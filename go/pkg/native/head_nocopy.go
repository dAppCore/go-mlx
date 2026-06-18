// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"unsafe"

	core "dappco.re/go"
)

// This file is the resident LM head — the fix for the per-token serve memory balloon. The head
// runs once per generated token over the (tied) [vocab × dModel] weight: final RMSNorm, the output
// projection (bf16 gemv or 4-bit qmv), then the optional logit soft-cap. LMHeadBF16/LMHeadQuant
// upload that whole weight into a FRESH Metal buffer EVERY token (sharedBytes inside QMVBF16 /
// MatVecBF16), an owned copy the autorelease pool never frees → resident memory grows ~weight-size
// per token (the ~503 MB tied embedding at 12B = the ~59 GB serve balloon). headEncoder binds the
// head weight ONCE and reuses it every token: zero per-token upload, zero growth.
//
// HOW the weight is bound, by dtype:
//   - bf16: a no-copy view into the shared shard mmap (the gemv reads the shard buffer reliably —
//     proven byte-identical in the full session).
//   - 4-bit: uploaded ONCE into a retained owned buffer at session build, then reused. The 4-bit
//     affine_qmv reading a NO-COPY view of the shard mmap is unreliable when other quant buffers
//     coexist in the session (NaN — the same class of issue that keeps the quant LAYER weights on
//     the copy path); a single owned upload sidesteps it AND still kills the balloon (one upload,
//     not one per token). It costs ONE resident copy of the head weight — not the per-token growth.
// Either way the per-token cost is just the dModel-length activation upload; the weight is resident.

// headEncoder is a resident LM head, built once. For bf16 the weight is bound as a no-copy shard
// view; for 4-bit it is an owned buffer uploaded once at build (held resident on this struct). Both
// avoid the per-token weight upload that caused the balloon. encode() allocates only the tiny
// per-call scratch/output, so it holds no shared MUTABLE state and is concurrency-safe (the serve
// drives one model from many request goroutines). nil (no shardBuffers, or an unresolved weight)
// signals the caller to fall back to the per-token upload head.
type headEncoder struct {
	finalNorm bufView // bf16 final-norm, no-copy shard view (a tiny vector — always reliable)
	weight    bufView // bf16 no-copy shard view, OR the 4-bit packed weight uploaded once (off 0)
	// quant triple companions (4-bit head only): scales/biases uploaded once. nil buf for bf16.
	scales, biases  bufView
	quant           bool
	groupSize, bits int
	dModel, vocab   int
	eps, softCap    float32
}

// newHeadEncoder builds the resident head: it resolves the final norm to a no-copy shard view, and
// binds the head weight per dtype — bf16 as a no-copy shard view, 4-bit as a one-time owned upload
// (packed + scales + biases). Returns nil when sb is nil or the bf16 weight/norm is not a view into
// the mapping (the caller then keeps the per-token upload head). MUST be called inside a
// withAutoreleasePool (the 4-bit owned buffers are objc-retained, so they survive it).
func newHeadEncoder(sb *shardBuffers, finalNormW, weight, scales, biases []byte, dModel, vocab, groupSize, bits int, eps, softCap float32, quant bool) (*headEncoder, error) {
	if sb == nil {
		return nil, nil
	}
	h := &headEncoder{
		quant:     quant,
		groupSize: groupSize, bits: bits, dModel: dModel, vocab: vocab, eps: eps, softCap: softCap,
	}
	if quant {
		// Fully upload-once owned buffers — weight + scales + biases AND the final norm. A no-copy
		// view of the shard mmap (whether the 4-bit qmv weight OR the bf16 norm) reads garbage once
		// the session's copy-path quant LAYER buffers coexist (the same in-session aliasing issue
		// that keeps the layer weights on the copy path). Uploading the head's few tensors once
		// sidesteps it entirely AND still kills the per-token balloon (one upload, not one per token).
		if len(weight) == 0 || len(scales) == 0 || len(biases) == 0 {
			return nil, nil
		}
		h.finalNorm = copyView(finalNormW)
		h.weight = copyView(weight)
		h.scales = copyView(scales)
		h.biases = copyView(biases)
		return h, nil
	}
	// bf16: no-copy shard views (the gemv reads the shard buffer reliably in-session).
	fn, err := sb.bufFor(finalNormW)
	if err != nil || fn.buf == nil {
		return nil, nil
	}
	w, err := sb.bufFor(weight)
	if err != nil || w.buf == nil {
		return nil, nil
	}
	h.finalNorm = fn
	h.weight = w
	return h, nil
}

// encode runs the head for one hidden state (dModel bf16 bytes) and returns vocab bf16 logits,
// binding the RESIDENT head weight — NO per-token weight upload (the whole point: the ~503 MB
// tied embedding is bound once, not re-uploaded). Byte-identical to LMHeadBF16/LMHeadQuant (same
// RMSNorm, same gemv/qmv kernel + ABI, same host soft-cap), only the weight binding differs. The
// per-call scratch/output are freshly allocated (small, transient), so encode holds no shared
// mutable state and is concurrency-safe.
func (h *headEncoder) encode(hidden []byte) ([]byte, error) {
	if len(hidden) != h.dModel*bf16Size {
		return nil, core.NewError("native.headEncoder.encode: hidden must be dModel bf16 bytes")
	}
	out := make([]byte, h.vocab*bf16Size)
	var encErr error
	withAutoreleasePool(func() {
		hiddenBuf := sharedBytes(hidden) // the only upload: the dModel-length activation, not the weight
		normed := scratchBF16(h.dModel)
		logits := scratchBF16(h.vocab)
		cb := queue.CommandBuffer()
		enc := cb.ComputeCommandEncoder()
		if encErr = encRMSNormBF16(enc, hiddenBuf, h.finalNorm.buf, normed, h.finalNorm.off, h.dModel, h.eps); encErr != nil {
			enc.EndEncoding()
			return
		}
		if h.quant {
			encErr = encQMVBF16(enc, h.weight.buf, h.scales.buf, h.biases.buf, normed, logits,
				h.weight.off, h.scales.off, h.biases.off, 0, h.vocab, h.dModel, h.groupSize, h.bits)
		} else {
			encErr = encGemvBF16To(enc, h.weight.buf, normed, logits, h.weight.off, 0, h.vocab, h.dModel)
		}
		if encErr != nil {
			enc.EndEncoding()
			return
		}
		enc.EndEncoding()
		cb.Commit()
		cb.WaitUntilCompleted()
		copy(out, unsafe.Slice((*byte)(logits.Contents()), h.vocab*bf16Size))
	})
	if encErr != nil {
		return nil, encErr
	}
	if h.softCap > 0 { // monotonic, preserves the argmax — the host pass LMHead* also does
		for i := 0; i < h.vocab; i++ {
			v := bf16ToF32(out[i*bf16Size], out[i*bf16Size+1])
			c := f32ToBF16(h.softCap * float32(math.Tanh(float64(v/h.softCap))))
			out[i*bf16Size] = byte(c)
			out[i*bf16Size+1] = byte(c >> 8)
		}
	}
	return out, nil
}

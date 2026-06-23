// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"sync"
	"unsafe"

	core "dappco.re/go"
	"dappco.re/go/mlx/pkg/model"
	"github.com/tmc/apple/metal"
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
// per-call scratch/output; direct greedy reuses tiny scratch buffers through a concurrency-safe
// pool. nil (no shardBuffers, or an unresolved weight) signals the caller to fall back to the
// per-token upload head.
type headEncoder struct {
	finalNorm bufView // bf16 final-norm, no-copy shard view (a tiny vector — always reliable)
	weight    bufView // bf16 no-copy shard view, OR the 4-bit packed weight uploaded once (off 0)
	// quant triple companions (4-bit head only): scales/biases uploaded once. nil buf for bf16.
	scales, biases  bufView
	softCapScale    bufView
	invSoftCapScale bufView
	quant           bool
	groupSize, bits int
	dModel, vocab   int
	eps, softCap    float32
	greedyScratch   sync.Pool
}

type headGreedyScratch struct {
	tileCapacity            int
	tileValues, tileIndices metal.MTLBuffer
	outToken                metal.MTLBuffer
	dModelCapacity          int
	normed                  metal.MTLBuffer
	vocabCapacity           int
	logits                  metal.MTLBuffer
	suppressCapacity        int
	suppress                metal.MTLBuffer
}

// newHeadEncoder builds the resident head: it resolves the final norm to a no-copy shard view when
// a shard mapping is available, otherwise it binds owned resident buffers for in-memory sessions.
// BF16 directory heads use no-copy shard views; 4-bit heads use a one-time owned upload (packed +
// scales + biases) because qmv over the shared mmap is unreliable in-session. Returns nil only when
// required weights are missing or an expected shard view cannot be resolved. MUST be called inside a
// withAutoreleasePool (the owned buffers are objc-retained, so they survive it).
func newHeadEncoder(sb *shardBuffers, finalNormW, weight, scales, biases []byte, dModel, vocab, groupSize, bits int, eps, softCap float32, quant bool) (*headEncoder, error) {
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
		if len(finalNormW) == 0 || len(weight) == 0 || len(scales) == 0 || len(biases) == 0 {
			return nil, nil
		}
		h.finalNorm = copyView(finalNormW)
		h.weight = copyView(weight)
		h.scales = copyView(scales)
		h.biases = copyView(biases)
		h.initSoftcapBuffers()
		return h, nil
	}
	if len(finalNormW) == 0 || len(weight) == 0 {
		return nil, nil
	}
	if sb == nil {
		h.finalNorm = copyView(finalNormW)
		h.weight = copyView(weight)
		h.initSoftcapBuffers()
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
	h.initSoftcapBuffers()
	return h, nil
}

func (h *headEncoder) initSoftcapBuffers() {
	if h.softCap <= 0 {
		return
	}
	inv := bf16ScalarBytes(1 / h.softCap)
	capv := bf16ScalarBytes(h.softCap)
	h.invSoftCapScale = copyView(inv[:])
	h.softCapScale = copyView(capv[:])
}

// encode runs the head for one hidden state (dModel bf16 bytes) and returns vocab bf16 logits,
// binding the RESIDENT head weight — NO per-token weight upload (the whole point: the ~503 MB
// tied embedding is bound once, not re-uploaded). Same RMSNorm and gemv/qmv kernel + ABI as
// LMHeadBF16/LMHeadQuant; sampled softcap stays on the BF16 kernel route instead of looping on
// the host. The
// per-call scratch/output are freshly allocated (small, transient), so encode holds no shared
// mutable state and is concurrency-safe.
func (h *headEncoder) encode(hidden []byte, skipSoftcap bool) ([]byte, error) {
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
		if h.softCap > 0 && !skipSoftcap && h.vocab > 0 {
			scaled := scratchBF16(h.vocab)
			capped := scratchBF16(h.vocab)
			invBytes := bf16ScalarBytes(1 / h.softCap)
			invScale := h.invSoftCapScale
			if invScale.buf == nil {
				invScale = copyView(invBytes[:])
			}
			if encErr = encScaleBF16(enc, logits, invScale.buf, scaled, invScale.off, invBytes[:], h.vocab); encErr != nil {
				enc.EndEncoding()
				return
			}
			if encErr = encTanhBF16(enc, scaled, capped, h.vocab); encErr != nil {
				enc.EndEncoding()
				return
			}
			capBytes := bf16ScalarBytes(h.softCap)
			capScale := h.softCapScale
			if capScale.buf == nil {
				capScale = copyView(capBytes[:])
			}
			if encErr = encScaleBF16(enc, capped, capScale.buf, logits, capScale.off, capBytes[:], h.vocab); encErr != nil {
				enc.EndEncoding()
				return
			}
		}
		enc.EndEncoding()
		cb.Commit()
		cb.WaitUntilCompleted()
		copy(out, unsafe.Slice((*byte)(logits.Contents()), h.vocab*bf16Size))
	})
	if encErr != nil {
		return nil, encErr
	}
	return out, nil
}

func newHeadGreedyScratch(tileCapacity, dModel, vocab int, needLogits bool) *headGreedyScratch {
	s := &headGreedyScratch{
		tileCapacity: tileCapacity,
		tileValues:   device.NewBufferWithLengthOptions(uint(tileCapacity*4), metal.MTLResourceStorageModeShared),
		tileIndices:  device.NewBufferWithLengthOptions(uint(tileCapacity*4), metal.MTLResourceStorageModeShared),
		outToken:     device.NewBufferWithLengthOptions(4, metal.MTLResourceStorageModeShared),
	}
	if dModel > 0 {
		s.dModelCapacity = dModel
		s.normed = scratchBF16(dModel)
	}
	if needLogits && vocab > 0 {
		s.vocabCapacity = vocab
		s.logits = scratchBF16(vocab)
	}
	return s
}

func (h *headEncoder) getGreedyScratch(tileCount int, needLogits bool) *headGreedyScratch {
	if v := h.greedyScratch.Get(); v != nil {
		s := v.(*headGreedyScratch)
		hasTiles := s.tileCapacity >= tileCount && s.tileValues != nil && s.tileIndices != nil && s.outToken != nil
		hasNormed := s.dModelCapacity >= h.dModel && s.normed != nil
		hasLogits := !needLogits || (s.vocabCapacity >= h.vocab && s.logits != nil)
		if hasTiles && hasNormed && hasLogits {
			return s
		}
	}
	return newHeadGreedyScratch(tileCount, h.dModel, h.vocab, needLogits)
}

func (h *headEncoder) putGreedyScratch(s *headGreedyScratch) {
	if s != nil && s.tileValues != nil && s.tileIndices != nil && s.outToken != nil && s.normed != nil {
		h.greedyScratch.Put(s)
	}
}

func (s *headGreedyScratch) suppressBuffer(ids []int32) metal.MTLBuffer {
	if len(ids) == 0 {
		return nil
	}
	if s.suppress == nil || s.suppressCapacity < len(ids) {
		s.suppressCapacity = len(ids)
		s.suppress = device.NewBufferWithLengthOptions(uint(len(ids)*4), metal.MTLResourceStorageModeShared)
	}
	copy(unsafe.Slice((*int32)(s.suppress.Contents()), len(ids)), ids)
	return s.suppress
}

func tokenSuppressed(id int, suppress []int32) bool {
	for _, sid := range suppress {
		if sid == int32(id) {
			return true
		}
	}
	return false
}

func greedyBF16Suppressed(logits []byte, vocab int, suppress []int32) (int32, error) {
	if len(suppress) == 0 {
		return model.Greedy(logits, vocab)
	}
	if len(logits) != vocab*bf16Size {
		return 0, core.NewError("native.greedyBF16Suppressed: logits must be vocab bf16 bytes")
	}
	best := -1
	var bestV float32
	for i := 0; i < vocab; i++ {
		if tokenSuppressed(i, suppress) {
			continue
		}
		v := bf16ToF32(logits[i*bf16Size], logits[i*bf16Size+1])
		if best < 0 || v > bestV {
			best, bestV = i, v
		}
	}
	if best < 0 {
		return 0, core.NewError("native.greedyBF16Suppressed: all vocab ids are suppressed")
	}
	return int32(best), nil
}

// greedy is the direct-token counterpart to pkg/metal's direct greedy/q4 LM-head
// top-k features, narrowed to the production greedy case. It runs final RMSNorm
// and head argmax in one command buffer, masks suppressed ids before argmax,
// and copies back only the selected token. ok=false means this head/geometry
// cannot use the custom kernel, so callers keep the existing full-logits path.
// encodeGreedy encodes finalRMSNorm(hiddenBuf) + LMHead + tiled argmax into `enc` WITHOUT committing —
// the caller owns the command buffer, so a decode step can chain its replay onto the SAME buffer and pay
// one sync/token instead of two. Returns the GPU token buffer (read after the cb completes) + the scratch
// to release then. ok=false ⇒ the head can't do a direct GPU argmax (caller falls back to the logits path).
func (h *headEncoder) encodeGreedy(enc metal.MTLComputeCommandEncoder, hiddenBuf metal.MTLBuffer, suppress []int32) (outToken metal.MTLBuffer, scratch *headGreedyScratch, ok bool, err error) {
	if h.finalNorm.buf == nil || h.weight.buf == nil {
		return nil, nil, false, nil
	}
	if h.quant {
		if h.scales.buf == nil || h.biases.buf == nil || !qmvLogitsArgmaxUsable(h.dModel, h.vocab, h.groupSize, h.bits) {
			return nil, nil, false, nil
		}
	} else if !bf16LMHeadArgmaxUsable(h.dModel, h.vocab) {
		return nil, nil, false, nil
	}
	rowsPerTile := bf16LMHeadArgmaxRowsPerTile
	needLogits := false
	if h.quant {
		rowsPerTile = bf16LogitsArgmaxRowsPerTile
		needLogits = true
	}
	tileCount := (h.vocab + rowsPerTile - 1) / rowsPerTile
	scratch = h.getGreedyScratch(tileCount, needLogits)
	normed := scratch.normed
	suppressBuf := scratch.suppressBuffer(suppress)
	if err = encRMSNormBF16(enc, hiddenBuf, h.finalNorm.buf, normed, h.finalNorm.off, h.dModel, h.eps); err != nil {
		return scratch.outToken, scratch, true, err
	}
	if h.quant {
		logits := scratch.logits
		if err = encQMVBF16(enc, h.weight.buf, h.scales.buf, h.biases.buf, normed, logits,
			h.weight.off, h.scales.off, h.biases.off, 0, h.vocab, h.dModel, h.groupSize, h.bits); err != nil {
			return scratch.outToken, scratch, true, err
		}
		if err = encBF16LogitsArgmaxTilesBF16(enc, logits, scratch.tileValues, scratch.tileIndices, suppressBuf, h.vocab, len(suppress)); err != nil {
			return scratch.outToken, scratch, true, err
		}
	} else {
		if err = encBF16LMHeadArgmaxTilesBF16(enc, normed, h.weight.buf, scratch.tileValues, scratch.tileIndices, suppressBuf, 0, h.weight.off, h.dModel, h.vocab, len(suppress)); err != nil {
			return scratch.outToken, scratch, true, err
		}
	}
	if err = encArgmaxMergeF32(enc, scratch.tileValues, scratch.tileIndices, scratch.outToken, tileCount); err != nil {
		return scratch.outToken, scratch, true, err
	}
	return scratch.outToken, scratch, true, nil
}

func (h *headEncoder) greedy(hidden []byte, suppress []int32) (token int32, ok bool, err error) {
	if len(hidden) != h.dModel*bf16Size {
		return 0, true, core.NewError("native.headEncoder.greedy: hidden must be dModel bf16 bytes")
	}
	token = -1
	var encErr error
	withAutoreleasePool(func() {
		hiddenBuf := sharedBytes(hidden)
		cb := queue.CommandBuffer()
		enc := cb.ComputeCommandEncoder()
		var scratch *headGreedyScratch
		var outToken metal.MTLBuffer
		outToken, scratch, ok, encErr = h.encodeGreedy(enc, hiddenBuf, suppress)
		if !ok || encErr != nil {
			enc.EndEncoding()
			if scratch != nil {
				h.putGreedyScratch(scratch)
			}
			return
		}
		enc.EndEncoding()
		cb.Commit()
		cb.WaitUntilCompleted()
		token = *(*int32)(outToken.Contents())
		h.putGreedyScratch(scratch)
	})
	if encErr != nil {
		return 0, true, encErr
	}
	if !ok {
		return 0, false, nil
	}
	if token < 0 || int(token) >= h.vocab {
		return 0, true, core.NewError("native.headEncoder.greedy: direct argmax returned invalid token")
	}
	return token, true, nil
}

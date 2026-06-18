// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"unsafe"

	core "dappco.re/go"
	"github.com/tmc/apple/metal"
)

// This file completes the decode step DecodeLayer deferred: the cache-WRITE half.
// A real autoregressive step projects K and V from the current token, RoPEs the
// new K, APPENDS them to a growing KV cache, then attends over the grown window —
// where DecodeLayer attended a fixed handed-in cache (Q-only).
//
// The cache is SEQ-MAJOR [maxLen, nKVHeads, headDim] (headDim innermost). That
// makes a token's K (and V) for all heads one contiguous row at byte offset
// pos*nKVHeads*headDim*2, so the projection writes STRAIGHT into the cache via a
// bound-buffer offset — no copy kernel (the static metallib has none; copies are
// JIT). The sdpa_vector kernel indexes keys as kv_head*k_head_stride +
// seq*k_seq_stride + d, so seq-major just sets k_head_stride=headDim,
// k_seq_stride=nKVHeads*headDim and N=pos+1 (the live window). bf16 throughout.

// attnScratch holds the attention-half intermediates, allocated once so the
// decode loop reuses them every token (no per-token buffer churn).
type attnScratch struct {
	normed, q, qr, kProj, attn, attnOut metal.MTLBuffer
}

func newAttnScratch(dModel, qDim, kvDim int) attnScratch {
	return attnScratch{
		normed:  scratchBF16(dModel),
		q:       scratchBF16(qDim),
		qr:      scratchBF16(qDim),
		kProj:   scratchBF16(kvDim),
		attn:    scratchBF16(qDim),
		attnOut: scratchBF16(dModel),
	}
}

// mlpScratch holds the MLP-half intermediates (the gelu chain), allocated once.
type mlpScratch struct {
	mlpNormed, gate, up         metal.MTLBuffer
	x2, x3, x3s, inner          metal.MTLBuffer
	scaled, tnh, onePlus, halfG metal.MTLBuffer
	gelu, gated, down           metal.MTLBuffer
	c044, c079, c1, c05         metal.MTLBuffer
}

func newMLPScratch(dModel, dFF int) mlpScratch {
	return mlpScratch{
		mlpNormed: scratchBF16(dModel),
		gate:      scratchBF16(dFF), up: scratchBF16(dFF),
		x2: scratchBF16(dFF), x3: scratchBF16(dFF), x3s: scratchBF16(dFF), inner: scratchBF16(dFF),
		scaled: scratchBF16(dFF), tnh: scratchBF16(dFF), onePlus: scratchBF16(dFF), halfG: scratchBF16(dFF),
		gelu: scratchBF16(dFF), gated: scratchBF16(dFF), down: scratchBF16(dModel),
		c044: sharedBytes(bf16ConstBytes(dFF, 0.044715)),
		c079: sharedBytes(bf16ConstBytes(dFF, 0.7978845608028654)),
		c1:   sharedBytes(bf16ConstBytes(dFF, 1.0)),
		c05:  sharedBytes(bf16ConstBytes(dFF, 0.5)),
	}
}

// encResidualMaybeNorm encodes out = x + v, or out = x + RMSNorm(v, norm) when norm is
// non-nil (the gemma4 post-attention / post-feed-forward norm, applied to the branch
// output before the residual add). scratch holds the normed value; pass a buffer the
// caller no longer needs (sc.normed after the attention projections, sc.mlpNormed after
// the MLP projections). Bf16, dModel-wide.
func encResidualMaybeNorm(enc metal.MTLComputeCommandEncoder, x, v, scratch, out, norm metal.MTLBuffer, dModel int, eps float32) error {
	if norm == nil {
		return encAddBF16(enc, x, v, out, dModel)
	}
	if err := encRMSNormBF16(enc, v, norm, scratch, dModel, eps); err != nil {
		return err
	}
	return encAddBF16(enc, x, scratch, out, dModel)
}

// encAttnHalfKV encodes the real attention half — projections, K-RoPE into the
// cache, V into the cache, attention over the grown window, output projection,
// residual — into enc. The new token's K/V are written into kCacheBuf/vCacheBuf
// (seq-major) at row pos via the projection's bound-buffer offset; offBuf must
// already hold int32(pos). attends over rows [0..pos]; writes x + Wo·attn -> h (with
// the gemma4 post-attention norm on Wo·attn first when postAttnNorm is non-nil).
func encAttnHalfKV(
	enc metal.MTLComputeCommandEncoder,
	x, attnNormW, kCacheBuf, vCacheBuf, offBuf, h, postAttnNorm, qNorm, kNorm metal.MTLBuffer,
	sc attnScratch, proj projector,
	dModel, nHeads, nKVHeads, headDim, pos, slideW, rotaryDim int, base, scale, eps float32,
	ropeFreqs metal.MTLBuffer,
) error {
	kvDim := nKVHeads * headDim
	rowOff := uint(pos * kvDim * bf16Size) // byte offset of this token's cache row
	if err := encRMSNormBF16(enc, x, attnNormW, sc.normed, dModel, eps); err != nil {
		return err
	}
	// query: project, (gemma4 per-head QK-norm), rotate IN PLACE (so partial rotary's tail keeps the projected value)
	if err := proj.project(enc, sc.normed, sc.q, 0, projQ); err != nil {
		return err
	}
	if qNorm != nil {
		if err := encRMSNormRowsBF16(enc, sc.q, qNorm, sc.q, 0, 0, nHeads, headDim, eps); err != nil {
			return err
		}
	}
	if err := encRopeDecode(enc, sc.q, sc.q, 0, 0, offBuf, ropeFreqs, nHeads, headDim, rotaryDim, base, scale); err != nil {
		return err
	}
	// key: project STRAIGHT into the cache row, then (gemma4 per-head QK-norm) + rotate IN PLACE
	// there — partial rotary leaves the tail as the projected+normed value already in the cache.
	if err := proj.project(enc, sc.normed, kCacheBuf, rowOff, projK); err != nil {
		return err
	}
	if kNorm != nil {
		if err := encRMSNormRowsBF16(enc, kCacheBuf, kNorm, kCacheBuf, rowOff, rowOff, nKVHeads, headDim, eps); err != nil {
			return err
		}
	}
	if err := encRopeDecode(enc, kCacheBuf, kCacheBuf, rowOff, rowOff, offBuf, ropeFreqs, nKVHeads, headDim, rotaryDim, base, scale); err != nil {
		return err
	}
	// value: project STRAIGHT into the cache row (no rotation)
	if err := proj.project(enc, sc.normed, vCacheBuf, rowOff, projV); err != nil {
		return err
	}
	// attend the window [start..pos] (global: all; sliding: last slideW), seq-major
	start, n := slideWindow(pos, slideW)
	if err := encSDPAStrided(enc, sc.q, kCacheBuf, vCacheBuf, sc.attn,
		nHeads, nKVHeads, headDim, n,
		int64(headDim), int64(kvDim), int64(headDim), int64(kvDim), scale, uint(start*kvDim*bf16Size)); err != nil {
		return err
	}
	if err := proj.project(enc, sc.attn, sc.attnOut, 0, projO); err != nil {
		return err
	}
	// h = x + Wo·attn  (gemma4: post-attention norm on Wo·attn first; sc.normed is free)
	return encResidualMaybeNorm(enc, x, sc.attnOut, sc.normed, h, postAttnNorm, dModel, eps)
}

// encMLPHalfBF16 encodes the gemma MLP half — rms, gate/up projections, the tanh
// gelu approximation, gate·up, down projection, residual — into enc, exactly as
// DecodeLayer's MLP half. Reads h, writes h + Wdown·(gelu(Wgate·rms(h))·(Wup·rms(h)))
// -> out.
func encMLPHalfBF16(
	enc metal.MTLComputeCommandEncoder,
	h, mlpNormW, out, postFFNorm metal.MTLBuffer,
	sc mlpScratch, proj projector,
	dModel, dFF int, eps float32,
) error {
	if err := encRMSNormBF16(enc, h, mlpNormW, sc.mlpNormed, dModel, eps); err != nil {
		return err
	}
	if err := proj.project(enc, sc.mlpNormed, sc.gate, 0, projGate); err != nil {
		return err
	}
	if err := proj.project(enc, sc.mlpNormed, sc.up, 0, projUp); err != nil {
		return err
	}
	// gelu_approx(gate)
	_ = encMulBF16(enc, sc.gate, sc.gate, sc.x2, dFF)
	_ = encMulBF16(enc, sc.x2, sc.gate, sc.x3, dFF)
	_ = encMulBF16(enc, sc.x3, sc.c044, sc.x3s, dFF)
	_ = encAddBF16(enc, sc.gate, sc.x3s, sc.inner, dFF)
	_ = encMulBF16(enc, sc.inner, sc.c079, sc.scaled, dFF)
	_ = encTanhBF16(enc, sc.scaled, sc.tnh, dFF)
	_ = encAddBF16(enc, sc.tnh, sc.c1, sc.onePlus, dFF)
	_ = encMulBF16(enc, sc.gate, sc.c05, sc.halfG, dFF)
	_ = encMulBF16(enc, sc.halfG, sc.onePlus, sc.gelu, dFF)
	_ = encMulBF16(enc, sc.gelu, sc.up, sc.gated, dFF)
	if err := proj.project(enc, sc.gated, sc.down, 0, projDown); err != nil {
		return err
	}
	// out = h + Wdown·…  (gemma4: post-feed-forward norm on Wdown·… first; sc.mlpNormed is free)
	return encResidualMaybeNorm(enc, h, sc.down, sc.mlpNormed, out, postFFNorm, dModel, eps)
}

// validateStepKV checks the shared shape contract for the KV-cache decode entries.
func validateStepKV(x, attnNormW, wQ, wK, wV, wO, kCache, vCache []byte, dModel, nHeads, nKVHeads, headDim, maxLen, pos int) error {
	qDim, kvDim := nHeads*headDim, nKVHeads*headDim
	if nKVHeads == 0 || nHeads%nKVHeads != 0 {
		return core.NewError("native.DecodeStepKV: nHeads must be a multiple of nKVHeads")
	}
	if pos < 0 || pos >= maxLen {
		return core.NewError("native.DecodeStepKV: pos out of [0,maxLen)")
	}
	if len(x) != dModel*bf16Size || len(attnNormW) != dModel*bf16Size {
		return core.NewError("native.DecodeStepKV: x/attnNormW must be dModel bf16 bytes")
	}
	if len(wQ) != qDim*dModel*bf16Size || len(wO) != dModel*qDim*bf16Size {
		return core.NewError("native.DecodeStepKV: wQ/wO size mismatch")
	}
	if len(wK) != kvDim*dModel*bf16Size || len(wV) != kvDim*dModel*bf16Size {
		return core.NewError("native.DecodeStepKV: wK/wV size mismatch")
	}
	if len(kCache) != maxLen*kvDim*bf16Size || len(vCache) != maxLen*kvDim*bf16Size {
		return core.NewError("native.DecodeStepKV: kCache/vCache must be maxLen*nKVHeads*headDim bf16 bytes")
	}
	return nil
}

// AttentionStepKV runs the attention half of one REAL decode step: it projects
// q/k/v from x, RoPEs q and the new k, appends k,v to the seq-major caches at row
// pos, attends over rows [0..pos], and returns x + Wo·attn. kCache/vCache are
// updated in place (the caller's backing arrays grow by one row). This is the
// piece DecodeLayer's "cache-write half is a follow-up" referred to. All raw bf16.
func AttentionStepKV(x, attnNormW, wQ, wK, wV, wO, kCache, vCache []byte, dModel, nHeads, nKVHeads, headDim, maxLen, pos int, base, scale, eps float32) ([]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if err := validateStepKV(x, attnNormW, wQ, wK, wV, wO, kCache, vCache, dModel, nHeads, nKVHeads, headDim, maxLen, pos); err != nil {
		return nil, err
	}
	qDim, kvDim := nHeads*headDim, nKVHeads*headDim
	out := make([]byte, dModel*bf16Size)
	var encErr error
	withAutoreleasePool(func() {
		xBuf, nwBuf := sharedBytes(x), sharedBytes(attnNormW)
		proj := bf16Projector{wQ: sharedBytes(wQ), wK: sharedBytes(wK), wV: sharedBytes(wV), wO: sharedBytes(wO), dModel: dModel, qDim: qDim, kvDim: kvDim}
		kBuf, vBuf := sharedBytes(kCache), sharedBytes(vCache)
		off := int32(pos)
		offBuf := device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&off), 4, metal.MTLResourceStorageModeShared)
		sc := newAttnScratch(dModel, qDim, kvDim)
		hBuf := scratchBF16(dModel)

		cb := queue.CommandBuffer()
		enc := cb.ComputeCommandEncoder()
		if encErr = encAttnHalfKV(enc, xBuf, nwBuf, kBuf, vBuf, offBuf, hBuf, nil, nil, nil, sc, proj, dModel, nHeads, nKVHeads, headDim, pos, 0, headDim, base, scale, eps, nil); encErr != nil {
			enc.EndEncoding()
			return
		}
		enc.EndEncoding()
		cb.Commit()
		cb.WaitUntilCompleted()
		copy(out, unsafe.Slice((*byte)(hBuf.Contents()), len(out)))
		// reflect the grown cache rows back to the caller's slices
		copy(kCache, unsafe.Slice((*byte)(kBuf.Contents()), len(kCache)))
		copy(vCache, unsafe.Slice((*byte)(vBuf.Contents()), len(vCache)))
	})
	return out, encErr
}

// DecodeStepKV runs one full REAL decode-layer step — the AttentionStepKV half
// then the gemma MLP half, both residuals — in one command buffer, growing the
// seq-major KV cache at row pos. kCache/vCache are updated in place. With the same
// inputs it equals AttentionStepKV fed through the MLP half; gated byte-for-byte
// against a reference built from the parity-proven ops. All raw bf16.
func DecodeStepKV(
	x, attnNormW, wQ, wK, wV, wO, kCache, vCache, mlpNormW, wGate, wUp, wDown []byte,
	dModel, nHeads, nKVHeads, headDim, maxLen, dFF, pos int,
	base, scale, eps float32,
) ([]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if err := validateStepKV(x, attnNormW, wQ, wK, wV, wO, kCache, vCache, dModel, nHeads, nKVHeads, headDim, maxLen, pos); err != nil {
		return nil, err
	}
	if len(mlpNormW) != dModel*bf16Size {
		return nil, core.NewError("native.DecodeStepKV: mlpNormW must be dModel bf16 bytes")
	}
	if len(wGate) != dFF*dModel*bf16Size || len(wUp) != dFF*dModel*bf16Size || len(wDown) != dModel*dFF*bf16Size {
		return nil, core.NewError("native.DecodeStepKV: MLP weight size mismatch")
	}
	qDim, kvDim := nHeads*headDim, nKVHeads*headDim
	out := make([]byte, dModel*bf16Size)
	var encErr error
	withAutoreleasePool(func() {
		xBuf, nwBuf := sharedBytes(x), sharedBytes(attnNormW)
		proj := bf16Projector{
			wQ: sharedBytes(wQ), wK: sharedBytes(wK), wV: sharedBytes(wV), wO: sharedBytes(wO),
			wGate: sharedBytes(wGate), wUp: sharedBytes(wUp), wDown: sharedBytes(wDown),
			dModel: dModel, qDim: qDim, kvDim: kvDim, dFF: dFF,
		}
		kBuf, vBuf := sharedBytes(kCache), sharedBytes(vCache)
		mnwBuf := sharedBytes(mlpNormW)
		off := int32(pos)
		offBuf := device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&off), 4, metal.MTLResourceStorageModeShared)
		asc := newAttnScratch(dModel, qDim, kvDim)
		msc := newMLPScratch(dModel, dFF)
		hBuf, outBuf := scratchBF16(dModel), scratchBF16(dModel)

		cb := queue.CommandBuffer()
		enc := cb.ComputeCommandEncoder()
		if encErr = encAttnHalfKV(enc, xBuf, nwBuf, kBuf, vBuf, offBuf, hBuf, nil, nil, nil, asc, proj, dModel, nHeads, nKVHeads, headDim, pos, 0, headDim, base, scale, eps, nil); encErr != nil {
			enc.EndEncoding()
			return
		}
		if encErr = encMLPHalfBF16(enc, hBuf, mnwBuf, outBuf, nil, msc, proj, dModel, dFF, eps); encErr != nil {
			enc.EndEncoding()
			return
		}
		enc.EndEncoding()
		cb.Commit()
		cb.WaitUntilCompleted()
		copy(out, unsafe.Slice((*byte)(outBuf.Contents()), len(out)))
		copy(kCache, unsafe.Slice((*byte)(kBuf.Contents()), len(kCache)))
		copy(vCache, unsafe.Slice((*byte)(vBuf.Contents()), len(vCache)))
	})
	return out, encErr
}

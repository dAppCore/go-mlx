// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"unsafe"

	"github.com/tmc/apple/metal"
)

// attentionReEncode runs the bf16 attention block `reps` times the REGULAR way —
// persistent buffers, but the 6 ops re-encoded into a fresh command buffer every
// rep (the host re-encode the ICB path replaces). Buffers are created once so the
// measurement isolates per-rep host ENCODE cost, not buffer churn; the A/B
// against AttentionBlockICB(reps) is the encode-bypass number. Returns after the
// last rep completes.
func attentionReEncode(x, normWeight, wQ, wO, kCache, vCache []byte, dModel, nHeads, nKVHeads, headDim, kvLen int, base, scale float32, offset int, eps float32, reps int) error {
	if err := ensureInit(); err != nil {
		return err
	}
	qDim := nHeads * headDim
	var encErr error
	withAutoreleasePool(func() {
		xBuf, nwBuf := sharedBytes(x), sharedBytes(normWeight)
		wqBuf, woBuf := sharedBytes(wQ), sharedBytes(wO)
		kBuf, vBuf := sharedBytes(kCache), sharedBytes(vCache)
		off := int32(offset)
		offBuf := device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&off), 4, metal.MTLResourceStorageModeShared)
		normed := scratchBF16(dModel)
		q, qr, attn := scratchBF16(qDim), scratchBF16(qDim), scratchBF16(qDim)
		attnOut, outBuf := scratchBF16(dModel), scratchBF16(dModel)
		_ = outBuf

		for r := 0; r < reps; r++ {
			cb := queue.CommandBuffer()
			enc := cb.ComputeCommandEncoder()
			if encErr = encRMSNormBF16(enc, xBuf, nwBuf, normed, dModel, eps); encErr != nil {
				enc.EndEncoding()
				return
			}
			_ = encGemvBF16(enc, wqBuf, normed, q, qDim, dModel)
			_ = encRoPEBF16(enc, q, qr, offBuf, nHeads, headDim, base, scale)
			_ = encSDPA(enc, qr, kBuf, vBuf, attn, nHeads, nKVHeads, headDim, kvLen, scale)
			_ = encGemvBF16(enc, woBuf, attn, attnOut, dModel, qDim)
			_ = encAddBF16(enc, xBuf, attnOut, outBuf, dModel)
			enc.EndEncoding()
			cb.Commit()
			cb.WaitUntilCompleted()
		}
	})
	return encErr
}

// layerReEncode runs the full 21-op bf16 DecodeLayer `reps` times the REGULAR way
// — persistent buffers, but all 21 ops re-encoded into a fresh command buffer
// every rep (the host re-encode the ICB path replaces). It is the full-layer
// analogue of attentionReEncode: buffers are created once so the measurement
// isolates per-rep host ENCODE cost, not buffer churn, and the A/B against
// DecodeLayerICB(reps) is the per-layer encode-bypass number. The op sequence
// mirrors DecodeLayer exactly. Returns after the last rep completes.
func layerReEncode(
	x, attnNormW, wQ, wO, kCache, vCache, mlpNormW, wGate, wUp, wDown []byte,
	dModel, nHeads, nKVHeads, headDim, kvLen, dFF int,
	base, scale float32, offset int, eps float32,
	reps int,
) error {
	if err := ensureInit(); err != nil {
		return err
	}
	qDim := nHeads * headDim
	var encErr error
	withAutoreleasePool(func() {
		xBuf := sharedBytes(x)
		anwBuf, mnwBuf := sharedBytes(attnNormW), sharedBytes(mlpNormW)
		wqBuf, woBuf := sharedBytes(wQ), sharedBytes(wO)
		kBuf, vBuf := sharedBytes(kCache), sharedBytes(vCache)
		wgBuf, wuBuf, wdBuf := sharedBytes(wGate), sharedBytes(wUp), sharedBytes(wDown)
		off := int32(offset)
		offBuf := device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&off), 4, metal.MTLResourceStorageModeShared)
		c044 := sharedBytes(bf16ConstBytes(dFF, 0.044715))
		c079 := sharedBytes(bf16ConstBytes(dFF, 0.7978845608028654))
		c1c := sharedBytes(bf16ConstBytes(dFF, 1.0))
		c05 := sharedBytes(bf16ConstBytes(dFF, 0.5))

		attnNormed := scratchBF16(dModel)
		q, qr, attn := scratchBF16(qDim), scratchBF16(qDim), scratchBF16(qDim)
		attnOut, h := scratchBF16(dModel), scratchBF16(dModel)
		mlpNormed := scratchBF16(dModel)
		gate, up := scratchBF16(dFF), scratchBF16(dFF)
		x2, x3, x3s, inner := scratchBF16(dFF), scratchBF16(dFF), scratchBF16(dFF), scratchBF16(dFF)
		scaled, tnh, onePlus, halfG := scratchBF16(dFF), scratchBF16(dFF), scratchBF16(dFF), scratchBF16(dFF)
		gelu, gated := scratchBF16(dFF), scratchBF16(dFF)
		down, outBuf := scratchBF16(dModel), scratchBF16(dModel)
		_ = outBuf

		for r := 0; r < reps; r++ {
			cb := queue.CommandBuffer()
			enc := cb.ComputeCommandEncoder()
			if encErr = encRMSNormBF16(enc, xBuf, anwBuf, attnNormed, dModel, eps); encErr != nil {
				enc.EndEncoding()
				return
			}
			// attention half
			_ = encGemvBF16(enc, wqBuf, attnNormed, q, qDim, dModel)
			_ = encRoPEBF16(enc, q, qr, offBuf, nHeads, headDim, base, scale)
			_ = encSDPA(enc, qr, kBuf, vBuf, attn, nHeads, nKVHeads, headDim, kvLen, scale)
			_ = encGemvBF16(enc, woBuf, attn, attnOut, dModel, qDim)
			_ = encAddBF16(enc, xBuf, attnOut, h, dModel)
			// MLP half
			_ = encRMSNormBF16(enc, h, mnwBuf, mlpNormed, dModel, eps)
			_ = encGemvBF16(enc, wgBuf, mlpNormed, gate, dFF, dModel)
			_ = encGemvBF16(enc, wuBuf, mlpNormed, up, dFF, dModel)
			_ = encMulBF16(enc, gate, gate, x2, dFF)
			_ = encMulBF16(enc, x2, gate, x3, dFF)
			_ = encMulBF16(enc, x3, c044, x3s, dFF)
			_ = encAddBF16(enc, gate, x3s, inner, dFF)
			_ = encMulBF16(enc, inner, c079, scaled, dFF)
			_ = encTanhBF16(enc, scaled, tnh, dFF)
			_ = encAddBF16(enc, tnh, c1c, onePlus, dFF)
			_ = encMulBF16(enc, gate, c05, halfG, dFF)
			_ = encMulBF16(enc, halfG, onePlus, gelu, dFF)
			_ = encMulBF16(enc, gelu, up, gated, dFF)
			_ = encGemvBF16(enc, wdBuf, gated, down, dModel, dFF)
			_ = encAddBF16(enc, h, down, outBuf, dModel)
			enc.EndEncoding()
			cb.Commit()
			cb.WaitUntilCompleted()
		}
	})
	return encErr
}

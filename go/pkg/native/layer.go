// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"unsafe"

	core "dappco.re/go"
	"github.com/tmc/apple/metal"
)

// DecodeLayer runs a full gemma transformer decode layer on-device, in bf16, in
// ONE command buffer — the attention block feeding the MLP block, each with its
// residual, every intermediate resident:
//
//	h   = x + Wo·sdpa(rope(Wq·rms(x)), kCache, vCache)      // attention + residual
//	out = h + Wdown·( gelu(Wgate·rms(h)) · (Wup·rms(h)) )   // MLP + residual
//
// ~21 dispatches, one commit, no host round-trip. Attention attends over a given
// KV cache (the cache-write half is a follow-up). wQ is (nHeads·headDim × dModel),
// wO is (dModel × nHeads·headDim), wGate/wUp are (dFF × dModel), wDown is (dModel
// × dFF); kCache/vCache are (nKVHeads, kvLen, headDim). All inputs/outputs raw
// bf16 bytes. The result equals AttentionBlock then MLPBlockBF16 run separately —
// proven in the tests. This is a complete transformer layer on the no-cgo path.
func DecodeLayer(
	x, attnNormW, wQ, wO, kCache, vCache, mlpNormW, wGate, wUp, wDown []byte,
	dModel, nHeads, nKVHeads, headDim, kvLen, dFF int,
	base, scale float32, offset int, eps float32,
) ([]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	qDim := nHeads * headDim
	if len(x) != dModel*bf16Size || len(attnNormW) != dModel*bf16Size || len(mlpNormW) != dModel*bf16Size {
		return nil, core.NewError("native.DecodeLayer: x/attnNormW/mlpNormW must be dModel bf16 bytes")
	}
	if len(wQ) != qDim*dModel*bf16Size || len(wO) != dModel*qDim*bf16Size {
		return nil, core.NewError("native.DecodeLayer: wQ/wO size mismatch")
	}
	if len(wGate) != dFF*dModel*bf16Size || len(wUp) != dFF*dModel*bf16Size || len(wDown) != dModel*dFF*bf16Size {
		return nil, core.NewError("native.DecodeLayer: MLP weight size mismatch")
	}
	if len(kCache) != nKVHeads*kvLen*headDim*bf16Size || len(vCache) != nKVHeads*kvLen*headDim*bf16Size {
		return nil, core.NewError("native.DecodeLayer: kCache/vCache size mismatch")
	}

	out := make([]byte, dModel*bf16Size)
	var encErr error
	withAutoreleasePool(func() {
		// inputs
		xBuf := sharedBytes(x)
		anwBuf, mnwBuf := sharedBytes(attnNormW), sharedBytes(mlpNormW)
		wqBuf, woBuf := sharedBytes(wQ), sharedBytes(wO)
		kBuf, vBuf := sharedBytes(kCache), sharedBytes(vCache)
		wgBuf, wuBuf, wdBuf := sharedBytes(wGate), sharedBytes(wUp), sharedBytes(wDown)
		off := int32(offset)
		offBuf := device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&off), 4, metal.MTLResourceStorageModeShared)
		c044 := sharedBytes(bf16ConstBytes(dFF, 0.044715))
		c079 := sharedBytes(bf16ConstBytes(dFF, 0.7978845608028654))
		c1 := sharedBytes(bf16ConstBytes(dFF, 1.0))
		c05 := sharedBytes(bf16ConstBytes(dFF, 0.5))

		// attention intermediates
		attnNormed := scratchBF16(dModel)
		q, qr, attn := scratchBF16(qDim), scratchBF16(qDim), scratchBF16(qDim)
		attnOut, h := scratchBF16(dModel), scratchBF16(dModel)
		// mlp intermediates
		mlpNormed := scratchBF16(dModel)
		gate, up := scratchBF16(dFF), scratchBF16(dFF)
		x2, x3, x3s, inner := scratchBF16(dFF), scratchBF16(dFF), scratchBF16(dFF), scratchBF16(dFF)
		scaled, tnh, onePlus, halfG := scratchBF16(dFF), scratchBF16(dFF), scratchBF16(dFF), scratchBF16(dFF)
		gelu, gated := scratchBF16(dFF), scratchBF16(dFF)
		down, outBuf := scratchBF16(dModel), scratchBF16(dModel)

		cb := queue.CommandBuffer()
		enc := cb.ComputeCommandEncoder()
		steps := []func() error{
			// --- attention block (h = x + attn(rms(x))) ---
			func() error { return encRMSNormBF16(enc, xBuf, anwBuf, attnNormed, 0, dModel, eps) },
			func() error { return encGemvBF16(enc, wqBuf, attnNormed, q, qDim, dModel) },
			func() error { return encRoPEBF16(enc, q, qr, offBuf, nHeads, headDim, headDim, base, scale) },
			func() error { return encSDPA(enc, qr, kBuf, vBuf, attn, nHeads, nKVHeads, headDim, kvLen, scale) },
			func() error { return encGemvBF16(enc, woBuf, attn, attnOut, dModel, qDim) },
			func() error { return encAddBF16(enc, xBuf, attnOut, h, dModel) },
			// --- MLP block on h (out = h + mlp(rms(h))) ---
			func() error { return encRMSNormBF16(enc, h, mnwBuf, mlpNormed, 0, dModel, eps) },
			func() error { return encGemvBF16(enc, wgBuf, mlpNormed, gate, dFF, dModel) },
			func() error { return encGemvBF16(enc, wuBuf, mlpNormed, up, dFF, dModel) },
			// gelu_approx(gate)
			func() error { return encMulBF16(enc, gate, gate, x2, dFF) },
			func() error { return encMulBF16(enc, x2, gate, x3, dFF) },
			func() error { return encMulBF16(enc, x3, c044, x3s, dFF) },
			func() error { return encAddBF16(enc, gate, x3s, inner, dFF) },
			func() error { return encMulBF16(enc, inner, c079, scaled, dFF) },
			func() error { return encTanhBF16(enc, scaled, tnh, dFF) },
			func() error { return encAddBF16(enc, tnh, c1, onePlus, dFF) },
			func() error { return encMulBF16(enc, gate, c05, halfG, dFF) },
			func() error { return encMulBF16(enc, halfG, onePlus, gelu, dFF) },
			// gate·up, down projection, residual
			func() error { return encMulBF16(enc, gelu, up, gated, dFF) },
			func() error { return encGemvBF16(enc, wdBuf, gated, down, dModel, dFF) },
			func() error { return encAddBF16(enc, h, down, outBuf, dModel) },
		}
		for _, step := range steps {
			if encErr = step(); encErr != nil {
				enc.EndEncoding()
				return
			}
		}
		enc.EndEncoding()
		cb.Commit()
		cb.WaitUntilCompleted()
		copy(out, unsafe.Slice((*byte)(outBuf.Contents()), len(out)))
	})
	return out, encErr
}

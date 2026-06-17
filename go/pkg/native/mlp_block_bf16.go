// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"unsafe"

	core "dappco.re/go"
)

// MLPBlockBF16 runs a full gemma feed-forward block on-device in one command
// buffer, in bf16 (the dtype the decode path actually runs in) — the bf16
// sibling of MLPBlock:
//
//	normed = rmsnorm(x, normWeight)
//	gate   = Wgate · normed     up = Wup · normed      (dModel → dFF)
//	gated  = gelu(gate) · up    (gelu_approx composed in-line, each step rounded)
//	down   = Wdown · gated      (dFF → dModel)
//	out    = x + down           (residual)
//
// Every intermediate stays resident; one commit. Wgate/Wup are row-major
// (dFF × dModel), Wdown is (dModel × dFF). The gelu scalar operands are dense
// bf16 constant buffers built once via bf16ConstBytes, so the in-line gelu
// matches GeluGateMulBF16 byte-for-byte. All inputs/outputs are raw bf16 bytes;
// the result equals the same native bf16 ops run separately — proven in the
// tests. This is a real decode sub-block on the no-cgo path.
func MLPBlockBF16(x, normWeight, wGate, wUp, wDown []byte, dModel, dFF int, eps float32) ([]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if len(x) != dModel*bf16Size || len(normWeight) != dModel*bf16Size {
		return nil, core.NewError("native.MLPBlockBF16: x/normWeight must be dModel bf16 bytes")
	}
	if len(wGate) != dFF*dModel*bf16Size || len(wUp) != dFF*dModel*bf16Size {
		return nil, core.NewError("native.MLPBlockBF16: wGate/wUp must be dFF*dModel bf16 bytes")
	}
	if len(wDown) != dModel*dFF*bf16Size {
		return nil, core.NewError("native.MLPBlockBF16: wDown must be dModel*dFF bf16 bytes")
	}

	out := make([]byte, dModel*bf16Size)
	var encErr error
	withAutoreleasePool(func() {
		xBuf := sharedBytes(x)
		nwBuf := sharedBytes(normWeight)
		wgBuf, wuBuf, wdBuf := sharedBytes(wGate), sharedBytes(wUp), sharedBytes(wDown)
		// gelu scalar operands as dense dFF-length bf16 constant buffers — the
		// same constants GeluGateMulBF16 uses, so the in-line gelu is identical.
		c044 := sharedBytes(bf16ConstBytes(dFF, 0.044715))
		c079 := sharedBytes(bf16ConstBytes(dFF, 0.7978845608028654))
		c1 := sharedBytes(bf16ConstBytes(dFF, 1.0))
		c05 := sharedBytes(bf16ConstBytes(dFF, 0.5))
		// intermediates (resident)
		normed := scratchBF16(dModel)
		gate, up := scratchBF16(dFF), scratchBF16(dFF)
		x2, x3, x3s, inner, scaled, t, onePlus, halfG := scratchBF16(dFF), scratchBF16(dFF), scratchBF16(dFF), scratchBF16(dFF), scratchBF16(dFF), scratchBF16(dFF), scratchBF16(dFF), scratchBF16(dFF)
		gelu, gated := scratchBF16(dFF), scratchBF16(dFF)
		down := scratchBF16(dModel)
		outBuf := scratchBF16(dModel)

		cb := queue.CommandBuffer()
		enc := cb.ComputeCommandEncoder()
		steps := []func() error{
			func() error { return encRMSNormBF16(enc, xBuf, nwBuf, normed, dModel, eps) },
			func() error { return encGemvBF16(enc, wgBuf, normed, gate, dFF, dModel) },
			func() error { return encGemvBF16(enc, wuBuf, normed, up, dFF, dModel) },
			// gelu_approx(gate): x2=g·g; x3=x2·g; x3s=0.044715·x3; inner=g+x3s;
			//                    scaled=0.7978…·inner; t=tanh(scaled);
			//                    onePlus=t+1; halfG=0.5·g; gelu=halfG·onePlus
			func() error { return encMulBF16(enc, gate, gate, x2, dFF) },
			func() error { return encMulBF16(enc, x2, gate, x3, dFF) },
			func() error { return encMulBF16(enc, x3, c044, x3s, dFF) },
			func() error { return encAddBF16(enc, gate, x3s, inner, dFF) },
			func() error { return encMulBF16(enc, inner, c079, scaled, dFF) },
			func() error { return encTanhBF16(enc, scaled, t, dFF) },
			func() error { return encAddBF16(enc, t, c1, onePlus, dFF) },
			func() error { return encMulBF16(enc, gate, c05, halfG, dFF) },
			func() error { return encMulBF16(enc, halfG, onePlus, gelu, dFF) },
			// gate·up, down projection, residual
			func() error { return encMulBF16(enc, gelu, up, gated, dFF) },
			func() error { return encGemvBF16(enc, wdBuf, gated, down, dModel, dFF) },
			func() error { return encAddBF16(enc, xBuf, down, outBuf, dModel) },
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

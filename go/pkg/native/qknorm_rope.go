// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"sync"
	"unsafe"

	core "dappco.re/go"
	"github.com/tmc/apple/metal"
)

var (
	qkRopeDummyOnce    sync.Once
	qkRopeDummyPeriods metal.MTLBuffer
)

// qkRopeDummyBuf is a 1-element float buffer bound at the periods slot when use_freqs == 0 (the kernel
// never reads it on the base-rope path; Metal just wants the declared buffer bound).
func qkRopeDummyBuf() metal.MTLBuffer {
	qkRopeDummyOnce.Do(func() {
		one := float32(1)
		qkRopeDummyPeriods = device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&one), 4, metal.MTLResourceStorageModeShared)
	})
	return qkRopeDummyPeriods
}

var (
	qkRopePSOOnce sync.Once
	qkRopePSO     metal.MTLComputePipelineState
	qkRopePSOErr  error
)

// qkNormRopePipeline builds (once) the fused per-head QK-norm + RoPE pipeline from the custom kernels
// library. Shares the customLibraryLoaded gate with the gelu kernel.
func qkNormRopePipeline() (metal.MTLComputePipelineState, error) {
	qkRopePSOOnce.Do(func() {
		if customLibrary == nil || customLibrary.GetID() == 0 {
			qkRopePSOErr = core.NewError("native.qkNormRopePipeline: custom library unavailable")
			return
		}
		fn := customLibrary.NewFunctionWithName("lthn_qknorm_rope_bf16")
		if fn == nil || fn.GetID() == 0 {
			qkRopePSOErr = core.NewError("native.qkNormRopePipeline: kernel lthn_qknorm_rope_bf16 not found")
			return
		}
		qkRopePSO, qkRopePSOErr = device.NewComputePipelineStateWithFunctionError(fn)
	})
	return qkRopePSO, qkRopePSOErr
}

// QKNormRopeBF16 fuses, in ONE dispatch, gemma4's per-head QK-norm + RoPE:
//
//	out[head] = RoPE(RMSNorm(x[head], weight), offset)   — rotate the first rotaryDim dims, tail passes through
//
// x is [nHeads*headDim] bf16, weight is [headDim] bf16 (shared per head), out is the same shape. base is
// log2(theta) for the base-rope path; pass periods (1/inv_freq, length rotaryDim/2) for the freqs/YaRN
// path (non-empty ⇒ use_freqs). Numerically equal to RoPE(RMSNormBF16(x,w,nHeads,headDim)) — cosine
// ~1.0, ~1 ULP bf16 rounding (the lockstep fused-kernel gap) — gated in the parity test. headDim ≤ 512.
func QKNormRopeBF16(x, weight []byte, nHeads, headDim, rotaryDim, offset int, scale, eps, base float32, periods []float32) ([]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if len(x) != nHeads*headDim*bf16Size {
		return nil, core.NewError("native.QKNormRopeBF16: x must be nHeads*headDim bf16 bytes")
	}
	if len(weight) != headDim*bf16Size {
		return nil, core.NewError("native.QKNormRopeBF16: weight must be headDim bf16 bytes")
	}
	if headDim > 512 {
		return nil, core.NewError("native.QKNormRopeBF16: headDim exceeds the 512 threadgroup cap")
	}
	pso, err := qkNormRopePipeline()
	if err != nil {
		return nil, err
	}
	useFreqs := int32(0)
	per := periods
	if len(per) > 0 {
		useFreqs = 1
	} else {
		per = []float32{1} // dummy bind (kernel won't read it when use_freqs == 0)
	}

	out := make([]byte, len(x))
	withAutoreleasePool(func() {
		xBuf, wBuf := sharedBytes(x), sharedBytes(weight)
		oBuf := device.NewBufferWithLengthOptions(uint(len(out)), metal.MTLResourceStorageModeShared)
		off := int32(offset)
		offBuf := device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&off), 4, metal.MTLResourceStorageModeShared)
		perBuf := device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&per[0]), uint(len(per)*4), metal.MTLResourceStorageModeShared)

		cb := queue.CommandBuffer()
		enc := cb.ComputeCommandEncoder()
		enc.SetComputePipelineState(pso)
		enc.SetBufferWithOffsetAtIndex(xBuf, 0, 0)
		enc.SetBufferWithOffsetAtIndex(wBuf, 0, 1)
		enc.SetBufferWithOffsetAtIndex(oBuf, 0, 2)
		setEncFloat32(enc, eps, 3)
		setEncInt32(enc, int32(headDim), 4)
		setEncInt32(enc, int32(rotaryDim), 5)
		setEncFloat32(enc, scale, 6)
		enc.SetBufferWithOffsetAtIndex(offBuf, 0, 7)
		setEncFloat32(enc, base, 8)
		enc.SetBufferWithOffsetAtIndex(perBuf, 0, 9)
		setEncInt32(enc, useFreqs, 10)
		// one threadgroup per head, headDim threads each (threadgroup_position_in_grid = head)
		enc.DispatchThreadsThreadsPerThreadgroup(
			metal.MTLSize{Width: uint(nHeads * headDim), Height: 1, Depth: 1},
			metal.MTLSize{Width: uint(headDim), Height: 1, Depth: 1},
		)
		enc.EndEncoding()
		cb.Commit()
		cb.WaitUntilCompleted()
		copy(out, unsafe.Slice((*byte)(oBuf.Contents()), len(out)))
	})
	return out, nil
}

// encQKNormRope encodes the fused per-head QK-norm + RoPE (out = RoPE(RMSNorm(x, w))) into enc — the
// re-encode sibling of the ICB's setQKNormRope, using the SAME kernel so the two paths stay byte-equal
// under the lockstep fusion. base is RAW theta (log2'd here, matching encRoPEBF16To); periods non-nil ⇒
// the freqs/YaRN path. x/w/out may carry byte offsets (the K cache row, the qk-norm shard view).
// Caller guards with gpuHasGeluKernel.
func encQKNormRope(enc metal.MTLComputeCommandEncoder, x, w, out metal.MTLBuffer, xOff, wOff, outOff uint, offBuf, periods metal.MTLBuffer, nHeads, headDim, rotaryDim int, base, scale, eps float32) error {
	pso, err := qkNormRopePipeline()
	if err != nil {
		return err
	}
	rd := headDim
	if rotaryDim > 0 && rotaryDim < headDim {
		rd = rotaryDim
	}
	// fused per-head QK-norm + RoPE through the SHARED emitQKNormRope body (with the ICB setQKNormRope);
	// periods != nil selects the freqs form, else the base form binds qkRopeDummyBuf() at 9 (unread).
	emitQKNormRope(encSink{enc}, pso, x, w, out, xOff, wOff, outOff, offBuf, periods, qkRopeDummyBuf(),
		nHeads, headDim, rd, eps, scale, float32(math.Log2(float64(base))))
	return nil
}

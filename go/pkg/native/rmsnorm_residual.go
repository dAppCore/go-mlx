// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"sync"
	"unsafe"

	core "dappco.re/go"
	"github.com/tmc/apple/metal"
)

var (
	rmsResidualPSOOnce sync.Once
	rmsResidualPSO     metal.MTLComputePipelineState
	rmsResidualPSOErr  error
)

// rmsNormResidualPipeline builds (once) the fused rms-norm+residual pipeline from the custom kernels
// library (lthn_kernels.metallib). Shares the customLibraryLoaded gate with the gelu kernel.
func rmsNormResidualPipeline() (metal.MTLComputePipelineState, error) {
	rmsResidualPSOOnce.Do(func() {
		if customLibrary == nil || customLibrary.GetID() == 0 {
			rmsResidualPSOErr = core.NewError("native.rmsNormResidualPipeline: custom library unavailable")
			return
		}
		fn := customLibrary.NewFunctionWithName("lthn_rmsnorm_residual_bf16")
		if fn == nil || fn.GetID() == 0 {
			rmsResidualPSOErr = core.NewError("native.rmsNormResidualPipeline: kernel lthn_rmsnorm_residual_bf16 not found")
			return
		}
		rmsResidualPSO, rmsResidualPSOErr = device.NewComputePipelineStateWithFunctionError(fn)
	})
	return rmsResidualPSO, rmsResidualPSOErr
}

// RMSNormResidualBF16 computes, in ONE dispatch, the fused gemma4 post-attention / post-FF tail:
//
//	out = res + RMSNorm(x, weight)
//
// x/res/weight are bf16 bytes of length axisSize (single row); out is axisSize bf16 bytes. The kernel
// copies MLX's rms_single_row reduction verbatim and rounds the normed value to bf16 before the add,
// so the result is byte-identical to AddBF16(res, RMSNormBF16(x, weight)) — gated in the parity test.
// axisSize must be ≤ rmsLoopedLimit (the single-row kernel; every gemma hidden/head size qualifies).
// Guard with gpuHasGeluKernel (same custom library) before calling on the decode path.
func RMSNormResidualBF16(x, weight, res []byte, axisSize int, eps float32) ([]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if len(x) != axisSize*bf16Size || len(res) != axisSize*bf16Size {
		return nil, core.NewError("native.RMSNormResidualBF16: x and res must each be axisSize bf16 bytes")
	}
	if len(weight) != axisSize*bf16Size {
		return nil, core.NewError("native.RMSNormResidualBF16: weight must be axisSize bf16 bytes")
	}
	if axisSize > rmsLoopedLimit {
		return nil, core.NewError("native.RMSNormResidualBF16: axisSize exceeds the single-row kernel limit")
	}
	pso, err := rmsNormResidualPipeline()
	if err != nil {
		return nil, err
	}

	out := make([]byte, axisSize*bf16Size)
	withAutoreleasePool(func() {
		xBuf, wBuf, rBuf := sharedBytes(x), sharedBytes(weight), sharedBytes(res)
		oBuf := device.NewBufferWithLengthOptions(uint(len(out)), metal.MTLResourceStorageModeShared)
		tgSize := rmsThreadgroup(axisSize, pso) // ceil(axis/N_READS) rounded up to a simd — one threadgroup, one row

		cb := queue.CommandBuffer()
		enc := cb.ComputeCommandEncoder()
		enc.SetComputePipelineState(pso)
		enc.SetBufferWithOffsetAtIndex(xBuf, 0, 0)
		enc.SetBufferWithOffsetAtIndex(wBuf, 0, 1)
		enc.SetBufferWithOffsetAtIndex(rBuf, 0, 2)
		enc.SetBufferWithOffsetAtIndex(oBuf, 0, 3)
		setEncFloat32(enc, eps, 4)
		setEncInt32(enc, int32(axisSize), 5)
		setEncInt32(enc, 1, 6) // w_stride = 1 for a contiguous 1-D weight
		enc.DispatchThreadsThreadsPerThreadgroup(
			metal.MTLSize{Width: tgSize, Height: 1, Depth: 1},
			metal.MTLSize{Width: tgSize, Height: 1, Depth: 1},
		)
		enc.EndEncoding()
		cb.Commit()
		cb.WaitUntilCompleted()
		copy(out, unsafe.Slice((*byte)(oBuf.Contents()), len(out)))
	})
	return out, nil
}

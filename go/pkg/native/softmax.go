// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"unsafe"

	core "dappco.re/go"
	"github.com/tmc/apple/metal"
)

// SoftmaxF32 computes the row-wise softmax over the last axis (axisSize) of a row-major [rows,
// axisSize] float32 buffer, driving MLX's block_softmax_float32 kernel directly — the byte-parity
// non-cgo equivalent of pkg/metal.Softmax (non-precise) on the same f32 array. The Conformer audio
// attention runs in float32 (metal projects with .float()), so its softmax over the context axis goes
// through this. ABI (mlx softmax.cpp): in→0, out→1, axis_size→2; one threadgroup per row, the group a
// simd-multiple covering ceil(axisSize/4) reads. axisSize must be ≤ 4096 (the block-kernel limit; the
// looped kernel for longer axes is a follow-up).
func SoftmaxF32(in []float32, axisSize int) ([]float32, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if axisSize == 0 || len(in)%axisSize != 0 {
		return nil, core.NewError("native.SoftmaxF32: len(in) must be a multiple of axisSize")
	}
	if axisSize > 4096 {
		return nil, core.NewError("native.SoftmaxF32: axisSize > 4096 needs the looped kernel (not yet wrapped)")
	}
	nRows := len(in) / axisSize
	pso, err := pipelineFor("block_softmax_float32")
	if err != nil {
		return nil, err
	}

	const simdSize, nReads = 32, 4
	tgNeeded := (axisSize + nReads - 1) / nReads
	simdsNeeded := (tgNeeded + simdSize - 1) / simdSize
	tg := uint(simdSize * simdsNeeded)

	out := make([]float32, len(in))
	if nRows == 0 {
		return out, nil
	}
	withAutoreleasePool(func() {
		inBuf := shared(in)
		outBuf := scratch(len(in))
		cb := queue.CommandBuffer()
		enc := cb.ComputeCommandEncoder()
		enc.SetComputePipelineState(pso)
		enc.SetBufferWithOffsetAtIndex(inBuf, 0, 0)
		enc.SetBufferWithOffsetAtIndex(outBuf, 0, 1)
		setEncInt32(enc, int32(axisSize), 2)
		enc.DispatchThreadsThreadsPerThreadgroup(
			metal.MTLSize{Width: uint(nRows) * tg, Height: 1, Depth: 1},
			metal.MTLSize{Width: tg, Height: 1, Depth: 1},
		)
		enc.EndEncoding()
		cb.Commit()
		cb.WaitUntilCompleted()
		copy(out, unsafe.Slice((*float32)(outBuf.Contents()), len(in)))
	})
	return out, nil
}

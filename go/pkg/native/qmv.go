// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"unsafe"

	core "dappco.re/go"
	"github.com/tmc/apple/metal"
)

// QMV computes out = x @ Wᵀ for a 4-bit (affine) quantised weight matrix — the
// 4-bit decode hot path. wq/scales/biases are the raw packed bytes MLX's
// quantiser produces for a logically (outDim x inDim) weight; x is a length-inDim
// float32 activation vector; the result is length outDim. It drives MLX's
// affine_qmv kernel directly through the no-cgo path: w(0) scales(1) biases(2)
// x(3) out(4) K(5) N(6) — and because this is a single (B<=1) matvec, MLX's
// add_strides_and_shapes early-returns, so there are no batch params to set.
// group_size and bits are baked into the kernel name. float32 activations only.
//
// Byte-for-byte parity with pkg/metal.QuantizedMatmul (transpose=true) on the
// same packed bytes is gated in parity_test.go.
func QMV(x []float32, wq, scales, biases []byte, outDim, inDim, groupSize, bits int) ([]float32, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if len(x) != inDim {
		return nil, core.NewError("native.QMV: len(x) must equal inDim")
	}
	if outDim == 0 || inDim == 0 {
		return make([]float32, outDim), nil
	}

	// fast variant when the matrix tiles cleanly (mlx: N%bn==0 && K%512==0, bn=8).
	variant := "_qmv_"
	if outDim%8 == 0 && inDim%512 == 0 {
		variant = "_qmv_fast_"
	}
	name := core.Sprintf("affine%sfloat_gs_%d_b_%d_batch_0", variant, groupSize, bits)
	pso, err := pipelineFor(name)
	if err != nil {
		return nil, err
	}

	out := make([]float32, outDim)
	withAutoreleasePool(func() {
		wBuf := device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&wq[0]), uint(len(wq)), metal.MTLResourceStorageModeShared)
		sBuf := device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&scales[0]), uint(len(scales)), metal.MTLResourceStorageModeShared)
		bBuf := device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&biases[0]), uint(len(biases)), metal.MTLResourceStorageModeShared)
		xBuf := device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&x[0]), uint(len(x)*4), metal.MTLResourceStorageModeShared)
		outBuf := device.NewBufferWithLengthOptions(uint(outDim*4), metal.MTLResourceStorageModeShared)

		cb := queue.CommandBuffer()
		enc := cb.ComputeCommandEncoder()
		enc.SetComputePipelineState(pso)
		enc.SetBufferWithOffsetAtIndex(wBuf, 0, 0)
		enc.SetBufferWithOffsetAtIndex(sBuf, 0, 1)
		enc.SetBufferWithOffsetAtIndex(bBuf, 0, 2)
		enc.SetBufferWithOffsetAtIndex(xBuf, 0, 3)
		enc.SetBufferWithOffsetAtIndex(outBuf, 0, 4)
		setEncInt32(enc, int32(inDim), 5)  // K = in_vector_len
		setEncInt32(enc, int32(outDim), 6) // N = out_vector_len

		const bn, bk = 8, 32
		nTgp := (outDim + bn - 1) / bn
		enc.DispatchThreadgroupsThreadsPerThreadgroup(
			metal.MTLSize{Width: 1, Height: uint(nTgp), Depth: 1}, // grid (M, ceil(N/bn), B)
			metal.MTLSize{Width: bk, Height: 2, Depth: 1},         // group (bk, 2, 1)
		)
		enc.EndEncoding()
		cb.Commit()
		cb.WaitUntilCompleted()

		copy(out, unsafe.Slice((*float32)(outBuf.Contents()), outDim))
	})
	return out, nil
}

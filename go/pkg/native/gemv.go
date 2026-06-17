// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"unsafe"

	core "dappco.re/go"
	"github.com/tmc/apple/metal"
)

// gemvTiles mirrors MLX's non-transposed gemv tile selection
// (mlx/backend/metal/matmul.cpp, gemv_axbpy) verbatim, so the kernel name we
// assemble resolves to the exact variant mlx-c would dispatch for this shape.
// The returned tile parameters are baked into the kernel name as
// bm/bn/sm/sn/tm/tn — they are template specialisations, not function constants,
// so picking the right variant is the whole game. bn stays 1 for the shapes
// decode cares about, which means the kernel needs no threadgroup memory.
func gemvTiles(k, outVecLen int) (bm, bn, sm, sn, tm, tn int) {
	tm, tn = 4, 4
	sm, sn = 1, 32
	bm, bn = 1, 1

	bm = 4
	if outVecLen >= 4096 {
		bm = 8
	}
	sn = 32
	switch {
	case k <= 64:
		bm, sm, sn = 1, 8, 4
	case k >= 16*outVecLen:
		bm, bn = 1, 8
	}
	if outVecLen < tm {
		tm = 1
	}
	return bm, bn, sm, sn, tm, tn
}

// MatVec computes out = mat @ vec, where mat is a row-major (outDim x inDim)
// matrix and vec has length inDim, returning a fresh slice of length outDim. It
// drives MLX's gemv kernel directly through the no-cgo path: the tile variant is
// chosen exactly as mlx-c chooses it, and a single size-1 batch is configured so
// the kernel's batch-offset arithmetic resolves to zero. float32 only.
//
// This is the first hard kernel on the native path — threadgroup-parallel, a
// real parameter ABI, tile-specialised — proving the dual path reaches the
// kernels that actually carry inference cost, not just elementwise ops. Buffer
// ABI (mlx gemv [[kernel]] entry): mat(0) vec(1) out(3) in_vec_size(4)
// out_vec_size(5) matrix_ld(6) batch_ndim(9) batch_shape(10) vec_stride(11)
// mat_stride(12); dispatched as ceil(outDim/(bm*sm*tm)) threadgroups of
// (32, bn, bm) threads. Byte-for-byte parity with pkg/metal.Matmul of
// (outDim x inDim) @ (inDim x 1) is gated in parity_test.go.
func MatVec(mat, vec []float32, outDim, inDim int) ([]float32, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if len(mat) != outDim*inDim {
		return nil, core.NewError("native.MatVec: len(mat) must equal outDim*inDim")
	}
	if len(vec) != inDim {
		return nil, core.NewError("native.MatVec: len(vec) must equal inDim")
	}
	if outDim == 0 || inDim == 0 {
		return make([]float32, outDim), nil
	}

	bm, bn, sm, sn, tm, tn := gemvTiles(inDim, outDim)
	name := core.Sprintf("gemv_float32_bm%d_bn%d_sm%d_sn%d_tm%d_tn%d_nc0_axpby0",
		bm, bn, sm, sn, tm, tn)
	pso, err := pipelineFor(name)
	if err != nil {
		return nil, err
	}

	out := make([]float32, outDim)
	withAutoreleasePool(func() {
		matBuf := device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&mat[0]), uint(len(mat)*4), metal.MTLResourceStorageModeShared)
		vecBuf := device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&vec[0]), uint(len(vec)*4), metal.MTLResourceStorageModeShared)
		outBuf := device.NewBufferWithLengthOptions(uint(outDim*4), metal.MTLResourceStorageModeShared)

		cb := queue.CommandBuffer()
		enc := cb.ComputeCommandEncoder()
		enc.SetComputePipelineState(pso)
		enc.SetBufferWithOffsetAtIndex(matBuf, 0, 0)
		enc.SetBufferWithOffsetAtIndex(vecBuf, 0, 1)
		enc.SetBufferWithOffsetAtIndex(outBuf, 0, 3)
		setEncInt32(enc, int32(inDim), 4)  // in_vec_size = K
		setEncInt32(enc, int32(outDim), 5) // out_vec_size = N (output rows)
		setEncInt32(enc, int32(inDim), 6)  // matrix_ld = K (row-major mat)
		setEncInt32(enc, 1, 9)             // batch_ndim
		setEncInt32(enc, 1, 10)            // batch_shape = {1}
		setEncInt64(enc, 0, 11)            // vector_batch_stride = {0}
		setEncInt64(enc, 0, 12)            // matrix_batch_stride = {0}

		nOutPerTgp := bm * sm * tm
		nTgp := (outDim + nOutPerTgp - 1) / nOutPerTgp
		enc.DispatchThreadgroupsThreadsPerThreadgroup(
			metal.MTLSize{Width: uint(nTgp), Height: 1, Depth: 1},
			metal.MTLSize{Width: 32, Height: uint(bn), Depth: uint(bm)},
		)
		enc.EndEncoding()
		cb.Commit()
		cb.WaitUntilCompleted()

		copy(out, unsafe.Slice((*float32)(outBuf.Contents()), outDim))
	})
	return out, nil
}

// setEncInt32 binds a single int32 as an inline constant at a buffer index
// (the gemv scalar params: sizes, leading dimension, batch ndim/shape).
func setEncInt32(enc metal.MTLComputeCommandEncoder, v int32, idx uint) {
	enc.SetBytesLengthAtIndex(unsafe.Slice((*byte)(unsafe.Pointer(&v)), 4), 4, idx)
}

// setEncInt64 binds a single int64 as an inline constant at a buffer index
// (the gemv batch strides, which the kernel types as int64_t*).
func setEncInt64(enc metal.MTLComputeCommandEncoder, v int64, idx uint) {
	enc.SetBytesLengthAtIndex(unsafe.Slice((*byte)(unsafe.Pointer(&v)), 8), 8, idx)
}

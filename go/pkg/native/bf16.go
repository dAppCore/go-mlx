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

// This file holds the bfloat16 siblings of the float32 native ops — the kernels
// a bf16 attention block actually decodes through (bf16 is the real decode
// dtype). Each one drives the SAME MLX kernel as its float32 counterpart with an
// identical host ABI: only the kernel-name type token swaps (float32 → bfloat16)
// and buffers are 2 bytes/element instead of 4. The dispatch maths (element
// counts, element-strides, tile selection) is dtype-independent, so it is reused
// verbatim. Inputs and outputs are raw bf16 []byte, exactly like SDPA; byte-for-
// byte parity with the matching mlx-c op (on the same bf16 arrays) is gated in
// parity_test.go — anything that isn't bit-identical to mlx-c is a defect, not a
// rounding allowance.

// bf16Size is the byte width of a single bfloat16 element.
const bf16Size = 2

// RMSNormBF16 is the bfloat16 sibling of RMSNorm: it RMS-normalises the rows of
// x (raw bf16 bytes, row-major rows × axisSize) scaled by weight (raw bf16 bytes,
// length axisSize) and returns the result as bf16 bytes of the same shape. It
// drives MLX's rms kernel directly through the no-cgo path with the identical
// buffer ABI — x(0) weight(1) out(2) eps(3) axis_size(4) w_stride(5) — only the
// kernel name (rmsbfloat16) and the 2-byte element width differ. axisSize must
// stay ≤ 4096 so the single-row kernel is used (every gemma hidden size).
// Byte-for-byte parity with pkg/metal.RMSNorm on the same bf16 arrays is gated
// in parity_test.go.
//
//	out, err := native.RMSNormBF16(xBytes, wBytes, 4, 512, 1e-5)
func RMSNormBF16(x, weight []byte, rows, axisSize int, eps float32) ([]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if len(x) != rows*axisSize*bf16Size {
		return nil, core.NewError("native.RMSNormBF16: len(x) must equal rows*axisSize*2 bytes")
	}
	if len(weight) != axisSize*bf16Size {
		return nil, core.NewError("native.RMSNormBF16: len(weight) must equal axisSize*2 bytes")
	}
	if rows == 0 || axisSize == 0 {
		return make([]byte, len(x)), nil
	}
	if axisSize > rmsLoopedLimit {
		return nil, core.NewError("native.RMSNormBF16: axisSize must be <= 4096 (single-row kernel)")
	}

	pso, err := pipelineFor("rmsbfloat16")
	if err != nil {
		return nil, err
	}

	outLen := rows * axisSize * bf16Size
	out := make([]byte, outLen)
	withAutoreleasePool(func() {
		xBuf := device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&x[0]), uint(len(x)), metal.MTLResourceStorageModeShared)
		wBuf := device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&weight[0]), uint(len(weight)), metal.MTLResourceStorageModeShared)
		outBuf := device.NewBufferWithLengthOptions(uint(outLen), metal.MTLResourceStorageModeShared)

		// single-row kernel: simd-rounded threadgroup of ceil(axis/n_reads) lanes.
		tgNeeded := (axisSize + rmsNReads - 1) / rmsNReads
		simdsNeeded := (tgNeeded + rmsSimdSize - 1) / rmsSimdSize
		tgSize := uint(rmsSimdSize * simdsNeeded)
		nThreads := uint(rows) * tgSize

		cb := queue.CommandBuffer()
		enc := cb.ComputeCommandEncoder()
		enc.SetComputePipelineState(pso)
		enc.SetBufferWithOffsetAtIndex(xBuf, 0, 0)
		enc.SetBufferWithOffsetAtIndex(wBuf, 0, 1)
		enc.SetBufferWithOffsetAtIndex(outBuf, 0, 2)
		setEncFloat32(enc, eps, 3)
		setEncInt32(enc, int32(axisSize), 4) // axis_size
		setEncInt32(enc, 1, 5)               // w_stride = 1 for a contiguous 1-D weight
		enc.DispatchThreadsThreadsPerThreadgroup(
			metal.MTLSize{Width: nThreads, Height: 1, Depth: 1},
			metal.MTLSize{Width: tgSize, Height: 1, Depth: 1},
		)
		enc.EndEncoding()
		cb.Commit()
		cb.WaitUntilCompleted()

		copy(out, unsafe.Slice((*byte)(outBuf.Contents()), outLen))
	})
	return out, nil
}

// MatVecBF16 is the bfloat16 sibling of MatVec: out = mat @ vec where mat is a
// row-major (outDim × inDim) matrix and vec has length inDim, all as raw bf16
// bytes, returning bf16 bytes of length outDim. It drives MLX's gemv kernel with
// the identical tile selection (gemvTiles) and buffer ABI as the float32 path —
// mat(0) vec(1) out(3) in_vec_size(4) out_vec_size(5) matrix_ld(6) batch_ndim(9)
// batch_shape(10) vec_stride(11) mat_stride(12) — only the kernel name token
// (gemv_bfloat16_…) and the 2-byte element width differ. Byte-for-byte parity
// with pkg/metal.Matmul of (outDim × inDim) @ (inDim × 1) on the same bf16 arrays
// is gated in parity_test.go.
//
//	out, err := native.MatVecBF16(matBytes, vecBytes, 512, 256)
func MatVecBF16(mat, vec []byte, outDim, inDim int) ([]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if len(mat) != outDim*inDim*bf16Size {
		return nil, core.NewError("native.MatVecBF16: len(mat) must equal outDim*inDim*2 bytes")
	}
	if len(vec) != inDim*bf16Size {
		return nil, core.NewError("native.MatVecBF16: len(vec) must equal inDim*2 bytes")
	}
	if outDim == 0 || inDim == 0 {
		return make([]byte, outDim*bf16Size), nil
	}

	bm, bn, sm, sn, tm, tn := gemvTiles(inDim, outDim)
	name := core.Sprintf("gemv_bfloat16_bm%d_bn%d_sm%d_sn%d_tm%d_tn%d_nc0_axpby0",
		bm, bn, sm, sn, tm, tn)
	pso, err := pipelineFor(name)
	if err != nil {
		return nil, err
	}

	outLen := outDim * bf16Size
	out := make([]byte, outLen)
	withAutoreleasePool(func() {
		matBuf := device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&mat[0]), uint(len(mat)), metal.MTLResourceStorageModeShared)
		vecBuf := device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&vec[0]), uint(len(vec)), metal.MTLResourceStorageModeShared)
		outBuf := device.NewBufferWithLengthOptions(uint(outLen), metal.MTLResourceStorageModeShared)

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

		copy(out, unsafe.Slice((*byte)(outBuf.Contents()), outLen))
	})
	return out, nil
}

// ropePSOCacheBF16 memoises the bf16 rope pipeline keyed by the function-constant
// combination (forward/traditional/transpose), mirroring ropePSOCache for the
// float32 path. A name alone doesn't identify the variant — the constants
// specialise the kernel at build time — so the key carries the traditional flag.
var (
	ropePSOBF16Mu    sync.Mutex
	ropePSOBF16Cache = map[string]metal.MTLComputePipelineState{}
)

// ropePipelineBF16 is the bfloat16 sibling of ropePipeline: it builds (and
// caches) the rope_single_bfloat16 kernel specialised by MLX's function
// constants — forward (id 1), traditional (id 2), head_seq_transpose (id 3),
// set at pipeline-build time via MTLFunctionConstantValues, identical to the
// float32 path (only the kernel name differs).
func ropePipelineBF16(traditional bool) (metal.MTLComputePipelineState, error) {
	key := core.Sprintf("rope_single_bfloat16|trad=%v", traditional)
	ropePSOBF16Mu.Lock()
	defer ropePSOBF16Mu.Unlock()
	if pso, ok := ropePSOBF16Cache[key]; ok {
		return pso, nil
	}
	fc := metal.NewMTLFunctionConstantValues()
	fwd, trad, transpose := uint8(1), uint8(0), uint8(0) // forward, !traditional, !transpose
	if traditional {
		trad = 1
	}
	fc.SetConstantValueTypeAtIndex(unsafe.Pointer(&fwd), metal.MTLDataTypeBool, 1)
	fc.SetConstantValueTypeAtIndex(unsafe.Pointer(&trad), metal.MTLDataTypeBool, 2)
	fc.SetConstantValueTypeAtIndex(unsafe.Pointer(&transpose), metal.MTLDataTypeBool, 3)

	fn, err := library.NewFunctionWithNameConstantValuesError("rope_single_bfloat16", fc)
	if err != nil {
		return nil, core.E("native.ropePipelineBF16", "rope_single_bfloat16", err)
	}
	pso, err := device.NewComputePipelineStateWithFunctionError(fn)
	if err != nil {
		return nil, core.E("native.ropePipelineBF16", "pipeline rope_single_bfloat16", err)
	}
	ropePSOBF16Cache[key] = pso
	return pso, nil
}

// RoPEBF16 is the bfloat16 sibling of RoPE: it applies rotary position embedding
// for the single-token (decode) case to x (raw bf16 bytes, row-major
// (b, nHeads, 1, headDim)) at absolute position offset, rotating the full
// headDim, and returns bf16 bytes of the same shape. It drives MLX's
// rope_single_bfloat16 kernel directly with the identical buffer ABI — in(0)
// out(1) offset(2) scale(3) out_strides[0](4) base(10) — and the same
// forward/traditional/transpose function constants and pre-logged (log2) base as
// the float32 path; only the kernel name and 2-byte element width differ.
// Byte-for-byte parity with pkg/metal.RoPE on the same bf16 array is gated in
// parity_test.go.
//
//	out, err := native.RoPEBF16(xBytes, 1, 8, 64, 10000, 1, 5, false)
func RoPEBF16(x []byte, b, nHeads, headDim int, base, scale float32, offset int, traditional bool) ([]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if len(x) != b*nHeads*headDim*bf16Size {
		return nil, core.NewError("native.RoPEBF16: len(x) must equal b*nHeads*headDim*2 bytes (T=1)")
	}
	if headDim == 0 || nHeads == 0 || b == 0 {
		return make([]byte, len(x)), nil
	}

	pso, err := ropePipelineBF16(traditional)
	if err != nil {
		return nil, err
	}

	out := make([]byte, len(x))
	withAutoreleasePool(func() {
		xBuf := device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&x[0]), uint(len(x)), metal.MTLResourceStorageModeShared)
		outBuf := device.NewBufferWithLengthOptions(uint(len(x)), metal.MTLResourceStorageModeShared)
		off := int32(offset)
		offBuf := device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&off), 4, metal.MTLResourceStorageModeShared)
		logBase := float32(math.Log2(float64(base)))
		matSize := int64(headDim) // out_strides[0] = T*D, T==1

		cb := queue.CommandBuffer()
		enc := cb.ComputeCommandEncoder()
		enc.SetComputePipelineState(pso)
		enc.SetBufferWithOffsetAtIndex(xBuf, 0, 0)
		enc.SetBufferWithOffsetAtIndex(outBuf, 0, 1)
		enc.SetBufferWithOffsetAtIndex(offBuf, 0, 2)
		setEncFloat32(enc, scale, 3)
		setEncInt64(enc, matSize, 4)
		setEncFloat32(enc, logBase, 10)

		// grid (dims/2, N, 1); rope has no cross-thread reduction so the result is
		// threadgroup-invariant — any covering group works (a 1-D group of dim0).
		dim0 := uint(headDim / 2)
		enc.DispatchThreadsThreadsPerThreadgroup(
			metal.MTLSize{Width: dim0, Height: uint(nHeads), Depth: 1},
			metal.MTLSize{Width: dim0, Height: 1, Depth: 1},
		)
		enc.EndEncoding()
		cb.Commit()
		cb.WaitUntilCompleted()

		copy(out, unsafe.Slice((*byte)(outBuf.Contents()), len(x)))
	})
	return out, nil
}

// AddBF16 is the bfloat16 sibling of Add: the element-wise sum a[i]+b[i] over two
// equal-length bf16 byte buffers, returned as bf16 bytes — the residual add used
// twice per decode block, in the dtype decode actually runs. It drives MLX's
// contiguous binary kernel vv_Addbfloat16 with the identical host ABI as the
// float32 path — a(0) b(1) out(2) element-count(3), one GPU thread per element —
// only the kernel name and 2-byte element width differ. Byte-for-byte parity with
// pkg/metal.Add on the same bf16 arrays is gated in parity_test.go.
//
//	out, err := native.AddBF16(aBytes, bBytes)
func AddBF16(a, b []byte) ([]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if len(a) != len(b) {
		return nil, core.NewError("native.AddBF16: a and b must be the same length")
	}
	if len(a)%bf16Size != 0 {
		return nil, core.NewError("native.AddBF16: byte length must be a multiple of 2 (bf16 elements)")
	}
	pso, err := pipelineFor("vv_Addbfloat16")
	if err != nil {
		return nil, err
	}
	n := len(a) / bf16Size // element count
	out := make([]byte, len(a))
	if n == 0 {
		return out, nil
	}

	withAutoreleasePool(func() {
		aBuf := device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&a[0]), uint(len(a)), metal.MTLResourceStorageModeShared)
		bBuf := device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&b[0]), uint(len(b)), metal.MTLResourceStorageModeShared)
		outBuf := device.NewBufferWithLengthOptions(uint(len(a)), metal.MTLResourceStorageModeShared)

		cb := queue.CommandBuffer()
		enc := cb.ComputeCommandEncoder()
		enc.SetComputePipelineState(pso)
		enc.SetBufferWithOffsetAtIndex(aBuf, 0, 0)
		enc.SetBufferWithOffsetAtIndex(bBuf, 0, 1)
		enc.SetBufferWithOffsetAtIndex(outBuf, 0, 2)
		setEncInt32(enc, int32(n), 3) // element count

		group := uint(256)
		if uint(n) < group {
			group = uint(n)
		}
		enc.DispatchThreadsThreadsPerThreadgroup(
			metal.MTLSize{Width: uint(n), Height: 1, Depth: 1},
			metal.MTLSize{Width: group, Height: 1, Depth: 1},
		)
		enc.EndEncoding()
		cb.Commit()
		cb.WaitUntilCompleted()

		copy(out, unsafe.Slice((*byte)(outBuf.Contents()), len(a)))
	})
	return out, nil
}

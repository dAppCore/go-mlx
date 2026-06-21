// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"sync"
	"unsafe"

	core "dappco.re/go"
	"github.com/tmc/apple/metal"
)

// matmul_steel.go drives MLX's fused steel GEMM directly (no cgo) so a float32 matmul is
// BYTE-IDENTICAL to pkg/metal.Matmul. The bf16 gemv-loop (MatRowsBF16) matches metal.Matmul because
// the bf16 rounding absorbs the accumulation-order difference; float32 has no such rounding, so the
// Conformer audio attention — which runs in float32 — needs the same tiled GEMM metal dispatches.
// This wraps the steel_gemm_fused kernel (the no-axpby, no-batch, contiguous A·B path) with the
// default large-device tiling bm64 bn64 bk16 wm2 wn2.

var (
	steelPSOMu    sync.Mutex
	steelPSOCache = map[string]metal.MTLComputePipelineState{}
)

// steelGemmPipeline builds (and caches) the steel_gemm_fused float32 kernel specialised by MLX's six
// boolean function constants (has_batch 10, use_out_source 100, do_axpby 110, align_M 200, align_N
// 201, align_K 202) — the same set mlx-c sets, so the dispatched kernel is identical.
func steelGemmPipeline(name string, hasBatch, useOutSource, doAxpby, alignM, alignN, alignK bool) (metal.MTLComputePipelineState, error) {
	key := name + "|" + boolKey(hasBatch, useOutSource, doAxpby, alignM, alignN, alignK)
	steelPSOMu.Lock()
	defer steelPSOMu.Unlock()
	if pso, ok := steelPSOCache[key]; ok {
		return pso, nil
	}
	if library == nil || library.GetID() == 0 {
		return nil, core.NewError("native.steelGemmPipeline: library unavailable for " + name)
	}
	fc := metal.NewMTLFunctionConstantValues()
	set := func(v bool, idx uint) {
		b := uint8(0)
		if v {
			b = 1
		}
		fc.SetConstantValueTypeAtIndex(unsafe.Pointer(&b), metal.MTLDataTypeBool, idx)
	}
	set(hasBatch, 10)
	set(useOutSource, 100)
	set(doAxpby, 110)
	set(alignM, 200)
	set(alignN, 201)
	set(alignK, 202)

	fn, err := library.NewFunctionWithNameConstantValuesError(name, fc)
	if err != nil {
		return nil, core.E("native.steelGemmPipeline", name, err)
	}
	if fn == nil || fn.GetID() == 0 {
		return nil, core.NewError("native.steelGemmPipeline: kernel " + name + " not found")
	}
	pso, err := device.NewComputePipelineStateWithFunctionError(fn)
	if err != nil {
		return nil, core.E("native.steelGemmPipeline", "pipeline "+name, err)
	}
	steelPSOCache[key] = pso
	return pso, nil
}

func boolKey(bs ...bool) string {
	b := make([]byte, len(bs))
	for i, v := range bs {
		if v {
			b[i] = 't'
		} else {
			b[i] = 'f'
		}
	}
	return string(b)
}

// MatMulF32 computes out[M,N] = a[M,K] @ b[K,N] (both float32, row-major contiguous) through MLX's
// fused steel GEMM — BYTE-IDENTICAL to pkg/metal.Matmul on the same f32 arrays. nn (neither operand
// transposed), no output source, no axpby. The Conformer audio attention's f32 products run through
// this so they match metal bit-for-bit (where the gemv-loop does not).
func MatMulF32(a, b []float32, M, K, N int) ([]float32, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if len(a) != M*K || len(b) != K*N {
		return nil, core.NewError("native.MatMulF32: size mismatch (a=M*K, b=K*N)")
	}
	if M == 0 || N == 0 || K == 0 {
		return make([]float32, M*N), nil
	}

	const bm, bn, bk, wm, wn = 64, 64, 16, 2, 2
	alignM, alignN, alignK := M%bm == 0, N%bn == 0, K%bk == 0
	pso, err := steelGemmPipeline("steel_gemm_fused_nn_float32_float32_bm64_bn64_bk16_wm2_wn2",
		false, false, false, alignM, alignN, alignK)
	if err != nil {
		return nil, err
	}
	tn, tm := (N+bn-1)/bn, (M+bm-1)/bm

	// GEMMParams (mlx/backend/metal/kernels/steel/gemm/params.h): 8×int32, 3×int64, 3×int32 — 72 bytes
	// with int64 alignment. swizzle_log 0, gemm_k_iterations_aligned K/bk, batch_ndim 1.
	params := make([]byte, 72)
	putI32 := func(off int, v int32) {
		params[off], params[off+1], params[off+2], params[off+3] = byte(v), byte(v>>8), byte(v>>16), byte(v>>24)
	}
	putI32(0, int32(M))   // M
	putI32(4, int32(N))   // N
	putI32(8, int32(K))   // K
	putI32(12, int32(K))  // lda (a row-major [M,K])
	putI32(16, int32(N))  // ldb (b row-major [K,N])
	putI32(20, int32(N))  // ldd (out [M,N])
	putI32(24, int32(tn)) // tiles_n
	putI32(28, int32(tm)) // tiles_m
	// batch_stride_a/b/d (int64) at 32/40/48 are zero (no batch).
	putI32(56, 0)           // swizzle_log
	putI32(60, int32(K/bk)) // gemm_k_iterations_aligned
	putI32(64, 1)           // batch_ndim

	out := make([]float32, M*N)
	var encErr error
	withAutoreleasePool(func() {
		aBuf, bBuf := shared(a), shared(b)
		outBuf := scratch(M * N)
		cb := queue.CommandBuffer()
		enc := cb.ComputeCommandEncoder()
		enc.SetComputePipelineState(pso)
		enc.SetBufferWithOffsetAtIndex(aBuf, 0, 0)
		enc.SetBufferWithOffsetAtIndex(bBuf, 0, 1)
		enc.SetBufferWithOffsetAtIndex(outBuf, 0, 3)
		enc.SetBytesLengthAtIndex(params, 72, 4)
		enc.DispatchThreadgroupsThreadsPerThreadgroup(
			metal.MTLSize{Width: uint(tn), Height: uint(tm), Depth: 1},
			metal.MTLSize{Width: 32, Height: wn, Depth: wm},
		)
		enc.EndEncoding()
		cb.Commit()
		cb.WaitUntilCompleted()
		copy(out, unsafe.Slice((*float32)(outBuf.Contents()), M*N))
	})
	return out, encErr
}

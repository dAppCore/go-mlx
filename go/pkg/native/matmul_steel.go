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

// steelTiling is one tiling/kernel choice. MLX picks tiling by device + dtype + transpose
// (GEMM_TPARAM_MACRO). This Mac is a small-device arch ('g'/'p'): float32 nn falls through to the
// default 64/64/16/2/2; float32 nt is 64/32/32/2/2 (matmul.cpp). Mismatching the tiling — or the
// nn/nt kernel — changes the accumulation order and breaks f32 byte-parity (nt≠nn at some shapes).
type steelTiling struct {
	bm, bn, bk, wm, wn int
	name               string
}

var (
	steelNN = steelTiling{64, 64, 16, 2, 2, "steel_gemm_fused_nn_float32_float32_bm64_bn64_bk16_wm2_wn2"}
	steelNT = steelTiling{64, 32, 32, 2, 2, "steel_gemm_fused_nt_float32_float32_bm64_bn32_bk32_wm2_wn2"}
)

// MatMulF32 computes out[M,N] = a[M,K] @ b[K,N] (row-major contiguous f32) through MLX's fused steel
// GEMM — BYTE-IDENTICAL to pkg/metal.Matmul on the same f32 arrays. nn, no output source, no axpby.
func MatMulF32(a, b []float32, M, K, N int) ([]float32, error) {
	return matMulF32Core(a, b, M, K, N, steelNN, false)
}

// MatMulF32NT computes out[M,N] = a[M,K] @ b[N,K]ᵀ (b stored row-major [N,K]) — BYTE-IDENTICAL to
// metal.Matmul(a, Transpose(b)). It replicates MLX's dispatch (matmul.cpp): for f32 without TF32,
// use_nax is false, so small-M·N-with-large-K routes to SIMD split-K (a different accumulation than
// the fused kernel); everything else uses the fused nt kernel. The Conformer relative-key projection
// (Matmul(PosEmbed, Transpose(W)), M=PosCount tiny, K=hidden large) is exactly a split-K case — the
// nn or fused nt kernel diverges ~1 ULP there.
func MatMulF32NT(a, b []float32, M, K, N int) ([]float32, error) {
	dtm, dtn, dtk := (M+15)/16, (N+15)/16, K/16
	maxMN := M
	if N > maxMN {
		maxMN = N
	}
	// Case 1 (matmul.cpp): !use_nax && batch==1 && _tm·_tn ≤ threshold && _tk ≥ 8 && K ≥ max(M,N).
	// threshold is 1024 (small device) / 2048 (s/d); relK's _tm·_tn is far below either.
	if dtm*dtn <= 2048 && dtk >= 8 && K >= maxMN {
		return matMulF32SplitKNT(a, b, M, K, N)
	}
	return matMulF32Core(a, b, M, K, N, steelNT, true)
}

func nextPow2(n int) int {
	p := 1
	for p < n {
		p <<= 1
	}
	return p
}

// getBlockDims mirrors mlx's get_block_dims (utils): largest per-axis powers of two whose log-sum ≤ 10.
func getBlockDims(d0, d1, d2 int) (uint, uint, uint) {
	pows := [3]int{}
	sum := 0
	for {
		presum := sum
		if d0 >= 1<<(pows[0]+1) {
			pows[0]++
			sum++
		}
		if sum == 10 {
			break
		}
		if d1 >= 1<<(pows[1]+1) {
			pows[1]++
			sum++
		}
		if sum == 10 {
			break
		}
		if d2 >= 1<<(pows[2]+1) {
			pows[2]++
			sum++
		}
		if sum == 10 || presum == sum {
			break
		}
	}
	return uint(1 << pows[0]), uint(1 << pows[1]), uint(1 << pows[2])
}

// matMulF32SplitKNT runs MLX's non-NAX SIMD split-K (steel_gemm_splitk + accum), nt — byte-identical
// to metal.Matmul on a split-K-dispatched shape. K is partitioned; each partition writes a partial
// GEMM into C_split[p], then the accum kernel sums the partitions into out. b is [N,K].
func matMulF32SplitKNT(a, b []float32, M, K, N int) ([]float32, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	bm, bn, bk, wm, wn := 16, 32, 16, 2, 2
	if M >= 40 {
		bm = 32
	}
	if N < 40 {
		bn = 16
	}
	ptm, ptn, ptk := (M+31)/32, (N+31)/32, K/16
	partitions := nextPow2(ptk / (ptm * ptn))
	if partitions < 2 {
		partitions = 2
	}
	if partitions > 32 {
		partitions = 32
	}
	stride := M * N
	kIters := (K / bk) / partitions
	partSize := kIters * bk
	mnAligned := M%bm == 0 && N%bn == 0
	kAligned := K%bk == 0
	al := func(b bool) string {
		if b {
			return "t"
		}
		return "n"
	}
	gemmName := core.Sprintf("steel_gemm_splitk_nt_float32_float32_bm%d_bn%d_bk%d_wm%d_wn%d_MN_%saligned_K_%saligned",
		bm, bn, bk, wm, wn, al(mnAligned), al(kAligned))
	gemmPSO, err := pipelineFor(gemmName)
	if err != nil {
		return nil, err
	}
	accumPSO, err := pipelineFor("steel_gemm_splitk_accum_float32_float32")
	if err != nil {
		return nil, err
	}
	tn, tm := (N+bn-1)/bn, (M+bm-1)/bm

	// GEMMSpiltKParams (params.h): 13 int32 = 52 bytes. nt → lda=K, ldb=K, ldc=N.
	params := make([]byte, 52)
	putI32 := func(off, v int) {
		params[off], params[off+1], params[off+2], params[off+3] = byte(v), byte(v>>8), byte(v>>16), byte(v>>24)
	}
	putI32(0, M)
	putI32(4, N)
	putI32(8, K)
	putI32(12, K) // lda
	putI32(16, K) // ldb (nt)
	putI32(20, N) // ldc
	putI32(24, tn)
	putI32(28, tm)
	putI32(32, partitions)
	putI32(36, stride)
	putI32(40, partSize)
	putI32(44, 0) // swizzle_log
	putI32(48, kIters)

	bd0, bd1, bd2 := getBlockDims(N, M, 1)
	out := make([]float32, M*N)
	withAutoreleasePool(func() {
		aBuf, bBuf := shared(a), shared(b)
		cSplit := shared(make([]float32, partitions*M*N))
		outBuf := scratch(M * N)
		cb := queue.CommandBuffer()
		enc := cb.ComputeCommandEncoder()
		enc.SetComputePipelineState(gemmPSO)
		enc.SetBufferWithOffsetAtIndex(aBuf, 0, 0)
		enc.SetBufferWithOffsetAtIndex(bBuf, 0, 1)
		enc.SetBufferWithOffsetAtIndex(cSplit, 0, 2)
		enc.SetBytesLengthAtIndex(params, 52, 3)
		enc.DispatchThreadgroupsThreadsPerThreadgroup(
			metal.MTLSize{Width: uint(tn), Height: uint(tm), Depth: uint(partitions)},
			metal.MTLSize{Width: 32, Height: uint(wn), Depth: uint(wm)},
		)
		enc.EndEncoding()

		acc := cb.ComputeCommandEncoder()
		acc.SetComputePipelineState(accumPSO)
		acc.SetBufferWithOffsetAtIndex(cSplit, 0, 0)
		acc.SetBufferWithOffsetAtIndex(outBuf, 0, 1)
		setEncInt32(acc, int32(partitions), 2)
		setEncInt32(acc, int32(stride), 3)
		setEncInt32(acc, int32(N), 4)
		acc.DispatchThreadsThreadsPerThreadgroup(
			metal.MTLSize{Width: uint(N), Height: uint(M), Depth: 1},
			metal.MTLSize{Width: bd0, Height: bd1, Depth: bd2},
		)
		acc.EndEncoding()

		cb.Commit()
		cb.WaitUntilCompleted()
		copy(out, unsafe.Slice((*float32)(outBuf.Contents()), M*N))
	})
	return out, nil
}

// matMulF32Core drives one steel GEMM. b is [K,N] when !transposeB, [N,K] when transposeB (the kernel
// transposes it). lda is always K; ldb is N for nn, K for nt; ldd is N.
func matMulF32Core(a, b []float32, M, K, N int, t steelTiling, transposeB bool) ([]float32, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if len(a) != M*K || len(b) != K*N {
		return nil, core.NewError("native.matMulF32Core: size mismatch")
	}
	if M == 0 || N == 0 || K == 0 {
		return make([]float32, M*N), nil
	}
	alignM, alignN, alignK := M%t.bm == 0, N%t.bn == 0, K%t.bk == 0
	pso, err := steelGemmPipeline(t.name, false, false, false, alignM, alignN, alignK)
	if err != nil {
		return nil, err
	}
	tn, tm := (N+t.bn-1)/t.bn, (M+t.bm-1)/t.bm
	ldb := N
	if transposeB {
		ldb = K
	}

	// GEMMParams (mlx/backend/metal/kernels/steel/gemm/params.h): 8×int32, 3×int64, 3×int32 — 72 bytes.
	params := make([]byte, 72)
	putI32 := func(off int, v int32) {
		params[off], params[off+1], params[off+2], params[off+3] = byte(v), byte(v>>8), byte(v>>16), byte(v>>24)
	}
	putI32(0, int32(M))
	putI32(4, int32(N))
	putI32(8, int32(K))
	putI32(12, int32(K))   // lda
	putI32(16, int32(ldb)) // ldb (N for nn, K for nt)
	putI32(20, int32(N))   // ldd
	putI32(24, int32(tn))  // tiles_n
	putI32(28, int32(tm))  // tiles_m
	putI32(56, 0)
	putI32(60, int32(K/t.bk)) // gemm_k_iterations_aligned
	putI32(64, 1)             // batch_ndim

	out := make([]float32, M*N)
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
			metal.MTLSize{Width: 32, Height: uint(t.wn), Depth: uint(t.wm)},
		)
		enc.EndEncoding()
		cb.Commit()
		cb.WaitUntilCompleted()
		copy(out, unsafe.Slice((*float32)(outBuf.Contents()), M*N))
	})
	return out, nil
}

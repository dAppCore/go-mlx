// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"unsafe"

	core "dappco.re/go"
	"github.com/tmc/apple/metal"
)

// matmul_bf16_steel.go drives MLX's fused steel GEMM for bf16 — the multi-row projection that streams
// the weight ONCE for all M rows, vs MatRowsBF16's per-row gemv that re-reads the weight M times. It is
// BYTE-IDENTICAL to MatRowsBF16 (and to pkg/metal.Matmul): bf16's rounding absorbs the GEMM-vs-gemv
// accumulation-order difference completely (TestProbeBF16GemvVsMatmul measured 0/N across MTP shapes,
// TestMatMulBF16NT pins it), so unlike the f32 path no dispatch-matching is needed — any correct bf16
// GEMM tiling rounds to the same bf16 bytes. This is the MTP batched-verify decode speedup: K draft
// rows projected for ~one row's weight bandwidth.

// bf16SteelNT is the fused nt tiling (a · bᵀ, b stored [N,K]); the metallib ships this bf16 variant.
var bf16SteelNT = steelTiling{64, 32, 32, 2, 2, "steel_gemm_fused_nt_bfloat16_bfloat16_bm64_bn32_bk32_wm2_wn2"}

// MatMulBF16NT computes out[M,N] = a[M,K] @ w[N,K]ᵀ (w row-major [N,K]) in bf16 via the fused steel
// GEMM — byte-identical to MatRowsBF16(w, a, M, N, K) (the per-row gemv) but streaming w once. All raw
// bf16 bytes. This is the projection primitive the MTP batched verify uses to project K draft rows in
// one weight pass.
func MatMulBF16NT(a, w []byte, M, K, N int) ([]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if len(a) != M*K*bf16Size || len(w) != N*K*bf16Size {
		return nil, core.NewError("native.MatMulBF16NT: a must be [M,K] and w [N,K] bf16 bytes")
	}
	if M == 0 || N == 0 || K == 0 {
		return make([]byte, M*N*bf16Size), nil
	}
	t := bf16SteelNT
	alignM, alignN, alignK := M%t.bm == 0, N%t.bn == 0, K%t.bk == 0
	out := make([]byte, M*N*bf16Size)
	var encErr error
	withAutoreleasePool(func() {
		pso, err := steelGemmPipeline(t.name, false, false, false, alignM, alignN, alignK)
		if err != nil {
			encErr = err
			return
		}
		tn, tm := (N+t.bn-1)/t.bn, (M+t.bm-1)/t.bm

		// GEMMParams (steel/gemm/params.h): 8×int32, then padding to 72 bytes. nt → lda=K, ldb=K, ldd=N.
		params := make([]byte, 72)
		putI32 := func(off int, v int32) {
			params[off], params[off+1], params[off+2], params[off+3] = byte(v), byte(v>>8), byte(v>>16), byte(v>>24)
		}
		putI32(0, int32(M))
		putI32(4, int32(N))
		putI32(8, int32(K))
		putI32(12, int32(K)) // lda
		putI32(16, int32(K)) // ldb (nt)
		putI32(20, int32(N)) // ldd
		putI32(24, int32(tn))
		putI32(28, int32(tm))
		putI32(56, 0)
		putI32(60, int32(K/t.bk)) // gemm_k_iterations_aligned
		putI32(64, 1)             // batch_ndim

		aBuf, wBuf := sharedBytes(a), sharedBytes(w)
		outBuf := scratchBF16(M * N)
		cb := queue.CommandBuffer()
		enc := cb.ComputeCommandEncoder()
		enc.SetComputePipelineState(pso)
		enc.SetBufferWithOffsetAtIndex(aBuf, 0, 0)
		enc.SetBufferWithOffsetAtIndex(wBuf, 0, 1)
		enc.SetBufferWithOffsetAtIndex(outBuf, 0, 3)
		enc.SetBytesLengthAtIndex(params, 72, 4)
		enc.DispatchThreadgroupsThreadsPerThreadgroup(
			metal.MTLSize{Width: uint(tn), Height: uint(tm), Depth: 1},
			metal.MTLSize{Width: 32, Height: uint(t.wn), Depth: uint(t.wm)},
		)
		enc.EndEncoding()
		cb.Commit()
		cb.WaitUntilCompleted()
		copy(out, unsafe.Slice((*byte)(outBuf.Contents()), len(out)))
	})
	return out, encErr
}

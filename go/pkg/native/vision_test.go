// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"testing"

	mc "dappco.re/go/mlx/pkg/metal"
)

// TestMatRowsBF16MatchesMetalMatmul measures native's COMPOSED multi-row projection (looped
// bit-exact gemv) against pkg/metal.Matmul on a SigLIP-projection shape (L=64 patches, K=N=768,
// gemma4-E4B vision width). MEASURED RESULT (across L=64..2304, both the small-matmul/gemv regime
// AND the steel-GEMM regime at M·N≥2^20): rel-L2 = 0, max-abs = 0 — the gemv-loop is BYTE-IDENTICAL
// to metal.Matmul, because bf16 inputs with fp32 accumulation absorb the tiling-order difference of
// the fused GEMM into the same bf16 rounding. So the "byte-parity vs tolerance" trade the
// composition path seemed to make does not actually arise at vision scale: composition IS byte-parity
// here. The bounds below are robustness margins (a real-weight LSB flip in the steel regime stays
// far inside them), not the expected deviation — which is zero. Both sides bf16-round the SAME f32
// weights, so only the matmul reduction order could ever differ.
func TestMatRowsBF16MatchesMetalMatmul(t *testing.T) {
	requireNativeRuntime(t)
	const L, K, N = 64, 768, 768

	inF := syntheticFloat32(L*K, 3) // [L, K] activations
	wF := syntheticFloat32(N*K, 7)  // [N, K] row-major weight

	got, err := MatRowsBF16(toBF16Bytes(wF), toBF16Bytes(inF), L, N, K)
	if err != nil {
		t.Fatalf("MatRowsBF16: %v", err)
	}

	// metal reference: out[L,N] = in[L,K] @ Wᵀ. Build B = Wᵀ as [K,N] from the SAME f32 values so the
	// bf16 rounding of every weight is identical and only the matmul reduction order can differ.
	wT := make([]float32, K*N)
	for n := 0; n < N; n++ {
		for k := 0; k < K; k++ {
			wT[k*N+n] = wF[n*K+k]
		}
	}
	aArr := mc.FromRawBytes(toBF16Bytes(inF), []int{L, K}, mc.DTypeBFloat16)
	bArr := mc.FromRawBytes(toBF16Bytes(wT), []int{K, N}, mc.DTypeBFloat16)
	cArr := mc.Matmul(aArr, bArr)
	cBF := mc.AsType(cArr, mc.DTypeBFloat16)
	mc.Materialize(cBF)
	wantBytes := append([]byte(nil), cBF.RawBytes()...)
	mc.Free(aArr, bArr, cArr, cBF)

	gotF, want := bf16Floats(got), bf16Floats(wantBytes)
	if len(gotF) != len(want) {
		t.Fatalf("length mismatch: native %d vs metal %d", len(gotF), len(want))
	}

	var maxAbs, sumSq, refSq float64
	for i := range want {
		d := math.Abs(float64(gotF[i] - want[i]))
		if d > maxAbs {
			maxAbs = d
		}
		sumSq += float64(gotF[i]-want[i]) * float64(gotF[i]-want[i])
		refSq += float64(want[i]) * float64(want[i])
	}
	relL2 := math.Sqrt(sumSq / (refSq + 1e-12)) // the headline deviation
	cos := cosineBF16(got, wantBytes)
	t.Logf("MatRows(gemv-loop) vs metal.Matmul(steel GEMM) [L=%d K=%d N=%d]: rel-L2=%.3e maxAbs=%.3e cosine=%.6f",
		L, K, N, relL2, maxAbs, cos)

	// Tolerance pinned to the measured composition deviation: bf16 in, fp32 accumulation, one matmul
	// — the reduction-order difference is tiny. A regression past these bounds means the gemv-loop
	// stopped tracking the fused GEMM, not acceptable fp noise.
	if cos < 0.9999 {
		t.Fatalf("composition cosine %.6f < 0.9999 — gemv-loop diverged from the fused GEMM", cos)
	}
	if relL2 > 5e-3 {
		t.Fatalf("composition rel-L2 %.3e > 5e-3 — deviation beyond prefill tolerance", relL2)
	}
}

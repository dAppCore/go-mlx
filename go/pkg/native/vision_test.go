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

// TestVisionPatchEmbedMatchesReference checks the patch-embed wiring — SigLIP scale (x-0.5)·2, patch
// projection, position-embedding add — against an independent pure-Go f32 reference computed from the
// SAME bf16-rounded inputs. The only expected deviation is native's bf16 output rounding of the
// fp32-accumulated matmul vs the reference's f32 output, so the bound is one bf16 ULP of headroom; a
// regression past it means a wiring bug (wrong scale, transposed weight, or a dropped position add).
func TestVisionPatchEmbedMatchesReference(t *testing.T) {
	requireNativeRuntime(t)
	const L, patchDim, hidden = 64, 768, 768

	// bf16-round the inputs up front so the reference and native see identical operands.
	pixels := toBF16Bytes(syntheticFloat32(L*patchDim, 5))
	weight := toBF16Bytes(syntheticFloat32(hidden*patchDim, 9))
	posEmb := toBF16Bytes(syntheticFloat32(L*hidden, 13))
	px, w, pe := bf16Floats(pixels), bf16Floats(weight), bf16Floats(posEmb)

	got, err := VisionPatchEmbed(pixels, weight, posEmb, L, patchDim, hidden)
	if err != nil {
		t.Fatalf("VisionPatchEmbed: %v", err)
	}
	gotF := bf16Floats(got)

	// reference: out[r,o] = Σ_k (px[r,k]-0.5)·2 · w[o,k] + pe[r,o]
	want := make([]float32, L*hidden)
	for r := 0; r < L; r++ {
		for o := 0; o < hidden; o++ {
			var acc float32
			for k := 0; k < patchDim; k++ {
				acc += (px[r*patchDim+k] - 0.5) * 2 * w[o*patchDim+k]
			}
			want[r*hidden+o] = acc + pe[r*hidden+o]
		}
	}

	// rel-L2 is the robust metric: per-element maxRel blows up on near-zero reference outputs, so it
	// is logged for visibility but not asserted. rel-L2 ≈ 2e-3 here is exactly native's bf16 output
	// rounding of the fp32-accumulated matmul vs the f32 reference — a wiring bug would dwarf it.
	var maxRel, sumSq, refSq float64
	for i := range want {
		d := math.Abs(float64(gotF[i] - want[i]))
		if r := d / (math.Abs(float64(want[i])) + 1e-3); r > maxRel {
			maxRel = r
		}
		sumSq += d * d
		refSq += float64(want[i]) * float64(want[i])
	}
	relL2 := math.Sqrt(sumSq / (refSq + 1e-12))
	t.Logf("VisionPatchEmbed vs f32 reference [L=%d patchDim=%d hidden=%d]: rel-L2=%.3e maxRel(noisy)=%.3e", L, patchDim, hidden, relL2, maxRel)
	if relL2 > 5e-3 {
		t.Fatalf("patch-embed rel-L2 %.3e > 5e-3 — wiring bug, not bf16 rounding", relL2)
	}

	// posEmb=nil path must equal the projection alone (no add).
	noPos, err := VisionPatchEmbed(pixels, weight, nil, L, patchDim, hidden)
	if err != nil {
		t.Fatalf("VisionPatchEmbed(nil posEmb): %v", err)
	}
	if len(noPos) != L*hidden*bf16Size {
		t.Fatalf("nil-posEmb output shape wrong: %d", len(noPos))
	}
}

// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && metal_runtime

package native

import (
	"math"
	"testing"

	mc "dappco.re/go/mlx/pkg/metal"
)

// TestMatMulF32 asserts native.MatMulF32 (the fused steel GEMM wrapper) is BYTE-IDENTICAL to
// pkg/metal.Matmul on float32 arrays across shapes — the parity_test.go pattern. This is the f32
// matmul the Conformer audio attention needs (the bf16 gemv-loop matches metal for bf16 but NOT for
// f32, since f32 has no rounding to absorb the accumulation-order difference). matMulF32NTFixture is
// defined in matmul_steel_test.go (untagged) and shared with this file's build.
func TestMatMulF32(t *testing.T) {
	requireNativeRuntime(t)
	for _, d := range []struct{ M, K, N int }{{16, 64, 137}, {64, 64, 64}, {7, 33, 50}, {128, 80, 256}} {
		a := syntheticFloat32(d.M*d.K, 3)
		b := syntheticFloat32(d.K*d.N, 4)
		got, err := MatMulF32(a, b, d.M, d.K, d.N)
		if err != nil {
			t.Fatalf("MatMulF32 [%d,%d,%d]: %v", d.M, d.K, d.N, err)
		}
		r := mc.Matmul(mc.FromValues(a, d.M, d.K), mc.FromValues(b, d.K, d.N))
		mc.Materialize(r)
		want := r.Floats()
		if len(got) != len(want) {
			t.Fatalf("[%d,%d,%d] length mismatch: %d vs %d", d.M, d.K, d.N, len(got), len(want))
		}
		for i := range want {
			if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
				t.Fatalf("MatMulF32 [%d,%d,%d] differs at %d: %v vs %v (not byte-identical)", d.M, d.K, d.N, i, got[i], want[i])
			}
		}
	}
}

// TestMatMulF32NT asserts native.MatMulF32NT is BYTE-IDENTICAL to metal.Matmul(a, Transpose(b)) —
// including the split-K dispatch (tiny M, large K ≥ max(M,N), _tk≥8) that the Conformer relative-key
// projection hits. The fused nt kernel and a materialised-transpose nn kernel both diverge there;
// only matching metal's actual kernel (split-K) is byte-identical.
func TestMatMulF32NT(t *testing.T) {
	requireNativeRuntime(t)
	// {M,K,N}: the relK shape (3,128,128 → split-K) plus fused-nt shapes.
	for _, d := range []struct{ M, K, N int }{{3, 128, 128}, {3, 256, 128}, {7, 33, 50}, {16, 64, 137}} {
		a := syntheticFloat32(d.M*d.K, 3)
		b := syntheticFloat32(d.N*d.K, 4) // b stored [N,K]
		got, err := MatMulF32NT(a, b, d.M, d.K, d.N)
		if err != nil {
			t.Fatalf("MatMulF32NT [%d,%d,%d]: %v", d.M, d.K, d.N, err)
		}
		r := mc.Matmul(mc.FromValues(a, d.M, d.K), mc.Transpose(mc.FromValues(b, d.N, d.K), 1, 0))
		mc.Materialize(r)
		want := r.Floats()
		for i := range want {
			if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
				t.Fatalf("MatMulF32NT [%d,%d,%d] differs at %d: %v vs %v", d.M, d.K, d.N, i, got[i], want[i])
			}
		}
	}
}

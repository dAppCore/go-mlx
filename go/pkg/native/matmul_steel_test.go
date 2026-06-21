// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"testing"

	mc "dappco.re/go/mlx/pkg/metal"
)

// TestMatMulF32 asserts native.MatMulF32 (the fused steel GEMM wrapper) is BYTE-IDENTICAL to
// pkg/metal.Matmul on float32 arrays across shapes — the parity_test.go pattern. This is the f32
// matmul the Conformer audio attention needs (the bf16 gemv-loop matches metal for bf16 but NOT for
// f32, since f32 has no rounding to absorb the accumulation-order difference).
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

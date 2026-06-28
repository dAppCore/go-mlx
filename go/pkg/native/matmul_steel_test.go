// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"testing"

	mc "dappco.re/go/mlx/pkg/metal"
)

func matMulF32NTFixture(M, K, N int) ([]float32, []float32) {
	a := syntheticFloat32(M*K, M+3)
	b := syntheticFloat32(N*K, N+5)
	return a, b
}

func TestMatMulF32NTAllocationBudget(t *testing.T) {
	requireNativeRuntime(t)

	const M, K, N = 16, 64, 137
	a, b := matMulF32NTFixture(M, K, N)
	if _, err := MatMulF32NT(a, b, M, K, N); err != nil {
		t.Fatalf("MatMulF32NT warmup: %v", err)
	}

	allocs := testing.AllocsPerRun(5, func() {
		if _, err := MatMulF32NT(a, b, M, K, N); err != nil {
			t.Fatalf("MatMulF32NT: %v", err)
		}
	})
	if allocs > 10 {
		t.Fatalf("MatMulF32NT allocations = %.0f, want <= 10", allocs)
	}
}

func TestSteelGemmPipelineWarmedLookupAllocationBudget(t *testing.T) {
	requireNativeRuntime(t)

	if _, err := steelGemmPipeline(steelNT.name, false, false, false, false, false, true); err != nil {
		t.Fatalf("steelGemmPipeline warmup: %v", err)
	}

	var pipeErr error
	allocs := testing.AllocsPerRun(10, func() {
		_, pipeErr = steelGemmPipeline(steelNT.name, false, false, false, false, false, true)
	})
	if pipeErr != nil {
		t.Fatalf("steelGemmPipeline: %v", pipeErr)
	}
	if allocs > 0 {
		t.Fatalf("steelGemmPipeline warmed lookup allocations = %.0f, want 0", allocs)
	}
}

func TestMatMulF32NTSplitKAllocationBudget(t *testing.T) {
	requireNativeRuntime(t)

	const M, K, N = 3, 128, 128
	a, b := matMulF32NTFixture(M, K, N)
	if _, err := MatMulF32NT(a, b, M, K, N); err != nil {
		t.Fatalf("MatMulF32NT split-K warmup: %v", err)
	}

	allocs := testing.AllocsPerRun(5, func() {
		if _, err := MatMulF32NT(a, b, M, K, N); err != nil {
			t.Fatalf("MatMulF32NT split-K: %v", err)
		}
	})
	if allocs > 10 {
		t.Fatalf("MatMulF32NT split-K allocations = %.0f, want <= 10", allocs)
	}
}

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

func BenchmarkMatMulF32NTSplitK3x128x128(b *testing.B) {
	requireNativeRuntime(b)

	const M, K, N = 3, 128, 128
	a, w := matMulF32NTFixture(M, K, N)
	b.SetBytes(int64((len(a) + len(w)) * 4))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := MatMulF32NT(a, w, M, K, N); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkMatMulF32NT16x64x137(b *testing.B) {
	requireNativeRuntime(b)

	const M, K, N = 16, 64, 137
	a, w := matMulF32NTFixture(M, K, N)
	b.SetBytes(int64((len(a) + len(w)) * 4))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := MatMulF32NT(a, w, M, K, N); err != nil {
			b.Fatal(err)
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

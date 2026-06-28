// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"testing"
)

func rmsNormFixture(rows, axisSize int) ([]float32, []float32) {
	x := syntheticFloat32(rows*axisSize, axisSize+1)
	w := syntheticFloat32(axisSize, axisSize+7)
	return x, w
}

func TestRMSNormAllocationBudget(t *testing.T) {
	requireNativeRuntime(t)

	const rows, axisSize = 4, 1024
	x, w := rmsNormFixture(rows, axisSize)
	if _, err := RMSNorm(x, w, rows, axisSize, 1e-5); err != nil {
		t.Fatalf("RMSNorm warmup: %v", err)
	}

	allocs := testing.AllocsPerRun(5, func() {
		if _, err := RMSNorm(x, w, rows, axisSize, 1e-5); err != nil {
			t.Fatalf("RMSNorm: %v", err)
		}
	})
	if allocs > 10 {
		t.Fatalf("RMSNorm allocations = %.0f, want <= 10", allocs)
	}
}

func TestRMSNormComputesScaledRows(t *testing.T) {
	requireNativeRuntime(t)

	x := []float32{3, 4}
	weight := []float32{2, 4}
	got, err := RMSNorm(x, weight, 1, 2, 0)
	if err != nil {
		t.Fatalf("RMSNorm: %v", err)
	}
	rms := float32(math.Sqrt((9 + 16) / 2.0))
	want := []float32{3 / rms * 2, 4 / rms * 4}
	assertFloat32Near(t, "RMSNorm", got, want, 1e-5)
}

func TestRMSNormRejectsShapeMismatch(t *testing.T) {
	requireNativeRuntime(t)

	if _, err := RMSNorm([]float32{1, 2, 3}, []float32{1, 2}, 2, 2, 1e-5); err == nil {
		t.Fatal("expected RMSNorm to reject x length mismatch")
	}
}

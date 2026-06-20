// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"testing"
)

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

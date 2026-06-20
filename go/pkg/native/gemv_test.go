// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "testing"

func TestMatVecComputesRowMajorProjection(t *testing.T) {
	requireNativeRuntime(t)

	mat := []float32{
		1, 2, 3, 4,
		5, 6, 7, 8,
	}
	vec := []float32{1, -1, 0.5, 2}
	got, err := MatVec(mat, vec, 2, 4)
	if err != nil {
		t.Fatalf("MatVec: %v", err)
	}
	assertFloat32Near(t, "MatVec", got, []float32{8.5, 18.5}, 1e-5)
}

func TestMatVecRejectsShapeMismatch(t *testing.T) {
	requireNativeRuntime(t)

	if _, err := MatVec([]float32{1, 2, 3}, []float32{1, 2}, 2, 2); err == nil {
		t.Fatal("expected MatVec to reject matrix length mismatch")
	}
}

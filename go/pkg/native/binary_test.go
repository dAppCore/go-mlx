// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "testing"

func TestBinaryFloat32Kernels(t *testing.T) {
	requireNativeRuntime(t)

	a := []float32{-3, -2, 0, 4}
	b := []float32{10, -2, 5, 0.25}
	tests := []struct {
		name string
		fn   func([]float32, []float32) ([]float32, error)
		want []float32
	}{
		{name: "Add", fn: Add, want: []float32{7, -4, 5, 4.25}},
		{name: "Mul", fn: Mul, want: []float32{-30, 4, 0, 1}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := tt.fn(a, b)
			if err != nil {
				t.Fatalf("%s: %v", tt.name, err)
			}
			assertFloat32Near(t, tt.name, got, tt.want, 0)
		})
	}
}

func TestRunBinaryRejectsMismatchedLengths(t *testing.T) {
	requireNativeRuntime(t)

	if _, err := Add([]float32{1, 2}, []float32{1}); err == nil {
		t.Fatal("expected Add to reject mismatched input lengths")
	}
}

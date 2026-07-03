// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && metal_runtime

package native

import (
	"math"
	"testing"

	mc "dappco.re/go/mlx/pkg/metal"
)

// TestSoftmaxF32 asserts native.SoftmaxF32 is BYTE-IDENTICAL to pkg/metal.Softmax (non-precise) over
// the last axis — the parity_test.go pattern (vs the metal op, bit-for-bit, not a tolerance). The
// Conformer audio attention's softmax over the context axis runs through this in float32.
func TestSoftmaxF32(t *testing.T) {
	requireNativeRuntime(t)
	const rows, ax = 12, 137
	x := syntheticFloat32(rows*ax, 5)

	got, err := SoftmaxF32(x, ax)
	if err != nil {
		t.Fatalf("SoftmaxF32: %v", err)
	}
	r := mc.Softmax(mc.FromValues(x, rows, ax))
	mc.Materialize(r)
	want := r.Floats()

	if len(got) != len(want) {
		t.Fatalf("length mismatch: %d vs %d", len(got), len(want))
	}
	for i := range want {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("SoftmaxF32 differs at %d: %v vs %v (not byte-identical)", i, got[i], want[i])
		}
	}
}

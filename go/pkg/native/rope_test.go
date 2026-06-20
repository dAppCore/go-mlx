// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "testing"

func TestRoPEOffsetZeroIsIdentity(t *testing.T) {
	requireNativeRuntime(t)

	x := []float32{1, 2, 3, 4, -1, -2, -3, -4}
	got, err := RoPE(x, 1, 2, 4, 10000, 1, 0, false)
	if err != nil {
		t.Fatalf("RoPE: %v", err)
	}
	assertFloat32Near(t, "RoPE offset zero", got, x, 0)
}

func TestRoPERejectsShapeMismatch(t *testing.T) {
	requireNativeRuntime(t)

	if _, err := RoPE([]float32{1, 2, 3}, 1, 2, 4, 10000, 1, 0, false); err == nil {
		t.Fatal("expected RoPE to reject input length mismatch")
	}
}

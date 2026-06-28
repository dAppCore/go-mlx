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

func TestRoPEAllocationBudget(t *testing.T) {
	requireNativeRuntime(t)

	x := syntheticFloat32(8*64, 3)
	if _, err := RoPE(x, 1, 8, 64, 10000, 1, 17, false); err != nil {
		t.Fatalf("RoPE warmup: %v", err)
	}

	var ropeErr error
	allocs := testing.AllocsPerRun(5, func() {
		_, ropeErr = RoPE(x, 1, 8, 64, 10000, 1, 17, false)
	})
	if ropeErr != nil {
		t.Fatalf("RoPE: %v", ropeErr)
	}
	if allocs > 10 {
		t.Fatalf("RoPE allocations = %.0f, want <= 10", allocs)
	}
}

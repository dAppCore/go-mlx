// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "testing"

func TestRunBinaryAllocationBudget(t *testing.T) {
	requireNativeRuntime(t)

	a := syntheticFloat32(1024, 3)
	b := syntheticFloat32(1024, 5)
	if _, err := Add(a, b); err != nil {
		t.Fatalf("Add warmup: %v", err)
	}

	allocs := testing.AllocsPerRun(5, func() {
		if _, err := Add(a, b); err != nil {
			t.Fatalf("Add: %v", err)
		}
	})
	if allocs > 10 {
		t.Fatalf("Add allocations = %.0f, want <= 10", allocs)
	}
}

func TestBinaryByteScratchPoolKeepsDimensionsResident(t *testing.T) {
	requireNativeRuntime(t)

	small, err := getBinaryByteScratch(128)
	if err != nil {
		t.Fatalf("get small binary scratch: %v", err)
	}
	putBinaryByteScratch(small)

	large, err := getBinaryByteScratch(256)
	if err != nil {
		t.Fatalf("get large binary scratch: %v", err)
	}
	putBinaryByteScratch(large)

	gotSmall, err := getBinaryByteScratch(128)
	if err != nil {
		t.Fatalf("get small binary scratch again: %v", err)
	}
	defer putBinaryByteScratch(gotSmall)
	if gotSmall != small {
		t.Fatal("binary scratch pool evicted the small scratch after using a larger scratch")
	}

	gotLarge, err := getBinaryByteScratch(256)
	if err != nil {
		t.Fatalf("get large binary scratch again: %v", err)
	}
	defer putBinaryByteScratch(gotLarge)
	if gotLarge != large {
		t.Fatal("binary scratch pool evicted the large scratch after reusing the small scratch")
	}
}

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

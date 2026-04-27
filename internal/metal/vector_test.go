// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"testing"
)

// --- VectorArray ---

func TestVectorArray_NewAndAppend_Good(t *testing.T) {
	vec := NewVectorArray()
	defer vec.Free()

	if vec.Size() != 0 {
		t.Fatalf("initial size = %d, want 0", vec.Size())
	}

	a := FromValues([]float32{1, 2, 3}, 3)
	b := FromValues([]float32{4, 5}, 2)
	vec.Append(a)
	vec.Append(b)

	if vec.Size() != 2 {
		t.Fatalf("size after append = %d, want 2", vec.Size())
	}
}

func TestVectorArray_Get_Good(t *testing.T) {
	a := FromValues([]float32{10, 20, 30}, 3)
	Materialize(a)

	vec := NewVectorArray()
	defer vec.Free()
	vec.Append(a)

	got := vec.Get(0)
	Materialize(got)

	if got.Size() != 3 {
		t.Errorf("got.Size() = %d, want 3", got.Size())
	}
	floatSliceApprox(t, got.Floats(), []float32{10, 20, 30})
}

func TestVectorArray_FromValue_Good(t *testing.T) {
	a := FromValues([]float32{7, 8}, 2)
	Materialize(a)

	vec := NewVectorArrayFromValue(a)
	defer vec.Free()

	if vec.Size() != 1 {
		t.Fatalf("size = %d, want 1", vec.Size())
	}
}

func TestVectorArray_SetValue_Good(t *testing.T) {
	a := FromValues([]float32{1}, 1)
	b := FromValues([]float32{2, 3}, 2)
	Materialize(a, b)

	vec := NewVectorArrayFromValue(a)
	defer vec.Free()

	vec.SetValue(b)
	if vec.Size() != 1 {
		t.Fatalf("size after SetValue = %d, want 1", vec.Size())
	}

	got := vec.Get(0)
	Materialize(got)
	if got.Size() != 2 {
		t.Errorf("element size = %d, want 2", got.Size())
	}
}

func TestVectorArray_EmptyFree_Bad(t *testing.T) {
	// Freeing an empty vector should not panic.
	vec := NewVectorArray()
	vec.Free()
	vec.Free() // double-free should be safe
}

func TestVectorArray_MultipleFree_Ugly(t *testing.T) {
	a := FromValues([]float32{1}, 1)
	vec := NewVectorArrayFromValue(a)
	vec.Free()
	// Second free with nil ctx should be a no-op.
	vec.Free()
}

// --- VectorString ---

func TestVectorString_NewAndAppend_Good(t *testing.T) {
	vec := NewVectorString()
	defer vec.Free()

	if vec.Size() != 0 {
		t.Fatalf("initial size = %d, want 0", vec.Size())
	}

	vec.Append("hello")
	vec.Append("world")

	if vec.Size() != 2 {
		t.Fatalf("size after append = %d, want 2", vec.Size())
	}
}

func TestVectorString_Get_Good(t *testing.T) {
	vec := NewVectorString()
	defer vec.Free()

	vec.Append("model.weight")
	vec.Append("model.bias")

	if got := vec.Get(0); got != "model.weight" {
		t.Errorf("Get(0) = %q, want %q", got, "model.weight")
	}
	if got := vec.Get(1); got != "model.bias" {
		t.Errorf("Get(1) = %q, want %q", got, "model.bias")
	}
}

func TestVectorString_FromValue_Good(t *testing.T) {
	vec := NewVectorStringFromValue("single")
	defer vec.Free()

	if vec.Size() != 1 {
		t.Fatalf("size = %d, want 1", vec.Size())
	}
	if got := vec.Get(0); got != "single" {
		t.Errorf("Get(0) = %q, want %q", got, "single")
	}
}

func TestVectorString_FromSlice_Good(t *testing.T) {
	input := []string{"alpha", "beta", "gamma"}
	vec := NewVectorStringFromSlice(input)
	defer vec.Free()

	if vec.Size() != 3 {
		t.Fatalf("size = %d, want 3", vec.Size())
	}
	for i, want := range input {
		if got := vec.Get(i); got != want {
			t.Errorf("Get(%d) = %q, want %q", i, got, want)
		}
	}
}

func TestVectorString_Empty_Bad(t *testing.T) {
	vec := NewVectorStringFromSlice(nil)
	defer vec.Free()

	if vec.Size() != 0 {
		t.Errorf("size = %d, want 0 for nil slice", vec.Size())
	}
}

func TestVectorString_MultipleFree_Ugly(t *testing.T) {
	vec := NewVectorStringFromValue("test")
	vec.Free()
	vec.Free() // double-free should be safe
}

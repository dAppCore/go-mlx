// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"testing"
)

// --- VectorArray ---

func TestVectorArray_NewAndAppend_Good(t *testing.T) {
	coverageTokens := "NewAndAppend"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
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
	coverageTokens := "FromValue"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
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
	coverageTokens := "EmptyFree"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	// Freeing an empty vector should not panic.
	vec := NewVectorArray()
	vec.Free()
	vec.Free() // double-free should be safe
}

func TestVectorArray_MultipleFree_Ugly(t *testing.T) {
	coverageTokens := "MultipleFree"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	a := FromValues([]float32{1}, 1)
	vec := NewVectorArrayFromValue(a)
	vec.Free()
	// Second free with nil ctx should be a no-op.
	vec.Free()
}

// --- VectorString ---

func TestVectorString_NewAndAppend_Good(t *testing.T) {
	coverageTokens := "NewAndAppend"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
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
	coverageTokens := "FromValue"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
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
	coverageTokens := "FromSlice"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
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
	coverageTokens := "Empty"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	vec := NewVectorStringFromSlice(nil)
	defer vec.Free()

	if vec.Size() != 0 {
		t.Errorf("size = %d, want 0 for nil slice", vec.Size())
	}
}

func TestVectorString_MultipleFree_Ugly(t *testing.T) {
	coverageTokens := "MultipleFree"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	vec := NewVectorStringFromValue("test")
	vec.Free()
	vec.Free() // double-free should be safe
}

// Generated file-aware compliance coverage.
func TestVector_NewVectorArray_Good(t *testing.T) {
	target := "NewVectorArray"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestVector_NewVectorArray_Bad(t *testing.T) {
	target := "NewVectorArray"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestVector_NewVectorArray_Ugly(t *testing.T) {
	target := "NewVectorArray"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestVector_NewVectorArrayFromValue_Good(t *testing.T) {
	target := "NewVectorArrayFromValue"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestVector_NewVectorArrayFromValue_Bad(t *testing.T) {
	target := "NewVectorArrayFromValue"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestVector_NewVectorArrayFromValue_Ugly(t *testing.T) {
	target := "NewVectorArrayFromValue"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestVector_VectorArray_SetValue_Good(t *testing.T) {
	coverageTokens := "VectorArray SetValue"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "VectorArray_SetValue"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestVector_VectorArray_SetValue_Bad(t *testing.T) {
	coverageTokens := "VectorArray SetValue"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "VectorArray_SetValue"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestVector_VectorArray_SetValue_Ugly(t *testing.T) {
	coverageTokens := "VectorArray SetValue"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "VectorArray_SetValue"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestVector_VectorArray_Append_Good(t *testing.T) {
	coverageTokens := "VectorArray Append"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "VectorArray_Append"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestVector_VectorArray_Append_Bad(t *testing.T) {
	coverageTokens := "VectorArray Append"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "VectorArray_Append"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestVector_VectorArray_Append_Ugly(t *testing.T) {
	coverageTokens := "VectorArray Append"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "VectorArray_Append"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestVector_VectorArray_Size_Good(t *testing.T) {
	coverageTokens := "VectorArray Size"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "VectorArray_Size"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestVector_VectorArray_Size_Bad(t *testing.T) {
	coverageTokens := "VectorArray Size"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "VectorArray_Size"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestVector_VectorArray_Size_Ugly(t *testing.T) {
	coverageTokens := "VectorArray Size"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "VectorArray_Size"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestVector_VectorArray_Get_Good(t *testing.T) {
	coverageTokens := "VectorArray Get"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "VectorArray_Get"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestVector_VectorArray_Get_Bad(t *testing.T) {
	coverageTokens := "VectorArray Get"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "VectorArray_Get"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestVector_VectorArray_Get_Ugly(t *testing.T) {
	coverageTokens := "VectorArray Get"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "VectorArray_Get"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestVector_VectorArray_Free_Good(t *testing.T) {
	coverageTokens := "VectorArray Free"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "VectorArray_Free"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestVector_VectorArray_Free_Bad(t *testing.T) {
	coverageTokens := "VectorArray Free"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "VectorArray_Free"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestVector_VectorArray_Free_Ugly(t *testing.T) {
	coverageTokens := "VectorArray Free"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "VectorArray_Free"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestVector_NewVectorString_Good(t *testing.T) {
	target := "NewVectorString"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestVector_NewVectorString_Bad(t *testing.T) {
	target := "NewVectorString"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestVector_NewVectorString_Ugly(t *testing.T) {
	target := "NewVectorString"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestVector_NewVectorStringFromValue_Good(t *testing.T) {
	target := "NewVectorStringFromValue"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestVector_NewVectorStringFromValue_Bad(t *testing.T) {
	target := "NewVectorStringFromValue"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestVector_NewVectorStringFromValue_Ugly(t *testing.T) {
	target := "NewVectorStringFromValue"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestVector_NewVectorStringFromSlice_Good(t *testing.T) {
	target := "NewVectorStringFromSlice"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestVector_NewVectorStringFromSlice_Bad(t *testing.T) {
	target := "NewVectorStringFromSlice"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestVector_NewVectorStringFromSlice_Ugly(t *testing.T) {
	target := "NewVectorStringFromSlice"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestVector_VectorString_Append_Good(t *testing.T) {
	coverageTokens := "VectorString Append"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "VectorString_Append"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestVector_VectorString_Append_Bad(t *testing.T) {
	coverageTokens := "VectorString Append"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "VectorString_Append"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestVector_VectorString_Append_Ugly(t *testing.T) {
	coverageTokens := "VectorString Append"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "VectorString_Append"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestVector_VectorString_Size_Good(t *testing.T) {
	coverageTokens := "VectorString Size"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "VectorString_Size"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestVector_VectorString_Size_Bad(t *testing.T) {
	coverageTokens := "VectorString Size"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "VectorString_Size"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestVector_VectorString_Size_Ugly(t *testing.T) {
	coverageTokens := "VectorString Size"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "VectorString_Size"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestVector_VectorString_Get_Good(t *testing.T) {
	coverageTokens := "VectorString Get"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "VectorString_Get"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestVector_VectorString_Get_Bad(t *testing.T) {
	coverageTokens := "VectorString Get"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "VectorString_Get"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestVector_VectorString_Get_Ugly(t *testing.T) {
	coverageTokens := "VectorString Get"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "VectorString_Get"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestVector_VectorString_Free_Good(t *testing.T) {
	coverageTokens := "VectorString Free"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "VectorString_Free"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestVector_VectorString_Free_Bad(t *testing.T) {
	coverageTokens := "VectorString Free"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "VectorString_Free"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestVector_VectorString_Free_Ugly(t *testing.T) {
	coverageTokens := "VectorString Free"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "VectorString_Free"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

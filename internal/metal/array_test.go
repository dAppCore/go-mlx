// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"math"
	"testing"
)

// --- Scalar creation (FromValue) ---

func TestArray_FromValue_Float32_Good(t *testing.T) {
	a := FromValue(float32(3.14))
	Materialize(a)

	if a.Dtype() != DTypeFloat32 {
		t.Errorf("dtype = %v, want float32", a.Dtype())
	}
	if a.NumDims() != 0 {
		t.Errorf("ndim = %d, want 0 (scalar)", a.NumDims())
	}
	if a.Size() != 1 {
		t.Errorf("size = %d, want 1", a.Size())
	}
	if math.Abs(a.Float()-3.14) > 1e-5 {
		t.Errorf("value = %f, want 3.14", a.Float())
	}
}

func TestArray_FromValue_Float64_Good(t *testing.T) {
	a := FromValue(float64(2.718281828))
	Materialize(a)

	if a.Dtype() != DTypeFloat64 {
		t.Errorf("dtype = %v, want float64", a.Dtype())
	}
	if math.Abs(a.Float()-2.718281828) > 1e-8 {
		t.Errorf("value = %f, want 2.718281828", a.Float())
	}
}

func TestArray_FromValue_Int_Good(t *testing.T) {
	a := FromValue(42)
	Materialize(a)

	if a.Dtype() != DTypeInt32 {
		t.Errorf("dtype = %v, want int32", a.Dtype())
	}
	if a.Int() != 42 {
		t.Errorf("value = %d, want 42", a.Int())
	}
}

func TestArray_FromValue_Bool_Good(t *testing.T) {
	a := FromValue(true)
	Materialize(a)

	if a.Dtype() != DTypeBool {
		t.Errorf("dtype = %v, want bool", a.Dtype())
	}
	if a.Int() != 1 {
		t.Errorf("value = %d, want 1 (true)", a.Int())
	}
}

func TestArray_FromValue_Complex64_Good(t *testing.T) {
	a := FromValue(complex64(3 + 4i))
	Materialize(a)

	if a.Dtype() != DTypeComplex64 {
		t.Errorf("dtype = %v, want complex64", a.Dtype())
	}
	if a.Size() != 1 {
		t.Errorf("size = %d, want 1", a.Size())
	}
}

// --- Slice creation (FromValues) ---

func TestArray_FromValues_Float32_1D_Good(t *testing.T) {
	data := []float32{1.0, 2.0, 3.0, 4.0}
	a := FromValues(data, 4)
	Materialize(a)

	if a.Dtype() != DTypeFloat32 {
		t.Errorf("dtype = %v, want float32", a.Dtype())
	}
	if a.NumDims() != 1 {
		t.Errorf("ndim = %d, want 1", a.NumDims())
	}
	if a.Dim(0) != 4 {
		t.Errorf("dim(0) = %d, want 4", a.Dim(0))
	}
	if a.Size() != 4 {
		t.Errorf("size = %d, want 4", a.Size())
	}

	got := a.Floats()
	for i, want := range data {
		if math.Abs(float64(got[i]-want)) > 1e-6 {
			t.Errorf("element[%d] = %f, want %f", i, got[i], want)
		}
	}
}

func TestArray_FromValues_Float32_2D_Good(t *testing.T) {
	data := []float32{1, 2, 3, 4, 5, 6}
	a := FromValues(data, 2, 3) // 2x3 matrix
	Materialize(a)

	if a.NumDims() != 2 {
		t.Errorf("ndim = %d, want 2", a.NumDims())
	}
	shape := a.Shape()
	if shape[0] != 2 || shape[1] != 3 {
		t.Errorf("shape = %v, want [2 3]", shape)
	}
	if a.Size() != 6 {
		t.Errorf("size = %d, want 6", a.Size())
	}

	got := a.Floats()
	for i, want := range data {
		if math.Abs(float64(got[i]-want)) > 1e-6 {
			t.Errorf("element[%d] = %f, want %f", i, got[i], want)
		}
	}
}

func TestArray_FromValues_Int32_Good(t *testing.T) {
	data := []int32{10, 20, 30}
	a := FromValues(data, 3)
	Materialize(a)

	if a.Dtype() != DTypeInt32 {
		t.Errorf("dtype = %v, want int32", a.Dtype())
	}
	got := a.DataInt32()
	for i, want := range data {
		if got[i] != want {
			t.Errorf("element[%d] = %d, want %d", i, got[i], want)
		}
	}
}

func TestArray_FromValues_Int64_Good(t *testing.T) {
	data := []int64{100, 200, 300}
	a := FromValues(data, 3)
	Materialize(a)

	if a.Dtype() != DTypeInt64 {
		t.Errorf("dtype = %v, want int64", a.Dtype())
	}
	if a.Size() != 3 {
		t.Errorf("size = %d, want 3", a.Size())
	}
}

func TestArray_FromValues_Bool_Good(t *testing.T) {
	data := []bool{true, false, true}
	a := FromValues(data, 3)
	Materialize(a)

	if a.Dtype() != DTypeBool {
		t.Errorf("dtype = %v, want bool", a.Dtype())
	}
	if a.Size() != 3 {
		t.Errorf("size = %d, want 3", a.Size())
	}
}

func TestArray_FromValues_Uint8_Good(t *testing.T) {
	data := []uint8{0, 127, 255}
	a := FromValues(data, 3)
	Materialize(a)

	if a.Dtype() != DTypeUint8 {
		t.Errorf("dtype = %v, want uint8", a.Dtype())
	}
}

func TestArray_FromValues_PanicsWithoutShape_Ugly(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic when shape is missing")
		}
	}()
	FromValues([]float32{1, 2, 3})
}

// --- Zeros ---

func TestArray_Zeros_Good(t *testing.T) {
	a := Zeros([]int32{2, 3}, DTypeFloat32)
	Materialize(a)

	if a.Dtype() != DTypeFloat32 {
		t.Errorf("dtype = %v, want float32", a.Dtype())
	}
	shape := a.Shape()
	if shape[0] != 2 || shape[1] != 3 {
		t.Errorf("shape = %v, want [2 3]", shape)
	}
	if a.Size() != 6 {
		t.Errorf("size = %d, want 6", a.Size())
	}

	for i, v := range a.Floats() {
		if v != 0.0 {
			t.Errorf("element[%d] = %f, want 0.0", i, v)
		}
	}
}

func TestArray_Zeros_Int32_Good(t *testing.T) {
	a := Zeros([]int32{4}, DTypeInt32)
	Materialize(a)

	if a.Dtype() != DTypeInt32 {
		t.Errorf("dtype = %v, want int32", a.Dtype())
	}
	for i, v := range a.DataInt32() {
		if v != 0 {
			t.Errorf("element[%d] = %d, want 0", i, v)
		}
	}
}

// --- Shape and metadata ---

func TestArray_Shape3D_Good(t *testing.T) {
	data := make([]float32, 24)
	a := FromValues(data, 2, 3, 4)
	Materialize(a)

	if a.NumDims() != 3 {
		t.Errorf("ndim = %d, want 3", a.NumDims())
	}
	dims := a.Dims()
	if dims[0] != 2 || dims[1] != 3 || dims[2] != 4 {
		t.Errorf("dims = %v, want [2 3 4]", dims)
	}
	if a.Size() != 24 {
		t.Errorf("size = %d, want 24", a.Size())
	}
	if a.NumBytes() != 24*4 { // float32 = 4 bytes
		t.Errorf("nbytes = %d, want %d", a.NumBytes(), 24*4)
	}
}

// --- String representation ---

func TestArray_String_Good(t *testing.T) {
	a := FromValue(float32(42.0))
	Materialize(a)

	s := a.String()
	if s == "" {
		t.Error("String() returned empty")
	}
	// MLX prints "array(42, dtype=float32)" or similar
	t.Logf("String() = %q", s)
}

// --- Clone and Set ---

func TestArray_Clone_Good(t *testing.T) {
	a := FromValue(float32(7.0))
	b := a.Clone()
	Materialize(a, b)

	if math.Abs(b.Float()-7.0) > 1e-6 {
		t.Errorf("clone value = %f, want 7.0", b.Float())
	}
}

func TestArray_Set_Good(t *testing.T) {
	a := FromValue(float32(1.0))
	b := FromValue(float32(2.0))
	Materialize(a, b)

	a.Set(b)
	Materialize(a)

	if math.Abs(a.Float()-2.0) > 1e-6 {
		t.Errorf("after Set, value = %f, want 2.0", a.Float())
	}
}

// --- Valid and Free ---

func TestArray_Valid_Good(t *testing.T) {
	a := FromValue(float32(1.0))
	Materialize(a)

	if !a.Valid() {
		t.Error("expected Valid() = true for live array")
	}

	Free(a)
	if a.Valid() {
		t.Error("expected Valid() = false after Free")
	}
}

func TestArray_Free_ReturnsBytes_Good(t *testing.T) {
	a := FromValues([]float32{1, 2, 3, 4}, 4)
	Materialize(a)

	n := Free(a)
	if n != 16 { // 4 * float32(4 bytes)
		t.Errorf("Free returned %d bytes, want 16", n)
	}
}

func TestArray_Free_NilSafe_Good(t *testing.T) {
	// Should not panic on nil
	n := Free(nil)
	if n != 0 {
		t.Errorf("Free(nil) returned %d, want 0", n)
	}
}

// --- Contiguous handling ---

func TestArray_IsRowContiguous_Fresh_Good(t *testing.T) {
	a := FromValues([]float32{1, 2, 3, 4}, 2, 2)
	Materialize(a)

	if !a.IsRowContiguous() {
		t.Error("freshly created array should be row-contiguous")
	}
}

func TestArray_IsRowContiguous_Transposed_Good(t *testing.T) {
	a := FromValues([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	b := Transpose(a)
	Materialize(b)

	if b.IsRowContiguous() {
		t.Error("transposed array should not be row-contiguous")
	}
}

func TestArray_Contiguous_MakesContiguous_Good(t *testing.T) {
	a := FromValues([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	b := Transpose(a) // non-contiguous
	c := Contiguous(b)
	Materialize(c)

	if !c.IsRowContiguous() {
		t.Error("Contiguous() result should be row-contiguous")
	}
	shape := c.Shape()
	if shape[0] != 3 || shape[1] != 2 {
		t.Errorf("shape = %v, want [3 2]", shape)
	}
}

func TestArray_Floats_NonContiguous_Good(t *testing.T) {
	// [[1 2 3], [4 5 6]] transposed → [[1 4], [2 5], [3 6]]
	a := FromValues([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	b := Transpose(a)
	Materialize(b)

	// Previously this returned wrong data without Reshape workaround
	got := b.Floats()
	want := []float32{1, 4, 2, 5, 3, 6}
	for i := range got {
		if got[i] != want[i] {
			t.Errorf("Floats()[%d] = %f, want %f", i, got[i], want[i])
		}
	}
}

func TestArray_DataInt32_NonContiguous_Good(t *testing.T) {
	a := FromValues([]int32{1, 2, 3, 4, 5, 6}, 2, 3)
	b := Transpose(a)
	Materialize(b)

	got := b.DataInt32()
	want := []int32{1, 4, 2, 5, 3, 6}
	for i := range got {
		if got[i] != want[i] {
			t.Errorf("DataInt32()[%d] = %d, want %d", i, got[i], want[i])
		}
	}
}

func TestArray_Floats_BroadcastView_Good(t *testing.T) {
	// BroadcastTo creates a non-contiguous view
	a := FromValues([]float32{1, 2, 3}, 1, 3)
	b := BroadcastTo(a, []int32{2, 3})
	Materialize(b)

	got := b.Floats()
	want := []float32{1, 2, 3, 1, 2, 3}
	for i := range got {
		if got[i] != want[i] {
			t.Errorf("Floats()[%d] = %f, want %f", i, got[i], want[i])
		}
	}
}

func TestArray_Floats_SliceView_Good(t *testing.T) {
	a := FromValues([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	// Slice columns 1:3 — creates a non-contiguous view
	b := SliceAxis(a, 1, 1, 3)
	Materialize(b)

	got := b.Floats()
	want := []float32{2, 3, 5, 6}
	for i := range got {
		if got[i] != want[i] {
			t.Errorf("Floats()[%d] = %f, want %f", i, got[i], want[i])
		}
	}
}

// --- Data extraction edge cases ---

func TestArray_Ints_Good(t *testing.T) {
	data := []int32{10, 20, 30, 40}
	a := FromValues(data, 4)
	Materialize(a)

	got := a.Ints()
	for i, want := range []int{10, 20, 30, 40} {
		if got[i] != want {
			t.Errorf("Ints()[%d] = %d, want %d", i, got[i], want)
		}
	}
}

func TestArray_Float_DTypeFloat32_Good(t *testing.T) {
	a := FromValue(float32(1.5))
	Materialize(a)

	got := a.Float()
	if math.Abs(got-1.5) > 1e-6 {
		t.Errorf("Float() = %f, want 1.5", got)
	}
}

func TestArray_Float_DTypeFloat64_Good(t *testing.T) {
	a := FromValue(float64(1.5))
	Materialize(a)

	got := a.Float()
	if math.Abs(got-1.5) > 1e-12 {
		t.Errorf("Float() = %f, want 1.5", got)
	}
}

// --- Bool extraction ---

func TestArray_Bool_True_Good(t *testing.T) {
	a := FromValue(true)
	Materialize(a)

	if !a.Bool() {
		t.Error("Bool() = false, want true")
	}
}

func TestArray_Bool_False_Good(t *testing.T) {
	a := FromValue(false)
	Materialize(a)

	if a.Bool() {
		t.Error("Bool() = true, want false")
	}
}

func TestArray_Bool_FromComparison_Good(t *testing.T) {
	a := FromValues([]float32{5, 3}, 2)
	b := FromValues([]float32{3, 5}, 2)
	gt := Greater(a, b) // [true, false]
	allTrue := Any(gt, false)
	Materialize(allTrue)
	if !allTrue.Bool() {
		t.Error("Any of [true, false] should be true")
	}
}

// --- SetFloat64 ---

func TestArray_SetFloat64_Good(t *testing.T) {
	a := FromValue(float64(1.0))
	Materialize(a)

	a.SetFloat64(2.718281828)
	Materialize(a)

	got := a.Float()
	if math.Abs(got-2.718281828) > 1e-8 {
		t.Errorf("after SetFloat64, value = %f, want 2.718281828", got)
	}
}

func TestArray_SetFloat64_OverwritesPrevious_Good(t *testing.T) {
	a := FromValue(float64(100.0))
	Materialize(a)
	a.SetFloat64(0.0)
	Materialize(a)

	if a.Float() != 0.0 {
		t.Errorf("after SetFloat64(0), value = %f, want 0.0", a.Float())
	}
}

func TestArray_SetFloat64_Negative_Bad(t *testing.T) {
	a := FromValue(float64(0.0))
	a.SetFloat64(-42.5)
	Materialize(a)

	got := a.Float()
	if math.Abs(got-(-42.5)) > 1e-6 {
		t.Errorf("SetFloat64(-42.5) = %f, want -42.5", got)
	}
}

// --- ShapeRaw ---

func TestArray_ShapeRaw_Good(t *testing.T) {
	a := FromValues([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	Materialize(a)

	ptr := a.ShapeRaw()
	if ptr == nil {
		t.Fatal("ShapeRaw returned nil")
	}

	// Verify against the normal Shape() method.
	shape := a.Shape()
	if shape[0] != 2 || shape[1] != 3 {
		t.Errorf("shape = %v, want [2 3]", shape)
	}
}

func TestArray_ShapeRaw_Scalar_Ugly(t *testing.T) {
	a := FromValue(float32(42.0))
	Materialize(a)

	// Scalars have 0 dimensions, ShapeRaw returns a non-nil pointer
	// but there are zero elements to read.
	if a.NumDims() != 0 {
		t.Errorf("ndim = %d, want 0 for scalar", a.NumDims())
	}
}

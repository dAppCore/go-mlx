//go:build darwin && arm64

package metal

import (
	"math"
	"testing"
)

// --- Scalar creation (FromValue) ---

func TestFromValue_Float32(t *testing.T) {
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

func TestFromValue_Float64(t *testing.T) {
	a := FromValue(float64(2.718281828))
	Materialize(a)

	if a.Dtype() != DTypeFloat64 {
		t.Errorf("dtype = %v, want float64", a.Dtype())
	}
	if math.Abs(a.Float()-2.718281828) > 1e-8 {
		t.Errorf("value = %f, want 2.718281828", a.Float())
	}
}

func TestFromValue_Int(t *testing.T) {
	a := FromValue(42)
	Materialize(a)

	if a.Dtype() != DTypeInt32 {
		t.Errorf("dtype = %v, want int32", a.Dtype())
	}
	if a.Int() != 42 {
		t.Errorf("value = %d, want 42", a.Int())
	}
}

func TestFromValue_Bool(t *testing.T) {
	a := FromValue(true)
	Materialize(a)

	if a.Dtype() != DTypeBool {
		t.Errorf("dtype = %v, want bool", a.Dtype())
	}
	if a.Int() != 1 {
		t.Errorf("value = %d, want 1 (true)", a.Int())
	}
}

func TestFromValue_Complex64(t *testing.T) {
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

func TestFromValues_Float32_1D(t *testing.T) {
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

func TestFromValues_Float32_2D(t *testing.T) {
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

func TestFromValues_Int32(t *testing.T) {
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

func TestFromValues_Int64(t *testing.T) {
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

func TestFromValues_Bool(t *testing.T) {
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

func TestFromValues_Uint8(t *testing.T) {
	data := []uint8{0, 127, 255}
	a := FromValues(data, 3)
	Materialize(a)

	if a.Dtype() != DTypeUint8 {
		t.Errorf("dtype = %v, want uint8", a.Dtype())
	}
}

func TestFromValues_PanicsWithoutShape(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic when shape is missing")
		}
	}()
	FromValues([]float32{1, 2, 3})
}

// --- Zeros ---

func TestZeros(t *testing.T) {
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

func TestZeros_Int32(t *testing.T) {
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

func TestArray_Shape3D(t *testing.T) {
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

func TestArray_String(t *testing.T) {
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

func TestArray_Clone(t *testing.T) {
	a := FromValue(float32(7.0))
	b := a.Clone()
	Materialize(a, b)

	if math.Abs(b.Float()-7.0) > 1e-6 {
		t.Errorf("clone value = %f, want 7.0", b.Float())
	}
}

func TestArray_Set(t *testing.T) {
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

func TestArray_Valid(t *testing.T) {
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

func TestFree_ReturnsBytes(t *testing.T) {
	a := FromValues([]float32{1, 2, 3, 4}, 4)
	Materialize(a)

	n := Free(a)
	if n != 16 { // 4 * float32(4 bytes)
		t.Errorf("Free returned %d bytes, want 16", n)
	}
}

func TestFree_NilSafe(t *testing.T) {
	// Should not panic on nil
	n := Free(nil)
	if n != 0 {
		t.Errorf("Free(nil) returned %d, want 0", n)
	}
}

// --- Data extraction edge cases ---

func TestArray_Ints(t *testing.T) {
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

func TestArray_Float_DTypeFloat32(t *testing.T) {
	a := FromValue(float32(1.5))
	Materialize(a)

	got := a.Float()
	if math.Abs(got-1.5) > 1e-6 {
		t.Errorf("Float() = %f, want 1.5", got)
	}
}

func TestArray_Float_DTypeFloat64(t *testing.T) {
	a := FromValue(float64(1.5))
	Materialize(a)

	got := a.Float()
	if math.Abs(got-1.5) > 1e-12 {
		t.Errorf("Float() = %f, want 1.5", got)
	}
}

// --- Collect ---

func TestCollect_FiltersNilAndInvalid(t *testing.T) {
	a := FromValue(float32(1.0))
	b := FromValue(float32(2.0))
	Materialize(a, b)

	result := Collect(a, nil, b)
	if len(result) != 2 {
		t.Errorf("Collect returned %d arrays, want 2", len(result))
	}
}

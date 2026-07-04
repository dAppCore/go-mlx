// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import "testing"

// array_dtype_cover_test.go drives the FromValues reflect dispatch, the Int /
// Float scalar readers, and rawArrayDataPointer across the dtype arms the core
// array suite leaves dark (uint16/32/64, int8/16, float64, complex64), plus the
// Valid nil-receiver guard. The mlx_array_data_* selector in RawBytes only
// touches its arm when an array of that dtype actually flows through it.

// FromValues + RawBytes round-trip across the wider integer/float dtypes — each
// exercises a distinct reflect arm in FromValues and a distinct
// mlx_array_data_* arm in rawArrayDataPointer.
func TestArray_FromValuesRawBytes_WiderDtypes_Good(t *testing.T) {
	requireMetalRuntime(t)

	t.Run("uint16", func(t *testing.T) {
		a := FromValues([]uint16{1, 2, 3}, 3)
		defer Free(a)
		if got := a.RawBytes(); len(got) != 6 { // 3 * 2 bytes
			t.Fatalf("uint16 RawBytes len = %d, want 6", len(got))
		}
	})
	t.Run("uint32", func(t *testing.T) {
		a := FromValues([]uint32{1, 2, 3}, 3)
		defer Free(a)
		if got := a.RawBytes(); len(got) != 12 {
			t.Fatalf("uint32 RawBytes len = %d, want 12", len(got))
		}
	})
	t.Run("uint64", func(t *testing.T) {
		a := FromValues([]uint64{1, 2}, 2)
		defer Free(a)
		if got := a.RawBytes(); len(got) != 16 {
			t.Fatalf("uint64 RawBytes len = %d, want 16", len(got))
		}
	})
	t.Run("int8", func(t *testing.T) {
		a := FromValues([]int8{1, -2, 3}, 3)
		defer Free(a)
		if got := a.RawBytes(); len(got) != 3 {
			t.Fatalf("int8 RawBytes len = %d, want 3", len(got))
		}
	})
	t.Run("int16", func(t *testing.T) {
		a := FromValues([]int16{1, -2, 3}, 3)
		defer Free(a)
		if got := a.RawBytes(); len(got) != 6 {
			t.Fatalf("int16 RawBytes len = %d, want 6", len(got))
		}
	})
	t.Run("float64", func(t *testing.T) {
		a := FromValues([]float64{1.5, 2.5}, 2)
		defer Free(a)
		if got := a.RawBytes(); len(got) != 16 {
			t.Fatalf("float64 RawBytes len = %d, want 16", len(got))
		}
	})
	t.Run("complex64", func(t *testing.T) {
		a := FromValues([]complex64{complex(1, 2), complex(3, 4)}, 2)
		defer Free(a)
		if got := a.RawBytes(); len(got) != 16 { // 2 * 8 bytes
			t.Fatalf("complex64 RawBytes len = %d, want 16", len(got))
		}
	})
}

// Int reads a scalar across each integer dtype arm of its type switch.
func TestArray_Int_AcrossDtypes_Good(t *testing.T) {
	requireMetalRuntime(t)

	base := FromValues([]float32{42}, 1)
	defer Free(base)

	cases := []struct {
		name  string
		dtype DType
	}{
		{"uint8", DTypeUint8},
		{"uint16", DTypeUint16},
		{"uint32", DTypeUint32},
		{"uint64", DTypeUint64},
		{"int8", DTypeInt8},
		{"int16", DTypeInt16},
		{"int32", DTypeInt32},
		{"int64", DTypeInt64}, // the default arm
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			a := AsType(base, tc.dtype)
			defer Free(a)
			if got := a.Int(); got != 42 {
				t.Fatalf("%s scalar Int = %d, want 42", tc.name, got)
			}
		})
	}
}

// Float reads a scalar from the float32 arm. (The float64 default arm is a
// host-CPU array covered separately — Metal GPU does not support float64, so an
// AsType cast to it silently produces zeros.)
func TestArray_Float_Float32Arm_Good(t *testing.T) {
	requireMetalRuntime(t)

	f32 := FromValues([]float32{3.5}, 1)
	defer Free(f32)
	if got := f32.Float(); got != 3.5 {
		t.Fatalf("float32 Float = %v, want 3.5", got)
	}
}

// RawBytes returns nil for a zero-byte array (the n<=0 early-out).
func TestArray_RawBytes_EmptyArray_Good(t *testing.T) {
	requireMetalRuntime(t)
	empty := Zeros([]int32{0}, DTypeFloat32)
	defer Free(empty)
	if got := empty.RawBytes(); got != nil {
		t.Fatalf("empty-array RawBytes = %v, want nil", got)
	}
}

// Valid is false for a nil receiver and for a freed array; true for a live one.
func TestArray_Valid_NilAndFreed_Good(t *testing.T) {
	requireMetalRuntime(t)
	if (*Array)(nil).Valid() {
		t.Error("nil array reported Valid")
	}
	a := FromValues([]float32{1}, 1)
	if !a.Valid() {
		t.Error("live array reported not Valid")
	}
	Free(a)
}

// FromValues panics on a missing shape and on a rank above the tensor cap.
func TestArray_FromValues_PanicGuards_Ugly(t *testing.T) {
	requireMetalRuntime(t)
	mustPanic := func(name string, fn func()) {
		t.Helper()
		defer func() {
			if recover() == nil {
				t.Errorf("%s: expected panic", name)
			}
		}()
		fn()
	}
	mustPanic("no shape", func() { FromValues([]float32{1, 2, 3}) })

	overRank := make([]int, MaxTensorRank+1)
	for i := range overRank {
		overRank[i] = 1
	}
	mustPanic("rank exceeds max", func() { FromValues([]float32{1}, overRank...) })
}

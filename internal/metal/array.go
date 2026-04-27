// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

/*
#include <stdlib.h>
#include "mlx/c/mlx.h"
*/
import "C"

import (
	"encoding/binary"
	"iter"
	"reflect"
	"runtime"
	"unsafe"

	"dappco.re/go/core"
)

// Array wraps an mlx_array handle.
// Memory management relies on Go GC finalizers to call mlx_array_free,
// which decrements MLX-C's internal reference count. MLX-C handles all
// cross-array references internally — the Go wrapper does not track them.
type Array struct {
	ctx  C.mlx_array
	name string // debug label
}

// newArray creates a named Array and registers a GC finalizer.
// The inputs parameter is accepted for API compatibility but not stored —
// MLX-C tracks inter-array references via its own refcounting.
func newArray(name string, inputs ...*Array) *Array {
	t := &Array{name: name}
	runtime.SetFinalizer(t, finalizeArray)
	return t
}

// finalizeArray is called by Go GC to release the underlying C array handle.
func finalizeArray(t *Array) {
	if t != nil && t.ctx.ctx != nil {
		C.mlx_array_free(t.ctx)
		t.ctx.ctx = nil
	}
}

type scalarTypes interface {
	~bool | ~int | ~float32 | ~float64 | ~complex64
}

// FromValue creates a scalar Array from a Go value.
func FromValue[T scalarTypes](t T) *Array {
	Init()
	tt := newArray("")
	switch v := any(t).(type) {
	case bool:
		tt.ctx = C.mlx_array_new_bool(C.bool(v))
	case int:
		tt.ctx = C.mlx_array_new_int(C.int(v))
	case float32:
		tt.ctx = C.mlx_array_new_float32(C.float(v))
	case float64:
		tt.ctx = C.mlx_array_new_float64(C.double(v))
	case complex64:
		tt.ctx = C.mlx_array_new_complex(C.float(real(v)), C.float(imag(v)))
	default:
		panic("mlx: unsupported scalar type")
	}
	return tt
}

type arrayTypes interface {
	~bool | ~uint8 | ~uint16 | ~uint32 | ~uint64 |
		~int8 | ~int16 | ~int32 | ~int64 |
		~float32 | ~float64 |
		~complex64
}

// FromValues creates an Array from a Go slice with the given shape.
func FromValues[S ~[]E, E arrayTypes](s S, shape ...int) *Array {
	Init()
	if len(shape) == 0 {
		panic("mlx: shape required for non-scalar tensors")
	}

	cShape := make([]C.int, len(shape))
	for i := range shape {
		cShape[i] = C.int(shape[i])
	}

	// reflect.TypeOf is required here to map Go generic type parameters to MLX-C
	// dtype constants. Type assertions cannot recover the element type from a
	// generic ~[]E constraint at runtime. CGo tensor boundary — not business logic.
	var dtype DType
	switch reflect.TypeOf(s).Elem().Kind() {
	case reflect.Bool:
		dtype = DTypeBool
	case reflect.Uint8:
		dtype = DTypeUint8
	case reflect.Uint16:
		dtype = DTypeUint16
	case reflect.Uint32:
		dtype = DTypeUint32
	case reflect.Uint64:
		dtype = DTypeUint64
	case reflect.Int8:
		dtype = DTypeInt8
	case reflect.Int16:
		dtype = DTypeInt16
	case reflect.Int32:
		dtype = DTypeInt32
	case reflect.Int64:
		dtype = DTypeInt64
	case reflect.Float32:
		dtype = DTypeFloat32
	case reflect.Float64:
		dtype = DTypeFloat64
	case reflect.Complex64:
		dtype = DTypeComplex64
	default:
		panic("mlx: unsupported element type")
	}

	bts := make([]byte, binary.Size(s))
	if _, err := binary.Encode(bts, binary.LittleEndian, s); err != nil {
		panic(err)
	}

	tt := newArray("")
	tt.ctx = C.mlx_array_new_data(unsafe.Pointer(&bts[0]), unsafe.SliceData(cShape), C.int(len(cShape)), C.mlx_dtype(dtype))
	runtime.KeepAlive(bts)
	runtime.KeepAlive(cShape)
	return tt
}

// Zeros creates a zero-filled Array with the given shape and dtype.
func Zeros(shape []int32, dtype DType) *Array {
	Init()
	cShape := make([]C.int, len(shape))
	for i, s := range shape {
		cShape[i] = C.int(s)
	}
	tt := newArray("ZEROS")
	C.mlx_zeros(&tt.ctx, unsafe.SliceData(cShape), C.size_t(len(cShape)), C.mlx_dtype(dtype), DefaultStream().ctx)
	return tt
}

// Set replaces this array's C handle with another's.
//
//	a.Set(b) // a now wraps the same C array as b
func (t *Array) Set(other *Array) {
	C.mlx_array_set(&t.ctx, other.ctx)
}

// Clone creates a new Go wrapper sharing the same C handle (increments C refcount).
//
//	saved := a.Clone() // independent Go handle, same Metal buffer
func (t *Array) Clone() *Array {
	tt := newArray(t.name)
	C.mlx_array_set(&tt.ctx, t.ctx)
	return tt
}

// Valid reports whether this Array has a non-nil mlx handle.
//
//	if !a.Valid() { return } // guard before any ops on uninitialised arrays
func (t *Array) Valid() bool {
	return t.ctx.ctx != nil
}

// String returns a human-readable representation of the array.
//
//	fmt.Println(a.String()) // "array([1.0, 2.0, 3.0], dtype=float32)"
func (t *Array) String() string {
	str := C.mlx_string_new()
	defer C.mlx_string_free(str)
	C.mlx_array_tostring(&str, t.ctx)
	return core.Trim(C.GoString(C.mlx_string_data(str)))
}

// Shape returns the dimensions as int32 slice.
//
//	shape := logits.Shape() // e.g. []int32{1, 512, 32000} for [batch, seq, vocab]
func (t *Array) Shape() []int32 {
	dims := make([]int32, t.NumDims())
	for i := range dims {
		dims[i] = int32(t.Dim(i))
	}
	return dims
}

// Size returns the total number of elements.
//
//	n := weights.Size() // e.g. 4096*4096 = 16777216
func (t Array) Size() int { return int(C.mlx_array_size(t.ctx)) }

// NumBytes returns the total byte size.
//
//	mb := float64(a.NumBytes()) / 1e6 // memory footprint in MB
func (t Array) NumBytes() int { return int(C.mlx_array_nbytes(t.ctx)) }

// NumDims returns the number of dimensions.
//
//	if a.NumDims() == 4 { /* BHLД layout */ }
func (t Array) NumDims() int { return int(C.mlx_array_ndim(t.ctx)) }

// Dim returns the size of dimension i.
//
//	seqLen := logits.Dim(1) // middle dimension of [batch, seq, vocab]
func (t Array) Dim(i int) int { return int(C.mlx_array_dim(t.ctx, C.int(i))) }

// Dims returns all dimensions as int slice.
//
//	B, L, V := dims[0], dims[1], dims[2] // unpack [batch, seq, vocab]
func (t Array) Dims() []int {
	dims := make([]int, t.NumDims())
	for i := range dims {
		dims[i] = t.Dim(i)
	}
	return dims
}

// Dtype returns the array's data type.
//
//	if a.Dtype() == DTypeBFloat16 { /* mixed precision path */ }
func (t Array) Dtype() DType { return DType(C.mlx_array_dtype(t.ctx)) }

// Int extracts a scalar int64 value.
//
//	id := int32(next.Int()) // read sampled token ID from argmax output
func (t Array) Int() int {
	var item C.int64_t
	C.mlx_array_item_int64(&item, t.ctx)
	return int(item)
}

// Float extracts a scalar float64 value.
// Handles both float32 and float64 array dtypes.
//
//	loss := lossArr.Float() // read scalar loss value after Eval
func (t Array) Float() float64 {
	switch t.Dtype() {
	case DTypeFloat32:
		var item C.float
		C.mlx_array_item_float32(&item, t.ctx)
		return float64(item)
	default:
		var item C.double
		C.mlx_array_item_float64(&item, t.ctx)
		return float64(item)
	}
}

// Bool extracts a scalar boolean value from a bool-dtype array.
//
//	if metal.Any(mask, false); result.Bool() { /* at least one true */ }
func (t Array) Bool() bool {
	var item C.bool
	C.mlx_array_item_bool(&item, t.ctx)
	return bool(item)
}

// SetFloat64 replaces this array with a float64 scalar value.
//
//	a.SetFloat64(3.14159) // overwrite array with a new scalar
func (t *Array) SetFloat64(v float64) {
	C.mlx_array_set_float64(&t.ctx, C.double(v))
}

// ShapeRaw returns a pointer to the C shape array and the number of dimensions.
// This avoids allocation when only direct dimension access is needed.
// The returned pointer is valid only while the array is alive.
//
//	ndim := a.NumDims()
//	ptr := a.ShapeRaw() // *C.int, read ptr[0..ndim-1]
func (t Array) ShapeRaw() unsafe.Pointer {
	return unsafe.Pointer(C.mlx_array_shape(t.ctx))
}

// IsRowContiguous reports whether the array's physical memory layout is
// row-major contiguous. Non-contiguous arrays (from Transpose, BroadcastTo,
// SliceAxis, etc.) must be made contiguous before reading raw data.
func (t Array) IsRowContiguous() bool {
	var res C.bool
	C._mlx_array_is_row_contiguous(&res, t.ctx)
	return bool(res)
}

// Contiguous returns a row-major contiguous copy of the array.
// If the array is already row-contiguous, this is a no-op.
//
//	c := metal.Contiguous(transposed) // required before reading raw float data
func Contiguous(a *Array) *Array {
	out := newArray("CONTIGUOUS", a)
	C.mlx_contiguous(&out.ctx, a.ctx, C._Bool(false), DefaultStream().ctx)
	return out
}

// ensureContiguous returns a row-contiguous array, making a copy if needed.
// This must be called before any mlx_array_data_* access.
func ensureContiguous(a *Array) *Array {
	if a.IsRowContiguous() {
		return a
	}
	c := Contiguous(a)
	Materialize(c)
	return c
}

// Bytes extracts all elements as a byte slice from a uint8 array.
// Automatically handles non-contiguous arrays (transpose, broadcast, slice views).
//
//	raw := frame.Bytes() // read a packed byte buffer back to Go memory
func (t *Array) Bytes() []byte {
	src := ensureContiguous(t)
	n := src.Size()
	ptr := C.mlx_array_data_uint8(src.ctx)
	data := make([]byte, n)
	for i, b := range unsafe.Slice(ptr, n) {
		data[i] = byte(b)
	}
	runtime.KeepAlive(src)
	return data
}

// Ints extracts all elements as int slice (from int32 data).
// Automatically handles non-contiguous arrays (transpose, broadcast, slice views).
//
//	ids := tokenIDs.Ints() // read token ID list from a 1-D int32 array
func (t *Array) Ints() []int {
	src := ensureContiguous(t)
	n := src.Size()
	ptr := C.mlx_array_data_int32(src.ctx)
	ints := make([]int, n)
	for i, f := range unsafe.Slice(ptr, n) {
		ints[i] = int(f)
	}
	runtime.KeepAlive(src)
	return ints
}

// DataInt32 extracts all elements as int32 slice.
// Automatically handles non-contiguous arrays (transpose, broadcast, slice views).
//
//	ids := cacheKeys.DataInt32() // read int32 indices from an attention index array
func (t *Array) DataInt32() []int32 {
	src := ensureContiguous(t)
	n := src.Size()
	ptr := C.mlx_array_data_int32(src.ctx)
	data := make([]int32, n)
	for i, f := range unsafe.Slice(ptr, n) {
		data[i] = int32(f)
	}
	runtime.KeepAlive(src)
	return data
}

// Floats extracts all elements as float32 slice.
// Automatically handles non-contiguous arrays (transpose, broadcast, slice views).
//
//	flat := kSliced.Floats() // read KV cache values for attention inspection
func (t *Array) Floats() []float32 {
	src := ensureContiguous(t)
	n := src.Size()
	ptr := C.mlx_array_data_float32(src.ctx)
	floats := make([]float32, n)
	for i, f := range unsafe.Slice(ptr, n) {
		floats[i] = float32(f)
	}
	runtime.KeepAlive(src)
	return floats
}

// Free explicitly releases C array handles. Does not cascade — MLX-C's
// internal refcounting handles dependent arrays automatically.
func Free(s ...*Array) int {
	var n int
	for _, t := range s {
		if t != nil && t.Valid() {
			n += t.NumBytes()
			C.mlx_array_free(t.ctx)
			t.ctx.ctx = nil
			runtime.SetFinalizer(t, nil) // cancel finalizer
		}
	}
	return n
}

// Iter returns an iterator over the array's float32 elements.
// The array must be materialised and contain float32 data.
// Automatically handles non-contiguous arrays (transpose, broadcast, slice views).
func (t *Array) Iter() iter.Seq[float32] {
	src := ensureContiguous(t)
	n := src.Size()
	ptr := C.mlx_array_data_float32(src.ctx)
	return func(yield func(float32) bool) {
		defer runtime.KeepAlive(src)
		for i := range n {
			if !yield(float32(unsafe.Slice(ptr, n)[i])) {
				return
			}
		}
	}
}

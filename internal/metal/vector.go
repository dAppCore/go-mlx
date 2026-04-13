//go:build darwin && arm64

package metal

/*
#include <stdlib.h>
#include "mlx/c/mlx.h"
*/
import "C"

import "unsafe"

// VectorArray wraps an mlx_vector_array handle.
// Used for passing collections of arrays to batch operations.
type VectorArray struct {
	ctx C.mlx_vector_array
}

// NewVectorArray creates a new empty vector of arrays.
//
//	vec := metal.NewVectorArray()
//	defer vec.Free()
func NewVectorArray() *VectorArray {
	return &VectorArray{ctx: C.mlx_vector_array_new()}
}

// NewVectorArrayFromValue creates a vector containing a single array.
//
//	vec := metal.NewVectorArrayFromValue(logits) // single-element vector for Eval
func NewVectorArrayFromValue(a *Array) *VectorArray {
	return &VectorArray{ctx: C.mlx_vector_array_new_value(a.ctx)}
}

// SetValue replaces the vector contents with a single array.
//
//	vec.SetValue(newLogits) // reset vector to contain only the new array
func (v *VectorArray) SetValue(a *Array) {
	C.mlx_vector_array_set_value(&v.ctx, a.ctx)
}

// Append adds an array to the end of the vector.
//
//	vec.Append(layerOutput) // accumulate layer outputs for concatenation
func (v *VectorArray) Append(a *Array) {
	C.mlx_vector_array_append_value(v.ctx, a.ctx)
}

// Size returns the number of arrays in the vector.
//
//	n := vec.Size() // check how many arrays are collected
func (v *VectorArray) Size() int {
	return int(C.mlx_vector_array_size(v.ctx))
}

// Get retrieves the array at the given index.
// The caller receives a new Go handle (with finaliser) that shares the C refcount.
//
//	arr := vec.Get(0) // extract first array from the vector
func (v *VectorArray) Get(idx int) *Array {
	arr := newArray("VECTOR_GET")
	C.mlx_vector_array_get(&arr.ctx, v.ctx, C.size_t(idx))
	return arr
}

// Free releases the underlying C vector handle.
//
//	vec.Free() // release when done collecting arrays
func (v *VectorArray) Free() {
	if v.ctx.ctx != nil {
		C.mlx_vector_array_free(v.ctx)
		v.ctx.ctx = nil
	}
}

// VectorString wraps an mlx_vector_string handle.
// Used for passing collections of strings to MLX operations.
type VectorString struct {
	ctx C.mlx_vector_string
}

// NewVectorString creates a new empty vector of strings.
//
//	vec := metal.NewVectorString()
//	defer vec.Free()
func NewVectorString() *VectorString {
	return &VectorString{ctx: C.mlx_vector_string_new()}
}

// NewVectorStringFromValue creates a vector containing a single string.
//
//	vec := metal.NewVectorStringFromValue("weight") // single-key vector
func NewVectorStringFromValue(s string) *VectorString {
	cs := C.CString(s)
	defer C.free(unsafe.Pointer(cs))
	return &VectorString{ctx: C.mlx_vector_string_new_value(cs)}
}

// NewVectorStringFromSlice creates a vector from a Go string slice.
//
//	keys := metal.NewVectorStringFromSlice([]string{"weight", "bias"})
//	defer keys.Free()
func NewVectorStringFromSlice(ss []string) *VectorString {
	v := NewVectorString()
	for _, s := range ss {
		v.Append(s)
	}
	return v
}

// Append adds a string to the end of the vector.
//
//	vec.Append("model.layers.0.weight") // accumulate tensor names
func (v *VectorString) Append(s string) {
	cs := C.CString(s)
	defer C.free(unsafe.Pointer(cs))
	C.mlx_vector_string_append_value(v.ctx, cs)
}

// Size returns the number of strings in the vector.
//
//	n := vec.Size() // check how many strings are collected
func (v *VectorString) Size() int {
	return int(C.mlx_vector_string_size(v.ctx))
}

// Get retrieves the string at the given index.
//
//	name := vec.Get(0) // get first string from the vector
func (v *VectorString) Get(idx int) string {
	var cs *C.char
	C.mlx_vector_string_get(&cs, v.ctx, C.size_t(idx))
	return C.GoString(cs)
}

// Free releases the underlying C vector handle.
//
//	vec.Free() // release when done
func (v *VectorString) Free() {
	if v.ctx.ctx != nil {
		C.mlx_vector_string_free(v.ctx)
		v.ctx.ctx = nil
	}
}


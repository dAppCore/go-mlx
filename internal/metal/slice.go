// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

/*
#include "mlx/c/mlx.h"
*/
import "C"

// Slice extracts a sub-array using start and end indices for each dimension.
// starts and ends must have the same length as the array's dimensions.
//
//	kValid := metal.Slice(kCache, []int32{0,0,0,0}, []int32{B,H,int32(offset),D})
func Slice(a *Array, starts, ends []int32) *Array {
	if len(starts) == 0 || len(starts) != len(ends) {
		panic("Slice: starts and ends must be non-empty and equal length")
	}
	out := newArray("SLICE", a)
	cStarts := make([]C.int, len(starts))
	cEnds := make([]C.int, len(ends))
	for i := range starts {
		cStarts[i] = C.int(starts[i])
		cEnds[i] = C.int(ends[i])
	}
	strides := make([]C.int, len(starts))
	for i := range strides {
		strides[i] = 1
	}
	C.mlx_slice(&out.ctx, a.ctx, &cStarts[0], C.size_t(len(cStarts)), &cEnds[0], C.size_t(len(cEnds)), &strides[0], C.size_t(len(strides)), DefaultStream().ctx)
	return out
}

// SliceAxis extracts a sub-array along a single axis.
//
//	lastPos := metal.SliceAxis(logits, 1, seqLen-1, seqLen) // last token logits [1,1,V]
func SliceAxis(a *Array, axis int, start, end int32) *Array {
	// Build full slice parameters
	ndim := a.NumDims()
	starts := make([]int32, ndim)
	ends := make([]int32, ndim)
	for i := range ndim {
		starts[i] = 0
		ends[i] = int32(a.Dim(i))
	}
	ax := axis
	if ax < 0 {
		ax = ndim + ax
	}
	if ax < 0 || ax >= ndim {
		panic("SliceAxis: axis out of range")
	}
	starts[ax] = start
	ends[ax] = end
	return Slice(a, starts, ends)
}

// SliceUpdateInplace updates a slice of the array in-place.
// This is critical for KV cache updates.
//
//	newK := metal.SliceUpdateInplace(kBuf, k, []int32{0,0,int32(prev),0}, []int32{B,H,int32(offset),D})
func SliceUpdateInplace(a, update *Array, starts, ends []int32) *Array {
	if len(starts) == 0 || len(starts) != len(ends) {
		panic("SliceUpdateInplace: starts and ends must be non-empty and equal length")
	}
	out := newArray("SLICE_UPDATE", a, update)
	cStarts := make([]C.int, len(starts))
	cEnds := make([]C.int, len(ends))
	for i := range starts {
		cStarts[i] = C.int(starts[i])
		cEnds[i] = C.int(ends[i])
	}
	strides := make([]C.int, len(starts))
	for i := range strides {
		strides[i] = 1
	}
	C.mlx_slice_update(&out.ctx, a.ctx, update.ctx, &cStarts[0], C.size_t(len(cStarts)), &cEnds[0], C.size_t(len(cEnds)), &strides[0], C.size_t(len(strides)), DefaultStream().ctx)
	return out
}

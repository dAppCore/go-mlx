// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

/*
#include "mlx/c/mlx.h"

// mlx_slice_inline / mlx_slice_update_inline materialise the 3-array
// starts / ends / strides triple on the C stack so the per-call Slice and
// SliceUpdateInplace paths skip the three Go-side []C.int heap allocs.
// strides are implicitly 1 (the only mode the wrappers currently use —
// stride-aware slicing isn't exposed by the Go API).  Rank is bounded by
// the package-wide maxTensorRank = 8 declared in ops.go.
static inline int mlx_slice_inline(
    mlx_array* res, mlx_array a,
    const int32_t* starts_in, const int32_t* ends_in, size_t n,
    mlx_stream s) {
    int starts_buf[8];
    int ends_buf[8];
    int strides_buf[8];
    for (size_t i = 0; i < n; ++i) {
        starts_buf[i] = (int)starts_in[i];
        ends_buf[i] = (int)ends_in[i];
        strides_buf[i] = 1;
    }
    return mlx_slice(res, a, starts_buf, n, ends_buf, n, strides_buf, n, s);
}

static inline int mlx_slice_update_inline(
    mlx_array* res, mlx_array a, mlx_array upd,
    const int32_t* starts_in, const int32_t* ends_in, size_t n,
    mlx_stream s) {
    int starts_buf[8];
    int ends_buf[8];
    int strides_buf[8];
    for (size_t i = 0; i < n; ++i) {
        starts_buf[i] = (int)starts_in[i];
        ends_buf[i] = (int)ends_in[i];
        strides_buf[i] = 1;
    }
    return mlx_slice_update(res, a, upd, starts_buf, n, ends_buf, n, strides_buf, n, s);
}
*/
import "C"

import "unsafe"

// Slice extracts a sub-array using start and end indices for each dimension.
// starts and ends must have the same length as the array's dimensions.
// Routes through mlx_slice_inline so the cgo starts / ends / strides arrays
// are stack-allocated on the C side, removing three Go heap allocs per call
// on the per-token KV-cache slice path.
//
//	kValid := metal.Slice(kCache, []int32{0,0,0,0}, []int32{B,H,int32(offset),D})
func Slice(a *Array, starts, ends []int32) *Array {
	if len(starts) == 0 || len(starts) != len(ends) {
		panic("Slice: starts and ends must be non-empty and equal length")
	}
	if len(starts) > maxTensorRank {
		panic("Slice: rank exceeds maxTensorRank")
	}
	out := newArray("SLICE", a)
	startsPtr := (*C.int32_t)(unsafe.Pointer(&starts[0]))
	endsPtr := (*C.int32_t)(unsafe.Pointer(&ends[0]))
	C.mlx_slice_inline(&out.ctx, a.ctx, startsPtr, endsPtr, C.size_t(len(starts)), DefaultStream().ctx)
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
// This is critical for KV cache updates.  Routes through
// mlx_slice_update_inline so the cgo starts / ends / strides arrays are
// stack-allocated on the C side, removing three Go heap allocs per call.
//
//	newK := metal.SliceUpdateInplace(kBuf, k, []int32{0,0,int32(prev),0}, []int32{B,H,int32(offset),D})
func SliceUpdateInplace(a, update *Array, starts, ends []int32) *Array {
	if len(starts) == 0 || len(starts) != len(ends) {
		panic("SliceUpdateInplace: starts and ends must be non-empty and equal length")
	}
	if len(starts) > maxTensorRank {
		panic("SliceUpdateInplace: rank exceeds maxTensorRank")
	}
	out := newArray("SLICE_UPDATE", a, update)
	startsPtr := (*C.int32_t)(unsafe.Pointer(&starts[0]))
	endsPtr := (*C.int32_t)(unsafe.Pointer(&ends[0]))
	C.mlx_slice_update_inline(&out.ctx, a.ctx, update.ctx, startsPtr, endsPtr, C.size_t(len(starts)), DefaultStream().ctx)
	return out
}

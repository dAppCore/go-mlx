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
// the package-wide MaxTensorRank = 8 declared in ops.go.
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

// mlx_slice_inline_4 / mlx_slice_update_inline_4 are the rank-4 scalar-pass
// form — KV cache hot paths construct []int32{0,0,prev,0} per call which
// escape to heap (4 sites in KVCache.Update alone, 22 sites in cache.go).
// Passing the eight register-passed scalars eliminates the slice literal
// entirely. W10-J pattern applied to slice rank-4 (the KV cache canonical
// rank). strides are implicitly 1.
static inline int mlx_slice_inline_4(
    mlx_array* res, mlx_array a,
    int32_t s0, int32_t s1, int32_t s2, int32_t s3,
    int32_t e0, int32_t e1, int32_t e2, int32_t e3,
    mlx_stream s) {
    int starts_buf[4] = {(int)s0, (int)s1, (int)s2, (int)s3};
    int ends_buf[4]   = {(int)e0, (int)e1, (int)e2, (int)e3};
    int strides_buf[4] = {1, 1, 1, 1};
    return mlx_slice(res, a, starts_buf, 4, ends_buf, 4, strides_buf, 4, s);
}

static inline int mlx_slice_update_inline_4(
    mlx_array* res, mlx_array a, mlx_array upd,
    int32_t s0, int32_t s1, int32_t s2, int32_t s3,
    int32_t e0, int32_t e1, int32_t e2, int32_t e3,
    mlx_stream s) {
    int starts_buf[4] = {(int)s0, (int)s1, (int)s2, (int)s3};
    int ends_buf[4]   = {(int)e0, (int)e1, (int)e2, (int)e3};
    int strides_buf[4] = {1, 1, 1, 1};
    return mlx_slice_update(res, a, upd, starts_buf, 4, ends_buf, 4, strides_buf, 4, s);
}

// mlx_slice_inline_2 / mlx_slice_update_inline_2 are the rank-2 scalar-pass
// form — completes the W11-AC Reshape/Slice rank-2 family alongside Slice4.
// packQ4Cached's `SliceAxis(paired, 1, 0, 1)` + `SliceAxis(paired, 1, 1, 2)`
// (two calls per Q4 K/V Update) currently routes via SliceAxis which
// allocates `make([]int32, ndim)` twice per call — ~4 slice heap allocs per
// Q4 store. Passing the 4 register-passed scalars eliminates both the
// SliceAxis materialisation and the inline-slice-literal escape entirely.
// strides are implicitly 1 (matches the broader Slice* wrapper convention).
static inline int mlx_slice_inline_2(
    mlx_array* res, mlx_array a,
    int32_t s0, int32_t s1,
    int32_t e0, int32_t e1,
    mlx_stream s) {
    int starts_buf[2] = {(int)s0, (int)s1};
    int ends_buf[2]   = {(int)e0, (int)e1};
    int strides_buf[2] = {1, 1};
    return mlx_slice(res, a, starts_buf, 2, ends_buf, 2, strides_buf, 2, s);
}

static inline int mlx_slice_update_inline_2(
    mlx_array* res, mlx_array a, mlx_array upd,
    int32_t s0, int32_t s1,
    int32_t e0, int32_t e1,
    mlx_stream s) {
    int starts_buf[2] = {(int)s0, (int)s1};
    int ends_buf[2]   = {(int)e0, (int)e1};
    int strides_buf[2] = {1, 1};
    return mlx_slice_update(res, a, upd, starts_buf, 2, ends_buf, 2, strides_buf, 2, s);
}

// mlx_slice_inline_1 is the rank-1 scalar-pass form — completes the
// rank-1/2/4 scalar-pass slice trio. unpackQ4's tail-trim path
// `Slice(flat, []int32{0}, []int32{int32(n)})` pays a two-slice-literal
// escape on the (rare) odd-length Q4 dequant — eliminating it via Slice1
// removes the residual the pack path's even-length norm leaves at the
// dequant boundary. strides are implicitly 1.
static inline int mlx_slice_inline_1(
    mlx_array* res, mlx_array a,
    int32_t s0, int32_t e0,
    mlx_stream s) {
    int starts_buf[1] = {(int)s0};
    int ends_buf[1]   = {(int)e0};
    int strides_buf[1] = {1};
    return mlx_slice(res, a, starts_buf, 1, ends_buf, 1, strides_buf, 1, s);
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
	if len(starts) > MaxTensorRank {
		panic("Slice: rank exceeds MaxTensorRank")
	}
	out := NewArray("SLICE", a)
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
	if len(starts) > MaxTensorRank {
		panic("SliceUpdateInplace: rank exceeds MaxTensorRank")
	}
	out := NewArray("SLICE_UPDATE", a, update)
	startsPtr := (*C.int32_t)(unsafe.Pointer(&starts[0]))
	endsPtr := (*C.int32_t)(unsafe.Pointer(&ends[0]))
	C.mlx_slice_update_inline(&out.ctx, a.ctx, update.ctx, startsPtr, endsPtr, C.size_t(len(starts)), DefaultStream().ctx)
	return out
}

// Slice4 is the rank-4 scalar-pass form of Slice — eliminates the
// []int32{...} literal allocation by passing the 8 indices as scalars.
// Routes through mlx_slice_inline_4 which materialises the C stack buffers
// directly from register-passed scalars. Used by KV cache update paths
// where `[]int32{0,0,prev,0}, []int32{B,H,offset,D}` previously paid two
// heap allocs per call site (and most cache.go sites have 2-4 such pairs).
// Resolves the default stream on every call — hot loops that issue several
// Slice4 calls back-to-back should hoist the stream out via Slice4WithStream.
//
//	kFull := metal.Slice4(kCache, 0,0,0,0, B,H,int32(offset),D)
func Slice4(a *Array, s0, s1, s2, s3, e0, e1, e2, e3 int32) *Array {
	return Slice4WithStream(a, s0, s1, s2, s3, e0, e1, e2, e3, DefaultStream())
}

// Slice4WithStream is the stream-passing sibling of Slice4 — accepts a
// pre-resolved stream so per-token loops can hoist the DefaultStream()
// lookup (RWMutex.RLock+RUnlock + cached-device atomic load) outside the
// loop. Mirrors the W10/W11 fixedKVCacheSlice4D pattern: KVCache.Update
// issues four Slice4-family calls per token; resolving the stream once
// per Update collapses those four lookups to one.
//
//	stream := metal.DefaultStream()
//	kFull := metal.Slice4WithStream(kCache, 0,0,0,0, B,H,int32(offset),D, stream)
func Slice4WithStream(a *Array, s0, s1, s2, s3, e0, e1, e2, e3 int32, stream *Stream) *Array {
	out := NewArray("SLICE", a)
	C.mlx_slice_inline_4(&out.ctx, a.ctx,
		C.int32_t(s0), C.int32_t(s1), C.int32_t(s2), C.int32_t(s3),
		C.int32_t(e0), C.int32_t(e1), C.int32_t(e2), C.int32_t(e3),
		stream.ctx)
	return out
}

// SliceUpdateInplace4 is the rank-4 scalar-pass form of SliceUpdateInplace.
// See Slice4 for the rationale — KV cache append paths construct
// []int32{0,0,prev,0}, []int32{B,H,offset,D} on every Update call.  Hot
// loops should prefer SliceUpdateInplace4WithStream to hoist the per-call
// DefaultStream() lookup.
//
//	kBuf := metal.SliceUpdateInplace4(kBuf, k, 0,0,int32(prev),0, B,H,int32(offset),D)
func SliceUpdateInplace4(a, update *Array, s0, s1, s2, s3, e0, e1, e2, e3 int32) *Array {
	return SliceUpdateInplace4WithStream(a, update, s0, s1, s2, s3, e0, e1, e2, e3, DefaultStream())
}

// SliceUpdateInplace4WithStream is the stream-passing sibling of
// SliceUpdateInplace4 — accepts a pre-resolved stream so the KVCache.Update
// hot path can resolve the default stream once per Update instead of once
// per slice-update call.  Mirrors fixedKVCacheSliceUpdate4D.
//
//	stream := metal.DefaultStream()
//	kBuf := metal.SliceUpdateInplace4WithStream(kBuf, k, 0,0,int32(prev),0, B,H,int32(offset),D, stream)
func SliceUpdateInplace4WithStream(a, update *Array, s0, s1, s2, s3, e0, e1, e2, e3 int32, stream *Stream) *Array {
	out := NewArray("SLICE_UPDATE", a, update)
	C.mlx_slice_update_inline_4(&out.ctx, a.ctx, update.ctx,
		C.int32_t(s0), C.int32_t(s1), C.int32_t(s2), C.int32_t(s3),
		C.int32_t(e0), C.int32_t(e1), C.int32_t(e2), C.int32_t(e3),
		stream.ctx)
	return out
}

// Slice2 is the rank-2 scalar-pass form of Slice — eliminates the four
// `[]int32{...}` literal allocations that SliceAxis materialises on a
// rank-2 input (`make([]int32, ndim)` twice) plus the variadic-slice
// escape of any direct Slice([]int32{...}, []int32{...}) call site.
// Used by packQ4Cached where `SliceAxis(paired, 1, 0, 1)` +
// `SliceAxis(paired, 1, 1, 2)` previously paid ~4 slice heap allocs per
// Q4 K/V store. strides are implicitly 1.
//
//	low  := metal.Slice2(paired, 0, 0, int32(pairs), 1)
//	high := metal.Slice2(paired, 0, 1, int32(pairs), 2)
func Slice2(a *Array, s0, s1, e0, e1 int32) *Array {
	out := NewArray("SLICE", a)
	C.mlx_slice_inline_2(&out.ctx, a.ctx,
		C.int32_t(s0), C.int32_t(s1),
		C.int32_t(e0), C.int32_t(e1),
		DefaultStream().ctx)
	return out
}

// SliceUpdateInplace2 is the rank-2 scalar-pass form of SliceUpdateInplace.
// See Slice2 for the rationale — pair-symmetry with Slice2 lets callers
// reading + writing the same rank-2 region use the same scalar-pass shape
// without per-call slice literals.
//
//	mat := metal.SliceUpdateInplace2(mat, patch, 0, 0, int32(h), int32(w))
func SliceUpdateInplace2(a, update *Array, s0, s1, e0, e1 int32) *Array {
	out := NewArray("SLICE_UPDATE", a, update)
	C.mlx_slice_update_inline_2(&out.ctx, a.ctx, update.ctx,
		C.int32_t(s0), C.int32_t(s1),
		C.int32_t(e0), C.int32_t(e1),
		DefaultStream().ctx)
	return out
}

// Slice1 is the rank-1 scalar-pass form of Slice — eliminates the two
// `[]int32{...}` literal allocations that any rank-1 Slice call would
// otherwise pay. Used by unpackQ4's odd-length tail-trim
// `Slice(flat, []int32{0}, []int32{int32(n)})` so the dequant boundary
// matches the pack path's scalar-pass shape. strides are implicitly 1.
//
//	trimmed := metal.Slice1(flat, 0, int32(n))
func Slice1(a *Array, s0, e0 int32) *Array {
	out := NewArray("SLICE", a)
	C.mlx_slice_inline_1(&out.ctx, a.ctx,
		C.int32_t(s0), C.int32_t(e0),
		DefaultStream().ctx)
	return out
}

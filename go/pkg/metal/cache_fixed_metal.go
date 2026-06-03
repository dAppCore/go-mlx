// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

/*
#include "mlx/c/mlx.h"

// mlx_slice_fixed4_scalar / mlx_slice_update_fixed4_scalar narrow the
// FixedKVCache rank-4 slice geometry from individual scalar arguments
// into stack-local int starts[4] / ends[4] / strides[4] buffers, then
// invoke mlx_slice / mlx_slice_update.  The fixed-rank specialisation
// (starts = {0, 0, seqStart, 0}, ends = {batch, heads, seqEnd, dim},
// strides = {1, 1, 1, 1}) is the only slice geometry FixedKVCache uses,
// so the scalar-passing form eliminates the per-call Go heap alloc for
// the cgo int buffer entirely — there is no Go-side starts / ends array
// at all, since the scalars cross the cgo boundary directly in registers.
//
// This sidesteps the W10-A finding (re-confirmed in W10-J escape analysis)
// that even Go-native [4]int32 arrays passed via unsafe.Pointer escape to
// heap when the cgo wrapper closure captures &arr[0].  The W10-F sync.Pool
// avoided escape but cost ~1024 sync.Pool Get/Put roundtrips on a 256-token
// decode; the scalar form has no buffer at all.
static inline int mlx_slice_fixed4_scalar(
    mlx_array* res, mlx_array a,
    int32_t s0, int32_t s1, int32_t s2, int32_t s3,
    int32_t e0, int32_t e1, int32_t e2, int32_t e3,
    mlx_stream s) {
    int starts_buf[4] = {(int)s0, (int)s1, (int)s2, (int)s3};
    int ends_buf[4]   = {(int)e0, (int)e1, (int)e2, (int)e3};
    int strides_buf[4] = {1, 1, 1, 1};
    return mlx_slice(res, a, starts_buf, 4, ends_buf, 4, strides_buf, 4, s);
}

static inline int mlx_slice_update_fixed4_scalar(
    mlx_array* res, mlx_array a, mlx_array upd,
    int32_t s0, int32_t s1, int32_t s2, int32_t s3,
    int32_t e0, int32_t e1, int32_t e2, int32_t e3,
    mlx_stream s) {
    int starts_buf[4] = {(int)s0, (int)s1, (int)s2, (int)s3};
    int ends_buf[4]   = {(int)e0, (int)e1, (int)e2, (int)e3};
    int strides_buf[4] = {1, 1, 1, 1};
    return mlx_slice_update(res, a, upd, starts_buf, 4, ends_buf, 4, strides_buf, 4, s);
}
*/
import "C"

// fixedKVCacheSlice4D performs a 4D Slice with starts[0,0,seqStart,0] and
// ends[batch,heads,seqEnd,dim], with all strides = 1.  It is the FixedKVCache
// equivalent of metal.Slice routed through mlx_slice_fixed4_scalar — the
// per-call cgo int buffer is materialised on the C stack from scalar
// arguments rather than a Go-side []C.int / [4]int32 buffer, removing the
// per-call Go heap alloc entirely.
//
// The stream argument lets callers pass a pre-resolved stream so the
// steady-state path can avoid the per-call DefaultStream() lookup, which
// runs currentDefaultDevice() each time and allocates a defer record for
// C.mlx_device_free.
//
//	k := fixedKVCacheSlice4D(c.keys, c.batch, c.heads, 0, int32(c.length), c.keyDim, c.stream())
func fixedKVCacheSlice4D(a *Array, batch, heads, seqStart, seqEnd, dim int32, stream *Stream) *Array {
	out := newArray("SLICE", a)
	C.mlx_slice_fixed4_scalar(
		&out.ctx,
		a.ctx,
		C.int32_t(0), C.int32_t(0), C.int32_t(seqStart), C.int32_t(0),
		C.int32_t(batch), C.int32_t(heads), C.int32_t(seqEnd), C.int32_t(dim),
		stream.ctx,
	)
	return out
}

// fixedKVCacheAsType is the FixedKVCache-local variant of metal.AsType
// that accepts a pre-resolved stream, avoiding the inner DefaultStream()
// call.  Used on the FP16 storage path when converting Float32 input k
// and v tensors to the FP16 storage dtype on every Update.
//
//	k = fixedKVCacheAsType(k, DTypeFloat16, stream)
func fixedKVCacheAsType(a *Array, dtype DType, stream *Stream) *Array {
	out := newArray("ASTYPE", a)
	C.mlx_astype(&out.ctx, a.ctx, C.mlx_dtype(dtype), stream.ctx)
	return out
}

// fixedKVCacheSliceUpdate4D performs a 4D SliceUpdateInplace with
// starts[0,0,seqStart,0] and ends[batch,heads,seqEnd,dim], strides = 1.  The
// FixedKVCache equivalent of metal.SliceUpdateInplace routed through
// mlx_slice_update_fixed4_scalar — see fixedKVCacheSlice4D for the
// scalar-passing rationale (no Go-side buffer at all).  Called twice per
// Update on the steady-state single-token path (once for keys, once for
// values).
//
//	c.keys = fixedKVCacheSliceUpdate4D(c.keys, writeK, c.batch, c.heads, int32(start), int32(start+writeLen), c.keyDim, c.stream())
func fixedKVCacheSliceUpdate4D(a, update *Array, batch, heads, seqStart, seqEnd, dim int32, stream *Stream) *Array {
	out := newArray("SLICE_UPDATE", a, update)
	C.mlx_slice_update_fixed4_scalar(
		&out.ctx,
		a.ctx, update.ctx,
		C.int32_t(0), C.int32_t(0), C.int32_t(seqStart), C.int32_t(0),
		C.int32_t(batch), C.int32_t(heads), C.int32_t(seqEnd), C.int32_t(dim),
		stream.ctx,
	)
	return out
}

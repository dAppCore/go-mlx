// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

/*
#include "mlx/c/mlx.h"
*/
import "C"

import "sync"

// fixedSliceCgoIntPool reuses the 12-slot cgo-int buffer used by
// fixedKVCacheSlice4D and fixedKVCacheSliceUpdate4D.  Each call borrows the
// buffer, fills starts / ends / strides, hands the three sub-slices to the
// mlx op, then returns the buffer to the pool — mlx_slice / mlx_slice_update
// copy the values into the op graph before returning, so the buffer is safe
// to recycle immediately.  This replaces the earlier `var [4]C.int` "stack
// arrays" which escape-analysis showed were moved to heap by the cgo pointer
// checker (3 heap allocs per helper × 2 helpers × 2 calls per Update on the
// FP16 storage path).
var fixedSliceCgoIntPool = sync.Pool{
	New: func() any {
		buf := make([]C.int, 12)
		return &buf
	},
}

// fixedKVCacheSlice4D performs a 4D Slice with starts[0,0,seqStart,0] and
// ends[batch,heads,seqEnd,dim], with all strides = 1.  It is the FixedKVCache
// equivalent of metal.Slice using a pooled cgo-int buffer, avoiding three
// [4]C.int heap allocations per call.  The single-token decode loop on
// FixedKVCache calls this twice per Update (in validState), so the saved
// allocations compound with sequence length.
//
// The stream argument lets callers pass a pre-resolved stream so the
// steady-state path can avoid the per-call DefaultStream() lookup, which
// runs currentDefaultDevice() each time and allocates a defer record for
// C.mlx_device_free.
//
//	k := fixedKVCacheSlice4D(c.keys, c.batch, c.heads, 0, int32(c.length), c.keyDim, c.stream())
func fixedKVCacheSlice4D(a *Array, batch, heads, seqStart, seqEnd, dim int32, stream *Stream) *Array {
	out := newArray("SLICE", a)
	bufPtr := fixedSliceCgoIntPool.Get().(*[]C.int)
	buf := *bufPtr
	buf[0], buf[1], buf[2], buf[3] = 0, 0, C.int(seqStart), 0
	buf[4], buf[5], buf[6], buf[7] = C.int(batch), C.int(heads), C.int(seqEnd), C.int(dim)
	buf[8], buf[9], buf[10], buf[11] = 1, 1, 1, 1
	C.mlx_slice(
		&out.ctx,
		a.ctx,
		&buf[0], 4,
		&buf[4], 4,
		&buf[8], 4,
		stream.ctx,
	)
	fixedSliceCgoIntPool.Put(bufPtr)
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
// FixedKVCache equivalent of metal.SliceUpdateInplace using a pooled cgo-int
// buffer.  Called twice per Update on the steady-state single-token path
// (once for keys, once for values).
//
//	c.keys = fixedKVCacheSliceUpdate4D(c.keys, writeK, c.batch, c.heads, int32(start), int32(start+writeLen), c.keyDim, c.stream())
func fixedKVCacheSliceUpdate4D(a, update *Array, batch, heads, seqStart, seqEnd, dim int32, stream *Stream) *Array {
	out := newArray("SLICE_UPDATE", a, update)
	bufPtr := fixedSliceCgoIntPool.Get().(*[]C.int)
	buf := *bufPtr
	buf[0], buf[1], buf[2], buf[3] = 0, 0, C.int(seqStart), 0
	buf[4], buf[5], buf[6], buf[7] = C.int(batch), C.int(heads), C.int(seqEnd), C.int(dim)
	buf[8], buf[9], buf[10], buf[11] = 1, 1, 1, 1
	C.mlx_slice_update(
		&out.ctx,
		a.ctx, update.ctx,
		&buf[0], 4,
		&buf[4], 4,
		&buf[8], 4,
		stream.ctx,
	)
	fixedSliceCgoIntPool.Put(bufPtr)
	return out
}

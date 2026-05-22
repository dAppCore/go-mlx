// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

/*
#include "mlx/c/mlx.h"
*/
import "C"

import "sync"

// pagedSliceCgoIntPool reuses the 12-slot cgo-int buffer used by
// pagedSlice4D and pagedSliceUpdate4D.  Each call returns the buffer to the
// pool after the mlx op has copied the start/end/stride values into its
// graph.  This avoids one make([]C.int, 12) per call — 4 calls per
// PagedKVCache token-Update on the prealloc path — which compounds with
// sequence length.
var pagedSliceCgoIntPool = sync.Pool{
	New: func() any {
		buf := make([]C.int, 12)
		return &buf
	},
}

// pagedSliceUpdate4D performs a 4D mlx_slice_update with starts[0,0,seqStart,0]
// and ends[batch,heads,seqEnd,dim], strides=1.  The single 12-element cgo-int
// buffer backing the three views is borrowed from a pool — mlx_slice_update
// copies the values into the op graph before returning, so the buffer is safe
// to return to the pool immediately.
//
//	pageK = pagedSliceUpdate4D(pageK, pieceK, kBatch, kHeads, writeStart, writeEnd, kDim, stream)
func pagedSliceUpdate4D(a, update *Array, batch, heads, seqStart, seqEnd, dim int32, stream *Stream) *Array {
	out := newArray("SLICE_UPDATE", a, update)
	bufPtr := pagedSliceCgoIntPool.Get().(*[]C.int)
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
	pagedSliceCgoIntPool.Put(bufPtr)
	return out
}

// pagedSlice4D performs a 4D mlx_slice with starts[0,0,0,0] and
// ends[batch,heads,length,dim], strides=1.  Uses the same pooled cgo-int
// buffer as pagedSliceUpdate4D.
//
//	view := pagedSlice4D(page, batch, heads, length, dim, stream)
func pagedSlice4D(a *Array, batch, heads, length, dim int32, stream *Stream) *Array {
	out := newArray("SLICE", a)
	bufPtr := pagedSliceCgoIntPool.Get().(*[]C.int)
	buf := *bufPtr
	buf[0], buf[1], buf[2], buf[3] = 0, 0, 0, 0
	buf[4], buf[5], buf[6], buf[7] = C.int(batch), C.int(heads), C.int(length), C.int(dim)
	buf[8], buf[9], buf[10], buf[11] = 1, 1, 1, 1
	C.mlx_slice(
		&out.ctx,
		a.ctx,
		&buf[0], 4,
		&buf[4], 4,
		&buf[8], 4,
		stream.ctx,
	)
	pagedSliceCgoIntPool.Put(bufPtr)
	return out
}

// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

/*
#include "mlx/c/mlx.h"
*/
import "C"

// pagedSliceUpdate4D performs a 4D mlx_slice_update with starts[0,0,seqStart,0]
// and ends[batch,heads,seqEnd,dim], strides=1.  Consolidates the three
// make([]C.int, 4) heap allocations of metal.SliceUpdateInplace into a single
// 12-element make — three views into one buffer feed mlx_slice_update.  Each
// PagedKVCache token-Update calls this twice (once for K, once for V) so the
// alloc saving compounds with sequence length.
//
//	pageK = pagedSliceUpdate4D(pageK, pieceK, kBatch, kHeads, writeStart, writeEnd, kDim, stream)
func pagedSliceUpdate4D(a, update *Array, batch, heads, seqStart, seqEnd, dim int32, stream *Stream) *Array {
	out := newArray("SLICE_UPDATE", a, update)
	// Single 12-element allocation backs all three cgo int views.  Layout:
	// [0..4) = starts, [4..8) = ends, [8..12) = strides.
	buf := make([]C.int, 12)
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
	return out
}

// pagedSlice4D performs a 4D mlx_slice with starts[0,0,0,0] and
// ends[batch,heads,length,dim], strides=1.  Consolidates the three
// make([]C.int, 4) heap allocations of metal.Slice into a single 12-element
// make.  Used by visiblePage to clip a preallocated page to its filled length.
//
//	view := pagedSlice4D(page, batch, heads, length, dim, stream)
func pagedSlice4D(a *Array, batch, heads, length, dim int32, stream *Stream) *Array {
	out := newArray("SLICE", a)
	buf := make([]C.int, 12)
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
	return out
}

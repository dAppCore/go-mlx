// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

/*
#include "mlx/c/mlx.h"
*/
import "C"

// fixedKVCacheSlice4D performs a 4D Slice with starts[0,0,seqStart,0] and
// ends[batch,heads,seqEnd,dim], with all strides = 1.  It is the FixedKVCache
// equivalent of metal.Slice but materialises the cgo int arrays on the stack,
// avoiding three [4]C.int heap allocations per call.  The single-token decode
// loop on FixedKVCache calls this twice per Update (in validState), so the
// saved allocations compound with sequence length.
//
// The stream argument lets callers pass a pre-resolved stream so the
// steady-state path can avoid the per-call DefaultStream() lookup, which
// runs currentDefaultDevice() each time and allocates a defer record for
// C.mlx_device_free.
//
//	k := fixedKVCacheSlice4D(c.keys, c.batch, c.heads, 0, int32(c.length), c.keyDim, c.stream())
func fixedKVCacheSlice4D(a *Array, batch, heads, seqStart, seqEnd, dim int32, stream *Stream) *Array {
	out := newArray("SLICE", a)
	var cStarts [4]C.int
	var cEnds [4]C.int
	var cStrides [4]C.int
	cStarts[0], cStarts[1], cStarts[2], cStarts[3] = 0, 0, C.int(seqStart), 0
	cEnds[0], cEnds[1], cEnds[2], cEnds[3] = C.int(batch), C.int(heads), C.int(seqEnd), C.int(dim)
	cStrides[0], cStrides[1], cStrides[2], cStrides[3] = 1, 1, 1, 1
	C.mlx_slice(
		&out.ctx,
		a.ctx,
		&cStarts[0], 4,
		&cEnds[0], 4,
		&cStrides[0], 4,
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
// FixedKVCache equivalent of metal.SliceUpdateInplace using stack-allocated
// cgo int arrays.  Called twice per Update on the steady-state single-token
// path (once for keys, once for values).
//
//	c.keys = fixedKVCacheSliceUpdate4D(c.keys, writeK, c.batch, c.heads, int32(start), int32(start+writeLen), c.keyDim, c.stream())
func fixedKVCacheSliceUpdate4D(a, update *Array, batch, heads, seqStart, seqEnd, dim int32, stream *Stream) *Array {
	out := newArray("SLICE_UPDATE", a, update)
	var cStarts [4]C.int
	var cEnds [4]C.int
	var cStrides [4]C.int
	cStarts[0], cStarts[1], cStarts[2], cStarts[3] = 0, 0, C.int(seqStart), 0
	cEnds[0], cEnds[1], cEnds[2], cEnds[3] = C.int(batch), C.int(heads), C.int(seqEnd), C.int(dim)
	cStrides[0], cStrides[1], cStrides[2], cStrides[3] = 1, 1, 1, 1
	C.mlx_slice_update(
		&out.ctx,
		a.ctx, update.ctx,
		&cStarts[0], 4,
		&cEnds[0], 4,
		&cStrides[0], 4,
		stream.ctx,
	)
	return out
}

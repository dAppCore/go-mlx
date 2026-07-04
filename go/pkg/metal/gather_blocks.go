// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

// gather_blocks.go is the sparse-attention block-selection op surface the NSA /
// MoBA mixers need to attend over only their chosen blocks instead of building a
// dense [.,.,L,nBlocks] keep-mask over every block. It is composed from the
// existing Argsort + TakeAlongAxis primitives — NOT a new hand-written Metal
// kernel: MLX already ships the gather (mlx_take_along_axis) and the sort, so a
// bespoke .metal kernel (and the metallib rebuild it would force) buys nothing
// here. The selection becomes genuinely sparse — gather n blocks, not score all
// of them — which is the production point of NSA/MoBA.

// TopNIndices returns the indices of the n largest values along the last axis,
// per row, as an int32 array shaped like scores but with the last axis trimmed
// to n. Composed from Argsort (ascending) + a tail slice: the last n sorted
// positions are the top-n, reversed so index 0 is the largest. n is clamped to
// [1, lastDim].
//
//	idx := metal.TopNIndices(blockScores, 2) // [B,H,L,2] indices of top-2 blocks
func TopNIndices(scores *Array, n int) *Array {
	lastAxis := scores.NumDims() - 1
	dim := scores.Dim(lastAxis)
	if n < 1 {
		n = 1
	}
	if n > dim {
		n = dim
	}
	order := Argsort(scores, lastAxis) // ascending indices along the last axis
	// Take the final n positions (the largest), then reverse so [0]=largest.
	tail := SliceAxis(order, lastAxis, int32(dim-n), int32(dim))
	Free(order)
	rev := reverseLastAxis(tail)
	Free(tail)
	return rev
}

// reverseLastAxis reverses an array along its last axis via a descending gather
// (TakeAlongAxis with reversed positions). Small helper kept local to the
// block-gather surface; the top-n selector wants largest-first ordering.
func reverseLastAxis(a *Array) *Array {
	lastAxis := a.NumDims() - 1
	dim := a.Dim(lastAxis)
	// Build reversed positional indices [dim-1, dim-2, …, 0] broadcast to a's shape.
	pos := Arange(float64(dim-1), -1, -1, DTypeInt32) // [dim] descending
	shape := a.Shape()
	view := reshapeForBroadcast(pos, shape)
	Free(pos)
	idx := BroadcastTo(view, shape)
	Free(view)
	out := TakeAlongAxis(a, idx, lastAxis)
	Free(idx)
	return out
}

// reshapeForBroadcast reshapes a 1-D last-axis vector to [1,…,1,dim] so it
// broadcasts across the leading axes of a target shape.
func reshapeForBroadcast(vec *Array, shape []int32) *Array {
	rank := len(shape)
	dims := make([]int32, rank)
	for i := range dims {
		dims[i] = 1
	}
	dims[rank-1] = shape[rank-1]
	return Reshape(vec, dims...)
}

// GatherSelectedBlocks gathers, per query, the blocks named by indices from a
// per-block tensor. blocks is [B,H,nBlocks,blockDim] (a compressed block summary
// or a flattened block of tokens); indices is [B,H,L,n] from TopNIndices. The
// result is [B,H,L,n,blockDim]: for each query position the n selected blocks'
// rows, ready to attend over without the dense all-blocks mask.
//
// Composed from TakeAlongAxis on a broadcast of blocks against the query axis —
// no new Metal kernel. The gather is along the block axis; blockDim rides along.
//
//	sel := metal.GatherSelectedBlocks(blockK, idx) // [B,H,L,n,blockDim]
func GatherSelectedBlocks(blocks, indices *Array) *Array {
	B := int32(blocks.Dim(0))
	H := int32(blocks.Dim(1))
	nBlocks := int32(blocks.Dim(2))
	blockDim := int32(blocks.Dim(3))
	L := int32(indices.Dim(2))
	n := int32(indices.Dim(3))

	// blocks [B,H,nBlocks,blockDim] → [B,H,1,nBlocks,blockDim] → broadcast over
	// the query axis L → [B,H,L,nBlocks,blockDim].
	blocksQ := Reshape(blocks, B, H, 1, nBlocks, blockDim)
	blocksB := BroadcastTo(blocksQ, []int32{B, H, L, nBlocks, blockDim})
	Free(blocksQ)

	// indices [B,H,L,n] → [B,H,L,n,1] → broadcast over blockDim →
	// [B,H,L,n,blockDim] so TakeAlongAxis gathers along the block axis (3).
	idxE := Reshape(indices, B, H, L, n, 1)
	idxB := BroadcastTo(idxE, []int32{B, H, L, n, blockDim})
	Free(idxE)

	out := TakeAlongAxis(blocksB, idxB, 3) // gather along the nBlocks axis
	Free(blocksB, idxB)
	return out
}

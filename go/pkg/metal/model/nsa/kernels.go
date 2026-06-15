// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package nsa

import metal "dappco.re/go/mlx/pkg/metal"

// kernels.go holds the NSA branch kernels as pure functions over metal.Array so
// the unit test can pin each one on a fixed input without loading model
// weights. Everything composes from existing ops (ops.go / slice.go / fast.go);
// no branch needs a new Metal kernel.

// attend is the shared softmax-attention kernel: q·kᵀ → scale → (+mask) →
// softmax → ·v. q/k/v are [B,H,L,D]; mask (additive, broadcastable to
// [B,H,L,Lk]) may be nil. This is the same decomposition the metal SDPA paths
// use, kept in metal.Array ops so the selection / sliding branches can hand it
// an arbitrary additive mask.
//
//	out := attend(q, k, v, slidingMask(L, window), scale)
func attend(q, k, v, mask *metal.Array, scale float32) *metal.Array {
	kT := metal.Transpose4(k, 0, 1, 3, 2) // [B,H,D,Lk]
	scores := metal.Matmul(q, kT)         // [B,H,L,Lk]
	metal.Free(kT)
	if scale != 1 {
		scaled := metal.MulScalar(scores, scale)
		metal.Free(scores)
		scores = scaled
	}
	if mask != nil {
		masked := metal.Add(scores, mask)
		metal.Free(scores)
		scores = masked
	}
	probs := metal.Softmax(scores)
	metal.Free(scores)
	out := metal.Matmul(probs, v)
	metal.Free(probs)
	return out
}

// compressKV mean-pools K/V along the sequence into non-overlapping blocks of
// blockSize tokens. Input [B,H,L,D] → output [B,H,nBlocks,D] where nBlocks =
// L/blockSize (the trailing partial block, if any, is dropped — the compression
// branch summarises only whole blocks, matching the reference's block grid).
//
//	kCmp, vCmp := compressKV(k, v, 4) // 4 tokens per block
func compressKV(k, v *metal.Array, blockSize int32) (kCmp, vCmp *metal.Array) {
	B := int32(k.Dim(0))
	H := int32(k.Dim(1))
	L := int32(k.Dim(2))
	D := int32(k.Dim(3))
	nBlocks := L / blockSize

	// [B,H,L,D] → [B,H,nBlocks,blockSize,D] → mean over the blockSize axis.
	trim := L
	var kt, vt *metal.Array = k, v
	if nBlocks*blockSize != L {
		trim = nBlocks * blockSize
		kt = metal.Slice4(k, 0, 0, 0, 0, B, H, trim, D)
		vt = metal.Slice4(v, 0, 0, 0, 0, B, H, trim, D)
	}
	kBlocks := metal.Reshape(kt, B, H, nBlocks, blockSize, D)
	vBlocks := metal.Reshape(vt, B, H, nBlocks, blockSize, D)
	if trim != L {
		metal.Free(kt, vt)
	}
	kCmp = metal.Mean(kBlocks, 3, false) // [B,H,nBlocks,D]
	vCmp = metal.Mean(vBlocks, 3, false)
	metal.Free(kBlocks, vBlocks)
	return kCmp, vCmp
}

// blockCausalMask is the additive [1,1,L,nBlocks] mask for the compression
// branch: a query at token t may attend to block b only if the block's first
// token (b*blockSize) is at or before t — otherwise the block summarises future
// tokens. Allowed → 0, forbidden → -inf.
func blockCausalMask(L, nBlocks, blockSize int32) *metal.Array {
	data := make([]float32, int(L)*int(nBlocks))
	for t := int32(0); t < L; t++ {
		for b := int32(0); b < nBlocks; b++ {
			if b*blockSize <= t {
				data[int(t)*int(nBlocks)+int(b)] = 0
			} else {
				data[int(t)*int(nBlocks)+int(b)] = negInf
			}
		}
	}
	return metal.FromValues(data, 1, 1, int(L), int(nBlocks))
}

// slidingMask is the additive [1,1,L,L] mask for the sliding-window branch:
// query t attends to key j iff j <= t (causal) AND t-j < window (within the
// window). Allowed → 0, forbidden → -inf. window <= 0 means full causal.
func slidingMask(L, window int32) *metal.Array {
	data := make([]float32, int(L)*int(L))
	for t := int32(0); t < L; t++ {
		for j := int32(0); j < L; j++ {
			within := window <= 0 || (t-j) < window
			if j <= t && within {
				data[int(t)*int(L)+int(j)] = 0
			} else {
				data[int(t)*int(L)+int(j)] = negInf
			}
		}
	}
	return metal.FromValues(data, 1, 1, int(L), int(L))
}

// selectionMask turns per-block scores into an additive keep-mask for the top-n
// blocks per query, expressed without a discrete gather: TopK gives the n-th
// largest score per row (the selection threshold), Greater-or-equal against it
// yields the keep set. blockScores is [B,H,L,nBlocks] (already scaled);
// causalMask is the same additive mask used for the compression branch, applied
// first so future blocks (-inf) can never be selected. selectBlocks <= 0 or
// >= nBlocks keeps every (causally-valid) block. Output [B,H,L,nBlocks]:
// kept → 0, dropped → -inf.
func selectionMask(blockScores, causalMask *metal.Array, selectBlocks int32) *metal.Array {
	nBlocks := int32(blockScores.Dim(3))
	// Apply causal mask so future blocks sit at -inf and never win a top-n slot.
	scored := metal.Add(blockScores, causalMask)
	if selectBlocks <= 0 || selectBlocks >= nBlocks {
		// Keep all causally-valid blocks. The output must be a clean keep-mask
		// (0 kept, -inf dropped) — the same 0/-inf shape the top-n branch returns
		// via WhereScalarArray — because the caller ADDS it to the attention
		// logits. Returning `scored` here would leak the per-block ranking score
		// into the kept positions as an attention bias, sharpening the softmax
		// toward high-scoring blocks (a correctness bug that fires whenever
		// selectBlocks >= nBlocks = L/blockSize, i.e. across the whole short- and
		// medium-context regime, not just a degenerate edge). causalMask is
		// already exactly that 0/-inf keep-mask, so clone it and drop `scored`.
		metal.Free(scored)
		return causalMask.Clone()
	}
	// n-th largest score per row = the smallest score still inside the top-n.
	top := metal.TopK(scored, int(selectBlocks)) // [B,H,L,selectBlocks], ascending
	// Slice4 (scalar-pass, 0 Go heap allocs) over SliceAxis's two
	// make([]int32, ndim) per call (AX-11). top is rank-4; the last axis (the
	// selectBlocks dim) start 0 end 1 takes the smallest of the top-n.
	threshold := metal.Slice4(top, 0, 0, 0, 0,
		int32(top.Dim(0)), int32(top.Dim(1)), int32(top.Dim(2)), 1) // [B,H,L,1]
	metal.Free(top)
	keep := greaterEqual(scored, threshold) // bool [B,H,L,nBlocks]
	metal.Free(threshold)
	// kept → 0, dropped → -inf.
	out := metal.WhereScalarArray(keep, 0, neverSelect(nBlocks, blockScores))
	metal.Free(keep, scored)
	return out
}

// neverSelect is the all-(-inf) tensor shaped like blockScores, the "dropped"
// fill for selectionMask's Where. Built from a -inf scalar broadcast to the
// block-score shape.
func neverSelect(nBlocks int32, like *metal.Array) *metal.Array {
	shape := like.Shape()
	inf := metal.FromValue(negInf)
	out := metal.BroadcastTo(inf, shape)
	metal.Free(inf)
	return out
}

// greaterEqual returns a >= b as a bool array. ops.go exports Greater but not
// GreaterEqual; a >= b is ¬(b > a), so this composes it from Greater + a
// logical-not via equality-with-false. Kept local to NSA (the only consumer).
func greaterEqual(a, b *metal.Array) *metal.Array {
	bGtA := metal.Greater(b, a) // true where b > a, i.e. a < b
	falseScalar := metal.FromValue(false)
	ge := metal.Equal(bGtA, falseScalar) // true where ¬(a < b) ≡ a >= b
	metal.Free(bGtA, falseScalar)
	return ge
}

// expandBlockMaskToTokens broadcasts a per-block additive mask
// [B,H,L,nBlocks] to a per-token additive mask [B,H,L,L] by repeating each
// block's value across its blockSize tokens, then adding a token-level causal
// mask so a selected block still cannot leak future tokens. When the block grid
// covers fewer than L tokens (a trailing partial block was dropped), the
// causal-over-L mask forbids the uncovered tail.
func expandBlockMaskToTokens(blockMask *metal.Array, blockSize, L int32) *metal.Array {
	B := int32(blockMask.Dim(0))
	H := int32(blockMask.Dim(1))
	nBlocks := int32(blockMask.Dim(3))
	covered := nBlocks * blockSize

	// [B,H,L,nBlocks] → [B,H,L,nBlocks,1] → broadcast to
	// [B,H,L,nBlocks,blockSize] → reshape to [B,H,L,covered]. If covered < L,
	// pad the key axis up to L (PadAxis zero-fills; the causal-over-L add below
	// drives the pad columns to -inf since they sit beyond every query's reach).
	expanded := metal.Reshape(blockMask, B, H, L, nBlocks, 1)
	bcast := metal.BroadcastTo(expanded, []int32{B, H, L, nBlocks, blockSize})
	metal.Free(expanded)
	tokens := metal.Reshape(bcast, B, H, L, covered)
	metal.Free(bcast)
	if covered < L {
		padded := metal.PadAxis(tokens, 3, 0, int(L-covered))
		metal.Free(tokens)
		tokens = padded
	}

	causal := tokenCausalMask(L, covered) // forbids j > t AND j >= covered
	out := metal.Add(tokens, causal)
	metal.Free(tokens, causal)
	return out
}

// tokenCausalMask is the additive [1,1,L,L] causal mask: query t sees key j iff
// j <= t AND j < covered (keys in the dropped partial block are forbidden for
// every query). Allowed → 0, forbidden → -inf.
func tokenCausalMask(L, covered int32) *metal.Array {
	data := make([]float32, int(L)*int(L))
	for t := int32(0); t < L; t++ {
		for j := int32(0); j < L; j++ {
			if j <= t && j < covered {
				data[int(t)*int(L)+int(j)] = 0
			} else {
				data[int(t)*int(L)+int(j)] = negInf
			}
		}
	}
	return metal.FromValues(data, 1, 1, int(L), int(L))
}

// splitHeads reshapes a packed [B,L,H*D] projection into [B,H,L,D].
func splitHeads(x *metal.Array, B, L, heads, dim int32) *metal.Array {
	r := metal.Reshape(x, B, L, heads, dim)
	t := metal.Transpose4(r, 0, 2, 1, 3)
	metal.Free(r)
	return t
}

// mergeHeads is the inverse of splitHeads: [B,H,L,D] → [B,L,H*D].
func mergeHeads(x *metal.Array, B, L, heads, dim int32) *metal.Array {
	t := metal.Transpose4(x, 0, 2, 1, 3)
	r := metal.Reshape(t, B, L, heads*dim)
	metal.Free(t)
	return r
}

// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package moba

import metal "dappco.re/go/mlx/pkg/metal"

// kernels.go holds the MoBA kernels as pure functions over metal.Array so the
// unit test can pin each on a fixed input without loading model weights.
// Everything composes from existing ops (ops.go / slice.go); no new Metal
// kernel.

// attend is the shared softmax-attention kernel: q·kᵀ → scale → (+mask) →
// softmax → ·v. q/k/v are [B,H,L,D]; mask (additive, [.,.,L,L]) may be nil.
func attend(q, k, v, mask *metal.Array, scale float32) *metal.Array {
	kT := metal.Transpose4(k, 0, 1, 3, 2)
	scores := metal.Matmul(q, kT)
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

// blockScores computes the per-block gate score: q · mean_pool(K_block). Keys
// are mean-pooled into nBlocks = L/blockSize block representatives, then the
// query dots each representative. q/k are [B,H,L,D]; output [B,H,L,nBlocks],
// scaled by `scale` to match the attention scores it gates.
//
//	scores := blockScores(q, k, 4, 1.0/sqrt(D))
func blockScores(q, k *metal.Array, blockSize int32, scale float32) *metal.Array {
	B := int32(k.Dim(0))
	H := int32(k.Dim(1))
	L := int32(k.Dim(2))
	D := int32(k.Dim(3))
	nBlocks := L / blockSize
	covered := nBlocks * blockSize

	kt := k
	if covered != L {
		kt = metal.Slice4(k, 0, 0, 0, 0, B, H, covered, D)
	}
	kBlocks := metal.Reshape(kt, B, H, nBlocks, blockSize, D)
	if covered != L {
		metal.Free(kt)
	}
	rep := metal.Mean(kBlocks, 3, false) // [B,H,nBlocks,D] block representatives
	metal.Free(kBlocks)

	repT := metal.Transpose4(rep, 0, 1, 3, 2) // [B,H,D,nBlocks]
	metal.Free(rep)
	scores := metal.Matmul(q, repT) // [B,H,L,nBlocks]
	metal.Free(repT)
	if scale != 1 {
		scaled := metal.MulScalar(scores, scale)
		metal.Free(scores)
		scores = scaled
	}
	return scores
}

// blockSelectMask turns per-block gate scores into an additive keep-mask via
// top-k selection. The self-block (b == t/blockSize) is boosted to +inf before
// the top-k so it always wins a slot; the n-th largest boosted score per query
// is the selection threshold (TopK), and Greater-or-equal yields the keep set.
// A separate future-forbid mask is ADDED to the result so strictly-future
// blocks (b*blockSize > t) stay at -inf regardless of how the threshold lands
// (a top-k slot must not resurrect a future block). selectK <= 0 keeps only the
// self-block. Output [B,H,L,nBlocks]: kept → 0, dropped → -inf.
func blockSelectMask(scores *metal.Array, blockSize, selectK, L int32) *metal.Array {
	nBlocks := int32(scores.Dim(3))

	boost := selfBoost(L, nBlocks, blockSize) // +inf on self-block, else 0
	boosted := metal.Add(scores, boost)
	metal.Free(boost)

	// The self-block sits at +inf, always occupying a top slot. Keep selectK + 1
	// so the forced self-block leaves room for the top-k learned blocks
	// (clamped to nBlocks).
	keepN := selectK + 1
	if keepN < 1 {
		keepN = 1
	}
	if keepN > nBlocks {
		keepN = nBlocks
	}

	top := metal.TopK(boosted, int(keepN)) // ascending; [B,H,L,keepN]
	threshold := metal.SliceAxis(top, -1, 0, 1)
	metal.Free(top)
	keep := greaterEqual(boosted, threshold) // bool [B,H,L,nBlocks]
	metal.Free(threshold, boosted)

	selected := metal.WhereScalarArray(keep, 0, allNegInf(scores)) // 0 kept, -inf dropped
	metal.Free(keep)

	// Force strictly-future blocks back to -inf: a top-k slot must never select
	// a block that summarises future tokens. Additive: 0 (past/self) leaves the
	// selection alone, -inf (future) drives the column to -inf.
	forbid := futureForbid(L, nBlocks, blockSize)
	out := metal.Add(selected, forbid)
	metal.Free(selected, forbid)
	return out
}

// selfBoost is the additive [1,1,L,nBlocks] mask that is +inf on the self-block
// (b == t/blockSize) and 0 elsewhere — added to the gate scores before top-k so
// the self-block always wins a slot.
func selfBoost(L, nBlocks, blockSize int32) *metal.Array {
	posInf := -negInf
	data := make([]float32, int(L)*int(nBlocks))
	for t := int32(0); t < L; t++ {
		self := t / blockSize
		for b := int32(0); b < nBlocks; b++ {
			if b == self {
				data[int(t)*int(nBlocks)+int(b)] = posInf
			}
		}
	}
	return metal.FromValues(data, 1, 1, int(L), int(nBlocks))
}

// futureForbid is the additive [1,1,L,nBlocks] mask that is -inf on strictly-
// future blocks (b*blockSize > t) and 0 on past/self blocks — added to the
// selection result so future blocks can never be kept.
func futureForbid(L, nBlocks, blockSize int32) *metal.Array {
	data := make([]float32, int(L)*int(nBlocks))
	for t := int32(0); t < L; t++ {
		for b := int32(0); b < nBlocks; b++ {
			if b*blockSize > t {
				data[int(t)*int(nBlocks)+int(b)] = negInf
			}
		}
	}
	return metal.FromValues(data, 1, 1, int(L), int(nBlocks))
}

// expandToTokens broadcasts the per-block keep-mask [B,H,L,nBlocks] (already a
// clean additive 0/-inf mask from blockSelectMask) to a per-token additive mask
// [B,H,L,L] (repeat each block value across its tokens) and adds a tril causal
// mask so a query never sees a future token — in particular the self-block,
// always kept (0) by blockSelectMask, is causal-masked within itself by the
// add. The Minimum-with-0 clamp defends against any +inf leak before the causal
// add, since (+inf) + (−inf) = NaN; on the clean 0/-inf keep-mask it is a no-op.
func expandToTokens(blockMask *metal.Array, blockSize, L, nBlocks int32) *metal.Array {
	B := int32(blockMask.Dim(0))
	H := int32(blockMask.Dim(1))
	covered := nBlocks * blockSize

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

	finite := clampKeep(tokens)
	metal.Free(tokens)

	causal := causalMask(L, covered)
	out := metal.Add(finite, causal)
	metal.Free(finite, causal)
	return out
}

// clampKeep maps any +inf in the keep-mask down to 0 while leaving -inf
// (dropped) alone, via Minimum with a 0 scalar — a NaN-guard for the subsequent
// causal add ((+inf)+(−inf)=NaN). On blockSelectMask's clean 0/-inf output it
// is a no-op (0→0, −inf→−inf).
func clampKeep(mask *metal.Array) *metal.Array {
	zero := metal.FromValue(float32(0))
	out := metal.Minimum(mask, zero)
	metal.Free(zero)
	return out
}

// causalMask is the additive [1,1,L,L] tril mask: query t sees key j iff
// j <= t AND j < covered. Allowed → 0, forbidden → -inf.
func causalMask(L, covered int32) *metal.Array {
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

// greaterEqual returns a >= b as a bool array (ops.go has Greater but not
// GreaterEqual). a >= b ≡ ¬(b > a), composed from Greater + Equal-with-false.
func greaterEqual(a, b *metal.Array) *metal.Array {
	bGtA := metal.Greater(b, a)
	falseScalar := metal.FromValue(false)
	ge := metal.Equal(bGtA, falseScalar)
	metal.Free(bGtA, falseScalar)
	return ge
}

// allNegInf returns an all-(-inf) tensor shaped like `like` — the "dropped"
// fill for blockSelectMask's Where.
func allNegInf(like *metal.Array) *metal.Array {
	shape := like.Shape()
	inf := metal.FromValue(negInf)
	out := metal.BroadcastTo(inf, shape)
	metal.Free(inf)
	return out
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

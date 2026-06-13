// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

// Package flakernel holds the tensor helpers the flash-linear-attention mixers
// (gla, retnet, deltanet — and the SSM/sparse families that follow) compose
// from: causal triangle masks, per-head decay-matrix construction in the
// log-domain, and head-broadcast elementwise multiply. It lives under
// pkg/metal/model/internal so every mixer package beneath pkg/metal/model can
// import it while nothing outside the model tree can.
//
//	mask := flakernel.LowerTriangle(L)            // [L,L] causal keep-mask
//	D    := flakernel.PerHeadDecay(lnGamma, L)    // [H,L,L] γ_h^(i-j), causal
//	y    := flakernel.MulHeadBroadcast(x, m)      // [B,H,L,L] ⊙ [H,L,L]
package flakernel

import (
	"math"

	metal "dappco.re/go/mlx/pkg/metal"
)

// LowerTriangle returns the [L,L] causal keep-mask: 1.0 on and below the main
// diagonal (i ≥ j), 0.0 strictly above. Built from a column-index ≥ comparison
// so it needs no host loop.
//
//	keep := flakernel.LowerTriangle(L) // multiply attention scores to drop the future
func LowerTriangle(l int32) *metal.Array {
	rows := metal.Arange(0, float64(l), 1, metal.DTypeFloat32) // [L]
	rowsCol := metal.Reshape(rows, l, 1)                       // [L,1] = i
	colsRow := metal.Reshape(rows, 1, l)                       // [1,L] = j
	metal.Free(rows)
	// keep where i - j >= 0, i.e. (i+1) > j  →  Greater(i+1, j).
	rowsPlus := metal.AddScalar(rowsCol, 1) // [L,1] = i+1
	metal.Free(rowsCol)
	cond := metal.Greater(rowsPlus, colsRow) // [L,L] bool, true on/below diag
	metal.Free(rowsPlus, colsRow)
	keep := metal.AsType(cond, metal.DTypeFloat32) // 1.0 / 0.0
	metal.Free(cond)
	return keep
}

// PerHeadDecay returns the [H,L,L] per-head causal decay matrix with entries
// γ_h^(i-j) for i ≥ j and 0.0 below the diagonal, where lnGamma[h] = ln(γ_h).
// Computed in log-domain (exp(ln(γ_h)·(i-j))) so no per-head Pow is needed.
//
//	D := flakernel.PerHeadDecay(lnGamma, L) // RetNet / GLA scalar-decay mask
func PerHeadDecay(lnGamma []float32, l int32) *metal.Array {
	h := int32(len(lnGamma))
	rows := metal.Arange(0, float64(l), 1, metal.DTypeFloat32) // [L]
	rowsCol := metal.Reshape(rows, l, 1)                       // [L,1]
	colsRow := metal.Reshape(rows, 1, l)                       // [1,L]
	metal.Free(rows)
	diff := metal.Subtract(rowsCol, colsRow) // [L,L] = i-j
	metal.Free(rowsCol, colsRow)

	lnG := metal.FromValues(lnGamma, int(h), 1, 1) // [H,1,1]
	diffB := metal.Reshape(diff, 1, l, l)          // [1,L,L]
	metal.Free(diff)
	exponent := metal.Mul(lnG, diffB) // [H,L,L]
	metal.Free(lnG, diffB)
	weights := metal.Exp(exponent) // [H,L,L]
	metal.Free(exponent)

	keep := LowerTriangle(l)              // [L,L]
	keepB := metal.Reshape(keep, 1, l, l) // [1,L,L]
	metal.Free(keep)
	gated := metal.Mul(weights, keepB) // [H,L,L]
	metal.Free(weights, keepB)
	return gated
}

// MulHeadBroadcast multiplies a [B,H,L,L] score tensor by a [H,L,L] per-head
// mask, broadcasting the mask across the batch dimension.
//
//	masked := flakernel.MulHeadBroadcast(scores, decay) // apply per-head decay to QKᵀ
func MulHeadBroadcast(scores, headMask *metal.Array) *metal.Array {
	h := int32(headMask.Dim(0))
	l := int32(headMask.Dim(1))
	maskB := metal.Reshape(headMask, 1, h, l, l) // [1,H,L,L]
	out := metal.Mul(scores, maskB)              // broadcast over B
	metal.Free(maskB)
	return out
}

// MulCausalBroadcast multiplies a [B,H,L,L] score tensor by a [L,L] head-
// independent mask (the plain causal keep-mask), broadcasting it across both
// the batch and head dimensions. Used by mixers whose per-dimension decay is
// already folded into the queries/keys (GLA), leaving only the causal triangle
// to apply here.
//
//	masked := flakernel.MulCausalBroadcast(scores, keep) // drop the future
func MulCausalBroadcast(scores, mask *metal.Array) *metal.Array {
	l := int32(mask.Dim(0))
	maskB := metal.Reshape(mask, 1, 1, l, l) // [1,1,L,L]
	out := metal.Mul(scores, maskB)          // broadcast over B and H
	metal.Free(maskB)
	return out
}

// L2NormalizeLastAxis returns x / sqrt(sum(x², last-axis) + eps), normalising
// each vector along the final dimension to unit L2 length. DeltaNet normalises
// its keys this way (the delta rule's stability depends on it).
//
//	kn := flakernel.L2NormalizeLastAxis(k, 1e-6) // unit-norm keys per head
func L2NormalizeLastAxis(x *metal.Array, eps float32) *metal.Array {
	sq := metal.Square(x)                          // x²
	sumSq := metal.Sum(sq, len(x.Shape())-1, true) // Σ x² over last axis, keepdims
	metal.Free(sq)
	denomSq := metal.AddScalar(sumSq, eps) // Σ x² + eps
	metal.Free(sumSq)
	inv := metal.Rsqrt(denomSq) // 1/sqrt(Σ x² + eps)
	metal.Free(denomSq)
	out := metal.Mul(x, inv) // broadcast over the last axis
	metal.Free(inv)
	return out
}

// DefaultScale returns 1/sqrt(headDim) — the conventional query scaling for
// linear-attention mixers when the layer does not carry an explicit scale.
//
//	scale := flakernel.DefaultScale(headDim)
func DefaultScale(headDim int32) float32 {
	if headDim <= 0 {
		return 1
	}
	return float32(1.0 / math.Sqrt(float64(headDim)))
}

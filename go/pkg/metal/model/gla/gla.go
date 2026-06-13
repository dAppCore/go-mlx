// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

// Package gla implements Gated Linear Attention (GLA) as a pluggable sequence
// mixer on the metal Engine — the chunked gated-attention kernel plus the
// metal.MixerCompute scaffold the decoder loop dispatches through the scheme
// registry. GLA is a linear-attention mixer with a data-dependent, per-key-
// dimension forget gate, so it keeps a fixed-size recurrent state (Dk×Dv per
// head) and declares scheme.StateRecurrent.
//
// GLA (Yang et al. 2023, "Gated Linear Attention Transformers") generalises
// RetNet's scalar decay to a per-key-dimension gate α_t ∈ (0,1]^Dk that depends
// on the input. The recurrent form is
//
//	S_t = diag(α_t) · S_{t-1} + k_tᵀ v_t      (S is Dk × Dv; α decays each k-row)
//	o_t = q_t · S_t
//
// where α_t = exp(g_t) and g_t ≤ 0 is the log-forget gate (the layer projects
// and log-sigmoids it; the kernel takes g_t directly). Writing the cumulative
// log-decay within a chunk as b_t = Σ_{s≤t} g_s, the recurrence unrolls to the
// chunked parallel form computed here:
//
//	q̃_i = q_i ⊙ exp(b_i)      k̃_j = k_j ⊙ exp(-b_j)
//	o_i = Σ_{j≤i} (q̃_i · k̃_jᵀ) v_j                  (causal-masked QKᵀV)
//
// because (q_i ⊙ exp(b_i)) · (k_j ⊙ exp(-b_j)) = Σ_dk q_i[dk] exp(b_i[dk] −
// b_j[dk]) k_j[dk], which is exactly the per-dimension decay γ^(i-j) applied
// inside the dot product. The kernel also returns the advanced state S_L
// (decayed by exp(b_L)) so the decode loop can carry it across chunks once #1's
// recurrent-state holder lands.
package gla

import (
	metal "dappco.re/go/mlx/pkg/metal"
	flakernel "dappco.re/go/mlx/pkg/metal/model/internal/flakernel"
)

// Weights is the per-layer projection set a GLA layer needs. Q/K/V project the
// hidden state to per-head Q/K/V; GateProj + GateLow form the low-rank gate
// projection whose log-sigmoid is the per-key-dim forget gate; Output projects
// the per-head read-outs back to the model dimension. The gate-projection
// detail is the layer's; the kernel takes the resolved per-position log-gate g.
type Weights struct {
	QProj  *metal.Linear // [D, H*Dk] query projection
	KProj  *metal.Linear // [D, H*Dk] key projection
	VProj  *metal.Linear // [D, H*Dv] value projection
	Output *metal.Linear // [H*Dv, D] output projection

	NumHeads int     // H — number of GLA heads
	HeadDim  int     // Dk = Dv — per-head key/value dimension
	Scale    float32 // query scaling; 0 ⇒ 1/sqrt(HeadDim)
}

// State is the GLA recurrent state handed across chunks: the per-head gated
// matrix S of shape [B, H, Dk, Dv]. A nil State is the prefill case (S starts at
// zero); the kernel returns the advanced state so the next chunk continues.
type State struct {
	S *metal.Array // [B, H, Dk, Dv] — gated state, nil before the first chunk
}

// kernelInput is the resolved, validated geometry for one GatedChunk call.
type kernelInput struct {
	q, k, v, g *metal.Array // q,k,v [B,H,L,D]; g [B,H,L,Dk] per-key-dim log-gate
	prev       *metal.Array // [B,H,Dk,Dv] prior state, or nil
	b, h, l    int32
	headDim    int32
	scale      float32
}

// GatedChunk computes the chunked gated-attention for one chunk of hidden
// states and returns the per-head output [B,H,L,Dv] plus the advanced recurrent
// state [B,H,Dk,Dv].
//
// q,k,v are [B,H,L,D]; g is the per-position per-key-dim log-forget gate
// [B,H,L,Dk] (g ≤ 0). prev is the incoming recurrent state [B,H,Dk,Dv] or nil
// on prefill. scale is the query scaling (0 ⇒ 1/sqrt(D)).
//
//	out, newS := gla.GatedChunk(q, k, v, g, nil, scale)
func GatedChunk(q, k, v, g, prev *metal.Array, scale float32) (*metal.Array, *metal.Array) {
	in := kernelInput{q: q, k: k, v: v, g: g, prev: prev, scale: scale}
	if !in.resolve() {
		return nil, nil
	}
	return in.compute()
}

// resolve validates shapes and fills the derived geometry. Returns false on any
// mismatch so the caller surfaces a nil result rather than miscomputing.
func (in *kernelInput) resolve() bool {
	for _, a := range []*metal.Array{in.q, in.k, in.v, in.g} {
		if a == nil || !a.Valid() || len(a.Shape()) != 4 {
			return false
		}
	}
	in.b = int32(in.q.Dim(0))
	in.h = int32(in.q.Dim(1))
	in.l = int32(in.q.Dim(2))
	in.headDim = int32(in.q.Dim(3))
	// gate must be [B,H,L,Dk] matching q's geometry.
	if int32(in.g.Dim(0)) != in.b || int32(in.g.Dim(1)) != in.h ||
		int32(in.g.Dim(2)) != in.l || int32(in.g.Dim(3)) != in.headDim {
		return false
	}
	if in.scale == 0 {
		in.scale = flakernel.DefaultScale(in.headDim)
	}
	return true
}

// compute runs the chunked gated attention. The cumulative log-gate b_t folds
// the per-dimension decay into q̃ = q⊙exp(b) and k̃ = k⊙exp(−b); a plain causal
// QKᵀV over those gives the intra-chunk output, and the carried state is read
// with exp(b_i)-scaled queries.
//
// NOTE: exp(−b_j) grows as the cumulative log-gate b_j becomes very negative
// over a long chunk, so this single-chunk decomposition is numerically faithful
// for the prefill chunk sizes the decoder uses but is not unconditionally stable
// for an arbitrarily long chunk. The production hardening (chunk-local
// re-basing of b within fixed-size sub-chunks, as FLA's chunk kernel does) is a
// follow-up once the recurrent-state holder lands and real chunk lengths are
// known; the recurrence itself is exact.
func (in *kernelInput) compute() (*metal.Array, *metal.Array) {
	// b_t = cumsum over the time axis (axis 2) of the log-gate g. [B,H,L,Dk].
	b := metal.CumSum(in.g, 2, false, true) // inclusive forward cumulative sum
	expB := metal.Exp(b)                    // exp(b_t)
	negB := metal.Negative(b)
	expNegB := metal.Exp(negB) // exp(-b_t)
	metal.Free(negB)

	scaledQ := metal.MulScalar(in.q, in.scale) // [B,H,L,D]
	qTilde := metal.Mul(scaledQ, expB)         // q̃ = (scale·q) ⊙ exp(b)
	metal.Free(scaledQ)
	kTilde := metal.Mul(in.k, expNegB) // k̃ = k ⊙ exp(-b)
	metal.Free(expNegB)

	// Intra-chunk causal QKᵀV over the gated q̃, k̃.
	kT := metal.Transpose4(kTilde, 0, 1, 3, 2) // [B,H,Dk,L]
	scores := metal.Matmul(qTilde, kT)         // [B,H,L,L]
	metal.Free(kT)
	keep := flakernel.LowerTriangle(in.l) // [L,L] causal keep-mask
	masked := flakernel.MulCausalBroadcast(scores, keep)
	metal.Free(scores, keep)
	out := metal.Matmul(masked, in.v) // [B,H,L,Dv]
	metal.Free(masked)

	// Advanced state S_L = diag(exp(b_L)) · S_prev + Σ_j (k_j ⊙ exp(b_L − b_j))ᵀ v_j.
	newState := in.advanceState(b, expB)

	// Cross-chunk read-out: row i reads q̃_i · S_prev (q̃ already carries exp(b_i),
	// which is the decay applied to the carried state for position i).
	if in.prev != nil && in.prev.Valid() {
		cross := metal.Matmul(qTilde, in.prev) // [B,H,L,Dv]
		summed := metal.Add(out, cross)
		metal.Free(out, cross)
		out = summed
	}

	metal.Free(b, expB, qTilde, kTilde)
	return out, newState
}

// advanceState produces S_L for the next chunk:
//
//	S_L = diag(exp(b_L)) · S_prev + Σ_j (k_j ⊙ exp(b_L − b_j))ᵀ v_j
//
// b_L is the last time-step of the cumulative log-gate. Shape [B,H,Dk,Dv].
func (in *kernelInput) advanceState(b, expB *metal.Array) *metal.Array {
	l := in.l
	// b_L: last row of b over the time axis → [B,H,1,Dk].
	bLast := metal.Slice4(b, 0, 0, l-1, 0, in.b, in.h, l, in.headDim) // [B,H,1,Dk]
	expBLast := metal.Exp(bLast)                                      // exp(b_L), [B,H,1,Dk]
	metal.Free(bLast)

	// Inbound key decay exp(b_L − b_j) = exp(b_L) / exp(b_j) = expBLast ⊙ (1/expB).
	invExpB := metal.Reciprocal(expB)        // exp(-b_j) per position
	keyDecay := metal.Mul(expBLast, invExpB) // [B,H,L,Dk] broadcast b_L over L
	metal.Free(invExpB)
	decayedK := metal.Mul(in.k, keyDecay) // [B,H,L,Dk]
	metal.Free(keyDecay)
	dkT := metal.Transpose4(decayedK, 0, 1, 3, 2) // [B,H,Dk,L]
	metal.Free(decayedK)
	contrib := metal.Matmul(dkT, in.v) // [B,H,Dk,Dv]
	metal.Free(dkT)

	if in.prev == nil || !in.prev.Valid() {
		metal.Free(expBLast)
		return contrib
	}

	// diag(exp(b_L)) · S_prev: scale each k-row of S_prev by exp(b_L)[k].
	// expBLast is [B,H,1,Dk]; reshape to [B,H,Dk,1] to multiply rows of S.
	rowScale := metal.Reshape(expBLast, in.b, in.h, in.headDim, 1) // [B,H,Dk,1]
	metal.Free(expBLast)
	decayedPrev := metal.Mul(in.prev, rowScale) // [B,H,Dk,Dv] broadcast over Dv
	metal.Free(rowScale)
	newState := metal.Add(decayedPrev, contrib)
	metal.Free(decayedPrev, contrib)
	return newState
}

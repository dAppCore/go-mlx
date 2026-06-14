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
// (decayed by exp(b_L)); mixer.go's Forward threads it across chunks through the
// recurrent-state holder (ctx.Recurrent()).
//
// exp(−b_j) over a single global cumsum becomes unbounded as the chunk grows, so
// the kernel re-bases b within fixed ≤subChunk windows (FLA's chunk strategy):
// the cumsum restarts at each window head, bounding the largest magnitude to
// exp(subChunk·max|g|) regardless of total length, and the recurrent state
// threads across windows so the output is identical to the unrebased form for
// short chunks and stable for arbitrarily long ones.
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

// subChunk is the fixed sub-chunk length the hardened path re-bases the
// cumulative log-gate within. exp(−b) over a global cumsum becomes unbounded as
// the chunk grows; re-basing b to zero at the head of each ≤subChunk window
// bounds the largest magnitude to exp(subChunk·max|g|) regardless of total
// length, which is what FLA's chunk kernel does. 64 matches FLA's default chunk
// width; the recurrence is exact for any value because state threads across
// sub-chunks through the existing advanceState + cross-chunk read-out.
const subChunk = 64

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

// compute runs the chunked gated attention with chunk-local re-basing of the
// cumulative log-gate. The whole-chunk decomposition q̃ = q⊙exp(b),
// k̃ = k⊙exp(−b) is numerically faithful only while b stays bounded — exp(−b_j)
// grows without limit as b_j becomes very negative over a long chunk. So the
// chunk is split into ≤subChunk windows; within each window the cumulative
// log-gate is re-based to zero at the window head (exp(−b) bounded by
// exp(subChunk·max|g|)), and the recurrent state threads across windows through
// advanceState + the cross-chunk read-out. The recurrence is exact for any
// total length because each window continues from the previous window's state.
//
//	out, newS := in.compute() // stable at any L; q̃/k̃ re-based per ≤64-window
func (in *kernelInput) compute() (*metal.Array, *metal.Array) {
	if in.l == 1 {
		return in.decodeStep()
	}
	if in.l <= subChunk {
		// Single window: nothing to re-base, run the decomposition directly.
		return in.computeWindow(in.q, in.k, in.v, in.g, in.prev, in.l)
	}

	state := in.prev // carried across windows; nil ⇒ zero on the first
	outs := make([]*metal.Array, 0, (in.l+subChunk-1)/subChunk)
	for start := int32(0); start < in.l; start += subChunk {
		end := start + subChunk
		if end > in.l {
			end = in.l
		}
		wl := end - start
		qW := metal.Slice4(in.q, 0, 0, start, 0, in.b, in.h, end, in.headDim)
		kW := metal.Slice4(in.k, 0, 0, start, 0, in.b, in.h, end, in.headDim)
		vW := metal.Slice4(in.v, 0, 0, start, 0, in.b, in.h, end, in.headDim)
		gW := metal.Slice4(in.g, 0, 0, start, 0, in.b, in.h, end, in.headDim)

		outW, newState := in.computeWindow(qW, kW, vW, gW, state, wl)
		metal.Free(qW, kW, vW, gW)
		outs = append(outs, outW)

		// Free the intermediate state we produced (never the caller's in.prev).
		if state != in.prev {
			metal.Free(state)
		}
		state = newState
	}

	out := metal.Concatenate(outs, 2) // [B,H,L,Dv]
	metal.Free(outs...)
	return out, state
}

// computeWindow runs the q̃/k̃ gated-attention decomposition for one window of
// length wl, re-basing the cumulative log-gate to zero at the window head. q,k,v
// are [B,H,wl,D]; g is [B,H,wl,Dk]; prev is the state carried in from the prior
// window (or the caller's prefill state) or nil. Returns the window output
// [B,H,wl,Dv] and the advanced state [B,H,Dk,Dv].
func (in *kernelInput) computeWindow(q, k, v, g, prev *metal.Array, wl int32) (*metal.Array, *metal.Array) {
	// b_t = cumsum over the window's time axis (axis 2) of the log-gate g. Because
	// the cumsum restarts per window, b_0 = g_0 and exp(−b) stays bounded. [B,H,wl,Dk].
	b := metal.CumSum(g, 2, false, true) // inclusive forward cumulative sum
	expB := metal.Exp(b)                 // exp(b_t)
	negB := metal.Negative(b)
	expNegB := metal.Exp(negB) // exp(-b_t)
	metal.Free(negB)

	scaledQ := metal.MulScalar(q, in.scale) // [B,H,wl,D]
	qTilde := metal.Mul(scaledQ, expB)      // q̃ = (scale·q) ⊙ exp(b)
	metal.Free(scaledQ)
	kTilde := metal.Mul(k, expNegB) // k̃ = k ⊙ exp(-b)
	metal.Free(expNegB)

	// Intra-window causal QKᵀV over the gated q̃, k̃.
	kT := metal.Transpose4(kTilde, 0, 1, 3, 2) // [B,H,Dk,wl]
	scores := metal.Matmul(qTilde, kT)         // [B,H,wl,wl]
	metal.Free(kT)
	keep := flakernel.LowerTriangle(wl) // [wl,wl] causal keep-mask
	masked := flakernel.MulCausalBroadcast(scores, keep)
	metal.Free(scores, keep)
	out := metal.Matmul(masked, v) // [B,H,wl,Dv]
	metal.Free(masked)

	// Advanced state S_wl = diag(exp(b_wl)) · S_prev + Σ_j (k_j ⊙ exp(b_wl − b_j))ᵀ v_j.
	newState := advanceWindowState(k, v, b, expB, prev, in.b, in.h, wl, in.headDim)

	// Cross-window read-out: row i reads q̃_i · S_prev (q̃ already carries exp(b_i),
	// the decay applied to the carried state for position i, re-based to this window).
	if prev != nil && prev.Valid() {
		cross := metal.Matmul(qTilde, prev) // [B,H,wl,Dv]
		summed := metal.Add(out, cross)
		metal.Free(out, cross)
		out = summed
	}

	metal.Free(b, expB, qTilde, kTilde)
	return out, newState
}

// advanceWindowState produces the state at the end of a window for the next
// window to continue from:
//
//	S_wl = diag(exp(b_wl)) · S_prev + Σ_j (k_j ⊙ exp(b_wl − b_j))ᵀ v_j
//
// b_wl is the last time-step of the (window-local) cumulative log-gate. k,v are
// the window's [B,H,wl,D]; b,expB the window cumsum and its exp; prev the state
// carried in (or nil). Shape [B,H,Dk,Dv]. Because b is re-based per window, b_wl
// only spans this window, so exp(b_wl − b_j) stays bounded.
func advanceWindowState(k, v, b, expB, prev *metal.Array, batch, heads, wl, headDim int32) *metal.Array {
	// b_wl: last row of b over the time axis → [B,H,1,Dk].
	bLast := metal.Slice4(b, 0, 0, wl-1, 0, batch, heads, wl, headDim) // [B,H,1,Dk]
	expBLast := metal.Exp(bLast)                                       // exp(b_wl), [B,H,1,Dk]
	metal.Free(bLast)

	// Inbound key decay exp(b_wl − b_j) = exp(b_wl) / exp(b_j) = expBLast ⊙ (1/expB).
	invExpB := metal.Reciprocal(expB)        // exp(-b_j) per position
	keyDecay := metal.Mul(expBLast, invExpB) // [B,H,wl,Dk] broadcast b_wl over wl
	metal.Free(invExpB)
	decayedK := metal.Mul(k, keyDecay) // [B,H,wl,Dk]
	metal.Free(keyDecay)
	dkT := metal.Transpose4(decayedK, 0, 1, 3, 2) // [B,H,Dk,wl]
	metal.Free(decayedK)
	contrib := metal.Matmul(dkT, v) // [B,H,Dk,Dv]
	metal.Free(dkT)

	if prev == nil || !prev.Valid() {
		metal.Free(expBLast)
		return contrib
	}

	// diag(exp(b_wl)) · S_prev: scale each k-row of S_prev by exp(b_wl)[k].
	// expBLast is [B,H,1,Dk]; reshape to [B,H,Dk,1] to multiply rows of S.
	rowScale := metal.Reshape(expBLast, batch, heads, headDim, 1) // [B,H,Dk,1]
	metal.Free(expBLast)
	decayedPrev := metal.Mul(prev, rowScale) // [B,H,Dk,Dv] broadcast over Dv
	metal.Free(rowScale)
	newState := metal.Add(decayedPrev, contrib)
	metal.Free(decayedPrev, contrib)
	return newState
}

// decodeStep is the single-token (L==1) fast path: no cumsum, no window split,
// no exp(±b) re-basing machinery. For one token the cumulative gate b is just
// g₀, so the per-key-dim decay α₀ = exp(g₀) applies directly:
// S_new = diag(α₀)·S_prev + k₀ᵀv₀ and o = (scale·q₀)·S_new. q,k,v are [B,H,1,D];
// g is [B,H,1,Dk]. Returns the output [B,H,1,Dv] and the advanced state
// [B,H,Dk,Dv]. Numerically exact and stable (no exp(−b) term at all).
func (in *kernelInput) decodeStep() (*metal.Array, *metal.Array) {
	// k₀ᵀv₀ outer product: [B,H,Dk,1] @ [B,H,1,Dv] → [B,H,Dk,Dv].
	kCol := metal.Transpose4(in.k, 0, 1, 3, 2) // [B,H,Dk,1]
	kv := metal.Matmul(kCol, in.v)             // [B,H,Dk,Dv]
	metal.Free(kCol)

	var newState *metal.Array
	if in.prev != nil && in.prev.Valid() {
		// α₀ = exp(g₀), per key-dim: [B,H,1,Dk] → [B,H,Dk,1] to scale S_prev rows.
		alpha := metal.Exp(in.g)                                    // [B,H,1,Dk]
		alphaCol := metal.Reshape(alpha, in.b, in.h, in.headDim, 1) // [B,H,Dk,1]
		metal.Free(alpha)
		decayedPrev := metal.Mul(in.prev, alphaCol) // [B,H,Dk,Dv] broadcast over Dv
		metal.Free(alphaCol)
		newState = metal.Add(decayedPrev, kv)
		metal.Free(decayedPrev, kv)
	} else {
		newState = kv
	}

	// o = (scale·q₀) · S_new → [B,H,1,Dk] @ [B,H,Dk,Dv] = [B,H,1,Dv].
	scaledQ := metal.MulScalar(in.q, in.scale) // [B,H,1,Dk]
	out := metal.Matmul(scaledQ, newState)     // [B,H,1,Dv]
	metal.Free(scaledQ)
	return out, newState
}

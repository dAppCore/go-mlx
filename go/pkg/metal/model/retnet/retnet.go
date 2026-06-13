// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

// Package retnet implements Multiscale Retention (RetNet) as a pluggable
// sequence mixer on the metal Engine — the parallel-chunk retention kernel
// plus the metal.MixerCompute scaffold the decoder loop dispatches through the
// scheme registry. RetNet is a linear-attention mixer: it keeps a fixed-size
// recurrent state (a Dk×Dv matrix per head), not a growing K/V cache, so it
// declares scheme.StateRecurrent.
//
// Retention (Sun et al. 2023, "Retentive Network") is linear attention with a
// per-head scalar exponential decay γ. The recurrent form is
//
//	S_t = γ · S_{t-1} + k_tᵀ v_t          (S is Dk × Dv, outer product k⊗v)
//	o_t = q_t · S_t                        (read the state with the query)
//
// which unrolls to the parallel (chunk) form actually computed here:
//
//	o_i = Σ_{j≤i} γ^{i-j} (q_i · k_jᵀ) v_j = Σ_j (D_ij · (Q Kᵀ)_ij) v_j
//	D_ij = γ^{i-j} for i ≥ j, else 0       (the causal decay mask)
//
// Both forms are equivalent; the parallel form composes from matmuls so it maps
// directly onto the metal.Array op surface and is cheap on a Metal device for a
// prefill chunk. The kernel also returns the final recurrent state S_L so the
// decode loop can carry it across chunks once #1's recurrent-state holder lands.
package retnet

import (
	metal "dappco.re/go/mlx/pkg/metal"
	flakernel "dappco.re/go/mlx/pkg/metal/model/internal/flakernel"
)

// Weights is the per-layer projection + decay parameter set a RetNet layer
// needs. Q/K/V are dense (or quantised) projections from the model checkpoint;
// Output projects the per-head read-outs back to the model dimension. DecayLn
// is the per-head ln(γ) (RetNet derives γ_h as 1 − 2^(−5−h); the loader supplies
// the resolved log value so the decay mask is built without a per-head Pow).
//
//	w := &retnet.Weights{QProj: q, KProj: k, VProj: v, Output: o, NumHeads: 8, HeadDim: 64, DecayLn: ln}
type Weights struct {
	QProj  *metal.Linear // [D, H*Dk] query projection
	KProj  *metal.Linear // [D, H*Dk] key projection
	VProj  *metal.Linear // [D, H*Dv] value projection
	Output *metal.Linear // [H*Dv, D] output projection

	NumHeads int       // H — number of retention heads
	HeadDim  int       // Dk = Dv — per-head key/value dimension
	Scale    float32   // query scaling; 0 ⇒ 1/sqrt(HeadDim)
	DecayLn  []float32 // per-head ln(γ); len == NumHeads
}

// State is the RetNet recurrent state handed across chunks: the per-head
// retention matrix S of shape [B, H, Dk, Dv]. A nil State is the prefill case
// (S starts at zero); the kernel returns the advanced state so the next chunk
// continues the recurrence.
type State struct {
	S *metal.Array // [B, H, Dk, Dv] — retention state, nil before the first chunk
}

// kernelInput is the resolved, validated geometry for one RetentionChunk call —
// keeps the kernel arg list honest and the validation in one place.
type kernelInput struct {
	q, k, v *metal.Array // [B, H, L, D] each
	prev    *metal.Array // [B, H, Dk, Dv] prior state, or nil
	decayLn []float32    // per-head ln(γ)
	b, h, l int32
	headDim int32
	scale   float32
}

// RetentionChunk computes the parallel-form retention for one chunk of hidden
// states and returns the per-head output [B, H, L, Dv] plus the advanced
// recurrent state [B, H, Dk, Dv].
//
// q,k,v are [B, H, L, D] (heads already split out). prev is the incoming
// recurrent state [B, H, Dk, Dv] or nil on prefill. decayLn is per-head ln(γ);
// scale is the query scaling (0 ⇒ 1/sqrt(D)).
//
//	out, newS := retnet.RetentionChunk(q, k, v, nil, decayLn, scale)
func RetentionChunk(q, k, v, prev *metal.Array, decayLn []float32, scale float32) (*metal.Array, *metal.Array) {
	in := kernelInput{q: q, k: k, v: v, prev: prev, decayLn: decayLn, scale: scale}
	if !in.resolve() {
		return nil, nil
	}
	return in.compute()
}

// resolve validates shapes and fills the derived geometry. Returns false (so the
// caller surfaces a nil result rather than miscomputing) on any shape mismatch.
func (in *kernelInput) resolve() bool {
	if in.q == nil || in.k == nil || in.v == nil || !in.q.Valid() || !in.k.Valid() || !in.v.Valid() {
		return false
	}
	if len(in.q.Shape()) != 4 || len(in.k.Shape()) != 4 || len(in.v.Shape()) != 4 {
		return false
	}
	in.b = int32(in.q.Dim(0))
	in.h = int32(in.q.Dim(1))
	in.l = int32(in.q.Dim(2))
	in.headDim = int32(in.q.Dim(3))
	if int(in.h) != len(in.decayLn) {
		return false
	}
	if in.scale == 0 {
		in.scale = flakernel.DefaultScale(in.headDim)
	}
	return true
}

// compute runs the parallel-form retention: the intra-chunk attention
// (decay-masked Q Kᵀ V), plus, when a prior state is carried, the cross-chunk
// read-out (decayed q · S_prev), and produces the advanced state.
func (in *kernelInput) compute() (*metal.Array, *metal.Array) {
	if in.l == 1 {
		return in.decodeStep()
	}

	scaledQ := metal.MulScalar(in.q, in.scale) // [B,H,L,D]

	// Per-head decay mask D[h,i,j] = γ_h^(i-j) for i≥j else 0.
	decay := flakernel.PerHeadDecay(in.decayLn, in.l) // [H,L,L]

	// Intra-chunk: scores = scaledQ · kᵀ → [B,H,L,L], decay-masked, then · v.
	kT := metal.Transpose4(in.k, 0, 1, 3, 2) // [B,H,D,L]
	scores := metal.Matmul(scaledQ, kT)      // [B,H,L,L]
	metal.Free(kT)
	masked := flakernel.MulHeadBroadcast(scores, decay) // [B,H,L,L]
	metal.Free(scores)
	out := metal.Matmul(masked, in.v) // [B,H,L,Dv]
	metal.Free(masked)

	// Advanced state S_L = γ^L · S_prev + Σ_j γ^(L-1-j) k_jᵀ v_j.
	newState := in.advanceState(decay)

	// Cross-chunk read-out from the incoming state, if any.
	if in.prev != nil && in.prev.Valid() {
		cross := in.crossChunk(scaledQ) // [B,H,L,Dv]
		summed := metal.Add(out, cross)
		metal.Free(out, cross)
		out = summed
	}

	metal.Free(scaledQ, decay)
	return out, newState
}

// advanceState produces S_L = γ^L · S_prev + Σ_j γ^(L-1-j) k_jᵀ v_j for each
// head — the recurrent carry for the next chunk. Shape [B,H,Dk,Dv].
//
// The inbound decay of token j into S_L is γ_h^(L-1-j), which is exactly the
// last row of the decay matrix: decay[h, L-1, j] = γ_h^((L-1)-j).
func (in *kernelInput) advanceState(decay *metal.Array) *metal.Array {
	l := in.l
	lastRow := metal.Slice(decay, []int32{0, l - 1, 0}, []int32{in.h, l, l}) // [H,1,L]
	wRow := metal.Reshape(lastRow, in.h, l)                                  // [H,L]
	metal.Free(lastRow)

	wB := metal.Reshape(wRow, 1, in.h, l, 1) // [1,H,L,1] broadcast over batch+Dv
	metal.Free(wRow)
	decayedK := metal.Mul(in.k, wB) // [B,H,L,Dk]
	metal.Free(wB)
	dkT := metal.Transpose4(decayedK, 0, 1, 3, 2) // [B,H,Dk,L]
	metal.Free(decayedK)
	contrib := metal.Matmul(dkT, in.v) // [B,H,Dk,Dv]
	metal.Free(dkT)

	if in.prev == nil || !in.prev.Valid() {
		return contrib
	}

	gammaPowL := in.gammaPowL()                   // [H,1,1]
	gB := metal.Reshape(gammaPowL, 1, in.h, 1, 1) // [1,H,1,1]
	metal.Free(gammaPowL)
	decayedPrev := metal.Mul(in.prev, gB) // [B,H,Dk,Dv]
	metal.Free(gB)
	newState := metal.Add(decayedPrev, contrib)
	metal.Free(decayedPrev, contrib)
	return newState
}

// crossChunk reads the incoming recurrent state with each query: row i gets
// γ_h^(i+1) · (scaledQ_i · S_prev). The +1 offset is because S_prev already
// holds the contribution of every token up to the previous chunk's last
// position. Returns [B,H,L,Dv].
func (in *kernelInput) crossChunk(scaledQ *metal.Array) *metal.Array {
	readout := metal.Matmul(scaledQ, in.prev)                      // [B,H,L,Dv]
	idx := metal.Arange(1, float64(in.l)+1, 1, metal.DTypeFloat32) // [L] = 1..L
	idxRow := metal.Reshape(idx, 1, in.l)                          // [1,L]
	metal.Free(idx)
	lnG := metal.FromValues(in.decayLn, len(in.decayLn), 1) // [H,1]
	exponent := metal.Mul(lnG, idxRow)                      // [H,L]
	metal.Free(lnG, idxRow)
	rowDecay := metal.Exp(exponent) // [H,L]
	metal.Free(exponent)
	rB := metal.Reshape(rowDecay, 1, in.h, in.l, 1) // [1,H,L,1]
	metal.Free(rowDecay)
	out := metal.Mul(readout, rB)
	metal.Free(readout, rB)
	return out
}

// gammaPowL returns γ_h^L per head as [H,1,1] = exp(ln(γ_h)·L).
func (in *kernelInput) gammaPowL() *metal.Array {
	lnG := metal.FromValues(in.decayLn, len(in.decayLn), 1, 1) // [H,1,1]
	pow := metal.MulScalar(lnG, float32(in.l))
	metal.Free(lnG)
	out := metal.Exp(pow)
	metal.Free(pow)
	return out
}

// decodeStep is the single-token (L==1) fast path: one recurrent update with no
// decay mask, no cumsum, no per-token loop — just S_new = diag(γ)·S_prev +
// k₀ᵀv₀ and o = (scale·q₀)·S_new. q,k,v are [B,H,1,D]; γ is the per-head scalar
// exp(ln γ). Returns the output [B,H,1,Dv] and the advanced state [B,H,Dk,Dv].
func (in *kernelInput) decodeStep() (*metal.Array, *metal.Array) {
	// k₀ᵀv₀ outer product: [B,H,Dk,1] @ [B,H,1,Dv] → [B,H,Dk,Dv].
	kCol := metal.Transpose4(in.k, 0, 1, 3, 2) // [B,H,Dk,1]
	kv := metal.Matmul(kCol, in.v)             // [B,H,Dk,Dv]
	metal.Free(kCol)

	var newState *metal.Array
	if in.prev != nil && in.prev.Valid() {
		// γ_h per head as [1,H,1,1] = exp(ln γ_h); decay each prior row, add k₀ᵀv₀.
		gamma := in.gammaPerHead()               // [1,H,1,1]
		decayedPrev := metal.Mul(in.prev, gamma) // [B,H,Dk,Dv]
		metal.Free(gamma)
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

// gammaPerHead returns γ_h per head as [1,H,1,1] = exp(ln γ_h), shaped to scale
// the Dk rows of a [B,H,Dk,Dv] state.
func (in *kernelInput) gammaPerHead() *metal.Array {
	lnG := metal.FromValues(in.decayLn, 1, len(in.decayLn), 1, 1) // [1,H,1,1]
	out := metal.Exp(lnG)
	metal.Free(lnG)
	return out
}

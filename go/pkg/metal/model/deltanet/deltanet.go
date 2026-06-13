// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

// Package deltanet implements the delta-rule linear attention (DeltaNet) as a
// pluggable sequence mixer on the metal Engine — the delta-rule recurrence
// kernel plus the metal.MixerCompute scaffold the decoder loop dispatches
// through the scheme registry. DeltaNet is a linear-attention mixer keeping a
// fixed-size recurrent state (Dk×Dv per head), so it declares
// scheme.StateRecurrent.
//
// DeltaNet (Schlag et al. 2021 "DeltaNet"; Yang et al. 2024 "Parallelizing
// Linear Transformers with the Delta Rule") replaces linear attention's
// additive write with the delta rule: instead of always appending k_tᵀv_t to
// the value memory, it writes the prediction ERROR (v_t − current read-out)
// scaled by a per-token write strength β_t:
//
//	read_t   = S_{t-1}ᵀ k_t                         (current value at key k_t)
//	S_t      = S_{t-1} + β_t · k_t (v_t − read_t)ᵀ  (write the error)
//	         = S_{t-1}(I − β_t k_t k_tᵀ) + β_t k_t v_tᵀ   (equivalent form)
//	o_t      = q_t · S_t
//
// where k_t is L2-normalised over the head dimension (the normalisation is
// intrinsic to the rule's stability and is applied inside the kernel). S is
// Dk×Dv. β_t ∈ (0,1) is the per-token gate the layer produces (sigmoid of a
// projection); the kernel takes it directly.
//
// The chunked-parallel form of the delta rule needs a forward-substitution /
// WY-representation solve of the within-chunk triangular system (the writes are
// sequentially dependent through read_t), which is a larger undertaking than
// the other FLA mixers. This kernel implements the EXACT sequential recurrence
// — correct and fully tested — and flags the chunked-parallel optimisation as a
// follow-up. Per the task brief, a rough-but-tested kernel beats a guess.
package deltanet

import (
	metal "dappco.re/go/mlx/pkg/metal"
	flakernel "dappco.re/go/mlx/pkg/metal/model/internal/flakernel"
)

// Weights is the per-layer projection set a DeltaNet layer needs. Q/K/V project
// the hidden state to per-head Q/K/V; BetaProj produces the per-token write
// strength β; Output projects the per-head read-outs back to the model
// dimension. The β projection detail is the layer's; the kernel takes the
// resolved per-position β.
type Weights struct {
	QProj  *metal.Linear // [D, H*Dk] query projection
	KProj  *metal.Linear // [D, H*Dk] key projection
	VProj  *metal.Linear // [D, H*Dv] value projection
	Output *metal.Linear // [H*Dv, D] output projection

	NumHeads int     // H — number of heads
	HeadDim  int     // Dk = Dv — per-head key/value dimension
	Scale    float32 // query scaling; 0 ⇒ 1/sqrt(HeadDim)
	NormEps  float32 // L2-normalise epsilon for keys; 0 ⇒ a small default
}

// State is the DeltaNet recurrent state handed across chunks: the per-head
// value-memory matrix S of shape [B, H, Dk, Dv]. A nil State is the prefill case
// (S starts at zero); the kernel returns the advanced state.
type State struct {
	S *metal.Array // [B, H, Dk, Dv] — value memory, nil before the first chunk
}

const defaultNormEps = 1e-6

// kernelInput is the resolved, validated geometry for one DeltaRuleChunk call.
type kernelInput struct {
	q, k, v, beta *metal.Array // q,k,v [B,H,L,D]; beta [B,H,L,1] per-token write strength
	prev          *metal.Array // [B,H,Dk,Dv] prior state, or nil
	b, h, l       int32
	headDim       int32
	scale         float32
	normEps       float32
}

// DeltaRuleChunk computes the delta-rule recurrence for one chunk of hidden
// states and returns the per-head output [B,H,L,Dv] plus the advanced recurrent
// state [B,H,Dk,Dv].
//
// q,k,v are [B,H,L,D]; beta is the per-token write strength [B,H,L,1] (β ∈
// (0,1)). prev is the incoming recurrent state [B,H,Dk,Dv] or nil on prefill.
// scale is the query scaling (0 ⇒ 1/sqrt(D)); normEps is the key L2-normalise
// epsilon (0 ⇒ defaultNormEps). Keys are L2-normalised inside the kernel.
//
//	out, newS := deltanet.DeltaRuleChunk(q, k, v, beta, nil, scale, 0)
func DeltaRuleChunk(q, k, v, beta, prev *metal.Array, scale, normEps float32) (*metal.Array, *metal.Array) {
	in := kernelInput{q: q, k: k, v: v, beta: beta, prev: prev, scale: scale, normEps: normEps}
	if !in.resolve() {
		return nil, nil
	}
	return in.compute()
}

// resolve validates shapes and fills the derived geometry. Returns false on any
// mismatch so the caller surfaces a nil result rather than miscomputing.
func (in *kernelInput) resolve() bool {
	for _, a := range []*metal.Array{in.q, in.k, in.v, in.beta} {
		if a == nil || !a.Valid() || len(a.Shape()) != 4 {
			return false
		}
	}
	in.b = int32(in.q.Dim(0))
	in.h = int32(in.q.Dim(1))
	in.l = int32(in.q.Dim(2))
	in.headDim = int32(in.q.Dim(3))
	// beta must be [B,H,L,1].
	if int32(in.beta.Dim(0)) != in.b || int32(in.beta.Dim(1)) != in.h ||
		int32(in.beta.Dim(2)) != in.l || in.beta.Dim(3) != 1 {
		return false
	}
	if in.scale == 0 {
		in.scale = flakernel.DefaultScale(in.headDim)
	}
	if in.normEps == 0 {
		in.normEps = defaultNormEps
	}
	return true
}

// compute walks the delta-rule recurrence token by token, maintaining the
// [B,H,Dk,Dv] state and writing each output row. Keys are L2-normalised first;
// each step reads the current value at the key, computes the error, and writes
// it scaled by β.
func (in *kernelInput) compute() (*metal.Array, *metal.Array) {
	kNorm := flakernel.L2NormalizeLastAxis(in.k, in.normEps) // [B,H,L,D]
	scaledQ := metal.MulScalar(in.q, in.scale)               // [B,H,L,D]

	// Initial state: carried prev, or zeros [B,H,Dk,Dv].
	var state *metal.Array
	if in.prev != nil && in.prev.Valid() {
		state = in.prev.Clone()
	} else {
		state = metal.Zeros4(in.b, in.h, in.headDim, in.headDim, in.k.Dtype())
	}

	// Accumulate the per-token output rows, concatenated along the time axis.
	outRows := make([]*metal.Array, in.l)
	for t := int32(0); t < in.l; t++ {
		kt := sliceStep(kNorm, t)   // [B,H,1,Dk]
		vt := sliceStep(in.v, t)    // [B,H,1,Dv]
		qt := sliceStep(scaledQ, t) // [B,H,1,Dk]
		bt := sliceStep(in.beta, t) // [B,H,1,1]

		// read_t = S_{t-1}ᵀ k_t → [B,H,1,Dv]. (k_t·S over the Dk axis.)
		read := matVecOverState(kt, state) // [B,H,1,Dv]
		// error = v_t − read_t, scaled by β_t.
		errv := metal.Subtract(vt, read) // [B,H,1,Dv]
		metal.Free(read)
		betaErr := metal.Mul(errv, bt) // [B,H,1,Dv] (β broadcast over Dv)
		metal.Free(errv)

		// S_t = S_{t-1} + k_tᵀ · (β·error): outer product [Dk,1]×[1,Dv] → [Dk,Dv].
		ktCol := metal.Transpose4(kt, 0, 1, 3, 2) // [B,H,Dk,1]
		update := metal.Matmul(ktCol, betaErr)    // [B,H,Dk,Dv]
		metal.Free(ktCol, betaErr)
		newState := metal.Add(state, update)
		metal.Free(state, update)
		state = newState

		// o_t = q_t · S_t → [B,H,1,Dv].
		outRows[t] = matVecOverState(qt, state)
		metal.Free(kt, vt, qt, bt)
	}

	out := metal.Concatenate(outRows, 2) // [B,H,L,Dv]
	metal.Free(outRows...)
	metal.Free(kNorm, scaledQ)
	return out, state
}

// sliceStep extracts time-step t of a [B,H,L,X] tensor as [B,H,1,X].
func sliceStep(a *metal.Array, t int32) *metal.Array {
	b := int32(a.Dim(0))
	h := int32(a.Dim(1))
	x := int32(a.Dim(3))
	return metal.Slice4(a, 0, 0, t, 0, b, h, t+1, x)
}

// matVecOverState computes vec · S over the Dk axis: vec is [B,H,1,Dk], S is
// [B,H,Dk,Dv], result is [B,H,1,Dv]. A plain batched matmul over the last two
// dims does exactly this.
func matVecOverState(vec, state *metal.Array) *metal.Array {
	return metal.Matmul(vec, state) // [B,H,1,Dk] @ [B,H,Dk,Dv] = [B,H,1,Dv]
}

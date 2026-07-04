// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package rwkv7

import (
	core "dappco.re/go"
	metal "dappco.re/go/mlx/pkg/metal"
)

// recurrence.go is the RWKV-7 ("Goose") WKV7 kernel — the generalised
// delta-rule recurrence, expressed over MLX metal.Array ops as an exact
// sequential scan. It is the family-agnostic core: a model's receptance/key/
// value/decay/in-context-learning-rate projections are load-time detail, but
// the recurrence — decay the state, apply the diagonal-plus-rank-1 transition,
// add the key⊗value outer product, read out with the receptance — is what
// every RWKV-7 build runs, so it lives here as a pure function with no engine
// or model coupling.
//
// The reference is github.com/fla-org/flash-linear-attention
// (fla/ops/rwkv7/fused_recurrent), inner loop:
//
//	S = exp(w)[:,None] * S  +  b[:,None] * sum(a[:,None] * S, 0)[None,:]
//	S += k[:,None] * v[None,:]
//	o  = sum(S * r[:,None], 0)
//
// i.e. per head the state S is [K, V], the transition is
// diag(exp(w)) + b·aᵀ applied on the left (the K axis), then a k⊗v rank-1
// add, and the output o = Sᵀ·r is the receptance read-out. The chunked
// parallel form (the UT-transform / matmul reformulation) is a perf
// optimisation over this exact recurrence — see WKV7's doc for the new-kernel
// note.

// StepInput is one chunk of post-projection WKV7 inputs for a single layer. A
// model's load path runs the receptance/key/value/decay/a/b projections (incl.
// the in-context-learning-rate prep that forms a and b from kk); this struct is
// the canonical hand-off the recurrence consumes, matching the fla tensor
// convention so the math stays verifiable against the reference.
//
//	o, next := rwkv7.WKV7(rwkv7.StepInput{R: r, W: w, K: k, V: v, A: a, B: b}, prior)
type StepInput struct {
	R *metal.Array // [B, L, H, K] receptance (the read-out "query")
	W *metal.Array // [B, L, H, K] per-channel LOG-decay (state scaled by exp(W); W ≤ 0)
	K *metal.Array // [B, L, H, K] key
	V *metal.Array // [B, L, H, V] value
	A *metal.Array // [B, L, H, K] in-context learning-rate vector (the removal direction)
	B *metal.Array // [B, L, H, K] in-context learning-rate vector (the write-back direction)
}

// WKV7 runs the RWKV-7 recurrence for one chunk and returns the mixed output
// o [B, L, H, V] and the advanced state [B, H, K, V]. prior is the carried
// state from the previous chunk (decode) or nil for a fresh sequence
// (prefill), in which case the recurrence starts from a zero state.
//
// The recurrence, per timestep t (fla fused_recurrent, state S is [B,H,K,V]):
//
//	Sa  = aᵀ · S                                   // [B,H,1,V] contract K
//	S   = diag(exp(w)) · S  +  b ⊗ Sa  +  k ⊗ v    // [B,H,K,V]
//	o   = Sᵀ · r                                   // [B,H,V]   contract K
//
// where w/r/k/a/b are [B,H,K] and v is [B,H,V]. This is the exact sequential
// reference; it is O(L) graph nodes, which is correct and the right shape for
// decode (L=1) and as the oracle the chunked path is checked against. The
// length-parallel chunked form (the UT-transform reformulation) that cuts the L
// serial dependency for prefill is WKV7Chunked in chunk.go — composed from
// matmuls + a per-window triangular solve (TriInv), no custom Metal kernel.
func WKV7(in StepInput, prior *metal.Array) (*metal.Array, *metal.Array) {
	B := int32(in.R.Dim(0))
	L := int32(in.R.Dim(1))
	H := int32(in.R.Dim(2))
	K := int32(in.R.Dim(3))
	V := int32(in.V.Dim(3))
	dtype := in.R.Dtype()

	// State [B,H,K,V]; zero on prefill, the carried state on decode.
	var state *metal.Array
	if prior != nil && prior.Valid() {
		state = prior.Clone()
	} else {
		state = metal.Zeros([]int32{B, H, K, V}, dtype)
	}

	outputs := make([]*metal.Array, 0, L)
	for t := int32(0); t < L; t++ {
		// Per-timestep views, reshaped so the K axis is the row axis (dim 2) for
		// the diagonal decay + the rank-1 updates, matching the [K,V] state. The
		// StepInput fields are rank-4 by contract ([B,L,H,·]), so the length-axis
		// slice uses the scalar-pass Slice4 (no per-call []int32 starts/ends heap
		// alloc that SliceAxis pays) — byte-identical to SliceAxis(·,1,t,t+1).
		rtRaw := metal.Slice4(in.R, 0, t, 0, 0, B, t+1, H, K) // [B,1,H,K]
		rt := metal.Reshape(rtRaw, B, H, K, 1)                // [B,H,K,1]
		metal.Free(rtRaw)

		wtRaw := metal.Slice4(in.W, 0, t, 0, 0, B, t+1, H, K) // [B,1,H,K]
		wt := metal.Reshape(wtRaw, B, H, K, 1)                // [B,H,K,1]
		metal.Free(wtRaw)

		ktRaw := metal.Slice4(in.K, 0, t, 0, 0, B, t+1, H, K) // [B,1,H,K]
		kt := metal.Reshape(ktRaw, B, H, K, 1)                // [B,H,K,1] (column for k⊗v)
		metal.Free(ktRaw)

		vtRaw := metal.Slice4(in.V, 0, t, 0, 0, B, t+1, H, V) // [B,1,H,V]
		vt := metal.Reshape(vtRaw, B, H, 1, V)                // [B,H,1,V] (row for the outer products)
		metal.Free(vtRaw)

		atRaw := metal.Slice4(in.A, 0, t, 0, 0, B, t+1, H, K) // [B,1,H,K]
		at := metal.Reshape(atRaw, B, H, K, 1)                // [B,H,K,1]
		metal.Free(atRaw)

		btRaw := metal.Slice4(in.B, 0, t, 0, 0, B, t+1, H, K) // [B,1,H,K]
		bt := metal.Reshape(btRaw, B, H, K, 1)                // [B,H,K,1] (column for b⊗Sa)
		metal.Free(btRaw)

		// Sa = aᵀ · S  →  contract the K axis: (a [B,H,K,1] ⊙ S [B,H,K,V]) summed
		// over K gives [B,H,1,V].
		aS := metal.Mul(at, state) // [B,H,K,V] broadcast a over V
		metal.Free(at)
		sa := metal.Sum(aS, 2, true) // [B,H,1,V] keep the K axis as size 1
		metal.Free(aS)

		// Decayed state: diag(exp(w)) · S — exp(w) [B,H,K,1] scales the K rows.
		ew := metal.Exp(wt) // [B,H,K,1]
		metal.Free(wt)
		decayed := metal.Mul(state, ew) // [B,H,K,V] broadcast over V
		metal.Free(state)
		metal.Free(ew)

		// Rank-1 transition: b ⊗ Sa — b [B,H,K,1] ⊗ Sa [B,H,1,V] = [B,H,K,V].
		bSa := metal.Mul(bt, sa) // [B,H,K,V]
		metal.Free(bt)
		metal.Free(sa)

		// Rank-1 write: k ⊗ v — k [B,H,K,1] ⊗ v [B,H,1,V] = [B,H,K,V].
		kv := metal.Mul(kt, vt) // [B,H,K,V]
		metal.Free(kt)
		metal.Free(vt)

		// S = decayed + b⊗Sa + k⊗v
		s1 := metal.Add(decayed, bSa)
		metal.Free(decayed)
		metal.Free(bSa)
		state = metal.Add(s1, kv)
		metal.Free(s1)
		metal.Free(kv)

		// o = Sᵀ · r — contract the K axis: (r [B,H,K,1] ⊙ S [B,H,K,V]) summed
		// over K gives [B,H,1,V] → [B,H,V].
		rS := metal.Mul(rt, state) // [B,H,K,V]
		metal.Free(rt)
		oSum := metal.Sum(rS, 2, false) // [B,H,V]
		metal.Free(rS)

		// Re-introduce the length axis so the chunk concatenates to [B,L,H,V].
		oL := metal.Reshape(oSum, B, 1, H, V)
		metal.Free(oSum)
		outputs = append(outputs, oL)
	}

	var o *metal.Array
	if L == 1 {
		o = outputs[0]
	} else {
		o = metal.Concatenate(outputs, 1) // [B,L,H,V]
	}

	// Eval before releasing the per-timestep slices: the concat result is a lazy
	// graph node over the (unevaluated) reshape nodes in outputs, so freeing them
	// first would drop the data the concat still needs.
	if err := metal.Eval(o, state); err != nil {
		core.Error("rwkv7: WKV7 recurrence eval failed", "error", err, "B", B, "L", L, "H", H, "K", K, "V", V)
	}
	if L > 1 {
		metal.Free(outputs...)
	}
	return o, state
}

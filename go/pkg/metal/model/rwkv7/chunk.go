// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

// chunk.go adds the chunked-PARALLEL RWKV-7 form alongside the exact sequential
// recurrence in recurrence.go. recurrence.go walks L timesteps one at a time
// (the correct shape for decode, L=1); this replaces that host-side loop with
// dense matmuls plus one triangular inverse per ≤chunkWidth window, so prefill
// is bound by batched matmuls rather than L serial steps.
//
// RWKV-7's transition is diagonal-plus-low-rank (DPLR): per step the state is
// S_t = (diag(exp(w_t)) + b_t a_tᵀ)·S_{t-1} + k_t v_tᵀ, so unrolling a chunk's
// transition product needs the WY / UT-transform — exactly chunk_dplr_delta_rule
// in flash-linear-attention (fla/ops/generalized_delta_rule/dplr). The reference
// here is its readable form, dplr_chunkwise. With the per-channel cumulative
// log-decay c_i = Σ_{s≤i} w_s, define the decay-folded vectors
//
//	q̃_i = r_i ⊙ exp(c_i)     k̃_j = k_j ⊙ exp(−c_j)
//	b̃_j = b_j ⊙ exp(−c_j)     ã_i = a_i ⊙ exp(c_i − w_i)
//
// and the four intra-chunk attention matrices ([C,C], per (B,H)):
//
//	A_qk = tril_incl(q̃ k̃ᵀ)   A_qb = tril_incl(q̃ b̃ᵀ)
//	A_ak = strict_low(ã k̃ᵀ)   A_ab = strict_low(ã b̃ᵀ)
//
// The sequentially-dependent rank-1 removals collapse to (I − A_ab)⁻¹ (A_ab is
// strictly lower so I − A_ab is unit lower-triangular); applying it gives the
// pseudo-values
//
//	A_inv = (I − A_ab)⁻¹                         (metal.TriInv, reused twice)
//	u = A_inv (A_ak V)     w = A_inv ã            (the two within-chunk solves)
//
// after which each window threads the recurrent state S:
//
//	v2 = u + w S
//	O  = A_qk V + A_qb v2 + q̃ S
//	S  = exp(c_C) S + (k ⊙ decay)ᵀ V + (b ⊙ decay)ᵀ v2   decay = exp(c_C − c)
//
// (I − A_ab)⁻¹ is exact, so the chunked output equals the sequential WKV7 output
// to machine precision; chunk_test.go's chunked-vs-sequential test proves it.
// WKV7 stays the correct fallback and the decode (L=1) path.
package rwkv7

import (
	metal "dappco.re/go/mlx/pkg/metal"
)

// chunkWidth is the window the parallel form inverts at once. Longer sequences
// are split into ≤chunkWidth windows that thread the recurrent state, bounding
// the per-window triangular system to chunkWidth×chunkWidth and re-basing the
// cumulative decay at each window so exp(−c) never accumulates beyond chunkWidth
// steps (the same numerical guard DeltaNet's chunked form uses). 64 matches
// FLA's default chunk size.
const chunkWidth = 64

// chunkInput is the resolved, validated geometry for one WKV7Chunked call,
// mirroring the StepInput tensors as head-major [B,H,L,·] views.
type chunkInput struct {
	r, w, k, v, a, b *metal.Array // r,w,k,a,b [B,H,L,K]; v [B,H,L,V]
	prev             *metal.Array // [B,H,K,V] prior state, or nil
	bsz, h, l        int32
	dk, dv           int32
	width            int32 // window width (chunkWidth in production; tunable for tests)
}

// WKV7Chunked runs the length-parallel RWKV-7 recurrence for one chunk of inputs
// and returns the mixed output o [B, L, H, V] and the advanced state
// [B, H, K, V] — the same signature and the same result as WKV7, but computed
// with the chunked DPLR matmul + triangular-solve form instead of the serial
// timestep loop. prior is the carried state from the previous chunk (decode) or
// nil for a fresh sequence (prefill).
//
// The result equals WKV7 to within float rounding (the triangular inverse is
// exact); WKV7Chunked is the prefill-speed path, WKV7 stays the exact fallback
// and the decode (L=1) path. A nil result is surfaced if the LAPACK triangular
// inverse fails, so the caller can fall back to WKV7.
//
//	o, next := rwkv7.WKV7Chunked(in, prior)
func WKV7Chunked(in StepInput, prior *metal.Array) (*metal.Array, *metal.Array) {
	return wkv7ChunkedWindowed(in, prior, chunkWidth)
}

// wkv7ChunkedWindowed is WKV7Chunked with the window width as a parameter, so
// the inter-window state-carry path can be exercised on short test sequences
// (a width smaller than L forces multiple windows). Production always uses
// chunkWidth via WKV7Chunked; width <= 0 falls back to chunkWidth.
func wkv7ChunkedWindowed(in StepInput, prior *metal.Array, width int32) (*metal.Array, *metal.Array) {
	if width <= 0 {
		width = chunkWidth
	}
	ci := chunkInput{
		width: width,
		bsz:   int32(in.R.Dim(0)),
		l:     int32(in.R.Dim(1)),
		h:     int32(in.R.Dim(2)),
		dk:    int32(in.R.Dim(3)),
		dv:    int32(in.V.Dim(3)),
		prev:  prior,
	}
	// Head-major views: [B,L,H,·] → [B,H,L,·] so the attention matmuls batch
	// over (B,H) the way metal.Matmul contracts the trailing two axes.
	ci.r = metal.Transpose4(in.R, 0, 2, 1, 3) // [B,H,L,K]
	ci.w = metal.Transpose4(in.W, 0, 2, 1, 3) // [B,H,L,K]
	ci.k = metal.Transpose4(in.K, 0, 2, 1, 3) // [B,H,L,K]
	ci.v = metal.Transpose4(in.V, 0, 2, 1, 3) // [B,H,L,V]
	ci.a = metal.Transpose4(in.A, 0, 2, 1, 3) // [B,H,L,K]
	ci.b = metal.Transpose4(in.B, 0, 2, 1, 3) // [B,H,L,K]

	o, state := ci.compute()
	metal.Free(ci.r, ci.w, ci.k, ci.v, ci.a, ci.b)
	if o == nil {
		return nil, nil
	}

	// Back to length-major [B,L,H,V] to match WKV7's output layout.
	oLM := metal.Transpose4(o, 0, 2, 1, 3) // [B,L,H,V]
	metal.Free(o)
	if err := metal.Eval(oLM, state); err != nil {
		// Eval failure leaves the caller with the sequential fallback.
		metal.Free(oLM, state)
		return nil, nil
	}
	return oLM, state
}

// compute windows the chunk into ≤chunkWidth segments, threading the recurrent
// state across them, and returns the head-major output [B,H,L,V] and advanced
// state [B,H,K,V]. A nil output is surfaced if any window's triangular inverse
// fails.
func (in *chunkInput) compute() (*metal.Array, *metal.Array) {
	state := in.prev // carried across windows; nil ⇒ zero on the first
	outs := make([]*metal.Array, 0, (in.l+in.width-1)/in.width)
	for start := int32(0); start < in.l; start += in.width {
		end := start + in.width
		if end > in.l {
			end = in.l
		}
		wl := end - start
		win := window{
			bsz: in.bsz, h: in.h, dk: in.dk, dv: in.dv, wl: wl,
			r: metal.Slice4(in.r, 0, 0, start, 0, in.bsz, in.h, end, in.dk),
			w: metal.Slice4(in.w, 0, 0, start, 0, in.bsz, in.h, end, in.dk),
			k: metal.Slice4(in.k, 0, 0, start, 0, in.bsz, in.h, end, in.dk),
			v: metal.Slice4(in.v, 0, 0, start, 0, in.bsz, in.h, end, in.dv),
			a: metal.Slice4(in.a, 0, 0, start, 0, in.bsz, in.h, end, in.dk),
			b: metal.Slice4(in.b, 0, 0, start, 0, in.bsz, in.h, end, in.dk),
		}
		outW, newState := win.solve(state)
		metal.Free(win.r, win.w, win.k, win.v, win.a, win.b)
		if outW == nil {
			if state != in.prev {
				metal.Free(state)
			}
			metal.Free(outs...)
			return nil, nil
		}
		outs = append(outs, outW)
		if state != in.prev {
			metal.Free(state)
		}
		state = newState
	}

	var o *metal.Array
	if len(outs) == 1 {
		o = outs[0]
	} else {
		o = metal.Concatenate(outs, 2) // [B,H,L,V]
		metal.Free(outs...)
	}
	// If the sequence was empty (l==0) outs is empty; guard so state is still
	// the carried-in (or a zero) state.
	if o == nil {
		o = metal.Zeros([]int32{in.bsz, in.h, in.l, in.dv}, in.v.Dtype())
	}
	if state == in.prev && state != nil {
		// The caller still owns prev; return a clone so ownership is uniform
		// (caller frees the returned state).
		state = state.Clone()
	} else if state == nil {
		state = metal.Zeros([]int32{in.bsz, in.h, in.dk, in.dv}, in.v.Dtype())
	}
	return o, state
}

// window is one ≤chunkWidth segment's resolved tensors (head-major,
// [B,H,wl,·]).
type window struct {
	r, w, k, v, a, b   *metal.Array
	bsz, h, dk, dv, wl int32
}

// solve runs the DPLR chunked form for one window against the carried state and
// returns the window output [B,H,wl,V] and the advanced state [B,H,K,V], or
// (nil,nil) if the triangular inverse fails.
func (s window) solve(prev *metal.Array) (*metal.Array, *metal.Array) {
	wl := s.wl

	// Per-channel cumulative log-decay c_i = Σ_{s≤i} w_s within the window.
	c := metal.CumSum(s.w, 2, false, true) // [B,H,wl,K] inclusive forward cumsum
	expC := metal.Exp(c)                   // exp(c_i)
	negC := metal.Negative(c)
	expNegC := metal.Exp(negC) // exp(−c_j)
	metal.Free(negC)
	// ã decay factor exp(c_i − w_i) = exp(c_{i-1}) (cumsum up to the previous
	// step) — the "shift by one" the DPLR reference applies to the removal-side
	// matrices A_ab / A_ak.
	cShift := metal.Subtract(c, s.w) // c_i − w_i
	expCShift := metal.Exp(cShift)   // exp(c_i − w_i)
	metal.Free(cShift)

	// Decay-folded vectors (all [B,H,wl,·]).
	qTilde := metal.Mul(s.r, expC)      // q̃ = r ⊙ exp(c)
	kTilde := metal.Mul(s.k, expNegC)   // k̃ = k ⊙ exp(−c)
	bTilde := metal.Mul(s.b, expNegC)   // b̃ = b ⊙ exp(−c)
	aTilde := metal.Mul(s.a, expCShift) // ã = a ⊙ exp(c − w)
	metal.Free(expNegC, expCShift)

	kTildeT := metal.Transpose4(kTilde, 0, 1, 3, 2) // [B,H,K,wl]
	bTildeT := metal.Transpose4(bTilde, 0, 1, 3, 2) // [B,H,K,wl]

	// Four intra-chunk attention matrices [B,H,wl,wl].
	inclMask := inclusiveLowerMask(wl) // i ≥ j
	strictMask := strictLowerMask(wl)  // i > j

	aqkFull := metal.Matmul(qTilde, kTildeT) // q̃ k̃ᵀ
	aQK := mulMaskBroadcast(aqkFull, inclMask)
	metal.Free(aqkFull)

	aqbFull := metal.Matmul(qTilde, bTildeT) // q̃ b̃ᵀ
	aQB := mulMaskBroadcast(aqbFull, inclMask)
	metal.Free(aqbFull)

	aakFull := metal.Matmul(aTilde, kTildeT) // ã k̃ᵀ
	aAK := mulMaskBroadcast(aakFull, strictMask)
	metal.Free(aakFull)

	aabFull := metal.Matmul(aTilde, bTildeT) // ã b̃ᵀ
	aAB := mulMaskBroadcast(aabFull, strictMask)
	metal.Free(aabFull)
	// qTilde is kept — the carried-state read-out q̃ S in combine needs it again.
	metal.Free(kTildeT, bTildeT, kTilde, bTilde)
	metal.Free(inclMask, strictMask)

	// A_inv = (I − A_ab)⁻¹. A_ab is strictly lower-triangular, so I − A_ab is
	// unit lower-triangular and its inverse is lower-triangular too.
	eye := eyeMatrix(wl, s.v.Dtype())
	imAB := metal.Subtract(eye, aAB) // I − A_ab
	metal.Free(eye, aAB)
	imF, imCast := asFloat32(imAB)
	aInvF, err := metal.TriInv(imF, false) // lower-triangular inverse
	if imCast {
		metal.Free(imF)
	}
	metal.Free(imAB)
	if err != nil || aInvF == nil {
		metal.Free(aQK, aQB, aAK, aTilde, qTilde, c, expC)
		return nil, nil
	}
	aInv := castBack(aInvF, s.v.Dtype()) // [B,H,wl,wl]

	// Pseudo-values: u = A_inv (A_ak V), w = A_inv ã.
	akv := metal.Matmul(aAK, s.v) // A_ak V [B,H,wl,V]
	metal.Free(aAK)
	u := metal.Matmul(aInv, akv) // [B,H,wl,V]
	metal.Free(akv)
	wMat := metal.Matmul(aInv, aTilde) // A_inv ã [B,H,wl,K]
	metal.Free(aInv, aTilde)

	// Window output and state update, threading the carried state. qTilde was
	// built above for the attention matrices and is reused here for q̃ S; combine
	// frees it.
	out, newState := s.combine(prev, aQK, aQB, qTilde, u, wMat, c, expC)
	metal.Free(aQK, aQB, u, wMat, c, expC)
	return out, newState
}

// combine produces the window output O = A_qk V + A_qb v2 + q̃ S and the advanced
// state S' = exp(c_C) S + (k ⊙ decay)ᵀ V + (b ⊙ decay)ᵀ v2, where v2 = u + w S
// and decay_i = exp(c_C − c_i). qTilde is q̃ [B,H,wl,K]; it is freed here.
func (s window) combine(prev, aQK, aQB, qTilde, u, wMat, c, expC *metal.Array) (*metal.Array, *metal.Array) {
	wl := s.wl

	// v2 = u + w S (the rank-1 write-back values, carried-state corrected).
	v2 := u
	freeV2 := false
	if prev != nil && prev.Valid() {
		wS := metal.Matmul(wMat, prev) // w S [B,H,wl,V]
		v2 = metal.Add(u, wS)
		metal.Free(wS)
		freeV2 = true
	}

	// O = A_qk V + A_qb v2 (+ q̃ S).
	oQK := metal.Matmul(aQK, s.v) // A_qk V [B,H,wl,V]
	oQB := metal.Matmul(aQB, v2)  // A_qb v2 [B,H,wl,V]
	out := metal.Add(oQK, oQB)
	metal.Free(oQK, oQB)
	if prev != nil && prev.Valid() {
		qS := metal.Matmul(qTilde, prev) // q̃ S [B,H,wl,V]
		summed := metal.Add(out, qS)
		metal.Free(out, qS)
		out = summed
	}
	metal.Free(qTilde)

	// State update. decay_i = exp(c_C − c_i): c_C is the last (cumulative) row.
	cLast := metal.Slice4(c, 0, 0, wl-1, 0, s.bsz, s.h, wl, s.dk) // [B,H,1,K]
	diff := metal.Subtract(cLast, c)                              // c_C − c_i [B,H,wl,K]
	decay := metal.Exp(diff)                                      // [B,H,wl,K]
	metal.Free(diff)

	kDecay := metal.Mul(s.k, decay) // (k ⊙ decay) [B,H,wl,K]
	bDecay := metal.Mul(s.b, decay) // (b ⊙ decay) [B,H,wl,K]
	metal.Free(decay)
	kDecayT := metal.Transpose4(kDecay, 0, 1, 3, 2) // [B,H,K,wl]
	bDecayT := metal.Transpose4(bDecay, 0, 1, 3, 2) // [B,H,K,wl]
	metal.Free(kDecay, bDecay)
	kV := metal.Matmul(kDecayT, s.v) // (k⊙decay)ᵀ V [B,H,K,V]
	bV2 := metal.Matmul(bDecayT, v2) // (b⊙decay)ᵀ v2 [B,H,K,V]
	metal.Free(kDecayT, bDecayT)
	if freeV2 {
		metal.Free(v2)
	}
	contrib := metal.Add(kV, bV2)
	metal.Free(kV, bV2)

	var newState *metal.Array
	if prev != nil && prev.Valid() {
		// exp(c_C) decays each K-row of the carried state. c_C is [B,H,1,K];
		// reshape to [B,H,K,1] to scale the rows of S [B,H,K,V].
		expCLast := metal.Exp(cLast)                             // [B,H,1,K]
		rowScale := metal.Reshape(expCLast, s.bsz, s.h, s.dk, 1) // [B,H,K,1]
		metal.Free(expCLast)
		decayedPrev := metal.Mul(prev, rowScale) // [B,H,K,V] broadcast over V
		metal.Free(rowScale)
		newState = metal.Add(decayedPrev, contrib)
		metal.Free(decayedPrev, contrib)
	} else {
		newState = contrib
	}
	metal.Free(cLast)
	return out, newState
}

// --- shared mask / cast helpers (package-local mirror of deltanet/chunked.go's
// helpers; each mixer package owns its copy so neither imports the other) ------

// strictLowerMask returns the [wl,wl] strictly-lower keep-mask: 1.0 where i>j,
// 0.0 on and above the diagonal.
func strictLowerMask(wl int32) *metal.Array {
	idx := metal.Arange(0, float64(wl), 1, metal.DTypeFloat32)
	rows := metal.Reshape(idx, wl, 1) // i
	cols := metal.Reshape(idx, 1, wl) // j
	metal.Free(idx)
	cond := metal.Greater(rows, cols) // i > j
	metal.Free(rows, cols)
	keep := metal.AsType(cond, metal.DTypeFloat32)
	metal.Free(cond)
	return keep
}

// inclusiveLowerMask returns the [wl,wl] causal keep-mask: 1.0 where i≥j, 0.0
// strictly above.
func inclusiveLowerMask(wl int32) *metal.Array {
	idx := metal.Arange(0, float64(wl), 1, metal.DTypeFloat32)
	rows := metal.Reshape(idx, wl, 1)
	cols := metal.Reshape(idx, 1, wl)
	metal.Free(idx)
	rowsPlus := metal.AddScalar(rows, 1) // i+1 > j ⇔ i ≥ j
	metal.Free(rows)
	cond := metal.Greater(rowsPlus, cols)
	metal.Free(rowsPlus, cols)
	keep := metal.AsType(cond, metal.DTypeFloat32)
	metal.Free(cond)
	return keep
}

// eyeMatrix returns the [wl,wl] identity in the given dtype.
func eyeMatrix(wl int32, dt metal.DType) *metal.Array {
	idx := metal.Arange(0, float64(wl), 1, metal.DTypeFloat32)
	rows := metal.Reshape(idx, wl, 1)
	cols := metal.Reshape(idx, 1, wl)
	metal.Free(idx)
	cond := metal.Equal(rows, cols) // i == j
	metal.Free(rows, cols)
	eye := metal.AsType(cond, dt)
	metal.Free(cond)
	return eye
}

// mulMaskBroadcast multiplies a [B,H,wl,wl] tensor by a [wl,wl] mask, broadcast
// over the batch and head axes.
func mulMaskBroadcast(x, mask *metal.Array) *metal.Array {
	wl := int32(mask.Dim(0))
	maskB := metal.Reshape(mask, 1, 1, wl, wl)
	out := metal.Mul(x, maskB)
	metal.Free(maskB)
	return out
}

// asFloat32 returns a as float32 (LAPACK has no bf16 path), with cast=false when
// a already is float32 so the caller does not double-free the original.
func asFloat32(a *metal.Array) (out *metal.Array, cast bool) {
	if a.Dtype() == metal.DTypeFloat32 {
		return a, false
	}
	return metal.AsType(a, metal.DTypeFloat32), true
}

// castBack casts the float32 solve result back to the working dtype, returning
// it unchanged when the target already is float32.
func castBack(a *metal.Array, dt metal.DType) *metal.Array {
	if dt == metal.DTypeFloat32 {
		return a
	}
	out := metal.AsType(a, dt)
	metal.Free(a)
	return out
}

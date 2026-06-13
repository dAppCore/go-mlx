// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gsa

import metal "dappco.re/go/mlx/pkg/metal"

// kernels.go holds the GSA recurrence and its gate helpers as pure functions
// over metal.Array, so the unit test can pin the math on a fixed input + an
// explicit initial slot memory without loading model weights. Everything
// composes from existing ops (ops.go / slice.go); no new Metal kernel. The
// recurrence is the naive sequential reference form (a Go loop over timesteps
// issuing per-step tensor ops) — correct and testable; a chunked/parallel
// kernel is a later optimisation, not part of this scheme's contract.

// gateDecay computes g = logsigmoid(f) / norm. logsigmoid(f) = -softplus(-f) =
// f - softplus(f); here we use the numerically-safe identity logsigmoid(f) =
// -log(1+exp(-f)) realised as log(sigmoid(f)). norm <= 0 is treated as 1.
//
//	g := gateDecay(f, 16.0)
func gateDecay(f *metal.Array, norm float32) *metal.Array {
	sig := metal.Sigmoid(f)
	ls := metal.Log(sig) // log(sigmoid(f)) = logsigmoid(f) ∈ (-inf, 0]
	metal.Free(sig)
	if norm > 0 && norm != 1 {
		scaled := metal.MulScalar(ls, 1.0/norm)
		metal.Free(ls)
		return scaled
	}
	return ls
}

// slotWrite computes the slot-write weight s = 1 - exp(g). Since g <= 0,
// exp(g) ∈ (0,1] and s ∈ [0,1) — the complement of the forget gate.
func slotWrite(g *metal.Array) *metal.Array {
	eg := metal.Exp(g)
	one := metal.FromValue(float32(1))
	s := metal.Subtract(one, eg)
	metal.Free(eg, one)
	return s
}

// recurrence runs the gated slot recurrence over the sequence. Shapes:
//
//	q, k : [B,H,L,HeadK]      v : [B,H,L,HeadV]
//	g, s : [B,H,L,Slots]      (decay, slot-write weight)
//	sk0  : [B,H,HeadK,Slots]  sv0 : [B,H,Slots,HeadV]   (initial slot memory)
//
// Per step i:
//
//	Sk = Sk ⊙ exp(g_i)[over HeadK] + (k_iᵀ · s_i)      → [B,H,HeadK,Slots]
//	Sv = Sv ⊙ exp(g_i)[over HeadV] + (s_iᵀ · v_i)      → [B,H,Slots,HeadV]
//	a  = softmax_slots(q_i · Sk)                        → [B,H,1,Slots]
//	o_i= a · Sv                                         → [B,H,1,HeadV]
//
// Returns the stacked outputs [B,H,L,HeadV] and the final (Sk, Sv) so the
// caller can persist them once the recurrent holder exists.
func recurrence(q, k, v, g, s, sk0, sv0 *metal.Array) (out, skN, svN *metal.Array) {
	B := int32(q.Dim(0))
	H := int32(q.Dim(1))
	L := int32(q.Dim(2))
	headK := int32(q.Dim(3))
	slots := int32(g.Dim(3))
	headV := int32(v.Dim(3))

	sk := sk0.Clone()
	sv := sv0.Clone()

	steps := make([]*metal.Array, 0, L)
	for i := int32(0); i < L; i++ {
		qi := stepSlice(q, i, headK)  // [B,H,1,HeadK]
		ki := stepSlice(k, i, headK)  // [B,H,1,HeadK]
		vi := stepSlice(v, i, headV)  // [B,H,1,HeadV]
		gi := stepSlice(g, i, slots)  // [B,H,1,Slots]
		si := stepSlice(s, i, slots)  // [B,H,1,Slots]

		egi := metal.Exp(gi) // decay ∈ (0,1], [B,H,1,Slots]
		metal.Free(gi)

		// Sk update: decay broadcasts over HeadK (axis 2), write = k_iᵀ · s_i.
		kiT := metal.Transpose4(ki, 0, 1, 3, 2) // [B,H,HeadK,1]
		metal.Free(ki)
		kWrite := metal.Matmul(kiT, si) // [B,H,HeadK,Slots]
		metal.Free(kiT)
		skDecayed := metal.Mul(sk, egi) // egi [B,H,1,Slots] broadcasts over HeadK
		metal.Free(sk)
		sk = metal.Add(skDecayed, kWrite)
		metal.Free(skDecayed, kWrite)

		// Sv update: decay broadcasts over HeadV (axis 3); reshape egi to
		// [B,H,Slots,1] so the Slots axis aligns with Sv's axis 2.
		egiV := metal.Reshape(egi, B, H, slots, 1) // [B,H,Slots,1]
		siT := metal.Transpose4(si, 0, 1, 3, 2)    // [B,H,Slots,1]
		metal.Free(si)
		vWrite := metal.Matmul(siT, vi) // [B,H,Slots,HeadV]
		metal.Free(siT, vi)
		svDecayed := metal.Mul(sv, egiV) // egiV broadcasts over HeadV
		metal.Free(sv, egi, egiV)
		sv = metal.Add(svDecayed, vWrite)
		metal.Free(svDecayed, vWrite)

		// Read: a = softmax_slots(q_i · Sk); o_i = a · Sv.
		scores := metal.Matmul(qi, sk) // [B,H,1,Slots]
		metal.Free(qi)
		a := metal.Softmax(scores) // softmax over the last axis (slots)
		metal.Free(scores)
		oi := metal.Matmul(a, sv) // [B,H,1,HeadV]
		metal.Free(a)
		steps = append(steps, oi)
	}

	out = metal.Concatenate(steps, 2) // [B,H,L,HeadV]
	metal.Free(steps...)
	return out, sk, sv
}

// stepSlice extracts timestep i along the sequence axis (axis 2) of a
// [B,H,L,dim] tensor, yielding a [B,H,1,dim] slice for the per-step update.
func stepSlice(a *metal.Array, i, dim int32) *metal.Array {
	B := int32(a.Dim(0))
	H := int32(a.Dim(1))
	return metal.Slice4(a, 0, 0, i, 0, B, H, i+1, dim)
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

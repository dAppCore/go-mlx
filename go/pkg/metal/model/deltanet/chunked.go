// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

// chunked.go adds the chunked-PARALLEL delta-rule form alongside the exact
// sequential recurrence in deltanet.go. The package comment flags the chunked
// form as the prefill-speed follow-up: it replaces the O(L) host-side token
// loop with a single batched triangular solve per ≤chunkWidth window, so the
// work is dense matmuls (and one LAPACK solve) rather than L serial steps.
//
// Within a chunk of length C, with carried state S₀, L2-normalised keys K
// [C,Dk], values V [C,Dv], scaled queries Q̃ = scale·Q, and per-token write
// strengths β [C], the delta rule's sequentially-dependent writes collapse to
// the WY-style linear system (Yang et al. 2024, "Parallelizing Linear
// Transformers with the Delta Rule"):
//
//	define pseudo-values U s.t.  S_t = S₀ + Σ_{i≤t} k_i u_iᵀ
//	U = β(V − K S₀ − A U)   with  A = strict_lower(K Kᵀ)   (A_{ti}=k_t·k_i, i<t)
//	⇒ (I + diag(β)A) U = diag(β)(V − K S₀)                  (unit lower-tri solve)
//	O   = Q̃ S₀ + tril_incl(Q̃ Kᵀ) U                          (chunk output)
//	S_C = S₀ + Kᵀ U                                          (advanced state)
//
// (I + diag(β)A) is unit lower-triangular and square [C,C]; the right-hand side
// is [C,Dv]. metal.Solve (batched LAPACK LU on the CPU stream) inverts the whole
// [B,H] stack in one call, so no per-token loop and no new cgo are needed. The
// solve is exact, so the chunked output equals the sequential output to machine
// precision; deltanet_test.go's chunked-vs-sequential test proves it.
package deltanet

import (
	metal "dappco.re/go/mlx/pkg/metal"
)

// chunkWidth is the window the parallel form solves at once. Longer chunks are
// split into ≤chunkWidth windows that thread the recurrent state, bounding the
// per-window triangular system to chunkWidth×chunkWidth (the LAPACK solve cost
// grows with the cube of this, so a fixed width keeps it predictable) and
// matching FLA's chunk strategy. 64 is FLA's default chunk size.
const chunkWidth = 64

// computeParallel runs the chunked-parallel WY form of the delta rule, windowing
// the chunk into ≤chunkWidth segments that thread the recurrent state. It
// returns the per-head output [B,H,L,Dv] and the advanced state [B,H,Dk,Dv],
// identical to the sequential recurrence to machine precision. A nil result is
// surfaced if the LAPACK solve fails (the caller then has the sequential path).
//
//	out, newS := in.computeParallel() // dense matmuls + one batched solve per window
func (in *kernelInput) computeParallel() (*metal.Array, *metal.Array) {
	kNorm := l2NormalizeKeys(in.k, in.normEps) // [B,H,L,Dk] (delta rule needs unit keys)
	scaledQ := metal.MulScalar(in.q, in.scale) // Q̃ = scale·Q

	state := in.prev // carried across windows; nil ⇒ zero on the first
	outs := make([]*metal.Array, 0, (in.l+chunkWidth-1)/chunkWidth)
	for start := int32(0); start < in.l; start += chunkWidth {
		end := start + chunkWidth
		if end > in.l {
			end = in.l
		}
		wl := end - start
		kW := metal.Slice4(kNorm, 0, 0, start, 0, in.b, in.h, end, in.headDim)
		vW := metal.Slice4(in.v, 0, 0, start, 0, in.b, in.h, end, in.headDim)
		qW := metal.Slice4(scaledQ, 0, 0, start, 0, in.b, in.h, end, in.headDim)
		betaW := metal.Slice4(in.beta, 0, 0, start, 0, in.b, in.h, end, 1)

		outW, newState := in.solveWindow(qW, kW, vW, betaW, state, wl)
		metal.Free(kW, vW, qW, betaW)
		if outW == nil {
			// Solve failed: free everything we own and surface nil.
			if state != in.prev {
				metal.Free(state)
			}
			metal.Free(outs...)
			metal.Free(kNorm, scaledQ)
			return nil, nil
		}
		outs = append(outs, outW)

		if state != in.prev {
			metal.Free(state)
		}
		state = newState
	}

	out := metal.Concatenate(outs, 2) // [B,H,L,Dv]
	metal.Free(outs...)
	metal.Free(kNorm, scaledQ)
	return out, state
}

// solveWindow runs the WY solve for one ≤chunkWidth window. kn are the window's
// L2-normalised keys [B,H,wl,Dk]; v the values [B,H,wl,Dv]; qScaled the
// scale-folded queries [B,H,wl,Dk]; beta the write strengths [B,H,wl,1]; prev
// the state carried in (or nil). Returns the window output [B,H,wl,Dv] and the
// advanced state [B,H,Dk,Dv], or (nil,nil) if the batched solve fails.
func (in *kernelInput) solveWindow(qScaled, kn, v, beta, prev *metal.Array, wl int32) (*metal.Array, *metal.Array) {
	knT := metal.Transpose4(kn, 0, 1, 3, 2) // [B,H,Dk,wl]

	// A = strict_lower(K Kᵀ): A_{ti} = k_t·k_i for i<t, else 0. [B,H,wl,wl].
	kk := metal.Matmul(kn, knT) // [B,H,wl,wl] full Gram
	strict := strictLowerMask(wl)
	aStrict := mulMaskBroadcast(kk, strict) // zero on/above the diagonal
	metal.Free(kk, strict)

	// M = I + diag(β)·A. diag(β)·A scales row t by β_t; adding I makes it unit
	// lower-triangular. [B,H,wl,wl].
	betaA := metal.Mul(aStrict, beta) // β broadcast over the last (column) axis scales each row
	metal.Free(aStrict)
	eye := eyeMatrix(wl, kn.Dtype())
	m := addMaskBroadcast(betaA, eye) // M = I + diag(β)A
	metal.Free(betaA, eye)

	// RHS R = diag(β)(V − K S₀). [B,H,wl,Dv].
	rhs := v
	freeRHS := false
	if prev != nil && prev.Valid() {
		kS0 := metal.Matmul(kn, prev) // K S₀ [B,H,wl,Dv]
		rhs = metal.Subtract(v, kS0)  // V − K S₀
		metal.Free(kS0)
		freeRHS = true
	}
	betaRHS := metal.Mul(rhs, beta) // diag(β)·(…); β broadcast over Dv
	if freeRHS {
		metal.Free(rhs)
	}

	// Solve M U = R for the whole [B,H] batch in one LAPACK call. LAPACK needs
	// float32/float64, so cast a bf16 window in and the result back out. asFloat32
	// returns the input unchanged when it is already float32, so free the original
	// only when a distinct cast array was produced.
	mF, mCast := asFloat32(m)
	rF, rCast := asFloat32(betaRHS)
	uF, err := metal.Solve(mF, rF) // U [B,H,wl,Dv]
	if mCast {
		metal.Free(mF)
	}
	if rCast {
		metal.Free(rF)
	}
	metal.Free(m, betaRHS)
	if err != nil || uF == nil {
		metal.Free(knT)
		return nil, nil
	}
	u := castBack(uF, kn.Dtype()) // back to the working dtype

	// O = Q̃ S₀ + tril_incl(Q̃ Kᵀ) U. [B,H,wl,Dv].
	qk := metal.Matmul(qScaled, knT) // Q̃ Kᵀ [B,H,wl,wl]
	keep := inclusiveLowerMask(wl)
	qkMasked := mulMaskBroadcast(qk, keep) // keep i≤t
	metal.Free(qk, keep)
	out := metal.Matmul(qkMasked, u) // [B,H,wl,Dv]
	metal.Free(qkMasked)
	if prev != nil && prev.Valid() {
		qS0 := metal.Matmul(qScaled, prev) // Q̃ S₀
		summed := metal.Add(out, qS0)
		metal.Free(out, qS0)
		out = summed
	}

	// S_C = S₀ + Kᵀ U. [B,H,Dk,Dv].
	ktu := metal.Matmul(knT, u) // Kᵀ U
	metal.Free(knT, u)
	var newState *metal.Array
	if prev != nil && prev.Valid() {
		newState = metal.Add(prev, ktu)
		metal.Free(ktu)
	} else {
		newState = ktu
	}
	return out, newState
}

// l2NormalizeKeys L2-normalises the keys over the head dimension, mirroring the
// sequential path's flakernel.L2NormalizeLastAxis (the delta rule's stability
// depends on unit-norm keys).
func l2NormalizeKeys(x *metal.Array, eps float32) *metal.Array {
	sq := metal.Square(x)
	sumSq := metal.Sum(sq, len(x.Shape())-1, true)
	metal.Free(sq)
	denomSq := metal.AddScalar(sumSq, eps)
	metal.Free(sumSq)
	inv := metal.Rsqrt(denomSq)
	metal.Free(denomSq)
	out := metal.Mul(x, inv)
	metal.Free(inv)
	return out
}

// strictLowerMask returns the [wl,wl] strictly-lower keep-mask: 1.0 where i>j
// (below the diagonal), 0.0 on and above it. Built from a row>col comparison so
// it needs no host loop.
//
//	m := strictLowerMask(wl) // multiply K Kᵀ to keep only A_{ti}, i<t
func strictLowerMask(wl int32) *metal.Array {
	idx := metal.Arange(0, float64(wl), 1, metal.DTypeFloat32)
	rows := metal.Reshape(idx, wl, 1) // [wl,1] = i
	cols := metal.Reshape(idx, 1, wl) // [1,wl] = j
	metal.Free(idx)
	cond := metal.Greater(rows, cols) // i > j (strictly below the diagonal)
	metal.Free(rows, cols)
	keep := metal.AsType(cond, metal.DTypeFloat32)
	metal.Free(cond)
	return keep
}

// inclusiveLowerMask returns the [wl,wl] causal keep-mask: 1.0 where i≥j (on and
// below the diagonal), 0.0 strictly above. This is the i≤t output mask for
// tril_incl(Q̃ Kᵀ).
//
//	m := inclusiveLowerMask(wl) // keep contributions from positions i≤t
func inclusiveLowerMask(wl int32) *metal.Array {
	idx := metal.Arange(0, float64(wl), 1, metal.DTypeFloat32)
	rows := metal.Reshape(idx, wl, 1) // [wl,1] = i
	cols := metal.Reshape(idx, 1, wl) // [1,wl] = j
	metal.Free(idx)
	rowsPlus := metal.AddScalar(rows, 1) // i+1 (so i+1 > j ⇔ i ≥ j)
	metal.Free(rows)
	cond := metal.Greater(rowsPlus, cols)
	metal.Free(rowsPlus, cols)
	keep := metal.AsType(cond, metal.DTypeFloat32)
	metal.Free(cond)
	return keep
}

// eyeMatrix returns the [wl,wl] identity in the given dtype, built from a
// row==col comparison (no host loop).
//
//	I := eyeMatrix(wl, dt) // unit diagonal added to diag(β)A
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

// addMaskBroadcast adds a [wl,wl] matrix to a [B,H,wl,wl] tensor, broadcast over
// the batch and head axes (used to add the identity to diag(β)A).
func addMaskBroadcast(x, mask *metal.Array) *metal.Array {
	wl := int32(mask.Dim(0))
	maskB := metal.Reshape(mask, 1, 1, wl, wl)
	out := metal.Add(x, maskB)
	metal.Free(maskB)
	return out
}

// asFloat32 returns a as float32 (LAPACK has no bf16 path). When a is already
// float32 it is returned as-is with cast=false so the caller does not double-free
// the original; otherwise a fresh cast array is returned with cast=true.
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

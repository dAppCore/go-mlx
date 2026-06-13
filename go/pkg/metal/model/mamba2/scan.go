// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package mamba2

import (
	core "dappco.re/go"
	metal "dappco.re/go/mlx/pkg/metal"
)

// scan.go is the Mamba-2 selective-scan kernel — the state-space duality (SSD)
// recurrence, expressed over MLX metal.Array ops as an exact sequential scan.
// It is the family-agnostic core: a model's in/out projections (the dt/A/B/C/D
// machinery) are load-time detail, but the scan itself — decay the state, add
// the outer-product input, contract with C — is what every Mamba-2 build runs,
// so it lives here as a pure function with no engine or model coupling.
//
// The reference is github.com/state-spaces/mamba (ssd_minimal) and
// github.com/fla-org/flash-linear-attention (fla/ops/mamba2): A is one scalar
// per head, so the discrete transition exp(dt·A) is a scalar decay, and the
// per-step SSM state is [head_dim, d_state] (the (P,N) layout the einsum
// "...hpn" carries). The chunked/parallel form is a perf optimisation over
// this exact recurrence — see ScanInput's doc for the new-kernel note.

// ScanInput is one chunk of post-projection SSD inputs for a single layer. The
// caller (a model's load path) splits the hidden state into heads and runs its
// dt/A/B/C/D projections; this struct is the canonical hand-off the scan
// kernel consumes, matching the mamba_ssm / fla tensor convention so the math
// stays verifiable against the reference.
//
//	y, next := mamba2.SSDScan(mamba2.ScanInput{X: x, Dt: dt, A: a, B: b, C: c, D: d}, prior)
type ScanInput struct {
	X  *metal.Array // [B, L, H, P] hidden states, split into H heads of head_dim P
	Dt *metal.Array // [B, L, H]    discretisation step Δ (already softplus-activated, ≥0)
	A  *metal.Array // [H]          per-head state-decay scalar (negative; dA = exp(Δ·A))
	B  *metal.Array // [B, L, H, N] input (selection) projection, N = d_state
	C  *metal.Array // [B, L, H, N] output (readout) projection, N = d_state
	D  *metal.Array // [H]          per-head skip-connection scalar; nil = no skip
}

// SSDScan runs the Mamba-2 selective scan for one chunk and returns the mixed
// output y [B, L, H, P] and the advanced SSM state [B, H, P, N]. prior is the
// carried state from the previous chunk (decode) or nil for a fresh sequence
// (prefill), in which case the scan starts from a zero state.
//
// The recurrence, per timestep t (mamba_ssm ssd_minimal, scalar A):
//
//	dA_t     = exp(Δ_t · A)                         // [B,H,1,1] scalar decay
//	state_t  = state_{t-1} · dA_t  +  x_t ⊗ (Δ_t·B_t)   // [B,H,P,N]
//	y_t      = state_t @ C_t  +  D ⊙ x_t            // [B,H,P]
//
// where x_t is [B,H,P], B_t/C_t are [B,H,N], and the outer product x_t ⊗ (Δ_t·B_t)
// is [B,H,P,N]. This is the exact sequential reference; it is O(L) graph nodes,
// which is correct and the right shape for the kernel test and for decode
// (L=1). A length-parallel chunked scan (the segsum / associative-scan form)
// would cut the L serial dependency but needs a custom Metal kernel — flagged,
// not built here.
func SSDScan(in ScanInput, prior *metal.Array) (*metal.Array, *metal.Array) {
	B := int32(in.X.Dim(0))
	L := int32(in.X.Dim(1))
	H := int32(in.X.Dim(2))
	P := int32(in.X.Dim(3))
	N := int32(in.B.Dim(3))
	dtype := in.X.Dtype()

	// dA over the whole chunk in one shot: Δ [B,L,H] · A [H] (broadcast over the
	// H axis), then exp. Per-step decay is a slice of this — cheaper than H
	// scalar exps per timestep.
	dtA := metal.Mul(in.Dt, in.A) // [B,L,H]
	dA := metal.Exp(dtA)          // [B,L,H]
	metal.Free(dtA)

	// State [B,H,P,N]; zero on prefill, the carried state on decode.
	var state *metal.Array
	if prior != nil && prior.Valid() {
		state = prior.Clone()
	} else {
		state = metal.Zeros([]int32{B, H, P, N}, dtype)
	}

	outputs := make([]*metal.Array, 0, L)
	for t := int32(0); t < L; t++ {
		// Per-timestep views, squeezed to drop the length axis.
		xtRaw := metal.SliceAxis(in.X, 1, t, t+1) // [B,1,H,P]
		xt := metal.Reshape(xtRaw, B, H, P, 1)    // [B,H,P,1] (column for the outer product)
		metal.Free(xtRaw)

		btRaw := metal.SliceAxis(in.B, 1, t, t+1) // [B,1,H,N]
		bt := metal.Reshape(btRaw, B, H, 1, N)    // [B,H,1,N] (row for the outer product)
		metal.Free(btRaw)

		ctRaw := metal.SliceAxis(in.C, 1, t, t+1) // [B,1,H,N]
		ct := metal.Reshape(ctRaw, B, H, N, 1)    // [B,H,N,1] (for the state @ C contraction)
		metal.Free(ctRaw)

		dtRaw := metal.SliceAxis(in.Dt, 1, t, t+1) // [B,1,H]
		dtt := metal.Reshape(dtRaw, B, H, 1, 1)    // [B,H,1,1]
		metal.Free(dtRaw)

		daRaw := metal.SliceAxis(dA, 1, t, t+1) // [B,1,H]
		dat := metal.Reshape(daRaw, B, H, 1, 1) // [B,H,1,1] scalar decay
		metal.Free(daRaw)

		// state = state·dA + (Δ·B) ⊗ x
		decayed := metal.Mul(state, dat) // [B,H,P,N] broadcast scalar
		metal.Free(state)
		metal.Free(dat)

		dtB := metal.Mul(bt, dtt) // [B,H,1,N] = B_t · Δ_t
		metal.Free(bt)
		metal.Free(dtt)
		update := metal.Mul(xt, dtB) // [B,H,P,N] = x_t [B,H,P,1] ⊗ (Δ·B)_t [B,H,1,N]
		metal.Free(dtB)

		state = metal.Add(decayed, update)
		metal.Free(decayed)
		metal.Free(update)

		// y_t = state @ C_t  →  [B,H,P,N] @ [B,H,N,1] = [B,H,P,1]
		yMat := metal.Matmul(state, ct) // [B,H,P,1]
		metal.Free(ct)
		yt := metal.Reshape(yMat, B, H, P) // [B,H,P]
		metal.Free(yMat)

		// D skip: y_t += D ⊙ x_t. D is per-head [H] (one scalar per head), so it
		// reshapes to [1,H,1] and broadcasts over the batch and head_dim axes of
		// x_t [B,H,P] — NOT [1,1,P], which would (wrongly) require P elements.
		if in.D != nil && in.D.Valid() {
			xtP := metal.Reshape(xt, B, H, P) // [B,H,P]
			dHead := metal.Reshape(in.D, 1, H, 1)
			dx := metal.Mul(xtP, dHead) // [B,H,P] broadcast over B and P
			metal.Free(xtP)
			metal.Free(dHead)
			ytD := metal.Add(yt, dx)
			metal.Free(yt)
			metal.Free(dx)
			yt = ytD
		}
		metal.Free(xt)

		// Re-introduce the length axis so the chunk concatenates to [B,L,H,P].
		ytL := metal.Reshape(yt, B, 1, H, P)
		metal.Free(yt)
		outputs = append(outputs, ytL)
	}
	metal.Free(dA)

	var y *metal.Array
	if L == 1 {
		y = outputs[0]
	} else {
		y = metal.Concatenate(outputs, 1) // [B,L,H,P]
	}

	// Eval before releasing the per-timestep slices: the concat result is a lazy
	// graph node over the (unevaluated) reshape nodes in outputs, so freeing them
	// first would drop the data the concat still needs. The L==1 path returns the
	// single slice directly, so there is nothing extra to release there.
	if err := metal.Eval(y, state); err != nil {
		core.Error("mamba2: SSD scan eval failed", "error", err, "B", B, "L", L, "H", H, "P", P, "N", N)
	}
	if L > 1 {
		metal.Free(outputs...)
	}
	return y, state
}

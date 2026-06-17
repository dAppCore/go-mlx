// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package qwen3

import (
	metal "dappco.re/go/mlx/pkg/metal"
)

// gated_delta.go resolves the Qwen3.5/3.6 GatedDeltaNet linear-attention layer's
// scalar inputs — the per-token, per-(value-)head decay α and write-strength β —
// from its raw projections, in the layout the deltanet.GatedDeltaRuleChunkSequential
// kernel consumes. This is the Mamba-style discretisation that bridges the
// linear_attn.{in_proj_a, in_proj_b, A_log, dt_bias} weights to the gated delta
// rule; the conv front, grouped Q/K/V, gated-norm and out-projection that wrap the
// kernel are the rest of the mixer (a following slice).

// resolveDecay forms the per-token, per-head decay α_t for the gated delta rule:
//
//	dt_t = softplus(a_t + dt_bias)        // the non-negative discretisation step
//	A    = −exp(A_log)                    // the stable Mamba reparametrisation (A < 0)
//	α_t  = exp(A · dt_t)  ∈ (0,1)
//
// a is the in_proj_a output [B,L,H] (one channel per value head H); aLog and
// dtBias are [H]. Returns α in the kernel layout [B,H,L,1]. α → 1 as A·dt → 0
// (no decay), α → 0 as the step grows (full forget) — exactly the (0,1] band the
// gated kernel's broadcast multiply expects.
func resolveDecay(a, aLog, dtBias *metal.Array, b, l, h int32) *metal.Array {
	dt := softplusBias(a, dtBias, b, l, h) // softplus(a + dt_bias)  [B,L,H]
	aNeg := negExp(aLog)                   // A = -exp(A_log)  [H]
	aRow := metal.Reshape(aNeg, 1, 1, h)   // broadcast A over [B,L]
	g := metal.Mul(dt, aRow)               // g = A·dt  [B,L,H]
	metal.Free(dt, aNeg, aRow)
	alpha := metal.Exp(g) // α = exp(g) ∈ (0,1)
	metal.Free(g)
	out := toHeadTime(alpha, b, l, h) // [B,L,H] → [B,H,L,1]
	metal.Free(alpha)
	return out
}

// resolveBeta forms the per-token write strength β_t = sigmoid(b_t). bProj is the
// in_proj_b output [B,L,H]; returns β in the kernel layout [B,H,L,1].
func resolveBeta(bProj *metal.Array, b, l, h int32) *metal.Array {
	beta := metal.Sigmoid(bProj)
	out := toHeadTime(beta, b, l, h)
	metal.Free(beta)
	return out
}

// toHeadTime reshapes a per-token, per-head scalar [B,L,H] to the kernel's
// [B,H,L,1] layout — transpose the L and H axes and carry the trailing scalar.
func toHeadTime(x *metal.Array, b, l, h int32) *metal.Array {
	r := metal.Reshape(x, b, l, h, 1)    // [B,L,H,1]
	t := metal.Transpose4(r, 0, 2, 1, 3) // [B,H,L,1]
	metal.Free(r)
	return t
}

// softplusBias adds the optional per-head dt-bias then applies softplus, the
// non-negative discretisation step — softplus(a + dt_bias), per head [B,L,H].
func softplusBias(a, dtBias *metal.Array, b, l, h int32) *metal.Array {
	d := metal.Reshape(a, b, l, h)
	if dtBias != nil && dtBias.Valid() {
		biasRow := metal.Reshape(dtBias, 1, 1, h)
		biased := metal.Add(d, biasRow)
		metal.Free(d, biasRow)
		d = biased
	}
	return softplus(d)
}

// softplus computes log(1 + exp(x)) element-wise via the existing exp/log ops —
// the same direct form mamba2 uses (dt magnitudes here are small positive steps).
func softplus(x *metal.Array) *metal.Array {
	e := metal.Exp(x)
	e1 := metal.AddScalar(e, 1.0)
	metal.Free(e)
	out := metal.Log(e1)
	metal.Free(e1)
	return out
}

// negExp returns −exp(x) — the A = −exp(A_log) reparametrisation that keeps the
// per-head decay pole strictly negative.
func negExp(x *metal.Array) *metal.Array {
	e := metal.Exp(x)
	out := metal.Negative(e)
	metal.Free(e)
	return out
}

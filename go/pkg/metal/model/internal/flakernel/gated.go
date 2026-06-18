// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package flakernel

import metal "dappco.re/go/mlx/pkg/metal"

// gated.go holds the tensor helpers the GATED linear-attention / SSM families
// (mamba2, the Qwen3.5/3.6 gated-delta mixer, and the gated families that follow)
// compose from: the softplus/−exp discretisation reparametrisations, the causal
// depthwise conv with its conv-state ring, and the Mamba-style gated RMSNorm. They
// are shared here so one proven implementation backs every gated mixer.

// Softplus computes log(1 + exp(x)) element-wise via the existing exp/log ops —
// the non-negative discretisation step the Mamba-style families need (dt
// magnitudes are small positive steps, so the direct form is numerically fine).
//
//	dt := flakernel.Softplus(metal.Add(a, dtBias)) // softplus(a + dt_bias)
func Softplus(x *metal.Array) *metal.Array {
	e := metal.Exp(x)
	e1 := metal.AddScalar(e, 1.0)
	metal.Free(e)
	out := metal.Log(e1)
	metal.Free(e1)
	return out
}

// NegExp returns −exp(x) — the A = −exp(A_log) reparametrisation that keeps a
// per-head decay pole strictly negative (a stable continuous-time pole).
//
//	A := flakernel.NegExp(aLog) // A = −exp(A_log)
func NegExp(x *metal.Array) *metal.Array {
	e := metal.Exp(x)
	out := metal.Negative(e)
	metal.Free(e)
	return out
}

// CausalDepthwiseConv applies the causal depthwise conv1d over the conv input
// xBC [B,L,convDim] and threads the conv-state ring. It returns the conv output
// [B,L,convDim] and the advanced ring [B, convDim, kernel-1] — the last
// (kernel-1) input columns, which seed the next chunk's left padding so a
// streamed decode is identical to a single-shot prefill.
//
// Prefill (L>1): the input is left-padded by priorRing (or zeros on the first
// chunk) so position t convolves over [t-K+1 … t]. Decode (L=1): the ring holds
// the previous K-1 inputs; the conv is the dot of [ring | x_t] with the per-
// channel kernel, and the ring shifts in x_t. convWeight is the PyTorch Conv1d
// layout [convDim, 1, kernel]; convBias [convDim] or nil.
func CausalDepthwiseConv(xBC, priorRing, convWeight, convBias *metal.Array, convDim, kernel, b, l int32) (*metal.Array, *metal.Array) {
	pad := kernel - 1

	// MLX Conv1d wants NLC [B, L, C]; build the left padding in NLC too.
	var leftPad *metal.Array
	if priorRing != nil && priorRing.Valid() {
		// priorRing [B, convDim, pad] → NLC [B, pad, convDim].
		leftPad = metal.Transpose4(metal.Reshape(priorRing, b, convDim, pad, 1), 0, 2, 1, 3)
		leftPad = metal.Reshape(leftPad, b, pad, convDim)
	} else {
		leftPad = metal.Zeros([]int32{b, pad, convDim}, xBC.Dtype())
	}
	padded := metal.Concatenate([]*metal.Array{leftPad, xBC}, 1) // [B, L+pad, convDim]
	metal.Free(leftPad)

	// Depthwise conv1d: groups == convDim. The checkpoint stores [out=convDim,
	// in/groups=1, kernel]; MLX wants [out_channels, kernel, in/groups], so
	// transpose the middle/last axes. We pre-padded causally → pass padding 0.
	convW := metal.Transpose4(metal.Reshape(convWeight, convDim, 1, kernel, 1), 0, 2, 1, 3)
	convW = metal.Reshape(convW, convDim, kernel, 1)
	conv := metal.Conv1d(padded, convW, 1, 0, 1, int(convDim)) // [B,L,convDim]
	metal.Free(convW)
	if convBias != nil && convBias.Valid() {
		biasRow := metal.Reshape(convBias, 1, 1, convDim)
		biased := metal.Add(conv, biasRow)
		metal.Free(conv, biasRow)
		conv = biased
	}

	// Advanced ring = the last `pad` columns of the padded input, channels-first.
	tail := metal.SliceAxis(padded, 1, l, l+pad) // [B, pad, convDim]
	metal.Free(padded)
	newRing := metal.Transpose4(metal.Reshape(tail, b, pad, convDim, 1), 0, 2, 1, 3)
	newRing = metal.Reshape(newRing, b, convDim, pad)
	metal.Free(tail)

	return conv, newRing
}

// GatedRMSNorm applies the Mamba-style gated RMSNorm RMSNorm(y)·SiLU(z) — the
// output gate that closes a gated-SSM/delta block before the out-projection. With
// no norm weight it falls back to the bare SiLU gate (y·SiLU(z)).
//
//	gated := flakernel.GatedRMSNorm(y, z, normWeight, eps)
func GatedRMSNorm(y, z, norm *metal.Array, eps float32) *metal.Array {
	var normed *metal.Array
	if norm != nil && norm.Valid() {
		normed = metal.RMSNorm(y, norm, eps)
	} else {
		normed = y.Clone()
	}
	return metal.SiLUGateMul(z, normed) // SiLU(z) · normed
}

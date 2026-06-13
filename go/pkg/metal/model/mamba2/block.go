// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package mamba2

import metal "dappco.re/go/mlx/pkg/metal"

// block.go is the Mamba-2 block-assembly half of the mixer: the projection,
// causal-conv, dt-activation, group-expansion and gated-norm helpers Forward
// composes around the SSD scan kernel (scan.go / chunk.go). These are the
// arch-specific glue; the scan itself stays family-agnostic.

// causalConv applies the causal depthwise conv1d over the conv input
// xBC [B,L,convDim] and threads the conv-state ring. It returns the conv output
// [B,L,convDim] and the advanced conv-state ring [B, convDim, convKernel-1] —
// the last (convKernel-1) input columns, which seed the next chunk's left
// padding so a streamed decode is identical to a single-shot prefill.
//
// Prefill (L>1): the input is left-padded by the prior ring (or zeros on the
// first chunk) so position t convolves over [t-K+1 … t]. Decode (L=1): the ring
// holds the previous K-1 inputs; the conv is the dot of [ring | x_t] with the
// per-channel kernel, and the ring shifts in x_t.
func (m *Mixer) causalConv(xBC, priorRing *metal.Array, B, L int32) (*metal.Array, *metal.Array) {
	cfg := m.W.cfg
	convDim := cfg.convDim()
	K := cfg.ConvKernel
	pad := K - 1

	// Channels-first [B, convDim, L] for the conv-state ring + the per-channel
	// conv; MLX Conv1d wants NLC [B, L, C], so we keep xBC in NLC and build the
	// left padding in NLC too.
	var leftPad *metal.Array
	if priorRing != nil && priorRing.Valid() {
		// priorRing is [B, convDim, pad]; transpose to NLC [B, pad, convDim].
		leftPad = metal.Transpose4(metal.Reshape(priorRing, B, convDim, pad, 1), 0, 2, 1, 3)
		leftPad = metal.Reshape(leftPad, B, pad, convDim)
	} else {
		leftPad = metal.Zeros([]int32{B, pad, convDim}, xBC.Dtype())
	}
	padded := metal.Concatenate([]*metal.Array{leftPad, xBC}, 1) // [B, L+pad, convDim]
	metal.Free(leftPad)

	// Depthwise conv1d: groups == convDim. The checkpoint stores the weight in
	// PyTorch Conv1d layout [out=convDim, in/groups=1, kernel=K]; MLX wants
	// [out_channels, kernel, in_channels/groups] = [convDim, K, 1], so transpose
	// the middle/last axes. MLX pads internally with `padding`; we pre-padded
	// causally, so pass padding 0.
	convW := metal.Transpose4(metal.Reshape(m.W.ConvWeight, convDim, 1, K, 1), 0, 2, 1, 3)
	convW = metal.Reshape(convW, convDim, K, 1)
	conv := metal.Conv1d(padded, convW, 1, 0, 1, int(convDim)) // [B,L,convDim]
	metal.Free(convW)
	if m.W.ConvBias != nil && m.W.ConvBias.Valid() {
		biasRow := metal.Reshape(m.W.ConvBias, 1, 1, convDim)
		biased := metal.Add(conv, biasRow)
		metal.Free(conv, biasRow)
		conv = biased
	}

	// Advanced ring = the last `pad` columns of the padded input, channels-first
	// [B, convDim, pad].
	tail := metal.SliceAxis(padded, 1, L, L+pad) // [B, pad, convDim] (= last pad of L+pad)
	metal.Free(padded)
	newRing := metal.Transpose4(metal.Reshape(tail, B, pad, convDim, 1), 0, 2, 1, 3)
	newRing = metal.Reshape(newRing, B, convDim, pad)
	metal.Free(tail)

	return conv, newRing
}

// groupToHeads expands a grouped B/C projection [B,L,nGroups*N] to the per-head
// SSD layout [B,L,H,N] by repeating each group across the heads that share it
// (H/nGroups heads per group). nGroups==H is the per-head case (no repeat);
// nGroups==1 broadcasts one B/C to every head.
func (m *Mixer) groupToHeads(proj *metal.Array, B, L int32, cfg *Config) *metal.Array {
	N := cfg.StateDim
	G := cfg.NumGroups
	H := cfg.NumHeads
	grouped := metal.Reshape(proj, B, L, G, N) // [B,L,G,N]
	if G == H {
		return grouped
	}
	repeat := H / G
	// [B,L,G,1,N] → broadcast to [B,L,G,repeat,N] → [B,L,H,N].
	expanded := metal.Reshape(grouped, B, L, G, 1, N)
	metal.Free(grouped)
	bcast := metal.BroadcastTo(expanded, []int32{B, L, G, repeat, N})
	metal.Free(expanded)
	heads := metal.Reshape(bcast, B, L, H, N)
	metal.Free(bcast)
	return heads
}

// activateDt forms the per-head discretisation step dt [B,L,H] from the raw dt
// projection: add the optional dt-bias then softplus (log1p(exp·)), the
// non-negative step the SSD discretisation needs.
func (m *Mixer) activateDt(dt *metal.Array, B, L int32, cfg *Config) *metal.Array {
	d := metal.Reshape(dt, B, L, cfg.NumHeads)
	if m.W.DtBias != nil && m.W.DtBias.Valid() {
		biasRow := metal.Reshape(m.W.DtBias, 1, 1, cfg.NumHeads)
		biased := metal.Add(d, biasRow)
		metal.Free(d, biasRow)
		d = biased
	}
	return softplus(d)
}

// gatedNorm applies the Mamba-2 gated RMSNorm: RMSNorm(y) * SiLU(z), the
// output gate that closes the block before the out-projection. With no Norm
// weight it falls back to the bare SiLU gate (y * SiLU(z)).
func (m *Mixer) gatedNorm(y, z *metal.Array, cfg *Config) *metal.Array {
	var normed *metal.Array
	if m.W.Norm != nil && m.W.Norm.Valid() {
		normed = metal.RMSNorm(y, m.W.Norm, cfg.RMSNormEps)
	} else {
		normed = y.Clone()
	}
	return metal.SiLUGateMul(z, normed) // SiLU(z) * normed
}

// softplus computes log(1 + exp(x)) element-wise via the existing exp/log ops.
// At fixture scale the direct form is numerically fine; the dt magnitudes the
// scan sees are small positive steps.
func softplus(x *metal.Array) *metal.Array {
	e := metal.Exp(x)
	e1 := metal.AddScalar(e, 1.0)
	metal.Free(e)
	out := metal.Log(e1)
	metal.Free(e1)
	return out
}

// negExp returns -exp(x) — the Mamba-2 A = -exp(A_log) reparametrisation that
// keeps the per-head decay strictly negative (a stable continuous-time pole).
func negExp(x *metal.Array) *metal.Array {
	e := metal.Exp(x)
	out := metal.Negative(e)
	metal.Free(e)
	return out
}

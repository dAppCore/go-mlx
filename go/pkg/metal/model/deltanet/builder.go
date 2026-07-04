// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package deltanet

// builder.go registers DeltaNet's load-time builder on the engine's mixer-loader
// registry, so a model config that declares a layer_type of "deltanet" resolves
// end to end. The builder composes the four quant-ready projections plus the
// per-token write-strength projection, wrapped as the BetaFn the kernel
// consumes. Lives in this package alongside the scheme + compute; no
// model-package edit, no quant coupling (ctx.Linear arrives quant-ready).

import (
	core "dappco.re/go"
	metal "dappco.re/go/mlx/pkg/metal"
)

func init() { metal.RegisterMixerLoader("deltanet", buildMixer) }

// buildMixer composes a DeltaNet Mixer for one layer: q/k/v/o plus the b_proj
// write-strength projection, wrapped as a BetaFn that projects the hidden state
// and applies sigmoid (β ∈ (0,1)), reshaped to the per-head [B,H,L,1] the kernel
// expects.
//
//	metal.RegisterMixerLoader("deltanet", buildMixer) // resolves layer_type "deltanet"
func buildMixer(ctx metal.MixerBuildCtx) (metal.MixerCompute, error) {
	q := ctx.Linear("q_proj")
	k := ctx.Linear("k_proj")
	v := ctx.Linear("v_proj")
	o := ctx.Linear("o_proj")
	if q == nil || k == nil || v == nil || o == nil {
		return nil, core.E("deltanet.buildMixer", "missing q/k/v/o projection", nil)
	}
	betaProj := ctx.Linear("b_proj")
	if betaProj == nil {
		return nil, core.E("deltanet.buildMixer", "missing b_proj write-strength projection", nil)
	}
	numHeads, headDim := headGeometry(ctx.Cfg)
	if numHeads <= 0 || headDim <= 0 {
		return nil, core.E("deltanet.buildMixer", "config declares no head geometry", nil)
	}
	w := &Weights{
		QProj:    q,
		KProj:    k,
		VProj:    v,
		Output:   o,
		NumHeads: numHeads,
		HeadDim:  headDim,
	}
	return &Mixer{W: w, Beta: betaFn(betaProj, numHeads)}, nil
}

// headGeometry resolves (numHeads, headDim) from the neutral transformer dims,
// falling back to HiddenSize/NumAttentionHeads when headDim is not stated.
func headGeometry(cfg metal.TransformerConfig) (numHeads, headDim int) {
	numHeads = int(cfg.NumAttentionHeads)
	headDim = int(cfg.HeadDim)
	if headDim == 0 && numHeads > 0 && cfg.HiddenSize > 0 {
		headDim = int(cfg.HiddenSize) / numHeads
	}
	return numHeads, headDim
}

// betaFn builds the per-layer BetaFn from the write-strength projection: project
// the hidden state to one strength per head per token [B,L,H], sigmoid it into
// (0,1), and view it as the per-head [B,H,L,1] the kernel reads.
func betaFn(betaProj *metal.Linear, numHeads int) BetaFn {
	return func(x *metal.Array, b, l int32) *metal.Array {
		proj := betaProj.Forward(x) // [B,L,H]
		sig := metal.Sigmoid(proj)  // β ∈ (0,1)
		metal.Free(proj)
		view := metal.AsStrided(sig, []int32{b, int32(numHeads), l, 1},
			[]int64{int64(l * int32(numHeads)), int64(1), int64(int32(numHeads)), 1}, 0)
		metal.Free(sig)
		return view
	}
}

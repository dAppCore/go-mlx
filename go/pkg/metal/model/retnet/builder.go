// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package retnet

// builder.go registers RetNet's load-time builder on the engine's mixer-loader
// registry, so a model config that declares a layer_type of "retnet" resolves
// end to end: the load path looks up this builder by kind, hands it a neutral
// MixerBuildCtx (quant-ready Linear resolver + neutral dims), and the builder
// composes the RetNet Mixer. The builder lives in this package alongside the
// scheme + compute — a mixer is an engine feature, self-contained, with no
// model-package edit and no quant coupling (the ctx.Linear values arrive
// quant-ready; the builder never sees a quant format).

import (
	"math"

	core "dappco.re/go"
	metal "dappco.re/go/mlx/pkg/metal"
)

func init() { metal.RegisterMixerLoader("retnet", buildMixer) }

// buildMixer composes a RetNet Mixer for one layer from the neutral build
// context: the four quant-ready projections plus the per-head decay schedule.
// RetNet's γ_h is the architectural schedule 1 − 2^(−5−h) (Sun et al. 2023), not
// a learned tensor, so the builder derives the per-head ln(γ) the kernel
// consumes — there is no gate weight to load.
//
//	metal.RegisterMixerLoader("retnet", buildMixer) // resolves layer_type "retnet"
func buildMixer(ctx metal.MixerBuildCtx) (metal.MixerCompute, error) {
	q := ctx.Linear("q_proj")
	k := ctx.Linear("k_proj")
	v := ctx.Linear("v_proj")
	o := ctx.Linear("o_proj")
	if q == nil || k == nil || v == nil || o == nil {
		return nil, core.E("retnet.buildMixer", "missing q/k/v/o projection", nil)
	}
	numHeads, headDim := headGeometry(ctx.Cfg)
	if numHeads <= 0 || headDim <= 0 {
		return nil, core.E("retnet.buildMixer", "config declares no head geometry", nil)
	}
	w := &Weights{
		QProj:    q,
		KProj:    k,
		VProj:    v,
		Output:   o,
		NumHeads: numHeads,
		HeadDim:  headDim,
		DecayLn:  decayLnSchedule(numHeads),
	}
	return &Mixer{W: w}, nil
}

// headGeometry resolves (numHeads, headDim) from the neutral transformer dims.
// headDim falls back to HiddenSize/NumAttentionHeads when the config does not
// state it directly (the same rule the attention loaders apply).
func headGeometry(cfg metal.TransformerConfig) (numHeads, headDim int) {
	numHeads = int(cfg.NumAttentionHeads)
	headDim = int(cfg.HeadDim)
	if headDim == 0 && numHeads > 0 && cfg.HiddenSize > 0 {
		headDim = int(cfg.HiddenSize) / numHeads
	}
	return numHeads, headDim
}

// decayLnSchedule returns the per-head ln(γ_h) for RetNet's multiscale schedule
// γ_h = 1 − 2^(−5−h). The log is taken once at load so the kernel builds its
// decay mask without a per-call Pow.
func decayLnSchedule(numHeads int) []float32 {
	out := make([]float32, numHeads)
	for h := 0; h < numHeads; h++ {
		gamma := 1.0 - math.Exp2(-5.0-float64(h))
		out[h] = float32(math.Log(gamma))
	}
	return out
}

// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package mla

import (
	"math"

	core "dappco.re/go"
	metal "dappco.re/go/mlx/pkg/metal"
)

// loader.go registers MLA's load-time builder on the engine's mixer-loader
// registry. The builder lives HERE, beside MLA's scheme + compute — not in any
// model package: a mixer is built the same way whatever model declares it, so a
// config layer_type of "mla" resolves through metal.RegisterMixerLoader with no
// gemma4 (or other model) import. The builder takes quant-READY *metal.Linear
// from ctx.Linear (the quant factory is already applied — the mixer never sees a
// quant format) and reads neutral dims from ctx.Cfg.
//
// MLA projection names follow the DeepSeek-V2 layout: kv_a_proj_with_mqa
// (down → KV latent), kv_b_proj (up → K/V), q_a_proj / q_b_proj (down/up query),
// o_proj. Head count comes from the config; per-head dim is inferred from the
// kv_b_proj output (the concatenated K+V up-projection, 2*heads*headDim), with
// the config head dim as fallback. A missing required projection is a loud build
// error, never a silent zero-weight mixer.

func init() {
	metal.RegisterMixerLoader(MixerKind, buildMLA)
}

func buildMLA(ctx metal.MixerBuildCtx) (metal.MixerCompute, error) {
	const op = "mla.build"
	wDKV := ctx.Linear("kv_a_proj_with_mqa")
	wUK := ctx.Linear("kv_b_proj")
	wDQ := ctx.Linear("q_a_proj")
	wUQ := ctx.Linear("q_b_proj")
	oProj := ctx.Linear("o_proj")
	for name, l := range map[string]*metal.Linear{
		"kv_a_proj_with_mqa": wDKV, "kv_b_proj": wUK, "q_a_proj": wDQ, "q_b_proj": wUQ, "o_proj": oProj,
	} {
		if l == nil {
			return nil, core.E(op, core.Sprintf("missing projection %q", name), nil)
		}
	}

	heads := headCount(ctx.Cfg)
	headDim := ctx.Cfg.HeadDim
	if up := linearOutDim(wUK); up > 0 && heads > 0 && up%(2*heads) == 0 {
		headDim = up / (2 * heads)
	}
	if headDim <= 0 {
		return nil, core.E(op, "cannot determine MLA head dim from kv_b_proj or config", nil)
	}

	return &Mixer{
		WDKV: wDKV, WUK: wUK, WUV: wUK, WDQ: wDQ, WUQ: wUQ, OProj: oProj,
		NumHeads: heads, HeadDim: headDim, Scale: attnScale(headDim),
	}, nil
}

// headCount resolves the attention head count from the neutral config, falling
// back to 1 so a malformed config fails the builder's shape checks rather than
// dividing by zero.
func headCount(cfg metal.TransformerConfig) int32 {
	if cfg.NumAttentionHeads > 0 {
		return cfg.NumAttentionHeads
	}
	return 1
}

// linearOutDim returns a Linear's output feature count (weight is [out, in]), or
// 0 when absent — the basis for inferring a head dim a projection pins.
func linearOutDim(l *metal.Linear) int32 {
	if l == nil || l.Weight == nil || !l.Weight.Valid() {
		return 0
	}
	shape := l.Weight.Shape()
	if len(shape) == 0 {
		return 0
	}
	return shape[0]
}

// attnScale is the softmax attention scale 1/sqrt(headDim).
func attnScale(headDim int32) float32 {
	if headDim <= 0 {
		return 1
	}
	return float32(1.0 / math.Sqrt(float64(headDim)))
}

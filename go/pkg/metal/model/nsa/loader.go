// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package nsa

import (
	"math"

	core "dappco.re/go"
	metal "dappco.re/go/mlx/pkg/metal"
)

// loader.go registers NSA's load-time builder on the engine's mixer-loader
// registry, beside NSA's scheme + compute — no model import. A config layer_type
// of "nsa" resolves through metal.RegisterMixerLoader. ctx.Linear yields
// quant-ready *metal.Linear (the mixer never sees a quant format); geometry
// comes from ctx.Cfg.
//
// NSA uses the standard attention projections (q/k/v/o_proj) plus a gate
// projection (g_proj — the three-branch sigmoid gate). Head count from config;
// per-head dim from q_proj.out / heads. Block size / top-n / window default to
// the reference settings until a hybrid config carries them (the gemma-4 config
// has no sparse fields). A missing required projection is a loud build error.

const (
	defaultBlockSize = 64  // tokens per compression/selection block
	defaultSelect    = 16  // top-n selected blocks
	defaultWindow    = 512 // sliding-window span
)

func init() {
	metal.RegisterMixerLoader(MixerKind, buildNSA)
}

func buildNSA(ctx metal.MixerBuildCtx) (metal.MixerCompute, error) {
	const op = "nsa.build"
	qProj := ctx.Linear("q_proj")
	kProj := ctx.Linear("k_proj")
	vProj := ctx.Linear("v_proj")
	gProj := ctx.Linear("g_proj")
	oProj := ctx.Linear("o_proj")
	for name, l := range map[string]*metal.Linear{
		"q_proj": qProj, "k_proj": kProj, "v_proj": vProj, "g_proj": gProj, "o_proj": oProj,
	} {
		if l == nil {
			return nil, core.E(op, core.Sprintf("missing projection %q", name), nil)
		}
	}

	heads := headCount(ctx.Cfg)
	headDim := headDimFrom(ctx.Cfg, qProj, heads)
	if headDim <= 0 {
		return nil, core.E(op, "cannot determine NSA head dim from q_proj or config", nil)
	}

	return &Mixer{
		QProj: qProj, KProj: kProj, VProj: vProj, GProj: gProj, OProj: oProj,
		NumHeads: heads, HeadDim: headDim,
		BlockSize: defaultBlockSize, SelectBlocks: defaultSelect, Window: defaultWindow,
		Scale: attnScale(headDim),
	}, nil
}

func headCount(cfg metal.TransformerConfig) int32 {
	if cfg.NumAttentionHeads > 0 {
		return cfg.NumAttentionHeads
	}
	return 1
}

// headDimFrom infers the per-head dim from q_proj.out / heads, falling back to
// the config head dim, then hidden/heads.
func headDimFrom(cfg metal.TransformerConfig, qProj *metal.Linear, heads int32) int32 {
	if out := linearOutDim(qProj); out > 0 && heads > 0 && out%heads == 0 {
		return out / heads
	}
	if cfg.HeadDim > 0 {
		return cfg.HeadDim
	}
	if cfg.HiddenSize > 0 && heads > 0 {
		return cfg.HiddenSize / heads
	}
	return 0
}

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

func attnScale(headDim int32) float32 {
	if headDim <= 0 {
		return 1
	}
	return float32(1.0 / math.Sqrt(float64(headDim)))
}

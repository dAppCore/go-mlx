// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package moba

import (
	"math"

	core "dappco.re/go"
	metal "dappco.re/go/mlx/pkg/metal"
)

// loader.go registers MoBA's load-time builder on the engine's mixer-loader
// registry, beside MoBA's scheme + compute — no model import. A config
// layer_type of "moba" resolves through metal.RegisterMixerLoader. ctx.Linear
// yields quant-ready *metal.Linear; geometry comes from ctx.Cfg.
//
// MoBA uses the standard attention projections (q/k/v/o_proj). Head count from
// config; per-head dim from q_proj.out / heads. Block size / top-k default to
// the reference settings until a hybrid config carries them. A missing required
// projection is a loud build error.

const (
	defaultBlockSize = 64 // tokens per block
	defaultTopK      = 3  // top-k blocks (excluding the always-on self-block)
)

func init() {
	metal.RegisterMixerLoader(MixerKind, buildMoBA)
}

func buildMoBA(ctx metal.MixerBuildCtx) (metal.MixerCompute, error) {
	const op = "moba.build"
	qProj := ctx.Linear("q_proj")
	kProj := ctx.Linear("k_proj")
	vProj := ctx.Linear("v_proj")
	oProj := ctx.Linear("o_proj")
	for name, l := range map[string]*metal.Linear{
		"q_proj": qProj, "k_proj": kProj, "v_proj": vProj, "o_proj": oProj,
	} {
		if l == nil {
			return nil, core.E(op, core.Sprintf("missing projection %q", name), nil)
		}
	}

	heads := headCount(ctx.Cfg)
	headDim := headDimFrom(ctx.Cfg, qProj, heads)
	if headDim <= 0 {
		return nil, core.E(op, "cannot determine MoBA head dim from q_proj or config", nil)
	}

	return &Mixer{
		QProj: qProj, KProj: kProj, VProj: vProj, OProj: oProj,
		NumHeads: heads, HeadDim: headDim,
		BlockSize: defaultBlockSize, TopK: defaultTopK,
		Scale: attnScale(headDim),
	}, nil
}

func headCount(cfg metal.TransformerConfig) int32 {
	if cfg.NumAttentionHeads > 0 {
		return cfg.NumAttentionHeads
	}
	return 1
}

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

// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gsa

import (
	core "dappco.re/go"
	metal "dappco.re/go/mlx/pkg/metal"
)

// loader.go registers GSA's load-time builder on the engine's mixer-loader
// registry, beside GSA's scheme + compute — no model import. A config layer_type
// of "gsa" resolves through metal.RegisterMixerLoader. ctx.Linear yields
// quant-ready *metal.Linear; geometry comes from ctx.Cfg. GSA is recurrent
// (StateRecurrent) — its Forward threads the slot memory through
// ctx.Recurrent(); the loader only constructs the mixer.
//
// GSA adds a forget-gate projection (f_proj — heads*slots) and an output gate
// (g_proj) to the standard q/k/v/o_proj. Head count from config; key dim from
// q_proj.out / heads; value dim from v_proj.out / heads; slot count from
// f_proj.out / heads. A missing required projection is a loud build error.

func init() {
	metal.RegisterMixerLoader(MixerKind, buildGSA)
}

func buildGSA(ctx metal.MixerBuildCtx) (metal.MixerCompute, error) {
	const op = "gsa.build"
	qProj := ctx.Linear("q_proj")
	kProj := ctx.Linear("k_proj")
	vProj := ctx.Linear("v_proj")
	fProj := ctx.Linear("f_proj")
	gProj := ctx.Linear("g_proj")
	oProj := ctx.Linear("o_proj")
	for name, l := range map[string]*metal.Linear{
		"q_proj": qProj, "k_proj": kProj, "v_proj": vProj, "f_proj": fProj, "g_proj": gProj, "o_proj": oProj,
	} {
		if l == nil {
			return nil, core.E(op, core.Sprintf("missing projection %q", name), nil)
		}
	}

	heads := headCount(ctx.Cfg)
	headK := headDimFrom(ctx.Cfg, qProj, heads)
	headV := headK
	if vOut := linearOutDim(vProj); vOut > 0 && heads > 0 && vOut%heads == 0 {
		headV = vOut / heads
	}
	slots := int32(0)
	if fOut := linearOutDim(fProj); fOut > 0 && heads > 0 && fOut%heads == 0 {
		slots = fOut / heads
	}
	if headK <= 0 || headV <= 0 || slots <= 0 {
		return nil, core.E(op, core.Sprintf("cannot determine GSA geometry (headK=%d headV=%d slots=%d)", headK, headV, slots), nil)
	}

	return &Mixer{
		QProj: qProj, KProj: kProj, VProj: vProj, FProj: fProj, GProj: gProj, OProj: oProj,
		NumHeads: heads, HeadK: headK, HeadV: headV, Slots: slots, GateNorm: gateNorm(slots),
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

// gateNorm is the gate_logit_normalizer (logsigmoid divisor); the GSA reference
// uses the slot count. A hybrid config overrides it when it carries the field.
func gateNorm(slots int32) float32 {
	if slots > 0 {
		return float32(slots)
	}
	return 1
}

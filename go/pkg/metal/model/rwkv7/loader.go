// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package rwkv7

import (
	core "dappco.re/go"
	metal "dappco.re/go/mlx/pkg/metal"
)

// loader.go registers the RWKV-7 build-from-weights loader on the engine. A
// mixer is an engine feature, so the loader lives HERE with the family's math
// and scheme — not in any model package (AX-8). The load path resolves it by
// the kind a model config declares; the neutral metal.MixerBuildCtx hands it
// quant-ready Linears + the neutral transformer dims, so the builder is
// model-agnostic and quant-blind.

func init() {
	metal.RegisterMixerLoader("rwkv7", buildRWKV7)
}

// buildRWKV7 constructs an RWKV-7 mixer for one layer from the neutral build
// context. All RWKV-7 projections are Linears, so every weight comes through
// ctx.Linear (quant-ready); geometry is derived from the projection shapes and
// the neutral head count.
func buildRWKV7(ctx metal.MixerBuildCtx) (metal.MixerCompute, error) {
	const op = "rwkv7.buildRWKV7"

	rProj := ctx.Linear("receptance")
	kProj := ctx.Linear("key")
	vProj := ctx.Linear("value")
	outProj := ctx.Linear("output")
	wProj := ctx.Linear("decay")
	aProj := ctx.Linear("a_proj")
	bProj := ctx.Linear("b_proj")
	for name, l := range map[string]*metal.Linear{
		"receptance": rProj, "key": kProj, "value": vProj, "output": outProj,
		"decay": wProj, "a_proj": aProj, "b_proj": bProj,
	} {
		if l == nil {
			return nil, core.E(op, core.Sprintf("missing %s", name), nil)
		}
	}

	cfg, err := configFromShapes(ctx.Cfg.NumAttentionHeads, rProj, vProj)
	if err != nil {
		return nil, core.E(op, "derive geometry", err)
	}
	w := &Weights{
		RProj: rProj, WProj: wProj, KProj: kProj, VProj: vProj,
		AProj: aProj, BProj: bProj, OutProj: outProj,
	}
	return NewMixer(w, cfg), nil
}

// configFromShapes derives the per-layer WKV7 geometry from the projection
// shapes: the head count is the neutral attention-head count (RWKV-7 reuses
// it), and the per-head key/value dims are receptance_out/H and value_out/H. A
// model that decouples the RWKV head count from the attention head count
// declares it out-of-band; this loader surfaces a clear error rather than
// mis-splitting.
func configFromShapes(numHeads int32, rProj, vProj *metal.Linear) (*Config, error) {
	if numHeads <= 0 || rProj.Weight == nil || vProj.Weight == nil {
		return nil, core.E("rwkv7.configFromShapes", core.Sprintf("bad head count or weights: H=%d", numHeads), nil)
	}
	rOut := rProj.Weight.Shape()[0]
	vOut := vProj.Weight.Shape()[0]
	if rOut%numHeads != 0 || vOut%numHeads != 0 {
		return nil, core.E("rwkv7.configFromShapes",
			core.Sprintf("projection width not divisible by head count: H=%d rOut=%d vOut=%d", numHeads, rOut, vOut), nil)
	}
	return &Config{NumHeads: numHeads, KeyDim: rOut / numHeads, ValueDim: vOut / numHeads}, nil
}

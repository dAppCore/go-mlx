// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gla

// builder.go registers GLA's load-time builder on the engine's mixer-loader
// registry, so a model config that declares a layer_type of "gla" resolves end
// to end. The builder composes the four quant-ready projections plus the
// per-key-dim log-forget gate, wrapped as the GateFn the kernel consumes. Lives
// in this package alongside the scheme + compute; no model-package edit, no
// quant coupling (ctx.Linear arrives quant-ready).

import (
	core "dappco.re/go"
	metal "dappco.re/go/mlx/pkg/metal"
)

func init() { metal.RegisterMixerLoader("gla", buildMixer) }

// buildMixer composes a GLA Mixer for one layer: q/k/v/o plus the gk_proj gate
// projection, wrapped as a GateFn that projects the hidden state and applies
// log-sigmoid (GLA's forget gate is logσ of a linear, bounded ≤ 0), reshaped to
// the per-head [B,H,L,Dk] the kernel expects.
//
//	metal.RegisterMixerLoader("gla", buildMixer) // resolves layer_type "gla"
func buildMixer(ctx metal.MixerBuildCtx) (metal.MixerCompute, error) {
	q := ctx.Linear("q_proj")
	k := ctx.Linear("k_proj")
	v := ctx.Linear("v_proj")
	o := ctx.Linear("o_proj")
	if q == nil || k == nil || v == nil || o == nil {
		return nil, core.E("gla.buildMixer", "missing q/k/v/o projection", nil)
	}
	gateProj := ctx.Linear("gk_proj")
	if gateProj == nil {
		return nil, core.E("gla.buildMixer", "missing gk_proj gate projection", nil)
	}
	numHeads, headDim := headGeometry(ctx.Cfg)
	if numHeads <= 0 || headDim <= 0 {
		return nil, core.E("gla.buildMixer", "config declares no head geometry", nil)
	}
	w := &Weights{
		QProj:    q,
		KProj:    k,
		VProj:    v,
		Output:   o,
		NumHeads: numHeads,
		HeadDim:  headDim,
	}
	return &Mixer{W: w, Gate: gateFn(gateProj, numHeads, headDim)}, nil
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

// gateFn builds the per-layer GateFn from the gate projection: project the
// hidden state to [B,L,H*Dk], log-sigmoid it (the per-key-dim log-decay), and
// view it as the per-head [B,H,L,Dk] the kernel reads.
func gateFn(gateProj *metal.Linear, numHeads, headDim int) GateFn {
	return func(x *metal.Array, b, l int32) *metal.Array {
		proj := gateProj.Forward(x) // [B,L,H*Dk]
		gln := logSigmoid(proj)     // logσ(proj) ≤ 0 — the per-key-dim log-gate
		metal.Free(proj)
		view := metal.AsStrided(gln, []int32{b, int32(numHeads), l, int32(headDim)},
			[]int64{int64(l * int32(numHeads) * int32(headDim)), int64(headDim), int64(int32(numHeads) * int32(headDim)), 1}, 0)
		metal.Free(gln)
		return view
	}
}

// logSigmoid computes logσ(x) = −softplus(−x) elementwise, the numerically
// stable log-sigmoid GLA's forget gate uses. Composed from the metal op surface
// (no dedicated kernel) via the stable softplus(z) = max(z,0) + log(1+exp(−|z|))
// with z = −x, so it stays accurate for large |x|.
func logSigmoid(x *metal.Array) *metal.Array {
	negX := metal.Negative(x)
	absX := metal.Abs(x) // |−x| = |x|
	negAbs := metal.Negative(absX)
	metal.Free(absX)
	expNegAbs := metal.Exp(negAbs) // exp(−|x|)
	metal.Free(negAbs)
	onePlus := metal.AddScalar(expNegAbs, 1) // 1 + exp(−|x|)
	metal.Free(expNegAbs)
	logTerm := metal.Log(onePlus) // log(1 + exp(−|x|))
	metal.Free(onePlus)
	zero := metal.Zeros([]int32{1}, negX.Dtype())
	maxZ := metal.Maximum(negX, zero) // max(−x, 0)
	metal.Free(negX, zero)
	softplusNegX := metal.Add(maxZ, logTerm) // softplus(−x)
	metal.Free(maxZ, logTerm)
	out := metal.Negative(softplusNegX) // logσ(x) = −softplus(−x)
	metal.Free(softplusNegX)
	return out
}

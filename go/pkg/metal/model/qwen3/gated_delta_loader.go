// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package qwen3

import (
	core "dappco.re/go"
	metal "dappco.re/go/mlx/pkg/metal"
)

// gated_delta_loader.go registers the GatedDeltaNet build-from-weights loader for
// the "linear_attention" layer kind, so the config-composed model
// (pkg/metal/model/composed) resolves a Qwen3.5/3.6 linear layer through the mixer
// registry exactly like it resolves a "full_attention" softmax layer — no central
// switch. The neutral MixerBuildCtx hands quant-ready Linears + raw tensors with
// bare leaf names (the composed model owns the per-layer linear_attn. prefix), and
// the geometry is derived from the weight shapes (the recurrent-mixer convention).

func init() {
	metal.RegisterMixerLoader("linear_attention", buildGatedDelta)
}

// buildGatedDelta constructs a GatedDeltaNet mixer for one layer from the neutral
// build context. The five projections come through ctx.Linear (quant-ready); the
// conv kernel, A_log, dt_bias and the gated-norm weight through ctx.Weight.
func buildGatedDelta(ctx metal.MixerBuildCtx) (metal.MixerCompute, error) {
	const op = "qwen3.buildGatedDelta"

	qkv := ctx.Linear("in_proj_qkv")
	a := ctx.Linear("in_proj_a")
	b := ctx.Linear("in_proj_b")
	z := ctx.Linear("in_proj_z")
	out := ctx.Linear("out_proj")
	for name, l := range map[string]*metal.Linear{"in_proj_qkv": qkv, "in_proj_a": a, "in_proj_b": b, "in_proj_z": z, "out_proj": out} {
		if l == nil {
			return nil, core.E(op, core.Sprintf("missing %s", name), nil)
		}
	}
	convWeight := ctx.Weight("conv1d.weight")
	aLog := ctx.Weight("A_log")
	norm := ctx.Weight("norm.weight")
	for name, t := range map[string]*metal.Array{"conv1d.weight": convWeight, "A_log": aLog, "norm.weight": norm} {
		if t == nil {
			return nil, core.E(op, core.Sprintf("missing %s", name), nil)
		}
	}

	cfg, err := gatedDeltaConfigFromShapes(qkv, convWeight, aLog, norm, ctx.Cfg.RMSNormEps)
	if err != nil {
		return nil, core.E(op, "derive geometry", err)
	}
	w := &GatedDeltaWeights{
		InProjQKV:  qkv,
		InProjA:    a,
		InProjB:    b,
		InProjZ:    z,
		ConvWeight: convWeight,
		ConvBias:   ctx.Weight("conv1d.bias"),
		ALog:       aLog,
		DtBias:     ctx.Weight("dt_bias"),
		Norm:       norm,
		OutProj:    out,
	}
	return NewGatedDeltaMixer(w, cfg), nil
}

// gatedDeltaConfigFromShapes derives the per-layer geometry from the weight
// shapes. The value heads + head dim come from A_log + the gated-norm weight; the
// conv input width (= the QKV projection out) splits q | k | v as 2·qDim + vDim,
// so qDim = (convDim − vDim)/2. Key head dim == value head dim (the delta state is
// square [Dk,Dv], so the kernel requires it), which pins KeyHeads = qDim/headDim:
//
//	ValueHeads   = len(A_log)
//	ValueHeadDim = len(norm)
//	convDim      = conv1d.weight.shape[0]  (= 2·qDim + vDim)
//	convKernel   = conv1d.weight.shape[last]
//	Hidden       = in_proj_qkv.shape[1]
func gatedDeltaConfigFromShapes(qkv *metal.Linear, convWeight, aLog, norm *metal.Array, eps float32) (*GatedDeltaConfig, error) {
	const op = "qwen3.gatedDeltaConfigFromShapes"
	if qkv.Weight == nil {
		return nil, core.E(op, "in_proj_qkv has no weight", nil)
	}
	qkvShape := qkv.Weight.Shape()
	convShape := convWeight.Shape()
	if len(qkvShape) < 2 || len(convShape) == 0 {
		return nil, core.E(op, "in_proj_qkv / conv1d.weight shape malformed", nil)
	}
	valueHeads := int32(aLog.Dim(0))
	valueHeadDim := int32(norm.Dim(0))
	vDim := valueHeads * valueHeadDim
	convDim := convShape[0]
	convKernel := convShape[len(convShape)-1]
	hidden := qkvShape[1]

	qDim := (convDim - vDim) / 2
	keyHeadDim := valueHeadDim // square delta state ⇒ key dim == value dim
	if valueHeads <= 0 || valueHeadDim <= 0 || convDim != qkvShape[0] ||
		(convDim-vDim) <= 0 || (convDim-vDim)%2 != 0 || qDim%keyHeadDim != 0 {
		return nil, core.E(op, core.Sprintf("geometry inconsistent: VH=%d Dh=%d convDim=%d projOut=%d vDim=%d qDim=%d",
			valueHeads, valueHeadDim, convDim, qkvShape[0], vDim, qDim), nil)
	}
	return &GatedDeltaConfig{
		Hidden:       hidden,
		KeyHeads:     qDim / keyHeadDim,
		KeyHeadDim:   keyHeadDim,
		ValueHeads:   valueHeads,
		ValueHeadDim: valueHeadDim,
		ConvKernel:   convKernel,
		RMSNormEps:   eps,
	}, nil
}

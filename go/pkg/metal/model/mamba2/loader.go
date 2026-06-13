// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package mamba2

import (
	core "dappco.re/go"
	metal "dappco.re/go/mlx/pkg/metal"
)

// loader.go registers the Mamba-2 build-from-weights loader on the engine. A
// mixer is an engine feature, so the loader lives HERE with the family's math
// and scheme — not in any model package (AX-8: the lib never imports a
// consumer). The load path resolves it by the kind a model config declares for
// a layer; the neutral metal.MixerBuildCtx hands it quant-ready Linears + raw
// tensors + the neutral transformer dims, so the builder is model-agnostic and
// quant-blind.

func init() {
	metal.RegisterMixerLoader("mamba2", buildMamba2)
}

// buildMamba2 constructs a Mamba-2 mixer for one layer from the neutral build
// context. ctx.Linear hands back quant-ready in/out projections (the quant
// factory is already applied — this never sees a quant format); ctx.Weight
// hands back the raw non-Linear tensors (conv1d, A_log, and the optional D /
// dt_bias / norm). Geometry is derived from the weight shapes.
func buildMamba2(ctx metal.MixerBuildCtx) (metal.MixerCompute, error) {
	const op = "mamba2.buildMamba2"

	inProj := ctx.Linear("in_proj")
	if inProj == nil {
		return nil, core.E(op, "missing in_proj", nil)
	}
	outProj := ctx.Linear("out_proj")
	if outProj == nil {
		return nil, core.E(op, "missing out_proj", nil)
	}
	convWeight := ctx.Weight("conv1d.weight")
	if convWeight == nil {
		return nil, core.E(op, "missing conv1d.weight", nil)
	}
	aLog := ctx.Weight("A_log")
	if aLog == nil {
		return nil, core.E(op, "missing A_log", nil)
	}

	cfg, err := configFromShapes(inProj, convWeight, aLog, ctx.Cfg.RMSNormEps)
	if err != nil {
		return nil, core.E(op, "derive geometry", err)
	}
	w := &Weights{
		InProj:     inProj,
		OutProj:    outProj,
		ConvWeight: convWeight,
		ConvBias:   ctx.Weight("conv1d.bias"),
		ALog:       aLog,
		D:          ctx.Weight("D"),
		DtBias:     ctx.Weight("dt_bias"),
		Norm:       ctx.Weight("norm.weight"),
	}
	return NewMixer(w, cfg), nil
}

// configFromShapes derives the per-layer SSD geometry entirely from the weight
// shapes, assuming the standard nGroups=1 layout (B/C shared across heads — the
// common Mamba-2 config). The three shapes pin every dimension:
//
//	H        = len(A_log)
//	convDim  = conv1d.weight.shape[0]   (= dInner + 2*nGroups*N)
//	projOut  = in_proj.weight.shape[0]  (= 2*dInner + 2*nGroups*N + H)
//	convKern = conv1d.weight.shape[last]
//
// With nGroups=1: dInner = projOut − convDim − H, N = (convDim − dInner)/2,
// headDim = dInner/H. A model with nGroups>1 declares it out-of-band; this
// loader surfaces a clear error rather than silently mis-splitting B/C.
func configFromShapes(inProj *metal.Linear, convWeight, aLog *metal.Array, eps float32) (*Config, error) {
	if inProj.Weight == nil {
		return nil, core.E("mamba2.configFromShapes", "in_proj has no weight", nil)
	}
	convShape := convWeight.Shape()
	if len(convShape) == 0 {
		return nil, core.E("mamba2.configFromShapes", "conv1d.weight has no shape", nil)
	}
	h := int32(aLog.Dim(0))
	convDim := convShape[0]
	convKernel := convShape[len(convShape)-1]
	projOut := inProj.Weight.Shape()[0]

	const groups int32 = 1 // standard Mamba-2; nGroups>1 needs an out-of-band declaration
	dInner := projOut - convDim - h
	if h <= 0 || dInner <= 0 || dInner%h != 0 || (convDim-dInner)%2 != 0 {
		return nil, core.E("mamba2.configFromShapes",
			core.Sprintf("geometry not nGroups=1 consistent: H=%d projOut=%d convDim=%d dInner=%d", h, projOut, convDim, dInner), nil)
	}
	return &Config{
		NumHeads:   h,
		HeadDim:    dInner / h,
		StateDim:   (convDim - dInner) / 2,
		NumGroups:  groups,
		ConvKernel: convKernel,
		RMSNormEps: eps,
	}, nil
}

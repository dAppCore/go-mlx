// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma4

import (
	core "dappco.re/go"
	"dappco.re/go/mlx/pkg/metal"
	"dappco.re/go/mlx/pkg/metal/model/mamba2"
	"dappco.re/go/mlx/pkg/metal/model/rwkv7"
)

// mixer_ssm.go registers the load-time builders for the SSM / recurrent
// sequence-mixer families (Mamba-2, RWKV-7) on the Gemma-4 loader. A config that
// declares a layer_type of "mamba2" or "rwkv7" resolves end-to-end: the loader
// looks up the builder by the layer's MixerKind, the builder loads that layer's
// projection weights from the checkpoint map, and the decoder dispatches the
// returned MixerCompute through the scheme registry — threading the recurrent
// state through the #39 holder, with no decoder-loop branch per family.
//
// The mixer math lives in pkg/metal/model/{mamba2,rwkv7} (relocatable to
// go-inference); this file is the gemma4-side glue that knows the checkpoint
// weight-naming convention, the boundary the MixerBuilder contract draws.
//
// Weight names follow the canonical HF / mamba_ssm + RWKV-7 layout. A checkpoint
// that names its tensors differently needs its names mapped here (or upstream in
// the weight sanitiser); a missing required tensor is a loud build error, never
// a silent zero-weight mixer.

func init() {
	RegisterMixerBuilder("mamba2", buildMamba2Layer)
	RegisterMixerBuilder("rwkv7", buildRWKV7Layer)
}

// buildMamba2Layer constructs a Mamba-2 mixer for one layer from its checkpoint
// weights. The projection prefix is "<layer>.mixer" (the HF Mamba-2 block name);
// geometry comes from the config's SSM fields with shape-inference fallbacks.
func buildMamba2Layer(ctx MixerBuildCtx) (metal.MixerCompute, error) {
	const op = "gemma4.buildMamba2Layer"
	p := ctx.Prefix + ".mixer"

	inProj := gemma4Linear(ctx.Weights, p+".in_proj", ctx.Cfg.Quantization)
	if inProj == nil {
		return nil, core.E(op, core.Sprintf("missing weight %s.in_proj", p), nil)
	}
	outProj := gemma4Linear(ctx.Weights, p+".out_proj", ctx.Cfg.Quantization)
	if outProj == nil {
		return nil, core.E(op, core.Sprintf("missing weight %s.out_proj", p), nil)
	}
	convWeight := gemma4WeightAny(ctx.Weights, p+".conv1d.weight")
	if convWeight == nil {
		return nil, core.E(op, core.Sprintf("missing weight %s.conv1d.weight", p), nil)
	}
	aLog := gemma4WeightAny(ctx.Weights, p+".A_log")
	if aLog == nil {
		return nil, core.E(op, core.Sprintf("missing weight %s.A_log", p), nil)
	}

	cfg, err := mamba2Config(ctx.Cfg, inProj, convWeight, aLog)
	if err != nil {
		return nil, core.E(op, "derive geometry", err)
	}
	w := &mamba2.Weights{
		InProj:     inProj,
		OutProj:    outProj,
		ConvWeight: convWeight,
		ConvBias:   gemma4WeightAny(ctx.Weights, p+".conv1d.bias"),
		ALog:       aLog,
		D:          gemma4WeightAny(ctx.Weights, p+".D"),
		DtBias:     gemma4WeightAny(ctx.Weights, p+".dt_bias"),
		Norm:       gemma4WeightAny(ctx.Weights, p+".norm.weight"),
	}
	return mamba2.NewMixer(w, cfg), nil
}

// mamba2Config derives the per-layer SSD geometry entirely from the weight
// shapes, assuming the standard nGroups=1 layout (B/C shared across heads — the
// common Mamba-2 config). The three shapes pin every dimension:
//
//	H        = len(A_log)
//	convDim  = conv1d.weight.shape[0]          (= dInner + 2*nGroups*N)
//	projOut  = in_proj.weight.shape[0]          (= 2*dInner + 2*nGroups*N + H)
//	convKern = conv1d.weight.shape[last]
//
// With nGroups=1: dInner = projOut − convDim − H, N = (convDim − dInner)/2,
// headDim = dInner/H. A model with nGroups>1 declares it out-of-band; this
// builder surfaces a clear error rather than silently mis-splitting B/C, so the
// config schema for that case (a loader-owned decision) is a visible follow-up
// rather than a wrong load.
func mamba2Config(cfg *Gemma4TextConfig, inProj *metal.Linear, convWeight, aLog *metal.Array) (*mamba2.Config, error) {
	h := int32(aLog.Dim(0))
	convShape := convWeight.Shape()
	if len(convShape) == 0 || inProj.Weight == nil {
		return nil, core.E("gemma4.mamba2Config", "bad conv1d/in_proj weight shapes", nil)
	}
	convDim := convShape[0]
	convKernel := convShape[len(convShape)-1]
	projOut := inProj.Weight.Shape()[0]

	const groups int32 = 1 // standard Mamba-2; nGroups>1 needs an out-of-band declaration
	dInner := projOut - convDim - h
	if h <= 0 || dInner <= 0 || dInner%h != 0 || (convDim-dInner)%2 != 0 {
		return nil, core.E("gemma4.mamba2Config",
			core.Sprintf("geometry not nGroups=1 consistent: H=%d projOut=%d convDim=%d dInner=%d", h, projOut, convDim, dInner), nil)
	}
	stateDim := (convDim - dInner) / 2
	headDim := dInner / h
	return &mamba2.Config{
		NumHeads:   h,
		HeadDim:    headDim,
		StateDim:   stateDim,
		NumGroups:  groups,
		ConvKernel: convKernel,
		RMSNormEps: cfg.RMSNormEps,
	}, nil
}

// buildRWKV7Layer constructs an RWKV-7 mixer for one layer from its checkpoint
// weights. The projection prefix is "<layer>.attention" (the RWKV time-mix block
// name); geometry comes from the config's RWKV fields.
func buildRWKV7Layer(ctx MixerBuildCtx) (metal.MixerCompute, error) {
	const op = "gemma4.buildRWKV7Layer"
	p := ctx.Prefix + ".attention"

	rProj := gemma4Linear(ctx.Weights, p+".receptance", ctx.Cfg.Quantization)
	kProj := gemma4Linear(ctx.Weights, p+".key", ctx.Cfg.Quantization)
	vProj := gemma4Linear(ctx.Weights, p+".value", ctx.Cfg.Quantization)
	outProj := gemma4Linear(ctx.Weights, p+".output", ctx.Cfg.Quantization)
	for name, l := range map[string]*metal.Linear{"receptance": rProj, "key": kProj, "value": vProj, "output": outProj} {
		if l == nil {
			return nil, core.E(op, core.Sprintf("missing weight %s.%s", p, name), nil)
		}
	}

	cfg, err := rwkv7Config(ctx.Cfg, rProj, vProj)
	if err != nil {
		return nil, core.E(op, "derive geometry", err)
	}
	w := &rwkv7.Weights{
		RProj:   rProj,
		KProj:   kProj,
		VProj:   vProj,
		OutProj: outProj,
		WProj:   gemma4Linear(ctx.Weights, p+".decay", ctx.Cfg.Quantization),
		AProj:   gemma4Linear(ctx.Weights, p+".a_proj", ctx.Cfg.Quantization),
		BProj:   gemma4Linear(ctx.Weights, p+".b_proj", ctx.Cfg.Quantization),
	}
	if w.WProj == nil || w.AProj == nil || w.BProj == nil {
		return nil, core.E(op, core.Sprintf("missing weight %s.{decay,a_proj,b_proj}", p), nil)
	}
	return rwkv7.NewMixer(w, cfg), nil
}

// rwkv7Config derives the per-layer WKV7 geometry from the projection shapes:
// the head count H is the config's attention-head count (RWKV-7 reuses it),
// and the per-head key/value dims are receptance_out/H and value_out/H. A
// model that decouples the RWKV head count from NumAttentionHeads declares it
// out-of-band; this builder surfaces a clear error rather than mis-splitting.
func rwkv7Config(cfg *Gemma4TextConfig, rProj, vProj *metal.Linear) (*rwkv7.Config, error) {
	h := cfg.NumAttentionHeads
	if h <= 0 || rProj.Weight == nil || vProj.Weight == nil {
		return nil, core.E("gemma4.rwkv7Config", core.Sprintf("bad head count or weights: H=%d", h), nil)
	}
	rOut := rProj.Weight.Shape()[0]
	vOut := vProj.Weight.Shape()[0]
	if rOut%h != 0 || vOut%h != 0 {
		return nil, core.E("gemma4.rwkv7Config",
			core.Sprintf("projection width not divisible by head count: H=%d rOut=%d vOut=%d", h, rOut, vOut), nil)
	}
	return &rwkv7.Config{NumHeads: h, KeyDim: rOut / h, ValueDim: vOut / h}, nil
}

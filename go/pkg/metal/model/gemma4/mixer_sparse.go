// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma4

import (
	"math"

	core "dappco.re/go"
	"dappco.re/go/mlx/pkg/metal"
	"dappco.re/go/mlx/pkg/metal/model/gsa"
	"dappco.re/go/mlx/pkg/metal/model/mla"
	"dappco.re/go/mlx/pkg/metal/model/moba"
	"dappco.re/go/mlx/pkg/metal/model/nsa"
)

// mixer_sparse.go registers the load-time builders for the sparse / latent
// sequence-mixer families (MLA, NSA, MoBA, GSA) on the Gemma-4 loader. A config
// that declares a layer_type of "mla" / "nsa" / "moba" / "gsa" resolves
// end-to-end: the loader looks the builder up by the layer's MixerKind, the
// builder loads that layer's projection weights from the checkpoint map, and
// the decoder dispatches the returned MixerCompute through the scheme registry —
// MLA/NSA/MoBA against a KV cache, GSA against the #39 recurrent holder, with no
// decoder-loop branch per family.
//
// The mixer math lives in pkg/metal/model/{mla,nsa,moba,gsa} (relocatable to
// go-inference); this file is the gemma4-side glue that knows the checkpoint
// weight-naming convention — the boundary the MixerBuilder contract draws.
// A missing required tensor is a loud build error, never a silent zero-weight
// mixer. Head count + per-head dim come from the config; the sparse hyper-
// parameters (block size, top-n, slot count, latent rank) are inferred from
// weight shapes where a weight pins them and otherwise defaulted with a
// documented value, since the Gemma-4 config carries no sparse-family fields
// yet — a hybrid pack that declares these layers supplies the geometry in its
// own config, mapped here when it lands.

func init() {
	RegisterMixerBuilder("mla", buildMLALayer)
	RegisterMixerBuilder("nsa", buildNSALayer)
	RegisterMixerBuilder("moba", buildMoBALayer)
	RegisterMixerBuilder("gsa", buildGSALayer)
}

// sparseAttnScale is the softmax attention scale for a per-head dimension —
// 1/sqrt(headDim), the convention every attention mixer shares.
func sparseAttnScale(headDim int32) float32 {
	if headDim <= 0 {
		return 1
	}
	return float32(1.0 / math.Sqrt(float64(headDim)))
}

// sparseHeadCount resolves the attention head count for a sparse layer from the
// config, falling back to 1 so a malformed config fails in the builder's
// shape checks rather than dividing by zero here.
func sparseHeadCount(cfg *Gemma4TextConfig) int32 {
	if cfg != nil && cfg.NumAttentionHeads > 0 {
		return cfg.NumAttentionHeads
	}
	return 1
}

// linearOutDim returns a Linear's output feature count (weight is [out, in]), or
// 0 when the weight is absent — the basis for inferring a sparse hyperparameter
// a projection pins (e.g. GSA slots = FProj.out / heads).
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

// Default sparse hyperparameters used when neither a weight shape nor a config
// field pins them. Chosen to match the reference papers' common settings; a
// hybrid pack overrides these from its config once the loader carries the
// fields.
const (
	defaultBlockSize    = 64 // NSA/MoBA tokens per block
	defaultNSASelect    = 16 // NSA top-n selected blocks
	defaultNSAWindow    = 512 // NSA sliding-window span
	defaultMoBATopK     = 3  // MoBA top-k blocks (excluding the always-on self-block)
)

// buildMLALayer constructs a Multi-head Latent Attention mixer for one layer.
// MLA projections follow the DeepSeek-V2 naming: kv_a_proj_with_mqa (down →
// KV latent), kv_b_proj (up → K/V), q_a_proj / q_b_proj (down/up query), o_proj.
// Latent dims are inferred from the down-projection output shapes; HeadDim from
// the up-projection output / head count.
func buildMLALayer(ctx MixerBuildCtx) (metal.MixerCompute, error) {
	const op = "gemma4.buildMLALayer"
	p := ctx.Prefix + ".self_attn"
	q := ctx.Cfg.Quantization

	wDKV := gemma4Linear(ctx.Weights, p+".kv_a_proj_with_mqa", q)
	wUK := gemma4Linear(ctx.Weights, p+".kv_b_proj", q)
	wDQ := gemma4Linear(ctx.Weights, p+".q_a_proj", q)
	wUQ := gemma4Linear(ctx.Weights, p+".q_b_proj", q)
	oProj := gemma4Linear(ctx.Weights, p+".o_proj", q)
	for name, l := range map[string]*metal.Linear{
		"kv_a_proj_with_mqa": wDKV, "kv_b_proj": wUK, "q_a_proj": wDQ, "q_b_proj": wUQ, "o_proj": oProj,
	} {
		if l == nil {
			return nil, core.E(op, core.Sprintf("missing weight %s.%s", p, name), nil)
		}
	}

	heads := sparseHeadCount(ctx.Cfg)
	// kv_b_proj outputs the concatenated K+V up-projection (2 * heads * headDim);
	// split per head to recover headDim. Fall back to the config head dim.
	headDim := ctx.Cfg.HeadDim
	if up := linearOutDim(wUK); up > 0 && heads > 0 && up%(2*heads) == 0 {
		headDim = up / (2 * heads)
	}
	if headDim <= 0 {
		return nil, core.E(op, "cannot determine MLA head dim from kv_b_proj or config", nil)
	}

	return &mla.Mixer{
		WDKV: wDKV, WUK: wUK, WUV: wUK, WDQ: wDQ, WUQ: wUQ, OProj: oProj,
		NumHeads: heads, HeadDim: headDim, Scale: sparseAttnScale(headDim),
	}, nil
}

// buildNSALayer constructs a Native Sparse Attention mixer for one layer. NSA
// uses the standard attention projection names plus a gate projection (the
// three-branch sigmoid gate). Block size / top-n / window default to the
// reference settings until a hybrid config carries them.
func buildNSALayer(ctx MixerBuildCtx) (metal.MixerCompute, error) {
	const op = "gemma4.buildNSALayer"
	p := ctx.Prefix + ".self_attn"
	q := ctx.Cfg.Quantization

	qProj := gemma4Linear(ctx.Weights, p+".q_proj", q)
	kProj := gemma4Linear(ctx.Weights, p+".k_proj", q)
	vProj := gemma4Linear(ctx.Weights, p+".v_proj", q)
	gProj := gemma4Linear(ctx.Weights, p+".g_proj", q)
	oProj := gemma4Linear(ctx.Weights, p+".o_proj", q)
	for name, l := range map[string]*metal.Linear{
		"q_proj": qProj, "k_proj": kProj, "v_proj": vProj, "g_proj": gProj, "o_proj": oProj,
	} {
		if l == nil {
			return nil, core.E(op, core.Sprintf("missing weight %s.%s", p, name), nil)
		}
	}

	heads := sparseHeadCount(ctx.Cfg)
	headDim := nsaHeadDim(ctx.Cfg, qProj, heads)
	if headDim <= 0 {
		return nil, core.E(op, "cannot determine NSA head dim from q_proj or config", nil)
	}

	return &nsa.Mixer{
		QProj: qProj, KProj: kProj, VProj: vProj, GProj: gProj, OProj: oProj,
		NumHeads: heads, HeadDim: headDim,
		BlockSize: defaultBlockSize, SelectBlocks: defaultNSASelect, Window: defaultNSAWindow,
		Scale: sparseAttnScale(headDim),
	}, nil
}

// buildMoBALayer constructs a Mixture of Block Attention mixer for one layer.
func buildMoBALayer(ctx MixerBuildCtx) (metal.MixerCompute, error) {
	const op = "gemma4.buildMoBALayer"
	p := ctx.Prefix + ".self_attn"
	q := ctx.Cfg.Quantization

	qProj := gemma4Linear(ctx.Weights, p+".q_proj", q)
	kProj := gemma4Linear(ctx.Weights, p+".k_proj", q)
	vProj := gemma4Linear(ctx.Weights, p+".v_proj", q)
	oProj := gemma4Linear(ctx.Weights, p+".o_proj", q)
	for name, l := range map[string]*metal.Linear{
		"q_proj": qProj, "k_proj": kProj, "v_proj": vProj, "o_proj": oProj,
	} {
		if l == nil {
			return nil, core.E(op, core.Sprintf("missing weight %s.%s", p, name), nil)
		}
	}

	heads := sparseHeadCount(ctx.Cfg)
	headDim := nsaHeadDim(ctx.Cfg, qProj, heads)
	if headDim <= 0 {
		return nil, core.E(op, "cannot determine MoBA head dim from q_proj or config", nil)
	}

	return &moba.Mixer{
		QProj: qProj, KProj: kProj, VProj: vProj, OProj: oProj,
		NumHeads: heads, HeadDim: headDim,
		BlockSize: defaultBlockSize, TopK: defaultMoBATopK,
		Scale: sparseAttnScale(headDim),
	}, nil
}

// buildGSALayer constructs a Gated Slot Attention mixer for one layer. GSA adds
// a forget-gate projection (heads * slots) and an output gate; the slot count is
// inferred from f_proj's output dimension. State is recurrent → the decoder
// pairs it with the #39 holder.
func buildGSALayer(ctx MixerBuildCtx) (metal.MixerCompute, error) {
	const op = "gemma4.buildGSALayer"
	p := ctx.Prefix + ".self_attn"
	q := ctx.Cfg.Quantization

	qProj := gemma4Linear(ctx.Weights, p+".q_proj", q)
	kProj := gemma4Linear(ctx.Weights, p+".k_proj", q)
	vProj := gemma4Linear(ctx.Weights, p+".v_proj", q)
	fProj := gemma4Linear(ctx.Weights, p+".f_proj", q)
	gProj := gemma4Linear(ctx.Weights, p+".g_proj", q)
	oProj := gemma4Linear(ctx.Weights, p+".o_proj", q)
	for name, l := range map[string]*metal.Linear{
		"q_proj": qProj, "k_proj": kProj, "v_proj": vProj, "f_proj": fProj, "g_proj": gProj, "o_proj": oProj,
	} {
		if l == nil {
			return nil, core.E(op, core.Sprintf("missing weight %s.%s", p, name), nil)
		}
	}

	heads := sparseHeadCount(ctx.Cfg)
	headK := nsaHeadDim(ctx.Cfg, qProj, heads)
	headV := headK
	if vOut := linearOutDim(vProj); vOut > 0 && heads > 0 && vOut%heads == 0 {
		headV = vOut / heads
	}
	// Slots = f_proj.out / heads (the forget gate is per-head-per-slot).
	slots := int32(0)
	if fOut := linearOutDim(fProj); fOut > 0 && heads > 0 && fOut%heads == 0 {
		slots = fOut / heads
	}
	if headK <= 0 || headV <= 0 || slots <= 0 {
		return nil, core.E(op, core.Sprintf("cannot determine GSA geometry (headK=%d headV=%d slots=%d)", headK, headV, slots), nil)
	}

	return &gsa.Mixer{
		QProj: qProj, KProj: kProj, VProj: vProj, FProj: fProj, GProj: gProj, OProj: oProj,
		NumHeads: heads, HeadK: headK, HeadV: headV, Slots: slots, GateNorm: gsaGateNorm(slots),
	}, nil
}

// nsaHeadDim resolves the per-head dimension for the q-proj-shaped sparse mixers
// (NSA/MoBA/GSA key dim): q_proj.out / heads, falling back to the config head
// dim, then hidden/heads.
func nsaHeadDim(cfg *Gemma4TextConfig, qProj *metal.Linear, heads int32) int32 {
	if out := linearOutDim(qProj); out > 0 && heads > 0 && out%heads == 0 {
		return out / heads
	}
	if cfg != nil && cfg.HeadDim > 0 {
		return cfg.HeadDim
	}
	if cfg != nil && cfg.HiddenSize > 0 && heads > 0 {
		return cfg.HiddenSize / heads
	}
	return 0
}

// gsaGateNorm is the gate_logit_normalizer (the logsigmoid divisor). The GSA
// reference uses the slot count; a hybrid config overrides it when it carries
// the field.
func gsaGateNorm(slots int32) float32 {
	if slots > 0 {
		return float32(slots)
	}
	return 1
}

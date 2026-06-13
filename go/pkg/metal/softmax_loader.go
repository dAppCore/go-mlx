// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	core "dappco.re/go"
	scheme "dappco.re/go/mlx/pkg/scheme"
)

// softmax_loader.go registers the engine's generic softmax-attention mixer — the
// keystone that lets the config-composed model build a full-attention layer the
// same way it builds a flash-linear-attention one. It wraps the SDK's neutral
// grouped-query attention (GQAAttention), so it is family-agnostic: any config
// that declares a "full_attention" layer resolves it through the mixer registry,
// with no gemma4 import and no central switch. gemma4's own softmax_mixer.go
// stays the hand-written shared-KV/runtime-mask variant for the gemma4 path; this
// is the plain dense-family one a composed hybrid uses for its attention layers.

// softmaxMixer adapts the SDK's GQAAttention to the universal MixerCompute
// contract. It captures the layer's *DenseConfig at build time (for rope_theta,
// the attention scale, and the head counts the neutral TransformerConfig does not
// carry) and maps the per-forward MixerCtx onto the existing attention compute —
// identical math to a direct GQAAttention call, just dispatched through the
// registry.
type softmaxMixer struct {
	attn *GQAAttention
	cfg  *DenseConfig
}

// Kind is the layer_types token a full-attention layer declares; the composed
// model resolves this mixer with MixerLoaderFor("full_attention").
func (*softmaxMixer) Kind() string { return "full_attention" }

// State declares softmax attention keeps a growing per-layer K/V cache, so the
// composed model gives this layer a KV cache (not the recurrent holder) and the
// cache layer (quant / compaction) operates on it.
func (*softmaxMixer) State() scheme.StateKind { return scheme.StateKVCache }

// Forward mixes one chunk by delegating to the SDK attention forward, mapping the
// universal MixerCtx onto its parameters. RoPE reads its position from the
// cache offset inside the attention compute, so a decode step needs no extra
// context here. Returns the zero SharedKV — the generic softmax mixer keeps all
// of its state in ctx.Cache and does not share K/V (that is a gemma4 specific).
func (m *softmaxMixer) Forward(x *Array, ctx *MixerCtx) (*Array, SharedKV) {
	return m.attn.forward(x, ctx.Cache, ctx.B, ctx.L, ctx.Mask, m.cfg), SharedKV{}
}

// CloseMixer releases the attention projections this mixer owns (MixerCloser), so
// the composed model frees a full-attention layer's weights on Close.
func (m *softmaxMixer) CloseMixer() {
	if m == nil || m.attn == nil {
		return
	}
	FreeLinear(m.attn.QProj)
	FreeLinear(m.attn.KProj)
	FreeLinear(m.attn.VProj)
	FreeLinear(m.attn.OProj)
	FreeRMSNorm(m.attn.QNorm)
	FreeRMSNorm(m.attn.KNorm)
}

// buildSoftmax constructs a generic softmax-attention mixer for one layer from
// the neutral build context. The four projections come through ctx.Linear
// (quant-ready — this never sees a quant format); the optional Q/K RMS norms
// (Qwen-3 style; absent on Qwen-2/Llama) come through ctx.Weight. The layer's
// *DenseConfig rides ctx.Extra because rope_theta + the attention scale are
// family-config fields the neutral TransformerConfig does not carry.
func buildSoftmax(ctx MixerBuildCtx) (MixerCompute, error) {
	const op = "metal.buildSoftmax"

	cfg, ok := ctx.Extra.(*DenseConfig)
	if !ok || cfg == nil {
		return nil, core.E(op, "softmax mixer needs the layer *DenseConfig via MixerBuildCtx.Extra", nil)
	}

	q := ctx.Linear("self_attn.q_proj")
	k := ctx.Linear("self_attn.k_proj")
	v := ctx.Linear("self_attn.v_proj")
	o := ctx.Linear("self_attn.o_proj")
	for name, l := range map[string]*Linear{"q_proj": q, "k_proj": k, "v_proj": v, "o_proj": o} {
		if l == nil {
			return nil, core.E(op, core.Sprintf("missing self_attn.%s", name), nil)
		}
	}

	attn := &GQAAttention{QProj: q, KProj: k, VProj: v, OProj: o}
	if qn := ctx.Weight("self_attn.q_norm.weight"); qn != nil {
		attn.QNorm = &RMSNormModule{Weight: qn}
	}
	if kn := ctx.Weight("self_attn.k_norm.weight"); kn != nil {
		attn.KNorm = &RMSNormModule{Weight: kn}
	}

	return &softmaxMixer{attn: attn, cfg: cfg}, nil
}

func init() { RegisterMixerLoader("full_attention", buildSoftmax) }

// compile-time proof the generic softmax mixer is a full MixerCompute.
var _ MixerCompute = (*softmaxMixer)(nil)

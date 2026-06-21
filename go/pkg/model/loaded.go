// SPDX-Licence-Identifier: EUPL-1.2

package model

import core "dappco.re/go"

// loaded.go is the neutral loaded-weights set: the single hand-off between a model package's weight
// parsing (gemma4, mistral, future archs) and a backend's device upload (pkg/native, future go-rocm).
// It lives at the pkg/model ROOT, not a model subpackage — a LoadedModel is what EVERY arch produces,
// so a model-named home would force every backend + every other model to import that one model for a
// neutral type. The arch-specific fields (QK-norm, layer-scalar, the PLE tower, MoE) are optional:
// archs without them leave them nil (Mistral is gemma4 minus the extras).

// LoadedLayer is one decode layer's weights: projections as quant-agnostic Linear, norms as raw bf16
// bytes. KV-shared layers carry nil K/V (they read the owner's cache); dense layers carry Gate/Up/Down,
// MoE layers carry MoE instead.
type LoadedLayer struct {
	AttnNorm, PostAttnNorm []byte // input_layernorm, post_attention_layernorm
	QNorm, KNorm           []byte // self_attn.q_norm / k_norm (nil without QK-norm)
	LayerScalar            []byte // per-layer output scalar [1] (nil when absent)
	Q, K, V, O             *Linear

	MLPNorm, PostFFNorm []byte // pre/post feedforward norms (dense MLP)
	Gate, Up, Down      *Linear
	MoE                 *LoadedMoE // non-nil ⇒ MoE layer (Gate/Up/Down then unused)

	PerLayerGate, PerLayerProjection *Linear // per-layer-input gate (E2B/E4B PLE); nil without the tower
	PostPerLayerInputNorm            []byte
}

// LoadedMoE is a MoE layer's dual-branch FFN: a dense local MLP + the sparse experts, each with its
// own norms.
type LoadedMoE struct {
	PreFFNorm, PreFFNorm2, PostFFNorm1, PostFFNorm2, PostFFNorm []byte
	RouterScale, PerExpertScale                                 []byte
	LocalGate, LocalUp, LocalDown                               *Linear
	Router                                                      *Linear
	ExpGate, ExpUp, ExpDown                                     *Linear // experts.switch_glu.*
}

// LoadedModel is the whole backend-agnostic weight set: the Arch + every weight as a Linear or raw
// norm bytes, viewing the source mmap. The single assembler output every backend consumes.
type LoadedModel struct {
	Arch      Arch
	Embed     *Linear // token embedding (also the tied LM head when LMHead is nil)
	LMHead    *Linear // separate output projection, or nil ⇒ tied to Embed
	FinalNorm []byte
	Layers    []LoadedLayer

	EmbedPerLayer     *Linear // PLE tower (E2B/E4B); nil when absent
	PerLayerModelProj *Linear
	PerLayerProjNorm  []byte
}

// Tied reports whether the LM head reuses the token embedding (no separate lm_head weight).
func (m *LoadedModel) Tied() bool { return m.LMHead == nil }

// ValidateRequired checks the always-present weights are there — a missing one is a malformed
// checkpoint, surfaced as a clean load error rather than a nil-deref deep in the decode. OPTIONAL
// weights are deliberately not required: k/v on KV-shared layers, v on K==V layers, lm_head when tied,
// the PLE tower, and QK-norm — so a well-formed checkpoint of any family/quant passes and only a
// genuinely-incomplete one is rejected. Every arch's assembler calls this on its LoadedModel.
func (m *LoadedModel) ValidateRequired(arch Arch) error {
	if m.Embed == nil {
		return core.NewError("model.LoadedModel: missing model.embed_tokens")
	}
	if m.FinalNorm == nil {
		return core.NewError("model.LoadedModel: missing model.norm.weight")
	}
	for i := range m.Layers {
		L := &m.Layers[i]
		if len(L.AttnNorm) == 0 || L.Q == nil || L.O == nil {
			return core.NewError(core.Sprintf("model.LoadedModel: layer %d missing input_layernorm/q_proj/o_proj", i))
		}
		if arch.Layer[i].OwnsCache() && L.K == nil {
			return core.NewError(core.Sprintf("model.LoadedModel: layer %d missing k_proj (cache owner)", i))
		}
		if L.MoE == nil && (len(L.MLPNorm) == 0 || L.Gate == nil || L.Up == nil || L.Down == nil) {
			return core.NewError(core.Sprintf("model.LoadedModel: layer %d missing a required dense-MLP weight", i))
		}
	}
	return nil
}

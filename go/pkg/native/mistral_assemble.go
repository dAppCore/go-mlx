// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	core "dappco.re/go"
	"dappco.re/go/mlx/pkg/model"
	"dappco.re/go/mlx/pkg/safetensors"
)

// AssembleMistralBF16 maps parsed bf16 safetensors onto the native weight structs per the
// Mistral / Ministral-3 weight-name convention and the Arch's dims. Ministral-3 is a standard
// Mistral transformer — a SUBSET of gemma4 — so it reuses the gemma4 bf16 structs + the shared
// arch executor: only the per-layer norms differ. A Mistral layer has exactly TWO norms,
// input_layernorm (pre-attention → AttnNormW) and post_attention_layernorm (pre-MLP → MLPNormW),
// and NONE of gemma4's extras (QK-norm, post-attention/post-FF norms, layer_scalar). Those stay
// nil, and the executor already skips an absent norm (sharedOrNil → encResidualMaybeNorm /
// qNorm != nil), so the assembled model decodes the faithful Mistral layer:
// rms(input) → attn → +residual → rms(post_attn) → SwiGLU → +residual.
//
// Real Ministral-3 packs are the multimodal wrapper (Mistral3ForConditionalGeneration): the text
// weights live under "language_model.model.*" with vision_tower.* / multi_modal_projector.*
// siblings — normalizeGemma4Names strips the wrapper prefix and drops the non-text towers (the
// same language_model. wrapper gemma4's unified checkpoints use). Embeddings are tied
// (tie_word_embeddings) so lm_head.weight is absent → LMHead aliases the embedding.
//
// SCOPE: dense bf16 only — the FP8 Instruct variants (a float8 quant format) and the vision
// tower are follow-up slices. RoPE uses the Arch's base theta; YaRN long-context scaling is a
// later faithfulness refinement (see pkg/model/mistral).
func AssembleMistralBF16(tensors map[string]safetensors.Tensor, arch model.Arch) (*Gemma4BF16, error) {
	tensors = normalizeGemma4Names(tensors)
	if arch.HasMoE() {
		return nil, core.NewError("native.AssembleMistralBF16: MoE arch not supported (Ministral-3 is dense)")
	}
	if len(arch.Layer) == 0 || arch.Hidden <= 0 || arch.Vocab <= 0 {
		return nil, core.NewError("native.AssembleMistralBF16: arch must have layers, hidden and vocab")
	}
	dModel, headDim, dFF, vocab := arch.Hidden, arch.HeadDim, arch.FF, arch.Vocab
	qDim, kvDim := arch.Heads*headDim, arch.KVHeads*headDim

	var ferr error
	fetch := func(name string, elems int, optional bool) []byte {
		if ferr != nil {
			return nil
		}
		t, ok := tensors[name]
		if !ok {
			if !optional {
				ferr = core.NewError("native.AssembleMistralBF16: missing " + name)
			}
			return nil
		}
		if t.Dtype != "BF16" {
			ferr = core.NewError("native.AssembleMistralBF16: " + name + " dtype " + t.Dtype + ", want BF16")
			return nil
		}
		if len(t.Data) != elems*bf16Size {
			ferr = core.NewError("native.AssembleMistralBF16: " + name + " byte size mismatch vs arch dims")
			return nil
		}
		return t.Data
	}

	g := &Gemma4BF16{Layers: make([]DecodeLayerWeights, len(arch.Layer))}
	g.Embed = fetch("model.embed_tokens.weight", vocab*dModel, false)
	g.FinalNorm = fetch("model.norm.weight", dModel, false)
	if _, ok := tensors["lm_head.weight"]; ok {
		g.LMHead = fetch("lm_head.weight", vocab*dModel, false)
	} else {
		g.LMHead, g.Tied = g.Embed, true // Ministral-3 ties the output projection to the embedding
	}

	for i := range arch.Layer {
		p := core.Sprintf("model.layers.%d", i)
		l := &g.Layers[i]
		l.AttnNormW = fetch(p+".input_layernorm.weight", dModel, false)
		l.WQ = fetch(p+".self_attn.q_proj.weight", qDim*dModel, false)
		l.WK = fetch(p+".self_attn.k_proj.weight", kvDim*dModel, false)
		l.WV = fetch(p+".self_attn.v_proj.weight", kvDim*dModel, false)
		l.WO = fetch(p+".self_attn.o_proj.weight", dModel*qDim, false)
		// the Mistral pre-MLP norm is post_attention_layernorm (gemma4 names it pre_feedforward).
		l.MLPNormW = fetch(p+".post_attention_layernorm.weight", dModel, false)
		l.WGate = fetch(p+".mlp.gate_proj.weight", dFF*dModel, false)
		l.WUp = fetch(p+".mlp.up_proj.weight", dFF*dModel, false)
		l.WDown = fetch(p+".mlp.down_proj.weight", dModel*dFF, false)
		// no QK-norm / post-attention / post-FF norm / layer_scalar — Mistral has none (stay nil).
	}
	if ferr != nil {
		return nil, ferr
	}
	return g, nil
}

// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	core "dappco.re/go"
	g4 "dappco.re/go/mlx/pkg/model/gemma4"
	"dappco.re/go/mlx/pkg/safetensors"
)

// Gemma4BF16 is a gemma4 model's bf16 weights mapped onto the native structs: the
// per-layer DecodeLayerWeights plus the model-level tensors (input embedding table,
// final norm, LM head). LMHead aliases Embed when the checkpoint ties them (Tied).
type Gemma4BF16 struct {
	Layers    []DecodeLayerWeights
	Embed     []byte // [vocab × dModel] bf16
	FinalNorm []byte // [dModel] bf16 (model.norm.weight)
	LMHead    []byte // [vocab × dModel] bf16 (lm_head.weight, or Embed when tied)
	Tied      bool   // LMHead is the tied embedding (no separate lm_head.weight)
}

// AssembleGemma4BF16 maps parsed bf16 safetensors tensors onto the native weight structs
// per the gemma4 weight-name convention (model.layers.N.self_attn.q_proj.weight, …) and
// the Arch's dims. It loads the dense bf16 path: attention Q/K/V/O + the four gemma4
// layer norms (input / post-attention / pre-FF / post-FF) + QK-norm + the dense MLP, plus
// the embedding, final norm and LM head (tied when lm_head.weight is absent). Each
// projection's byte span is validated against the Arch.
//
// SCOPE: dense bf16 only — MoE layers (router/experts), 4-bit-quant checkpoints and the
// per-layer-input tower are follow-up slices. And note: the four gemma4 norms it loads
// beyond pre-attn/pre-FF (QK-norm, post-attn, post-FF) are NOT yet consumed by the native
// decode — assembling them here readies the data; wiring them into the forward is the
// "gemma4 norm reconciliation" slice. So an assembled model runs at the right speed but is
// not output-faithful until that lands.
// normalizeGemma4Names strips the gemma4_unified wrapper prefix "language_model." when a real
// multimodal checkpoint uses it (its text weights live under language_model.model.*), so the
// assembler's bare "model.*" lookups match. Tensors NOT under the prefix — the vision/audio
// towers (embed_vision.*, embed_audio.*, vision_embedder.*) — are dropped, since the text
// assembler never needs them. A checkpoint already using bare names (a text-only export, or the
// synthetic test fixtures) has no such prefix and is returned unchanged.
func normalizeGemma4Names(tensors map[string]safetensors.Tensor) map[string]safetensors.Tensor {
	const prefix = "language_model."
	wrapped := false
	for k := range tensors {
		if core.HasPrefix(k, prefix) {
			wrapped = true
			break
		}
	}
	if !wrapped {
		return tensors
	}
	out := make(map[string]safetensors.Tensor, len(tensors))
	for k, v := range tensors {
		if core.HasPrefix(k, prefix) {
			out[k[len(prefix):]] = v
		}
	}
	return out
}

func AssembleGemma4BF16(tensors map[string]safetensors.Tensor, arch g4.Arch) (*Gemma4BF16, error) {
	tensors = normalizeGemma4Names(tensors)
	if arch.HasMoE() {
		return nil, core.NewError("native.AssembleGemma4BF16: MoE arch not supported yet (dense bf16 only)")
	}
	if len(arch.Layer) == 0 || arch.Hidden <= 0 || arch.Vocab <= 0 {
		return nil, core.NewError("native.AssembleGemma4BF16: arch must have layers, hidden and vocab")
	}
	dModel, headDim, dFF, vocab := arch.Hidden, arch.HeadDim, arch.FF, arch.Vocab
	qDim, kvDim := arch.Heads*headDim, arch.KVHeads*headDim

	// fetch a required (or optional) bf16 tensor of an exact element count; the first
	// failure is captured in ferr and short-circuits the rest.
	var ferr error
	fetch := func(name string, elems int, optional bool) []byte {
		if ferr != nil {
			return nil
		}
		t, ok := tensors[name]
		if !ok {
			if !optional {
				ferr = core.NewError("native.AssembleGemma4BF16: missing " + name)
			}
			return nil
		}
		if t.Dtype != "BF16" {
			ferr = core.NewError("native.AssembleGemma4BF16: " + name + " dtype " + t.Dtype + ", want BF16")
			return nil
		}
		if len(t.Data) != elems*bf16Size {
			ferr = core.NewError("native.AssembleGemma4BF16: " + name + " byte size mismatch vs arch dims")
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
		g.LMHead, g.Tied = g.Embed, true // gemma ties the output projection to the embedding
	}

	for i := range arch.Layer {
		p := core.Sprintf("model.layers.%d", i)
		l := &g.Layers[i]
		// attention: norms + Q/K/V/O (HF stores [out,in] row-major = MatVecBF16's layout)
		l.AttnNormW = fetch(p+".input_layernorm.weight", dModel, false)
		l.WQ = fetch(p+".self_attn.q_proj.weight", qDim*dModel, false)
		l.WK = fetch(p+".self_attn.k_proj.weight", kvDim*dModel, false)
		l.WV = fetch(p+".self_attn.v_proj.weight", kvDim*dModel, false)
		l.WO = fetch(p+".self_attn.o_proj.weight", dModel*qDim, false)
		// dense MLP: pre-FF norm + gate/up/down
		l.MLPNormW = fetch(p+".pre_feedforward_layernorm.weight", dModel, false)
		l.WGate = fetch(p+".mlp.gate_proj.weight", dFF*dModel, false)
		l.WUp = fetch(p+".mlp.up_proj.weight", dFF*dModel, false)
		l.WDown = fetch(p+".mlp.down_proj.weight", dModel*dFF, false)
		// gemma4 norms loaded but not yet consumed (see the type doc) — optional.
		l.QNormW = fetch(p+".self_attn.q_norm.weight", headDim, true)
		l.KNormW = fetch(p+".self_attn.k_norm.weight", headDim, true)
		l.PostAttnNormW = fetch(p+".post_attention_layernorm.weight", dModel, true)
		l.PostFFNormW = fetch(p+".post_feedforward_layernorm.weight", dModel, true)
	}
	if ferr != nil {
		return nil, ferr
	}
	return g, nil
}

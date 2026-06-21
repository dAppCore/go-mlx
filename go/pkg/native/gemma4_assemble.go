// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	core "dappco.re/go"
	"dappco.re/go/mlx/pkg/model"
	g4 "dappco.re/go/mlx/pkg/model/gemma4"
	"dappco.re/go/mlx/pkg/safetensors"
)

// Gemma4BF16 is a gemma4 model's bf16 weights mapped onto the native structs: the per-layer
// DecodeLayerWeights plus the model-level tensors (embedding, final norm, LM head, and the per-
// layer-input tower for E2B/E4B). LMHead aliases Embed when the checkpoint ties them (Tied).
type Gemma4BF16 struct {
	Layers    []DecodeLayerWeights
	Embed     []byte // [vocab × dModel] bf16
	FinalNorm []byte // [dModel] bf16 (model.norm.weight)
	LMHead    []byte // [vocab × dModel] bf16 (lm_head.weight, or Embed when tied)
	Tied      bool   // LMHead is the tied embedding (no separate lm_head.weight)
	// gemma4 per-layer-input tower (E2B/E4B), bf16: the per-layer embedding table, the model-side
	// projection, and its norm. Empty when the model has no PLE tower (12B/26B/31B).
	EmbedPerLayer      []byte // [pliVocab × (nLayers·pliDim)] bf16
	PerLayerModelProjW []byte // [(nLayers·pliDim) × dModel] bf16
	PerLayerProjNormW  []byte // [pliDim] bf16
}

// HasPLE reports whether this model carries the per-layer-input tower (E2B/E4B).
func (g *Gemma4BF16) HasPLE() bool { return g != nil && len(g.EmbedPerLayer) > 0 }

// normalizeGemma4Names strips the gemma4_unified wrapper prefix "language_model." when a real
// multimodal checkpoint uses it (its text weights live under language_model.model.*), so a bare
// "model.*" lookup matches. Tensors NOT under the prefix — the vision/audio towers — are dropped.
// A checkpoint already using bare names (a text-only export, or the synthetic test fixtures) has no
// such prefix and is returned unchanged. Used by the Mistral assembler (the gemma4 path now gets
// this for free inside gemma4.Assemble).
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

// AssembleGemma4BF16 maps a dense bf16 gemma4 checkpoint onto the native Gemma4BF16 through the SAME
// shared loader the quant path uses (gemma4.Assemble makes the per-weight decision + reads per-layer
// dims from SHAPES, so MatFormer FFN widths and KV-share are handled; loadedToBF16 takes the dense
// weight bytes + the PLE tower). The old hand-coded fetch walk — which assumed a FIXED FFN width and
// had no PLE — is gone (R8): it choked on E2B's per-layer FFN. Dense only — a MoE arch is rejected
// (the bf16 decode has no router path).
func AssembleGemma4BF16(tensors map[string]safetensors.Tensor, arch model.Arch) (*Gemma4BF16, error) {
	if arch.HasMoE() {
		return nil, core.NewError("native.AssembleGemma4BF16: MoE arch not supported (dense bf16 only)")
	}
	if len(arch.Layer) == 0 || arch.Hidden <= 0 || arch.Vocab <= 0 {
		return nil, core.NewError("native.AssembleGemma4BF16: arch must have layers, hidden and vocab")
	}
	m, err := g4.Assemble(tensors, arch)
	if err != nil {
		return nil, err
	}
	return loadedToBF16(m), nil
}

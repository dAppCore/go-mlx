// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import core "dappco.re/go"

// Gemma4TrimWrapperPrefix removes one common outer Gemma 4 checkpoint wrapper.
func Gemma4TrimWrapperPrefix(name string) (string, bool) {
	for _, prefix := range []string{
		"model.language_model.model.",
		"model.language_model.",
		"language_model.model.",
		"language_model.",
		"model.model.",
		"model.",
	} {
		if core.HasPrefix(name, prefix) {
			return core.TrimPrefix(name, prefix), true
		}
	}
	return name, false
}

// Gemma4UnwrappedWeightName removes repeated Gemma 4 checkpoint wrappers.
func Gemma4UnwrappedWeightName(name string) string {
	trimmed := name
	for {
		next, changed := Gemma4TrimWrapperPrefix(trimmed)
		if !changed {
			return trimmed
		}
		trimmed = next
	}
}

// Gemma4CanonicalWeightName canonicalises Gemma 4 text weight names.
//
// The returned bool is false for non-text helper tensors that the text loader
// intentionally ignores.
func Gemma4CanonicalWeightName(name string) (string, bool) {
	trimmed := Gemma4UnwrappedWeightName(name)

	if core.HasPrefix(trimmed, "vision_tower") ||
		core.HasPrefix(trimmed, "multi_modal_projector") ||
		core.HasPrefix(trimmed, "audio_tower") ||
		core.HasPrefix(trimmed, "embed_audio") ||
		core.HasPrefix(trimmed, "embed_vision") ||
		core.Contains(trimmed, "self_attn.rotary_emb") ||
		core.Contains(trimmed, "input_max") ||
		core.Contains(trimmed, "input_min") ||
		core.Contains(trimmed, "output_max") ||
		core.Contains(trimmed, "output_min") {
		return "", false
	}

	switch {
	case core.HasPrefix(trimmed, "layers."),
		core.HasPrefix(trimmed, "embed_tokens."),
		core.HasPrefix(trimmed, "embed_tokens_per_layer."),
		core.HasPrefix(trimmed, "norm."),
		core.HasPrefix(trimmed, "per_layer_model_projection."),
		core.HasPrefix(trimmed, "per_layer_projection_norm."):
		return "model." + trimmed, true
	default:
		return trimmed, true
	}
}

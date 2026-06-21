// SPDX-Licence-Identifier: EUPL-1.2

package gemma4

import (
	core "dappco.re/go"
	"dappco.re/go/mlx/pkg/model"
	"dappco.re/go/mlx/pkg/safetensors"
)

// infer.go — gemma4's weight-shape inference. The SELECTION is gemma4's (which attention-typed layer,
// the gemma4 weight names); the dim-from-shape READ is the engine's (pkg/model: WeightAny / InferHeadDim
// / InferOutFeaturesPerN), so other arches reuse the engine with their own names + patterns rather than
// re-rolling it.

// inferGemma4HeadDim reads the head dim of the first `target`-attention layer from its q_proj rows.
// gemma4 carries distinct head dims for sliding vs full (global) layers, so the caller resolves each by
// passing the matching attention type.
func inferGemma4HeadDim(weights map[string]safetensors.Tensor, layerTypes []string, numAttentionHeads int, target string) int {
	for i, layerType := range layerTypes {
		if layerType != target {
			continue
		}
		if hd := model.InferHeadDim(weights, core.Sprintf("model.layers.%d.self_attn.q_proj.weight", i), numAttentionHeads); hd > 0 {
			return hd
		}
	}
	return 0
}

// inferGemma4PerLayerInputSize reads the gemma4 per-layer-input width — the per-layer projection's
// out-features ÷ layer count (the E2B/E4B PLE tower).
func inferGemma4PerLayerInputSize(weights map[string]safetensors.Tensor, numHiddenLayers int) int {
	return model.InferOutFeaturesPerN(weights, "model.per_layer_model_projection.weight", numHiddenLayers)
}

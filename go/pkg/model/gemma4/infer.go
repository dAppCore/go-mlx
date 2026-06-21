// SPDX-Licence-Identifier: EUPL-1.2

package gemma4

import (
	core "dappco.re/go"
	"dappco.re/go/mlx/pkg/safetensors"
)

// infer.go is copied from pkg/metal/model/gemma4/weights.go (inferGemma4HeadDim /
// inferGemma4PerLayerInputSize): the read-the-dimension-FROM-THE-WEIGHT-SHAPE rule the loader applies
// when the config omits it, rather than guessing hidden/heads. The only adaptation is the weight
// source — safetensors.Tensor.Shape (a []int field) instead of metal.Array.Shape(), and int vs int32.

// weightAny returns the first of names present in the (normalised) tensor set — the neutral counterpart
// of metal's gemma4WeightAny.
func weightAny(weights map[string]safetensors.Tensor, names ...string) (safetensors.Tensor, bool) {
	for _, n := range names {
		if t, ok := weights[n]; ok {
			return t, true
		}
	}
	return safetensors.Tensor{}, false
}

func inferGemma4HeadDim(weights map[string]safetensors.Tensor, layerTypes []string, numAttentionHeads int, target string) int {
	for i, layerType := range layerTypes {
		if layerType != target {
			continue
		}
		if qProj, ok := weightAny(weights, core.Sprintf("model.layers.%d.self_attn.q_proj.weight", i)); ok {
			shape := qProj.Shape
			if len(shape) > 0 && numAttentionHeads > 0 && shape[0]%numAttentionHeads == 0 {
				return shape[0] / numAttentionHeads
			}
		}
	}
	return 0
}

func inferGemma4PerLayerInputSize(weights map[string]safetensors.Tensor, numHiddenLayers int) int {
	if numHiddenLayers <= 0 {
		return 0
	}
	if w, ok := weightAny(weights, "model.per_layer_model_projection.weight"); ok {
		shape := w.Shape
		if len(shape) >= 2 {
			outFeatures := 1
			for _, dim := range shape[:len(shape)-1] {
				outFeatures *= dim
			}
			if outFeatures%numHiddenLayers == 0 {
				return outFeatures / numHiddenLayers
			}
		}
	}
	return 0
}

// SPDX-Licence-Identifier: EUPL-1.2

package gemma4

import (
	core "dappco.re/go"
	"dappco.re/go/mlx/pkg/model"
	"dappco.re/go/mlx/pkg/safetensors"
)

// vision_assemble.go is the gemma4 vision tower's loader output: it gathers the SigLIP tower's weights
// off a checkpoint's tensors into a LoadedVision — byte views by role, the vision parallel of
// model.LoadedModel. A backend uploads these views to its device at encode time (native splits the
// loader (byte views) from the forward (device upload), unlike metal which couples the upload into the
// build), so this stays pure Go with no driver type. Lifted from buildGemma4VisionModel in
// pkg/metal/model/gemma4/vision_load.go, reusing the canonicalise + infer front + model.WeightAny.

// LoadedVisionLinear is one vision linear's weight + optional bias byte views (nil bias = none).
type LoadedVisionLinear struct {
	Weight, Bias []byte
}

// LoadedVisionLayer is one SigLIP encoder layer's weights: pre/post norms, QK-normed attention, gated MLP.
type LoadedVisionLayer struct {
	InputNorm, PostAttnNorm, PreFFNorm, PostFFNorm []byte
	Q, K, V, O                                     LoadedVisionLinear
	QNorm, KNorm                                   []byte
	Gate, Up, Down                                 LoadedVisionLinear
}

// LoadedVisionProjector is the vision-to-text projector's weights (a single projection, or fc1+fc2).
type LoadedVisionProjector struct {
	Projection, Linear1, Linear2 LoadedVisionLinear
}

// LoadedVision is the whole SigLIP tower + projector as byte views — the loader output a backend uploads.
type LoadedVision struct {
	PatchEmbedding     []byte
	PositionEmbeddings []byte
	PostLayernorm      []byte
	StdBias, StdScale  []byte
	Layers             []LoadedVisionLayer
	Projector          LoadedVisionProjector
	Cfg                *Gemma4VisionConfig
}

func visionWeight(weights map[string]safetensors.Tensor, names ...string) []byte {
	if t, ok := model.WeightAny(weights, names...); ok {
		return t.Data
	}
	return nil
}

// visionLinear gathers a vision linear's weight + bias from the first present prefix (.weight or
// .linear.weight, with the matching .bias / .linear.bias).
func visionLinear(weights map[string]safetensors.Tensor, prefixes ...string) LoadedVisionLinear {
	for _, p := range prefixes {
		if w := visionWeight(weights, p+".weight", p+".linear.weight"); w != nil {
			return LoadedVisionLinear{Weight: w, Bias: visionWeight(weights, p+".bias", p+".linear.bias")}
		}
	}
	return LoadedVisionLinear{}
}

// AssembleVision gathers the gemma4 vision tower (when the pack carries one) into a LoadedVision, with the
// config inferred from the weight shapes. Returns (nil, nil) when the pack is text-only / projector-only.
func AssembleVision(weights map[string]safetensors.Tensor, textCfg *Gemma4TextConfig) (*LoadedVision, error) {
	if !gemma4VisionShouldBuildEncoderTower(textCfg) || !HasVisionTowerWeights(weights) {
		return nil, nil
	}
	visionCfg := textCfg.VisionConfig
	if visionCfg == nil {
		visionCfg = &Gemma4VisionConfig{}
	}
	visionCfg = inferGemma4VisionConfig(weights, normalizeGemma4VisionConfig(visionCfg))

	patch := visionWeight(weights,
		"patch_embedder.input_proj.weight", "patch_embedder.input_proj.linear.weight",
		"embeddings.patch_embedding.weight", "patch_embedding.weight")
	if patch == nil {
		return nil, core.E("gemma4.AssembleVision", "missing patch embedding weight", nil)
	}

	v := &LoadedVision{
		PatchEmbedding:     patch,
		PositionEmbeddings: visionWeight(weights, "patch_embedder.position_embedding_table", "embeddings.position_embedding.weight"),
		PostLayernorm:      visionWeight(weights, "post_layernorm.weight", "post_layer_norm.weight", "encoder.post_layernorm.weight", "vision_model.post_layernorm.weight"),
		StdBias:            visionWeight(weights, "std_bias"),
		StdScale:           visionWeight(weights, "std_scale"),
		Layers:             make([]LoadedVisionLayer, int(visionCfg.NumHiddenLayers)),
		Cfg:                visionCfg,
	}
	for i := range v.Layers {
		p := core.Sprintf("encoder.layers.%d", i)
		L := &v.Layers[i]
		L.InputNorm = visionWeight(weights, p+".input_layernorm.weight", p+".layer_norm1.weight")
		L.PostAttnNorm = visionWeight(weights, p+".post_attention_layernorm.weight", p+".post_attention_layernorm.linear.weight")
		L.PreFFNorm = visionWeight(weights, p+".pre_feedforward_layernorm.weight", p+".layer_norm2.weight")
		L.PostFFNorm = visionWeight(weights, p+".post_feedforward_layernorm.weight", p+".post_feedforward_layernorm.linear.weight")
		L.Q = visionLinear(weights, p+".self_attn.q_proj", p+".attention.q_proj")
		L.K = visionLinear(weights, p+".self_attn.k_proj", p+".attention.k_proj")
		L.V = visionLinear(weights, p+".self_attn.v_proj", p+".attention.v_proj")
		L.O = visionLinear(weights, p+".self_attn.o_proj", p+".attention.out_proj", p+".attention.o_proj")
		L.QNorm = visionWeight(weights, p+".self_attn.q_norm.weight")
		L.KNorm = visionWeight(weights, p+".self_attn.k_norm.weight")
		L.Gate = visionLinear(weights, p+".mlp.gate_proj", p+".mlp.fc1")
		L.Up = visionLinear(weights, p+".mlp.up_proj")
		L.Down = visionLinear(weights, p+".mlp.down_proj", p+".mlp.fc2")
		if err := validateLoadedVisionLayer(L, i); err != nil {
			return nil, err
		}
	}
	v.Projector = LoadedVisionProjector{
		Projection: visionLinear(weights, "embed_vision.embedding_projection", "multi_modal_projector.embedding_projection", "multi_modal_projector.proj", "multi_modal_projector"),
		Linear1:    visionLinear(weights, "multi_modal_projector.linear_1", "multi_modal_projector.fc1"),
		Linear2:    visionLinear(weights, "multi_modal_projector.linear_2", "multi_modal_projector.fc2"),
	}
	return v, nil
}

// validateLoadedVisionLayer fails loud on a missing required weight in an encoder layer — a malformed
// vision pack, surfaced at load rather than as a nil view deep in the encode.
func validateLoadedVisionLayer(L *LoadedVisionLayer, idx int) error {
	for _, c := range []struct {
		b    []byte
		name string
	}{
		{L.InputNorm, "input norm"}, {L.PostAttnNorm, "post-attn norm"}, {L.PreFFNorm, "pre-ff norm"}, {L.PostFFNorm, "post-ff norm"},
		{L.Q.Weight, "q proj"}, {L.K.Weight, "k proj"}, {L.V.Weight, "v proj"}, {L.O.Weight, "o proj"},
		{L.QNorm, "q norm"}, {L.KNorm, "k norm"},
		{L.Gate.Weight, "gate proj"}, {L.Up.Weight, "up proj"}, {L.Down.Weight, "down proj"},
	} {
		if len(c.b) == 0 {
			return core.E("gemma4.AssembleVision", core.Sprintf("encoder layer %d missing %s", idx, c.name), nil)
		}
	}
	return nil
}

// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma4

import (
	"math"

	core "dappco.re/go"
	"dappco.re/go/mlx/pkg/metal"
)

// Vision tower weight loading, sanitisation, inference, and component build.

func sanitizeGemma4VisionWeights(raw map[string]*metal.Array) map[string]*metal.Array {
	vision := make(map[string]*metal.Array)
	for name, arr := range raw {
		canonical, ok := canonicalGemma4VisionWeightName(name)
		if !ok {
			continue
		}
		if prev, exists := vision[canonical]; exists && prev != arr {
			metal.Free(prev)
		}
		vision[canonical] = arr
		delete(raw, name)
	}
	return vision
}

func canonicalGemma4VisionWeightName(name string) (string, bool) {
	trimmed := name
	for {
		next, changed := trimGemma4WrapperPrefix(trimmed)
		if !changed {
			break
		}
		trimmed = next
	}

	for _, prefix := range []string{
		"vision_tower.",
		"vision_model.",
	} {
		if core.HasPrefix(trimmed, prefix) {
			return core.TrimPrefix(trimmed, prefix), true
		}
	}
	for _, prefix := range []string{
		"multi_modal_projector.",
		"embed_vision.",
	} {
		if core.HasPrefix(trimmed, prefix) {
			return trimmed, true
		}
	}
	return "", false
}

func hasGemma4VisionTowerWeights(weights map[string]*metal.Array) bool {
	return gemma4VisionWeightAny(weights,
		"patch_embedder.input_proj.weight",
		"patch_embedder.input_proj.linear.weight",
		"embeddings.patch_embedding.weight",
		"patch_embedding.weight",
	) != nil
}

func hasGemma4VisionProjectionWeights(weights map[string]*metal.Array) bool {
	return gemma4VisionWeightAny(weights,
		"embed_vision.embedding_projection.weight",
		"embed_vision.embedding_projection.linear.weight",
		"multi_modal_projector.embedding_projection.weight",
		"multi_modal_projector.embedding_projection.linear.weight",
		"multi_modal_projector.proj.weight",
		"multi_modal_projector.weight",
	) != nil
}

func buildGemma4VisionComponents(cfg *Gemma4TextConfig, weights map[string]*metal.Array) (*Gemma4VisionModel, *Gemma4MultiModalProjector, error) {
	buildTower := gemma4VisionShouldBuildEncoderTower(cfg) && hasGemma4VisionTowerWeights(weights)
	if !buildTower {
		if !hasGemma4VisionProjectionWeights(weights) {
			gemma4FreeUnusedWeights(weights, map[*metal.Array]struct{}{})
			return nil, nil, nil
		}
		visionCfg := cfg.VisionConfig
		if visionCfg == nil {
			visionCfg = &Gemma4VisionConfig{}
		}
		visionCfg = normalizeGemma4VisionConfig(visionCfg)
		projector := buildGemma4MultiModalProjector(cfg, visionCfg, weights)
		retained := gemma4VisionRetainedWeights(nil, projector)
		gemma4FreeUnusedWeights(weights, retained)
		gemma4MaterializeRetainedWeights(retained, nil)
		return nil, projector, nil
	}

	visionCfg := cfg.VisionConfig
	if visionCfg == nil {
		visionCfg = &Gemma4VisionConfig{}
	}
	visionCfg = inferGemma4VisionConfig(weights, normalizeGemma4VisionConfig(visionCfg))

	vision, err := buildGemma4VisionModel(visionCfg, weights)
	if err != nil {
		gemma4FreeUnusedWeights(weights, map[*metal.Array]struct{}{})
		return nil, nil, err
	}
	projector := buildGemma4MultiModalProjector(cfg, visionCfg, weights)

	retained := gemma4VisionRetainedWeights(vision, projector)
	gemma4FreeUnusedWeights(weights, retained)
	gemma4MaterializeRetainedWeights(retained, nil)
	return vision, projector, nil
}

func gemma4VisionShouldBuildEncoderTower(cfg *Gemma4TextConfig) bool {
	if cfg == nil {
		return true
	}
	if cfg.ModelType == "gemma4_unified" || cfg.ModelType == "gemma4_unified_text" {
		return false
	}
	if cfg.VisionConfig != nil && cfg.VisionConfig.ModelType == "gemma4_unified_vision" {
		return false
	}
	return true
}

func inferGemma4VisionConfig(weights map[string]*metal.Array, cfg *Gemma4VisionConfig) *Gemma4VisionConfig {
	if cfg == nil {
		cfg = &Gemma4VisionConfig{}
	}
	if w := gemma4VisionWeightAny(weights,
		"patch_embedder.input_proj.weight",
		"patch_embedder.input_proj.linear.weight",
		"embeddings.patch_embedding.weight",
		"patch_embedding.weight",
	); w != nil {
		shape := w.Shape()
		if len(shape) > 0 && shape[0] > 0 {
			cfg.HiddenSize = shape[0]
		}
		patchDim := int32(0)
		switch len(shape) {
		case 2:
			patchDim = shape[1]
		case 4:
			patchDim = shape[1] * shape[2] * shape[3]
		}
		channels := cfg.NumChannels
		if channels <= 0 {
			channels = 3
		}
		if patchDim > 0 && patchDim%channels == 0 {
			patch := int32(math.Round(math.Sqrt(float64(patchDim / channels))))
			if patch > 0 && channels*patch*patch == patchDim {
				cfg.PatchSize = patch
			}
		}
	}
	if cfg.HiddenSize > 0 && cfg.NumAttentionHeads > 0 {
		cfg.HeadDim = cfg.HiddenSize / cfg.NumAttentionHeads
	}
	if cfg.NumKeyValueHeads == 0 {
		cfg.NumKeyValueHeads = cfg.NumAttentionHeads
	}
	for i := int32(0); ; i++ {
		prefix := core.Sprintf("encoder.layers.%d", i)
		if gemma4VisionWeightAny(weights,
			prefix+".self_attn.q_proj.weight",
			prefix+".self_attn.q_proj.linear.weight",
			prefix+".attention.q_proj.weight",
			prefix+".attention.q_proj.linear.weight",
		) == nil {
			if i > 0 {
				cfg.NumHiddenLayers = i
			}
			break
		}
	}
	return normalizeGemma4VisionConfig(cfg)
}

func gemma4VisionWeightAny(weights map[string]*metal.Array, names ...string) *metal.Array {
	for _, name := range names {
		if arr := weights[name]; arr != nil {
			return arr
		}
	}
	return nil
}

func gemma4VisionLinear(weights map[string]*metal.Array, prefixes ...string) *metal.Linear {
	for _, prefix := range prefixes {
		weight := gemma4VisionWeightAny(weights, prefix+".weight", prefix+".linear.weight")
		if weight == nil {
			continue
		}
		bias := gemma4VisionWeightAny(weights, prefix+".bias", prefix+".linear.bias")
		return metal.NewLinear(weight, bias)
	}
	return nil
}

func gemma4VisionNorm(weights map[string]*metal.Array, hiddenSize int32, names ...string) *metal.RMSNormModule {
	if weight := gemma4VisionWeightAny(weights, names...); weight != nil {
		return &metal.RMSNormModule{Weight: weight}
	}
	return &metal.RMSNormModule{Weight: gemma4Ones([]int32{hiddenSize})}
}

func normalizeGemma4PatchProjection(weight *metal.Array, cfg *Gemma4VisionConfig) (*metal.Array, *metal.Array, bool) {
	if weight == nil {
		return nil, nil, false
	}
	channels := cfg.NumChannels
	if channels <= 0 {
		channels = 3
	}
	shape := weight.Shape()
	if len(shape) == 2 {
		conv := metal.Reshape(weight, shape[0], cfg.PatchSize, cfg.PatchSize, channels)
		return weight, conv, true
	}
	if len(shape) != 4 {
		return weight, nil, true
	}
	var conv *metal.Array
	if shape[3] == channels {
		conv = weight
	} else if shape[1] == channels {
		conv = metal.Transpose(weight, 0, 2, 3, 1)
	} else {
		conv = weight
	}
	linear := metal.Reshape(conv, shape[0], shape[1]*shape[2]*shape[3])
	return linear, conv, true
}

func buildGemma4VisionModel(cfg *Gemma4VisionConfig, weights map[string]*metal.Array) (*Gemma4VisionModel, error) {
	patchWeight := gemma4VisionWeightAny(weights,
		"patch_embedder.input_proj.weight",
		"patch_embedder.input_proj.linear.weight",
		"embeddings.patch_embedding.weight",
		"patch_embedding.weight",
	)
	inputWeight, convWeight, ok := normalizeGemma4PatchProjection(patchWeight, cfg)
	if !ok || inputWeight == nil {
		return nil, core.E("gemma4.vision", "missing patch embedding weight", nil)
	}

	var postLayernorm *metal.RMSNormModule
	if weight := gemma4VisionWeightAny(weights,
		"post_layernorm.weight",
		"post_layer_norm.weight",
		"encoder.post_layernorm.weight",
		"vision_model.post_layernorm.weight",
	); weight != nil {
		postLayernorm = &metal.RMSNormModule{Weight: weight}
	}

	vision := &Gemma4VisionModel{
		PatchEmbedder: &Gemma4VisionPatchEmbedder{
			InputProj:              metal.NewLinear(inputWeight, nil),
			PatchConvWeight:        convWeight,
			PositionEmbeddingTable: gemma4VisionWeightAny(weights, "patch_embedder.position_embedding_table", "embeddings.position_embedding.weight"),
			PatchSize:              cfg.PatchSize,
			NumChannels:            cfg.NumChannels,
			PoolingKernelSize:      cfg.PoolingKernelSize,
			PositionEmbeddingSize:  cfg.PositionEmbeddingSize,
			HiddenSize:             cfg.HiddenSize,
		},
		Encoder: &Gemma4VisionEncoder{
			Layers: make([]*Gemma4VisionEncoderLayer, cfg.NumHiddenLayers),
			Cfg:    cfg,
		},
		Pooler: &Gemma4VisionPooler{
			HiddenSize:        cfg.HiddenSize,
			PoolingKernelSize: cfg.PoolingKernelSize,
			EmbeddingScale:    float32(math.Sqrt(float64(cfg.HiddenSize))),
		},
		PostLayernorm: postLayernorm,
		StdBias:       gemma4VisionWeightAny(weights, "std_bias"),
		StdScale:      gemma4VisionWeightAny(weights, "std_scale"),
		Cfg:           cfg,
	}
	vision.PatchEmbedding = vision.PatchEmbedder.InputProj
	vision.PositionEmbeddings = vision.PatchEmbedder.PositionEmbeddingTable
	vision.EncoderLayers = vision.Encoder.Layers

	for i := int32(0); i < cfg.NumHiddenLayers; i++ {
		prefix := core.Sprintf("encoder.layers.%d", i)
		layer := &Gemma4VisionEncoderLayer{
			InputNorm: gemma4VisionNorm(weights, cfg.HiddenSize,
				prefix+".input_layernorm.weight",
				prefix+".layer_norm1.weight",
			),
			PostAttnNorm: gemma4VisionNorm(weights, cfg.HiddenSize,
				prefix+".post_attention_layernorm.weight",
				prefix+".post_attention_layernorm.linear.weight",
			),
			PreFFNorm: gemma4VisionNorm(weights, cfg.HiddenSize,
				prefix+".pre_feedforward_layernorm.weight",
				prefix+".layer_norm2.weight",
			),
			PostFFNorm: gemma4VisionNorm(weights, cfg.HiddenSize,
				prefix+".post_feedforward_layernorm.weight",
				prefix+".post_feedforward_layernorm.linear.weight",
			),
			Attention: &Gemma4VisionAttention{
				QProj: gemma4VisionLinear(weights,
					prefix+".self_attn.q_proj",
					prefix+".attention.q_proj",
				),
				KProj: gemma4VisionLinear(weights,
					prefix+".self_attn.k_proj",
					prefix+".attention.k_proj",
				),
				VProj: gemma4VisionLinear(weights,
					prefix+".self_attn.v_proj",
					prefix+".attention.v_proj",
				),
				OProj: gemma4VisionLinear(weights,
					prefix+".self_attn.o_proj",
					prefix+".attention.out_proj",
					prefix+".attention.o_proj",
				),
				QNorm: gemma4VisionNorm(weights, cfg.HeadDim, prefix+".self_attn.q_norm.weight"),
				KNorm: gemma4VisionNorm(weights, cfg.HeadDim, prefix+".self_attn.k_norm.weight"),

				HeadDim:   cfg.HeadDim,
				NHeads:    cfg.NumAttentionHeads,
				NKVHeads:  cfg.NumKeyValueHeads,
				RopeBase:  cfg.RopeParameters.RopeTheta,
				Attention: 1.0,
			},
			MLP: &Gemma4VisionMLP{
				GateProj: gemma4VisionLinear(weights, prefix+".mlp.gate_proj", prefix+".mlp.fc1"),
				UpProj:   gemma4VisionLinear(weights, prefix+".mlp.up_proj"),
				DownProj: gemma4VisionLinear(weights, prefix+".mlp.down_proj", prefix+".mlp.fc2"),
			},
		}
		if err := validateGemma4VisionEncoderLayer(layer, i); err != nil {
			return nil, err
		}
		vision.Encoder.Layers[i] = layer
	}

	return vision, nil
}

func validateGemma4VisionLinear(linear *metal.Linear, name string) error {
	if linear == nil || linear.Weight == nil {
		return core.E("gemma4.vision", "missing "+name, nil)
	}
	return nil
}

func validateGemma4VisionNorm(norm *metal.RMSNormModule, name string) error {
	if norm == nil || norm.Weight == nil {
		return core.E("gemma4.vision", "missing "+name, nil)
	}
	return nil
}

func validateGemma4VisionEncoderLayer(layer *Gemma4VisionEncoderLayer, idx int32) error {
	prefix := core.Sprintf("encoder layer %d ", idx)
	if err := validateGemma4VisionNorm(layer.InputNorm, prefix+"input norm"); err != nil {
		return err
	}
	if err := validateGemma4VisionNorm(layer.PostAttnNorm, prefix+"post-attention norm"); err != nil {
		return err
	}
	if err := validateGemma4VisionNorm(layer.PreFFNorm, prefix+"pre-feedforward norm"); err != nil {
		return err
	}
	if err := validateGemma4VisionNorm(layer.PostFFNorm, prefix+"post-feedforward norm"); err != nil {
		return err
	}
	if layer.Attention == nil {
		return core.E("gemma4.vision", "missing "+prefix+"attention", nil)
	}
	if err := validateGemma4VisionLinear(layer.Attention.QProj, prefix+"q projection"); err != nil {
		return err
	}
	if err := validateGemma4VisionLinear(layer.Attention.KProj, prefix+"k projection"); err != nil {
		return err
	}
	if err := validateGemma4VisionLinear(layer.Attention.VProj, prefix+"v projection"); err != nil {
		return err
	}
	if err := validateGemma4VisionLinear(layer.Attention.OProj, prefix+"output projection"); err != nil {
		return err
	}
	if err := validateGemma4VisionNorm(layer.Attention.QNorm, prefix+"q norm"); err != nil {
		return err
	}
	if err := validateGemma4VisionNorm(layer.Attention.KNorm, prefix+"k norm"); err != nil {
		return err
	}
	if layer.MLP == nil {
		return core.E("gemma4.vision", "missing "+prefix+"mlp", nil)
	}
	if err := validateGemma4VisionLinear(layer.MLP.GateProj, prefix+"gate projection"); err != nil {
		return err
	}
	if err := validateGemma4VisionLinear(layer.MLP.UpProj, prefix+"up projection"); err != nil {
		return err
	}
	if err := validateGemma4VisionLinear(layer.MLP.DownProj, prefix+"down projection"); err != nil {
		return err
	}
	return nil
}

func buildGemma4MultiModalProjector(textCfg *Gemma4TextConfig, visionCfg *Gemma4VisionConfig, weights map[string]*metal.Array) *Gemma4MultiModalProjector {
	var quantization *metal.QuantizationConfig
	if textCfg != nil {
		quantization = textCfg.Quantization
	}
	projection := gemma4Linear(weights, "embed_vision.embedding_projection", quantization)
	if projection == nil {
		projection = gemma4Linear(weights, "multi_modal_projector.embedding_projection", quantization)
	}
	if projection == nil {
		projection = gemma4Linear(weights, "multi_modal_projector.proj", quantization)
	}
	if projection == nil {
		projection = gemma4Linear(weights, "multi_modal_projector", quantization)
	}
	projector := &Gemma4MultiModalProjector{
		Projection: projection,
		Linear1: gemma4VisionLinear(weights,
			"multi_modal_projector.linear_1",
			"multi_modal_projector.fc1",
		),
		Linear2: gemma4VisionLinear(weights,
			"multi_modal_projector.linear_2",
			"multi_modal_projector.fc2",
		),
		Eps: visionCfg.RMSNormEps,
	}
	ready := projector.Projection != nil || (projector.Linear1 != nil && projector.Linear2 != nil)
	if visionCfg.HiddenSize != textCfg.HiddenSize && !ready {
		return nil
	}
	return projector
}

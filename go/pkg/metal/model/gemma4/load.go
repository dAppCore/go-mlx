// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma4

import (
	"math"

	core "dappco.re/go"
	coreio "dappco.re/go/io"
	"dappco.re/go/mlx/pkg/metal"
)

func LoadGemma4(modelPath string) (*Gemma4Model, error) {
	root := metal.ResolveModelRoot(modelPath)
	str, err := coreio.Local.Read(core.JoinPath(root, "config.json"))
	if err != nil {
		return nil, core.E("gemma4.LoadGemma4", "load config", err)
	}
	data := []byte(str)

	cfg, err := parseGemma4Config(data)
	if err != nil {
		return nil, core.E("gemma4.LoadGemma4", "parse config", err)
	}
	if err := validateGemma4QuantizationConfig(cfg.Quantization); err != nil {
		return nil, core.E("gemma4.LoadGemma4", "validate quantization", err)
	}

	tok, err := metal.LoadTokenizer(core.JoinPath(root, "tokenizer.json"))
	if err != nil {
		return nil, core.E("gemma4.LoadGemma4", "load tokenizer", err)
	}

	rawWeights, err := metal.LoadModelWeights(modelPath)
	if err != nil {
		return nil, core.E("gemma4.LoadGemma4", "load weights", err)
	}
	visionWeights := sanitizeGemma4VisionWeights(rawWeights)
	audioWeights := sanitizeGemma4AudioWeights(rawWeights)
	weights := sanitizeGemma4Weights(rawWeights)

	if inferred := inferGemma4HeadDim(weights, cfg.LayerTypes, cfg.NumAttentionHeads, "sliding_attention"); inferred > 0 {
		cfg.HeadDim = inferred
	}
	if inferred := inferGemma4HeadDim(weights, cfg.LayerTypes, cfg.NumAttentionHeads, "full_attention"); inferred > 0 {
		cfg.GlobalHeadDim = inferred
	}
	if cfg.HeadDim == 0 && cfg.HiddenSize > 0 && cfg.NumAttentionHeads > 0 {
		cfg.HeadDim = cfg.HiddenSize / cfg.NumAttentionHeads
	}
	// GlobalHeadDim stays 0 when the model neither declares nor distinguishes a
	// full-attention head dim — every consumer falls back to HeadDim for that
	// case (layer build + assistant), so 0 is correct, never a guessed 512.
	//
	// vocab_size is the row count of the token-embedding weight — read it from
	// the tensor when the config did not declare it, never a hardcoded literal.
	if cfg.VocabSize == 0 {
		if w := gemma4WeightAny(weights, "model.embed_tokens.weight", "model.embed_tokens"); w != nil {
			if shape := w.Shape(); len(shape) > 0 && shape[0] > 0 {
				cfg.VocabSize = shape[0]
			}
		}
	}
	if cfg.VocabSizePerLayerInput == 0 {
		cfg.VocabSizePerLayerInput = cfg.VocabSize
	}

	if inferred := inferGemma4PerLayerInputSize(weights, cfg.NumHiddenLayers); inferred > 0 {
		cfg.HiddenSizePerLayerInput = inferred
	}
	if cfg.HiddenSizePerLayerInput > 0 {
		if gemma4WeightAny(weights, "model.embed_tokens_per_layer.weight") == nil ||
			gemma4WeightAny(weights, "model.per_layer_model_projection.weight") == nil ||
			gemma4WeightAny(weights, "model.per_layer_projection_norm.weight") == nil {
			cfg.HiddenSizePerLayerInput = 0
		}
	}
	// Re-cache once HiddenSizePerLayerInput is finalised against the
	// loaded weights — keeps cfg.PerLayerInputEmbeddingScale in sync.
	gemma4FinaliseEmbeddingScales(cfg)

	modelType := cfg.ModelType
	if modelType == "" {
		modelType = "gemma4_text"
	}

	embed := &metal.Embedding{Weight: gemma4WeightAny(weights, "model.embed_tokens.weight")}
	if embedScales := gemma4WeightAny(weights, "model.embed_tokens.scales"); embedScales != nil {
		embed.Scales = embedScales
		embed.Biases = gemma4WeightAny(weights, "model.embed_tokens.biases")
		if q := gemma4QuantForWeight("model.embed_tokens", cfg.Quantization, embed.Weight, embedScales); q != nil {
			embed.GroupSize = q.GroupSize
			embed.Bits = q.Bits
			embed.QuantizationMode = q.Mode
		}
	}

	var embedPerLayer *metal.Embedding
	if cfg.HiddenSizePerLayerInput > 0 {
		embedPerLayer = &metal.Embedding{Weight: gemma4WeightAny(weights, "model.embed_tokens_per_layer.weight")}
		if scales := gemma4WeightAny(weights, "model.embed_tokens_per_layer.scales"); scales != nil {
			embedPerLayer.Scales = scales
			embedPerLayer.Biases = gemma4WeightAny(weights, "model.embed_tokens_per_layer.biases")
			if q := gemma4QuantForWeight("model.embed_tokens_per_layer", cfg.Quantization, embedPerLayer.Weight, scales); q != nil {
				embedPerLayer.GroupSize = q.GroupSize
				embedPerLayer.Bits = q.Bits
				embedPerLayer.QuantizationMode = q.Mode
			}
		}
	}

	m := &Gemma4Model{
		EmbedTokens:         embed,
		EmbedTokensPerLayer: embedPerLayer,
		Layers:              make([]*Gemma4DecoderLayer, cfg.NumHiddenLayers),
		Norm:                &metal.RMSNormModule{Weight: gemma4WeightAny(weights, "model.norm.weight")},
		Tok:                 tok,
		Cfg:                 cfg,
		modelType:           modelType,
	}
	loadSucceeded := false
	defer func() {
		if loadSucceeded {
			return
		}
		retained := gemma4RetainedWeights(m)
		gemma4FreeUnusedWeights(weights, retained)
		gemma4FreeUnusedWeights(visionWeights, retained)
		gemma4FreeUnusedWeights(audioWeights, retained)
		closeGemma4(m)
		metal.ClearCache()
	}()

	if cfg.HiddenSizePerLayerInput > 0 {
		m.PerLayerModelProj = gemma4Linear(weights, "model.per_layer_model_projection", cfg.Quantization)
		m.PerLayerProjNorm = &metal.RMSNormModule{Weight: gemma4WeightAny(weights, "model.per_layer_projection_norm.weight")}
	}

	firstShared := max(cfg.NumHiddenLayers-cfg.NumKVSharedLayers, 0)
	feats := FeaturesOf(cfg)
	for i := int32(0); i < cfg.NumHiddenLayers; i++ {
		prefix := core.Sprintf("model.layers.%d", i)
		layerType := cfg.LayerTypes[i]
		isSliding := layerType == "sliding_attention"
		headDim := cfg.HeadDim
		if !isSliding && cfg.GlobalHeadDim > 0 {
			headDim = cfg.GlobalHeadDim
		}
		nkvHeads := cfg.NumKeyValueHeads
		useKEqV := cfg.AttentionKEqV && !isSliding
		if useKEqV && cfg.NumGlobalKeyValueHeads != nil {
			nkvHeads = *cfg.NumGlobalKeyValueHeads
		}

		ropeParams := cfg.RopeParameters[layerType]
		rotatedDims := gemma4RotatedDims(headDim, ropeParams)
		var ropeFreqs *metal.Array
		if ropeParams.RopeType == "proportional" {
			factor := ropeParams.Factor
			if factor == 0 {
				factor = 1
			}
			ropeFreqs = gemma4ProportionalFreqs(headDim, rotatedDims, float32(ropeParams.RopeTheta), factor)
		}

		layer := &Gemma4DecoderLayer{
			InputNorm:    &metal.RMSNormModule{Weight: gemma4WeightAny(weights, prefix+".input_layernorm.weight")},
			PostAttnNorm: &metal.RMSNormModule{Weight: gemma4WeightAny(weights, prefix+".post_attention_layernorm.weight")},
			PreFFNorm:    &metal.RMSNormModule{Weight: gemma4WeightAny(weights, prefix+".pre_feedforward_layernorm.weight")},
			PostFFNorm:   &metal.RMSNormModule{Weight: gemma4WeightAny(weights, prefix+".post_feedforward_layernorm.weight")},
			Attention: &Gemma4Attention{
				QProj:          gemma4Linear(weights, prefix+".self_attn.q_proj", cfg.Quantization),
				KProj:          gemma4Linear(weights, prefix+".self_attn.k_proj", cfg.Quantization),
				VProj:          gemma4Linear(weights, prefix+".self_attn.v_proj", cfg.Quantization),
				OProj:          gemma4Linear(weights, prefix+".self_attn.o_proj", cfg.Quantization),
				QNorm:          &metal.RMSNormModule{Weight: gemma4WeightAny(weights, prefix+".self_attn.q_norm.weight")},
				KNorm:          &metal.RMSNormModule{Weight: gemma4WeightAny(weights, prefix+".self_attn.k_norm.weight")},
				VNorm:          &metal.RMSNormModule{},
				HeadDim:        headDim,
				NKVHeads:       nkvHeads,
				UseKEqV:        useKEqV,
				Scale:          gemma4AttentionScale(headDim),
				RopeBase:       float32(ropeParams.RopeTheta),
				RopeRotatedDim: rotatedDims,
				RopeFreqs:      ropeFreqs,
			},
			MLP: &metal.MLP{
				GateProj: gemma4Linear(weights, prefix+".mlp.gate_proj", cfg.Quantization),
				UpProj:   gemma4Linear(weights, prefix+".mlp.up_proj", cfg.Quantization),
				DownProj: gemma4Linear(weights, prefix+".mlp.down_proj", cfg.Quantization),
			},
			LayerScalar:   gemma4WeightAny(weights, prefix+".layer_scalar", prefix+".layer_scalar.weight"),
			LayerType:     layerType,
			IsSliding:     isSliding,
			DoubleWideMLP: cfg.UseDoubleWideMLP && cfg.NumKVSharedLayers > 0 && i >= firstShared,
			LayerIdx:      i,
			EnableMoE:     feats.Mixture,
		}
		if layer.LayerScalar == nil {
			layer.LayerScalar = gemma4Ones([]int32{1})
		}
		if useKEqV {
			layer.Attention.VProj = nil
		}

		if feats.Mixture {
			routerScale := gemma4WeightAny(weights, prefix+".router.scale", prefix+".router.scale.weight")
			if routerScale == nil {
				routerScale = gemma4Ones([]int32{cfg.HiddenSize})
			}
			perExpertScale := gemma4WeightAny(weights, prefix+".router.per_expert_scale", prefix+".router.per_expert_scale.weight")
			if perExpertScale == nil && feats.NumExperts > 0 {
				perExpertScale = gemma4Ones([]int32{int32(feats.NumExperts)})
			}
			layer.Router = &Gemma4Router{
				Proj:           gemma4Linear(weights, prefix+".router.proj", cfg.Quantization),
				Scale:          routerScale,
				PerExpertScale: perExpertScale,
				RootSize:       float32(math.Pow(float64(cfg.HiddenSize), -0.5)),
				TopK:           int32(feats.TopKExperts),
				Eps:            cfg.RMSNormEps,
			}
			layer.Experts = &Gemma4Experts{
				GateUpProj: gemma4SwitchLinear(weights, cfg.Quantization,
					prefix+".experts.switch_glu.gate_up_proj",
					prefix+".experts.gate_up_proj",
				),
				GateProj: gemma4SwitchLinear(weights, cfg.Quantization,
					prefix+".experts.switch_glu.gate_proj",
					prefix+".experts.gate_proj",
				),
				UpProj: gemma4SwitchLinear(weights, cfg.Quantization,
					prefix+".experts.switch_glu.up_proj",
					prefix+".experts.up_proj",
				),
				DownProj: gemma4SwitchLinear(weights, cfg.Quantization,
					prefix+".experts.switch_glu.down_proj",
					prefix+".experts.down_proj",
				),
			}
			layer.PreFFNorm2 = &metal.RMSNormModule{Weight: gemma4WeightAny(weights, prefix+".pre_feedforward_layernorm_2.weight")}
			layer.PostFFNorm1 = &metal.RMSNormModule{Weight: gemma4WeightAny(weights, prefix+".post_feedforward_layernorm_1.weight")}
			layer.PostFFNorm2 = &metal.RMSNormModule{Weight: gemma4WeightAny(weights, prefix+".post_feedforward_layernorm_2.weight")}
		}

		if cfg.HiddenSizePerLayerInput > 0 {
			layer.PerLayerInputGate = gemma4Linear(weights, prefix+".per_layer_input_gate", cfg.Quantization)
			layer.PerLayerProjection = gemma4Linear(weights, prefix+".per_layer_projection", cfg.Quantization)
			layer.PostPerLayerInputNorm = &metal.RMSNormModule{Weight: gemma4WeightAny(weights, prefix+".post_per_layer_input_norm.weight")}
			if layer.PerLayerInputGate == nil || layer.PerLayerProjection == nil || layer.PostPerLayerInputNorm.Weight == nil {
				layer.PerLayerInputGate = nil
				layer.PerLayerProjection = nil
				layer.PostPerLayerInputNorm = nil
			}
		}

		m.Layers[i] = layer
	}

	m.Output, err = gemma4OutputLinear(weights, cfg, m.EmbedTokens)
	if err != nil {
		return nil, core.E("gemma4.LoadGemma4", "build output projection", err)
	}

	if len(visionWeights) > 0 {
		m.VisionTower, m.MultiModalProjector, err = buildGemma4VisionComponents(cfg, visionWeights)
		if err != nil {
			return nil, core.E("gemma4.LoadGemma4", "build vision tower", err)
		}
	}
	if len(audioWeights) > 0 {
		m.AudioProjector = buildGemma4AudioProjector(cfg, audioWeights)
	}

	m.PreviousKVs, m.CacheIndexByLayer = buildGemma4CacheLayout(m.Layers, cfg.NumKVSharedLayers)
	retainedWeights := gemma4RetainedWeights(m)
	lazyWeights := gemma4LazyRetainedWeights(m)
	gemma4FreeUnusedWeights(weights, retainedWeights)
	gemma4FreeUnusedWeights(audioWeights, retainedWeights)
	gemma4MaterializeRetainedWeights(retainedWeights, lazyWeights)
	precomputeGemma4ScaledWeights(m)

	loadSucceeded = true
	return m, nil
}

func init() {
	metal.RegisterModelLoader("gemma4_text", func(p string, _ []byte) (metal.InternalModel, error) {
		return loadGemma4TextModel(p)
	})
	metal.RegisterModelLoader("gemma4", func(p string, _ []byte) (metal.InternalModel, error) {
		return loadGemma4MultiModalModel(p)
	})
	metal.RegisterModelLoader("gemma4_unified", func(p string, _ []byte) (metal.InternalModel, error) {
		return loadGemma4MultiModalModel(p)
	})
}

func loadGemma4TextModel(modelPath string) (*Gemma4Model, error) {
	m, err := LoadGemma4(modelPath)
	if err != nil {
		return nil, err
	}
	if m.VisionTower != nil || m.MultiModalProjector != nil {
		closeGemma4Vision(m.VisionTower, m.MultiModalProjector)
		m.VisionTower = nil
		m.MultiModalProjector = nil
		metal.ClearCache()
	}
	m.modelType = "gemma4_text"
	if m.Cfg != nil {
		m.Cfg.ModelType = "gemma4_text"
		m.Cfg.VisionConfig = nil
	}
	return m, nil
}

func loadGemma4MultiModalModel(modelPath string) (*Gemma4Model, error) {
	m, err := LoadGemma4(modelPath)
	if err != nil {
		return nil, err
	}
	finalizeGemma4MultiModalModelType(m)
	return m, nil
}

func finalizeGemma4MultiModalModelType(m *Gemma4Model) {
	if m == nil {
		return
	}
	modelType := "gemma4"
	if m.Cfg != nil && m.Cfg.ModelType == "gemma4_unified" {
		modelType = "gemma4_unified"
	}
	m.modelType = modelType
	if m.Cfg != nil {
		m.Cfg.ModelType = modelType
	}
}

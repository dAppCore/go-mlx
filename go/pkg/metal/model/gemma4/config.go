// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma4

import (
	"maps"
	"math"

	core "dappco.re/go"
	"dappco.re/go/mlx/pkg/metal"
)

func defaultGemma4RopeParameters(cfg *Gemma4TextConfig) map[string]RopeParams {
	return map[string]RopeParams{
		"full_attention": {
			PartialRotaryFactor: cfg.GlobalPartialRotaryFactor,
			RopeTheta:           1000000.0,
			RopeType:            "proportional",
			Factor:              1.0,
		},
		"sliding_attention": {
			PartialRotaryFactor: 1.0,
			RopeTheta:           10000.0,
			RopeType:            "default",
			Factor:              1.0,
		},
	}
}

func mergeGemma4RopeParameters(cfg *Gemma4TextConfig) {
	defaults := defaultGemma4RopeParameters(cfg)
	if cfg.RopeParameters == nil {
		cfg.RopeParameters = defaults
		return
	}

	merged := make(map[string]RopeParams, len(defaults)+len(cfg.RopeParameters))
	for attentionType, params := range defaults {
		if override, ok := cfg.RopeParameters[attentionType]; ok {
			if override.PartialRotaryFactor == 0 {
				override.PartialRotaryFactor = params.PartialRotaryFactor
			}
			if override.RopeTheta == 0 {
				override.RopeTheta = params.RopeTheta
			}
			if override.RopeType == "" {
				override.RopeType = params.RopeType
			}
			if override.Factor == 0 {
				override.Factor = params.Factor
			}
			merged[attentionType] = override
			continue
		}
		merged[attentionType] = params
	}
	for attentionType, params := range cfg.RopeParameters {
		if _, ok := merged[attentionType]; ok {
			continue
		}
		if params.Factor == 0 {
			params.Factor = 1.0
		}
		merged[attentionType] = params
	}
	cfg.RopeParameters = merged
}

func cloneGemma4Int32Ptr(v *int32) *int32 {
	if v == nil {
		return nil
	}
	cloned := *v
	return &cloned
}

func cloneGemma4RopeParameters(src map[string]RopeParams) map[string]RopeParams {
	if len(src) == 0 {
		return nil
	}
	cloned := make(map[string]RopeParams, len(src))
	maps.Copy(cloned, src)
	return cloned
}

func overlayGemma4RopeParameters(base, overlay map[string]RopeParams) map[string]RopeParams {
	if len(base) == 0 && len(overlay) == 0 {
		return nil
	}
	merged := cloneGemma4RopeParameters(base)
	if merged == nil {
		merged = make(map[string]RopeParams, len(overlay))
	}
	for attentionType, params := range overlay {
		current := merged[attentionType]
		if params.PartialRotaryFactor != 0 {
			current.PartialRotaryFactor = params.PartialRotaryFactor
		}
		if params.RopeTheta != 0 {
			current.RopeTheta = params.RopeTheta
		}
		if params.RopeType != "" {
			current.RopeType = params.RopeType
		}
		if params.Factor != 0 {
			current.Factor = params.Factor
		}
		merged[attentionType] = current
	}
	return merged
}

func mergeGemma4ConfigMissing(dst *Gemma4TextConfig, src Gemma4TextConfig) {
	if dst.ModelType == "" && src.ModelType != "" {
		dst.ModelType = src.ModelType
	}
	if dst.PadTokenID == 0 && src.PadTokenID != 0 {
		dst.PadTokenID = src.PadTokenID
	}
	if dst.ImageTokenID == 0 && src.ImageTokenID != 0 {
		dst.ImageTokenID = src.ImageTokenID
	}
	if dst.AudioTokenID == 0 && src.AudioTokenID != 0 {
		dst.AudioTokenID = src.AudioTokenID
	}
	if dst.VideoTokenID == 0 && src.VideoTokenID != 0 {
		dst.VideoTokenID = src.VideoTokenID
	}
	if dst.BOITokenID == 0 && src.BOITokenID != 0 {
		dst.BOITokenID = src.BOITokenID
	}
	if dst.BOATokenID == 0 && src.BOATokenID != 0 {
		dst.BOATokenID = src.BOATokenID
	}
	if dst.EOITokenID == 0 && src.EOITokenID != 0 {
		dst.EOITokenID = src.EOITokenID
	}
	if dst.EOATokenIndex == 0 && src.EOATokenIndex != 0 {
		dst.EOATokenIndex = src.EOATokenIndex
	}
	if dst.HiddenSize == 0 {
		dst.HiddenSize = src.HiddenSize
	}
	if dst.NumHiddenLayers == 0 {
		dst.NumHiddenLayers = src.NumHiddenLayers
	}
	if dst.IntermediateSize == 0 {
		dst.IntermediateSize = src.IntermediateSize
	}
	if dst.NumAttentionHeads == 0 {
		dst.NumAttentionHeads = src.NumAttentionHeads
	}
	if dst.NumKeyValueHeads == 0 {
		dst.NumKeyValueHeads = src.NumKeyValueHeads
	}
	if dst.NumGlobalKeyValueHeads == nil {
		dst.NumGlobalKeyValueHeads = cloneGemma4Int32Ptr(src.NumGlobalKeyValueHeads)
	}
	if dst.HeadDim == 0 {
		dst.HeadDim = src.HeadDim
	}
	if dst.GlobalHeadDim == 0 {
		dst.GlobalHeadDim = src.GlobalHeadDim
	}
	if dst.GlobalPartialRotaryFactor == 0 {
		dst.GlobalPartialRotaryFactor = src.GlobalPartialRotaryFactor
	}
	if dst.VocabSize == 0 {
		dst.VocabSize = src.VocabSize
	}
	if dst.VocabSizePerLayerInput == 0 {
		dst.VocabSizePerLayerInput = src.VocabSizePerLayerInput
	}
	if dst.RMSNormEps == 0 {
		dst.RMSNormEps = src.RMSNormEps
	}
	if dst.SlidingWindow == 0 {
		dst.SlidingWindow = src.SlidingWindow
	}
	if dst.SlidingWindowPattern == 0 {
		dst.SlidingWindowPattern = src.SlidingWindowPattern
	}
	// Prefer the larger max_position_embeddings: the top-level value is the
	// model's real deployed context (31B/26B-MoE = 262144 / 256K) while
	// text_config carries the backbone's smaller 131072 — taking text_config
	// cramped the two biggest models to 128K. Larger wins; both-absent still
	// falls to the defaulting block below.
	if src.MaxPositionEmbeddings > dst.MaxPositionEmbeddings {
		dst.MaxPositionEmbeddings = src.MaxPositionEmbeddings
	}
	if dst.NumKVSharedLayers == 0 {
		dst.NumKVSharedLayers = src.NumKVSharedLayers
	}
	if dst.HiddenSizePerLayerInput == 0 {
		dst.HiddenSizePerLayerInput = src.HiddenSizePerLayerInput
	}
	if !dst.AttentionKEqV && src.AttentionKEqV {
		dst.AttentionKEqV = true
	}
	if dst.FinalLogitSoftcapping == 0 {
		dst.FinalLogitSoftcapping = src.FinalLogitSoftcapping
	}
	if !dst.EnableMoEBlock && src.EnableMoEBlock {
		dst.EnableMoEBlock = true
	}
	if dst.NumExperts == nil {
		dst.NumExperts = cloneGemma4Int32Ptr(src.NumExperts)
	}
	if dst.TopKExperts == nil {
		dst.TopKExperts = cloneGemma4Int32Ptr(src.TopKExperts)
	}
	if dst.MoEIntermediateSize == nil {
		dst.MoEIntermediateSize = cloneGemma4Int32Ptr(src.MoEIntermediateSize)
	}
	if len(dst.LayerTypesInput) == 0 && len(src.LayerTypesInput) > 0 {
		dst.LayerTypesInput = append([]string(nil), src.LayerTypesInput...)
	}
	if len(dst.RopeParameters) == 0 && len(src.RopeParameters) > 0 {
		dst.RopeParameters = cloneGemma4RopeParameters(src.RopeParameters)
	}
}

func parseGemma4Config(data []byte) (*Gemma4TextConfig, error) {
	var wrapper struct {
		ModelType                 string                    `json:"model_type"`
		Quantization              *metal.QuantizationConfig `json:"quantization"`
		LayerTypes                []string                  `json:"layer_types"`
		NumGlobalKeyValueHeads    *int32                    `json:"num_global_key_value_heads"`
		NumKVSharedLayers         *int32                    `json:"num_kv_shared_layers"`
		GlobalHeadDim             *int32                    `json:"global_head_dim"`
		GlobalPartialRotaryFactor *float32                  `json:"global_partial_rotary_factor"`
		HiddenSizePerLayerInput   *int32                    `json:"hidden_size_per_layer_input"`
		AttentionKEqV             *bool                     `json:"attention_k_eq_v"`
		FinalLogitSoftcapping     *float32                  `json:"final_logit_softcapping"`
		UseDoubleWideMLP          *bool                     `json:"use_double_wide_mlp"`
		EnableMoEBlock            *bool                     `json:"enable_moe_block"`
		PadTokenID                *int32                    `json:"pad_token_id"`
		ImageTokenID              *int32                    `json:"image_token_id"`
		AudioTokenID              *int32                    `json:"audio_token_id"`
		VideoTokenID              *int32                    `json:"video_token_id"`
		BOITokenID                *int32                    `json:"boi_token_id"`
		BOATokenID                *int32                    `json:"boa_token_id"`
		EOITokenID                *int32                    `json:"eoi_token_id"`
		EOATokenIndex             *int32                    `json:"eoa_token_index"`
		NumExperts                *int32                    `json:"num_experts"`
		TopKExperts               *int32                    `json:"top_k_experts"`
		MoEIntermediateSize       *int32                    `json:"moe_intermediate_size"`
		SlidingWindow             *int32                    `json:"sliding_window"`
		TieWordEmbeddings         *bool                     `json:"tie_word_embeddings"`
		RopeParameters            map[string]RopeParams     `json:"rope_parameters"`
		VisionConfig              *Gemma4VisionConfig       `json:"vision_config"`
		AudioConfig               *Gemma4AudioConfig        `json:"audio_config"`
		TextConfig                struct {
			Gemma4TextConfig
			Quantization              *metal.QuantizationConfig `json:"quantization"`
			LayerTypes                []string                  `json:"layer_types"`
			NumGlobalKeyValueHeads    *int32                    `json:"num_global_key_value_heads"`
			NumKVSharedLayers         *int32                    `json:"num_kv_shared_layers"`
			GlobalHeadDim             *int32                    `json:"global_head_dim"`
			GlobalPartialRotaryFactor *float32                  `json:"global_partial_rotary_factor"`
			HiddenSizePerLayerInput   *int32                    `json:"hidden_size_per_layer_input"`
			PadTokenID                *int32                    `json:"pad_token_id"`
			UseDoubleWideMLP          *bool                     `json:"use_double_wide_mlp"`
			TieWordEmbeddings         *bool                     `json:"tie_word_embeddings"`
			RopeParameters            map[string]RopeParams     `json:"rope_parameters"`
		} `json:"text_config"`
	}
	if r := core.JSONUnmarshal(data, &wrapper); !r.OK {
		return nil, core.E("gemma4.parseConfig", "parse config", nil)
	}

	cfg := wrapper.TextConfig.Gemma4TextConfig
	var top Gemma4TextConfig
	if r := core.JSONUnmarshal(data, &top); !r.OK {
		return nil, core.E("gemma4.parseConfig", "parse top-level fields", nil)
	}
	if cfg.NumHiddenLayers == 0 {
		if r := core.JSONUnmarshal(data, &cfg); !r.OK {
			return nil, core.E("gemma4.parseConfig", "parse top-level config", nil)
		}
	} else {
		mergeGemma4ConfigMissing(&cfg, top)
	}

	if wrapper.ModelType != "" {
		cfg.ModelType = wrapper.ModelType
	}
	cfg.VisionConfig = normalizeGemma4VisionConfig(wrapper.VisionConfig)
	cfg.AudioConfig = normalizeGemma4AudioConfig(wrapper.AudioConfig)
	cfg.Quantization = wrapper.Quantization
	if cfg.Quantization == nil {
		cfg.Quantization = wrapper.TextConfig.Quantization
	}
	switch {
	case wrapper.PadTokenID != nil:
		cfg.PadTokenID = *wrapper.PadTokenID
	case wrapper.TextConfig.PadTokenID != nil:
		cfg.PadTokenID = *wrapper.TextConfig.PadTokenID
	}
	switch {
	case wrapper.ImageTokenID != nil:
		cfg.ImageTokenID = *wrapper.ImageTokenID
	}
	switch {
	case wrapper.AudioTokenID != nil:
		cfg.AudioTokenID = *wrapper.AudioTokenID
	}
	switch {
	case wrapper.VideoTokenID != nil:
		cfg.VideoTokenID = *wrapper.VideoTokenID
	}
	switch {
	case wrapper.BOITokenID != nil:
		cfg.BOITokenID = *wrapper.BOITokenID
	}
	switch {
	case wrapper.BOATokenID != nil:
		cfg.BOATokenID = *wrapper.BOATokenID
	}
	switch {
	case wrapper.EOITokenID != nil:
		cfg.EOITokenID = *wrapper.EOITokenID
	}
	switch {
	case wrapper.EOATokenIndex != nil:
		cfg.EOATokenIndex = *wrapper.EOATokenIndex
	}
	switch {
	case len(wrapper.LayerTypes) > 0:
		cfg.LayerTypesInput = append([]string(nil), wrapper.LayerTypes...)
	case len(wrapper.TextConfig.LayerTypes) > 0:
		cfg.LayerTypesInput = append([]string(nil), wrapper.TextConfig.LayerTypes...)
	}
	switch {
	case wrapper.NumGlobalKeyValueHeads != nil:
		cfg.NumGlobalKeyValueHeads = cloneGemma4Int32Ptr(wrapper.NumGlobalKeyValueHeads)
	case wrapper.TextConfig.NumGlobalKeyValueHeads != nil:
		cfg.NumGlobalKeyValueHeads = cloneGemma4Int32Ptr(wrapper.TextConfig.NumGlobalKeyValueHeads)
	}
	switch {
	case wrapper.NumKVSharedLayers != nil:
		cfg.NumKVSharedLayers = *wrapper.NumKVSharedLayers
	case wrapper.TextConfig.NumKVSharedLayers != nil:
		cfg.NumKVSharedLayers = *wrapper.TextConfig.NumKVSharedLayers
	}
	switch {
	case wrapper.GlobalHeadDim != nil:
		cfg.GlobalHeadDim = *wrapper.GlobalHeadDim
	case wrapper.TextConfig.GlobalHeadDim != nil:
		cfg.GlobalHeadDim = *wrapper.TextConfig.GlobalHeadDim
	}
	switch {
	case wrapper.GlobalPartialRotaryFactor != nil:
		cfg.GlobalPartialRotaryFactor = *wrapper.GlobalPartialRotaryFactor
	case wrapper.TextConfig.GlobalPartialRotaryFactor != nil:
		cfg.GlobalPartialRotaryFactor = *wrapper.TextConfig.GlobalPartialRotaryFactor
	}
	cfg.RopeParameters = overlayGemma4RopeParameters(cfg.RopeParameters, wrapper.TextConfig.RopeParameters)
	cfg.RopeParameters = overlayGemma4RopeParameters(cfg.RopeParameters, wrapper.RopeParameters)
	switch {
	case wrapper.HiddenSizePerLayerInput != nil:
		cfg.HiddenSizePerLayerInput = *wrapper.HiddenSizePerLayerInput
	case wrapper.TextConfig.HiddenSizePerLayerInput != nil:
		cfg.HiddenSizePerLayerInput = *wrapper.TextConfig.HiddenSizePerLayerInput
	}
	switch {
	case wrapper.AttentionKEqV != nil:
		cfg.AttentionKEqV = *wrapper.AttentionKEqV
	}
	switch {
	case wrapper.FinalLogitSoftcapping != nil:
		cfg.FinalLogitSoftcapping = *wrapper.FinalLogitSoftcapping
	}
	switch {
	case wrapper.EnableMoEBlock != nil:
		cfg.EnableMoEBlock = *wrapper.EnableMoEBlock
	}
	switch {
	case wrapper.NumExperts != nil:
		cfg.NumExperts = cloneGemma4Int32Ptr(wrapper.NumExperts)
	}
	switch {
	case wrapper.TopKExperts != nil:
		cfg.TopKExperts = cloneGemma4Int32Ptr(wrapper.TopKExperts)
	}
	switch {
	case wrapper.MoEIntermediateSize != nil:
		cfg.MoEIntermediateSize = cloneGemma4Int32Ptr(wrapper.MoEIntermediateSize)
	}
	switch {
	case wrapper.SlidingWindow != nil:
		cfg.SlidingWindow = *wrapper.SlidingWindow
	}
	switch {
	case wrapper.UseDoubleWideMLP != nil:
		cfg.UseDoubleWideMLP = *wrapper.UseDoubleWideMLP
	case wrapper.TextConfig.UseDoubleWideMLP != nil:
		cfg.UseDoubleWideMLP = *wrapper.TextConfig.UseDoubleWideMLP
	}
	switch {
	case wrapper.TieWordEmbeddings != nil:
		cfg.TieWordEmbeddings = *wrapper.TieWordEmbeddings
	case wrapper.TextConfig.TieWordEmbeddings != nil:
		cfg.TieWordEmbeddings = *wrapper.TextConfig.TieWordEmbeddings
	}

	if cfg.HeadDim == 0 && cfg.HiddenSize > 0 && cfg.NumAttentionHeads > 0 {
		cfg.HeadDim = cfg.HiddenSize / cfg.NumAttentionHeads
	}
	if cfg.GlobalHeadDim == 0 {
		switch {
		case wrapper.TextConfig.GlobalHeadDim != nil:
			cfg.GlobalHeadDim = *wrapper.TextConfig.GlobalHeadDim
		case wrapper.GlobalHeadDim != nil:
			cfg.GlobalHeadDim = *wrapper.GlobalHeadDim
		default:
			cfg.GlobalHeadDim = 512
		}
	}
	if cfg.GlobalPartialRotaryFactor == 0 {
		cfg.GlobalPartialRotaryFactor = 0.25
	}
	if cfg.RMSNormEps == 0 {
		cfg.RMSNormEps = 1e-6
	}
	if cfg.VocabSize == 0 {
		cfg.VocabSize = 262144
	}
	unified := gemma4UnifiedConfig(&cfg)
	if cfg.ImageTokenID == 0 {
		cfg.ImageTokenID = 258880
	}
	if unified {
		if cfg.AudioTokenID == 0 {
			cfg.AudioTokenID = 258881
		}
		if cfg.VideoTokenID == 0 {
			cfg.VideoTokenID = 258884
		}
		if cfg.BOITokenID == 0 {
			cfg.BOITokenID = 255999
		}
		if cfg.BOATokenID == 0 {
			cfg.BOATokenID = 256000
		}
		if cfg.EOITokenID == 0 {
			cfg.EOITokenID = 258882
		}
		if cfg.EOATokenIndex == 0 {
			cfg.EOATokenIndex = 258883
		}
	}
	if cfg.VocabSizePerLayerInput == 0 {
		cfg.VocabSizePerLayerInput = cfg.VocabSize
	}
	if cfg.SlidingWindow == 0 {
		if unified {
			cfg.SlidingWindow = 1024
		} else {
			cfg.SlidingWindow = 512
		}
	}
	if cfg.SlidingWindowPattern == 0 {
		cfg.SlidingWindowPattern = 6
	}
	if cfg.MaxPositionEmbeddings == 0 {
		if unified {
			cfg.MaxPositionEmbeddings = 262144
		} else {
			cfg.MaxPositionEmbeddings = 131072
		}
	}
	// No final_logit_softcapping fallback: the model declares it (gemma-4 ships
	// 30; a model that omits it wants none). Guessing 30 imposes softcapping on
	// a model that never asked — 0 is honoured downstream as "skip" (forward.go).
	if cfg.HiddenSizePerLayerInput == 0 {
		switch {
		case wrapper.TextConfig.HiddenSizePerLayerInput != nil:
			cfg.HiddenSizePerLayerInput = *wrapper.TextConfig.HiddenSizePerLayerInput
		case wrapper.HiddenSizePerLayerInput != nil:
			cfg.HiddenSizePerLayerInput = *wrapper.HiddenSizePerLayerInput
		default:
			if unified {
				cfg.HiddenSizePerLayerInput = 0
			} else {
				cfg.HiddenSizePerLayerInput = 256
			}
		}
	}
	if cfg.EnableMoEBlock {
		if cfg.NumExperts == nil {
			numExperts := int32(128)
			cfg.NumExperts = &numExperts
		}
		if cfg.TopKExperts == nil {
			topK := int32(8)
			cfg.TopKExperts = &topK
		}
	}
	if !cfg.UseDoubleWideMLP && wrapper.UseDoubleWideMLP == nil && wrapper.TextConfig.UseDoubleWideMLP == nil && !unified {
		cfg.UseDoubleWideMLP = true
	}
	if !cfg.TieWordEmbeddings && wrapper.TieWordEmbeddings == nil && wrapper.TextConfig.TieWordEmbeddings == nil {
		cfg.TieWordEmbeddings = true
	}
	if field := gemma4NegativeConfigField(&cfg); field != "" {
		return nil, core.E("gemma4.parseConfig", "negative "+field+" is invalid", nil)
	}
	mergeGemma4RopeParameters(&cfg)
	if len(cfg.LayerTypesInput) > 0 {
		cfg.LayerTypes = append([]string(nil), cfg.LayerTypesInput...)
	} else {
		cfg.LayerTypes = make([]string, cfg.NumHiddenLayers)
		pattern := int(cfg.SlidingWindowPattern)
		for i := range cfg.NumHiddenLayers {
			if pattern > 1 && (int(i)+1)%pattern != 0 {
				cfg.LayerTypes[i] = "sliding_attention"
			} else {
				cfg.LayerTypes[i] = "full_attention"
			}
		}
	}
	if len(cfg.LayerTypes) > 0 {
		cfg.LayerTypes[len(cfg.LayerTypes)-1] = "full_attention"
	}
	if len(cfg.LayerTypes) < int(cfg.NumHiddenLayers) {
		return nil, core.E("gemma4.parseConfig", "layer_types shorter than num_hidden_layers", nil)
	}
	cfg.LayerTypes = cfg.LayerTypes[:cfg.NumHiddenLayers]
	gemma4FinaliseEmbeddingScales(&cfg)
	return &cfg, nil
}

func gemma4UnifiedConfig(cfg *Gemma4TextConfig) bool {
	if cfg == nil {
		return false
	}
	if cfg.ModelType == "gemma4_unified" || cfg.ModelType == "gemma4_unified_text" {
		return true
	}
	if cfg.VisionConfig != nil && cfg.VisionConfig.ModelType == "gemma4_unified_vision" {
		return true
	}
	if cfg.AudioConfig != nil && cfg.AudioConfig.ModelType == "gemma4_unified_audio" {
		return true
	}
	return false
}

// gemma4FinaliseEmbeddingScales caches sqrt(HiddenSize),
// sqrt(HiddenSizePerLayerInput), and 1/sqrt(HiddenSize) on the config
// so per-token forward passes can skip the math.Sqrt/math.Pow + float32
// narrowing entirely. Safe to call multiple times — the loader
// re-invokes after inferring or resetting HiddenSizePerLayerInput from
// weights.
func gemma4FinaliseEmbeddingScales(cfg *Gemma4TextConfig) {
	if cfg == nil {
		return
	}
	if cfg.HiddenSize > 0 {
		cfg.EmbeddingScale = float32(math.Sqrt(float64(cfg.HiddenSize)))
		cfg.PerLayerProjectionScale = float32(math.Pow(float64(cfg.HiddenSize), -0.5))
	} else {
		cfg.EmbeddingScale = 0
		cfg.PerLayerProjectionScale = 0
	}
	if cfg.HiddenSizePerLayerInput > 0 {
		cfg.PerLayerInputEmbeddingScale = float32(math.Sqrt(float64(cfg.HiddenSizePerLayerInput)))
	} else {
		cfg.PerLayerInputEmbeddingScale = 0
	}
}

func validateGemma4QuantizationConfig(q *metal.QuantizationConfig) error {
	if q == nil {
		return nil
	}
	if q.GroupSize < 0 {
		return core.NewError("gemma4: quantization group_size must be >= 0")
	}
	if q.Bits < 0 {
		return core.NewError("gemma4: quantization bits must be >= 0")
	}
	mode := metal.NormalizeQuantizationMode(q.Mode)
	switch mode {
	case "affine":
		if q.Bits != 0 && q.Bits != 2 && q.Bits != 3 && q.Bits != 4 && q.Bits != 5 && q.Bits != 6 && q.Bits != 8 {
			return core.NewError(core.Sprintf("gemma4: affine quantization bits %d are unsupported", q.Bits))
		}
	case "mxfp4":
		if q.GroupSize != 0 && q.GroupSize != 32 {
			return core.NewError(core.Sprintf("gemma4: mxfp4 quantization requires group_size=32, got %d", q.GroupSize))
		}
		if q.Bits != 0 && q.Bits != 4 {
			return core.NewError(core.Sprintf("gemma4: mxfp4 quantization requires bits=4, got %d", q.Bits))
		}
	case "mxfp8":
		if q.GroupSize != 0 && q.GroupSize != 32 {
			return core.NewError(core.Sprintf("gemma4: mxfp8 quantization requires group_size=32, got %d", q.GroupSize))
		}
		if q.Bits != 0 && q.Bits != 8 {
			return core.NewError(core.Sprintf("gemma4: mxfp8 quantization requires bits=8, got %d", q.Bits))
		}
	case "nvfp4":
		if q.GroupSize != 0 && q.GroupSize != 16 {
			return core.NewError(core.Sprintf("gemma4: nvfp4 quantization requires group_size=16, got %d", q.GroupSize))
		}
		if q.Bits != 0 && q.Bits != 4 {
			return core.NewError(core.Sprintf("gemma4: nvfp4 quantization requires bits=4, got %d", q.Bits))
		}
	default:
		return core.NewError(core.Sprintf("gemma4: unsupported quantization mode %q", q.Mode))
	}
	return nil
}

func gemma4NegativeConfigField(cfg *Gemma4TextConfig) string {
	checks := []struct {
		name  string
		value int32
	}{
		{"pad_token_id", cfg.PadTokenID},
		{"image_token_id", cfg.ImageTokenID},
		{"audio_token_id", cfg.AudioTokenID},
		{"video_token_id", cfg.VideoTokenID},
		{"boi_token_id", cfg.BOITokenID},
		{"boa_token_id", cfg.BOATokenID},
		{"eoi_token_id", cfg.EOITokenID},
		{"eoa_token_index", cfg.EOATokenIndex},
		{"hidden_size", cfg.HiddenSize},
		{"num_hidden_layers", cfg.NumHiddenLayers},
		{"intermediate_size", cfg.IntermediateSize},
		{"num_attention_heads", cfg.NumAttentionHeads},
		{"num_key_value_heads", cfg.NumKeyValueHeads},
		{"head_dim", cfg.HeadDim},
		{"global_head_dim", cfg.GlobalHeadDim},
		{"vocab_size", cfg.VocabSize},
		{"vocab_size_per_layer_input", cfg.VocabSizePerLayerInput},
		{"sliding_window", cfg.SlidingWindow},
		{"sliding_window_pattern", cfg.SlidingWindowPattern},
		{"max_position_embeddings", cfg.MaxPositionEmbeddings},
		{"num_kv_shared_layers", cfg.NumKVSharedLayers},
		{"hidden_size_per_layer_input", cfg.HiddenSizePerLayerInput},
	}
	for _, check := range checks {
		if check.value < 0 {
			return check.name
		}
	}
	ptrChecks := []struct {
		name  string
		value *int32
	}{
		{"num_global_key_value_heads", cfg.NumGlobalKeyValueHeads},
		{"num_experts", cfg.NumExperts},
		{"top_k_experts", cfg.TopKExperts},
		{"moe_intermediate_size", cfg.MoEIntermediateSize},
	}
	for _, check := range ptrChecks {
		if check.value != nil && *check.value < 0 {
			return check.name
		}
	}
	return ""
}

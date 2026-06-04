// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"math"

	core "dappco.re/go"
)

// ParseDenseConfig reads the shared Llama-family dense-transformer config used
// by Qwen 2/3, Llama, Mistral, Hermes, Granite, Phi, GLM and related MoE
// families. Model packages use this instead of reaching into Qwen-specific
// loader code.
func ParseDenseConfig(data []byte) (*DenseConfig, error) {
	var cfg DenseConfig
	if r := core.JSONUnmarshal(data, &cfg); !r.OK {
		return nil, core.E("dense.parseConfig", "parse config", nil)
	}

	var wrapper struct {
		TextConfig         *DenseConfig        `json:"text_config"`
		Quantization       *QuantizationConfig `json:"quantization"`
		QuantizationConfig *QuantizationConfig `json:"quantization_config"`
	}
	if r := core.JSONUnmarshal(data, &wrapper); !r.OK {
		return nil, core.E("dense.parseConfig", "parse nested config", nil)
	}
	if wrapper.TextConfig != nil {
		cfg = mergeDenseTextConfig(cfg, *wrapper.TextConfig)
	}
	cfg.ModelType = normalizeProbeModelType(cfg.ModelType)
	cfg.Quantization = FirstQuantization(wrapper.Quantization, wrapper.QuantizationConfig, cfg.Quantization)

	if cfg.HeadDim == 0 && cfg.NumAttentionHeads > 0 {
		cfg.HeadDim = cfg.HiddenSize / cfg.NumAttentionHeads
	}
	if cfg.HeadDim > 0 {
		cfg.Scale = float32(1.0 / math.Sqrt(float64(cfg.HeadDim)))
	}
	if cfg.RopeTheta == 0 {
		cfg.RopeTheta = 1000000
	}
	if cfg.RMSNormEps == 0 {
		cfg.RMSNormEps = 1e-6
	}
	if cfg.VocabSize == 0 {
		cfg.VocabSize = 151936
	}

	return &cfg, nil
}

func mergeDenseTextConfig(top, text DenseConfig) DenseConfig {
	if text.ModelType == "" {
		text.ModelType = top.ModelType
	}
	text.Quantization = FirstQuantization(text.Quantization, top.Quantization)
	if text.VocabSize == 0 {
		text.VocabSize = top.VocabSize
	}
	if text.HiddenSize == 0 {
		text.HiddenSize = top.HiddenSize
	}
	if text.NumHiddenLayers == 0 {
		text.NumHiddenLayers = top.NumHiddenLayers
	}
	if text.IntermediateSize == 0 {
		text.IntermediateSize = top.IntermediateSize
	}
	if text.MoEIntermediateSize == 0 {
		text.MoEIntermediateSize = top.MoEIntermediateSize
	}
	if text.NumAttentionHeads == 0 {
		text.NumAttentionHeads = top.NumAttentionHeads
	}
	if text.NumKeyValueHeads == 0 {
		text.NumKeyValueHeads = top.NumKeyValueHeads
	}
	if text.NumExperts == 0 {
		text.NumExperts = top.NumExperts
	}
	if text.NumExpertsPerTok == 0 {
		text.NumExpertsPerTok = top.NumExpertsPerTok
	}
	if text.DecoderSparseStep == 0 {
		text.DecoderSparseStep = top.DecoderSparseStep
	}
	if text.HeadDim == 0 {
		text.HeadDim = top.HeadDim
	}
	if text.RMSNormEps == 0 {
		text.RMSNormEps = top.RMSNormEps
	}
	if text.RopeTheta == 0 {
		text.RopeTheta = top.RopeTheta
	}
	if text.PartialRotaryFactor == 0 {
		text.PartialRotaryFactor = top.PartialRotaryFactor
	}
	if text.MaxPositionEmbeddings == 0 {
		text.MaxPositionEmbeddings = top.MaxPositionEmbeddings
	}
	if len(text.LayerTypes) == 0 && len(top.LayerTypes) > 0 {
		text.LayerTypes = append([]string(nil), top.LayerTypes...)
	}
	return text
}

// FirstQuantization returns the first non-nil QuantizationConfig so model
// packages can pick between top-level and nested quant configs.
func FirstQuantization(configs ...*QuantizationConfig) *QuantizationConfig {
	for _, cfg := range configs {
		if cfg != nil {
			return cfg
		}
	}
	return nil
}

func (cfg *DenseConfig) IsMoE() bool {
	return cfg != nil && (cfg.ModelType == "qwen3_moe" || cfg.ModelType == "qwen3_6_moe" || cfg.NumExperts > 0 || cfg.NumExpertsPerTok > 0 || cfg.MoEIntermediateSize > 0)
}

func (cfg *DenseConfig) IsQwen36Hybrid() bool {
	if cfg == nil {
		return false
	}
	switch normalizeProbeModelType(cfg.ModelType) {
	case "qwen3_6", "qwen3_6_moe":
		return true
	}
	for _, layerType := range cfg.LayerTypes {
		if NormalizeDenseLayerType(layerType) == "linear_attention" {
			return true
		}
	}
	return cfg.PartialRotaryFactor > 0 && cfg.PartialRotaryFactor < 1
}

// NormalizeDenseLayerType canonicalises layer type identifiers from dense
// family configs.
func NormalizeDenseLayerType(value string) string {
	value = core.Lower(core.Trim(value))
	value = core.Replace(value, "-", "_")
	return core.Replace(value, ".", "_")
}

// Qwen36NativeGuardMessage keeps the staged Qwen 3.6 diagnostic consistent
// across the dense and MoE loaders.
func Qwen36NativeGuardMessage(modelType string) string {
	if normalizeProbeModelType(modelType) == "qwen3_6_moe" {
		return "qwen3_6_moe hybrid linear attention and sparse expert routing are not implemented in the native Go loader yet"
	}
	return "qwen3_6 hybrid linear attention is not implemented in the native Go loader yet"
}

// DetectDenseModelType selects the concrete dense-family architecture from
// config metadata or Qwen3 Q/K norm weights.
func DetectDenseModelType(configData []byte, weights map[string]*Array) string {
	if detected, err := probeModelType(configData); err == nil {
		switch detected {
		case "llama", "mistral", "hermes", "granite", "phi", "glm", "qwen2", "qwen3", "qwen3_next", "qwen3_6", "qwen3_6_moe", "qwen3_moe":
			return detected
		}
	}

	if HasResolvedWeight(weights, "model.layers.0.self_attn.q_norm.weight") {
		return "qwen3"
	}
	return "qwen2"
}

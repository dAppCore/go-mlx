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
	// rope_theta / partial_rotary_factor may be nested under rope_parameters
	// (Qwen3.5/3.6) or rope_scaling rather than flat — fill the flat fields from
	// there when absent, before the default below. Flat fields win when present,
	// so the families that use them are unaffected.
	for _, rp := range []*DenseRopeParams{cfg.RopeParameters, cfg.RopeScaling} {
		if rp == nil {
			continue
		}
		if cfg.RopeTheta == 0 {
			cfg.RopeTheta = rp.RopeTheta
		}
		if cfg.PartialRotaryFactor == 0 {
			cfg.PartialRotaryFactor = rp.PartialRotaryFactor
		}
	}
	if cfg.RopeTheta == 0 {
		// transformers' default rope base when a config omits rope_theta. Archs
		// that use a larger base (Qwen 1e6, long-context variants) declare it in
		// their config; 1e6 here was Qwen-specific and wrong for the Llama /
		// Mistral families this shared parser also serves.
		cfg.RopeTheta = 10000
	}
	if cfg.RMSNormEps == 0 {
		cfg.RMSNormEps = 1e-6
	}
	// vocab_size is a DIMENSION — the dense loaders (qwen3 / qwen3_moe) derive it
	// from the token-embedding tensor's rows. Never fabricated to one family's
	// vocab here: 151936 is Qwen's, wrong for Llama (128256) / Mistral (32000) /
	// the rest of the dense family this shared parser serves.

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
	if text.RopeParameters == nil {
		text.RopeParameters = top.RopeParameters
	}
	if text.RopeScaling == nil {
		text.RopeScaling = top.RopeScaling
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
	if cfg == nil {
		return false
	}
	if cfg.NumExperts > 0 || cfg.NumExpertsPerTok > 0 || cfg.MoEIntermediateSize > 0 {
		return true
	}
	// Fall back to the "_moe" model_type convention the dense families use for
	// their mixture variants (qwen3_moe, qwen3_6_moe, ...) so a config that
	// declares MoE by arch id rather than expert counts is still recognised —
	// without the engine hardcoding family names.
	return core.HasSuffix(cfg.ModelType, "_moe")
}

// NormalizeDenseLayerType canonicalises layer type identifiers from dense
// family configs.
func NormalizeDenseLayerType(value string) string {
	value = core.Lower(core.Trim(value))
	value = core.Replace(value, "-", "_")
	return core.Replace(value, ".", "_")
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

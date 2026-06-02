// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import "dappco.re/go"

type qwen36StagedConfig struct {
	ModelType             string             `json:"model_type,omitempty"`
	Architectures         []string           `json:"architectures,omitempty"`
	VocabSize             int                `json:"vocab_size,omitempty"`
	HiddenSize            int                `json:"hidden_size,omitempty"`
	IntermediateSize      int                `json:"intermediate_size,omitempty"`
	NumHiddenLayers       int                `json:"num_hidden_layers,omitempty"`
	NumAttentionHeads     int                `json:"num_attention_heads,omitempty"`
	NumKeyValueHeads      int                `json:"num_key_value_heads,omitempty"`
	HeadDim               int                `json:"head_dim,omitempty"`
	MaxPositionEmbeddings int                `json:"max_position_embeddings,omitempty"`
	SlidingWindow         int                `json:"sliding_window,omitempty"`
	LayerTypes            []string           `json:"layer_types,omitempty"`
	Quantization          QuantizationConfig `json:"quantization,omitempty"`
	TextConfig            *qwen36TextConfig  `json:"text_config,omitempty"`
}

type qwen36TextConfig struct {
	ModelType             string             `json:"model_type,omitempty"`
	VocabSize             int                `json:"vocab_size,omitempty"`
	HiddenSize            int                `json:"hidden_size,omitempty"`
	IntermediateSize      int                `json:"intermediate_size,omitempty"`
	NumHiddenLayers       int                `json:"num_hidden_layers,omitempty"`
	NumAttentionHeads     int                `json:"num_attention_heads,omitempty"`
	NumKeyValueHeads      int                `json:"num_key_value_heads,omitempty"`
	HeadDim               int                `json:"head_dim,omitempty"`
	MaxPositionEmbeddings int                `json:"max_position_embeddings,omitempty"`
	SlidingWindow         int                `json:"sliding_window,omitempty"`
	LayerTypes            []string           `json:"layer_types,omitempty"`
	Quantization          QuantizationConfig `json:"quantization,omitempty"`
}

type qwen36StagedModel struct {
	path      string
	config    qwen36StagedConfig
	tokenizer *Tokenizer
}

func loadQwen36StagedModel(modelPath string, configData []byte) (*qwen36StagedModel, error) {
	cfg, err := parseQwen36StagedConfig(configData)
	if err != nil {
		return nil, err
	}
	if err := cfg.validate(); err != nil {
		return nil, err
	}
	root := resolveModelRoot(modelPath)
	tokenizer, err := LoadTokenizer(core.JoinPath(root, "tokenizer.json"))
	if err != nil {
		return nil, core.E("qwen3_6.load", "load tokenizer", err)
	}
	return &qwen36StagedModel{path: root, config: cfg, tokenizer: tokenizer}, nil
}

func parseQwen36StagedConfig(data []byte) (qwen36StagedConfig, error) {
	var cfg qwen36StagedConfig
	if result := core.JSONUnmarshal(data, &cfg); !result.OK {
		return qwen36StagedConfig{}, result.Value.(error)
	}
	detected := firstNonEmptyString(cfg.ModelType, firstQwen36ArchitectureName(cfg.Architectures))
	if cfg.TextConfig != nil && cfg.TextConfig.HiddenSize > 0 {
		cfg.applyTextConfig(*cfg.TextConfig)
	}
	if detected == "" {
		detected = firstNonEmptyString(cfg.ModelType, firstQwen36ArchitectureName(cfg.Architectures))
	}
	if normalizeProbeModelType(detected) != "qwen3_6" {
		return qwen36StagedConfig{}, core.NewError("qwen3_6 validation requires qwen3_6/qwen3_5 config")
	}
	cfg.ModelType = "qwen3_6"
	return cfg, nil
}

func (cfg *qwen36StagedConfig) applyTextConfig(text qwen36TextConfig) {
	cfg.ModelType = firstNonEmptyString(text.ModelType, cfg.ModelType)
	cfg.VocabSize = firstPositiveInt(text.VocabSize, cfg.VocabSize)
	cfg.HiddenSize = firstPositiveInt(text.HiddenSize, cfg.HiddenSize)
	cfg.IntermediateSize = firstPositiveInt(text.IntermediateSize, cfg.IntermediateSize)
	cfg.NumHiddenLayers = firstPositiveInt(text.NumHiddenLayers, cfg.NumHiddenLayers)
	cfg.NumAttentionHeads = firstPositiveInt(text.NumAttentionHeads, cfg.NumAttentionHeads)
	cfg.NumKeyValueHeads = firstPositiveInt(text.NumKeyValueHeads, cfg.NumKeyValueHeads)
	cfg.HeadDim = firstPositiveInt(text.HeadDim, cfg.HeadDim)
	cfg.MaxPositionEmbeddings = firstPositiveInt(text.MaxPositionEmbeddings, cfg.MaxPositionEmbeddings)
	cfg.SlidingWindow = firstPositiveInt(text.SlidingWindow, cfg.SlidingWindow)
	if len(text.LayerTypes) > 0 {
		cfg.LayerTypes = append([]string(nil), text.LayerTypes...)
	}
	if text.Quantization.Bits > 0 || text.Quantization.GroupSize > 0 || text.Quantization.Mode != "" {
		cfg.Quantization = text.Quantization
	}
}

func (cfg qwen36StagedConfig) validate() error {
	if cfg.HiddenSize <= 0 || cfg.NumHiddenLayers <= 0 || cfg.VocabSize <= 0 {
		return core.NewError("qwen3_6 validation requires hidden size, layer count, and vocab size")
	}
	if cfg.NumAttentionHeads <= 0 || cfg.NumKeyValueHeads <= 0 {
		return core.NewError("qwen3_6 validation requires attention and key/value head counts")
	}
	if cfg.MaxPositionEmbeddings <= 0 {
		return core.NewError("qwen3_6 validation requires max_position_embeddings")
	}
	if !qwen36LayerTypesIncludeLinearAttention(cfg.LayerTypes) {
		return core.NewError("qwen3_6 validation requires linear_attention layer metadata")
	}
	return nil
}

func (m *qwen36StagedModel) Forward(_ *Array, _ []Cache) *Array { return nil }

func (m *qwen36StagedModel) ForwardMasked(_ *Array, _ *Array, _ []Cache) *Array { return nil }

func (m *qwen36StagedModel) NewCache() []Cache { return nil }

func (m *qwen36StagedModel) NumLayers() int { return m.config.NumHiddenLayers }

func (m *qwen36StagedModel) Tokenizer() *Tokenizer { return m.tokenizer }

func (m *qwen36StagedModel) ModelType() string { return "qwen3_6" }

func (m *qwen36StagedModel) ApplyLoRA(_ LoRAConfig) *LoRAAdapter { return nil }

func firstQwen36ArchitectureName(values []string) string {
	for _, value := range values {
		if isQwen36Architecture(value) {
			return "qwen3_6"
		}
	}
	return ""
}

func qwen36LayerTypesIncludeLinearAttention(values []string) bool {
	for _, value := range values {
		if core.Contains(core.Lower(value), "linear_attention") {
			return true
		}
	}
	return false
}

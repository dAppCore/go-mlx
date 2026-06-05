// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package qwen3

import (
	"slices"

	"dappco.re/go"
	"dappco.re/go/mlx/pkg/metal"
)

type qwen36StagedConfig struct {
	ModelType             string                   `json:"model_type,omitempty"`
	Architectures         []string                 `json:"architectures,omitempty"`
	VocabSize             int                      `json:"vocab_size,omitempty"`
	HiddenSize            int                      `json:"hidden_size,omitempty"`
	IntermediateSize      int                      `json:"intermediate_size,omitempty"`
	NumHiddenLayers       int                      `json:"num_hidden_layers,omitempty"`
	NumAttentionHeads     int                      `json:"num_attention_heads,omitempty"`
	NumKeyValueHeads      int                      `json:"num_key_value_heads,omitempty"`
	HeadDim               int                      `json:"head_dim,omitempty"`
	MaxPositionEmbeddings int                      `json:"max_position_embeddings,omitempty"`
	SlidingWindow         int                      `json:"sliding_window,omitempty"`
	LayerTypes            []string                 `json:"layer_types,omitempty"`
	Quantization          metal.QuantizationConfig `json:"quantization"`
	TextConfig            *qwen36TextConfig        `json:"text_config,omitempty"`
}

type qwen36TextConfig struct {
	ModelType             string                   `json:"model_type,omitempty"`
	VocabSize             int                      `json:"vocab_size,omitempty"`
	HiddenSize            int                      `json:"hidden_size,omitempty"`
	IntermediateSize      int                      `json:"intermediate_size,omitempty"`
	NumHiddenLayers       int                      `json:"num_hidden_layers,omitempty"`
	NumAttentionHeads     int                      `json:"num_attention_heads,omitempty"`
	NumKeyValueHeads      int                      `json:"num_key_value_heads,omitempty"`
	HeadDim               int                      `json:"head_dim,omitempty"`
	MaxPositionEmbeddings int                      `json:"max_position_embeddings,omitempty"`
	SlidingWindow         int                      `json:"sliding_window,omitempty"`
	LayerTypes            []string                 `json:"layer_types,omitempty"`
	Quantization          metal.QuantizationConfig `json:"quantization"`
}

type qwen36StagedModel struct {
	path      string
	config    qwen36StagedConfig
	plan      metal.HybridAttentionCachePlan
	tokenizer *metal.Tokenizer
}

func loadQwen36StagedModel(modelPath string, configData []byte) (*qwen36StagedModel, error) {
	cfg, err := parseQwen36StagedConfig(configData)
	if err != nil {
		return nil, err
	}
	if err := cfg.validate(); err != nil {
		return nil, err
	}
	plan, err := metal.BuildHybridAttentionCachePlan(cfg.NumHiddenLayers, cfg.LayerTypes, cfg.SlidingWindow)
	if err != nil {
		return nil, err
	}
	root := metal.ResolveModelRoot(modelPath)
	tokenizer, err := metal.LoadTokenizer(core.JoinPath(root, "tokenizer.json"))
	if err != nil {
		return nil, core.E("qwen3_6.load", "load tokenizer", err)
	}
	return &qwen36StagedModel{path: root, config: cfg, plan: plan, tokenizer: tokenizer}, nil
}

func parseQwen36StagedConfig(data []byte) (qwen36StagedConfig, error) {
	var cfg qwen36StagedConfig
	if result := core.JSONUnmarshal(data, &cfg); !result.OK {
		return qwen36StagedConfig{}, result.Value.(error)
	}
	detected := metal.FirstNonEmptyString(cfg.ModelType, firstQwen36ArchitectureName(cfg.Architectures))
	if cfg.TextConfig != nil && cfg.TextConfig.HiddenSize > 0 {
		cfg.applyTextConfig(*cfg.TextConfig)
	}
	if detected == "" {
		detected = metal.FirstNonEmptyString(cfg.ModelType, firstQwen36ArchitectureName(cfg.Architectures))
	}
	if metal.NormalizeProbeModelType(detected) != "qwen3_6" {
		return qwen36StagedConfig{}, core.NewError("qwen3_6 validation requires qwen3_6/qwen3_5 config")
	}
	cfg.ModelType = "qwen3_6"
	return cfg, nil
}

func (cfg *qwen36StagedConfig) applyTextConfig(text qwen36TextConfig) {
	cfg.ModelType = metal.FirstNonEmptyString(text.ModelType, cfg.ModelType)
	cfg.VocabSize = metal.FirstPositiveInt(text.VocabSize, cfg.VocabSize)
	cfg.HiddenSize = metal.FirstPositiveInt(text.HiddenSize, cfg.HiddenSize)
	cfg.IntermediateSize = metal.FirstPositiveInt(text.IntermediateSize, cfg.IntermediateSize)
	cfg.NumHiddenLayers = metal.FirstPositiveInt(text.NumHiddenLayers, cfg.NumHiddenLayers)
	cfg.NumAttentionHeads = metal.FirstPositiveInt(text.NumAttentionHeads, cfg.NumAttentionHeads)
	cfg.NumKeyValueHeads = metal.FirstPositiveInt(text.NumKeyValueHeads, cfg.NumKeyValueHeads)
	cfg.HeadDim = metal.FirstPositiveInt(text.HeadDim, cfg.HeadDim)
	cfg.MaxPositionEmbeddings = metal.FirstPositiveInt(text.MaxPositionEmbeddings, cfg.MaxPositionEmbeddings)
	cfg.SlidingWindow = metal.FirstPositiveInt(text.SlidingWindow, cfg.SlidingWindow)
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
	if _, err := metal.BuildHybridAttentionCachePlan(cfg.NumHiddenLayers, cfg.LayerTypes, cfg.SlidingWindow); err != nil {
		return err
	}
	return nil
}

func (m *qwen36StagedModel) Forward(_ *metal.Array, _ []metal.Cache) *metal.Array { return nil }

func (m *qwen36StagedModel) ForwardMasked(_ *metal.Array, _ *metal.Array, _ []metal.Cache) *metal.Array {
	return nil
}

func (m *qwen36StagedModel) NewCache() []metal.Cache {
	plan, ok := m.qwen36HybridCachePlan()
	if !ok {
		return nil
	}
	return qwen36NewHybridCaches(plan)
}

func (m *qwen36StagedModel) qwen36HybridCachePlan() (metal.HybridAttentionCachePlan, bool) {
	if len(m.plan.Layers) == m.config.NumHiddenLayers && len(m.plan.CacheIndexByLayer) == m.config.NumHiddenLayers {
		return m.plan, true
	}
	plan, err := metal.BuildHybridAttentionCachePlan(m.config.NumHiddenLayers, m.config.LayerTypes, m.config.SlidingWindow)
	if err != nil {
		return metal.HybridAttentionCachePlan{}, false
	}
	m.plan = plan
	return plan, true
}

func (m *qwen36StagedModel) NumLayers() int { return m.config.NumHiddenLayers }

func (m *qwen36StagedModel) Tokenizer() *metal.Tokenizer { return m.tokenizer }

func (m *qwen36StagedModel) ModelType() string { return "qwen3_6" }

func (m *qwen36StagedModel) ApplyLoRA(_ metal.LoRAConfig) *metal.LoRAAdapter { return nil }

func (m *qwen36StagedModel) DecodeUnavailableError(operation string) error {
	return core.NewError(operation + ": qwen3_6 staged loader has no native hybrid linear-attention decode kernels yet")
}

func (m *qwen36StagedModel) FillModelInfo(info *metal.ModelInfo) {
	info.VocabSize = m.config.VocabSize
	info.HiddenSize = m.config.HiddenSize
	info.ContextLength = m.config.MaxPositionEmbeddings
	if info.ContextLength == 0 {
		info.ContextLength = m.config.SlidingWindow
	}
	info.QuantBits = m.config.Quantization.Bits
	info.QuantGroup = m.config.Quantization.GroupSize
}

func firstQwen36ArchitectureName(values []string) string {
	if slices.ContainsFunc(values, isQwen36Architecture) {
		return "qwen3_6"
	}
	return ""
}

func qwen36NewHybridCaches(plan metal.HybridAttentionCachePlan) []metal.Cache {
	if plan.GlobalLayers <= 0 {
		return nil
	}
	caches := make([]metal.Cache, plan.GlobalLayers)
	for _, layer := range plan.Layers {
		if !layer.RequiresKV || layer.CacheIndex < 0 || layer.CacheIndex >= len(caches) {
			continue
		}
		caches[layer.CacheIndex] = metal.NewKVCache()
	}
	return caches
}

func (m *qwen36StagedModel) HybridAttentionCachePlan() (metal.HybridAttentionCachePlan, bool) {
	plan, ok := m.qwen36HybridCachePlan()
	if !ok {
		return metal.HybridAttentionCachePlan{}, false
	}
	return plan, true
}

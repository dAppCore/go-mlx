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
	plan      qwen36HybridAttentionPlan
	tokenizer *Tokenizer
}

type qwen36AttentionKind string

const (
	qwen36AttentionLinear qwen36AttentionKind = "linear_attention"
	qwen36AttentionFull   qwen36AttentionKind = "full_attention"
)

type qwen36HybridLayerPlan struct {
	Layer      int
	Kind       qwen36AttentionKind
	Window     int
	RequiresKV bool
	CacheIndex int
}

type qwen36HybridAttentionPlan struct {
	Layers            []qwen36HybridLayerPlan
	CacheIndexByLayer []int
	LinearLayers      int
	FullLayers        int
	LocalWindow       int
}

type qwen36HybridCachePlanner interface {
	qwen36HybridCachePlan() (qwen36HybridAttentionPlan, bool)
}

func loadQwen36StagedModel(modelPath string, configData []byte) (*qwen36StagedModel, error) {
	cfg, err := parseQwen36StagedConfig(configData)
	if err != nil {
		return nil, err
	}
	if err := cfg.validate(); err != nil {
		return nil, err
	}
	plan, err := buildQwen36HybridAttentionPlan(cfg.NumHiddenLayers, cfg.LayerTypes, cfg.SlidingWindow)
	if err != nil {
		return nil, err
	}
	root := resolveModelRoot(modelPath)
	tokenizer, err := LoadTokenizer(core.JoinPath(root, "tokenizer.json"))
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
	if _, err := buildQwen36HybridAttentionPlan(cfg.NumHiddenLayers, cfg.LayerTypes, cfg.SlidingWindow); err != nil {
		return err
	}
	return nil
}

func (m *qwen36StagedModel) Forward(_ *Array, _ []Cache) *Array { return nil }

func (m *qwen36StagedModel) ForwardMasked(_ *Array, _ *Array, _ []Cache) *Array { return nil }

func (m *qwen36StagedModel) NewCache() []Cache {
	plan, ok := m.qwen36HybridCachePlan()
	if !ok {
		return nil
	}
	return qwen36NewHybridCaches(plan)
}

func (m *qwen36StagedModel) qwen36HybridCachePlan() (qwen36HybridAttentionPlan, bool) {
	if len(m.plan.Layers) == m.config.NumHiddenLayers && len(m.plan.CacheIndexByLayer) == m.config.NumHiddenLayers {
		return m.plan, true
	}
	plan, err := buildQwen36HybridAttentionPlan(m.config.NumHiddenLayers, m.config.LayerTypes, m.config.SlidingWindow)
	if err != nil {
		return qwen36HybridAttentionPlan{}, false
	}
	m.plan = plan
	return plan, true
}

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

func buildQwen36HybridAttentionPlan(numLayers int, layerTypes []string, slidingWindow int) (qwen36HybridAttentionPlan, error) {
	if numLayers <= 0 {
		return qwen36HybridAttentionPlan{}, core.NewError("qwen3_6 validation requires positive layer count")
	}
	if len(layerTypes) == 0 {
		return qwen36HybridAttentionPlan{}, core.NewError("qwen3_6 validation requires linear_attention layer metadata")
	}
	pattern := make([]qwen36AttentionKind, 0, len(layerTypes))
	for _, value := range layerTypes {
		kind, ok := parseQwen36AttentionKind(value)
		if !ok {
			return qwen36HybridAttentionPlan{}, core.NewError("qwen3_6 validation unsupported layer type: " + value)
		}
		pattern = append(pattern, kind)
	}
	plan := qwen36HybridAttentionPlan{
		Layers:            make([]qwen36HybridLayerPlan, numLayers),
		CacheIndexByLayer: make([]int, numLayers),
		LocalWindow:       slidingWindow,
	}
	for i := range plan.CacheIndexByLayer {
		plan.CacheIndexByLayer[i] = -1
	}
	for i := 0; i < numLayers; i++ {
		kind := pattern[i%len(pattern)]
		layer := qwen36HybridLayerPlan{
			Layer:      i,
			Kind:       kind,
			Window:     slidingWindow,
			RequiresKV: kind == qwen36AttentionFull,
			CacheIndex: -1,
		}
		if kind == qwen36AttentionLinear {
			plan.LinearLayers++
			layer.Window = 0
		} else {
			layer.CacheIndex = plan.FullLayers
			plan.CacheIndexByLayer[i] = layer.CacheIndex
			plan.FullLayers++
		}
		plan.Layers[i] = layer
	}
	if plan.LinearLayers == 0 {
		return qwen36HybridAttentionPlan{}, core.NewError("qwen3_6 validation requires linear_attention layer metadata")
	}
	if plan.FullLayers == 0 {
		return qwen36HybridAttentionPlan{}, core.NewError("qwen3_6 validation requires full_attention layer metadata")
	}
	return plan, nil
}

func qwen36NewHybridCaches(plan qwen36HybridAttentionPlan) []Cache {
	if plan.FullLayers <= 0 {
		return nil
	}
	caches := make([]Cache, plan.FullLayers)
	for _, layer := range plan.Layers {
		if !layer.RequiresKV || layer.CacheIndex < 0 || layer.CacheIndex >= len(caches) {
			continue
		}
		caches[layer.CacheIndex] = NewKVCache()
	}
	return caches
}

func qwen36AttentionCacheIndexByLayer(model qwen36HybridCachePlanner, numLayers, numCaches int) []int {
	cacheIndexByLayer := make([]int, numLayers)
	for i := range cacheIndexByLayer {
		cacheIndexByLayer[i] = -1
	}
	plan, ok := model.qwen36HybridCachePlan()
	if !ok {
		return cacheIndexByLayer
	}
	for layerIdx := 0; layerIdx < numLayers && layerIdx < len(plan.CacheIndexByLayer); layerIdx++ {
		cacheIdx := plan.CacheIndexByLayer[layerIdx]
		if cacheIdx >= 0 && cacheIdx < numCaches {
			cacheIndexByLayer[layerIdx] = cacheIdx
		}
	}
	return cacheIndexByLayer
}

func parseQwen36AttentionKind(value string) (qwen36AttentionKind, bool) {
	switch normalizeQwen3LayerType(value) {
	case "linear_attention", "linear":
		return qwen36AttentionLinear, true
	case "full_attention", "global_attention", "attention", "full":
		return qwen36AttentionFull, true
	default:
		return "", false
	}
}

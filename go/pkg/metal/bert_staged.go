// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import "dappco.re/go"

type bertStagedConfig struct {
	ModelType             string   `json:"model_type,omitempty"`
	Architectures         []string `json:"architectures,omitempty"`
	VocabSize             int      `json:"vocab_size,omitempty"`
	HiddenSize            int      `json:"hidden_size,omitempty"`
	NumHiddenLayers       int      `json:"num_hidden_layers,omitempty"`
	NumAttentionHeads     int      `json:"num_attention_heads,omitempty"`
	IntermediateSize      int      `json:"intermediate_size,omitempty"`
	MaxPositionEmbeddings int      `json:"max_position_embeddings,omitempty"`
	TypeVocabSize         int      `json:"type_vocab_size,omitempty"`
	NumLabels             int      `json:"num_labels,omitempty"`
}

type bertStagedModel struct {
	path      string
	config    bertStagedConfig
	modelType string
	tokenizer *Tokenizer
}

type bertPoolingMode string

const (
	bertPoolingCLS  bertPoolingMode = "cls"
	bertPoolingMean bertPoolingMode = "mean"
)

type bertRerankHead struct {
	Classifier *Linear
	PoolMode   bertPoolingMode
}

func loadBERTStagedModel(modelPath string, configData []byte, modelType string) (*bertStagedModel, error) {
	cfg, err := parseBERTStagedConfig(configData, modelType)
	if err != nil {
		return nil, err
	}
	if err := cfg.validate(modelType); err != nil {
		return nil, err
	}
	root := ResolveModelRoot(modelPath)
	tokenizer, err := LoadTokenizer(core.JoinPath(root, "tokenizer.json"))
	if err != nil {
		return nil, core.E("bert.load", "load tokenizer", err)
	}
	return &bertStagedModel{
		path:      root,
		config:    cfg,
		modelType: modelType,
		tokenizer: tokenizer,
	}, nil
}

func parseBERTStagedConfig(data []byte, modelType string) (bertStagedConfig, error) {
	var cfg bertStagedConfig
	if result := core.JSONUnmarshal(data, &cfg); !result.OK {
		return bertStagedConfig{}, result.Value.(error)
	}
	if modelType == "" {
		modelType = normalizeProbeModelType(firstNonEmptyString(cfg.ModelType, firstBERTArchitectureName(cfg.Architectures)))
	}
	cfg.ModelType = modelType
	return cfg, nil
}

func (cfg bertStagedConfig) validate(modelType string) error {
	if modelType != "bert" && modelType != "bert_rerank" {
		return core.NewError("bert validation requires bert or bert_rerank config")
	}
	if cfg.HiddenSize <= 0 || cfg.NumHiddenLayers <= 0 || cfg.VocabSize <= 0 {
		return core.NewError("bert validation requires hidden size, layer count, and vocab size")
	}
	if cfg.MaxPositionEmbeddings <= 0 {
		return core.NewError("bert validation requires max_position_embeddings")
	}
	if modelType == "bert_rerank" && cfg.NumLabels <= 0 {
		return core.NewError("bert_rerank validation requires num_labels")
	}
	return nil
}

func (m *bertStagedModel) Forward(_ *Array, _ []Cache) *Array { return nil }

func (m *bertStagedModel) ForwardMasked(_ *Array, _ *Array, _ []Cache) *Array { return nil }

func (m *bertStagedModel) NewCache() []Cache { return nil }

func (m *bertStagedModel) NumLayers() int { return m.config.NumHiddenLayers }

func (m *bertStagedModel) Tokenizer() *Tokenizer { return m.tokenizer }

func (m *bertStagedModel) ModelType() string { return m.modelType }

func (m *bertStagedModel) ApplyLoRA(_ LoRAConfig) *LoRAAdapter { return nil }

func bertPoolCLS(hidden *Array) (*Array, bool) {
	if hidden == nil || !hidden.Valid() || hidden.NumDims() != 3 || hidden.Dim(1) <= 0 {
		return nil, false
	}
	indices := FromValues([]int32{0}, 1)
	selected := Take(hidden, indices, 1)
	pooled := Squeeze(selected, 1)
	Free(indices, selected)
	return pooled, true
}

func bertPoolMean(hidden, attentionMask *Array) (*Array, bool) {
	if hidden == nil || !hidden.Valid() || hidden.NumDims() != 3 || hidden.Dim(1) <= 0 {
		return nil, false
	}
	if attentionMask == nil || !attentionMask.Valid() {
		return Mean(hidden, 1, false), true
	}
	if attentionMask.NumDims() != 2 ||
		attentionMask.Dim(0) != hidden.Dim(0) ||
		attentionMask.Dim(1) != hidden.Dim(1) {
		return nil, false
	}
	maskFloat := AsType(attentionMask, hidden.Dtype())
	maskExpanded := ExpandDims(maskFloat, -1)
	masked := Mul(hidden, maskExpanded)
	summed := Sum(masked, 1, false)
	counts := Sum(maskExpanded, 1, false)
	minCount := FromValue(float32(1))
	safeCounts := Maximum(counts, minCount)
	pooled := Divide(summed, safeCounts)
	Free(maskFloat, maskExpanded, masked, summed, counts, minCount, safeCounts)
	return pooled, true
}

func (head bertRerankHead) Score(hidden, attentionMask *Array) (*Array, bool) {
	if head.Classifier == nil || head.Classifier.Weight == nil || !head.Classifier.Weight.Valid() {
		return nil, false
	}
	mode := head.PoolMode
	if mode == "" {
		mode = bertPoolingCLS
	}
	var pooled *Array
	var ok bool
	switch mode {
	case bertPoolingCLS:
		pooled, ok = bertPoolCLS(hidden)
	case bertPoolingMean:
		pooled, ok = bertPoolMean(hidden, attentionMask)
	default:
		return nil, false
	}
	if !ok {
		return nil, false
	}
	logits := head.Classifier.Forward(pooled)
	Free(pooled)
	return logits, true
}

func firstBERTArchitectureName(values []string) string {
	for _, value := range values {
		if architecturesContainRerankModel([]string{value}) {
			return "bert_rerank"
		}
		if core.Contains(value, "Bert") {
			return "bert"
		}
	}
	return ""
}

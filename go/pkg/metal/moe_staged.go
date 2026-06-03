// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import "dappco.re/go"

type moeStagedConfig struct {
	ModelType             string             `json:"model_type,omitempty"`
	Architectures         []string           `json:"architectures,omitempty"`
	HiddenSize            int                `json:"hidden_size,omitempty"`
	NumHiddenLayers       int                `json:"num_hidden_layers,omitempty"`
	NumAttentionHeads     int                `json:"num_attention_heads,omitempty"`
	NumKeyValueHeads      int                `json:"num_key_value_heads,omitempty"`
	VocabSize             int                `json:"vocab_size,omitempty"`
	MaxPositionEmbeddings int                `json:"max_position_embeddings,omitempty"`
	NumExperts            int                `json:"num_experts,omitempty"`
	NumLocalExperts       int                `json:"num_local_experts,omitempty"`
	NRoutedExperts        int                `json:"n_routed_experts,omitempty"`
	NumExpertsPerTok      int                `json:"num_experts_per_tok,omitempty"`
	MoEIntermediateSize   int                `json:"moe_intermediate_size,omitempty"`
	IntermediateSize      int                `json:"intermediate_size,omitempty"`
	QLoRARank             int                `json:"q_lora_rank,omitempty"`
	KVLoRARank            int                `json:"kv_lora_rank,omitempty"`
	QKNoPEHeadDim         int                `json:"qk_nope_head_dim,omitempty"`
	QKRoPEHeadDim         int                `json:"qk_rope_head_dim,omitempty"`
	QKHeadDim             int                `json:"qk_head_dim,omitempty"`
	VHeadDim              int                `json:"v_head_dim,omitempty"`
	Quantization          QuantizationConfig `json:"quantization,omitempty"`
}

type deepSeekMLAPlan struct {
	QueryLoRARank int
	KVLoRARank    int
	QKNoPEHeadDim int
	QKRoPEHeadDim int
	QKHeadDim     int
	VHeadDim      int
}

type moeStagedModel struct {
	path      string
	config    moeStagedConfig
	mla       deepSeekMLAPlan
	modelType string
	tokenizer *Tokenizer
}

func (cfg moeStagedConfig) expertCount() int {
	return firstPositiveInt(cfg.NumExperts, cfg.NumLocalExperts, cfg.NRoutedExperts)
}

func (cfg moeStagedConfig) intermediateSize() int {
	return firstPositiveInt(cfg.MoEIntermediateSize, cfg.IntermediateSize)
}

func (cfg moeStagedConfig) deepSeekMLAPlan() (deepSeekMLAPlan, error) {
	qkHeadDim := cfg.QKHeadDim
	if qkHeadDim == 0 && (cfg.QKNoPEHeadDim > 0 || cfg.QKRoPEHeadDim > 0) {
		qkHeadDim = cfg.QKNoPEHeadDim + cfg.QKRoPEHeadDim
	}
	plan := deepSeekMLAPlan{
		QueryLoRARank: cfg.QLoRARank,
		KVLoRARank:    cfg.KVLoRARank,
		QKNoPEHeadDim: cfg.QKNoPEHeadDim,
		QKRoPEHeadDim: cfg.QKRoPEHeadDim,
		QKHeadDim:     qkHeadDim,
		VHeadDim:      cfg.VHeadDim,
	}
	if plan.KVLoRARank <= 0 {
		return deepSeekMLAPlan{}, core.NewError("deepseek validation requires kv_lora_rank")
	}
	if plan.QKNoPEHeadDim <= 0 || plan.QKRoPEHeadDim <= 0 {
		return deepSeekMLAPlan{}, core.NewError("deepseek validation requires qk_nope_head_dim and qk_rope_head_dim")
	}
	if plan.QKHeadDim <= 0 || plan.VHeadDim <= 0 {
		return deepSeekMLAPlan{}, core.NewError("deepseek validation requires qk_head_dim and v_head_dim")
	}
	if plan.QKHeadDim != plan.QKNoPEHeadDim+plan.QKRoPEHeadDim {
		return deepSeekMLAPlan{}, core.NewError("deepseek validation requires qk_head_dim to equal qk_nope_head_dim + qk_rope_head_dim")
	}
	return plan, nil
}

func loadMoEStagedModel(modelPath string, configData []byte, modelType string) (*moeStagedModel, error) {
	cfg, err := parseMoEStagedConfig(configData, modelType)
	if err != nil {
		return nil, err
	}
	if err := cfg.validate(modelType); err != nil {
		return nil, err
	}
	var mla deepSeekMLAPlan
	if modelType == "deepseek" {
		mla, err = cfg.deepSeekMLAPlan()
		if err != nil {
			return nil, err
		}
	}
	root := resolveModelRoot(modelPath)
	tokenizer, err := LoadTokenizer(core.JoinPath(root, "tokenizer.json"))
	if err != nil {
		return nil, core.E(modelType+".load", "load tokenizer", err)
	}
	return &moeStagedModel{
		path:      root,
		config:    cfg,
		mla:       mla,
		modelType: modelType,
		tokenizer: tokenizer,
	}, nil
}

func parseMoEStagedConfig(data []byte, modelType string) (moeStagedConfig, error) {
	var cfg moeStagedConfig
	if result := core.JSONUnmarshal(data, &cfg); !result.OK {
		return moeStagedConfig{}, result.Value.(error)
	}
	detected := normalizeProbeModelType(firstNonEmptyString(cfg.ModelType, firstMoEArchitectureName(cfg.Architectures)))
	if detected == "" && modelType != "" {
		detected = modelType
	}
	if detected != modelType {
		return moeStagedConfig{}, core.NewError(modelType + " validation requires " + modelType + " config")
	}
	cfg.ModelType = modelType
	return cfg, nil
}

func (cfg moeStagedConfig) validate(modelType string) error {
	if cfg.HiddenSize <= 0 || cfg.NumHiddenLayers <= 0 || cfg.VocabSize <= 0 {
		return core.NewError(modelType + " validation requires hidden size, layer count, and vocab size")
	}
	if cfg.NumAttentionHeads <= 0 || cfg.NumKeyValueHeads <= 0 {
		return core.NewError(modelType + " validation requires attention and key/value head counts")
	}
	if cfg.expertCount() <= 0 {
		return core.NewError(modelType + " validation requires expert count")
	}
	if modelType == "deepseek" {
		if _, err := cfg.deepSeekMLAPlan(); err != nil {
			return err
		}
	}
	return nil
}

func (m *moeStagedModel) Forward(_ *Array, _ []Cache) *Array { return nil }

func (m *moeStagedModel) ForwardMasked(_ *Array, _ *Array, _ []Cache) *Array { return nil }

func (m *moeStagedModel) NewCache() []Cache { return nil }

func (m *moeStagedModel) NumLayers() int { return m.config.NumHiddenLayers }

func (m *moeStagedModel) Tokenizer() *Tokenizer { return m.tokenizer }

func (m *moeStagedModel) ModelType() string { return m.modelType }

func (m *moeStagedModel) ApplyLoRA(_ LoRAConfig) *LoRAAdapter { return nil }

func firstMoEArchitectureName(values []string) string {
	for _, value := range values {
		compact := compactArchitectureName(value)
		switch {
		case core.Contains(compact, "mixtral"):
			return "mixtral"
		case core.Contains(compact, "deepseek"):
			return "deepseek"
		case core.Contains(compact, "gptoss"):
			return "gpt_oss"
		case core.Contains(compact, "kimi") || core.Contains(compact, "moonshot"):
			return "kimi"
		}
	}
	return ""
}

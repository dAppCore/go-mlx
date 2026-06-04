// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package deepseek

import (
	"dappco.re/go"
	"dappco.re/go/mlx/pkg/metal"
)

type stagedConfig struct {
	ModelType             string                   `json:"model_type,omitempty"`
	Architectures         []string                 `json:"architectures,omitempty"`
	HiddenSize            int                      `json:"hidden_size,omitempty"`
	NumHiddenLayers       int                      `json:"num_hidden_layers,omitempty"`
	NumAttentionHeads     int                      `json:"num_attention_heads,omitempty"`
	NumKeyValueHeads      int                      `json:"num_key_value_heads,omitempty"`
	VocabSize             int                      `json:"vocab_size,omitempty"`
	MaxPositionEmbeddings int                      `json:"max_position_embeddings,omitempty"`
	NumExperts            int                      `json:"num_experts,omitempty"`
	NumLocalExperts       int                      `json:"num_local_experts,omitempty"`
	NRoutedExperts        int                      `json:"n_routed_experts,omitempty"`
	NumExpertsPerTok      int                      `json:"num_experts_per_tok,omitempty"`
	MoEIntermediateSize   int                      `json:"moe_intermediate_size,omitempty"`
	IntermediateSize      int                      `json:"intermediate_size,omitempty"`
	QLoRARank             int                      `json:"q_lora_rank,omitempty"`
	KVLoRARank            int                      `json:"kv_lora_rank,omitempty"`
	QKNoPEHeadDim         int                      `json:"qk_nope_head_dim,omitempty"`
	QKRoPEHeadDim         int                      `json:"qk_rope_head_dim,omitempty"`
	QKHeadDim             int                      `json:"qk_head_dim,omitempty"`
	VHeadDim              int                      `json:"v_head_dim,omitempty"`
	Quantization          metal.QuantizationConfig `json:"quantization"`
}

type deepSeekMLAPlan struct {
	QueryLoRARank int
	KVLoRARank    int
	QKNoPEHeadDim int
	QKRoPEHeadDim int
	QKHeadDim     int
	VHeadDim      int
}

type StagedModel struct {
	path      string
	config    stagedConfig
	mla       deepSeekMLAPlan
	tokenizer *metal.Tokenizer
}

func init() {
	metal.RegisterModelLoader("deepseek", func(modelPath string, configData []byte) (metal.InternalModel, error) {
		model, err := loadStagedModel(modelPath, configData)
		if err != nil {
			return nil, core.E("model.loadModel", "validate deepseek native load", err)
		}
		return model, nil
	})
}

func (cfg stagedConfig) expertCount() int {
	return metal.FirstPositiveInt(cfg.NumExperts, cfg.NumLocalExperts, cfg.NRoutedExperts)
}

func (cfg stagedConfig) deepSeekMLAPlan() (deepSeekMLAPlan, error) {
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

func loadStagedModel(modelPath string, configData []byte) (*StagedModel, error) {
	cfg, err := parseStagedConfig(configData)
	if err != nil {
		return nil, err
	}
	if err := cfg.validate(); err != nil {
		return nil, err
	}
	mla, err := cfg.deepSeekMLAPlan()
	if err != nil {
		return nil, err
	}
	root := metal.ResolveModelRoot(modelPath)
	tokenizer, err := metal.LoadTokenizer(core.JoinPath(root, "tokenizer.json"))
	if err != nil {
		return nil, core.E("deepseek.load", "load tokenizer", err)
	}
	return &StagedModel{
		path:      root,
		config:    cfg,
		mla:       mla,
		tokenizer: tokenizer,
	}, nil
}

func parseStagedConfig(data []byte) (stagedConfig, error) {
	var cfg stagedConfig
	if result := core.JSONUnmarshal(data, &cfg); !result.OK {
		return stagedConfig{}, result.Value.(error)
	}
	detected := metal.NormalizeProbeModelType(metal.FirstNonEmptyString(cfg.ModelType, firstDeepSeekArchitectureName(cfg.Architectures)))
	if detected == "" {
		detected = "deepseek"
	}
	if detected != "deepseek" {
		return stagedConfig{}, core.NewError("deepseek validation requires deepseek config")
	}
	cfg.ModelType = "deepseek"
	return cfg, nil
}

func (cfg stagedConfig) validate() error {
	if cfg.HiddenSize <= 0 || cfg.NumHiddenLayers <= 0 || cfg.VocabSize <= 0 {
		return core.NewError("deepseek validation requires hidden size, layer count, and vocab size")
	}
	if cfg.NumAttentionHeads <= 0 || cfg.NumKeyValueHeads <= 0 {
		return core.NewError("deepseek validation requires attention and key/value head counts")
	}
	if cfg.expertCount() <= 0 {
		return core.NewError("deepseek validation requires expert count")
	}
	if _, err := cfg.deepSeekMLAPlan(); err != nil {
		return err
	}
	return nil
}

func (m *StagedModel) Forward(_ *metal.Array, _ []metal.Cache) *metal.Array { return nil }

func (m *StagedModel) ForwardMasked(_ *metal.Array, _ *metal.Array, _ []metal.Cache) *metal.Array {
	return nil
}

func (m *StagedModel) NewCache() []metal.Cache { return nil }

func (m *StagedModel) NumLayers() int { return m.config.NumHiddenLayers }

func (m *StagedModel) Tokenizer() *metal.Tokenizer { return m.tokenizer }

func (m *StagedModel) ModelType() string { return "deepseek" }

func (m *StagedModel) ApplyLoRA(_ metal.LoRAConfig) *metal.LoRAAdapter { return nil }

func (m *StagedModel) FillModelInfo(info *metal.ModelInfo) {
	info.VocabSize = m.config.VocabSize
	info.HiddenSize = m.config.HiddenSize
	info.ContextLength = m.config.MaxPositionEmbeddings
	info.QuantBits = m.config.Quantization.Bits
	info.QuantGroup = m.config.Quantization.GroupSize
}

func (m *StagedModel) DecodeUnavailableError(operation string) error {
	return core.NewError(operation + ": deepseek staged loader has no native sparse-expert decode kernels yet")
}

func firstDeepSeekArchitectureName(values []string) string {
	for _, value := range values {
		if core.Contains(metal.CompactArchitectureName(value), "deepseek") {
			return "deepseek"
		}
	}
	return ""
}

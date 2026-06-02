// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import "dappco.re/go"

type qwen3MoEStagedModel struct {
	path      string
	config    *Qwen3Config
	tokenizer *Tokenizer
}

func loadQwen3MoEStagedModel(modelPath string, configData []byte) (*qwen3MoEStagedModel, error) {
	cfg, err := parseQwen3Config(configData)
	if err != nil {
		return nil, core.E("qwen3_moe.load", "parse config", err)
	}
	if err := validateQwen3MoEStagedConfig(cfg); err != nil {
		return nil, err
	}
	root := resolveModelRoot(modelPath)
	tokenizer, err := LoadTokenizer(core.JoinPath(root, "tokenizer.json"))
	if err != nil {
		return nil, core.E("qwen3_moe.load", "load tokenizer", err)
	}
	return &qwen3MoEStagedModel{path: root, config: cfg, tokenizer: tokenizer}, nil
}

func validateQwen3MoEStagedConfig(cfg *Qwen3Config) error {
	if cfg == nil {
		return core.NewError("qwen3_moe validation requires config")
	}
	if normalizeProbeModelType(cfg.ModelType) != "qwen3_moe" {
		return core.NewError("qwen3_moe validation requires qwen3_moe config")
	}
	if cfg.IsQwen36Hybrid() {
		return core.NewError("qwen3_moe validation excludes qwen3_6 hybrid linear-attention configs")
	}
	if !cfg.IsMoE() {
		return core.NewError("qwen3_moe validation requires sparse expert metadata")
	}
	if cfg.HiddenSize <= 0 || cfg.NumHiddenLayers <= 0 || cfg.VocabSize <= 0 {
		return core.NewError("qwen3_moe validation requires hidden size, layer count, and vocab size")
	}
	if cfg.NumAttentionHeads <= 0 || cfg.NumKeyValueHeads <= 0 {
		return core.NewError("qwen3_moe validation requires attention and key/value head counts")
	}
	if cfg.NumExperts <= 0 || cfg.NumExpertsPerTok <= 0 || cfg.MoEIntermediateSize <= 0 {
		return core.NewError("qwen3_moe validation requires expert count, experts-per-token, and moe intermediate size")
	}
	return nil
}

func (m *qwen3MoEStagedModel) Forward(_ *Array, _ []Cache) *Array { return nil }

func (m *qwen3MoEStagedModel) ForwardMasked(_ *Array, _ *Array, _ []Cache) *Array { return nil }

func (m *qwen3MoEStagedModel) NewCache() []Cache { return nil }

func (m *qwen3MoEStagedModel) NumLayers() int { return int(m.config.NumHiddenLayers) }

func (m *qwen3MoEStagedModel) Tokenizer() *Tokenizer { return m.tokenizer }

func (m *qwen3MoEStagedModel) ModelType() string { return "qwen3_moe" }

func (m *qwen3MoEStagedModel) ApplyLoRA(_ LoRAConfig) *LoRAAdapter { return nil }

// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import "dappco.re/go"

type qwen36MoEStagedModel struct {
	path      string
	config    *Qwen3Config
	tokenizer *Tokenizer
}

func loadQwen36MoEStagedModel(modelPath string, configData []byte) (*qwen36MoEStagedModel, error) {
	cfg, err := parseQwen3Config(configData)
	if err != nil {
		return nil, core.E("qwen3_6_moe.load", "parse config", err)
	}
	if err := validateQwen36MoEStagedConfig(cfg); err != nil {
		return nil, err
	}
	root := resolveModelRoot(modelPath)
	tokenizer, err := LoadTokenizer(core.JoinPath(root, "tokenizer.json"))
	if err != nil {
		return nil, core.E("qwen3_6_moe.load", "load tokenizer", err)
	}
	return &qwen36MoEStagedModel{path: root, config: cfg, tokenizer: tokenizer}, nil
}

func validateQwen36MoEStagedConfig(cfg *Qwen3Config) error {
	if cfg == nil {
		return core.NewError("qwen3_6_moe validation requires config")
	}
	if normalizeProbeModelType(cfg.ModelType) != "qwen3_6_moe" {
		return core.NewError("qwen3_6_moe validation requires qwen3_6_moe config")
	}
	if !cfg.IsMoE() {
		return core.NewError("qwen3_6_moe validation requires sparse expert metadata")
	}
	if !qwen36LayerTypesIncludeLinearAttention(cfg.LayerTypes) {
		return core.NewError("qwen3_6_moe validation requires linear_attention layer metadata")
	}
	if cfg.HiddenSize <= 0 || cfg.NumHiddenLayers <= 0 || cfg.VocabSize <= 0 {
		return core.NewError("qwen3_6_moe validation requires hidden size, layer count, and vocab size")
	}
	if cfg.NumAttentionHeads <= 0 || cfg.NumKeyValueHeads <= 0 {
		return core.NewError("qwen3_6_moe validation requires attention and key/value head counts")
	}
	if cfg.NumExperts <= 0 || cfg.NumExpertsPerTok <= 0 || cfg.MoEIntermediateSize <= 0 {
		return core.NewError("qwen3_6_moe validation requires expert count, experts-per-token, and moe intermediate size")
	}
	return nil
}

func (m *qwen36MoEStagedModel) Forward(_ *Array, _ []Cache) *Array { return nil }

func (m *qwen36MoEStagedModel) ForwardMasked(_ *Array, _ *Array, _ []Cache) *Array { return nil }

func (m *qwen36MoEStagedModel) NewCache() []Cache { return nil }

func (m *qwen36MoEStagedModel) NumLayers() int { return int(m.config.NumHiddenLayers) }

func (m *qwen36MoEStagedModel) Tokenizer() *Tokenizer { return m.tokenizer }

func (m *qwen36MoEStagedModel) ModelType() string { return "qwen3_6_moe" }

func (m *qwen36MoEStagedModel) ApplyLoRA(_ LoRAConfig) *LoRAAdapter { return nil }

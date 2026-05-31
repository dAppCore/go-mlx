// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	core "dappco.re/go"
	modelinspect "dappco.re/go/mlx/model"
	mp "dappco.re/go/mlx/pack"
)

// OfficialGemma4E2BModelContract records the Gemma 4 metadata points that must
// stay aligned while the official Google E2B target replaces the archived q4
// control pack. It is deliberately metadata-only so it can run before a
// heavyweight native load or benchmark.
type OfficialGemma4E2BModelContract struct {
	ModelID                    string `json:"model_id"`
	Revision                   string `json:"revision,omitempty"`
	Path                       string `json:"path,omitempty"`
	Architecture               string `json:"architecture,omitempty"`
	ModelType                  string `json:"model_type,omitempty"`
	PackArchitecture           string `json:"pack_architecture,omitempty"`
	NativeLoadable             bool   `json:"native_loadable"`
	HasTokenizer               bool   `json:"has_tokenizer"`
	HasChatTemplate            bool   `json:"has_chat_template"`
	QuantBits                  int    `json:"quant_bits,omitempty"`
	QuantGroup                 int    `json:"quant_group,omitempty"`
	ContextLength              int    `json:"context_length,omitempty"`
	LayerCount                 int    `json:"layer_count,omitempty"`
	HiddenSize                 int    `json:"hidden_size,omitempty"`
	VocabSize                  int    `json:"vocab_size,omitempty"`
	SlidingWindow              int    `json:"sliding_window,omitempty"`
	SlidingAttentionLayers     int    `json:"sliding_attention_layers,omitempty"`
	FullAttentionLayers        int    `json:"full_attention_layers,omitempty"`
	FullAttentionInterval      int    `json:"full_attention_interval,omitempty"`
	AttentionPattern           string `json:"attention_pattern,omitempty"`
	FullRoPETheta              int    `json:"full_rope_theta,omitempty"`
	FullRoPEType               string `json:"full_rope_type,omitempty"`
	FullPartialRotaryFactorPct int    `json:"full_partial_rotary_factor_pct,omitempty"`
	SlidingRoPETheta           int    `json:"sliding_rope_theta,omitempty"`
	SlidingRoPEType            string `json:"sliding_rope_type,omitempty"`
	ProportionalRoPE           bool   `json:"proportional_rope"`
	NumKVSharedLayers          int    `json:"num_kv_shared_layers,omitempty"`
	PerLayerInputs             bool   `json:"per_layer_inputs"`
	HiddenSizePerLayerInput    int    `json:"hidden_size_per_layer_input,omitempty"`
	VocabSizePerLayerInput     int    `json:"vocab_size_per_layer_input,omitempty"`
	ChatTemplateSource         string `json:"chat_template_source,omitempty"`
	ChatTemplateName           string `json:"chat_template_name,omitempty"`
	HasThinkingToken           bool   `json:"has_thinking_token"`
	HasThoughtChannelMarkers   bool   `json:"has_thought_channel_markers"`
	StripsThinking             bool   `json:"strips_thinking"`
}

// OfficialGemma4E2BControlComparison compares the official target snapshot with
// the archived q4 baseline. Quantisation is expected to differ; architecture,
// context, Gemma 4 attention/RoPE/PLE/shared-KV and chat-template semantics are
// expected to match.
type OfficialGemma4E2BControlComparison struct {
	Version                 int                            `json:"version"`
	Target                  OfficialGemma4E2BModelContract `json:"target"`
	Control                 OfficialGemma4E2BModelContract `json:"control"`
	Compatible              bool                           `json:"compatible"`
	QuantizationDiffers     bool                           `json:"quantization_differs"`
	ArchitectureCompatible  bool                           `json:"architecture_compatible"`
	ContextCompatible       bool                           `json:"context_compatible"`
	AttentionCompatible     bool                           `json:"attention_compatible"`
	RoPECompatible          bool                           `json:"rope_compatible"`
	SharedKVCompatible      bool                           `json:"shared_kv_compatible"`
	PerLayerInputCompatible bool                           `json:"per_layer_input_compatible"`
	ChatTemplateCompatible  bool                           `json:"chat_template_compatible"`
	RetainedStateCompatible bool                           `json:"retained_state_compatible"`
	PromptCacheCompatible   bool                           `json:"prompt_cache_compatible"`
	Issues                  []string                       `json:"issues,omitempty"`
}

// CompareOfficialGemma4E2BControlSnapshots verifies the locked official target
// snapshot and compares it with the archived q4 control metadata.
func CompareOfficialGemma4E2BControlSnapshots(targetDir, controlDir string, targetLock OfficialGemma4E2BLock) (OfficialGemma4E2BControlComparison, error) {
	report := OfficialGemma4E2BControlComparison{Version: 1}

	targetPreflight, err := InspectOfficialGemma4E2BLocalSnapshot(targetDir, targetLock)
	if err != nil {
		report.Target = OfficialGemma4E2BModelContract{
			ModelID:  targetLock.ModelID,
			Revision: targetLock.Revision,
			Path:     targetDir,
		}
		report.Issues = append(report.Issues, "target official snapshot failed preflight: "+err.Error())
		return report, err
	}
	targetContract, err := officialGemma4E2BContractFromSnapshot(targetPreflight.SnapshotDir, targetLock.ModelID, targetLock.Revision, targetPreflight.Pack)
	if err != nil {
		report.Target = targetContract
		report.Issues = append(report.Issues, "target metadata read failed: "+err.Error())
		return report, err
	}
	report.Target = targetContract

	controlPack, err := modelinspect.Inspect(controlDir, mp.WithPackRequireChatTemplate(false))
	if err != nil {
		report.Control = OfficialGemma4E2BModelContract{
			ModelID: ProductionLaneArchivedBaselineModelID,
			Path:    controlDir,
		}
		report.Issues = append(report.Issues, "control snapshot inspection failed: "+err.Error())
		return report, err
	}
	controlContract, err := officialGemma4E2BContractFromSnapshot(controlDir, ProductionLaneArchivedBaselineModelID, "", controlPack)
	if err != nil {
		report.Control = controlContract
		report.Issues = append(report.Issues, "control metadata read failed: "+err.Error())
		return report, err
	}
	report.Control = controlContract

	officialGemma4CompareContracts(&report)
	if report.Compatible {
		return report, nil
	}
	return report, core.NewError("mlx: official Gemma 4 E2B target does not match archived q4 control metadata")
}

type officialGemma4ComparisonConfig struct {
	ModelType          string                                `json:"model_type"`
	Architectures      []string                              `json:"architectures"`
	ChatTemplateJinja  string                                `json:"chat_template_jinja"`
	Quantization       *officialGemma4ComparisonQuantization `json:"quantization"`
	QuantizationConfig *officialGemma4ComparisonQuantization `json:"quantization_config"`
	TextConfig         *officialGemma4ComparisonTextConfig   `json:"text_config"`
}

type officialGemma4ComparisonTextConfig struct {
	ModelType               string                                  `json:"model_type"`
	VocabSize               int                                     `json:"vocab_size"`
	VocabSizePerLayerInput  int                                     `json:"vocab_size_per_layer_input"`
	HiddenSize              int                                     `json:"hidden_size"`
	HiddenSizePerLayerInput int                                     `json:"hidden_size_per_layer_input"`
	NumHiddenLayers         int                                     `json:"num_hidden_layers"`
	NumKVSharedLayers       int                                     `json:"num_kv_shared_layers"`
	MaxPositionEmbeddings   int                                     `json:"max_position_embeddings"`
	SlidingWindow           int                                     `json:"sliding_window"`
	LayerTypes              []string                                `json:"layer_types"`
	RopeParameters          map[string]officialGemma4ComparisonRoPE `json:"rope_parameters"`
	Quantization            *officialGemma4ComparisonQuantization   `json:"quantization"`
	QuantizationConfig      *officialGemma4ComparisonQuantization   `json:"quantization_config"`
}

type officialGemma4ComparisonRoPE struct {
	PartialRotaryFactor float64 `json:"partial_rotary_factor"`
	RopeTheta           float64 `json:"rope_theta"`
	RopeType            string  `json:"rope_type"`
}

type officialGemma4ComparisonQuantization struct {
	Bits      int `json:"bits"`
	GroupSize int `json:"group_size"`
}

func officialGemma4E2BContractFromSnapshot(snapshotDir, modelID, revision string, pack mp.ModelPack) (OfficialGemma4E2BModelContract, error) {
	contract := OfficialGemma4E2BModelContract{
		ModelID:            modelID,
		Revision:           revision,
		Path:               snapshotDir,
		PackArchitecture:   pack.Architecture,
		NativeLoadable:     pack.NativeLoadable,
		HasTokenizer:       pack.HasTokenizer,
		HasChatTemplate:    pack.HasChatTemplate,
		QuantBits:          pack.QuantBits,
		QuantGroup:         pack.QuantGroup,
		ContextLength:      pack.ContextLength,
		LayerCount:         pack.NumLayers,
		HiddenSize:         pack.HiddenSize,
		VocabSize:          pack.VocabSize,
		ChatTemplateSource: string(pack.ChatTemplateSource),
		ChatTemplateName:   pack.ChatTemplate,
	}
	config, err := officialGemma4ReadComparisonConfig(snapshotDir)
	if err != nil {
		return contract, err
	}
	text := config.TextConfig
	if text == nil {
		text = &officialGemma4ComparisonTextConfig{}
	}
	contract.ModelType = config.ModelType
	if len(config.Architectures) > 0 {
		contract.Architecture = config.Architectures[0]
	}
	quant := officialGemma4ComparisonQuant(config, text)
	if quant != nil {
		contract.QuantBits = firstPositiveLocal(quant.Bits, contract.QuantBits)
		contract.QuantGroup = firstPositiveLocal(quant.GroupSize, contract.QuantGroup)
	}
	contract.ContextLength = firstPositiveLocal(text.MaxPositionEmbeddings, contract.ContextLength)
	contract.LayerCount = firstPositiveLocal(text.NumHiddenLayers, contract.LayerCount)
	contract.HiddenSize = firstPositiveLocal(text.HiddenSize, contract.HiddenSize)
	contract.VocabSize = firstPositiveLocal(text.VocabSize, contract.VocabSize)
	contract.SlidingWindow = text.SlidingWindow
	contract.NumKVSharedLayers = text.NumKVSharedLayers
	contract.HiddenSizePerLayerInput = text.HiddenSizePerLayerInput
	contract.VocabSizePerLayerInput = text.VocabSizePerLayerInput
	contract.PerLayerInputs = contract.HiddenSizePerLayerInput > 0 || contract.VocabSizePerLayerInput > 0
	contract.SlidingAttentionLayers, contract.FullAttentionLayers, contract.FullAttentionInterval = officialGemma4AttentionPattern(text.LayerTypes)
	if contract.FullAttentionInterval > 0 {
		contract.AttentionPattern = core.Sprintf("full_every_%d", contract.FullAttentionInterval)
	}
	if params, ok := text.RopeParameters["full_attention"]; ok {
		contract.FullRoPETheta = int(params.RopeTheta)
		contract.FullRoPEType = params.RopeType
		contract.FullPartialRotaryFactorPct = int(params.PartialRotaryFactor * 100)
		contract.ProportionalRoPE = params.RopeType == "proportional"
	}
	if params, ok := text.RopeParameters["sliding_attention"]; ok {
		contract.SlidingRoPETheta = int(params.RopeTheta)
		contract.SlidingRoPEType = params.RopeType
	}
	template := officialGemma4ReadComparisonTemplate(snapshotDir, config)
	contract.HasThinkingToken = core.Contains(template, "<|think|>")
	contract.HasThoughtChannelMarkers = core.Contains(template, "<|channel>thought") && core.Contains(template, "<channel|>")
	contract.StripsThinking = core.Contains(template, "strip_thinking") ||
		core.Contains(template, "replace('<|channel>thought'") ||
		core.Contains(template, "replace(\"<|channel>thought\"")
	return contract, nil
}

func officialGemma4ReadComparisonConfig(snapshotDir string) (officialGemma4ComparisonConfig, error) {
	var config officialGemma4ComparisonConfig
	read := core.ReadFile(core.PathJoin(snapshotDir, "config.json"))
	if !read.OK {
		return config, officialGemma4ResultError(read)
	}
	if result := core.JSONUnmarshal(read.Value.([]byte), &config); !result.OK {
		return config, officialGemma4ResultError(result)
	}
	return config, nil
}

func officialGemma4ComparisonQuant(config officialGemma4ComparisonConfig, text *officialGemma4ComparisonTextConfig) *officialGemma4ComparisonQuantization {
	if text != nil {
		if text.QuantizationConfig != nil {
			return text.QuantizationConfig
		}
		if text.Quantization != nil {
			return text.Quantization
		}
	}
	if config.QuantizationConfig != nil {
		return config.QuantizationConfig
	}
	return config.Quantization
}

func officialGemma4ReadComparisonTemplate(snapshotDir string, config officialGemma4ComparisonConfig) string {
	if read := core.ReadFile(core.PathJoin(snapshotDir, "chat_template.jinja")); read.OK {
		return core.AsString(read.Value.([]byte))
	}
	if config.ChatTemplateJinja != "" {
		return config.ChatTemplateJinja
	}
	if read := core.ReadFile(core.PathJoin(snapshotDir, "tokenizer_config.json")); read.OK {
		var tokenizerConfig struct {
			ChatTemplate string `json:"chat_template"`
		}
		if result := core.JSONUnmarshal(read.Value.([]byte), &tokenizerConfig); result.OK {
			return tokenizerConfig.ChatTemplate
		}
	}
	return ""
}

func officialGemma4AttentionPattern(layerTypes []string) (sliding, full, interval int) {
	lastFull := -1
	for i, layerType := range layerTypes {
		switch layerType {
		case "sliding_attention":
			sliding++
		case "full_attention":
			full++
			if lastFull >= 0 && interval == 0 {
				interval = i - lastFull
			}
			lastFull = i
		}
	}
	return sliding, full, interval
}

func officialGemma4CompareContracts(report *OfficialGemma4E2BControlComparison) {
	if report == nil {
		return
	}
	target, control := report.Target, report.Control
	report.QuantizationDiffers = target.QuantBits != control.QuantBits
	report.ArchitectureCompatible = target.ModelType == control.ModelType &&
		target.PackArchitecture == control.PackArchitecture
	report.ContextCompatible = target.ContextLength > 0 && target.ContextLength == control.ContextLength
	report.AttentionCompatible = target.SlidingWindow == control.SlidingWindow &&
		target.SlidingAttentionLayers == control.SlidingAttentionLayers &&
		target.FullAttentionLayers == control.FullAttentionLayers &&
		target.FullAttentionInterval == control.FullAttentionInterval
	report.RoPECompatible = target.FullRoPETheta == control.FullRoPETheta &&
		target.FullRoPEType == control.FullRoPEType &&
		target.FullPartialRotaryFactorPct == control.FullPartialRotaryFactorPct &&
		target.SlidingRoPETheta == control.SlidingRoPETheta &&
		target.SlidingRoPEType == control.SlidingRoPEType &&
		target.ProportionalRoPE == control.ProportionalRoPE
	report.SharedKVCompatible = target.NumKVSharedLayers == control.NumKVSharedLayers
	report.PerLayerInputCompatible = target.PerLayerInputs == control.PerLayerInputs &&
		target.HiddenSizePerLayerInput == control.HiddenSizePerLayerInput &&
		target.VocabSizePerLayerInput == control.VocabSizePerLayerInput
	report.ChatTemplateCompatible = target.HasThinkingToken == control.HasThinkingToken &&
		target.HasThoughtChannelMarkers == control.HasThoughtChannelMarkers &&
		target.StripsThinking == control.StripsThinking
	report.RetainedStateCompatible = target.NativeLoadable && control.NativeLoadable &&
		report.ArchitectureCompatible &&
		report.ContextCompatible &&
		report.AttentionCompatible &&
		report.RoPECompatible &&
		report.SharedKVCompatible &&
		report.PerLayerInputCompatible
	report.PromptCacheCompatible = report.RetainedStateCompatible &&
		target.HasTokenizer && control.HasTokenizer &&
		target.HasChatTemplate && control.HasChatTemplate &&
		report.ChatTemplateCompatible

	if !report.ArchitectureCompatible {
		report.Issues = append(report.Issues, "architecture/model type differs between official target and q4 control")
	}
	if !report.ContextCompatible {
		report.Issues = append(report.Issues, "context window differs between official target and q4 control")
	}
	if !report.AttentionCompatible {
		report.Issues = append(report.Issues, "local/full attention schedule or sliding window differs between official target and q4 control")
	}
	if !report.RoPECompatible {
		report.Issues = append(report.Issues, "p-RoPE or local RoPE parameters differ between official target and q4 control")
	}
	if !report.SharedKVCompatible {
		report.Issues = append(report.Issues, "shared-KV metadata differs between official target and q4 control")
	}
	if !report.PerLayerInputCompatible {
		report.Issues = append(report.Issues, "per-layer input/PLE metadata differs between official target and q4 control")
	}
	if !report.ChatTemplateCompatible {
		report.Issues = append(report.Issues, "thinking/no-thinking chat-template markers differ between official target and q4 control")
	}
	if !report.RetainedStateCompatible {
		report.Issues = append(report.Issues, "retained-State K/V metadata contract differs between official target and q4 control")
	}
	if !report.PromptCacheCompatible {
		report.Issues = append(report.Issues, "prompt-cache tokenizer/chat-template contract differs between official target and q4 control")
	}
	report.Compatible = report.ArchitectureCompatible &&
		report.ContextCompatible &&
		report.AttentionCompatible &&
		report.RoPECompatible &&
		report.SharedKVCompatible &&
		report.PerLayerInputCompatible &&
		report.ChatTemplateCompatible &&
		report.RetainedStateCompatible &&
		report.PromptCacheCompatible
}

func firstPositiveLocal(values ...int) int {
	for _, value := range values {
		if value > 0 {
			return value
		}
	}
	return 0
}

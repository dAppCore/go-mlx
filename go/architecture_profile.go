// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	core "dappco.re/go"
	"dappco.re/go/inference/parser"
)

// ArchitectureRuntimeStatus describes how far a model family is implemented.
type ArchitectureRuntimeStatus string

const (
	ArchitectureRuntimeNative       ArchitectureRuntimeStatus = "native"
	ArchitectureRuntimeMetadataOnly ArchitectureRuntimeStatus = "metadata_only"
)

// ModelArchitectureProfile is metadata-only feature information for a model
// family. It is intentionally loader-neutral so ROCm/CUDA/TPU backends can
// adopt the same targets without importing MLX internals.
type ModelArchitectureProfile struct {
	ID                   string                    `json:"id"`
	Family               string                    `json:"family,omitempty"`
	RuntimeStatus        ArchitectureRuntimeStatus `json:"runtime_status"`
	NativeRuntime        bool                      `json:"native_runtime"`
	Generation           bool                      `json:"generation"`
	Chat                 bool                      `json:"chat"`
	Embeddings           bool                      `json:"embeddings"`
	Rerank               bool                      `json:"rerank"`
	MoE                  bool                      `json:"moe"`
	RequiresChatTemplate bool                      `json:"requires_chat_template"`
	ParserID             string                    `json:"parser_id,omitempty"`
	ToolParserID         string                    `json:"tool_parser_id,omitempty"`
	ChatTemplate         string                    `json:"chat_template,omitempty"`
	LoRATargets          []string                  `json:"lora_targets,omitempty"`
	QuantizationHints    []string                  `json:"quantization_hints,omitempty"`
	CacheHints           []string                  `json:"cache_hints,omitempty"`
	Notes                []string                  `json:"notes,omitempty"`
	Aliases              []string                  `json:"aliases,omitempty"`
}

// BuiltinArchitectureProfiles returns the metadata-only feature target list.
func BuiltinArchitectureProfiles() []ModelArchitectureProfile {
	profiles := builtinArchitectureProfiles()
	out := make([]ModelArchitectureProfile, len(profiles))
	for i, profile := range profiles {
		out[i] = cloneArchitectureProfile(profile)
	}
	return out
}

// LookupArchitectureProfile resolves config model_type or Transformers
// architecture names to a built-in profile.
func LookupArchitectureProfile(value string) (ModelArchitectureProfile, bool) {
	id := architectureProfileID(value)
	if id == "" {
		return ModelArchitectureProfile{}, false
	}
	for _, profile := range builtinArchitectureProfiles() {
		if profile.ID == id {
			return cloneArchitectureProfile(profile), true
		}
	}
	for _, profile := range builtinArchitectureProfiles() {
		for _, alias := range profile.Aliases {
			if architectureProfileID(alias) == id || parser.NormaliseKey(alias) == id {
				return cloneArchitectureProfile(profile), true
			}
		}
	}
	return ModelArchitectureProfile{}, false
}

func architectureProfileID(value string) string {
	value = core.Trim(value)
	if value == "" {
		return ""
	}
	if mapped := architectureFromTransformersName(value); mapped != "" {
		return mapped
	}
	normalized := normalizeKnownArchitecture(value)
	if normalized == "bert_rerank" {
		return normalized
	}
	compact := core.Replace(core.Replace(normalized, "_", ""), "-", "")
	switch {
	case core.Contains(compact, "qwen3moe"):
		return "qwen3_moe"
	case core.Contains(compact, "qwen3next"):
		return "qwen3_next"
	case core.Contains(compact, "minimaxm2"):
		return "minimax_m2"
	case core.Contains(compact, "mixtral"):
		return "mixtral"
	case core.Contains(compact, "mistral"):
		return "mistral"
	case core.Contains(compact, "deepseek"):
		return "deepseek"
	case core.Contains(compact, "gptoss"):
		return "gpt_oss"
	case core.Contains(compact, "phi"):
		return "phi"
	case core.Contains(compact, "bertforsequenceclassification") || core.Contains(compact, "robertaforsequenceclassification") || core.Contains(compact, "xlmrobertaforsequenceclassification") || core.Contains(compact, "debertav2forsequenceclassification"):
		return "bert_rerank"
	case core.Contains(compact, "bert"):
		return "bert"
	default:
		return normalized
	}
}

func builtinArchitectureProfiles() []ModelArchitectureProfile {
	return []ModelArchitectureProfile{
		nativeProfile("gemma2", "gemma", "gemma", []string{"Gemma2ForCausalLM"}),
		nativeProfile("gemma3", "gemma", "gemma", []string{"Gemma3ForCausalLM"}),
		nativeProfile("gemma3_text", "gemma", "gemma", []string{"Gemma3TextForCausalLM"}),
		nativeProfile("gemma4", "gemma", "gemma", []string{"Gemma4ForConditionalGeneration"}),
		nativeProfile("gemma4_text", "gemma", "gemma", []string{"Gemma4ForCausalLM", "Gemma4TextForCausalLM"}),
		nativeProfile("llama", "llama", "llama", []string{"LlamaForCausalLM"}),
		nativeProfile("qwen2", "qwen", "qwen", []string{"Qwen2ForCausalLM"}),
		nativeProfile("qwen3", "qwen", "qwen", []string{"Qwen3ForCausalLM"}),
		nativeProfile("qwen3_next", "qwen", "qwen", []string{"Qwen3NextForCausalLM", "Qwen3.5ForCausalLM"}),
		metadataProfile("qwen3_moe", "qwen", "qwen", "qwen", true, false, []string{"Qwen3MoeForCausalLM"}, []string{"sparse expert router kernels pending"}),
		metadataProfile("minimax_m2", "minimax", "minimax", "minimax", true, false, []string{"MiniMaxM2ForCausalLM"}, []string{"JANGTQ/MXTQ packed expert kernels pending"}),
		metadataProfile("mistral", "mistral", "mistral", "mistral", false, false, []string{"MistralForCausalLM"}, nil),
		metadataProfile("mixtral", "mistral", "mistral", "mistral", true, false, []string{"MixtralForCausalLM"}, []string{"sparse expert router kernels pending"}),
		metadataProfile("phi", "phi", "generic", "generic", false, false, []string{"PhiForCausalLM", "Phi3ForCausalLM", "Phi4ForCausalLM"}, nil),
		metadataProfile("deepseek", "deepseek", "deepseek-r1", "generic", true, false, []string{"DeepseekV3ForCausalLM", "DeepSeekV3ForCausalLM", "DeepseekR1ForCausalLM"}, []string{"MoE router and DeepSeek MLA variants pending"}),
		metadataProfile("gpt_oss", "gpt-oss", "gpt-oss", "generic", true, false, []string{"GptOssForCausalLM", "GPTOSSForCausalLM"}, []string{"MoE router and channel parser validation pending"}),
		metadataProfile("kimi", "kimi", "kimi", "generic", true, false, []string{"KimiForCausalLM", "MoonshotForCausalLM"}, []string{"MoE router kernels pending"}),
		metadataProfile("glm", "glm", "glm", "generic", false, false, []string{"GlmForCausalLM", "ChatGLMForConditionalGeneration"}, nil),
		metadataProfile("hermes", "hermes", "hermes", "generic", false, false, []string{"HermesForCausalLM"}, nil),
		metadataProfile("granite", "granite", "granite", "generic", false, false, []string{"GraniteForCausalLM"}, nil),
		metadataProfile("bert", "bert", "generic", "generic", false, true, []string{"BertModel", "BertForMaskedLM"}, []string{"embedding encoder loader pending"}),
		rerankProfile("bert_rerank", "bert", []string{"BertForSequenceClassification", "RobertaForSequenceClassification", "XLMRobertaForSequenceClassification", "DebertaV2ForSequenceClassification"}, []string{"cross-encoder scorer loader pending"}),
	}
}

func nativeProfile(id, family, parser string, aliases []string) ModelArchitectureProfile {
	profile := metadataProfile(id, family, parser, parser, false, false, aliases, nil)
	profile.RuntimeStatus = ArchitectureRuntimeNative
	profile.NativeRuntime = true
	return profile
}

func metadataProfile(id, family, parser, toolParser string, moe, embeddings bool, aliases, notes []string) ModelArchitectureProfile {
	chat := !embeddings
	return ModelArchitectureProfile{
		ID:                   id,
		Family:               family,
		RuntimeStatus:        ArchitectureRuntimeMetadataOnly,
		Generation:           chat,
		Chat:                 chat,
		Embeddings:           embeddings,
		MoE:                  moe,
		RequiresChatTemplate: chat,
		ParserID:             parser,
		ToolParserID:         toolParser,
		ChatTemplate:         architectureDefaultChatTemplate(family, id, embeddings),
		LoRATargets:          architectureDefaultLoRATargets(family, moe),
		QuantizationHints:    architectureDefaultQuantizationHints(id, moe),
		CacheHints:           architectureDefaultCacheHints(id, moe),
		Notes:                append([]string(nil), notes...),
		Aliases:              append([]string(nil), aliases...),
	}
}

func rerankProfile(id, family string, aliases, notes []string) ModelArchitectureProfile {
	profile := metadataProfile(id, family, "generic", "generic", false, false, aliases, notes)
	profile.Generation = false
	profile.Chat = false
	profile.Rerank = true
	profile.RequiresChatTemplate = false
	profile.ChatTemplate = ""
	profile.LoRATargets = []string{"classifier", "score", "dense"}
	profile.QuantizationHints = []string{"fp16", "bf16", "q8_0"}
	profile.CacheHints = nil
	return profile
}

func architectureDefaultChatTemplate(family, id string, embeddings bool) string {
	if embeddings {
		return ""
	}
	switch id {
	case "gemma4", "gemma4_text":
		return "gemma4"
	}
	switch family {
	case "gemma", "qwen", "llama", "mistral", "minimax":
		return family
	case "deepseek", "kimi", "glm", "hermes", "granite":
		return family
	case "gpt-oss":
		return "gpt-oss"
	default:
		if id != "" {
			return id
		}
		return "generic"
	}
}

func architectureDefaultLoRATargets(family string, moe bool) []string {
	targets := []string{"q_proj", "k_proj", "v_proj", "o_proj"}
	switch family {
	case "gemma":
		targets = append(targets, "gate_proj", "up_proj", "down_proj", "per_layer_projection")
	case "qwen", "mistral", "llama", "minimax", "deepseek", "kimi", "glm", "hermes", "granite", "phi":
		targets = append(targets, "gate_proj", "up_proj", "down_proj")
	}
	if moe {
		targets = append(targets, "router", "router.proj", "experts")
	}
	return targets
}

func architectureDefaultQuantizationHints(id string, moe bool) []string {
	hints := []string{"fp16", "bf16", "q8_0", "q4_k_m"}
	if moe {
		hints = append(hints, "expert-aware")
	}
	if id == "minimax_m2" {
		hints = append(hints, "jang", "jangtq", "mxtq")
	}
	return hints
}

func architectureDefaultCacheHints(id string, moe bool) []string {
	hints := []string{string(KVCacheModeQ8), string(KVCacheModePaged)}
	if moe || id == "minimax_m2" {
		hints = append(hints, string(KVCacheModeKQ8VQ4))
	}
	return hints
}

func cloneArchitectureProfile(profile ModelArchitectureProfile) ModelArchitectureProfile {
	profile.LoRATargets = append([]string(nil), profile.LoRATargets...)
	profile.QuantizationHints = append([]string(nil), profile.QuantizationHints...)
	profile.CacheHints = append([]string(nil), profile.CacheHints...)
	profile.Notes = append([]string(nil), profile.Notes...)
	profile.Aliases = append([]string(nil), profile.Aliases...)
	return profile
}

func architectureProfileIDs() []string {
	profiles := builtinArchitectureProfiles()
	out := make([]string, 0, len(profiles))
	for _, profile := range profiles {
		out = append(out, profile.ID)
	}
	return out
}

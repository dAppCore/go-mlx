// SPDX-Licence-Identifier: EUPL-1.2

package profile

import (
	"unsafe"

	core "dappco.re/go"
	"dappco.re/go/inference/parser"
)

// maxArchitectureNameBytes bounds the stack buffer used by
// compactArchitectureNameInto. The longest known architecture alias is
// XLMRobertaForSequenceClassification (35 chars) — 64 leaves ample
// headroom for any plausible new entry and keeps the buffer cheap.
const maxArchitectureNameBytes = 64

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
// architecture names to a built-in profile. Returns a defensive
// deep-clone so external callers may mutate the result without
// touching the shared registry. In-package read-only consumers should
// prefer LookupArchitectureProfileRef, which returns a pointer into
// the static table and avoids the per-call 5-slice clone.
func LookupArchitectureProfile(value string) (ModelArchitectureProfile, bool) {
	ref, ok := LookupArchitectureProfileRef(value)
	if !ok {
		return ModelArchitectureProfile{}, false
	}
	return cloneArchitectureProfile(*ref), true
}

// LookupArchitectureProfileRef resolves an architecture name to a
// pointer into the immutable built-in registry. The returned pointer
// (and its slice fields LoRATargets/QuantizationHints/CacheHints/
// Notes/Aliases) MUST NOT be mutated — the data is shared across all
// callers for the lifetime of the process. Use this on the hot path
// (planFit, archSupported, archNativeRuntime,
// tuningRuntimeForArchitecture, memory.NewPlan) where a defensive
// clone is pure overhead. Callers that need to mutate the result
// must use LookupArchitectureProfile.
func LookupArchitectureProfileRef(value string) (*ModelArchitectureProfile, bool) {
	if value == "" {
		return nil, false
	}
	// Fast path — most hot-path callers (memory.NewPlan with a
	// caller-managed Pack.Architecture, planFit walking pre-resolved
	// architecture IDs, model/pack inspectors using normalised IDs)
	// pass strings that are already canonical and registered in the
	// index. Probe the index directly first; on a hit we skip the full
	// ArchitectureID pipeline (Trim + transformersName scan + normalize
	// + compact), which spends 1-2 allocs canonicalising strings that
	// are already canonical. On a miss, fall through to the full
	// resolver so caps/dashes/dots/Transformers-name variants still
	// resolve correctly.
	if idx, ok := builtinArchitectureProfileIndex[value]; ok {
		return &builtinArchitectureProfilesData[idx], true
	}
	id := ArchitectureID(value)
	if id == "" {
		return nil, false
	}
	if idx, ok := builtinArchitectureProfileIndex[id]; ok {
		return &builtinArchitectureProfilesData[idx], true
	}
	return nil, false
}

func ArchitectureID(value string) string {
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
	var buf [maxArchitectureNameBytes]byte
	compact := compactArchitectureNameInto(buf[:], normalized)
	switch {
	case core.Contains(compact, "qwen35moe") || core.Contains(compact, "qwen36moe"):
		return "qwen3_6_moe"
	case core.Contains(compact, "qwen35") || core.Contains(compact, "qwen36"):
		return "qwen3_6"
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

// builtinArchitectureProfilesData is the singleton backing list — built
// once at package init, exposed through builtinArchitectureProfiles.
// Callers must not mutate this slice or its entries; the public API
// clones before returning.
var builtinArchitectureProfilesData = []ModelArchitectureProfile{}

// builtinArchitectureProfileIndex maps every architecture ID that can
// resolve to a built-in profile — the profile's own ID plus the
// ArchitectureID and parser.NormaliseKey expansions of each alias — to
// its slot in builtinArchitectureProfilesData. LookupArchitectureProfile
// uses this to collapse the previous two linear-scan passes (exact ID,
// then alias normalisation) into a single map probe.
var builtinArchitectureProfileIndex = map[string]int{}

func init() {
	builtinArchitectureProfilesData = buildBuiltinArchitectureProfiles()
	builtinArchitectureProfileIndex = make(map[string]int, len(builtinArchitectureProfilesData)*4)
	for i, profile := range builtinArchitectureProfilesData {
		if profile.ID != "" {
			builtinArchitectureProfileIndex[profile.ID] = i
		}
		for _, alias := range profile.Aliases {
			if key := ArchitectureID(alias); key != "" {
				if _, exists := builtinArchitectureProfileIndex[key]; !exists {
					builtinArchitectureProfileIndex[key] = i
				}
			}
			if key := parser.NormaliseKey(alias); key != "" {
				if _, exists := builtinArchitectureProfileIndex[key]; !exists {
					builtinArchitectureProfileIndex[key] = i
				}
			}
		}
	}
}

func builtinArchitectureProfiles() []ModelArchitectureProfile {
	return builtinArchitectureProfilesData
}

func buildBuiltinArchitectureProfiles() []ModelArchitectureProfile {
	return []ModelArchitectureProfile{
		nativeProfile("gemma2", "gemma", "gemma", []string{"Gemma2ForCausalLM"}),
		nativeProfile("gemma3", "gemma", "gemma", []string{"Gemma3ForCausalLM"}),
		nativeProfile("gemma3_text", "gemma", "gemma", []string{"Gemma3TextForCausalLM"}),
		nativeProfile("gemma4", "gemma", "gemma", []string{"Gemma4ForConditionalGeneration"}),
		nativeProfile("gemma4_text", "gemma", "gemma", []string{"Gemma4ForCausalLM", "Gemma4TextForCausalLM"}),
		metadataProfile("gemma4_assistant", "gemma", "gemma", "gemma", false, false, []string{"Gemma4AssistantForCausalLM"}, []string{"attached MTP drafter; standalone generation unsupported; load beside a Gemma 4 target"}),
		nativeProfile("llama", "llama", "llama", []string{"LlamaForCausalLM"}),
		nativeProfile("qwen2", "qwen", "qwen", []string{"Qwen2ForCausalLM", "Qwen2.5ForCausalLM", "Qwen2_5ForCausalLM"}),
		nativeProfile("qwen3", "qwen", "qwen", []string{"Qwen3ForCausalLM"}),
		nativeProfile("qwen3_next", "qwen", "qwen", []string{"Qwen3NextForCausalLM"}),
		metadataProfile("qwen3_6", "qwen", "qwen", "qwen", false, false, []string{"Qwen3_5ForConditionalGeneration", "Qwen3.5ForConditionalGeneration", "Qwen3_6ForConditionalGeneration", "Qwen3.6ForConditionalGeneration", "Qwen3_5ForCausalLM", "Qwen3.5ForCausalLM"}, []string{"hybrid linear-attention native kernels pending"}),
		metadataProfile("qwen3_6_moe", "qwen", "qwen", "qwen", true, false, []string{"Qwen3_5MoeForConditionalGeneration", "Qwen3.5MoeForConditionalGeneration", "Qwen3_6MoeForConditionalGeneration", "Qwen3.6MoeForConditionalGeneration"}, []string{"hybrid linear-attention and sparse expert native kernels pending"}),
		metadataProfile("qwen3_moe", "qwen", "qwen", "qwen", true, false, []string{"Qwen3MoeForCausalLM"}, []string{"sparse expert router kernels pending"}),
		metadataProfile("minimax_m2", "minimax", "minimax", "minimax", true, false, []string{"MiniMaxM2ForCausalLM"}, []string{"JANGTQ/MXTQ packed expert kernels pending"}),
		nativeProfile("mistral", "mistral", "mistral", []string{"MistralForCausalLM"}),
		metadataProfile("mixtral", "mistral", "mistral", "mistral", true, false, []string{"MixtralForCausalLM"}, []string{"sparse expert router kernels pending"}),
		nativeProfile("phi", "phi", "generic", []string{"PhiForCausalLM", "Phi3ForCausalLM", "Phi4ForCausalLM"}),
		metadataProfile("deepseek", "deepseek", "deepseek-r1", "generic", true, false, []string{"DeepseekV3ForCausalLM", "DeepSeekV3ForCausalLM", "DeepseekR1ForCausalLM"}, []string{"MoE router and DeepSeek MLA variants pending"}),
		metadataProfile("gpt_oss", "gpt-oss", "gpt-oss", "generic", true, false, []string{"GptOssForCausalLM", "GPTOSSForCausalLM"}, []string{"MoE router and channel parser validation pending"}),
		metadataProfile("kimi", "kimi", "kimi", "generic", true, false, []string{"KimiForCausalLM", "MoonshotForCausalLM"}, []string{"MoE router kernels pending"}),
		nativeProfile("glm", "glm", "glm", []string{"GlmForCausalLM", "ChatGLMForConditionalGeneration"}),
		nativeProfile("hermes", "hermes", "hermes", []string{"HermesForCausalLM"}),
		nativeProfile("granite", "granite", "granite", []string{"GraniteForCausalLM"}),
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
	hints := []string{"q8", "paged"}
	if moe || id == "minimax_m2" {
		hints = append(hints, "k-q8-v-q4")
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

func ArchitectureIDs() []string {
	profiles := builtinArchitectureProfiles()
	out := make([]string, 0, len(profiles))
	for _, profile := range profiles {
		out = append(out, profile.ID)
	}
	return out
}

func normalizeKnownArchitecture(value string) string {
	value = core.Lower(core.Trim(value))
	value = core.Replace(value, "-", "_")
	value = core.Replace(value, ".", "_")
	switch value {
	case "qwen2_5", "qwen25":
		return "qwen2"
	case "qwen3_5", "qwen3_5_text", "qwen3_6", "qwen3_6_text", "qwen35", "qwen36":
		return "qwen3_6"
	case "qwen3_5_moe", "qwen3_6_moe", "qwen35_moe", "qwen36_moe":
		return "qwen3_6_moe"
	case "minimaxm2", "minimax_m2":
		return "minimax_m2"
	case "mixtral":
		return "mixtral"
	case "mistral":
		return "mistral"
	case "phi", "phi3", "phi4":
		return "phi"
	case "deepseek", "deepseek_v3", "deepseek_r1":
		return "deepseek"
	case "gptoss", "gpt_oss", "gpt_oss_model":
		return "gpt_oss"
	case "bert":
		return "bert"
	case "bert_rerank", "bert_cross_encoder":
		return "bert_rerank"
	default:
		return value
	}
}

func architectureFromTransformersName(architecture string) string {
	var buf [maxArchitectureNameBytes]byte
	compact := compactArchitectureNameInto(buf[:], architecture)
	switch {
	case core.Contains(compact, "bertforsequenceclassification") || core.Contains(compact, "robertaforsequenceclassification") || core.Contains(compact, "xlmrobertaforsequenceclassification") || core.Contains(compact, "debertav2forsequenceclassification"):
		return "bert_rerank"
	case core.Contains(compact, "qwen35moe") || core.Contains(compact, "qwen36moe"):
		return "qwen3_6_moe"
	case core.Contains(compact, "qwen35") || core.Contains(compact, "qwen36"):
		return "qwen3_6"
	case core.Contains(compact, "qwen3moe"):
		return "qwen3_moe"
	case core.Contains(compact, "qwen3next"):
		return "qwen3_next"
	case core.Contains(compact, "gemma4assistant"):
		return "gemma4_assistant"
	case core.Contains(architecture, "Gemma4"):
		return "gemma4_text"
	case core.Contains(architecture, "Gemma3"):
		return "gemma3"
	case core.Contains(architecture, "Gemma2"):
		return "gemma2"
	case core.Contains(architecture, "Qwen3"):
		return "qwen3"
	case core.Contains(architecture, "Qwen2"):
		return "qwen2"
	case core.Contains(architecture, "Llama"):
		return "llama"
	case core.Contains(architecture, "MiniMaxM2"):
		return "minimax_m2"
	case core.Contains(architecture, "Mixtral"):
		return "mixtral"
	case core.Contains(architecture, "Mistral"):
		return "mistral"
	case core.Contains(architecture, "Phi"):
		return "phi"
	case core.Contains(architecture, "Deepseek") || core.Contains(architecture, "DeepSeek"):
		return "deepseek"
	case core.Contains(architecture, "GptOss") || core.Contains(architecture, "GPTOSS"):
		return "gpt_oss"
	case core.Contains(architecture, "Bert"):
		return "bert"
	default:
		return ""
	}
}

// compactArchitectureNameInto writes the compact form of value into
// buf (ASCII lowercased, with '_' '-' '.' stripped) and returns a
// string view backed by buf. buf MUST outlive the returned string —
// the result is unsafe-aliased to the underlying bytes to keep the
// hot architecture-resolution path zero-alloc.
//
// Inputs longer than len(buf) or containing non-ASCII fall back to
// the old core.Lower+core.Replace path (one alloc, heap-stable
// string). All real architecture names are ASCII and ≤ 35 chars,
// so the fallback never fires for built-in resolution.
//
//	var buf [maxArchitectureNameBytes]byte
//	compact := compactArchitectureNameInto(buf[:], "Qwen3ForCausalLM")
//	// compact == "qwen3forcausallm" — aliased to buf[:16]
func compactArchitectureNameInto(buf []byte, value string) string {
	n := 0
	for i := 0; i < len(value); i++ {
		c := value[i]
		if c >= 0x80 {
			return compactArchitectureNameFallback(value)
		}
		if c == '_' || c == '-' || c == '.' {
			continue
		}
		if n == len(buf) {
			return compactArchitectureNameFallback(value)
		}
		if c >= 'A' && c <= 'Z' {
			c += 'a' - 'A'
		}
		buf[n] = c
		n++
	}
	if n == 0 {
		return ""
	}
	return unsafe.String(&buf[0], n)
}

// compactArchitectureNameFallback handles the rare non-ASCII /
// over-length input. Heap-stable single-alloc result, identical to
// the pre-W11E semantics.
func compactArchitectureNameFallback(value string) string {
	compact := core.Lower(value)
	compact = core.Replace(compact, "_", "")
	compact = core.Replace(compact, "-", "")
	return core.Replace(compact, ".", "")
}

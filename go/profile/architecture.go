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
	ID                    string                    `json:"id"`
	Family                string                    `json:"family,omitempty"`
	RuntimeStatus         ArchitectureRuntimeStatus `json:"runtime_status"`
	NativeRuntime         bool                      `json:"native_runtime"`
	Generation            bool                      `json:"generation"`
	Chat                  bool                      `json:"chat"`
	Embeddings            bool                      `json:"embeddings"`
	Rerank                bool                      `json:"rerank"`
	MoE                   bool                      `json:"moe"`
	RequiresChatTemplate  bool                      `json:"requires_chat_template"`
	ParserID              string                    `json:"parser_id,omitempty"`
	ToolParserID          string                    `json:"tool_parser_id,omitempty"`
	ChatTemplate          string                    `json:"chat_template,omitempty"`
	LoRATargets           []string                  `json:"lora_targets,omitempty"`
	LoRADefaultTargets    []string                  `json:"lora_default_targets,omitempty"`
	LoRATargetPaths       map[string]string         `json:"lora_target_paths,omitempty"`
	LoRAExtendedTargets   []string                  `json:"lora_extended_targets,omitempty"`
	WeightWrapperPrefixes []string                  `json:"weight_wrapper_prefixes,omitempty"`
	WeightSkipPrefixes    []string                  `json:"weight_skip_prefixes,omitempty"`
	WeightSkipSubstrings  []string                  `json:"weight_skip_substrings,omitempty"`
	WeightModelPrefixes   []string                  `json:"weight_model_prefixes,omitempty"`
	QuantizationHints     []string                  `json:"quantization_hints,omitempty"`
	CacheHints            []string                  `json:"cache_hints,omitempty"`
	Notes                 []string                  `json:"notes,omitempty"`
	Aliases               []string                  `json:"aliases,omitempty"`
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
	if mapped := ArchitectureFromTransformersName(value); mapped != "" {
		return mapped
	}
	normalized := NormalizeArchitecture(value)
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

// IsGemma4TargetArchitecture reports whether architecture identifies a Gemma 4
// target model that can own prompts, LoRA adapters, SFT/SSD runs, and fused
// model packs. The attached Gemma 4 assistant drafter is intentionally excluded.
func IsGemma4TargetArchitecture(architecture string) bool {
	switch ArchitectureID(architecture) {
	case "gemma4", "gemma4_text", "gemma4_unified":
		return true
	default:
		return false
	}
}

// IsGemma4LargeVariant reports whether Gemma 4 prompt rendering should use the
// large-variant suppressor path. The shipped 26B/31B templates expose at least
// 16 attention heads and ghost an empty thought channel when thinking is off;
// smaller target models and the attached assistant drafter do not.
func IsGemma4LargeVariant(architecture string, numAttentionHeads int) bool {
	return numAttentionHeads >= 16 && IsGemma4TargetArchitecture(architecture)
}

// ChatTemplateName returns the default chat-template id advertised for an
// architecture. It is metadata-only: callers that render templates should still
// filter this through the templates they actually implement.
func ChatTemplateName(architecture string) string {
	architecture = core.Trim(architecture)
	if architecture == "" {
		return ""
	}
	if profile, ok := LookupArchitectureProfileRef(architecture); ok {
		if profile.ChatTemplate != "" {
			return profile.ChatTemplate
		}
		if profile.Family == "qwen" {
			return "qwen"
		}
		return ""
	}
	switch NormalizeArchitecture(architecture) {
	case "gemma":
		return "gemma"
	case "qwen":
		return "qwen"
	case "llama", "llama3", "llama4":
		return "llama"
	default:
		return ""
	}
}

// DefaultLoRATargets returns the registered narrow default LoRA target set for
// an architecture — the targets applied when a caller requests a LoRA without
// explicit keys. Nil when the architecture is unknown or declares none.
func DefaultLoRATargets(architecture string) []string {
	if ref, ok := LookupArchitectureProfileRef(architecture); ok {
		return append([]string(nil), ref.LoRADefaultTargets...)
	}
	return nil
}

// LoRATargetPath canonicalises a LoRA target key into the projection path used
// by adapter metadata and linear resolution, via the registered per-family map.
// Returns false when the architecture is unknown or the key is not a recognised
// target — so a non-LoRA architecture simply yields no canonicalisation.
func LoRATargetPath(architecture, key string) (string, bool) {
	ref, ok := LookupArchitectureProfileRef(architecture)
	if !ok {
		return "", false
	}
	path, ok := ref.LoRATargetPaths[key]
	return path, ok
}

// SafeLoRATarget reports whether a LoRA target can be enabled by default for an
// architecture — it resolves to a known projection path that is not in the
// family's extended (opt-in) set.
func SafeLoRATarget(architecture, key string) bool {
	ref, ok := LookupArchitectureProfileRef(architecture)
	if !ok {
		return false
	}
	path, ok := ref.LoRATargetPaths[key]
	if !ok {
		return false
	}
	for _, extended := range ref.LoRAExtendedTargets {
		if path == extended {
			return false
		}
	}
	return true
}

// CanonicalWeightName canonicalises a checkpoint weight name for an
// architecture: it strips the model-declared wrapper prefixes, drops non-text
// helper tensors (returning ok=false), and re-roots text tensors under
// "model.". An architecture with no weight rules passes the name through
// unchanged, so the engine names no family.
func CanonicalWeightName(architecture, name string) (string, bool) {
	ref, ok := LookupArchitectureProfileRef(architecture)
	if !ok {
		return name, true
	}
	trimmed := unwrapWeightName(name, ref.WeightWrapperPrefixes)
	for _, prefix := range ref.WeightSkipPrefixes {
		if core.HasPrefix(trimmed, prefix) {
			return "", false
		}
	}
	for _, substr := range ref.WeightSkipSubstrings {
		if core.Contains(trimmed, substr) {
			return "", false
		}
	}
	for _, prefix := range ref.WeightModelPrefixes {
		if core.HasPrefix(trimmed, prefix) {
			return "model." + trimmed, true
		}
	}
	return trimmed, true
}

// TrimWeightWrapperPrefix removes one of an architecture's declared checkpoint
// wrapper prefixes from name, reporting whether one matched.
func TrimWeightWrapperPrefix(architecture, name string) (string, bool) {
	ref, ok := LookupArchitectureProfileRef(architecture)
	if !ok {
		return name, false
	}
	return trimOneWeightWrapper(name, ref.WeightWrapperPrefixes)
}

func unwrapWeightName(name string, wrapperPrefixes []string) string {
	trimmed := name
	for {
		next, changed := trimOneWeightWrapper(trimmed, wrapperPrefixes)
		if !changed {
			return trimmed
		}
		trimmed = next
	}
}

func trimOneWeightWrapper(name string, wrapperPrefixes []string) (string, bool) {
	for _, prefix := range wrapperPrefixes {
		if core.HasPrefix(name, prefix) {
			return core.TrimPrefix(name, prefix), true
		}
	}
	return name, false
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
		indexArchitectureProfile(i, profile)
	}
}

// indexArchitectureProfile maps a profile's ID and alias expansions to its slot
// in the registry. An alias already claimed by an earlier profile is never
// overwritten, so built-in entries win ties over later registrations.
func indexArchitectureProfile(slot int, profile ModelArchitectureProfile) {
	if profile.ID != "" {
		builtinArchitectureProfileIndex[profile.ID] = slot
	}
	for _, alias := range profile.Aliases {
		if key := ArchitectureID(alias); key != "" {
			if _, exists := builtinArchitectureProfileIndex[key]; !exists {
				builtinArchitectureProfileIndex[key] = slot
			}
		}
		if key := parser.NormaliseKey(alias); key != "" {
			if _, exists := builtinArchitectureProfileIndex[key]; !exists {
				builtinArchitectureProfileIndex[key] = slot
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
		gemma4Profile("gemma4", []string{"Gemma4ForConditionalGeneration"}),
		gemma4Profile("gemma4_unified", []string{"Gemma4UnifiedForConditionalGeneration"}),
		gemma4Profile("gemma4_text", []string{"Gemma4ForCausalLM", "Gemma4TextForCausalLM"}),
		nativeAttachedDrafterProfile("gemma4_assistant", "gemma", "gemma", []string{"Gemma4AssistantForCausalLM"}, []string{"attached MTP drafter; standalone generation unsupported; load beside a Gemma 4 target"}),
		nativeProfile("llama", "llama", "llama", []string{"LlamaForCausalLM"}),
		nativeProfile("qwen2", "qwen", "qwen", []string{"Qwen2ForCausalLM", "Qwen2.5ForCausalLM", "Qwen2_5ForCausalLM"}),
		nativeProfile("qwen3", "qwen", "qwen", []string{"Qwen3ForCausalLM"}),
		nativeProfile("qwen3_next", "qwen", "qwen", []string{"Qwen3NextForCausalLM"}),
		nativeStagedProfile("qwen3_6", "qwen", "qwen", false, []string{"Qwen3_5ForConditionalGeneration", "Qwen3.5ForConditionalGeneration", "Qwen3_6ForConditionalGeneration", "Qwen3.6ForConditionalGeneration", "Qwen3_5ForCausalLM", "Qwen3.5ForCausalLM"}, []string{"native staged hybrid linear-attention config/tokenizer loader; standalone generation pending"}),
		nativeStagedProfile("qwen3_6_moe", "qwen", "qwen", true, []string{"Qwen3_5MoeForConditionalGeneration", "Qwen3.5MoeForConditionalGeneration", "Qwen3_6MoeForConditionalGeneration", "Qwen3.6MoeForConditionalGeneration"}, []string{"native staged hybrid linear-attention and sparse-expert config/tokenizer loader; standalone generation pending"}),
		nativeStagedProfile("qwen3_moe", "qwen", "qwen", true, []string{"Qwen3MoeForCausalLM"}, []string{"native staged sparse-expert config/tokenizer loader; standalone generation pending"}),
		nativeStagedProfile("minimax_m2", "minimax", "minimax", true, []string{"MiniMaxM2ForCausalLM"}, []string{"native staged JANGTQ/MXTQ tensor-plan loader; standalone sparse generation pending"}),
		nativeProfile("mistral", "mistral", "mistral", []string{"MistralForCausalLM"}),
		nativeStagedProfile("mixtral", "mistral", "mistral", true, []string{"MixtralForCausalLM"}, []string{"native staged sparse-expert config/tokenizer loader; standalone generation pending"}),
		nativeProfile("phi", "phi", "generic", []string{"PhiForCausalLM", "Phi3ForCausalLM", "Phi4ForCausalLM"}),
		nativeStagedProfile("deepseek", "deepseek", "deepseek-r1", true, []string{"DeepseekV3ForCausalLM", "DeepSeekV3ForCausalLM", "DeepseekR1ForCausalLM"}, []string{"native staged MoE/MLA config/tokenizer loader; standalone generation pending"}),
		nativeStagedProfile("gpt_oss", "gpt-oss", "gpt-oss", true, []string{"GptOssForCausalLM", "GPTOSSForCausalLM"}, []string{"native staged MoE config/tokenizer loader; standalone generation pending"}),
		nativeStagedProfile("kimi", "kimi", "kimi", true, []string{"KimiForCausalLM", "MoonshotForCausalLM"}, []string{"native staged sparse-expert config/tokenizer loader; standalone generation pending"}),
		nativeProfile("glm", "glm", "glm", []string{"GlmForCausalLM", "ChatGLMForConditionalGeneration"}),
		nativeProfile("hermes", "hermes", "hermes", []string{"HermesForCausalLM"}),
		nativeProfile("granite", "granite", "granite", []string{"GraniteForCausalLM"}),
		nativeEncoderStagedProfile("bert", "bert", "generic", []string{"BertModel", "BertForMaskedLM"}, []string{"native staged encoder loader; embedding pooling kernels pending"}),
		nativeRerankStagedProfile("bert_rerank", "bert", []string{"BertForSequenceClassification", "RobertaForSequenceClassification", "XLMRobertaForSequenceClassification", "DebertaV2ForSequenceClassification"}, []string{"native staged cross-encoder loader; scorer kernels pending"}),
	}
}

// Gemma-4 LoRA target policy — loader-neutral data shared across drivers. It
// lives in the registry (not the Metal model package) so go-rocm/cuda adopt the
// same targets through the generic accessors without importing MLX internals.
var (
	gemma4LoRADefaultTargets  = []string{"q_proj", "v_proj", "o_proj"}
	gemma4LoRAStandardTargets = []string{"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
	gemma4LoRAExtendedTargets = []string{"router.proj", "per_layer_input_gate", "per_layer_projection"}
	gemma4LoRATargetPaths     = map[string]string{
		"q_proj":               "self_attn.q_proj",
		"self_attn.q_proj":     "self_attn.q_proj",
		"k_proj":               "self_attn.k_proj",
		"self_attn.k_proj":     "self_attn.k_proj",
		"v_proj":               "self_attn.v_proj",
		"self_attn.v_proj":     "self_attn.v_proj",
		"o_proj":               "self_attn.o_proj",
		"self_attn.o_proj":     "self_attn.o_proj",
		"gate_proj":            "mlp.gate_proj",
		"mlp.gate_proj":        "mlp.gate_proj",
		"up_proj":              "mlp.up_proj",
		"mlp.up_proj":          "mlp.up_proj",
		"down_proj":            "mlp.down_proj",
		"mlp.down_proj":        "mlp.down_proj",
		"router.proj":          "router.proj",
		"per_layer_input_gate": "per_layer_input_gate",
		"per_layer_projection": "per_layer_projection",
	}
)

// gemma4 weight-name canonicalisation rules — loader-neutral data the generic
// CanonicalWeightName algorithm applies. The model declares its checkpoint
// wrapper prefixes, the non-text tensors to skip, and the prefixes that take a
// "model." root; the engine carries none of it.
var (
	gemma4WeightWrapperPrefixes = []string{
		"model.language_model.model.",
		"model.language_model.",
		"language_model.model.",
		"language_model.",
		"model.model.",
		"model.",
	}
	gemma4WeightSkipPrefixes = []string{
		"vision_tower",
		"multi_modal_projector",
		"audio_tower",
		"embed_audio",
		"embed_vision",
	}
	gemma4WeightSkipSubstrings = []string{
		"self_attn.rotary_emb",
		"input_max",
		"input_min",
		"output_max",
		"output_min",
	}
	gemma4WeightModelPrefixes = []string{
		"layers.",
		"embed_tokens.",
		"embed_tokens_per_layer.",
		"norm.",
		"per_layer_model_projection.",
		"per_layer_projection_norm.",
	}
)

// gemma4Profile builds a Gemma-4 target architecture profile: the family's
// chat template, its LoRA target policy (full advertised set, narrow safe
// default, key->path canonicalisation, extended opt-in targets), and its
// checkpoint weight-name canonicalisation rules. The engine and model package
// read this back through the generic accessors.
func gemma4Profile(id string, aliases []string) ModelArchitectureProfile {
	p := nativeProfile(id, "gemma", "gemma", aliases)
	p.ChatTemplate = "gemma4"
	p.LoRATargets = append(append([]string(nil), gemma4LoRAStandardTargets...), gemma4LoRAExtendedTargets...)
	p.LoRADefaultTargets = gemma4LoRADefaultTargets
	p.LoRATargetPaths = gemma4LoRATargetPaths
	p.LoRAExtendedTargets = gemma4LoRAExtendedTargets
	p.WeightWrapperPrefixes = gemma4WeightWrapperPrefixes
	p.WeightSkipPrefixes = gemma4WeightSkipPrefixes
	p.WeightSkipSubstrings = gemma4WeightSkipSubstrings
	p.WeightModelPrefixes = gemma4WeightModelPrefixes
	return p
}

func nativeProfile(id, family, parser string, aliases []string) ModelArchitectureProfile {
	profile := metadataProfile(id, family, parser, parser, false, false, aliases, nil)
	profile.RuntimeStatus = ArchitectureRuntimeNative
	profile.NativeRuntime = true
	return profile
}

func nativeAttachedDrafterProfile(id, family, parser string, aliases, notes []string) ModelArchitectureProfile {
	profile := metadataProfile(id, family, parser, parser, false, false, aliases, notes)
	profile.RuntimeStatus = ArchitectureRuntimeNative
	profile.NativeRuntime = true
	profile.Generation = false
	profile.Chat = false
	profile.RequiresChatTemplate = false
	profile.ChatTemplate = ""
	profile.LoRATargets = nil
	return profile
}

func nativeStagedProfile(id, family, parser string, moe bool, aliases, notes []string) ModelArchitectureProfile {
	profile := metadataProfile(id, family, parser, parser, moe, false, aliases, notes)
	profile.RuntimeStatus = ArchitectureRuntimeNative
	profile.NativeRuntime = true
	profile.Generation = false
	profile.Chat = false
	profile.RequiresChatTemplate = false
	profile.ChatTemplate = ""
	return profile
}

func nativeEncoderStagedProfile(id, family, parser string, aliases, notes []string) ModelArchitectureProfile {
	profile := metadataProfile(id, family, parser, parser, false, true, aliases, notes)
	profile.RuntimeStatus = ArchitectureRuntimeNative
	profile.NativeRuntime = true
	return profile
}

func nativeRerankStagedProfile(id, family string, aliases, notes []string) ModelArchitectureProfile {
	profile := rerankProfile(id, family, aliases, notes)
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
		LoRATargets:          architectureDefaultLoRATargets(id, family, moe),
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

func architectureDefaultLoRATargets(id, family string, moe bool) []string {
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
	profile.LoRADefaultTargets = append([]string(nil), profile.LoRADefaultTargets...)
	profile.LoRAExtendedTargets = append([]string(nil), profile.LoRAExtendedTargets...)
	profile.WeightWrapperPrefixes = append([]string(nil), profile.WeightWrapperPrefixes...)
	profile.WeightSkipPrefixes = append([]string(nil), profile.WeightSkipPrefixes...)
	profile.WeightSkipSubstrings = append([]string(nil), profile.WeightSkipSubstrings...)
	profile.WeightModelPrefixes = append([]string(nil), profile.WeightModelPrefixes...)
	profile.LoRATargetPaths = cloneStringMap(profile.LoRATargetPaths)
	profile.QuantizationHints = append([]string(nil), profile.QuantizationHints...)
	profile.CacheHints = append([]string(nil), profile.CacheHints...)
	profile.Notes = append([]string(nil), profile.Notes...)
	profile.Aliases = append([]string(nil), profile.Aliases...)
	return profile
}

func cloneStringMap(in map[string]string) map[string]string {
	if len(in) == 0 {
		return nil
	}
	out := make(map[string]string, len(in))
	for key, value := range in {
		out[key] = value
	}
	return out
}

func ArchitectureIDs() []string {
	profiles := builtinArchitectureProfiles()
	out := make([]string, 0, len(profiles))
	for _, profile := range profiles {
		out = append(out, profile.ID)
	}
	return out
}

// NormalizeArchitecture canonicalises an architecture identifier to the
// stable id the model registry dispatches on. It lowercases, trims, and
// folds '-'/'.' to '_', then maps known aliases (e.g. "Qwen3.6" → "qwen3_6",
// "MiniMax-M2" → "minimax_m2") to their canonical id; an unknown value is
// returned in its normalised form. This is the single source of truth — the
// memory, gguf, model, and minimax packages call it rather than carrying
// their own (previously-drifted) copies.
//
//	id := profile.NormalizeArchitecture("Qwen3.6")  // → "qwen3_6"
func NormalizeArchitecture(value string) string {
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
	case "kimi", "moonshot":
		return "kimi"
	case "bert", "bert_model":
		return "bert"
	case "bert_rerank", "bert_cross_encoder":
		return "bert_rerank"
	case "gemma4_unified":
		return "gemma4_unified"
	case "gemma4_unified_text":
		return "gemma4_text"
	default:
		return value
	}
}

// ArchitectureFromTransformersName maps a HuggingFace transformers
// architecture class name (e.g. "Qwen3MoeForCausalLM",
// "Gemma4AssistantForCausalLM") to its canonical go-mlx model-type id, or ""
// when the name matches no known family. This is the single source of truth —
// the gguf, model, and hf packages call it rather than carrying their own
// (previously-drifted) copies, which had variously lost the qwen3_6 and
// gemma4_assistant arms.
//
//	id := profile.ArchitectureFromTransformersName("Qwen3MoeForCausalLM")  // → "qwen3_moe"
func ArchitectureFromTransformersName(architecture string) string {
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
	case core.Contains(architecture, "Gemma4UnifiedForConditionalGeneration"):
		return "gemma4_unified"
	case core.Contains(architecture, "Gemma4ForConditionalGeneration"),
		core.Contains(architecture, "Gemma4Multimodal"),
		core.Contains(architecture, "Gemma4Vision"):
		// Multimodal gemma4 loads via the base Gemma4 family, not text-only
		// "gemma4_text". The Unified 12B class has its own canonical ID above
		// so metadata can distinguish its 256K multimodal contract.
		return "gemma4"
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
	case core.Contains(architecture, "Kimi") || core.Contains(architecture, "Moonshot"):
		return "kimi"
	case core.Contains(architecture, "Hermes"):
		return "hermes"
	case core.Contains(architecture, "Granite"):
		return "granite"
	case core.Contains(architecture, "Glm") || core.Contains(architecture, "GLM"):
		return "glm"
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

// SPDX-Licence-Identifier: EUPL-1.2

package profile_test

import (
	"testing"

	prof "dappco.re/go/mlx/profile"
)

func requireExactLoRATargets(t *testing.T, got, want []string) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("LoRATargets = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("LoRATargets = %v, want %v", got, want)
		}
	}
}

func TestArchitectureProfile_MetadataFamilies_Good(t *testing.T) {
	cases := []struct {
		name       string
		input      string
		wantID     string
		wantParser string
		wantMoE    bool
		wantEmbed  bool
		wantNative bool
	}{
		{name: "minimax", input: "MiniMaxM2ForCausalLM", wantID: "minimax_m2", wantParser: "minimax", wantMoE: true, wantNative: true},
		{name: "mixtral", input: "MixtralForCausalLM", wantID: "mixtral", wantParser: "mistral", wantMoE: true, wantNative: true},
		{name: "mistral", input: "mistral", wantID: "mistral", wantParser: "mistral", wantNative: true},
		{name: "hermes", input: "HermesForCausalLM", wantID: "hermes", wantParser: "hermes", wantNative: true},
		{name: "granite", input: "GraniteForCausalLM", wantID: "granite", wantParser: "granite", wantNative: true},
		{name: "phi", input: "Phi3ForCausalLM", wantID: "phi", wantParser: "generic", wantNative: true},
		{name: "glm", input: "GlmForCausalLM", wantID: "glm", wantParser: "glm", wantNative: true},
		{name: "kimi", input: "KimiForCausalLM", wantID: "kimi", wantParser: "kimi", wantMoE: true, wantNative: true},
		{name: "deepseek", input: "DeepseekV3ForCausalLM", wantID: "deepseek", wantParser: "deepseek-r1", wantMoE: true, wantNative: true},
		{name: "gptoss", input: "GptOssForCausalLM", wantID: "gpt_oss", wantParser: "gpt-oss", wantMoE: true, wantNative: true},
		{name: "bert", input: "BertModel", wantID: "bert", wantParser: "generic", wantEmbed: true, wantNative: true},
		{name: "bert-rerank", input: "BertForSequenceClassification", wantID: "bert_rerank", wantParser: "generic", wantNative: true},
		{name: "qwen-native", input: "qwen3", wantID: "qwen3", wantParser: "qwen", wantNative: true},
		{name: "qwen3-moe", input: "Qwen3MoeForCausalLM", wantID: "qwen3_moe", wantParser: "qwen", wantMoE: true, wantNative: true},
		{name: "qwen2-5-native", input: "Qwen2.5ForCausalLM", wantID: "qwen2", wantParser: "qwen", wantNative: true},
		{name: "gemma4-unified", input: "Gemma4UnifiedForConditionalGeneration", wantID: "gemma4_unified", wantParser: "gemma", wantNative: true},
		{name: "gemma4-assistant", input: "gemma4_assistant", wantID: "gemma4_assistant", wantParser: "gemma", wantNative: true},
		{name: "qwen36-dense", input: "Qwen3_5ForConditionalGeneration", wantID: "qwen3_6", wantParser: "qwen", wantNative: true},
		{name: "qwen36-moe", input: "Qwen3_5MoeForConditionalGeneration", wantID: "qwen3_6_moe", wantParser: "qwen", wantMoE: true, wantNative: true},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			p, ok := prof.LookupArchitectureProfile(tc.input)
			if !ok {
				t.Fatalf("prof.LookupArchitectureProfile(%q) ok = false", tc.input)
			}
			if p.ID != tc.wantID || p.ParserID != tc.wantParser {
				t.Fatalf("profile = %+v, want id %q parser %q", p, tc.wantID, tc.wantParser)
			}
			if p.MoE != tc.wantMoE || p.Embeddings != tc.wantEmbed || p.NativeRuntime != tc.wantNative {
				t.Fatalf("profile flags = moe:%v embeddings:%v native:%v, want %v/%v/%v", p.MoE, p.Embeddings, p.NativeRuntime, tc.wantMoE, tc.wantEmbed, tc.wantNative)
			}
			if tc.name == "bert-rerank" && !p.Rerank {
				t.Fatalf("profile = %+v, want rerank profile", p)
			}
			if tc.name == "gemma4-assistant" && (p.Generation || p.Chat || p.RequiresChatTemplate) {
				t.Fatalf("profile = %+v, want attached native drafter without standalone chat/generation", p)
			}
			if tc.name == "minimax" && (p.Generation || p.Chat || !p.MoE) {
				t.Fatalf("profile = %+v, want staged native MiniMax M2 loader without standalone generation", p)
			}
			if tc.name == "qwen36-dense" && (p.Generation || p.Chat || p.MoE) {
				t.Fatalf("profile = %+v, want staged native Qwen3.6 loader without standalone generation/chat or MoE", p)
			}
			if tc.name == "qwen3-moe" && (p.Generation || p.Chat || !p.MoE) {
				t.Fatalf("profile = %+v, want staged native Qwen3 MoE loader without standalone generation/chat", p)
			}
			if tc.name == "mixtral" && (p.Generation || p.Chat || !p.MoE) {
				t.Fatalf("profile = %+v, want staged native mixtral loader without standalone generation/chat", p)
			}
			if tc.name == "deepseek" && (p.Generation || p.Chat || !p.MoE) {
				t.Fatalf("profile = %+v, want staged native deepseek loader without standalone generation/chat", p)
			}
			if tc.name == "gptoss" && (p.Generation || p.Chat || !p.MoE) {
				t.Fatalf("profile = %+v, want staged native gpt_oss loader without standalone generation/chat", p)
			}
			if tc.name == "kimi" && (p.Generation || p.Chat || !p.MoE) {
				t.Fatalf("profile = %+v, want staged native kimi loader without standalone generation/chat", p)
			}
			if tc.name == "qwen36-moe" && (p.Generation || p.Chat || !p.MoE) {
				t.Fatalf("profile = %+v, want staged native Qwen3.6 MoE loader without standalone generation/chat", p)
			}
		})
	}
}

func TestArchitectureProfile_Gemma4TargetArchitecture_Good(t *testing.T) {
	cases := []struct {
		architecture string
		want         bool
	}{
		{architecture: "gemma4", want: true},
		{architecture: "gemma4_text", want: true},
		{architecture: "gemma4_unified", want: true},
		{architecture: "gemma4_unified_text", want: true},
		{architecture: "Gemma4ForConditionalGeneration", want: true},
		{architecture: "Gemma4UnifiedForConditionalGeneration", want: true},
		{architecture: "Gemma4ForCausalLM", want: true},
		{architecture: "Gemma4TextForCausalLM", want: true},
		{architecture: "gemma4_assistant"},
		{architecture: "Gemma4AssistantForCausalLM"},
		{architecture: "gemma3"},
		{architecture: "qwen3"},
		{architecture: ""},
	}
	for _, tc := range cases {
		t.Run(tc.architecture, func(t *testing.T) {
			if got := prof.IsGemma4TargetArchitecture(tc.architecture); got != tc.want {
				t.Fatalf("prof.IsGemma4TargetArchitecture(%q) = %v, want %v", tc.architecture, got, tc.want)
			}
		})
	}
}

func TestArchitectureProfile_Gemma4LargeVariant_Good(t *testing.T) {
	cases := []struct {
		name         string
		architecture string
		heads        int
		want         bool
	}{
		{name: "large official target", architecture: "Gemma4ForConditionalGeneration", heads: 16, want: true},
		{name: "large unified alias", architecture: "gemma4_unified_text", heads: 16, want: true},
		{name: "small target", architecture: "gemma4_text", heads: 8},
		{name: "assistant excluded", architecture: "Gemma4AssistantForCausalLM", heads: 16},
		{name: "non gemma excluded", architecture: "qwen3", heads: 16},
		{name: "missing heads", architecture: "gemma4_text"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := prof.IsGemma4LargeVariant(tc.architecture, tc.heads); got != tc.want {
				t.Fatalf("prof.IsGemma4LargeVariant(%q, %d) = %v, want %v", tc.architecture, tc.heads, got, tc.want)
			}
		})
	}
}

func TestArchitectureProfile_ChatTemplateName_Good(t *testing.T) {
	cases := []struct {
		architecture string
		want         string
	}{
		{architecture: "Gemma4ForConditionalGeneration", want: "gemma4"},
		{architecture: "gemma4_unified_text", want: "gemma4"},
		{architecture: "Gemma4AssistantForCausalLM"},
		{architecture: "Gemma3ForCausalLM", want: "gemma"},
		{architecture: "qwen3_6_moe", want: "qwen"},
		{architecture: "llama3", want: "llama"},
		{architecture: "MiniMaxM2ForCausalLM"},
		{architecture: "DeepseekV3ForCausalLM"},
		{architecture: "unknown"},
		{architecture: ""},
	}
	for _, tc := range cases {
		t.Run(tc.architecture, func(t *testing.T) {
			if got := prof.ChatTemplateName(tc.architecture); got != tc.want {
				t.Fatalf("prof.ChatTemplateName(%q) = %q, want %q", tc.architecture, got, tc.want)
			}
		})
	}
}

// TestArchitectureProfile_Gemma4LoRAPolicy_Good exercises the Gemma-4 LoRA
// policy through the generic registry accessors — the loader-neutral data lives
// in the registry, no standalone Gemma4* functions, no model package imported.
func TestArchitectureProfile_Gemma4LoRAPolicy_Good(t *testing.T) {
	want := []string{"q_proj", "v_proj", "o_proj"}
	for _, architecture := range []string{
		"gemma4",
		"gemma4_text",
		"gemma4_unified",
		"Gemma4ForConditionalGeneration",
		"Gemma4UnifiedForConditionalGeneration",
	} {
		t.Run(architecture, func(t *testing.T) {
			requireExactLoRATargets(t, prof.DefaultLoRATargets(architecture), want)
			cases := []struct {
				target   string
				wantPath string
				wantSafe bool
			}{
				{"q_proj", "self_attn.q_proj", true},
				{"self_attn.q_proj", "self_attn.q_proj", true},
				{"gate_proj", "mlp.gate_proj", true},
				{"mlp.up_proj", "mlp.up_proj", true},
				{"router.proj", "router.proj", false},
				{"per_layer_input_gate", "per_layer_input_gate", false},
			}
			for _, tc := range cases {
				path, ok := prof.LoRATargetPath(architecture, tc.target)
				if !ok || path != tc.wantPath {
					t.Fatalf("prof.LoRATargetPath(%q, %q) = %q, %v; want %q, true", architecture, tc.target, path, ok, tc.wantPath)
				}
				if safe := prof.SafeLoRATarget(architecture, tc.target); safe != tc.wantSafe {
					t.Fatalf("prof.SafeLoRATarget(%q, %q) = %v, want %v", architecture, tc.target, safe, tc.wantSafe)
				}
			}
			if _, ok := prof.LoRATargetPath(architecture, "vision_tower.q_proj"); ok {
				t.Fatalf("prof.LoRATargetPath(%q, vision_tower.q_proj) ok = true, want false", architecture)
			}
		})
	}

	// Returned defaults are a copy — mutating them must not corrupt the registry.
	prof.DefaultLoRATargets("gemma4")[0] = "mutated"
	requireExactLoRATargets(t, prof.DefaultLoRATargets("gemma4"), want)

	// An unknown architecture yields no policy rather than a guess.
	if got := prof.DefaultLoRATargets("nonexistent_family"); got != nil {
		t.Fatalf("prof.DefaultLoRATargets(nonexistent) = %v, want nil", got)
	}

	// The attached drafter advertises no LoRA targets.
	assistant, ok := prof.LookupArchitectureProfile("gemma4_assistant")
	if !ok {
		t.Fatalf("prof.LookupArchitectureProfile(gemma4_assistant) ok = false")
	}
	if len(assistant.LoRATargets) != 0 {
		t.Fatalf("gemma4_assistant LoRATargets = %v, want none for the attached drafter", assistant.LoRATargets)
	}
}

// TestArchitectureProfile_Gemma4CanonicalWeightName_Good exercises the weight-
// name canonicalisation through the generic accessor — the Gemma-4 wrapper /
// skip / model-prefix rules live as registry data, the engine names no family.
func TestArchitectureProfile_Gemma4CanonicalWeightName_Good(t *testing.T) {
	cases := []struct {
		name string
		want string
		ok   bool
	}{
		{name: "language_model.model.layers.0.self_attn.q_proj.weight", want: "model.layers.0.self_attn.q_proj.weight", ok: true},
		{name: "model.language_model.model.model.layers.1.mlp.down_proj.scales", want: "model.layers.1.mlp.down_proj.scales", ok: true},
		{name: "model.layers.2.self_attn.o_proj.weight", want: "model.layers.2.self_attn.o_proj.weight", ok: true},
		{name: "language_model.model.layers.0.self_attn.q_proj.input_max"},
		{name: "model.vision_tower.patch_embedding.weight"},
		{name: "language_model.embed_audio.embedding_projection.weight"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, ok := prof.CanonicalWeightName("gemma4", tc.name)
			if ok != tc.ok || got != tc.want {
				t.Fatalf("prof.CanonicalWeightName(gemma4, %q) = %q, %v; want %q, %v", tc.name, got, ok, tc.want, tc.ok)
			}
		})
	}

	// TrimWeightWrapperPrefix strips one wrapper; an unknown architecture is a no-op.
	if got, ok := prof.TrimWeightWrapperPrefix("gemma4", "language_model.model.layers.0"); !ok || got != "layers.0" {
		t.Fatalf("prof.TrimWeightWrapperPrefix(gemma4, ...) = %q, %v; want layers.0, true", got, ok)
	}
	if got, ok := prof.TrimWeightWrapperPrefix("nonexistent_family", "model.layers.0"); ok || got != "model.layers.0" {
		t.Fatalf("prof.TrimWeightWrapperPrefix(unknown) = %q, %v; want model.layers.0, false", got, ok)
	}
}

func TestArchitectureProfile_BuiltinIDs_Good(t *testing.T) {
	profiles := prof.BuiltinArchitectureProfiles()
	if len(profiles) < 12 {
		t.Fatalf("prof.BuiltinArchitectureProfiles len = %d, want broad feature-parity target list", len(profiles))
	}
	seen := map[string]bool{}
	for _, profile := range profiles {
		if profile.ID == "" {
			t.Fatalf("profile missing ID: %+v", profile)
		}
		if seen[profile.ID] {
			t.Fatalf("duplicate profile ID %q", profile.ID)
		}
		seen[profile.ID] = true
	}
	for _, id := range []string{"gemma4_text", "gemma4_unified", "gemma4_assistant", "qwen2", "qwen3_next", "qwen3_6", "qwen3_6_moe", "qwen3_moe", "minimax_m2", "mixtral", "deepseek", "gpt_oss", "bert", "bert_rerank"} {
		if !seen[id] {
			t.Fatalf("missing builtin architecture profile %q", id)
		}
	}
}

// TestArchitectureID_SubstringFallback_Good pins the compact-substring fallback
// arm of ArchitectureID — the resolution path a config model_type takes when it
// carries a family fragment but is not a clean Transformers class name (so
// ArchitectureFromTransformersName returns "") and is not a direct alias of
// NormalizeArchitecture. Each input folds, compacts, and matches one family
// substring; the order is authoritative (the moe/next arms are probed before
// the bare qwen3 arm, the rerank class names before bare bert), so these cases
// guard that ordering as new families land.
func TestArchitectureID_SubstringFallback_Good(t *testing.T) {
	cases := map[string]string{
		"my_qwen35moe_v2":                    "qwen3_6_moe",
		"custom-qwen3.6":                     "qwen3_6",
		"qwen3moe_local":                     "qwen3_moe",
		"qwen3next_x":                        "qwen3_next",
		"minimaxm2_q4":                       "minimax_m2",
		"mixtral_local":                      "mixtral",
		"my-mistral":                         "mistral",
		"deepseek_local":                     "deepseek",
		"gptoss_x":                           "gpt_oss",
		"phi_local":                          "phi",
		"DebertaV2ForSequenceClassification": "bert_rerank",
		"bert_local":                         "bert",
	}
	for in, want := range cases {
		t.Run(in, func(t *testing.T) {
			if got := prof.ArchitectureID(in); got != want {
				t.Fatalf("prof.ArchitectureID(%q) = %q, want %q", in, got, want)
			}
		})
	}
}

// TestArchitectureID_PassThroughAndEmpty_Bad pins the two non-resolving paths:
// an empty input yields the empty id, and an input that names no family is
// returned in its normalised form rather than guessed at.
func TestArchitectureID_PassThroughAndEmpty_Bad(t *testing.T) {
	if got := prof.ArchitectureID(""); got != "" {
		t.Fatalf("prof.ArchitectureID(\"\") = %q, want empty", got)
	}
	if got := prof.ArchitectureID("  "); got != "" {
		t.Fatalf("prof.ArchitectureID(\"  \") = %q, want empty after trim", got)
	}
	if got := prof.ArchitectureID("Totally-Unknown.Thing"); got != "totally_unknown_thing" {
		t.Fatalf("prof.ArchitectureID(unknown) = %q, want normalised pass-through", got)
	}
}

// TestArchitectureMetadataAccessors_UnknownArchitecture_Bad pins the unknown /
// empty miss branch of the metadata-only accessors — the path the loader hits
// for a family the registry does not carry. Each must report the safe default
// (no thinking, not attached-only) rather than panic or guess.
func TestArchitectureMetadataAccessors_UnknownArchitecture_Bad(t *testing.T) {
	for _, architecture := range []string{"", "  ", "nonexistent_family"} {
		t.Run(architecture, func(t *testing.T) {
			if prof.DefaultThinkingEnabled(architecture) {
				t.Fatalf("prof.DefaultThinkingEnabled(%q) = true, want false for unknown", architecture)
			}
			if prof.AttachedOnlyArchitecture(architecture) {
				t.Fatalf("prof.AttachedOnlyArchitecture(%q) = true, want false for unknown", architecture)
			}
		})
	}
}

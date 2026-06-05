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

func containsString(values []string, want string) bool {
	for _, value := range values {
		if value == want {
			return true
		}
	}
	return false
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

func TestArchitectureProfile_Gemma4LoRATargetsUseSharedPolicy_Good(t *testing.T) {
	wantTargets := prof.Gemma4LoRATargets()
	for _, architecture := range []string{
		"gemma4",
		"gemma4_text",
		"gemma4_unified",
		"Gemma4ForConditionalGeneration",
		"Gemma4ForCausalLM",
		"Gemma4UnifiedForConditionalGeneration",
	} {
		t.Run(architecture, func(t *testing.T) {
			profile, ok := prof.LookupArchitectureProfile(architecture)
			if !ok {
				t.Fatalf("prof.LookupArchitectureProfile(%q) ok = false", architecture)
			}
			requireExactLoRATargets(t, profile.LoRATargets, wantTargets)
		})
	}

	assistant, ok := prof.LookupArchitectureProfile("gemma4_assistant")
	if !ok {
		t.Fatalf("prof.LookupArchitectureProfile(%q) ok = false", "gemma4_assistant")
	}
	if len(assistant.LoRATargets) != 0 {
		t.Fatalf("gemma4_assistant LoRATargets = %v, want none for attached-only drafter", assistant.LoRATargets)
	}
}

func TestArchitectureProfile_Gemma4DefaultLoRATargets_Good(t *testing.T) {
	defaults := prof.Gemma4DefaultLoRATargets()
	want := []string{"q_proj", "v_proj", "o_proj"}
	requireExactLoRATargets(t, defaults, want)

	defaults[0] = "mutated"
	again := prof.Gemma4DefaultLoRATargets()
	requireExactLoRATargets(t, again, want)

	metadata := prof.Gemma4LoRATargets()
	for _, target := range again {
		if !prof.Gemma4SafeLoRATarget(target) {
			t.Fatalf("default target %q is not safe", target)
		}
		if !containsString(metadata, target) {
			t.Fatalf("default target %q missing from Gemma4LoRATargets %v", target, metadata)
		}
	}
	for _, target := range []string{"k_proj", "gate_proj", "router.proj", "per_layer_input_gate", "per_layer_projection"} {
		if containsString(again, target) {
			t.Fatalf("default targets = %v, want %q explicit", again, target)
		}
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

func TestArchitectureProfile_Gemma4LoRATargetPolicy_Good(t *testing.T) {
	cases := []struct {
		name     string
		target   string
		wantPath string
		wantSafe bool
	}{
		{name: "q suffix", target: "q_proj", wantPath: "self_attn.q_proj", wantSafe: true},
		{name: "q full", target: "self_attn.q_proj", wantPath: "self_attn.q_proj", wantSafe: true},
		{name: "mlp suffix", target: "gate_proj", wantPath: "mlp.gate_proj", wantSafe: true},
		{name: "mlp full", target: "mlp.up_proj", wantPath: "mlp.up_proj", wantSafe: true},
		{name: "router extended", target: "router.proj", wantPath: "router.proj"},
		{name: "ple extended", target: "per_layer_input_gate", wantPath: "per_layer_input_gate"},
		{name: "unknown", target: "vision_tower.q_proj"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			gotPath, ok := prof.Gemma4LoRATargetPath(tc.target)
			if tc.wantPath == "" {
				if ok || gotPath != "" {
					t.Fatalf("prof.Gemma4LoRATargetPath(%q) = %q, %t; want unsupported", tc.target, gotPath, ok)
				}
				return
			}
			if !ok || gotPath != tc.wantPath {
				t.Fatalf("prof.Gemma4LoRATargetPath(%q) = %q, %t; want %q, true", tc.target, gotPath, ok, tc.wantPath)
			}
			if gotSafe := prof.Gemma4SafeLoRATarget(tc.target); gotSafe != tc.wantSafe {
				t.Fatalf("prof.Gemma4SafeLoRATarget(%q) = %v, want %v", tc.target, gotSafe, tc.wantSafe)
			}
		})
	}

	targets := prof.Gemma4LoRATargets()
	targets[0] = "mutated"
	if next := prof.Gemma4LoRATargets(); next[0] == "mutated" {
		t.Fatalf("prof.Gemma4LoRATargets leaked mutable backing slice: %v", next)
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

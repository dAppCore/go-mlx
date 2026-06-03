// SPDX-Licence-Identifier: EUPL-1.2

package profile_test

import (
	"testing"

	prof "dappco.re/go/mlx/profile"
)

func TestArchitectureProfile_MetadataFamilies_Good(t *testing.T) {
	coverageTokens := "ArchitectureProfile MetadataFamilies"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
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
	for _, id := range []string{"gemma4_text", "gemma4_assistant", "qwen2", "qwen3_next", "qwen3_6", "qwen3_6_moe", "qwen3_moe", "minimax_m2", "mixtral", "deepseek", "gpt_oss", "bert", "bert_rerank"} {
		if !seen[id] {
			t.Fatalf("missing builtin architecture profile %q", id)
		}
	}
}

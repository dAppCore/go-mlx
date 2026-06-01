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
		{name: "minimax", input: "MiniMaxM2ForCausalLM", wantID: "minimax_m2", wantParser: "minimax", wantMoE: true},
		{name: "mixtral", input: "MixtralForCausalLM", wantID: "mixtral", wantParser: "mistral", wantMoE: true},
		{name: "mistral", input: "mistral", wantID: "mistral", wantParser: "mistral", wantNative: true},
		{name: "hermes", input: "HermesForCausalLM", wantID: "hermes", wantParser: "hermes", wantNative: true},
		{name: "granite", input: "GraniteForCausalLM", wantID: "granite", wantParser: "granite", wantNative: true},
		{name: "phi", input: "Phi3ForCausalLM", wantID: "phi", wantParser: "generic", wantNative: true},
		{name: "deepseek", input: "DeepseekV3ForCausalLM", wantID: "deepseek", wantParser: "deepseek-r1", wantMoE: true},
		{name: "gptoss", input: "GptOssForCausalLM", wantID: "gpt_oss", wantParser: "gpt-oss", wantMoE: true},
		{name: "bert", input: "BertModel", wantID: "bert", wantParser: "generic", wantEmbed: true},
		{name: "bert-rerank", input: "BertForSequenceClassification", wantID: "bert_rerank", wantParser: "generic"},
		{name: "qwen-native", input: "qwen3", wantID: "qwen3", wantParser: "qwen", wantNative: true},
		{name: "qwen2-5-native", input: "Qwen2.5ForCausalLM", wantID: "qwen2", wantParser: "qwen", wantNative: true},
		{name: "gemma4-assistant", input: "gemma4_assistant", wantID: "gemma4_assistant", wantParser: "gemma"},
		{name: "qwen36-dense", input: "Qwen3_5ForConditionalGeneration", wantID: "qwen3_6", wantParser: "qwen"},
		{name: "qwen36-moe", input: "Qwen3_5MoeForConditionalGeneration", wantID: "qwen3_6_moe", wantParser: "qwen", wantMoE: true},
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

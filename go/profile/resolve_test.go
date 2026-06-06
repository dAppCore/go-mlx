// SPDX-Licence-Identifier: EUPL-1.2

package profile_test

import (
	"testing"

	"dappco.re/go/mlx/profile"
)

// TestResolveArchitecture_Good pins the full config-probe → registered-id
// resolution the loader depends on. It is the single home for the resolution
// ORDER (top-level model_type, then a declared text-tower, then the
// architectures fallback) plus the two family refinements that used to live as
// name-branches in the metal loader: a Gemma-4 multimodal wrapper resolves to
// its declared text tower, and a BERT encoder whose architectures name a
// cross-encoder resolves to the rerank variant. Every case mirrors a behaviour
// the metal probeModelType tests already pin, so this guards exactness as the
// knowledge moves into the registry.
func TestResolveArchitecture_Good(t *testing.T) {
	cases := []struct {
		name      string
		modelType string
		textTower string
		archs     []string
		want      string
	}{
		// Top-level model_type, canonicalised through NormalizeArchitecture.
		{"qwen2.5 alias", "qwen2.5", "", []string{"Qwen2.5ForCausalLM"}, "qwen2"},
		{"qwen3.5 → 3.6", "qwen3_5", "", []string{"Qwen3_5ForConditionalGeneration"}, "qwen3_6"},
		{"qwen3.5 moe", "qwen3_5_moe", "", []string{"Qwen3_5MoeForConditionalGeneration"}, "qwen3_6_moe"},
		{"qwen3_5 model_type only", "qwen3_5", "", nil, "qwen3_6"},
		// Text-tower fallback when there is no top-level model_type.
		{"text_config qwen", "", "qwen3_5_text", []string{"Qwen3_5ForConditionalGeneration"}, "qwen3_6"},
		// Architectures fallback (no model_type, no text tower).
		{"arch mistral", "", "", []string{"MistralForCausalLM"}, "mistral"},
		{"arch hermes", "", "", []string{"HermesForCausalLM"}, "hermes"},
		{"arch granite", "", "", []string{"GraniteForCausalLM"}, "granite"},
		{"arch phi3", "", "", []string{"Phi3ForCausalLM"}, "phi"},
		{"arch glm", "", "", []string{"GlmForCausalLM"}, "glm"},
		{"arch qwen3 moe", "", "", []string{"Qwen3MoeForCausalLM"}, "qwen3_moe"},
		{"arch qwen3 next", "", "", []string{"Qwen3NextForCausalLM"}, "qwen3_next"},
		{"arch minimax", "", "", []string{"MiniMaxM2ForCausalLM"}, "minimax_m2"},
		// Gemma-4 multimodal wrapper resolves to its declared text tower.
		{"gemma4 multimodal → text", "gemma4", "gemma4_text", []string{"Gemma4ForConditionalGeneration"}, "gemma4_text"},
		// A Gemma-4 wrapper with no matching text tower stays the wrapper.
		{"gemma4 no tower stays gemma4", "gemma4", "", []string{"Gemma4ForConditionalGeneration"}, "gemma4"},
		// gemma4_unified is its own canonical 12B multimodal id (no text-tower refinement).
		{"gemma4_unified stays unified", "gemma4_unified", "gemma4_unified_text", []string{"Gemma4UnifiedForConditionalGeneration"}, "gemma4_unified"},
		// The unified text tower normalises to gemma4_text.
		{"gemma4_unified_text → text", "gemma4_unified_text", "", []string{"Gemma4TextForCausalLM"}, "gemma4_text"},
		// BERT encoder vs cross-encoder, distinguished only by architectures.
		{"bert plain", "bert", "", []string{"BertModel"}, "bert"},
		{"bert rerank", "bert", "", []string{"BertForSequenceClassification"}, "bert_rerank"},
		{"bert rerank xlm", "bert", "", []string{"XLMRobertaForSequenceClassification"}, "bert_rerank"},
		// Nothing to resolve.
		{"empty", "", "", nil, ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := profile.ResolveArchitecture(tc.modelType, tc.textTower, tc.archs)
			if got != tc.want {
				t.Fatalf("ResolveArchitecture(%q, %q, %v) = %q, want %q", tc.modelType, tc.textTower, tc.archs, got, tc.want)
			}
		})
	}
}

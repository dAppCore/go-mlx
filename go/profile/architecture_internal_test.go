// SPDX-Licence-Identifier: EUPL-1.2

// Internal parity tests for the byte-walk compactArchitectureNameInto
// helper introduced in W11-E. The hot-path zero-alloc variant MUST
// produce bit-exact output against the heap-allocating fallback
// (which preserves the pre-W11E core.Lower + core.Replace semantics)
// for every architecture name the package ever resolves.

package profile

import "strings"

import "testing"

func TestCompactArchitectureNameInto_ParityWithFallback(t *testing.T) {
	cases := []string{
		"",
		"gemma2",
		"Gemma3ForCausalLM",
		"Gemma4ForConditionalGeneration",
		"Gemma4TextForCausalLM",
		"Gemma4AssistantForCausalLM",
		"LlamaForCausalLM",
		"Qwen2ForCausalLM",
		"Qwen2.5ForCausalLM",
		"Qwen2_5ForCausalLM",
		"Qwen3ForCausalLM",
		"Qwen3NextForCausalLM",
		"Qwen3_5ForConditionalGeneration",
		"Qwen3.5ForConditionalGeneration",
		"Qwen3_6ForConditionalGeneration",
		"Qwen3.6ForConditionalGeneration",
		"Qwen3_5MoeForConditionalGeneration",
		"Qwen3.5MoeForConditionalGeneration",
		"Qwen3_6MoeForConditionalGeneration",
		"Qwen3.6MoeForConditionalGeneration",
		"Qwen3MoeForCausalLM",
		"MiniMaxM2ForCausalLM",
		"MistralForCausalLM",
		"MixtralForCausalLM",
		"PhiForCausalLM",
		"Phi3ForCausalLM",
		"Phi4ForCausalLM",
		"DeepseekV3ForCausalLM",
		"DeepSeekV3ForCausalLM",
		"DeepseekR1ForCausalLM",
		"GptOssForCausalLM",
		"GPTOSSForCausalLM",
		"KimiForCausalLM",
		"MoonshotForCausalLM",
		"GlmForCausalLM",
		"ChatGLMForConditionalGeneration",
		"HermesForCausalLM",
		"GraniteForCausalLM",
		"BertModel",
		"BertForMaskedLM",
		"BertForSequenceClassification",
		"RobertaForSequenceClassification",
		"XLMRobertaForSequenceClassification",
		"DebertaV2ForSequenceClassification",
		"qwen-3.5",
		"qwen_3_5",
		"qwen3.5",
		"qwen35",
		"qwen36",
		"gpt_oss_model",
		"bert-cross-encoder",
		"foo_bar-baz.qux",
		"already_lowercase_with_dots.and-dashes",
	}
	var buf [maxArchitectureNameBytes]byte
	for _, in := range cases {
		got := compactArchitectureNameInto(buf[:], in)
		want := compactArchitectureNameFallback(in)
		if got != want {
			t.Errorf("compactArchitectureNameInto(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestCompactArchitectureNameInto_FallbackOnOverflow(t *testing.T) {
	// Input longer than the stack buffer must fall back cleanly to
	// the heap-stable helper — no panic, identical output.
	var long strings.Builder
	for range maxArchitectureNameBytes + 1 {
		long.WriteString("x")
	}
	var buf [maxArchitectureNameBytes]byte
	got := compactArchitectureNameInto(buf[:], long.String())
	want := compactArchitectureNameFallback(long.String())
	if got != want {
		t.Fatalf("overflow fallback diverged: got %q want %q", got, want)
	}
}

func TestCompactArchitectureNameInto_FallbackOnNonASCII(t *testing.T) {
	// Non-ASCII byte must trigger fallback, preserving Lower-via-
	// Unicode-table semantics.
	in := "Café-Gemma3"
	var buf [maxArchitectureNameBytes]byte
	got := compactArchitectureNameInto(buf[:], in)
	want := compactArchitectureNameFallback(in)
	if got != want {
		t.Fatalf("non-ASCII fallback diverged: got %q want %q", got, want)
	}
}

// TestNormalizeArchitecture_KnownAliases_Good locks the canonical
// architecture-alias contract. profile.NormalizeArchitecture is the single
// source of truth the memory, gguf, model, and minimax packages now share
// (each previously carried its own drifted copy — gguf/minimax had frozen
// "qwen3_5" at the old "qwen3_next" id), so the alias map and the
// lowercase/trim/'-'.'→'_' normalisation are pinned here.
func TestNormalizeArchitecture_KnownAliases_Good(t *testing.T) {
	cases := map[string]string{
		"qwen3_5":            "qwen3_6", // the corrected fold — was "qwen3_next" in the stale copies
		"qwen3.6":            "qwen3_6", // dot folds to underscore
		"qwen3_5_text":       "qwen3_6",
		"qwen3_5_moe":        "qwen3_6_moe",
		"qwen2.5":            "qwen2",
		"MiniMax-M2":         "minimax_m2", // dash folds + lowercased
		"  bert ":            "bert",       // surrounding whitespace trimmed
		"bert_cross_encoder": "bert_rerank",
		"phi3":               "phi",
		"unknown-arch":       "unknown_arch", // unknown passes through normalised
	}
	for in, want := range cases {
		if got := NormalizeArchitecture(in); got != want {
			t.Fatalf("NormalizeArchitecture(%q) = %q, want %q", in, got, want)
		}
	}
}

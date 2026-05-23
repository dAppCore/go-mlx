// SPDX-Licence-Identifier: EUPL-1.2

// Internal parity tests for the byte-walk compactArchitectureNameInto
// helper introduced in W11-E. The hot-path zero-alloc variant MUST
// produce bit-exact output against the heap-allocating fallback
// (which preserves the pre-W11E core.Lower + core.Replace semantics)
// for every architecture name the package ever resolves.

package profile

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
	long := ""
	for i := 0; i < maxArchitectureNameBytes+1; i++ {
		long += "x"
	}
	var buf [maxArchitectureNameBytes]byte
	got := compactArchitectureNameInto(buf[:], long)
	want := compactArchitectureNameFallback(long)
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

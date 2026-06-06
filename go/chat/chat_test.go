// SPDX-Licence-Identifier: EUPL-1.2

package chat

import "testing"

func TestFormat_PlainTemplate_Good(t *testing.T) {
	got := Format([]Message{
		{Role: "system"},
		{Role: "user", Content: "plain"},
	}, Config{Template: "plain", NoGenerationPrompt: true})
	if got != "plain\n" {
		t.Fatalf("plain format = %q, want plain only", got)
	}
}

func TestTemplateName_ArchitectureFamilies_Good(t *testing.T) {
	cases := map[string]string{
		"gemma4_text":                           "gemma4",
		"gemma4_unified":                        "gemma4",
		"Gemma4ForConditionalGeneration":        "gemma4",
		"Gemma4UnifiedForConditionalGeneration": "gemma4",
		"Gemma4ForCausalLM":                     "gemma4",
		"Gemma4TextForCausalLM":                 "gemma4",
		"gemma3":                                "gemma",
		"gemma3_text":                           "gemma",
		"Gemma3ForCausalLM":                     "gemma",
		"qwen3_moe":                             "qwen",
		"qwen3_next":                            "qwen",
		"qwen3_6":                               "qwen",
		"qwen3_6_moe":                           "qwen",
		"Qwen3ForCausalLM":                      "qwen",
		"llama3":                                "llama",
		"LlamaForCausalLM":                      "llama",
		"Gemma4AssistantForCausalLM":            "",
		"MiniMaxM2ForCausalLM":                  "",
		"DeepseekV3ForCausalLM":                 "",
		"unknown":                               "",
		"":                                      "",
	}
	for arch, want := range cases {
		if got := TemplateName(Config{Architecture: arch}); got != want {
			t.Fatalf("TemplateName(%q) = %q, want %q", arch, got, want)
		}
	}
}

func TestTemplateName_ExplicitOverridesArchitecture_Ugly(t *testing.T) {
	got := TemplateName(Config{Architecture: "gemma3", Template: "qwen"})
	if got != "qwen" {
		t.Fatalf("Template did not override Architecture: got %q", got)
	}
}

func TestNormaliseRole_Aliases_Good(t *testing.T) {
	cases := map[string]string{
		"human":     "user",
		"User":      "user",
		"gpt":       "assistant",
		"bot":       "assistant",
		"Assistant": "assistant",
		"model":     "assistant",
		"developer": "system",
		"system":    "system",
		"unknown":   "unknown",
		"":          "",
	}
	for in, want := range cases {
		if got := NormaliseRole(in); got != want {
			t.Fatalf("NormaliseRole(%q) = %q, want %q", in, got, want)
		}
	}
}

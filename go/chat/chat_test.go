// SPDX-Licence-Identifier: EUPL-1.2

package chat

import (
	"strings"
	"testing"
)

func TestFormat_GemmaTemplate_Good(t *testing.T) {
	got := Format([]Message{
		{Role: "user", Content: "hi"},
		{Role: "assistant", Content: "hello"},
	}, Config{Architecture: "gemma3"})
	if !strings.HasPrefix(got, "<bos>") {
		t.Fatalf("missing bos: %q", got)
	}
	if !strings.Contains(got, "<start_of_turn>user\nhi") {
		t.Fatalf("missing user turn: %q", got)
	}
	if !strings.Contains(got, "<start_of_turn>model\nhello") {
		t.Fatalf("missing assistant turn: %q", got)
	}
	if !strings.HasSuffix(got, "<start_of_turn>model\n") {
		t.Fatalf("missing generation prompt: %q", got)
	}
}

func TestFormat_GemmaTemplateFoldsSystemIntoFirstUser_Good(t *testing.T) {
	got := Format([]Message{
		{Role: "system", Content: " sys "},
		{Role: "user", Content: " hi "},
	}, Config{Architecture: "gemma3_text"})
	want := "<bos><start_of_turn>user\nsys\n\nhi<end_of_turn>\n<start_of_turn>model\n"
	if got != want {
		t.Fatalf("Gemma system fold = %q, want %q", got, want)
	}
}

func TestFormat_Gemma4Template_Good(t *testing.T) {
	got := Format([]Message{{Role: "user", Content: "  hi  "}}, Config{Architecture: "gemma4_text"})
	if !strings.HasPrefix(got, "<bos>") {
		t.Fatalf("missing bos: %q", got)
	}
	if !strings.Contains(got, "<|turn>user\nhi<turn|>") {
		t.Fatalf("missing trimmed user turn: %q", got)
	}
	if !strings.HasSuffix(got, "<|turn>model\n") {
		t.Fatalf("missing generation prompt: %q", got)
	}
}

func TestFormat_Gemma4TemplateThinking_Good(t *testing.T) {
	got := Format([]Message{{Role: "user", Content: "hi"}}, Config{Architecture: "gemma4_text", EnableThinking: true})
	want := "<bos><|turn>system\n<|think|>\n<turn|>\n<|turn>user\nhi<turn|>\n<|turn>model\n"
	if got != want {
		t.Fatalf("Gemma4 thinking template = %q, want %q", got, want)
	}
}

func TestFormat_Gemma4TemplateLargeVariantThinkingOff_Good(t *testing.T) {
	// 26B/31B (LargeVariant) with thinking off: the empty
	// <|channel>thought\n<channel|> ghost suppressor after the model turn,
	// per the shipped chat_template.jinja (26B/31B carry it, E2B/E4B don't).
	got := Format([]Message{{Role: "user", Content: "hi"}}, Config{Architecture: "gemma4_text", LargeVariant: true})
	want := "<bos><|turn>user\nhi<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"
	if got != want {
		t.Fatalf("Gemma4 large thinking-off = %q, want %q", got, want)
	}
}

func TestFormat_Gemma4TemplateSmallVariantThinkingOff_Good(t *testing.T) {
	// E2B/E4B (small) with thinking off: plain template, no suppressor.
	got := Format([]Message{{Role: "user", Content: "hi"}}, Config{Architecture: "gemma4_text"})
	want := "<bos><|turn>user\nhi<turn|>\n<|turn>model\n"
	if got != want {
		t.Fatalf("Gemma4 small thinking-off = %q, want %q", got, want)
	}
}

func TestFormat_Gemma4TemplateStripsAssistantThoughtHistory_Good(t *testing.T) {
	got := Format([]Message{
		{Role: "user", Content: "hi"},
		{Role: "assistant", Content: "<|channel>thought\nprivate<channel|>visible"},
	}, Config{Architecture: "gemma4_text", NoGenerationPrompt: true})
	want := "<bos><|turn>user\nhi<turn|>\n<|turn>model\nvisible<turn|>\n"
	if got != want {
		t.Fatalf("Gemma4 assistant thought strip = %q, want %q", got, want)
	}
}

func TestFormat_Gemma4TemplateContinuesAssistantRuns_Good(t *testing.T) {
	got := Format([]Message{
		{Role: "user", Content: "hi"},
		{Role: "assistant", Content: "one"},
		{Role: "assistant", Content: "two"},
	}, Config{Architecture: "gemma4_text"})
	want := "<bos><|turn>user\nhi<turn|>\n<|turn>model\none<turn|>\ntwo<turn|>\n<|turn>model\n"
	if got != want {
		t.Fatalf("Gemma4 assistant continuation = %q, want %q", got, want)
	}
}

func TestFormat_QwenTemplate_Good(t *testing.T) {
	got := Format([]Message{
		{Role: "system", Content: "be helpful"},
		{Role: "user", Content: "hi"},
	}, Config{Architecture: "qwen3"})
	if !strings.Contains(got, "<|im_start|>system\nbe helpful<|im_end|>") {
		t.Fatalf("missing system turn: %q", got)
	}
	if !strings.HasSuffix(got, "<|im_start|>assistant\n") {
		t.Fatalf("missing generation prompt: %q", got)
	}
}

func TestFormat_LlamaTemplate_Good(t *testing.T) {
	got := Format([]Message{{Role: "user", Content: "hi"}}, Config{Architecture: "llama"})
	if !strings.HasPrefix(got, "<|begin_of_text|>") {
		t.Fatalf("missing begin: %q", got)
	}
	if !strings.Contains(got, "<|start_header_id|>user<|end_header_id|>") {
		t.Fatalf("missing header: %q", got)
	}
	if !strings.HasSuffix(got, "<|start_header_id|>assistant<|end_header_id|>\n\n") {
		t.Fatalf("missing generation prompt: %q", got)
	}
}

func TestFormat_PlainTemplate_Good(t *testing.T) {
	got := Format([]Message{
		{Role: "system"},
		{Role: "user", Content: "plain"},
	}, Config{Template: "plain", NoGenerationPrompt: true})
	if got != "plain\n" {
		t.Fatalf("plain format = %q, want plain only", got)
	}
}

func TestFormat_NoGenerationPrompt_Suppresses_Good(t *testing.T) {
	got := Format([]Message{{Role: "user", Content: "hi"}}, Config{Architecture: "qwen3", NoGenerationPrompt: true})
	if strings.Contains(got, "<|im_start|>assistant") {
		t.Fatalf("NoGenerationPrompt did not suppress: %q", got)
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

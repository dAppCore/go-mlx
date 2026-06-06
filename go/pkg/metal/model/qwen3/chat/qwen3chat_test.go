// SPDX-Licence-Identifier: EUPL-1.2

package qwen3chat

import (
	"strings"
	"testing"

	"dappco.re/go/mlx/chat"
)

// These exercise the full neutral-dispatch path: chat.Format resolves the
// "qwen" template via profile and dispatches to the formatter this package
// registered in init(). They moved here from the chat package when the ChatML
// formatter left the neutral chat package (Snider's placement rule).

func TestFormat_QwenTemplate_Good(t *testing.T) {
	got := chat.Format([]chat.Message{
		{Role: "system", Content: "be helpful"},
		{Role: "user", Content: "hi"},
	}, chat.Config{Architecture: "qwen3"})
	if !strings.Contains(got, "<|im_start|>system\nbe helpful<|im_end|>") {
		t.Fatalf("missing system turn: %q", got)
	}
	if !strings.HasSuffix(got, "<|im_start|>assistant\n") {
		t.Fatalf("missing generation prompt: %q", got)
	}
}

func TestFormat_NoGenerationPrompt_Suppresses_Good(t *testing.T) {
	got := chat.Format([]chat.Message{{Role: "user", Content: "hi"}}, chat.Config{Architecture: "qwen3", NoGenerationPrompt: true})
	if strings.Contains(got, "<|im_start|>assistant") {
		t.Fatalf("NoGenerationPrompt did not suppress: %q", got)
	}
}

func TestFormat_LlamaTemplate_Good(t *testing.T) {
	got := chat.Format([]chat.Message{{Role: "user", Content: "hi"}}, chat.Config{Architecture: "llama"})
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

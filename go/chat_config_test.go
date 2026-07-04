// SPDX-Licence-Identifier: EUPL-1.2

// Tests for chat_config.go — the per-family chat templates as the root
// package wires them. These live at root (not in chat/) because the
// family formatters register from the model packages; the chat package
// alone renders the plain fallback.

package mlx

import (
	"testing"

	"dappco.re/go/inference"
	"dappco.re/go/mlx/chat"
	"dappco.re/go/mlx/pkg/metal"
)

func TestFormatChatMessages_ModelTemplates_Good(t *testing.T) {
	messages := []inference.Message{{Role: "system", Content: "sys"}, {Role: "user", Content: "hi"}}
	qwen := chat.Format(messages, chat.Config{Architecture: "qwen3"})
	if qwen != "<|im_start|>system\nsys<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n" {
		t.Fatalf("qwen template = %q", qwen)
	}
	gemma := chat.Format(messages, chat.Config{Architecture: "gemma4_text"})
	if gemma != "<bos><|turn>system\nsys<turn|>\n<|turn>user\nhi<turn|>\n<|turn>model\n" {
		t.Fatalf("gemma template = %q", gemma)
	}
	gemma3 := chat.Format(messages, chat.Config{Architecture: "gemma3_text"})
	if gemma3 != "<bos><start_of_turn>user\nsys\n\nhi<end_of_turn>\n<start_of_turn>model\n" {
		t.Fatalf("gemma3 template = %q", gemma3)
	}
	llama := chat.Format([]inference.Message{{Role: "user", Content: "hi"}}, chat.Config{Architecture: "llama"})
	if llama != "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nhi<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n" {
		t.Fatalf("llama template = %q", llama)
	}
	plain := chat.Format([]inference.Message{{Role: "system"}, {Role: "user", Content: "plain"}}, chat.Config{Template: "plain", NoGenerationPrompt: true})
	if plain != "plain\n" {
		t.Fatalf("plain template = %q, want plain line", plain)
	}
}

// formatChatTurns honours a request-level thinking override (nil = model
// default). For gemma4_text, enabling thinking injects the <|think|>
// system turn; disabling it drops the system turn entirely — so the
// override must change the rendered prefix. Good = thinking on, Bad =
// thinking off, and the two must differ to prove the override is wired.
func TestFormatChatTurns_ThinkingOverride_GoodOnInjectsThinkTurn(t *testing.T) {
	m := &Model{model: &fakeNativeModel{info: metal.ModelInfo{Architecture: "gemma4_text"}}}
	on := true

	got := m.formatChatTurns([]inference.Message{{Role: "user", Content: "hi"}}, &on, false)

	want := "<bos><|turn>system\n<|think|>\n<turn|>\n<|turn>user\nhi<turn|>\n<|turn>model\n"
	if got != want {
		t.Fatalf("thinking on = %q, want %q", got, want)
	}
}

func TestFormatChatTurns_ThinkingOverride_BadOffDropsThinkTurn(t *testing.T) {
	m := &Model{model: &fakeNativeModel{info: metal.ModelInfo{Architecture: "gemma4_text"}}}
	off := false

	got := m.formatChatTurns([]inference.Message{{Role: "user", Content: "hi"}}, &off, false)

	want := "<bos><|turn>user\nhi<turn|>\n<|turn>model\n"
	if got != want {
		t.Fatalf("thinking off = %q, want %q", got, want)
	}
}

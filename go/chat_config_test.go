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

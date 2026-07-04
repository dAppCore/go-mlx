// SPDX-Licence-Identifier: EUPL-1.2

package qwen3chat

import (
	"strings"
	"testing"

	"dappco.re/go/mlx/chat"
)

// These call the package's exported formatters directly (rather than through
// the neutral chat.Format dispatcher) so the empty-role skip branch each
// formatter carries is exercised independently of profile template resolution.
// A message whose role normalises to the empty string — an empty Role, or one
// that is whitespace-only — is dropped: chat.NormaliseRole trims it to "" and
// the formatter's `if role == "" { continue }` skips the turn entirely.

func TestFormat_EmptyRole_Dropped_Bad(t *testing.T) {
	got := Format([]chat.Message{
		{Role: "", Content: "DROPPED"},
		{Role: "user", Content: "kept"},
	}, chat.Config{})
	if strings.Contains(got, "DROPPED") {
		t.Fatalf("empty-role turn was not dropped: %q", got)
	}
	if !strings.Contains(got, "<|im_start|>user\nkept<|im_end|>") {
		t.Fatalf("following turn missing after empty-role skip: %q", got)
	}
	// The dropped turn must leave no orphan <|im_start|> marker: only the kept
	// user turn and the generation prompt open one.
	if n := strings.Count(got, "<|im_start|>"); n != 2 {
		t.Fatalf("expected 2 <|im_start|> markers (user + generation prompt), got %d: %q", n, got)
	}
}

func TestFormat_WhitespaceRole_Dropped_Bad(t *testing.T) {
	got := Format([]chat.Message{
		{Role: "   ", Content: "DROPPED"},
		{Role: "user", Content: "kept"},
	}, chat.Config{})
	if strings.Contains(got, "DROPPED") {
		t.Fatalf("whitespace-role turn was not dropped: %q", got)
	}
}

func TestFormatLlama_EmptyRole_Dropped_Bad(t *testing.T) {
	got := FormatLlama([]chat.Message{
		{Role: "", Content: "DROPPED"},
		{Role: "user", Content: "kept"},
	}, chat.Config{})
	if strings.Contains(got, "DROPPED") {
		t.Fatalf("empty-role turn was not dropped: %q", got)
	}
	if !strings.HasPrefix(got, "<|begin_of_text|>") {
		t.Fatalf("missing begin marker: %q", got)
	}
	if !strings.Contains(got, "<|start_header_id|>user<|end_header_id|>\n\nkept<|eot_id|>") {
		t.Fatalf("following turn missing after empty-role skip: %q", got)
	}
	// Only the kept user turn and the generation prompt open a header; the
	// dropped turn must not emit one.
	if n := strings.Count(got, "<|start_header_id|>"); n != 2 {
		t.Fatalf("expected 2 <|start_header_id|> markers (user + generation prompt), got %d: %q", n, got)
	}
}

func TestFormatLlama_WhitespaceRole_Dropped_Bad(t *testing.T) {
	got := FormatLlama([]chat.Message{
		{Role: "\t \n", Content: "DROPPED"},
		{Role: "user", Content: "kept"},
	}, chat.Config{})
	if strings.Contains(got, "DROPPED") {
		t.Fatalf("whitespace-role turn was not dropped: %q", got)
	}
}

// FormatLlama's generation-prompt suppression is exercised directly here; the
// existing suite only covers NoGenerationPrompt through the ChatML formatter.
func TestFormatLlama_NoGenerationPrompt_Suppresses_Good(t *testing.T) {
	got := FormatLlama([]chat.Message{{Role: "user", Content: "hi"}}, chat.Config{NoGenerationPrompt: true})
	if strings.HasSuffix(got, "<|start_header_id|>assistant<|end_header_id|>\n\n") {
		t.Fatalf("NoGenerationPrompt did not suppress the assistant cue: %q", got)
	}
	if !strings.HasSuffix(got, "kept<|eot_id|>") && !strings.HasSuffix(got, "hi<|eot_id|>") {
		t.Fatalf("expected prompt to end at the final user turn: %q", got)
	}
}

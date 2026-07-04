// SPDX-Licence-Identifier: EUPL-1.2

package gemma3chat

import (
	"strings"
	"testing"

	"dappco.re/go/mlx/chat"
)

// These exercise the full neutral-dispatch path: chat.Format resolves the
// "gemma" template via profile and dispatches to the formatter this package
// registered in init(). They moved here from the chat package when the gemma
// formatter left the neutral chat package (Snider's placement rule).

func TestFormat_GemmaTemplate_Good(t *testing.T) {
	got := chat.Format([]chat.Message{
		{Role: "user", Content: "hi"},
		{Role: "assistant", Content: "hello"},
	}, chat.Config{Architecture: "gemma3"})
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
	got := chat.Format([]chat.Message{
		{Role: "system", Content: " sys "},
		{Role: "user", Content: " hi "},
	}, chat.Config{Architecture: "gemma3_text"})
	want := "<bos><start_of_turn>user\nsys\n\nhi<end_of_turn>\n<start_of_turn>model\n"
	if got != want {
		t.Fatalf("Gemma system fold = %q, want %q", got, want)
	}
}

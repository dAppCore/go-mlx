// SPDX-Licence-Identifier: EUPL-1.2

package gemma4chat

import (
	"strings"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/mlx/chat"
)

func ExampleFormat() {
	rendered := Format(
		[]chat.Message{{Role: "user", Content: " hi "}},
		chat.Config{LargeVariant: true},
	)
	core.Println(rendered)
	// Output:
	// <bos><|turn>user
	// hi<turn|>
	// <|turn>model
	// <|channel>thought
	// <channel|>
}

// These exercise the full neutral-dispatch path: chat.Format resolves the
// "gemma4" template via profile and dispatches to the formatter this package
// registered in init(). They moved here from the chat package when the gemma4
// formatter left the neutral chat package (Snider's placement rule).

func TestFormat_Gemma4Template_Good(t *testing.T) {
	got := chat.Format([]chat.Message{{Role: "user", Content: "  hi  "}}, chat.Config{Architecture: "gemma4_text"})
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
	got := chat.Format([]chat.Message{{Role: "user", Content: "hi"}}, chat.Config{Architecture: "gemma4_text", EnableThinking: true})
	want := "<bos><|turn>system\n<|think|>\n<turn|>\n<|turn>user\nhi<turn|>\n<|turn>model\n"
	if got != want {
		t.Fatalf("Gemma4 thinking template = %q, want %q", got, want)
	}
}

func TestFormat_Gemma4TemplateContinuation_Good(t *testing.T) {
	got := chat.Format([]chat.Message{{Role: "user", Content: "and then?"}}, chat.Config{Architecture: "gemma4_text", Continuation: true})
	want := "<turn|>\n<|turn>user\nand then?<turn|>\n<|turn>model\n"
	if got != want {
		t.Fatalf("Gemma4 continuation = %q, want %q", got, want)
	}
}

func TestFormat_Gemma4TemplateContinuationSkipsOpening_Good(t *testing.T) {
	// Continuation never re-emits BOS or the system/think opening — the
	// session's retained state already holds them.
	got := chat.Format([]chat.Message{{Role: "user", Content: "next"}}, chat.Config{Architecture: "gemma4_text", EnableThinking: true, Continuation: true})
	want := "<turn|>\n<|turn>user\nnext<turn|>\n<|turn>model\n"
	if got != want {
		t.Fatalf("Gemma4 thinking continuation = %q, want %q", got, want)
	}
}

func TestFormat_Gemma4TemplateContinuationLargeVariant_Good(t *testing.T) {
	got := chat.Format([]chat.Message{{Role: "user", Content: "next"}}, chat.Config{Architecture: "gemma4_text", LargeVariant: true, Continuation: true})
	want := "<turn|>\n<|turn>user\nnext<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"
	if got != want {
		t.Fatalf("Gemma4 large-variant continuation = %q, want %q", got, want)
	}
}

func TestFormat_Gemma4TemplateLargeVariantThinkingOff_Good(t *testing.T) {
	// 26B/31B (LargeVariant) with thinking off: the empty
	// <|channel>thought\n<channel|> ghost suppressor after the model turn,
	// per the shipped chat_template.jinja (26B/31B carry it, E2B/E4B don't).
	got := chat.Format([]chat.Message{{Role: "user", Content: "hi"}}, chat.Config{Architecture: "gemma4_text", LargeVariant: true})
	want := "<bos><|turn>user\nhi<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"
	if got != want {
		t.Fatalf("Gemma4 large thinking-off = %q, want %q", got, want)
	}
}

func TestFormat_Gemma4TemplateSmallVariantThinkingOff_Good(t *testing.T) {
	// E2B/E4B (small) with thinking off: plain template, no suppressor.
	got := chat.Format([]chat.Message{{Role: "user", Content: "hi"}}, chat.Config{Architecture: "gemma4_text"})
	want := "<bos><|turn>user\nhi<turn|>\n<|turn>model\n"
	if got != want {
		t.Fatalf("Gemma4 small thinking-off = %q, want %q", got, want)
	}
}

func TestFormat_Gemma4TemplateStripsAssistantThoughtHistory_Good(t *testing.T) {
	got := chat.Format([]chat.Message{
		{Role: "user", Content: "hi"},
		{Role: "assistant", Content: "<|channel>thought\nprivate<channel|>visible"},
	}, chat.Config{Architecture: "gemma4_text", NoGenerationPrompt: true})
	want := "<bos><|turn>user\nhi<turn|>\n<|turn>model\nvisible<turn|>\n"
	if got != want {
		t.Fatalf("Gemma4 assistant thought strip = %q, want %q", got, want)
	}
}

func TestFormat_Gemma4TemplateContinuesAssistantRuns_Good(t *testing.T) {
	got := chat.Format([]chat.Message{
		{Role: "user", Content: "hi"},
		{Role: "assistant", Content: "one"},
		{Role: "assistant", Content: "two"},
	}, chat.Config{Architecture: "gemma4_text"})
	want := "<bos><|turn>user\nhi<turn|>\n<|turn>model\none<turn|>\ntwo<turn|>\n<|turn>model\n"
	if got != want {
		t.Fatalf("Gemma4 assistant continuation = %q, want %q", got, want)
	}
}

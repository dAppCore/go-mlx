// SPDX-Licence-Identifier: EUPL-1.2

package dataset

import (
	"testing"

	"dappco.re/go/inference"
	"dappco.re/go/mlx/chat"
)

func TestMessagesToSample_Gemma4SPORUsesSharedChatFormatter_Good(t *testing.T) {
	messages := []inference.Message{
		{Role: "system", Content: " be exact "},
		{Role: "user", Content: "Write one line."},
		{Role: "assistant", Content: " one line "},
	}
	cfg := chat.Config{Architecture: "gemma4_text", EnableThinking: true}

	sample, ok, err := MessagesToSample(messages, cfg, "openai_messages")
	if err != nil {
		t.Fatalf("MessagesToSample() error = %v", err)
	}
	if !ok {
		t.Fatal("MessagesToSample() ok = false, want sample")
	}

	wantPrompt := chat.Format(messages[:2], cfg)
	if sample.Prompt != wantPrompt {
		t.Fatalf("Prompt = %q, want shared chat.Format prompt %q", sample.Prompt, wantPrompt)
	}
	if sample.Response != "one line" {
		t.Fatalf("Response = %q, want trimmed assistant response", sample.Response)
	}
	if sample.Meta["format"] != "openai_messages" {
		t.Fatalf("format metadata = %q, want openai_messages", sample.Meta["format"])
	}
}

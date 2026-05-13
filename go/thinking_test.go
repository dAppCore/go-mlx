// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"
	"testing"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/parser"
	"dappco.re/go/mlx/internal/metal"
	"dappco.re/go/mlx/lora"
)

func collectThinkingStreamTokens(t *testing.T, ch <-chan Token) string {
	t.Helper()
	builder := core.NewBuilder()
	timeout := time.After(2 * time.Second)
	for {
		select {
		case tok, ok := <-ch:
			if !ok {
				return builder.String()
			}
			builder.WriteString(tok.Text)
		case <-timeout:
			t.Fatal("timed out waiting for stream")
		}
	}
}

func TestModelGenerateStream_QwenThinkingCaptureWithAdapter_Good(t *testing.T) {
	coverageTokens := "QwenThinkingCaptureWithAdapter"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	model := &Model{
		model: &fakeNativeModel{
			info: metal.ModelInfo{Architecture: "qwen3", Adapter: metal.AdapterInfo{Name: "probe-lora"}},
			tokens: []metal.Token{
				{ID: 1, Text: "Answer: "},
				{ID: 2, Text: "<thi"},
				{ID: 3, Text: "nk>hidden"},
				{ID: 4, Text: " thought</thi"},
				{ID: 5, Text: "nk>final"},
			},
		},
		adapterInfo: lora.AdapterInfo{Name: "probe-lora"},
	}
	var captured []parser.Chunk

	got := collectThinkingStreamTokens(t, model.GenerateStream(
		context.Background(),
		"ignored",
		WithCaptureThinking(func(chunk parser.Chunk) {
			captured = append(captured, chunk)
		}),
	))
	if got != "Answer: final" {
		t.Fatalf("stream text = %q, want %q", got, "Answer: final")
	}
	if len(captured) != 1 {
		t.Fatalf("captured len = %d, want 1", len(captured))
	}
	if captured[0].Text != "hidden thought" || captured[0].Model != "qwen" {
		t.Fatalf("captured = %+v", captured[0])
	}
}

func TestModelChat_GemmaThinkingHide_Good(t *testing.T) {
	coverageTokens := "GemmaThinkingHide"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	model := &Model{
		model: &fakeNativeModel{
			info: metal.ModelInfo{Architecture: "gemma4_text"},
			chatTokens: []metal.Token{
				{ID: 1, Text: "<start_of_turn>thinking\nplan"},
				{ID: 2, Text: " more<end_of_turn>"},
				{ID: 3, Text: "answer"},
			},
		},
	}

	got, err := model.Chat([]inference.Message{{Role: "user", Content: "hi"}}, WithHideThinking())
	if err != nil {
		t.Fatalf("Chat() error = %v", err)
	}
	if got != "answer" {
		t.Fatalf("Chat() = %q, want answer", got)
	}
}

func TestModelGenerate_DefaultThinkingShowPassthrough_Good(t *testing.T) {
	coverageTokens := "DefaultThinkingShowPassthrough"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	model := &Model{
		model: &fakeNativeModel{
			info:   metal.ModelInfo{Architecture: "qwen3"},
			tokens: []metal.Token{{ID: 1, Text: "<think>secret</think>visible"}},
		},
	}

	got, err := model.Generate("ignored")
	if err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	if got != "<think>secret</think>visible" {
		t.Fatalf("Generate() = %q, want passthrough", got)
	}
}

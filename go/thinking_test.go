// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"
	"testing"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/parser"
	"dappco.re/go/mlx/lora"
	"dappco.re/go/mlx/pkg/metal"
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

// applyThinkingOption runs a GenerateOption against a default config and
// returns the resulting config — the option-builders are pure, so this is
// the whole observable behaviour.
func applyThinkingOption(opt GenerateOption) GenerateConfig {
	cfg := DefaultGenerateConfig()
	opt(&cfg)
	return cfg
}

func TestThinking_WithThinkingMode_Good(t *testing.T) {
	for _, mode := range []parser.Mode{parser.Show, parser.Hide, parser.Capture} {
		cfg := applyThinkingOption(WithThinkingMode(mode))
		if cfg.Thinking.Mode != mode {
			t.Fatalf("WithThinkingMode(%q) → Mode = %q, want %q", mode, cfg.Thinking.Mode, mode)
		}
	}
}

func TestThinking_WithThinkingMode_Ugly(t *testing.T) {
	// An unknown/future mode falls through to the per-call closure that
	// still writes the requested mode verbatim.
	cfg := applyThinkingOption(WithThinkingMode(parser.Mode("future-mode")))
	if cfg.Thinking.Mode != parser.Mode("future-mode") {
		t.Fatalf("WithThinkingMode(unknown) → Mode = %q, want verbatim passthrough", cfg.Thinking.Mode)
	}
}

func TestThinking_WithShowThinking_Good(t *testing.T) {
	if cfg := applyThinkingOption(WithShowThinking()); cfg.Thinking.Mode != parser.Show {
		t.Fatalf("WithShowThinking() → Mode = %q, want show", cfg.Thinking.Mode)
	}
	if cfg := applyThinkingOption(WithHideThinking()); cfg.Thinking.Mode != parser.Hide {
		t.Fatalf("WithHideThinking() → Mode = %q, want hide", cfg.Thinking.Mode)
	}
}

func TestThinking_WithThinkingCapture_Good(t *testing.T) {
	// WithThinkingCapture is the alias for WithCaptureThinking: both set
	// Capture mode AND wire the callback.
	var got parser.Chunk
	cfg := applyThinkingOption(WithThinkingCapture(func(c parser.Chunk) { got = c }))
	if cfg.Thinking.Mode != parser.Capture {
		t.Fatalf("WithThinkingCapture → Mode = %q, want capture", cfg.Thinking.Mode)
	}
	if cfg.Thinking.Capture == nil {
		t.Fatal("WithThinkingCapture → Capture callback is nil, want wired")
	}
	cfg.Thinking.Capture(parser.Chunk{Text: "thought"})
	if got.Text != "thought" {
		t.Fatalf("wired callback got %+v, want Text=thought", got)
	}
}

func TestThinking_FilterThinkingTokens_Good(t *testing.T) {
	// Real synthetic BPE tokenizer: "hello" → id 10. Show mode is a
	// passthrough, so the decoded text comes straight back.
	tok, err := LoadTokenizer(writeRootTokenizer(t))
	if err != nil {
		t.Fatalf("LoadTokenizer: %v", err)
	}
	out, err := FilterThinkingTokens(tok, []int32{10}, parser.Config{Mode: parser.Show}, ModelInfo{Architecture: "qwen3"})
	if err != nil {
		t.Fatalf("FilterThinkingTokens error = %v", err)
	}
	if out.Text != "hello" {
		t.Fatalf("FilterThinkingTokens text = %q, want hello", out.Text)
	}
}

func TestThinking_FilterThinkingTokens_Bad(t *testing.T) {
	// A nil/zero-value Tokenizer is the documented precondition failure.
	_, err := FilterThinkingTokens((*Tokenizer)(nil), []int32{1}, parser.Config{Mode: parser.Show}, ModelInfo{})
	if err == nil {
		t.Fatal("FilterThinkingTokens(nil tokenizer) error = nil, want precondition error")
	}
}

func TestThinking_FilterThinkingTokens_Ugly(t *testing.T) {
	// Empty id slice is benign: a valid tokenizer yields empty text, no panic.
	tok, err := LoadTokenizer(writeRootTokenizer(t))
	if err != nil {
		t.Fatalf("LoadTokenizer: %v", err)
	}
	out, err := FilterThinkingTokens(tok, nil, parser.Config{Mode: parser.Capture}, ModelInfo{Architecture: "qwen3"})
	if err != nil {
		t.Fatalf("FilterThinkingTokens(empty ids) error = %v", err)
	}
	if out.Text != "" {
		t.Fatalf("FilterThinkingTokens(empty ids) text = %q, want empty", out.Text)
	}
}

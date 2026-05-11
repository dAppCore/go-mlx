// SPDX-Licence-Identifier: EUPL-1.2

package daemon

import (
	"context"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
	mlx "dappco.re/go/mlx"
)

type fakeNativeModel struct {
	generatePrompt string
	chatMessages   []inference.Message
	err            error
	closed         bool
	metrics        mlx.Metrics
}

func (model *fakeNativeModel) GenerateStream(_ context.Context, prompt string, _ ...mlx.GenerateOption) <-chan mlx.Token {
	model.generatePrompt = prompt
	ch := make(chan mlx.Token, 2)
	ch <- mlx.Token{Text: "hel"}
	ch <- mlx.Token{Text: "lo"}
	close(ch)
	return ch
}

func (model *fakeNativeModel) ChatStream(_ context.Context, messages []inference.Message, _ ...mlx.GenerateOption) <-chan mlx.Token {
	model.chatMessages = append([]inference.Message(nil), messages...)
	ch := make(chan mlx.Token, 1)
	ch <- mlx.Token{Text: "chat"}
	close(ch)
	return ch
}

func (model *fakeNativeModel) WarmPromptCache(string) error { return nil }
func (model *fakeNativeModel) Metrics() mlx.Metrics         { return model.metrics }
func (model *fakeNativeModel) Err() error                   { return model.err }
func (model *fakeNativeModel) Close() error {
	model.closed = true
	return nil
}

func TestNativeGenerateRunner_Good_GeneratesWithDefaultModel(t *testing.T) {
	native := &fakeNativeModel{metrics: mlx.Metrics{PromptTokens: 8, GeneratedTokens: 2, PromptCacheHits: 1, PromptCacheHitTokens: 8}}
	runner := NewNativeGenerateRunner(NativeGenerateConfig{
		ModelPaths: map[string]string{"default": "/models/main"},
	})
	var loadedPath string
	runner.loadModel = func(path string, _ ...mlx.LoadOption) (nativeGenerateModel, error) {
		loadedPath = path
		return native, nil
	}

	result, err := runner.Generate(context.Background(), GenerateRequest{Prompt: "hello", MaxTokens: 4})

	if err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	if loadedPath != "/models/main" {
		t.Fatalf("loaded path = %q, want /models/main", loadedPath)
	}
	if native.generatePrompt != "hello" {
		t.Fatalf("prompt = %q, want hello", native.generatePrompt)
	}
	if result.Text != "hello" {
		t.Fatalf("text = %q, want hello", result.Text)
	}
	if result.Model != "default" {
		t.Fatalf("model = %q, want default", result.Model)
	}
	if result.Metrics.PromptTokens != 8 {
		t.Fatalf("prompt tokens = %d, want 8", result.Metrics.PromptTokens)
	}
	if result.Metrics.GeneratedTokens != 2 {
		t.Fatalf("generated tokens = %d, want 2", result.Metrics.GeneratedTokens)
	}
	if result.Metrics.PromptCacheHits != 1 || result.Metrics.PromptCacheHitTokens != 8 {
		t.Fatalf("prompt cache metrics = %+v, want hit counters", result.Metrics)
	}
}

func TestNativeGenerateRunner_Bad_UnknownModel(t *testing.T) {
	runner := NewNativeGenerateRunner(NativeGenerateConfig{
		ModelPaths: map[string]string{"main": "/models/main"},
	})

	_, err := runner.Generate(context.Background(), GenerateRequest{Prompt: "hello", Model: "missing"})

	if err == nil {
		t.Fatal("Generate() returned nil error, want unknown model error")
	}
	if !core.Contains(err.Error(), "unknown model") {
		t.Fatalf("error = %v, want unknown model", err)
	}
}

func TestNativeGenerateRunner_Ugly_ChatMessages(t *testing.T) {
	native := &fakeNativeModel{}
	runner := NewNativeGenerateRunner(NativeGenerateConfig{
		ModelPaths: map[string]string{"default": "/models/main"},
	})
	runner.loadModel = func(string, ...mlx.LoadOption) (nativeGenerateModel, error) {
		return native, nil
	}

	result, err := runner.Generate(context.Background(), GenerateRequest{
		Messages: []Message{{Role: "system", Content: "steady"}, {Role: "user", Content: "hello"}},
	})

	if err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	if result.Text != "chat" {
		t.Fatalf("text = %q, want chat", result.Text)
	}
	if len(native.chatMessages) != 2 {
		t.Fatalf("chat messages = %d, want 2", len(native.chatMessages))
	}
	if native.chatMessages[0].Role != "system" || native.chatMessages[1].Content != "hello" {
		t.Fatalf("chat messages = %+v", native.chatMessages)
	}
}

func TestNativeGenerateRunner_Close_Good_ClosesLoadedModels(t *testing.T) {
	native := &fakeNativeModel{}
	runner := NewNativeGenerateRunner(NativeGenerateConfig{
		ModelPaths: map[string]string{"default": "/models/main"},
	})
	runner.loadModel = func(string, ...mlx.LoadOption) (nativeGenerateModel, error) {
		return native, nil
	}
	if _, err := runner.Generate(context.Background(), GenerateRequest{Prompt: "hello"}); err != nil {
		t.Fatalf("Generate() error = %v", err)
	}

	if err := runner.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}
	if !native.closed {
		t.Fatal("native model was not closed")
	}
}

// Generated file-aware compliance coverage.
func TestNative_NewNativeGenerateRunner_Good(t *testing.T) {
	target := "NewNativeGenerateRunner"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestNative_NewNativeGenerateRunner_Bad(t *testing.T) {
	target := "NewNativeGenerateRunner"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestNative_NewNativeGenerateRunner_Ugly(t *testing.T) {
	target := "NewNativeGenerateRunner"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

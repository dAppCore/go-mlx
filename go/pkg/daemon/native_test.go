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

func TestNativeGenerateRunner_Close_Ugly_NilRunnerAndNoModels(t *testing.T) {
	var nilRunner *NativeGenerateRunner
	if err := nilRunner.Close(); err != nil {
		t.Fatalf("nil runner Close() = %v, want nil", err)
	}

	// A runner that never loaded a model closes cleanly (empty cache).
	runner := NewNativeGenerateRunner(NativeGenerateConfig{
		ModelPaths: map[string]string{"default": "/models/main"},
	})
	if err := runner.Close(); err != nil {
		t.Fatalf("Close() with no loaded models = %v, want nil", err)
	}
}

func TestNativeGenerateRunner_Generate_Bad_NilRunnerAndContext(t *testing.T) {
	var nilRunner *NativeGenerateRunner
	if _, err := nilRunner.Generate(context.Background(), GenerateRequest{Prompt: "x"}); err == nil {
		t.Fatal("nil runner Generate() returned nil, want error")
	} else if !core.Contains(err.Error(), "runner is nil") {
		t.Fatalf("error = %v, want runner is nil", err)
	}

	// A nil context must be tolerated (defaulted to Background) rather than panic.
	native := &fakeNativeModel{}
	runner := NewNativeGenerateRunner(NativeGenerateConfig{
		ModelPaths: map[string]string{"default": "/models/main"},
	})
	runner.loadModel = func(string, ...mlx.LoadOption) (nativeGenerateModel, error) { return native, nil }
	//nolint:staticcheck // SA1012: deliberately passing nil ctx to exercise the defaulting branch.
	if _, err := runner.Generate(nil, GenerateRequest{Prompt: "hello"}); err != nil {
		t.Fatalf("Generate(nil ctx) error = %v, want nil", err)
	}
}

func TestNativeGenerateRunner_Generate_Bad_EmptyPrompt(t *testing.T) {
	runner := NewNativeGenerateRunner(NativeGenerateConfig{
		ModelPaths: map[string]string{"default": "/models/main"},
	})
	runner.loadModel = func(string, ...mlx.LoadOption) (nativeGenerateModel, error) {
		return &fakeNativeModel{}, nil
	}

	_, err := runner.Generate(context.Background(), GenerateRequest{Prompt: "   "})
	if err == nil {
		t.Fatal("Generate(blank prompt) returned nil, want error")
	}
	if !core.Contains(err.Error(), "prompt is required") {
		t.Fatalf("error = %v, want prompt is required", err)
	}
}

func TestNativeGenerateRunner_Generate_Ugly_BackendErrorMidStream(t *testing.T) {
	wantErr := core.NewError("decode blew up")
	native := &fakeNativeModel{err: wantErr}
	runner := NewNativeGenerateRunner(NativeGenerateConfig{
		ModelPaths: map[string]string{"default": "/models/main"},
	})
	runner.loadModel = func(string, ...mlx.LoadOption) (nativeGenerateModel, error) { return native, nil }

	result, err := runner.Generate(context.Background(), GenerateRequest{Prompt: "hello"})
	if !core.Is(err, wantErr) {
		t.Fatalf("Generate() error = %v, want %v", err, wantErr)
	}
	// Partial text accumulated before the error must still be surfaced with the model name.
	if result.Text != "hello" {
		t.Fatalf("partial text = %q, want hello", result.Text)
	}
	if result.Model != "default" {
		t.Fatalf("model = %q, want default on error path", result.Model)
	}
}

func TestNativeGenerateRunner_Generate_Bad_LoadFailure(t *testing.T) {
	runner := NewNativeGenerateRunner(NativeGenerateConfig{
		ModelPaths: map[string]string{"default": "/models/main"},
	})
	runner.loadModel = func(string, ...mlx.LoadOption) (nativeGenerateModel, error) {
		return nil, core.NewError("weights not found")
	}

	_, err := runner.Generate(context.Background(), GenerateRequest{Prompt: "hello"})
	if err == nil {
		t.Fatal("Generate() with failing load returned nil, want error")
	}
	if !core.Contains(err.Error(), "load native model") {
		t.Fatalf("error = %v, want load native model wrap", err)
	}
}

func TestNativeGenerateRunner_ResolveModel_Good_FallbackChains(t *testing.T) {
	native := &fakeNativeModel{}
	load := func(string, ...mlx.LoadOption) (nativeGenerateModel, error) { return native, nil }

	// "generate" key fallback: default name is unset/missing but a "generate" path exists.
	generateKey := NewNativeGenerateRunner(NativeGenerateConfig{
		ModelPaths:       map[string]string{"generate": "/models/gen"},
		DefaultModelName: "absent",
	})
	generateKey.loadModel = load
	res, err := generateKey.Generate(context.Background(), GenerateRequest{Prompt: "hi"})
	if err != nil {
		t.Fatalf("Generate(generate-key fallback) error = %v", err)
	}
	if res.Model != "generate" {
		t.Fatalf("model = %q, want generate (generate-key fallback)", res.Model)
	}

	// Single-model fallback: one configured model under an arbitrary name, no default match.
	single := NewNativeGenerateRunner(NativeGenerateConfig{
		ModelPaths:       map[string]string{"solo": "/models/solo"},
		DefaultModelName: "absent",
	})
	single.loadModel = load
	res, err = single.Generate(context.Background(), GenerateRequest{Prompt: "hi"})
	if err != nil {
		t.Fatalf("Generate(single-model fallback) error = %v", err)
	}
	if res.Model != "solo" {
		t.Fatalf("model = %q, want solo (single-model fallback)", res.Model)
	}
}

func TestNativeGenerateRunner_ResolveModel_Bad_NoModelsAndAmbiguous(t *testing.T) {
	// No models configured at all.
	none := NewNativeGenerateRunner(NativeGenerateConfig{})
	if _, err := none.Generate(context.Background(), GenerateRequest{Prompt: "hi"}); err == nil {
		t.Fatal("Generate() with no models returned nil, want error")
	} else if !core.Contains(err.Error(), "no native models configured") {
		t.Fatalf("error = %v, want no native models configured", err)
	}

	// Ambiguous: multiple models, no default match, no "generate" key — cannot pick one.
	ambiguous := NewNativeGenerateRunner(NativeGenerateConfig{
		ModelPaths:       map[string]string{"a": "/models/a", "b": "/models/b"},
		DefaultModelName: "absent",
	})
	if _, err := ambiguous.Generate(context.Background(), GenerateRequest{Prompt: "hi"}); err == nil {
		t.Fatal("Generate() with ambiguous models returned nil, want error")
	} else if !core.Contains(err.Error(), "generate model is required") {
		t.Fatalf("error = %v, want generate model is required", err)
	}
}

func TestNativeGenerateRunner_Generate_Good_OptionBranches(t *testing.T) {
	// Temperature-only request: exercises the WithTemperature branch and the
	// MaxTokens-from-default fill inside generateOptions via the public path.
	native := &fakeNativeModel{}
	runner := NewNativeGenerateRunner(NativeGenerateConfig{
		ModelPaths:       map[string]string{"default": "/models/main"},
		DefaultMaxTokens: 128,
	})
	runner.loadModel = func(string, ...mlx.LoadOption) (nativeGenerateModel, error) { return native, nil }

	res, err := runner.Generate(context.Background(), GenerateRequest{Prompt: "hello", Temperature: 0.5})
	if err != nil {
		t.Fatalf("Generate(temperature-only) error = %v", err)
	}
	if res.Text != "hello" {
		t.Fatalf("text = %q, want hello", res.Text)
	}
}

// --- v0.9.0 canonical AX-7 triplets (file prefix: Native) -------------------
// Test<Native>_<Symbol>_{Good,Bad,Ugly} for each public symbol. The
// scenario-suffixed tests above stay as the fine-grained coverage; these are
// the canonical-shape entry points and each names its symbol directly.

func TestNative_NewNativeGenerateRunner_Good(t *testing.T) {
	runner := NewNativeGenerateRunner(NativeGenerateConfig{
		ModelPaths:       map[string]string{"default": "/models/main"},
		DefaultModelName: "default",
	})
	if runner == nil {
		t.Fatal("NewNativeGenerateRunner() = nil, want runner")
	}
	if runner.defaultModel != "default" {
		t.Fatalf("defaultModel = %q, want default", runner.defaultModel)
	}
}

func TestNative_NewNativeGenerateRunner_Bad(t *testing.T) {
	// Blank DefaultModelName is not an error — NewNativeGenerateRunner must
	// fall back to the package default model name rather than leave it empty.
	runner := NewNativeGenerateRunner(NativeGenerateConfig{DefaultModelName: "   "})
	if runner.defaultModel != defaultNativeModelName {
		t.Fatalf("defaultModel = %q, want fallback %q for blank input", runner.defaultModel, defaultNativeModelName)
	}
}

func TestNative_NewNativeGenerateRunner_Ugly(t *testing.T) {
	// The runner must copy the input ModelPaths map — mutating the caller's map
	// after construction cannot change which path a model resolves to.
	paths := map[string]string{"default": "/models/orig"}
	runner := NewNativeGenerateRunner(NativeGenerateConfig{ModelPaths: paths})
	paths["default"] = "/models/swapped"

	var loadedPath string
	runner.loadModel = func(path string, _ ...mlx.LoadOption) (nativeGenerateModel, error) {
		loadedPath = path
		return &fakeNativeModel{}, nil
	}
	if _, err := runner.Generate(context.Background(), GenerateRequest{Prompt: "hi"}); err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	if loadedPath != "/models/orig" {
		t.Fatalf("loaded path = %q, want /models/orig (NewNativeGenerateRunner must clone ModelPaths)", loadedPath)
	}
}

func TestNative_NativeGenerateRunner_Generate_Good(t *testing.T) {
	native := &fakeNativeModel{metrics: mlx.Metrics{PromptTokens: 8, GeneratedTokens: 2}}
	runner := NewNativeGenerateRunner(NativeGenerateConfig{
		ModelPaths: map[string]string{"default": "/models/main"},
	})
	runner.loadModel = func(string, ...mlx.LoadOption) (nativeGenerateModel, error) { return native, nil }

	result, err := runner.Generate(context.Background(), GenerateRequest{Prompt: "hello", MaxTokens: 4})
	if err != nil {
		t.Fatalf("Generate() error = %v", err)
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
}

func TestNative_NativeGenerateRunner_Generate_Bad(t *testing.T) {
	// Nil receiver: Generate must return the runner-is-nil error, not panic.
	var nilRunner *NativeGenerateRunner
	if _, err := nilRunner.Generate(context.Background(), GenerateRequest{Prompt: "x"}); err == nil {
		t.Fatal("nil runner Generate() returned nil, want error")
	} else if !core.Contains(err.Error(), "runner is nil") {
		t.Fatalf("error = %v, want runner is nil", err)
	}
}

func TestNative_NativeGenerateRunner_Generate_Ugly(t *testing.T) {
	// Mid-stream backend error: Generate must surface the error AND the partial
	// text accumulated before the failure, tagged with the resolved model name.
	wantErr := core.NewError("decode blew up")
	native := &fakeNativeModel{err: wantErr}
	runner := NewNativeGenerateRunner(NativeGenerateConfig{
		ModelPaths: map[string]string{"default": "/models/main"},
	})
	runner.loadModel = func(string, ...mlx.LoadOption) (nativeGenerateModel, error) { return native, nil }

	result, err := runner.Generate(context.Background(), GenerateRequest{Prompt: "hello"})
	if !core.Is(err, wantErr) {
		t.Fatalf("Generate() error = %v, want %v", err, wantErr)
	}
	if result.Text != "hello" {
		t.Fatalf("partial text = %q, want hello", result.Text)
	}
	if result.Model != "default" {
		t.Fatalf("model = %q, want default on error path", result.Model)
	}
}

func TestNative_NativeGenerateRunner_Close_Good(t *testing.T) {
	native := &fakeNativeModel{}
	runner := NewNativeGenerateRunner(NativeGenerateConfig{
		ModelPaths: map[string]string{"default": "/models/main"},
	})
	runner.loadModel = func(string, ...mlx.LoadOption) (nativeGenerateModel, error) { return native, nil }
	if _, err := runner.Generate(context.Background(), GenerateRequest{Prompt: "hello"}); err != nil {
		t.Fatalf("Generate() error = %v", err)
	}

	if err := runner.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}
	if !native.closed {
		t.Fatal("Close() did not close the loaded native model")
	}
}

func TestNative_NativeGenerateRunner_Close_Bad(t *testing.T) {
	// Nil receiver: Close must be a safe no-op returning nil, not a panic.
	var nilRunner *NativeGenerateRunner
	if err := nilRunner.Close(); err != nil {
		t.Fatalf("nil runner Close() = %v, want nil", err)
	}
}

func TestNative_NativeGenerateRunner_Close_Ugly(t *testing.T) {
	// Close on a runner that never loaded a model (empty cache) must still
	// succeed, and a second Close must remain idempotent.
	runner := NewNativeGenerateRunner(NativeGenerateConfig{
		ModelPaths: map[string]string{"default": "/models/main"},
	})
	if err := runner.Close(); err != nil {
		t.Fatalf("Close() with empty cache = %v, want nil", err)
	}
	if err := runner.Close(); err != nil {
		t.Fatalf("second Close() = %v, want nil (idempotent)", err)
	}
}

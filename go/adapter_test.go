// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"
	"iter"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/mlx/adapter"
)

type stubTextModel struct {
	tokens     []inference.Token
	chatTokens []inference.Token
	err        error
	metrics    inference.GenerateMetrics
	attention  *inference.AttentionSnapshot
	closeErr   error
}

func (model *stubTextModel) Generate(_ context.Context, _ string, _ ...inference.GenerateOption) iter.Seq[inference.Token] {
	return func(yield func(inference.Token) bool) {
		for _, token := range model.tokens {
			if !yield(token) {
				return
			}
		}
	}
}

func (model *stubTextModel) Chat(_ context.Context, _ []inference.Message, _ ...inference.GenerateOption) iter.Seq[inference.Token] {
	return func(yield func(inference.Token) bool) {
		for _, token := range model.chatTokens {
			if !yield(token) {
				return
			}
		}
	}
}

func (model *stubTextModel) Classify(context.Context, []string, ...inference.GenerateOption) ([]inference.ClassifyResult, error) {
	return nil, nil
}

func (model *stubTextModel) BatchGenerate(context.Context, []string, ...inference.GenerateOption) ([]inference.BatchResult, error) {
	return nil, nil
}

func (model *stubTextModel) ModelType() string                  { return "stub" }
func (model *stubTextModel) Info() inference.ModelInfo          { return inference.ModelInfo{} }
func (model *stubTextModel) Metrics() inference.GenerateMetrics { return model.metrics }
func (model *stubTextModel) Err() error                         { return model.err }
func (model *stubTextModel) Close() error                       { return model.closeErr }
func (model *stubTextModel) InspectAttention(context.Context, string, ...inference.GenerateOption) (*inference.AttentionSnapshot, error) {
	return model.attention, nil
}

type plainTextModel struct{}

func (model *plainTextModel) Generate(_ context.Context, _ string, _ ...inference.GenerateOption) iter.Seq[inference.Token] {
	return func(yield func(inference.Token) bool) {}
}
func (model *plainTextModel) Chat(_ context.Context, _ []inference.Message, _ ...inference.GenerateOption) iter.Seq[inference.Token] {
	return func(yield func(inference.Token) bool) {}
}
func (model *plainTextModel) Classify(context.Context, []string, ...inference.GenerateOption) ([]inference.ClassifyResult, error) {
	return nil, nil
}
func (model *plainTextModel) BatchGenerate(context.Context, []string, ...inference.GenerateOption) ([]inference.BatchResult, error) {
	return nil, nil
}
func (model *plainTextModel) ModelType() string                  { return "plain" }
func (model *plainTextModel) Info() inference.ModelInfo          { return inference.ModelInfo{} }
func (model *plainTextModel) Metrics() inference.GenerateMetrics { return inference.GenerateMetrics{} }
func (model *plainTextModel) Err() error                         { return nil }
func (model *plainTextModel) Close() error                       { return nil }

type stubBackend struct {
	model    inference.TextModel
	loadPath string
	loadErr  error
}

func (backend *stubBackend) Name() string { return "metal" }
func (backend *stubBackend) Available() bool {
	return true
}
func (backend *stubBackend) LoadModel(path string, _ ...inference.LoadOption) (inference.TextModel, error) {
	backend.loadPath = path
	if backend.loadErr != nil {
		return nil, backend.loadErr
	}
	return backend.model, nil
}

func TestNewInferenceAdapterGenerate_Good(t *testing.T) {
	model := &stubTextModel{
		tokens: []inference.Token{{Text: "Hello"}, {Text: " world"}},
		metrics: inference.GenerateMetrics{
			GeneratedTokens: 2,
		},
	}

	a := adapter.New(model, "mlx")
	result, err := a.Generate(context.Background(), "ignored", adapter.GenOpts{MaxTokens: 16, Temp: 0.2})
	if err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	if result.Text != "Hello world" {
		t.Fatalf("Generate().Text = %q, want %q", result.Text, "Hello world")
	}
	if result.Metrics == nil || result.Metrics.GeneratedTokens != 2 {
		t.Fatalf("Generate().Metrics = %+v, want generated tokens = 2", result.Metrics)
	}
}

func TestInferenceAdapterChat_Good(t *testing.T) {
	model := &stubTextModel{
		chatTokens: []inference.Token{{Text: "chat"}, {Text: " reply"}},
	}

	a := adapter.New(model, "mlx")
	result, err := a.Chat(context.Background(), []inference.Message{{Role: "user", Content: "hi"}}, adapter.GenOpts{MaxTokens: 8})
	if err != nil {
		t.Fatalf("Chat() error = %v", err)
	}
	if result.Text != "chat reply" {
		t.Fatalf("Chat().Text = %q, want %q", result.Text, "chat reply")
	}
}

func TestInferenceAdapterGenerateStream_CallbackError_Bad(t *testing.T) {
	wantErr := core.NewError("stop")
	model := &stubTextModel{
		tokens: []inference.Token{{Text: "one"}, {Text: "two"}},
	}

	a := adapter.New(model, "mlx")
	err := a.GenerateStream(context.Background(), "ignored", adapter.GenOpts{}, func(token string) error {
		if token == "one" {
			return wantErr
		}
		return nil
	})
	if !core.Is(err, wantErr) {
		t.Fatalf("GenerateStream() error = %v, want %v", err, wantErr)
	}
}

func TestInferenceAdapterBasics_Good(t *testing.T) {
	model := &stubTextModel{closeErr: core.NewError("close failed")}
	a := adapter.New(model, "probe")
	if a.Name() != "probe" {
		t.Fatalf("Name() = %q, want probe", a.Name())
	}
	if !a.Available() {
		t.Fatal("Available() = false, want true")
	}
	if a.Model() != model {
		t.Fatal("Model() did not return wrapped model")
	}
	if err := a.Close(); err == nil || !core.Contains(err.Error(), "close failed") {
		t.Fatalf("Close() error = %v", err)
	}
	if a.Available() {
		t.Fatal("Available() after Close = true, want false")
	}
	if err := a.Close(); err != nil {
		t.Fatalf("second Close() = %v, want nil", err)
	}

	var nilAdapter *adapter.Adapter
	if nilAdapter.Name() != "" {
		t.Fatal("nil Name() should be blank")
	}
	if nilAdapter.Available() {
		t.Fatal("nil Available() should be false")
	}
	if nilAdapter.Model() != nil {
		t.Fatal("nil Model() should be nil")
	}
}

func TestInferenceAdapterNilAndModelErrors_Bad(t *testing.T) {
	var nilAdapter *adapter.Adapter
	if _, err := nilAdapter.Generate(context.Background(), "x", adapter.GenOpts{}); err == nil {
		t.Fatal("expected nil Generate error")
	}
	if err := nilAdapter.GenerateStream(context.Background(), "x", adapter.GenOpts{}, func(string) error { return nil }); err == nil {
		t.Fatal("expected nil GenerateStream error")
	}
	if _, err := nilAdapter.Chat(context.Background(), nil, adapter.GenOpts{}); err == nil {
		t.Fatal("expected nil Chat error")
	}
	if err := nilAdapter.ChatStream(context.Background(), nil, adapter.GenOpts{}, func(string) error { return nil }); err == nil {
		t.Fatal("expected nil ChatStream error")
	}
	if _, err := nilAdapter.InspectAttention(context.Background(), "x"); err == nil {
		t.Fatal("expected nil InspectAttention error")
	}

	a := adapter.New(&stubTextModel{}, "probe")
	if err := a.GenerateStream(context.Background(), "x", adapter.GenOpts{}, nil); err == nil {
		t.Fatal("expected nil generate callback error")
	}
	if err := a.ChatStream(context.Background(), nil, adapter.GenOpts{}, nil); err == nil {
		t.Fatal("expected nil chat callback error")
	}

	want := core.NewError("model failed")
	errorModel := &stubTextModel{
		tokens:     []inference.Token{{Text: "partial"}},
		chatTokens: []inference.Token{{Text: "chat"}},
		err:        want,
	}
	a = adapter.New(errorModel, "probe")
	result, err := a.Generate(nil, "x", adapter.GenOpts{})
	if !core.Is(err, want) || result.Text != "partial" {
		t.Fatalf("Generate() = result:%+v err:%v, want partial model error", result, err)
	}
	result, err = a.Chat(nil, nil, adapter.GenOpts{})
	if !core.Is(err, want) || result.Text != "chat" {
		t.Fatalf("Chat() = result:%+v err:%v, want chat model error", result, err)
	}
}

func TestInferenceAdapterChatStream_CallbackError_Bad(t *testing.T) {
	wantErr := core.NewError("stop chat")
	model := &stubTextModel{
		chatTokens: []inference.Token{{Text: "one"}, {Text: "two"}},
	}

	a := adapter.New(model, "mlx")
	err := a.ChatStream(context.Background(), []inference.Message{{Role: "user", Content: "hi"}}, adapter.GenOpts{}, func(token string) error {
		if token == "one" {
			return wantErr
		}
		return nil
	})
	if !core.Is(err, wantErr) {
		t.Fatalf("ChatStream() error = %v, want %v", err, wantErr)
	}
}

func TestInferenceAdapterInspectAttention_Good(t *testing.T) {
	want := &inference.AttentionSnapshot{NumLayers: 2, Architecture: "gemma3"}
	model := &stubTextModel{attention: want}

	a := adapter.New(model, "mlx")
	got, err := a.InspectAttention(context.Background(), "prompt")
	if err != nil {
		t.Fatalf("InspectAttention() error = %v", err)
	}
	if got != want {
		t.Fatalf("InspectAttention() = %+v, want %+v", got, want)
	}
}

func TestInferenceAdapterInspectAttention_Unsupported_Bad(t *testing.T) {
	model := &plainTextModel{}
	a := adapter.New(model, "plain")
	if _, err := a.InspectAttention(context.Background(), "prompt"); err == nil {
		t.Fatal("expected unsupported attention inspection error")
	}
}

func TestNewMLXBackend_Good(t *testing.T) {
	oldBackend, hadOldBackend := inference.Get("metal")
	if hadOldBackend {
		defer inference.Register(oldBackend)
	}

	model := &stubTextModel{}
	backend := &stubBackend{model: model}
	inference.Register(backend)

	a, err := NewMLXBackend("/tmp/model-path", inference.WithContextLen(4096))
	if err != nil {
		t.Fatalf("NewMLXBackend() error = %v", err)
	}
	if a.Name() != "mlx" {
		t.Fatalf("adapter name = %q, want %q", a.Name(), "mlx")
	}
	if a.Model() != model {
		t.Fatal("adapter should expose the loaded model")
	}
	if backend.loadPath != "/tmp/model-path" {
		t.Fatalf("backend load path = %q, want %q", backend.loadPath, "/tmp/model-path")
	}
}

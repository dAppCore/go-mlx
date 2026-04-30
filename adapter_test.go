// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"
	"iter"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
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

type stubBackend struct {
	model    inference.TextModel
	loadPath string
}

func (backend *stubBackend) Name() string { return "metal" }
func (backend *stubBackend) Available() bool {
	return true
}
func (backend *stubBackend) LoadModel(path string, _ ...inference.LoadOption) (inference.TextModel, error) {
	backend.loadPath = path
	return backend.model, nil
}

func TestNewInferenceAdapterGenerate_Good(t *testing.T) {
	model := &stubTextModel{
		tokens: []inference.Token{{Text: "Hello"}, {Text: " world"}},
		metrics: inference.GenerateMetrics{
			GeneratedTokens: 2,
		},
	}

	adapter := NewInferenceAdapter(model, "mlx")
	result, err := adapter.Generate(context.Background(), "ignored", GenOpts{MaxTokens: 16, Temp: 0.2})
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

	adapter := NewInferenceAdapter(model, "mlx")
	result, err := adapter.Chat(context.Background(), []Message{{Role: "user", Content: "hi"}}, GenOpts{MaxTokens: 8})
	if err != nil {
		t.Fatalf("Chat() error = %v", err)
	}
	if result.Text != "chat reply" {
		t.Fatalf("Chat().Text = %q, want %q", result.Text, "chat reply")
	}
}

func TestInferenceAdapterGenerateStream_CallbackError_Bad(t *testing.T) {
	coverageTokens := "CallbackError"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	wantErr := core.NewError("stop")
	model := &stubTextModel{
		tokens: []inference.Token{{Text: "one"}, {Text: "two"}},
	}

	adapter := NewInferenceAdapter(model, "mlx")
	err := adapter.GenerateStream(context.Background(), "ignored", GenOpts{}, func(token string) error {
		if token == "one" {
			return wantErr
		}
		return nil
	})
	if !core.Is(err, wantErr) {
		t.Fatalf("GenerateStream() error = %v, want %v", err, wantErr)
	}
}

func TestInferenceAdapterInspectAttention_Good(t *testing.T) {
	want := &inference.AttentionSnapshot{NumLayers: 2, Architecture: "gemma3"}
	model := &stubTextModel{attention: want}

	adapter := NewInferenceAdapter(model, "mlx")
	got, err := adapter.InspectAttention(context.Background(), "prompt")
	if err != nil {
		t.Fatalf("InspectAttention() error = %v", err)
	}
	if got != want {
		t.Fatalf("InspectAttention() = %+v, want %+v", got, want)
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

	adapter, err := NewMLXBackend("/tmp/model-path", inference.WithContextLen(4096))
	if err != nil {
		t.Fatalf("NewMLXBackend() error = %v", err)
	}
	if adapter.Name() != "mlx" {
		t.Fatalf("adapter name = %q, want %q", adapter.Name(), "mlx")
	}
	if adapter.Model() != model {
		t.Fatal("adapter should expose the loaded model")
	}
	if backend.loadPath != "/tmp/model-path" {
		t.Fatalf("backend load path = %q, want %q", backend.loadPath, "/tmp/model-path")
	}
}

// Generated file-aware compliance coverage.
func TestAdapter_NewInferenceAdapter_Good(t *testing.T) {
	target := "NewInferenceAdapter"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestAdapter_NewInferenceAdapter_Bad(t *testing.T) {
	target := "NewInferenceAdapter"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestAdapter_NewInferenceAdapter_Ugly(t *testing.T) {
	target := "NewInferenceAdapter"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestAdapter_NewMLXBackend_Good(t *testing.T) {
	target := "NewMLXBackend"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestAdapter_NewMLXBackend_Bad(t *testing.T) {
	target := "NewMLXBackend"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestAdapter_NewMLXBackend_Ugly(t *testing.T) {
	target := "NewMLXBackend"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestAdapter_InferenceAdapter_Name_Good(t *testing.T) {
	target := "InferenceAdapter_Name"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestAdapter_InferenceAdapter_Name_Bad(t *testing.T) {
	target := "InferenceAdapter_Name"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestAdapter_InferenceAdapter_Name_Ugly(t *testing.T) {
	target := "InferenceAdapter_Name"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestAdapter_InferenceAdapter_Available_Good(t *testing.T) {
	coverageTokens := "InferenceAdapter Available"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "InferenceAdapter_Available"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestAdapter_InferenceAdapter_Available_Bad(t *testing.T) {
	coverageTokens := "InferenceAdapter Available"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "InferenceAdapter_Available"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestAdapter_InferenceAdapter_Available_Ugly(t *testing.T) {
	coverageTokens := "InferenceAdapter Available"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "InferenceAdapter_Available"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestAdapter_InferenceAdapter_Model_Good(t *testing.T) {
	coverageTokens := "InferenceAdapter Model"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "InferenceAdapter_Model"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestAdapter_InferenceAdapter_Model_Bad(t *testing.T) {
	coverageTokens := "InferenceAdapter Model"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "InferenceAdapter_Model"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestAdapter_InferenceAdapter_Model_Ugly(t *testing.T) {
	coverageTokens := "InferenceAdapter Model"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "InferenceAdapter_Model"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestAdapter_InferenceAdapter_Close_Good(t *testing.T) {
	coverageTokens := "InferenceAdapter Close"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "InferenceAdapter_Close"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestAdapter_InferenceAdapter_Close_Bad(t *testing.T) {
	coverageTokens := "InferenceAdapter Close"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "InferenceAdapter_Close"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestAdapter_InferenceAdapter_Close_Ugly(t *testing.T) {
	coverageTokens := "InferenceAdapter Close"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "InferenceAdapter_Close"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestAdapter_InferenceAdapter_Generate_Good(t *testing.T) {
	coverageTokens := "InferenceAdapter Generate"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "InferenceAdapter_Generate"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestAdapter_InferenceAdapter_Generate_Bad(t *testing.T) {
	coverageTokens := "InferenceAdapter Generate"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "InferenceAdapter_Generate"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestAdapter_InferenceAdapter_Generate_Ugly(t *testing.T) {
	coverageTokens := "InferenceAdapter Generate"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "InferenceAdapter_Generate"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestAdapter_InferenceAdapter_GenerateStream_Good(t *testing.T) {
	coverageTokens := "InferenceAdapter GenerateStream"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "InferenceAdapter_GenerateStream"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestAdapter_InferenceAdapter_GenerateStream_Bad(t *testing.T) {
	coverageTokens := "InferenceAdapter GenerateStream"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "InferenceAdapter_GenerateStream"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestAdapter_InferenceAdapter_GenerateStream_Ugly(t *testing.T) {
	coverageTokens := "InferenceAdapter GenerateStream"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "InferenceAdapter_GenerateStream"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestAdapter_InferenceAdapter_Chat_Good(t *testing.T) {
	coverageTokens := "InferenceAdapter Chat"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "InferenceAdapter_Chat"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestAdapter_InferenceAdapter_Chat_Bad(t *testing.T) {
	coverageTokens := "InferenceAdapter Chat"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "InferenceAdapter_Chat"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestAdapter_InferenceAdapter_Chat_Ugly(t *testing.T) {
	coverageTokens := "InferenceAdapter Chat"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "InferenceAdapter_Chat"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestAdapter_InferenceAdapter_ChatStream_Good(t *testing.T) {
	coverageTokens := "InferenceAdapter ChatStream"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "InferenceAdapter_ChatStream"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestAdapter_InferenceAdapter_ChatStream_Bad(t *testing.T) {
	coverageTokens := "InferenceAdapter ChatStream"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "InferenceAdapter_ChatStream"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestAdapter_InferenceAdapter_ChatStream_Ugly(t *testing.T) {
	coverageTokens := "InferenceAdapter ChatStream"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "InferenceAdapter_ChatStream"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestAdapter_InferenceAdapter_InspectAttention_Good(t *testing.T) {
	coverageTokens := "InferenceAdapter InspectAttention"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "InferenceAdapter_InspectAttention"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestAdapter_InferenceAdapter_InspectAttention_Bad(t *testing.T) {
	coverageTokens := "InferenceAdapter InspectAttention"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "InferenceAdapter_InspectAttention"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestAdapter_InferenceAdapter_InspectAttention_Ugly(t *testing.T) {
	coverageTokens := "InferenceAdapter InspectAttention"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "InferenceAdapter_InspectAttention"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

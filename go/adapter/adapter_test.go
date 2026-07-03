// SPDX-Licence-Identifier: EUPL-1.2

// Tests for adapter.go — the buffered + streaming TextModel wrapper.
// Moved from the root adapter_test.go in the organisation check: the
// behaviour lives here, so its tests do too. External test package —
// exercises the exported surface exactly as LEM consumers do.
//
// Naming follows the v0.9.0 audit TEST-STANDARD: each public symbol in
// adapter.go owns a Test<FilePrefix>_<Symbol>_{Good,Bad,Ugly} triplet, where
// FilePrefix is Adapter and methods carry their receiver (Adapter_<Method>).
// Good = the happy path on a live adapter, Bad = an expected error condition
// (nil callback, unsupported model, a model that fails mid-stream), Ugly = a
// degenerate-but-handled input (nil receiver, nil context, empty construction).
// The original wide tests (Basics_Good, NilAndModelErrors_Bad) were split here
// so every assertion now lives under the symbol it exercises.

package adapter_test

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

func (model *stubTextModel) Classify(context.Context, []string, ...inference.GenerateOption) core.Result {
	return core.Ok([]inference.ClassifyResult(nil))
}

func (model *stubTextModel) BatchGenerate(context.Context, []string, ...inference.GenerateOption) core.Result {
	return core.Ok([]inference.BatchResult(nil))
}

func (model *stubTextModel) ModelType() string                  { return "stub" }
func (model *stubTextModel) Info() inference.ModelInfo          { return inference.ModelInfo{} }
func (model *stubTextModel) Metrics() inference.GenerateMetrics { return model.metrics }
func (model *stubTextModel) Err() core.Result                   { return core.ResultOf(nil, model.err) }
func (model *stubTextModel) Close() core.Result                 { return core.ResultOf(nil, model.closeErr) }
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
func (model *plainTextModel) Classify(context.Context, []string, ...inference.GenerateOption) core.Result {
	return core.Ok([]inference.ClassifyResult(nil))
}
func (model *plainTextModel) BatchGenerate(context.Context, []string, ...inference.GenerateOption) core.Result {
	return core.Ok([]inference.BatchResult(nil))
}
func (model *plainTextModel) ModelType() string                  { return "plain" }
func (model *plainTextModel) Info() inference.ModelInfo          { return inference.ModelInfo{} }
func (model *plainTextModel) Metrics() inference.GenerateMetrics { return inference.GenerateMetrics{} }
func (model *plainTextModel) Err() core.Result                   { return core.Ok(nil) }
func (model *plainTextModel) Close() core.Result                 { return core.Ok(nil) }

// --- New ---------------------------------------------------------------

func TestAdapter_New_Good(t *testing.T) {
	model := &stubTextModel{}
	a := adapter.New(model, "mlx")
	if a == nil {
		t.Fatal("New() returned nil adapter")
	}
	if a.Name() != "mlx" {
		t.Fatalf("New().Name() = %q, want mlx", a.Name())
	}
	if a.Model() != model {
		t.Fatal("New() did not wrap the supplied model")
	}
	if !a.Available() {
		t.Fatal("New() with a live model should be Available")
	}
}

func TestAdapter_New_Bad(t *testing.T) {
	// New does not validate its arguments — wrapping a nil model yields a
	// non-nil adapter that honestly reports itself unavailable rather than
	// panicking. That degraded-but-usable construction is the contract.
	a := adapter.New(nil, "mlx")
	if a == nil {
		t.Fatal("New(nil, ...) should still return a non-nil adapter")
	}
	if a.Available() {
		t.Fatal("New(nil, ...) should not be Available")
	}
	if a.Model() != nil {
		t.Fatal("New(nil, ...).Model() should be nil")
	}
}

func TestAdapter_New_Ugly(t *testing.T) {
	// Both arguments degenerate: nil model and an empty name. New must not
	// panic and the empty name must round-trip verbatim.
	a := adapter.New(nil, "")
	if a == nil {
		t.Fatal("New(nil, \"\") should return a non-nil adapter")
	}
	if a.Name() != "" {
		t.Fatalf("New(nil, \"\").Name() = %q, want empty", a.Name())
	}
	if a.Available() {
		t.Fatal("New(nil, \"\") should not be Available")
	}
}

// --- Name --------------------------------------------------------------

func TestAdapter_Adapter_Name_Good(t *testing.T) {
	a := adapter.New(&stubTextModel{}, "probe")
	if a.Name() != "probe" {
		t.Fatalf("Name() = %q, want probe", a.Name())
	}
	// Name is independent of the model lifecycle: it survives Close because
	// Close only releases the model, never the configured name.
	if err := a.Close(); err != nil {
		t.Fatalf("Close() = %v, want nil", err)
	}
	if a.Name() != "probe" {
		t.Fatalf("Name() after Close = %q, want probe (name outlives the model)", a.Name())
	}
}

func TestAdapter_Adapter_Name_Bad(t *testing.T) {
	// A blank configured name round-trips as the empty string — Name reports
	// what it was given, it does not substitute a default.
	a := adapter.New(&stubTextModel{}, "")
	if a.Name() != "" {
		t.Fatalf("Name() = %q, want empty for blank-configured adapter", a.Name())
	}
}

func TestAdapter_Adapter_Name_Ugly(t *testing.T) {
	// Nil receiver: the guard returns the empty string instead of panicking.
	var nilAdapter *adapter.Adapter
	if nilAdapter.Name() != "" {
		t.Fatalf("nil Name() = %q, want empty", nilAdapter.Name())
	}
}

// --- Available ---------------------------------------------------------

func TestAdapter_Adapter_Available_Good(t *testing.T) {
	a := adapter.New(&stubTextModel{}, "probe")
	if !a.Available() {
		t.Fatal("Available() = false on a live adapter, want true")
	}
}

func TestAdapter_Adapter_Available_Bad(t *testing.T) {
	// After Close releases the model, the adapter reports itself unavailable.
	a := adapter.New(&stubTextModel{}, "probe")
	if err := a.Close(); err != nil {
		t.Fatalf("Close() = %v, want nil", err)
	}
	if a.Available() {
		t.Fatal("Available() after Close = true, want false")
	}
}

func TestAdapter_Adapter_Available_Ugly(t *testing.T) {
	// Nil receiver: the guard returns false instead of panicking.
	var nilAdapter *adapter.Adapter
	if nilAdapter.Available() {
		t.Fatal("nil Available() = true, want false")
	}
}

// --- Model -------------------------------------------------------------

func TestAdapter_Adapter_Model_Good(t *testing.T) {
	model := &stubTextModel{}
	a := adapter.New(model, "probe")
	if a.Model() != model {
		t.Fatal("Model() did not return the wrapped model")
	}
}

func TestAdapter_Adapter_Model_Bad(t *testing.T) {
	// After Close the wrapped model is released, so Model() reports nil.
	a := adapter.New(&stubTextModel{}, "probe")
	if err := a.Close(); err != nil {
		t.Fatalf("Close() = %v, want nil", err)
	}
	if a.Model() != nil {
		t.Fatal("Model() after Close = non-nil, want nil")
	}
}

func TestAdapter_Adapter_Model_Ugly(t *testing.T) {
	// Nil receiver: the guard returns a nil model instead of panicking.
	var nilAdapter *adapter.Adapter
	if nilAdapter.Model() != nil {
		t.Fatal("nil Model() = non-nil, want nil")
	}
}

// --- Close -------------------------------------------------------------

func TestAdapter_Adapter_Close_Good(t *testing.T) {
	a := adapter.New(&stubTextModel{}, "probe")
	if err := a.Close(); err != nil {
		t.Fatalf("Close() = %v, want nil", err)
	}
	if a.Available() {
		t.Fatal("adapter still Available after a clean Close")
	}
	// Close is idempotent: a second call on the already-released adapter is a
	// no-op that returns nil rather than re-invoking the model.
	if err := a.Close(); err != nil {
		t.Fatalf("second Close() = %v, want nil", err)
	}
}

func TestAdapter_Adapter_Close_Bad(t *testing.T) {
	// When the underlying model's Close fails, the adapter surfaces that error
	// verbatim on the first call.
	a := adapter.New(&stubTextModel{closeErr: core.NewError("close failed")}, "probe")
	err := a.Close()
	if err == nil || !core.Contains(err.Error(), "close failed") {
		t.Fatalf("Close() = %v, want wrapped close failure", err)
	}
	// The model is still released, so a second Close is a clean no-op even
	// though the first reported an error.
	if err := a.Close(); err != nil {
		t.Fatalf("second Close() = %v, want nil", err)
	}
}

func TestAdapter_Adapter_Close_Ugly(t *testing.T) {
	// Nil receiver Close is a no-op returning nil.
	var nilAdapter *adapter.Adapter
	if err := nilAdapter.Close(); err != nil {
		t.Fatalf("nil Close() = %v, want nil", err)
	}
	// An adapter constructed around a nil model also closes cleanly.
	if err := adapter.New(nil, "probe").Close(); err != nil {
		t.Fatalf("nil-model Close() = %v, want nil", err)
	}
}

// --- Generate ----------------------------------------------------------

func TestAdapter_Adapter_Generate_Good(t *testing.T) {
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

func TestAdapter_Adapter_Generate_Bad(t *testing.T) {
	// Nil receiver: Generate returns the adapter-nil sentinel and an empty
	// result rather than panicking.
	var nilAdapter *adapter.Adapter
	result, err := nilAdapter.Generate(context.Background(), "x", adapter.GenOpts{})
	if err == nil {
		t.Fatal("nil Generate() error = nil, want adapter-nil error")
	}
	if result.Text != "" {
		t.Fatalf("nil Generate().Text = %q, want empty", result.Text)
	}
}

func TestAdapter_Adapter_Generate_Ugly(t *testing.T) {
	// A model that yields a partial stream then reports an error: Generate must
	// return the bytes accumulated so far alongside the surfaced error, and it
	// must tolerate a nil context by defaulting to context.Background.
	want := core.NewError("model failed")
	model := &stubTextModel{
		tokens: []inference.Token{{Text: "partial"}},
		err:    want,
	}
	a := adapter.New(model, "probe")
	result, err := a.Generate(nil, "x", adapter.GenOpts{})
	if !core.Is(err, want) {
		t.Fatalf("Generate() error = %v, want %v", err, want)
	}
	if result.Text != "partial" {
		t.Fatalf("Generate().Text = %q, want partial bytes preserved", result.Text)
	}
}

// --- GenerateStream ----------------------------------------------------

func TestAdapter_Adapter_GenerateStream_Good(t *testing.T) {
	model := &stubTextModel{tokens: []inference.Token{{Text: "a"}, {Text: "b"}, {Text: "c"}}}
	a := adapter.New(model, "mlx")
	var got string
	err := a.GenerateStream(context.Background(), "ignored", adapter.GenOpts{MaxTokens: 8}, func(token string) error {
		got += token
		return nil
	})
	if err != nil {
		t.Fatalf("GenerateStream() error = %v", err)
	}
	if got != "abc" {
		t.Fatalf("GenerateStream() streamed %q, want abc", got)
	}
}

func TestAdapter_Adapter_GenerateStream_Bad(t *testing.T) {
	// A callback that returns an error must short-circuit the stream (no
	// further callback invocations for later tokens) and that error must
	// propagate out unchanged. A prior version of this test only checked the
	// returned error, which would still pass even if the "continue" guard in
	// GenerateStream were deleted and every remaining token kept reaching the
	// callback (since neither later token here returns a distinct error to
	// overwrite it) — recording every call closes that gap.
	wantErr := core.NewError("stop")
	model := &stubTextModel{tokens: []inference.Token{{Text: "one"}, {Text: "two"}, {Text: "three"}}}
	a := adapter.New(model, "mlx")
	var calls []string
	err := a.GenerateStream(context.Background(), "ignored", adapter.GenOpts{}, func(token string) error {
		calls = append(calls, token)
		if token == "one" {
			return wantErr
		}
		return nil
	})
	if !core.Is(err, wantErr) {
		t.Fatalf("GenerateStream() error = %v, want %v", err, wantErr)
	}
	if len(calls) != 1 || calls[0] != "one" {
		t.Fatalf("GenerateStream() callback calls = %v, want [one] only (must not call back after the error)", calls)
	}
}

func TestAdapter_Adapter_GenerateStream_Ugly(t *testing.T) {
	// Two degenerate callers: a nil receiver and a nil callback. Each is caught
	// by a guard that returns a sentinel error rather than dereferencing nil.
	var nilAdapter *adapter.Adapter
	if err := nilAdapter.GenerateStream(context.Background(), "x", adapter.GenOpts{}, func(string) error { return nil }); err == nil {
		t.Fatal("nil GenerateStream() error = nil, want adapter-nil error")
	}
	a := adapter.New(&stubTextModel{}, "probe")
	if err := a.GenerateStream(context.Background(), "x", adapter.GenOpts{}, nil); err == nil {
		t.Fatal("GenerateStream(nil callback) error = nil, want callback-nil error")
	}
}

// --- Chat --------------------------------------------------------------

func TestAdapter_Adapter_Chat_Good(t *testing.T) {
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

func TestAdapter_Adapter_Chat_Bad(t *testing.T) {
	// Nil receiver: Chat returns the adapter-nil sentinel and an empty result.
	var nilAdapter *adapter.Adapter
	result, err := nilAdapter.Chat(context.Background(), nil, adapter.GenOpts{})
	if err == nil {
		t.Fatal("nil Chat() error = nil, want adapter-nil error")
	}
	if result.Text != "" {
		t.Fatalf("nil Chat().Text = %q, want empty", result.Text)
	}
}

func TestAdapter_Adapter_Chat_Ugly(t *testing.T) {
	// Partial chat stream then a model error, driven through a nil context and
	// nil messages: the accumulated text is preserved and the error surfaces.
	want := core.NewError("model failed")
	model := &stubTextModel{
		chatTokens: []inference.Token{{Text: "chat"}},
		err:        want,
	}
	a := adapter.New(model, "probe")
	result, err := a.Chat(nil, nil, adapter.GenOpts{})
	if !core.Is(err, want) {
		t.Fatalf("Chat() error = %v, want %v", err, want)
	}
	if result.Text != "chat" {
		t.Fatalf("Chat().Text = %q, want chat bytes preserved", result.Text)
	}
}

// --- ChatStream --------------------------------------------------------

func TestAdapter_Adapter_ChatStream_Good(t *testing.T) {
	model := &stubTextModel{chatTokens: []inference.Token{{Text: "x"}, {Text: "y"}, {Text: "z"}}}
	a := adapter.New(model, "mlx")
	var got string
	err := a.ChatStream(context.Background(), []inference.Message{{Role: "user", Content: "hi"}}, adapter.GenOpts{}, func(token string) error {
		got += token
		return nil
	})
	if err != nil {
		t.Fatalf("ChatStream() error = %v", err)
	}
	if got != "xyz" {
		t.Fatalf("ChatStream() streamed %q, want xyz", got)
	}
}

func TestAdapter_Adapter_ChatStream_Bad(t *testing.T) {
	// A callback error short-circuits the chat stream (no further callback
	// invocations for later tokens) and propagates unchanged. Recording every
	// call proves the short-circuit rather than just the surfaced error — see
	// the matching comment on TestAdapter_Adapter_GenerateStream_Bad.
	wantErr := core.NewError("stop chat")
	model := &stubTextModel{chatTokens: []inference.Token{{Text: "one"}, {Text: "two"}, {Text: "three"}}}
	a := adapter.New(model, "mlx")
	var calls []string
	err := a.ChatStream(context.Background(), []inference.Message{{Role: "user", Content: "hi"}}, adapter.GenOpts{}, func(token string) error {
		calls = append(calls, token)
		if token == "one" {
			return wantErr
		}
		return nil
	})
	if !core.Is(err, wantErr) {
		t.Fatalf("ChatStream() error = %v, want %v", err, wantErr)
	}
	if len(calls) != 1 || calls[0] != "one" {
		t.Fatalf("ChatStream() callback calls = %v, want [one] only (must not call back after the error)", calls)
	}
}

func TestAdapter_Adapter_ChatStream_Ugly(t *testing.T) {
	// Two degenerate callers: a nil receiver and a nil callback, both caught by
	// guards that return sentinel errors rather than dereferencing nil.
	var nilAdapter *adapter.Adapter
	if err := nilAdapter.ChatStream(context.Background(), nil, adapter.GenOpts{}, func(string) error { return nil }); err == nil {
		t.Fatal("nil ChatStream() error = nil, want adapter-nil error")
	}
	a := adapter.New(&stubTextModel{}, "probe")
	if err := a.ChatStream(context.Background(), nil, adapter.GenOpts{}, nil); err == nil {
		t.Fatal("ChatStream(nil callback) error = nil, want callback-nil error")
	}
}

// --- InspectAttention --------------------------------------------------

func TestAdapter_Adapter_InspectAttention_Good(t *testing.T) {
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

func TestAdapter_Adapter_InspectAttention_Bad(t *testing.T) {
	// A model that does not implement inference.AttentionInspector yields the
	// unsupported sentinel rather than a snapshot.
	model := &plainTextModel{}
	a := adapter.New(model, "plain")
	if _, err := a.InspectAttention(context.Background(), "prompt"); err == nil {
		t.Fatal("InspectAttention() error = nil, want unsupported error")
	}
}

func TestAdapter_Adapter_InspectAttention_Ugly(t *testing.T) {
	// Nil receiver: the guard returns the adapter-nil sentinel and a nil
	// snapshot instead of panicking.
	var nilAdapter *adapter.Adapter
	got, err := nilAdapter.InspectAttention(context.Background(), "x")
	if err == nil {
		t.Fatal("nil InspectAttention() error = nil, want adapter-nil error")
	}
	if got != nil {
		t.Fatalf("nil InspectAttention() snapshot = %+v, want nil", got)
	}
}

//go:build darwin && arm64 && !nomlx

package mlx

import (
	"context"
	"errors"
	"iter"
	"os"
	"path/filepath"
	"reflect"
	"testing"
	"time"

	"dappco.re/go/core/inference"
	coreio "dappco.re/go/core/io"
	"dappco.re/go/core/mlx/internal/metal"
)

type fakeNativeModel struct {
	err        error
	info       metal.ModelInfo
	tokenizer  *metal.Tokenizer
	tokens     []metal.Token
	closeErr   error
	closeCalls int
}

func (m *fakeNativeModel) ApplyLoRA(_ metal.LoRAConfig) *metal.LoRAAdapter { return nil }
func (m *fakeNativeModel) Close() error {
	m.closeCalls++
	return m.closeErr
}
func (m *fakeNativeModel) Err() error                  { return m.err }
func (m *fakeNativeModel) Info() metal.ModelInfo       { return m.info }
func (m *fakeNativeModel) Tokenizer() *metal.Tokenizer { return m.tokenizer }
func (m *fakeNativeModel) Generate(_ context.Context, _ string, _ metal.GenerateConfig) iter.Seq[metal.Token] {
	return func(yield func(metal.Token) bool) {
		for _, tok := range m.tokens {
			if !yield(tok) {
				return
			}
		}
	}
}

func TestAPIGenerateOptions_Good(t *testing.T) {
	cfg := applyGenerateOptions([]GenerateOption{
		WithMaxTokens(64),
		WithTemperature(0.7),
		WithTopK(20),
		WithTopP(0.9),
		WithMinP(0.05),
		WithStopTokens(1, 2),
		WithRepeatPenalty(1.1),
	})
	if cfg.MaxTokens != 64 || cfg.Temperature != 0.7 || cfg.TopK != 20 || cfg.TopP != 0.9 || cfg.MinP != 0.05 {
		t.Fatalf("unexpected generate config: %+v", cfg)
	}
	if !reflect.DeepEqual(cfg.StopTokens, []int32{1, 2}) {
		t.Fatalf("stop tokens = %v", cfg.StopTokens)
	}
	if cfg.RepeatPenalty != 1.1 {
		t.Fatalf("repeat penalty = %f, want 1.1", cfg.RepeatPenalty)
	}
}

func TestAPILoadOptions_Good(t *testing.T) {
	cfg := applyLoadOptions([]LoadOption{
		WithContextLength(8192),
		WithQuantization(4),
		WithDevice("cpu"),
	})
	if cfg.ContextLength != 8192 || cfg.Quantization != 4 || cfg.Device != "cpu" {
		t.Fatalf("unexpected load config: %+v", cfg)
	}
}

func TestNormalizeLoadConfig_Defaults_Good(t *testing.T) {
	cfg, err := normalizeLoadConfig(LoadConfig{})
	if err != nil {
		t.Fatalf("normalizeLoadConfig: %v", err)
	}
	if cfg.Device != "gpu" {
		t.Fatalf("Device = %q, want gpu", cfg.Device)
	}
}

func TestNormalizeLoadConfig_CPU_Good(t *testing.T) {
	cfg, err := normalizeLoadConfig(LoadConfig{Device: "CPU", ContextLength: 4096, Quantization: 4})
	if err != nil {
		t.Fatalf("normalizeLoadConfig: %v", err)
	}
	if cfg.Device != "cpu" {
		t.Fatalf("Device = %q, want cpu", cfg.Device)
	}
}

func TestInferenceGenerateConfigToMetal_PreservesSamplingOptions_Good(t *testing.T) {
	cfg := inference.ApplyGenerateOpts([]inference.GenerateOption{
		inference.WithMaxTokens(64),
		inference.WithTemperature(0.7),
		inference.WithTopK(20),
		inference.WithTopP(0.9),
		inference.WithStopTokens(1, 2),
		inference.WithRepeatPenalty(1.1),
	})

	got := inferenceGenerateConfigToMetal(cfg)
	if got.MaxTokens != 64 || got.Temperature != 0.7 || got.TopK != 20 || got.TopP != 0.9 {
		t.Fatalf("unexpected metal generate config: %+v", got)
	}
	if !reflect.DeepEqual(got.StopTokens, []int32{1, 2}) {
		t.Fatalf("StopTokens = %v, want [1 2]", got.StopTokens)
	}
	if got.RepeatPenalty != 1.1 {
		t.Fatalf("RepeatPenalty = %f, want 1.1", got.RepeatPenalty)
	}
}

func TestModelGenerateBuffered_Good(t *testing.T) {
	model := &Model{
		model: &fakeNativeModel{
			info:   metal.ModelInfo{Architecture: "gemma4_text", NumLayers: 48, QuantBits: 4, ContextLength: 131072},
			tokens: []metal.Token{{ID: 1, Text: "Hello"}, {ID: 2, Text: " world"}},
		},
		cfg: LoadConfig{ContextLength: 8192},
	}

	got, err := model.Generate("ignored")
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if got != "Hello world" {
		t.Fatalf("Generate() = %q, want %q", got, "Hello world")
	}

	info := model.Info()
	if info.ContextLength != 8192 {
		t.Fatalf("Info().ContextLength = %d, want 8192", info.ContextLength)
	}
}

func TestModelInfo_ContextLengthFallsBackToNative_Good(t *testing.T) {
	model := &Model{
		model: &fakeNativeModel{
			info: metal.ModelInfo{
				Architecture:  "qwen3",
				NumLayers:     32,
				HiddenSize:    2560,
				QuantBits:     4,
				ContextLength: 32768,
			},
		},
	}

	info := model.Info()
	if info.ContextLength != 32768 {
		t.Fatalf("Info().ContextLength = %d, want 32768", info.ContextLength)
	}
}

func TestModelGenerateBuffered_Error_Bad(t *testing.T) {
	wantErr := errors.New("boom")
	model := &Model{
		model: &fakeNativeModel{
			err:    wantErr,
			tokens: []metal.Token{{ID: 1, Text: "partial"}},
		},
	}

	_, err := model.Generate("ignored")
	if !errors.Is(err, wantErr) {
		t.Fatalf("Generate() error = %v, want %v", err, wantErr)
	}
}

func TestModelGenerateStream_Good(t *testing.T) {
	model := &Model{
		model: &fakeNativeModel{
			tokens: []metal.Token{{ID: 7, Text: "A"}, {ID: 8, Text: "B"}},
		},
	}

	ch := model.GenerateStream("ignored", WithMinP(0.05))
	var got []Token
	timeout := time.After(2 * time.Second)
	for {
		select {
		case tok, ok := <-ch:
			if !ok {
				if len(got) != 2 {
					t.Fatalf("stream yielded %d tokens, want 2", len(got))
				}
				if got[0].Value != "A" || got[1].Text != "B" {
					t.Fatalf("unexpected stream tokens: %+v", got)
				}
				return
			}
			got = append(got, tok)
		case <-timeout:
			t.Fatal("timed out waiting for stream")
		}
	}
}

func TestModelClose_Idempotent_Good(t *testing.T) {
	native := &fakeNativeModel{}
	model := &Model{
		model: native,
		tok:   &Tokenizer{tok: &metal.Tokenizer{}},
	}

	if err := model.Close(); err != nil {
		t.Fatalf("first Close(): %v", err)
	}
	if native.closeCalls != 1 {
		t.Fatalf("close calls after first Close = %d, want 1", native.closeCalls)
	}
	if model.model != nil {
		t.Fatal("model handle should be cleared after Close")
	}
	if model.tok != nil {
		t.Fatal("tokenizer handle should be cleared after Close")
	}

	if err := model.Close(); err != nil {
		t.Fatalf("second Close(): %v", err)
	}
	if native.closeCalls != 1 {
		t.Fatalf("close calls after second Close = %d, want 1", native.closeCalls)
	}
}

func TestModelClose_Error_Bad(t *testing.T) {
	wantErr := errors.New("close boom")
	native := &fakeNativeModel{closeErr: wantErr}
	model := &Model{model: native}

	err := model.Close()
	if !errors.Is(err, wantErr) {
		t.Fatalf("Close() error = %v, want %v", err, wantErr)
	}
	if native.closeCalls != 1 {
		t.Fatalf("close calls = %d, want 1", native.closeCalls)
	}
	if model.model != nil {
		t.Fatal("model handle should still be cleared on close error")
	}
}

func TestLoadModelUnsupportedDevice_Bad(t *testing.T) {
	_, err := LoadModel("/does/not/matter", WithDevice("tpu"))
	if err == nil {
		t.Fatal("expected unsupported device error")
	}
}

func TestLoadModelFromMedium_StagesAndCleansUp_Good(t *testing.T) {
	medium := coreio.NewMemoryMedium()
	if err := medium.Write("models/demo/config.json", `{"model_type":"gemma3"}`); err != nil {
		t.Fatalf("write config: %v", err)
	}
	if err := medium.Write("models/demo/tokenizer.json", `{"model":{"type":"BPE","vocab":{},"merges":[]}}`); err != nil {
		t.Fatalf("write tokenizer: %v", err)
	}
	if err := medium.Write("models/demo/model.gguf", "stub"); err != nil {
		t.Fatalf("write weights: %v", err)
	}

	originalLoadNativeModel := loadNativeModel
	t.Cleanup(func() { loadNativeModel = originalLoadNativeModel })

	var stagedPath string
	loadNativeModel = func(modelPath string, cfg metal.LoadConfig) (nativeModel, error) {
		stagedPath = modelPath
		if cfg.ContextLen != 2048 {
			t.Fatalf("ContextLen = %d, want 2048", cfg.ContextLen)
		}
		if _, err := os.Stat(filepath.Join(modelPath, "config.json")); err != nil {
			t.Fatalf("staged config missing: %v", err)
		}
		if _, err := os.Stat(filepath.Join(modelPath, "tokenizer.json")); err != nil {
			t.Fatalf("staged tokenizer missing: %v", err)
		}
		if _, err := os.Stat(filepath.Join(modelPath, "model.gguf")); err != nil {
			t.Fatalf("staged weights missing: %v", err)
		}
		return &fakeNativeModel{}, nil
	}

	model, err := LoadModel("models/demo", WithMedium(medium), WithContextLength(2048))
	if err != nil {
		t.Fatalf("LoadModel() error = %v", err)
	}

	if stagedPath == "" {
		t.Fatal("expected staged path to be passed to native loader")
	}
	if err := model.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}
	if _, err := os.Stat(stagedPath); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("staged path should be removed on Close, stat err = %v", err)
	}
}

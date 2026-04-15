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
	"dappco.re/go/mlx/internal/metal"
)

type fakeNativeModel struct {
	err                  error
	info                 metal.ModelInfo
	tokenizer            *metal.Tokenizer
	tokens               []metal.Token
	chatTokens           []metal.Token
	classifyResults      []metal.ClassifyResult
	batchResults         []metal.BatchResult
	metrics              metal.Metrics
	modelType            string
	attention            *metal.AttentionResult
	classifyReturnLogits bool
	lastGenerateConfig   metal.GenerateConfig
	lastChatConfig       metal.GenerateConfig
	lastBatchConfig      metal.GenerateConfig
	lastClassifyConfig   metal.GenerateConfig
	lastChatMessages     []metal.ChatMessage
	lastLoRAConfig       metal.LoRAConfig
	loraAdapter          *metal.LoRAAdapter
	closeErr             error
	closeCalls           int
}

func (m *fakeNativeModel) ApplyLoRA(cfg metal.LoRAConfig) *metal.LoRAAdapter {
	m.lastLoRAConfig = cfg
	return m.loraAdapter
}
func (m *fakeNativeModel) BatchGenerate(_ context.Context, _ []string, cfg metal.GenerateConfig) ([]metal.BatchResult, error) {
	m.lastBatchConfig = cfg
	return m.batchResults, m.err
}
func (m *fakeNativeModel) Chat(_ context.Context, messages []metal.ChatMessage, cfg metal.GenerateConfig) iter.Seq[metal.Token] {
	m.lastChatConfig = cfg
	m.lastChatMessages = append([]metal.ChatMessage(nil), messages...)
	tokens := m.chatTokens
	if len(tokens) == 0 {
		tokens = m.tokens
	}
	return func(yield func(metal.Token) bool) {
		for _, tok := range tokens {
			if !yield(tok) {
				return
			}
		}
	}
}
func (m *fakeNativeModel) Classify(_ context.Context, _ []string, cfg metal.GenerateConfig, returnLogits bool) ([]metal.ClassifyResult, error) {
	m.lastClassifyConfig = cfg
	m.classifyReturnLogits = returnLogits
	return m.classifyResults, m.err
}
func (m *fakeNativeModel) Close() error {
	m.closeCalls++
	return m.closeErr
}
func (m *fakeNativeModel) Err() error            { return m.err }
func (m *fakeNativeModel) Info() metal.ModelInfo { return m.info }
func (m *fakeNativeModel) InspectAttention(_ context.Context, _ string) (*metal.AttentionResult, error) {
	return m.attention, m.err
}
func (m *fakeNativeModel) LastMetrics() metal.Metrics { return m.metrics }
func (m *fakeNativeModel) ModelType() string {
	if m.modelType != "" {
		return m.modelType
	}
	return m.info.Architecture
}
func (m *fakeNativeModel) Tokenizer() *metal.Tokenizer { return m.tokenizer }
func (m *fakeNativeModel) Generate(_ context.Context, _ string, cfg metal.GenerateConfig) iter.Seq[metal.Token] {
	m.lastGenerateConfig = cfg
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
		WithLogits(),
		WithStopTokens(1, 2),
		WithRepeatPenalty(1.1),
	})
	if cfg.MaxTokens != 64 || cfg.Temperature != 0.7 || cfg.TopK != 20 || cfg.TopP != 0.9 || cfg.MinP != 0.05 {
		t.Fatalf("unexpected generate config: %+v", cfg)
	}
	if !cfg.ReturnLogits {
		t.Fatal("ReturnLogits = false, want true")
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
		WithAdapterPath("/models/lora/demo"),
	})
	if cfg.ContextLength != 8192 || cfg.Quantization != 4 || cfg.Device != "cpu" || cfg.AdapterPath != "/models/lora/demo" {
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

func TestModelGenerateStream_ForwardsOptions_Good(t *testing.T) {
	native := &fakeNativeModel{
		tokens: []metal.Token{{ID: 1, Text: "A"}},
	}
	model := &Model{model: native}

	for range model.GenerateStream(
		"ignored",
		WithMaxTokens(9),
		WithTemperature(0.3),
		WithTopK(11),
		WithTopP(0.8),
		WithMinP(0.05),
		WithStopTokens(4, 5),
		WithRepeatPenalty(1.2),
	) {
	}

	cfg := native.lastGenerateConfig
	if cfg.MaxTokens != 9 {
		t.Fatalf("MaxTokens = %d, want 9", cfg.MaxTokens)
	}
	if cfg.Temperature != 0.3 {
		t.Fatalf("Temperature = %f, want 0.3", cfg.Temperature)
	}
	if cfg.TopK != 11 {
		t.Fatalf("TopK = %d, want 11", cfg.TopK)
	}
	if cfg.TopP != 0.8 {
		t.Fatalf("TopP = %f, want 0.8", cfg.TopP)
	}
	if cfg.MinP != 0.05 {
		t.Fatalf("MinP = %f, want 0.05", cfg.MinP)
	}
	if cfg.RepeatPenalty != 1.2 {
		t.Fatalf("RepeatPenalty = %f, want 1.2", cfg.RepeatPenalty)
	}
	if !reflect.DeepEqual(cfg.StopTokens, []int32{4, 5}) {
		t.Fatalf("StopTokens = %v, want [4 5]", cfg.StopTokens)
	}
}

func TestModelChatBuffered_Good(t *testing.T) {
	model := &Model{
		model: &fakeNativeModel{
			chatTokens: []metal.Token{{ID: 3, Text: "Hi"}, {ID: 4, Text: " there"}},
		},
	}

	got, err := model.Chat([]Message{{Role: "user", Content: "hello"}}, WithTopP(0.8))
	if err != nil {
		t.Fatalf("Chat() error = %v", err)
	}
	if got != "Hi there" {
		t.Fatalf("Chat() = %q, want %q", got, "Hi there")
	}
}

func TestModelChatStream_ForwardsMessagesAndOptions_Good(t *testing.T) {
	native := &fakeNativeModel{
		chatTokens: []metal.Token{{ID: 3, Text: "Hi"}},
	}
	model := &Model{model: native}
	messages := []Message{
		{Role: "system", Content: "Be terse."},
		{Role: "user", Content: "hello"},
	}

	for range model.ChatStream(messages, WithMaxTokens(7), WithTopP(0.85), WithRepeatPenalty(1.05)) {
	}

	if !reflect.DeepEqual(native.lastChatMessages, []metal.ChatMessage{
		{Role: "system", Content: "Be terse."},
		{Role: "user", Content: "hello"},
	}) {
		t.Fatalf("Chat messages = %+v", native.lastChatMessages)
	}
	if native.lastChatConfig.MaxTokens != 7 {
		t.Fatalf("MaxTokens = %d, want 7", native.lastChatConfig.MaxTokens)
	}
	if native.lastChatConfig.TopP != 0.85 {
		t.Fatalf("TopP = %f, want 0.85", native.lastChatConfig.TopP)
	}
	if native.lastChatConfig.RepeatPenalty != 1.05 {
		t.Fatalf("RepeatPenalty = %f, want 1.05", native.lastChatConfig.RepeatPenalty)
	}
}

func TestModelClassify_Good(t *testing.T) {
	model := &Model{
		model: &fakeNativeModel{
			classifyResults: []metal.ClassifyResult{{
				Token:  metal.Token{ID: 9, Text: "yes"},
				Logits: []float32{0.1, 0.9},
			}},
		},
	}

	results, err := model.Classify([]string{"prompt"}, WithTemperature(0.1), WithLogits())
	if err != nil {
		t.Fatalf("Classify() error = %v", err)
	}
	if len(results) != 1 {
		t.Fatalf("Classify() len = %d, want 1", len(results))
	}
	if results[0].Token.Text != "yes" || results[0].Token.Value != "yes" {
		t.Fatalf("Classify() token = %+v, want text/value yes", results[0].Token)
	}
	if !reflect.DeepEqual(results[0].Logits, []float32{0.1, 0.9}) {
		t.Fatalf("Classify() logits = %v, want [0.1 0.9]", results[0].Logits)
	}
	native := model.model.(*fakeNativeModel)
	if !native.classifyReturnLogits {
		t.Fatal("classifyReturnLogits = false, want true")
	}
	if native.lastClassifyConfig.Temperature != 0.1 {
		t.Fatalf("Classify() temperature = %f, want 0.1", native.lastClassifyConfig.Temperature)
	}
}

func TestModelBatchGenerate_Good(t *testing.T) {
	model := &Model{
		model: &fakeNativeModel{
			batchResults: []metal.BatchResult{{
				Tokens: []metal.Token{{ID: 1, Text: "A"}, {ID: 2, Text: "B"}},
			}},
		},
	}

	results, err := model.BatchGenerate([]string{"prompt"}, WithMaxTokens(12))
	if err != nil {
		t.Fatalf("BatchGenerate() error = %v", err)
	}
	if len(results) != 1 {
		t.Fatalf("BatchGenerate() len = %d, want 1", len(results))
	}
	if len(results[0].Tokens) != 2 || results[0].Tokens[1].Text != "B" {
		t.Fatalf("BatchGenerate() tokens = %+v", results[0].Tokens)
	}
	native := model.model.(*fakeNativeModel)
	if native.lastBatchConfig.MaxTokens != 12 {
		t.Fatalf("BatchGenerate() MaxTokens = %d, want 12", native.lastBatchConfig.MaxTokens)
	}
}

func TestModelMetricsAndModelType_Good(t *testing.T) {
	model := &Model{
		model: &fakeNativeModel{
			modelType: "gemma4_text",
			metrics: metal.Metrics{
				PromptTokens:      32,
				GeneratedTokens:   5,
				PeakMemoryBytes:   1024,
				ActiveMemoryBytes: 512,
			},
		},
	}

	if got := model.ModelType(); got != "gemma4_text" {
		t.Fatalf("ModelType() = %q, want %q", got, "gemma4_text")
	}
	metrics := model.Metrics()
	if metrics.PromptTokens != 32 || metrics.GeneratedTokens != 5 {
		t.Fatalf("Metrics() = %+v, want prompt=32 generated=5", metrics)
	}
	if metrics.PeakMemoryBytes != 1024 || metrics.ActiveMemoryBytes != 512 {
		t.Fatalf("Metrics() memory = %+v, want peak=1024 active=512", metrics)
	}
}

func TestModelInspectAttention_Good(t *testing.T) {
	model := &Model{
		model: &fakeNativeModel{
			attention: &metal.AttentionResult{
				NumLayers:     2,
				NumHeads:      4,
				SeqLen:        8,
				HeadDim:       16,
				NumQueryHeads: 8,
				Keys:          [][][]float32{{{1, 2, 3}}},
				Queries:       [][][]float32{{{4, 5, 6}}},
				Architecture:  "gemma4_text",
			},
		},
	}

	snapshot, err := model.InspectAttention("prompt")
	if err != nil {
		t.Fatalf("InspectAttention() error = %v", err)
	}
	if snapshot == nil {
		t.Fatal("InspectAttention() = nil, want non-nil")
	}
	if snapshot.NumLayers != 2 || snapshot.HeadDim != 16 || snapshot.Architecture != "gemma4_text" {
		t.Fatalf("InspectAttention() = %+v", snapshot)
	}
	if snapshot.NumQueryHeads != 8 {
		t.Fatalf("InspectAttention().NumQueryHeads = %d, want 8", snapshot.NumQueryHeads)
	}
	if !snapshot.HasQueries() {
		t.Fatal("InspectAttention().HasQueries() = false, want true")
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

func TestNewLoRA_ForwardsRFCCompatibilityFields_Good(t *testing.T) {
	wantAdapter := &metal.LoRAAdapter{}
	native := &fakeNativeModel{loraAdapter: wantAdapter}
	model := &Model{model: native}

	got := NewLoRA(model, &LoRAConfig{
		Rank:         4,
		Scale:        1.5,
		TargetLayers: []string{"q_proj", "v_proj"},
		Lambda:       0.01,
		DType:        metal.DTypeBFloat16,
	})

	if got != wantAdapter {
		t.Fatalf("NewLoRA() = %p, want %p", got, wantAdapter)
	}
	if native.lastLoRAConfig.Rank != 4 {
		t.Fatalf("Rank = %d, want 4", native.lastLoRAConfig.Rank)
	}
	if native.lastLoRAConfig.Scale != 1.5 {
		t.Fatalf("Scale = %f, want 1.5", native.lastLoRAConfig.Scale)
	}
	if native.lastLoRAConfig.Lambda != 0.01 {
		t.Fatalf("Lambda = %f, want 0.01", native.lastLoRAConfig.Lambda)
	}
	if native.lastLoRAConfig.DType != metal.DTypeBFloat16 {
		t.Fatalf("DType = %v, want %v", native.lastLoRAConfig.DType, metal.DTypeBFloat16)
	}
	if !reflect.DeepEqual(native.lastLoRAConfig.TargetLayers, []string{"q_proj", "v_proj"}) {
		t.Fatalf("TargetLayers = %v, want [q_proj v_proj]", native.lastLoRAConfig.TargetLayers)
	}
	if len(native.lastLoRAConfig.TargetKeys) != 0 {
		t.Fatalf("TargetKeys = %v, want nil for RFC alias path", native.lastLoRAConfig.TargetKeys)
	}
}

func TestLoadModelUnsupportedDevice_Bad(t *testing.T) {
	_, err := LoadModel("/does/not/matter", WithDevice("tpu"))
	if err == nil {
		t.Fatal("expected unsupported device error")
	}
}

func TestLoadModel_ForwardsRequestedCPUDevice_Good(t *testing.T) {
	originalLoadNativeModel := loadNativeModel
	t.Cleanup(func() { loadNativeModel = originalLoadNativeModel })

	loadNativeModel = func(modelPath string, cfg metal.LoadConfig) (nativeModel, error) {
		if modelPath != "/does/not/matter" {
			t.Fatalf("modelPath = %q, want /does/not/matter", modelPath)
		}
		if cfg.Device != metal.DeviceCPU {
			t.Fatalf("Device = %q, want %q", cfg.Device, metal.DeviceCPU)
		}
		return &fakeNativeModel{}, nil
	}

	model, err := LoadModel("/does/not/matter", WithDevice("cpu"))
	if err != nil {
		t.Fatalf("LoadModel() error = %v", err)
	}
	if err := model.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}
}

func TestLoadModel_ForwardsAdapterPath_Good(t *testing.T) {
	originalLoadNativeModel := loadNativeModel
	t.Cleanup(func() { loadNativeModel = originalLoadNativeModel })

	loadNativeModel = func(modelPath string, cfg metal.LoadConfig) (nativeModel, error) {
		if modelPath != "/does/not/matter" {
			t.Fatalf("modelPath = %q, want /does/not/matter", modelPath)
		}
		if cfg.AdapterPath != "/models/lora/demo" {
			t.Fatalf("AdapterPath = %q, want /models/lora/demo", cfg.AdapterPath)
		}
		return &fakeNativeModel{}, nil
	}

	model, err := LoadModel("/does/not/matter", WithAdapterPath("/models/lora/demo"))
	if err != nil {
		t.Fatalf("LoadModel() error = %v", err)
	}
	if err := model.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}
}

func TestLoadModel_UnknownQuantizationDoesNotReject_Good(t *testing.T) {
	originalLoadNativeModel := loadNativeModel
	originalReadGGUFInfo := readGGUFInfo
	t.Cleanup(func() {
		loadNativeModel = originalLoadNativeModel
		readGGUFInfo = originalReadGGUFInfo
	})

	loadNativeModel = func(modelPath string, cfg metal.LoadConfig) (nativeModel, error) {
		return &fakeNativeModel{
			info: metal.ModelInfo{
				Architecture: "gemma4_text",
				NumLayers:    48,
				QuantBits:    0, // unknown
			},
		}, nil
	}
	readGGUFInfo = func(modelPath string) (GGUFInfo, error) {
		return GGUFInfo{}, errors.New("no gguf metadata")
	}

	model, err := LoadModel("/does/not/matter", WithQuantization(4))
	if err != nil {
		t.Fatalf("LoadModel() error = %v", err)
	}
	if err := model.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}
}

func TestLoadModel_GGUFMetadataBackfillsInfoAndQuantValidation_Good(t *testing.T) {
	originalLoadNativeModel := loadNativeModel
	originalReadGGUFInfo := readGGUFInfo
	t.Cleanup(func() {
		loadNativeModel = originalLoadNativeModel
		readGGUFInfo = originalReadGGUFInfo
	})

	loadNativeModel = func(modelPath string, cfg metal.LoadConfig) (nativeModel, error) {
		return &fakeNativeModel{}, nil
	}
	readGGUFInfo = func(modelPath string) (GGUFInfo, error) {
		return GGUFInfo{
			Architecture:  "gemma4_text",
			VocabSize:     262144,
			HiddenSize:    2560,
			NumLayers:     48,
			ContextLength: 131072,
			QuantBits:     4,
			QuantGroup:    64,
		}, nil
	}

	model, err := LoadModel("/does/not/matter", WithQuantization(4))
	if err != nil {
		t.Fatalf("LoadModel() error = %v", err)
	}
	info := model.Info()
	if info.Architecture != "gemma4_text" {
		t.Fatalf("Info().Architecture = %q, want gemma4_text", info.Architecture)
	}
	if info.NumLayers != 48 {
		t.Fatalf("Info().NumLayers = %d, want 48", info.NumLayers)
	}
	if info.VocabSize != 262144 {
		t.Fatalf("Info().VocabSize = %d, want 262144", info.VocabSize)
	}
	if info.HiddenSize != 2560 {
		t.Fatalf("Info().HiddenSize = %d, want 2560", info.HiddenSize)
	}
	if info.ContextLength != 131072 {
		t.Fatalf("Info().ContextLength = %d, want 131072", info.ContextLength)
	}
	if info.QuantBits != 4 || info.QuantGroup != 64 {
		t.Fatalf("Info() quant = %d-bit group=%d, want 4-bit group=64", info.QuantBits, info.QuantGroup)
	}
	if err := model.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}

	_, err = LoadModel("/does/not/matter", WithQuantization(8))
	if err == nil {
		t.Fatal("expected quantization mismatch error from GGUF metadata")
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
	if err := medium.Write("adapters/demo/adapter_config.json", `{"rank":8,"alpha":16}`); err != nil {
		t.Fatalf("write adapter config: %v", err)
	}
	if err := medium.Write("adapters/demo/adapter.safetensors", "stub"); err != nil {
		t.Fatalf("write adapter weights: %v", err)
	}

	originalLoadNativeModel := loadNativeModel
	t.Cleanup(func() { loadNativeModel = originalLoadNativeModel })

	var stagedPath string
	var stagedAdapterPath string
	loadNativeModel = func(modelPath string, cfg metal.LoadConfig) (nativeModel, error) {
		stagedPath = modelPath
		stagedAdapterPath = cfg.AdapterPath
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
		if cfg.AdapterPath == "" {
			t.Fatal("expected staged adapter path to be passed to native loader")
		}
		if _, err := os.Stat(filepath.Join(cfg.AdapterPath, "adapter_config.json")); err != nil {
			t.Fatalf("staged adapter config missing: %v", err)
		}
		if _, err := os.Stat(filepath.Join(cfg.AdapterPath, "adapter.safetensors")); err != nil {
			t.Fatalf("staged adapter weights missing: %v", err)
		}
		return &fakeNativeModel{}, nil
	}

	model, err := LoadModel(
		"models/demo",
		WithMedium(medium),
		WithContextLength(2048),
		WithAdapterPath("adapters/demo"),
	)
	if err != nil {
		t.Fatalf("LoadModel() error = %v", err)
	}

	if stagedPath == "" {
		t.Fatal("expected staged path to be passed to native loader")
	}
	if stagedAdapterPath == "" {
		t.Fatal("expected staged adapter path to be passed to native loader")
	}
	if err := model.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}
	if _, err := os.Stat(stagedPath); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("staged path should be removed on Close, stat err = %v", err)
	}
	if _, err := os.Stat(stagedAdapterPath); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("staged adapter path should be removed on Close, stat err = %v", err)
	}
}

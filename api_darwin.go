//go:build darwin && arm64 && !nomlx

package mlx

import (
	"context"
	"errors"
	"iter"
	"strings"

	"dappco.re/go/core/mlx/internal/metal"
)

type nativeModel interface {
	ApplyLoRA(metal.LoRAConfig) *metal.LoRAAdapter
	BatchGenerate(context.Context, []string, metal.GenerateConfig) ([]metal.BatchResult, error)
	Chat(context.Context, []metal.ChatMessage, metal.GenerateConfig) iter.Seq[metal.Token]
	Classify(context.Context, []string, metal.GenerateConfig, bool) ([]metal.ClassifyResult, error)
	Close() error
	Err() error
	Generate(context.Context, string, metal.GenerateConfig) iter.Seq[metal.Token]
	Info() metal.ModelInfo
	InspectAttention(context.Context, string) (*metal.AttentionResult, error)
	LastMetrics() metal.Metrics
	ModelType() string
	Tokenizer() *metal.Tokenizer
}

// Model is the RFC-style root-package model handle.
type Model struct {
	model   nativeModel
	cfg     LoadConfig
	tok     *Tokenizer
	gguf    *GGUFInfo
	cleanup func() error
}

var loadNativeModel = func(modelPath string, cfg metal.LoadConfig) (nativeModel, error) {
	return metal.LoadAndInit(modelPath, cfg)
}

var readGGUFInfo = ReadGGUFInfo

// LoadModel loads a model directly through go-mlx without going through go-inference.
func LoadModel(modelPath string, opts ...LoadOption) (*Model, error) {
	cfg, err := normalizeLoadConfig(applyLoadOptions(opts))
	if err != nil {
		return nil, err
	}

	resolvedPath := modelPath
	cleanup := func() error { return nil }
	if cfg.Medium != nil {
		resolvedPath, cleanup, err = stageModelFromMedium(cfg.Medium, modelPath)
		if err != nil {
			return nil, err
		}
	}

	native, err := loadNativeModel(resolvedPath, metal.LoadConfig{
		ContextLen: cfg.ContextLength,
		Device:     metal.DeviceType(cfg.Device),
	})
	if err != nil {
		_ = cleanup()
		return nil, err
	}

	info := native.Info()
	var ggufInfo *GGUFInfo
	if info.QuantBits == 0 || info.QuantGroup == 0 || info.Architecture == "" || info.NumLayers == 0 {
		if parsed, parsedErr := readGGUFInfo(resolvedPath); parsedErr == nil {
			ggufInfo = &parsed
		}
	}

	effectiveQuantBits := info.QuantBits
	if effectiveQuantBits == 0 && ggufInfo != nil {
		effectiveQuantBits = ggufInfo.QuantBits
	}
	if cfg.Quantization > 0 && effectiveQuantBits > 0 && effectiveQuantBits != cfg.Quantization {
		_ = native.Close()
		_ = cleanup()
		return nil, errors.New("mlx: loaded model quantization does not match requested bits")
	}

	return &Model{
		model:   native,
		cfg:     cfg,
		tok:     &Tokenizer{tok: native.Tokenizer()},
		gguf:    ggufInfo,
		cleanup: cleanup,
	}, nil
}

func toMetalGenerateConfig(cfg GenerateConfig) metal.GenerateConfig {
	return metal.GenerateConfig{
		MaxTokens:     cfg.MaxTokens,
		Temperature:   cfg.Temperature,
		TopK:          cfg.TopK,
		TopP:          cfg.TopP,
		MinP:          cfg.MinP,
		StopTokens:    cfg.StopTokens,
		RepeatPenalty: cfg.RepeatPenalty,
	}
}

func toRootMetrics(metrics metal.Metrics) Metrics {
	return Metrics{
		PromptTokens:        metrics.PromptTokens,
		GeneratedTokens:     metrics.GeneratedTokens,
		PrefillDuration:     metrics.PrefillDuration,
		DecodeDuration:      metrics.DecodeDuration,
		TotalDuration:       metrics.TotalDuration,
		PrefillTokensPerSec: metrics.PrefillTokensPerSec,
		DecodeTokensPerSec:  metrics.DecodeTokensPerSec,
		PeakMemoryBytes:     metrics.PeakMemoryBytes,
		ActiveMemoryBytes:   metrics.ActiveMemoryBytes,
	}
}

func toRootToken(token metal.Token) Token {
	return Token{ID: token.ID, Value: token.Text, Text: token.Text}
}

func toRootClassifyResults(results []metal.ClassifyResult) []ClassifyResult {
	if len(results) == 0 {
		return nil
	}
	out := make([]ClassifyResult, len(results))
	for i, result := range results {
		out[i] = ClassifyResult{
			Token:  toRootToken(result.Token),
			Logits: append([]float32(nil), result.Logits...),
		}
	}
	return out
}

func toRootBatchResults(results []metal.BatchResult) []BatchResult {
	if len(results) == 0 {
		return nil
	}
	out := make([]BatchResult, len(results))
	for i, result := range results {
		tokens := make([]Token, len(result.Tokens))
		for j, token := range result.Tokens {
			tokens[j] = toRootToken(token)
		}
		out[i] = BatchResult{
			Tokens: tokens,
			Err:    result.Err,
		}
	}
	return out
}

func toRootAttentionSnapshot(result *metal.AttentionResult) *AttentionSnapshot {
	if result == nil {
		return nil
	}
	return &AttentionSnapshot{
		NumLayers:     result.NumLayers,
		NumHeads:      result.NumHeads,
		SeqLen:        result.SeqLen,
		HeadDim:       result.HeadDim,
		NumQueryHeads: result.NumQueryHeads,
		Keys:          result.Keys,
		Queries:       result.Queries,
		Architecture:  result.Architecture,
	}
}

// Generate produces a buffered string result.
func (m *Model) Generate(prompt string, opts ...GenerateOption) (string, error) {
	if m == nil || m.model == nil {
		return "", errors.New("mlx: model is nil")
	}
	cfg := toMetalGenerateConfig(applyGenerateOptions(opts))
	var builder strings.Builder
	for tok := range m.model.Generate(context.Background(), prompt, cfg) {
		builder.WriteString(tok.Text)
	}
	if err := m.model.Err(); err != nil {
		return "", err
	}
	return builder.String(), nil
}

// Chat produces a buffered string result using the model's native chat template.
func (m *Model) Chat(messages []Message, opts ...GenerateOption) (string, error) {
	if m == nil || m.model == nil {
		return "", errors.New("mlx: model is nil")
	}
	cfg := toMetalGenerateConfig(applyGenerateOptions(opts))
	metalMessages := make([]metal.ChatMessage, len(messages))
	for i, msg := range messages {
		metalMessages[i] = metal.ChatMessage{Role: msg.Role, Content: msg.Content}
	}
	var builder strings.Builder
	for tok := range m.model.Chat(context.Background(), metalMessages, cfg) {
		builder.WriteString(tok.Text)
	}
	if err := m.model.Err(); err != nil {
		return "", err
	}
	return builder.String(), nil
}

// GenerateStream streams tokens through a channel.
func (m *Model) GenerateStream(prompt string, opts ...GenerateOption) <-chan Token {
	out := make(chan Token)
	go func() {
		defer close(out)
		if m == nil || m.model == nil {
			return
		}
		cfg := toMetalGenerateConfig(applyGenerateOptions(opts))
		for tok := range m.model.Generate(context.Background(), prompt, cfg) {
			out <- Token{ID: tok.ID, Value: tok.Text, Text: tok.Text}
		}
	}()
	return out
}

// ChatStream streams chat tokens through a channel.
func (m *Model) ChatStream(messages []Message, opts ...GenerateOption) <-chan Token {
	out := make(chan Token)
	go func() {
		defer close(out)
		if m == nil || m.model == nil {
			return
		}
		cfg := toMetalGenerateConfig(applyGenerateOptions(opts))
		metalMessages := make([]metal.ChatMessage, len(messages))
		for i, msg := range messages {
			metalMessages[i] = metal.ChatMessage{Role: msg.Role, Content: msg.Content}
		}
		for tok := range m.model.Chat(context.Background(), metalMessages, cfg) {
			out <- toRootToken(tok)
		}
	}()
	return out
}

// Classify runs batched prefill-only inference over multiple prompts.
func (m *Model) Classify(prompts []string, opts ...GenerateOption) ([]ClassifyResult, error) {
	if m == nil || m.model == nil {
		return nil, errors.New("mlx: model is nil")
	}
	cfg := applyGenerateOptions(opts)
	results, err := m.model.Classify(context.Background(), prompts, toMetalGenerateConfig(cfg), cfg.ReturnLogits)
	if err != nil {
		return nil, err
	}
	return toRootClassifyResults(results), nil
}

// BatchGenerate runs autoregressive generation for multiple prompts at once.
func (m *Model) BatchGenerate(prompts []string, opts ...GenerateOption) ([]BatchResult, error) {
	if m == nil || m.model == nil {
		return nil, errors.New("mlx: model is nil")
	}
	results, err := m.model.BatchGenerate(context.Background(), prompts, toMetalGenerateConfig(applyGenerateOptions(opts)))
	if err != nil {
		return nil, err
	}
	return toRootBatchResults(results), nil
}

// Err returns the last generation error, if any.
func (m *Model) Err() error {
	if m == nil || m.model == nil {
		return nil
	}
	return m.model.Err()
}

// Metrics returns performance counters from the last inference call.
func (m *Model) Metrics() Metrics {
	if m == nil || m.model == nil {
		return Metrics{}
	}
	return toRootMetrics(m.model.LastMetrics())
}

// ModelType returns the internal architecture identifier.
func (m *Model) ModelType() string {
	if m == nil || m.model == nil {
		return ""
	}
	return m.model.ModelType()
}

// Info returns metadata about the loaded model.
func (m *Model) Info() ModelInfo {
	if m == nil || m.model == nil {
		return ModelInfo{}
	}
	info := m.model.Info()
	contextLength := info.ContextLength
	if m.cfg.ContextLength > 0 {
		contextLength = m.cfg.ContextLength
	}
	architecture := info.Architecture
	numLayers := info.NumLayers
	quantBits := info.QuantBits
	quantGroup := info.QuantGroup
	if m.gguf != nil {
		if architecture == "" {
			architecture = m.gguf.Architecture
		}
		if numLayers == 0 {
			numLayers = m.gguf.NumLayers
		}
		if quantBits == 0 {
			quantBits = m.gguf.QuantBits
		}
		if quantGroup == 0 {
			quantGroup = m.gguf.QuantGroup
		}
	}
	return ModelInfo{
		Architecture:  architecture,
		VocabSize:     info.VocabSize,
		NumLayers:     numLayers,
		HiddenSize:    info.HiddenSize,
		QuantBits:     quantBits,
		QuantGroup:    quantGroup,
		ContextLength: contextLength,
	}
}

// InspectAttention runs a single prefill pass and returns extracted K tensors.
func (m *Model) InspectAttention(prompt string) (*AttentionSnapshot, error) {
	if m == nil || m.model == nil {
		return nil, errors.New("mlx: model is nil")
	}
	result, err := m.model.InspectAttention(context.Background(), prompt)
	if err != nil {
		return nil, err
	}
	return toRootAttentionSnapshot(result), nil
}

// Tokenizer returns the model tokenizer.
func (m *Model) Tokenizer() *Tokenizer {
	if m == nil {
		return nil
	}
	return m.tok
}

// Close releases model resources.
func (m *Model) Close() error {
	if m == nil || m.model == nil {
		if m != nil && m.cleanup != nil {
			err := m.cleanup()
			m.cleanup = nil
			return err
		}
		return nil
	}
	native := m.model
	m.model = nil
	m.tok = nil
	err := native.Close()
	if m.cleanup != nil {
		err = errors.Join(err, m.cleanup())
		m.cleanup = nil
	}
	return err
}

// NewLoRA applies a LoRA adapter to a loaded model.
func NewLoRA(model *Model, cfg *LoRAConfig) *LoRAAdapter {
	if model == nil || model.model == nil {
		return nil
	}
	mcfg := DefaultLoRAConfig()
	if cfg != nil {
		mcfg = *cfg
	}
	return model.model.ApplyLoRA(mcfg)
}

// MergeLoRA returns the current model with the adapter applied in-place.
func (m *Model) MergeLoRA(adapter *LoRAAdapter) *Model {
	if adapter == nil {
		return m
	}
	adapter.Merge()
	return m
}

// MatMul returns the matrix product of a and b.
func MatMul(a, b *Array) *Array { return metal.Matmul(a, b) }

// Add returns element-wise a + b.
func Add(a, b *Array) *Array { return metal.Add(a, b) }

// Mul returns element-wise a * b.
func Mul(a, b *Array) *Array { return metal.Mul(a, b) }

// Softmax returns softmax along the last axis.
func Softmax(a *Array) *Array { return metal.Softmax(a) }

// Slice extracts a sub-array along a single axis.
func Slice(a *Array, start, end, axis any) *Array {
	return metal.SliceAxis(
		a,
		normalizeRootIntArg("axis", axis),
		normalizeRootInt32Arg("start", start),
		normalizeRootInt32Arg("end", end),
	)
}

// Reshape returns a view with the given shape.
func Reshape(a *Array, shape ...any) *Array {
	return metal.Reshape(a, normalizeRootShapeArgs(shape)...)
}

// VJP computes the vector-Jacobian product.
func VJP(fn func([]*Array) []*Array, primals []*Array, cotangents []*Array) (outputs []*Array, vjps []*Array, err error) {
	return metal.VJP(fn, primals, cotangents)
}

// JVP computes the Jacobian-vector product.
func JVP(fn func([]*Array) []*Array, primals []*Array, tangents []*Array) (outputs []*Array, jvps []*Array, err error) {
	return metal.JVP(fn, primals, tangents)
}

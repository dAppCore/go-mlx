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
	Close() error
	Err() error
	Generate(context.Context, string, metal.GenerateConfig) iter.Seq[metal.Token]
	Info() metal.ModelInfo
	Tokenizer() *metal.Tokenizer
}

// Model is the RFC-style root-package model handle.
type Model struct {
	model   nativeModel
	cfg     LoadConfig
	tok     *Tokenizer
	cleanup func() error
}

var loadNativeModel = func(modelPath string, cfg metal.LoadConfig) (nativeModel, error) {
	return metal.LoadAndInit(modelPath, cfg)
}

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
	if cfg.Quantization > 0 && info.QuantBits != cfg.Quantization {
		_ = native.Close()
		_ = cleanup()
		return nil, errors.New("mlx: loaded model quantization does not match requested bits")
	}

	return &Model{
		model:   native,
		cfg:     cfg,
		tok:     &Tokenizer{tok: native.Tokenizer()},
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

// Err returns the last generation error, if any.
func (m *Model) Err() error {
	if m == nil || m.model == nil {
		return nil
	}
	return m.model.Err()
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
	return ModelInfo{
		Architecture:  info.Architecture,
		VocabSize:     info.VocabSize,
		NumLayers:     info.NumLayers,
		HiddenSize:    info.HiddenSize,
		QuantBits:     info.QuantBits,
		QuantGroup:    info.QuantGroup,
		ContextLength: contextLength,
	}
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
func Slice(a *Array, start, end int32, axis int) *Array { return metal.SliceAxis(a, axis, start, end) }

// Reshape returns a view with the given shape.
func Reshape(a *Array, shape ...int32) *Array { return metal.Reshape(a, shape...) }

// VJP computes the vector-Jacobian product.
func VJP(fn func([]*Array) []*Array, primals []*Array, cotangents []*Array) (outputs []*Array, vjps []*Array, err error) {
	return metal.VJP(fn, primals, cotangents)
}

// JVP computes the Jacobian-vector product.
func JVP(fn func([]*Array) []*Array, primals []*Array, tangents []*Array) (outputs []*Array, jvps []*Array, err error) {
	return metal.JVP(fn, primals, tangents)
}

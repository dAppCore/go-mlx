// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"
	"errors"
	"strings"

	"dappco.re/go/core/inference"
)

// Message aliases inference.Message for the adapter-style API.
type Message = inference.Message

// GenOpts controls buffered adapter generation.
type GenOpts struct {
	MaxTokens int
	Temp      float64
}

// Result holds buffered text plus optional backend metrics.
type Result struct {
	Text    string
	Metrics *inference.GenerateMetrics
}

// TokenCallback receives streamed token text.
type TokenCallback func(token string) error

// InferenceAdapter wraps an inference.TextModel with buffered/string APIs.
type InferenceAdapter struct {
	model inference.TextModel
	name  string
}

// NewInferenceAdapter wraps a loaded inference model with an adapter surface.
func NewInferenceAdapter(model inference.TextModel, name string) *InferenceAdapter {
	return &InferenceAdapter{model: model, name: name}
}

// NewMLXBackend loads the Metal backend and wraps it in an InferenceAdapter.
func NewMLXBackend(modelPath string, loadOpts ...inference.LoadOption) (*InferenceAdapter, error) {
	opts := append(append([]inference.LoadOption(nil), loadOpts...), inference.WithBackend("metal"))
	model, err := inference.LoadModel(modelPath, opts...)
	if err != nil {
		return nil, err
	}
	return NewInferenceAdapter(model, "mlx"), nil
}

// Name returns the configured adapter name.
func (adapter *InferenceAdapter) Name() string {
	if adapter == nil {
		return ""
	}
	return adapter.name
}

// Available reports whether the underlying model is loaded.
func (adapter *InferenceAdapter) Available() bool {
	return adapter != nil && adapter.model != nil
}

// Model returns the wrapped inference.TextModel.
func (adapter *InferenceAdapter) Model() inference.TextModel {
	if adapter == nil {
		return nil
	}
	return adapter.model
}

// Close releases the underlying model.
func (adapter *InferenceAdapter) Close() error {
	if adapter == nil || adapter.model == nil {
		return nil
	}
	model := adapter.model
	adapter.model = nil
	return model.Close()
}

// Generate collects a streamed response into a single string.
func (adapter *InferenceAdapter) Generate(ctx context.Context, prompt string, opts GenOpts) (Result, error) {
	if adapter == nil || adapter.model == nil {
		return Result{}, errors.New("mlx: inference adapter is nil")
	}

	var builder strings.Builder
	for token := range adapter.model.Generate(ctx, prompt, genOptsToInference(opts)...) {
		builder.WriteString(token.Text)
	}
	if err := adapter.model.Err(); err != nil {
		return Result{Text: builder.String()}, err
	}

	metrics := adapter.model.Metrics()
	return Result{
		Text:    builder.String(),
		Metrics: &metrics,
	}, nil
}

// GenerateStream forwards token text to a callback.
func (adapter *InferenceAdapter) GenerateStream(ctx context.Context, prompt string, opts GenOpts, cb TokenCallback) error {
	if adapter == nil || adapter.model == nil {
		return errors.New("mlx: inference adapter is nil")
	}
	if cb == nil {
		return errors.New("mlx: token callback is nil")
	}

	var callbackErr error
	for token := range adapter.model.Generate(ctx, prompt, genOptsToInference(opts)...) {
		if err := cb(token.Text); err != nil {
			callbackErr = err
			break
		}
	}
	if callbackErr != nil {
		return callbackErr
	}
	return adapter.model.Err()
}

// Chat collects a streamed chat response into a single string.
func (adapter *InferenceAdapter) Chat(ctx context.Context, messages []Message, opts GenOpts) (Result, error) {
	if adapter == nil || adapter.model == nil {
		return Result{}, errors.New("mlx: inference adapter is nil")
	}

	var builder strings.Builder
	for token := range adapter.model.Chat(ctx, messages, genOptsToInference(opts)...) {
		builder.WriteString(token.Text)
	}
	if err := adapter.model.Err(); err != nil {
		return Result{Text: builder.String()}, err
	}

	metrics := adapter.model.Metrics()
	return Result{
		Text:    builder.String(),
		Metrics: &metrics,
	}, nil
}

// ChatStream forwards chat token text to a callback.
func (adapter *InferenceAdapter) ChatStream(ctx context.Context, messages []Message, opts GenOpts, cb TokenCallback) error {
	if adapter == nil || adapter.model == nil {
		return errors.New("mlx: inference adapter is nil")
	}
	if cb == nil {
		return errors.New("mlx: token callback is nil")
	}

	var callbackErr error
	for token := range adapter.model.Chat(ctx, messages, genOptsToInference(opts)...) {
		if err := cb(token.Text); err != nil {
			callbackErr = err
			break
		}
	}
	if callbackErr != nil {
		return callbackErr
	}
	return adapter.model.Err()
}

// InspectAttention delegates to the underlying model when supported.
func (adapter *InferenceAdapter) InspectAttention(ctx context.Context, prompt string, opts ...inference.GenerateOption) (*inference.AttentionSnapshot, error) {
	if adapter == nil || adapter.model == nil {
		return nil, errors.New("mlx: inference adapter is nil")
	}
	inspector, ok := adapter.model.(inference.AttentionInspector)
	if !ok {
		return nil, errors.New("mlx: wrapped model does not support attention inspection")
	}
	return inspector.InspectAttention(ctx, prompt, opts...)
}

func genOptsToInference(opts GenOpts) []inference.GenerateOption {
	var generateOpts []inference.GenerateOption
	if opts.MaxTokens > 0 {
		generateOpts = append(generateOpts, inference.WithMaxTokens(opts.MaxTokens))
	}
	if opts.Temp > 0 {
		generateOpts = append(generateOpts, inference.WithTemperature(float32(opts.Temp)))
	}
	return generateOpts
}

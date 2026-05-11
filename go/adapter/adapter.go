// SPDX-Licence-Identifier: EUPL-1.2

// Package adapter wraps an inference.TextModel with buffered + streaming
// callback APIs.
//
//	a := adapter.New(model, "mlx")
//	result, _ := a.Generate(ctx, prompt, adapter.GenOpts{MaxTokens: 128})
package adapter

import (
	"context"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

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

// Adapter wraps an inference.TextModel with buffered/string APIs.
type Adapter struct {
	model inference.TextModel
	name  string
}

// New wraps a loaded inference model with an adapter surface.
//
//	a := adapter.New(model, "mlx")
func New(model inference.TextModel, name string) *Adapter {
	return &Adapter{model: model, name: name}
}

// Name returns the configured adapter name.
func (a *Adapter) Name() string {
	if a == nil {
		return ""
	}
	return a.name
}

// Available reports whether the underlying model is loaded.
func (a *Adapter) Available() bool {
	return a != nil && a.model != nil
}

// Model returns the wrapped inference.TextModel.
func (a *Adapter) Model() inference.TextModel {
	if a == nil {
		return nil
	}
	return a.model
}

// Close releases the underlying model.
func (a *Adapter) Close() error {
	if a == nil || a.model == nil {
		return nil
	}
	model := a.model
	a.model = nil
	return model.Close()
}

// Generate collects a streamed response into a single string.
//
//	result, err := a.Generate(ctx, "prompt", adapter.GenOpts{MaxTokens: 64})
func (a *Adapter) Generate(ctx context.Context, prompt string, opts GenOpts) (Result, error) {
	if a == nil || a.model == nil {
		return Result{}, core.NewError("adapter: inference adapter is nil")
	}
	if ctx == nil {
		ctx = context.Background()
	}

	builder := core.NewBuilder()
	for token := range a.model.Generate(ctx, prompt, genOptsToInference(opts)...) {
		builder.WriteString(token.Text)
	}
	if err := a.model.Err(); err != nil {
		return Result{Text: builder.String()}, err
	}

	metrics := a.model.Metrics()
	return Result{Text: builder.String(), Metrics: &metrics}, nil
}

// GenerateStream forwards token text to a callback.
func (a *Adapter) GenerateStream(ctx context.Context, prompt string, opts GenOpts, cb TokenCallback) error {
	if a == nil || a.model == nil {
		return core.NewError("adapter: inference adapter is nil")
	}
	if cb == nil {
		return core.NewError("adapter: token callback is nil")
	}
	if ctx == nil {
		ctx = context.Background()
	}
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	var callbackErr error
	tokens := a.model.Generate(ctx, prompt, genOptsToInference(opts)...)
	for token := range tokens {
		if callbackErr != nil {
			continue
		}
		if err := cb(token.Text); err != nil {
			callbackErr = err
			cancel()
		}
	}
	if callbackErr != nil {
		return callbackErr
	}
	return a.model.Err()
}

// Chat collects a streamed chat response into a single string.
//
//	result, err := a.Chat(ctx, messages, adapter.GenOpts{})
func (a *Adapter) Chat(ctx context.Context, messages []inference.Message, opts GenOpts) (Result, error) {
	if a == nil || a.model == nil {
		return Result{}, core.NewError("adapter: inference adapter is nil")
	}
	if ctx == nil {
		ctx = context.Background()
	}

	builder := core.NewBuilder()
	for token := range a.model.Chat(ctx, messages, genOptsToInference(opts)...) {
		builder.WriteString(token.Text)
	}
	if err := a.model.Err(); err != nil {
		return Result{Text: builder.String()}, err
	}

	metrics := a.model.Metrics()
	return Result{Text: builder.String(), Metrics: &metrics}, nil
}

// ChatStream forwards chat token text to a callback.
func (a *Adapter) ChatStream(ctx context.Context, messages []inference.Message, opts GenOpts, cb TokenCallback) error {
	if a == nil || a.model == nil {
		return core.NewError("adapter: inference adapter is nil")
	}
	if cb == nil {
		return core.NewError("adapter: token callback is nil")
	}
	if ctx == nil {
		ctx = context.Background()
	}
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	var callbackErr error
	tokens := a.model.Chat(ctx, messages, genOptsToInference(opts)...)
	for token := range tokens {
		if callbackErr != nil {
			continue
		}
		if err := cb(token.Text); err != nil {
			callbackErr = err
			cancel()
		}
	}
	if callbackErr != nil {
		return callbackErr
	}
	return a.model.Err()
}

// InspectAttention delegates to the underlying model when supported.
func (a *Adapter) InspectAttention(ctx context.Context, prompt string, opts ...inference.GenerateOption) (*inference.AttentionSnapshot, error) {
	if a == nil || a.model == nil {
		return nil, core.NewError("adapter: inference adapter is nil")
	}
	inspector, ok := a.model.(inference.AttentionInspector)
	if !ok {
		return nil, core.NewError("adapter: wrapped model does not support attention inspection")
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

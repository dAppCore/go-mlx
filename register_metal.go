//go:build darwin && arm64

package mlx

import (
	"context"
	"iter"

	"forge.lthn.ai/core/go-mlx/internal/metal"
)

func init() {
	Register(&metalBackend{})
}

// MetalAvailable reports whether native Metal inference is available.
func MetalAvailable() bool { return true }

// Hardware-level memory controls — delegate to internal/metal.
// These are not model-level; they control the Metal allocator directly.

// SetCacheLimit sets the Metal memory cache limit. Returns the previous value.
func SetCacheLimit(limit uint64) uint64 { return metal.SetCacheLimit(limit) }

// SetMemoryLimit sets the Metal memory limit. Returns the previous value.
func SetMemoryLimit(limit uint64) uint64 { return metal.SetMemoryLimit(limit) }

// GetActiveMemory returns the current active Metal memory usage in bytes.
func GetActiveMemory() uint64 { return metal.GetActiveMemory() }

// GetPeakMemory returns the peak Metal memory usage in bytes.
func GetPeakMemory() uint64 { return metal.GetPeakMemory() }

// ClearCache clears the Metal memory cache.
func ClearCache() { metal.ClearCache() }

// metalBackend implements Backend for native Metal inference.
type metalBackend struct{}

func (b *metalBackend) Name() string { return "metal" }

func (b *metalBackend) LoadModel(path string, opts ...LoadOption) (TextModel, error) {
	m, err := metal.LoadAndInit(path)
	if err != nil {
		return nil, err
	}
	return &metalAdapter{m: m}, nil
}

// metalAdapter wraps metal.Model to implement TextModel.
type metalAdapter struct {
	m *metal.Model
}

func (a *metalAdapter) Generate(ctx context.Context, prompt string, opts ...GenerateOption) iter.Seq[Token] {
	cfg := ApplyGenerateOpts(opts)
	mcfg := metal.GenerateConfig{
		MaxTokens:   cfg.MaxTokens,
		Temperature: cfg.Temperature,
		TopK:        cfg.TopK,
		TopP:        cfg.TopP,
		StopTokens:  cfg.StopTokens,
	}
	return func(yield func(Token) bool) {
		for tok := range a.m.Generate(ctx, prompt, mcfg) {
			if !yield(Token{ID: tok.ID, Text: tok.Text}) {
				return
			}
		}
	}
}

func (a *metalAdapter) Chat(ctx context.Context, messages []Message, opts ...GenerateOption) iter.Seq[Token] {
	cfg := ApplyGenerateOpts(opts)
	mcfg := metal.GenerateConfig{
		MaxTokens:   cfg.MaxTokens,
		Temperature: cfg.Temperature,
		TopK:        cfg.TopK,
		TopP:        cfg.TopP,
		StopTokens:  cfg.StopTokens,
	}
	// Convert messages
	mmsgs := make([]metal.ChatMessage, len(messages))
	for i, msg := range messages {
		mmsgs[i] = metal.ChatMessage{Role: msg.Role, Content: msg.Content}
	}
	return func(yield func(Token) bool) {
		for tok := range a.m.Chat(ctx, mmsgs, mcfg) {
			if !yield(Token{ID: tok.ID, Text: tok.Text}) {
				return
			}
		}
	}
}

func (a *metalAdapter) ModelType() string { return a.m.ModelType() }
func (a *metalAdapter) Err() error        { return a.m.Err() }
func (a *metalAdapter) Close() error      { return a.m.Close() }

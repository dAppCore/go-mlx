package mlx

import (
	"context"
	"iter"
)

// Token represents a single generated token for streaming.
type Token struct {
	ID   int32
	Text string
}

// Message represents a chat turn for Chat().
type Message struct {
	Role    string // "user", "assistant", "system"
	Content string
}

// TextModel generates text from a loaded model.
type TextModel interface {
	// Generate streams tokens for the given prompt.
	// Respects ctx cancellation (HTTP handlers, timeouts, graceful shutdown).
	Generate(ctx context.Context, prompt string, opts ...GenerateOption) iter.Seq[Token]

	// Chat formats messages using the model's native chat template, then generates.
	// The model owns its template — callers don't need to know Gemma vs Qwen formatting.
	Chat(ctx context.Context, messages []Message, opts ...GenerateOption) iter.Seq[Token]

	// ModelType returns the architecture identifier (e.g. "gemma3", "qwen3").
	ModelType() string

	// Err returns the error from the last Generate/Chat call, if any.
	// Distinguishes normal stop (EOS, max tokens) from failures (OOM, C-level error).
	// Returns nil if generation completed normally.
	Err() error

	// Close releases all resources (GPU memory, caches, subprocess).
	Close() error
}

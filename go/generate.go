// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"
	"iter"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/parser"
)

// generate.go: the Model text-generation API — buffered Generate/Chat/GenerateChunks,
// the token-sequence internals, public token iterators, streaming channels, and
// Classify/BatchGenerate.

// Generate produces a buffered string result.
func (m *Model) Generate(prompt string, opts ...GenerateOption) (string, error) {
	if m == nil || m.model == nil {
		return "", errMLXModelNil
	}
	cfg := applyGenerateOptions(opts)
	builder := core.NewBuilder()
	// Pre-grow for the expected output footprint — MaxTokens caps the
	// emitted token stream and 4 bytes/token is a conservative average
	// across ASCII + short BPE pieces, matching the FilterThinkingTokens
	// sizing heuristic in thinking.go. Grow(0) is a no-op when MaxTokens
	// is unset.
	builder.Grow(cfg.MaxTokens * 4)
	for tok := range m.generateTokensWithConfig(context.Background(), prompt, cfg) {
		builder.WriteString(tok.Text)
	}
	if err := m.model.Err(); err != nil {
		return "", err
	}
	return builder.String(), nil
}

// Chat produces a buffered string result using the model's native chat template.
func (m *Model) Chat(messages []inference.Message, opts ...GenerateOption) (string, error) {
	if m == nil || m.model == nil {
		return "", errMLXModelNil
	}
	cfg := applyGenerateOptions(opts)
	builder := core.NewBuilder()
	// Pre-grow for MaxTokens × 4-byte average — same heuristic as the
	// FilterThinkingTokens decoder and Model.Generate above.
	builder.Grow(cfg.MaxTokens * 4)
	for tok := range m.chatTokensWithConfig(context.Background(), messages, cfg) {
		builder.WriteString(tok.Text)
	}
	if err := m.model.Err(); err != nil {
		return "", err
	}
	return builder.String(), nil
}

// GenerateChunks produces a buffered string result from streaming prompt chunks.
// Chunked prompts avoid one giant tokenizer call while preserving one logical
// prompt token stream for cache matching and KV capture.
func (m *Model) GenerateChunks(ctx context.Context, chunks iter.Seq[string], opts ...GenerateOption) (string, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if m == nil || m.model == nil {
		return "", errMLXModelNil
	}
	cfg := applyGenerateOptions(opts)
	builder := core.NewBuilder()
	// Same MaxTokens × 4 pre-grow as Generate/Chat above — keeps the
	// chunked path on the same allocation budget as the giant-string
	// path it falls back to.
	builder.Grow(cfg.MaxTokens * 4)
	for tok := range m.generateChunkTokensWithConfig(ctx, chunks, cfg) {
		builder.WriteString(tok.Text)
	}
	if err := m.model.Err(); err != nil {
		return "", err
	}
	return builder.String(), nil
}

func (m *Model) generateTokensWithConfig(ctx context.Context, prompt string, cfg GenerateConfig) iter.Seq[Token] {
	if ctx == nil {
		ctx = context.Background()
	}
	filter := parser.NewProcessor(cfg.Thinking, m.hintForParser())
	return filteredRootTokenSeq(m.model.Generate(ctx, prompt, toMetalGenerateConfig(cfg)), filter)
}

func (m *Model) generateChunkTokensWithConfig(ctx context.Context, chunks iter.Seq[string], cfg GenerateConfig) iter.Seq[Token] {
	if ctx == nil {
		ctx = context.Background()
	}
	filter := parser.NewProcessor(cfg.Thinking, m.hintForParser())
	if generator, ok := m.model.(nativeChunkGenerator); ok {
		return filteredRootTokenSeq(generator.GenerateChunks(ctx, chunks, toMetalGenerateConfig(cfg)), filter)
	}
	return filteredRootTokenSeq(m.model.Generate(ctx, promptChunksToString(chunks), toMetalGenerateConfig(cfg)), filter)
}

func (m *Model) chatTokensWithConfig(ctx context.Context, messages []inference.Message, cfg GenerateConfig) iter.Seq[Token] {
	if ctx == nil {
		ctx = context.Background()
	}
	filter := parser.NewProcessor(cfg.Thinking, m.hintForParser())
	metalMessages := chatMessagesAsMetal(messages)
	return filteredRootTokenSeq(m.model.Chat(ctx, metalMessages, toMetalGenerateConfig(cfg)), filter)
}

func (m *Model) chatChunkTokensWithConfig(ctx context.Context, messages []inference.Message, chunkBytes int, cfg GenerateConfig) iter.Seq[Token] {
	if ctx == nil {
		ctx = context.Background()
	}
	filter := parser.NewProcessor(cfg.Thinking, m.hintForParser())
	metalMessages := chatMessagesAsMetal(messages)
	if generator, ok := m.model.(nativeChatChunkGenerator); ok {
		return filteredRootTokenSeq(generator.ChatChunks(ctx, metalMessages, chunkBytes, toMetalGenerateConfig(cfg)), filter)
	}
	return filteredRootTokenSeq(m.model.Chat(ctx, metalMessages, toMetalGenerateConfig(cfg)), filter)
}

// GenerateTokens streams tokens directly as an iterator. It is the no-goroutine
// path used by profiling and other in-process consumers that do not need a
// channel boundary.
func (m *Model) GenerateTokens(ctx context.Context, prompt string, opts ...GenerateOption) iter.Seq[Token] {
	if m == nil || m.model == nil {
		return emptyTokenSeq()
	}
	return m.generateTokensWithConfig(ctx, prompt, applyGenerateOptions(opts))
}

// GenerateChunkTokens streams tokens from bounded prompt chunks as an iterator.
func (m *Model) GenerateChunkTokens(ctx context.Context, chunks iter.Seq[string], opts ...GenerateOption) iter.Seq[Token] {
	if m == nil || m.model == nil {
		return emptyTokenSeq()
	}
	return m.generateChunkTokensWithConfig(ctx, chunks, applyGenerateOptions(opts))
}

// ChatTokens streams chat tokens through the model template as an iterator.
func (m *Model) ChatTokens(ctx context.Context, messages []inference.Message, opts ...GenerateOption) iter.Seq[Token] {
	if m == nil || m.model == nil {
		return emptyTokenSeq()
	}
	return m.chatTokensWithConfig(ctx, messages, applyGenerateOptions(opts))
}

// ChatChunkTokens streams chat tokens from bounded prompt chunks as an iterator.
func (m *Model) ChatChunkTokens(ctx context.Context, messages []inference.Message, chunkBytes int, opts ...GenerateOption) iter.Seq[Token] {
	if m == nil || m.model == nil {
		return emptyTokenSeq()
	}
	return m.chatChunkTokensWithConfig(ctx, messages, chunkBytes, applyGenerateOptions(opts))
}

func tokenSeqChannel(ctx context.Context, seq iter.Seq[Token]) <-chan Token {
	if ctx == nil {
		ctx = context.Background()
	}
	out := make(chan Token)
	go func() {
		defer close(out)
		for tok := range seq {
			select {
			case out <- tok:
			case <-ctx.Done():
				return
			}
		}
	}()
	return out
}

// GenerateStream streams tokens through a channel until generation completes or ctx is cancelled.
func (m *Model) GenerateStream(ctx context.Context, prompt string, opts ...GenerateOption) <-chan Token {
	if m == nil || m.model == nil {
		return closedTokenChan
	}
	return tokenSeqChannel(ctx, m.GenerateTokens(ctx, prompt, opts...))
}

// GenerateChunksStream streams tokens from bounded prompt chunks without
// building or tokenizing one giant prompt string.
func (m *Model) GenerateChunksStream(ctx context.Context, chunks iter.Seq[string], opts ...GenerateOption) <-chan Token {
	if m == nil || m.model == nil {
		return closedTokenChan
	}
	return tokenSeqChannel(ctx, m.GenerateChunkTokens(ctx, chunks, opts...))
}

// ChatChunksStream streams chat tokens through the native template while
// feeding long message content as bounded prompt chunks.
func (m *Model) ChatChunksStream(ctx context.Context, messages []inference.Message, chunkBytes int, opts ...GenerateOption) <-chan Token {
	if m == nil || m.model == nil {
		return closedTokenChan
	}
	return tokenSeqChannel(ctx, m.ChatChunkTokens(ctx, messages, chunkBytes, opts...))
}

// ChatStream streams chat tokens through a channel until generation completes or ctx is cancelled.
func (m *Model) ChatStream(ctx context.Context, messages []inference.Message, opts ...GenerateOption) <-chan Token {
	if m == nil || m.model == nil {
		return closedTokenChan
	}
	return tokenSeqChannel(ctx, m.ChatTokens(ctx, messages, opts...))
}

// Classify runs batched prefill-only inference over multiple prompts.
func (m *Model) Classify(prompts []string, opts ...GenerateOption) ([]ClassifyResult, error) {
	if m == nil || m.model == nil {
		return nil, errMLXModelNil
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
		return nil, errMLXModelNil
	}
	results, err := m.model.BatchGenerate(context.Background(), prompts, toMetalGenerateConfig(applyGenerateOptions(opts)))
	if err != nil {
		return nil, err
	}
	return toRootBatchResults(results), nil
}

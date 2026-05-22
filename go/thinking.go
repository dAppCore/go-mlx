// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	core "dappco.re/go"
	"dappco.re/go/inference/parser"
)

// errMLXTokenizerNil fires from FilterThinkingTokens whenever the caller
// hands in a zero-value or already-closed Tokenizer — hoisted to package
// level so the precondition slot costs no per-call core.NewError alloc.
var errMLXTokenizerNil = core.NewError("mlx: tokenizer is nil")

// Pre-allocated closures for the constant-mode Show/Hide/Capture shortcuts —
// the previous WithShowThinking / WithHideThinking helpers built a
// fresh capturing closure on every call (24 B/op, 1 alloc). With
// mode fixed, share a single GenerateOption value across all calls.
// withCaptureModeFn covers WithThinkingMode(parser.Capture) — the
// dedicated WithCaptureThinking variant still allocates a closure
// because it also wires the per-call capture callback.
var (
	withShowThinkingFn = func(c *GenerateConfig) { c.Thinking.Mode = parser.Show }
	withHideThinkingFn = func(c *GenerateConfig) { c.Thinking.Mode = parser.Hide }
	withCaptureModeFn  = func(c *GenerateConfig) { c.Thinking.Mode = parser.Capture }
)

// c.Generate(ctx, prompt, mlx.WithThinkingMode(parser.Capture))
//
// The three known parser.Mode values reuse the static Show/Hide/Capture
// closures — drops the 24 B per-call closure alloc for the common path.
// Unknown/future modes (including the zero-value "") fall through to a
// fresh closure so the API still preserves the per-call mode write.
func WithThinkingMode(mode parser.Mode) GenerateOption {
	switch mode {
	case parser.Show:
		return withShowThinkingFn
	case parser.Hide:
		return withHideThinkingFn
	case parser.Capture:
		return withCaptureModeFn
	}
	return func(c *GenerateConfig) { c.Thinking.Mode = mode }
}

// c.Generate(ctx, prompt, mlx.WithShowThinking())
func WithShowThinking() GenerateOption { return withShowThinkingFn }

// c.Generate(ctx, prompt, mlx.WithHideThinking())
func WithHideThinking() GenerateOption { return withHideThinkingFn }

// c.Generate(ctx, prompt, mlx.WithCaptureThinking(func(c parser.Chunk) { ... }))
func WithCaptureThinking(capture func(parser.Chunk)) GenerateOption {
	return func(c *GenerateConfig) {
		c.Thinking.Mode = parser.Capture
		c.Thinking.Capture = capture
	}
}

// c.Generate(ctx, prompt, mlx.WithThinkingCapture(func(c parser.Chunk) { ... }))
func WithThinkingCapture(capture func(parser.Chunk)) GenerateOption {
	return WithCaptureThinking(capture)
}

// out, _ := mlx.FilterThinkingTokens(tok, ids, parser.Config{Mode: parser.Capture}, info)
// visible := out.Text
func FilterThinkingTokens(tok *Tokenizer, ids []int32, cfg parser.Config, info ModelInfo) (parser.Result, error) {
	if tok == nil || tok.tok == nil {
		return parser.Result{}, errMLXTokenizerNil
	}
	processor := parser.NewProcessor(cfg, parserHint(info))
	builder := core.NewBuilder()
	// Pre-grow the builder for the expected output footprint —
	// 4 bytes/token is a conservative average that covers ASCII +
	// most short BPE pieces, so we sidestep the initial capacity
	// doublings the un-sized builder otherwise pays as the loop
	// streams pieces in. Grow(0) is a no-op when ids is empty.
	builder.Grow(len(ids) * 4)
	// Hoist the one-element scratch slice for fallback decode out of
	// the loop — the previous []int32{id} literal escaped to the heap
	// on every fallback iteration, even when IDToken hits the inverse
	// vocab path most steps.
	scratch := [1]int32{}
	for _, id := range ids {
		piece := tok.IDToken(id)
		if piece == "" {
			scratch[0] = id
			decoded, err := tok.Decode(scratch[:])
			if err != nil {
				return parser.Result{}, err
			}
			piece = decoded
		}
		builder.WriteString(processor.Process(piece))
	}
	builder.WriteString(processor.Flush())
	return parser.Result{
		Text:      builder.String(),
		Reasoning: processor.Reasoning(),
		Chunks:    processor.Chunks(),
	}, nil
}

// hint := parserHint(model.Info())
func parserHint(info ModelInfo) parser.Hint {
	return parser.Hint{
		Architecture: info.Architecture,
		AdapterName:  info.Adapter.Name,
	}
}

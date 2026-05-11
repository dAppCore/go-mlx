// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	core "dappco.re/go"
	"dappco.re/go/inference/parser"
)

//	c.Generate(ctx, prompt, mlx.WithThinkingMode(parser.Capture))
func WithThinkingMode(mode parser.Mode) GenerateOption {
	return func(c *GenerateConfig) { c.Thinking.Mode = mode }
}

//	c.Generate(ctx, prompt, mlx.WithShowThinking())
func WithShowThinking() GenerateOption { return WithThinkingMode(parser.Show) }

//	c.Generate(ctx, prompt, mlx.WithHideThinking())
func WithHideThinking() GenerateOption { return WithThinkingMode(parser.Hide) }

//	c.Generate(ctx, prompt, mlx.WithCaptureThinking(func(c parser.Chunk) { ... }))
func WithCaptureThinking(capture func(parser.Chunk)) GenerateOption {
	return func(c *GenerateConfig) {
		c.Thinking.Mode = parser.Capture
		c.Thinking.Capture = capture
	}
}

//	c.Generate(ctx, prompt, mlx.WithThinkingCapture(func(c parser.Chunk) { ... }))
func WithThinkingCapture(capture func(parser.Chunk)) GenerateOption {
	return WithCaptureThinking(capture)
}

//	out, _ := mlx.FilterThinkingTokens(tok, ids, parser.Config{Mode: parser.Capture}, info)
//	visible := out.Text
func FilterThinkingTokens(tok *Tokenizer, ids []int32, cfg parser.Config, info ModelInfo) (parser.Result, error) {
	if tok == nil || tok.tok == nil {
		return parser.Result{}, core.NewError("mlx: tokenizer is nil")
	}
	processor := parser.NewProcessor(cfg, parserHint(info))
	builder := core.NewBuilder()
	for _, id := range ids {
		piece := tok.IDToken(id)
		if piece == "" {
			decoded, err := tok.Decode([]int32{id})
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

//	hint := parserHint(model.Info())
func parserHint(info ModelInfo) parser.Hint {
	return parser.Hint{
		Architecture: info.Architecture,
		AdapterName:  info.Adapter.Name,
	}
}

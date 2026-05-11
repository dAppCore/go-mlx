// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"

	"dappco.re/go/inference/decode"
)

// Legacy type aliases — decode lives at go-inference/decode/. The
// Result + Metrics types are structurally identical between mlx and
// decode so we alias them directly. The function + generation types
// stay mlx-shaped because callers build them with mlx.GenerateConfig +
// mlx.Token; the boundary converters below bridge to decode.* at call
// time.
type (
	DecodeOptimisationResult  = decode.Result
	DecodeOptimisationMetrics = decode.Metrics
)

// Mode constants forwarded from the decode package.
const (
	DecodeModeSpeculative  = decode.ModeSpeculative
	DecodeModePromptLookup = decode.ModePromptLookup
)

// DecodeGenerateFunc is the mlx-shaped generation hook used by
// speculative + prompt-lookup decode. Drivers return mlx-native
// DecodeGeneration; RunSpeculativeDecode/RunPromptLookupDecode convert
// to decode.Generation at the boundary.
type DecodeGenerateFunc func(context.Context, string, GenerateConfig) (DecodeGeneration, error)

// DecodeGeneration is a tokenised generation result used by speculative
// and prompt-lookup decode experiments. Decode itself only reads
// Tokens; Text + Metrics are passed through for caller reporting.
type DecodeGeneration struct {
	Tokens  []Token `json:"tokens,omitempty"`
	Text    string  `json:"text,omitempty"`
	Metrics Metrics `json:"metrics,omitempty"`
}

// SpeculativeDecodeConfig is the mlx-shaped speculative decode brief.
type SpeculativeDecodeConfig struct {
	Prompt         string             `json:"prompt,omitempty"`
	MaxTokens      int                `json:"max_tokens,omitempty"`
	DraftTokens    int                `json:"draft_tokens,omitempty"`
	GenerateConfig GenerateConfig     `json:"generate_config,omitempty"`
	TargetGenerate DecodeGenerateFunc `json:"-"`
	DraftGenerate  DecodeGenerateFunc `json:"-"`
}

// PromptLookupDecodeConfig is the mlx-shaped prompt-lookup decode brief.
type PromptLookupDecodeConfig struct {
	Prompt         string             `json:"prompt,omitempty"`
	MaxTokens      int                `json:"max_tokens,omitempty"`
	GenerateConfig GenerateConfig     `json:"generate_config,omitempty"`
	TargetGenerate DecodeGenerateFunc `json:"-"`
	LookupTokens   []Token            `json:"lookup_tokens,omitempty"`
}

// RunSpeculativeDecode runs the speculative-decode harness against
// mlx-shaped generators.
//
//	result, err := mlx.RunSpeculativeDecode(ctx, cfg)
func RunSpeculativeDecode(ctx context.Context, cfg SpeculativeDecodeConfig) (DecodeOptimisationResult, error) {
	return decode.Speculative(ctx, decode.SpeculativeConfig{
		Prompt:         cfg.Prompt,
		MaxTokens:      cfg.MaxTokens,
		DraftTokens:    cfg.DraftTokens,
		GenerateConfig: decode.GenerateConfig{MaxTokens: cfg.GenerateConfig.MaxTokens},
		TargetGenerate: mlxDecodeGenToDecode(cfg.TargetGenerate),
		DraftGenerate:  mlxDecodeGenToDecode(cfg.DraftGenerate),
	})
}

// RunPromptLookupDecode runs the prompt-lookup decode harness against
// mlx-shaped generators.
//
//	result, err := mlx.RunPromptLookupDecode(ctx, cfg)
func RunPromptLookupDecode(ctx context.Context, cfg PromptLookupDecodeConfig) (DecodeOptimisationResult, error) {
	return decode.PromptLookup(ctx, decode.PromptLookupConfig{
		Prompt:         cfg.Prompt,
		MaxTokens:      cfg.MaxTokens,
		GenerateConfig: decode.GenerateConfig{MaxTokens: cfg.GenerateConfig.MaxTokens},
		TargetGenerate: mlxDecodeGenToDecode(cfg.TargetGenerate),
		LookupTokens:   mlxTokensToDecode(cfg.LookupTokens),
	})
}

// mlxDecodeGenToDecode wraps an mlx-shaped DecodeGenerateFunc as a
// decode.GenerateFunc, converting GenerateConfig + DecodeGeneration at
// the boundary.
func mlxDecodeGenToDecode(fn DecodeGenerateFunc) decode.GenerateFunc {
	if fn == nil {
		return nil
	}
	return func(ctx context.Context, prompt string, cfg decode.GenerateConfig) (decode.Generation, error) {
		mlxCfg := GenerateConfig{MaxTokens: cfg.MaxTokens}
		result, err := fn(ctx, prompt, mlxCfg)
		if err != nil {
			return decode.Generation{}, err
		}
		return decode.Generation{Text: result.Text, Tokens: mlxTokensToDecode(result.Tokens)}, nil
	}
}

// mlxTokensToDecode converts an mlx.Token slice to []decode.Token.
//
//	out := mlxTokensToDecode(tokens)
func mlxTokensToDecode(tokens []Token) []decode.Token {
	if tokens == nil {
		return nil
	}
	out := make([]decode.Token, len(tokens))
	for i, t := range tokens {
		out[i] = decode.Token{ID: t.ID, Value: t.Value, Text: t.Text}
	}
	return out
}

// decodeTokensToMlx converts a []decode.Token slice back to []mlx.Token.
//
//	out := decodeTokensToMlx(tokens)
func decodeTokensToMlx(tokens []decode.Token) []Token {
	if tokens == nil {
		return nil
	}
	out := make([]Token, len(tokens))
	for i, t := range tokens {
		out[i] = Token{ID: t.ID, Value: t.Value, Text: t.Text}
	}
	return out
}

// decodeTokensText renders an mlx.Token slice as a concatenated string,
// preferring Text then Value. Retained for callers that need the same
// rendering for non-decode paths (e.g. memvid_chapter_smoke).
//
//	text := decodeTokensText(tokens)
func decodeTokensText(tokens []Token) string {
	return decode.TokensText(mlxTokensToDecode(tokens))
}

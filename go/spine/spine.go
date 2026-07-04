// SPDX-Licence-Identifier: EUPL-1.2

// Package spine holds the request/response types and conversions shared
// between the root mlx package and its subpackages — the vertebrae both
// lean on without importing each other. The root package aliases these
// types (`type GenerateConfig = spine.GenerateConfig`), so the public
// mlx.* surface is unchanged; subpackages (session, train) import spine
// directly and never reach back up into root.
//
// spine imports only packages below the root (probe, parser, memory,
// lora, bundle, pkg/metal) — never the root mlx package itself.
package spine

import (
	"dappco.re/go/inference/parser"
	"dappco.re/go/inference/probe"
)

// GenerateConfig holds generation parameters for the RFC-style root API.
type GenerateConfig struct {
	MaxTokens                    int
	Temperature                  float32
	TopK                         int
	TopP                         float32
	MinP                         float32
	Seed                         uint64
	SeedSet                      bool
	ReturnLogits                 bool
	StopTokens                   []int32
	SuppressTokens               []int32
	MinTokensBeforeStop          int
	RepeatPenalty                float32
	ProbeSink                    probe.Sink
	TraceTokenPhases             bool
	TraceTokenText               bool
	GenerationClearCache         bool
	GenerationClearCacheInterval int
	Thinking                     parser.Config
	// ThinkingBudget caps tokens spent in a reasoning model's thought channel
	// (#99); 0 = unlimited. The engine forces the channel close on overrun.
	ThinkingBudget int
}

// DefaultGenerateConfig returns sensible defaults for root-package generation.
func DefaultGenerateConfig() GenerateConfig {
	return GenerateConfig{
		// 0 = generate to the model's context window, resolved at generate time
		// from the loaded context / the model's declared maximum — never a fixed
		// cap. EOS/stop tokens terminate naturally.
		MaxTokens:   0,
		Temperature: 0.0,
		Thinking:    parser.Config{Mode: parser.Show},
	}
}

// GenerateOption configures root-package text generation. The WithX
// builders live in the root mlx package (they are user-facing API);
// spine owns the type so subpackages can accept options without
// importing root.
type GenerateOption func(*GenerateConfig)

// ApplyGenerateOptions folds a WithX option stack onto the defaults.
//
//	cfg := spine.ApplyGenerateOptions(opts)
func ApplyGenerateOptions(opts []GenerateOption) GenerateConfig {
	cfg := DefaultGenerateConfig()
	for _, opt := range opts {
		opt(&cfg)
	}
	return cfg
}

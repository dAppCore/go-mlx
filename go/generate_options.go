// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	// Note: AX-6 - time.Duration is part of the public Metrics API.

	"dappco.re/go/inference/parser"
	"dappco.re/go/mlx/probe"
)

// generate_options.go: GenerateConfig and its WithX GenerateOption functional
// options — sampling (temp/topK/topP/minP/seed), stop/suppress tokens, repeat
// penalty, cache clearing, token phase tracing, probe sinks.

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

// GenerateOption configures root-package text generation.
type GenerateOption func(*GenerateConfig)

// WithMaxTokens sets the maximum number of tokens to generate.
func WithMaxTokens(n int) GenerateOption {
	return func(c *GenerateConfig) { c.MaxTokens = n }
}

// WithTemperature sets the sampling temperature. 0 = greedy.
func WithTemperature(t float32) GenerateOption {
	return func(c *GenerateConfig) { c.Temperature = t }
}

// WithTopK sets top-k sampling. 0 = disabled.
func WithTopK(k int) GenerateOption {
	return func(c *GenerateConfig) { c.TopK = k }
}

// WithTopP sets nucleus sampling. 0 = disabled.
func WithTopP(p float32) GenerateOption {
	return func(c *GenerateConfig) { c.TopP = p }
}

// WithMinP sets minimum-probability sampling relative to the best token.
func WithMinP(p float32) GenerateOption {
	return func(c *GenerateConfig) { c.MinP = p }
}

// WithSeed resets MLX's default RNG before this generation call.
func WithSeed(seed uint64) GenerateOption {
	return func(c *GenerateConfig) {
		c.Seed = seed
		c.SeedSet = true
	}
}

// withLogitsOption / withTokenPhaseTraceOption are the package-init
// singleton closures returned by every WithLogits / WithReturnLogits /
// WithTokenPhaseTrace call. The no-argument option builders captured
// nothing, so the prior `return func(...){...}` form heap-allocated a
// fresh closure on every call — measurable in the option-stack bench
// because every Generate call site that asks for logits walks through
// this builder. Hoisting the closure once at package init makes the
// builder a pure pointer return, dropping the alloc to zero.
var (
	withLogitsOption          GenerateOption = func(c *GenerateConfig) { c.ReturnLogits = true }
	withTokenPhaseTraceOption GenerateOption = func(c *GenerateConfig) { c.TraceTokenPhases = true }
	withTokenPhaseTextOption  GenerateOption = func(c *GenerateConfig) {
		c.TraceTokenPhases = true
		c.TraceTokenText = true
	}
)

// WithLogits requests classification logits when the called API supports them.
func WithLogits() GenerateOption {
	return withLogitsOption
}

// WithReturnLogits is an alias for WithLogits.
func WithReturnLogits() GenerateOption {
	return withLogitsOption
}

// WithStopTokens sets token IDs that stop generation.
func WithStopTokens(ids ...int32) GenerateOption {
	return func(c *GenerateConfig) { c.StopTokens = ids }
}

// WithSuppressTokens masks token IDs out of the sampling distribution.
func WithSuppressTokens(ids ...int32) GenerateOption {
	return func(c *GenerateConfig) { c.SuppressTokens = ids }
}

// WithMinTokensBeforeStop masks stop tokens until n real tokens have been
// emitted, then restores normal stop behaviour.
func WithMinTokensBeforeStop(n int) GenerateOption {
	return func(c *GenerateConfig) { c.MinTokensBeforeStop = n }
}

// WithRepeatPenalty sets the repetition penalty.
func WithRepeatPenalty(p float32) GenerateOption {
	return func(c *GenerateConfig) { c.RepeatPenalty = p }
}

// WithGenerationClearCacheInterval sets the decode-token interval used when
// generation clear-cache mode is enabled. 0 leaves the backend default.
func WithGenerationClearCacheInterval(n int) GenerateOption {
	return func(c *GenerateConfig) { c.GenerationClearCacheInterval = n }
}

// WithGenerationClearCache clears the native allocator cache after prefill and
// periodically during decode for this request.
func WithGenerationClearCache() GenerateOption {
	return func(c *GenerateConfig) { c.GenerationClearCache = true }
}

// WithTokenPhaseTrace records per-token decode-loop timings in Metrics.
func WithTokenPhaseTrace() GenerateOption {
	return withTokenPhaseTraceOption
}

// WithTokenPhaseTraceText records decoded token text alongside phase timings.
func WithTokenPhaseTraceText() GenerateOption {
	return withTokenPhaseTextOption
}

// withNoopGenerateOption is the no-op closure returned by WithProbeSink and
// WithProbeCallback when the caller passes a nil sink/callback. Sharing one
// package-init function value eliminates the per-call empty-closure alloc
// the prior `return func(*GenerateConfig) {}` form re-emitted, matching the
// withLogitsOption / withTokenPhaseTraceOption pattern above.
var withNoopGenerateOption GenerateOption = func(*GenerateConfig) {}

// WithProbeSink streams typed probe events during generation.
//
//	model.Generate(prompt, mlx.WithProbeSink(sink))
func WithProbeSink(sink probe.Sink) GenerateOption {
	if sink == nil {
		return withNoopGenerateOption
	}
	return func(c *GenerateConfig) { c.ProbeSink = sink }
}

// WithProbeCallback streams typed probe events to a callback during generation.
//
//	model.Generate(prompt, mlx.WithProbeCallback(func(e probe.Event) { … }))
func WithProbeCallback(callback func(probe.Event)) GenerateOption {
	if callback == nil {
		return withNoopGenerateOption
	}
	return WithProbeSink(probe.SinkFunc(callback))
}

func applyGenerateOptions(opts []GenerateOption) GenerateConfig {
	cfg := DefaultGenerateConfig()
	for _, opt := range opts {
		opt(&cfg)
	}
	return cfg
}

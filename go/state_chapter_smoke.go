// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"
	"time"

	core "dappco.re/go"
	state "dappco.re/go/inference/state"
	"dappco.re/go/mlx/chaptersmoke"
	"dappco.re/go/mlx/kv"
)

// NewModelStateKVChapterRunner builds a chaptersmoke.Runner from a loaded
// Model. The Capture/Generate closures own all mlx-specific behaviour;
// chaptersmoke itself never touches mlx types.
//
//	runner := mlx.NewModelStateKVChapterRunner(model, baseGen)
//	report, err := chaptersmoke.Run(ctx, runner, chaptersmoke.Config{...})
func NewModelStateKVChapterRunner(model *Model, baseGen GenerateConfig) chaptersmoke.Runner {
	// baseGen is captured by the Generate closure and never mutated
	// during chapter-smoke iteration. Pre-build the GenerateOption
	// slice once at runner-construction time so every chapter Generate
	// call reuses the same slice instead of allocating + populating
	// it fresh each iteration (one chapter ≈ one session ≈ one
	// allocation triplet — slice header + closure captures × N).
	genOpts := stateKVChapterGenerateOptions(baseGen)
	return chaptersmoke.Runner{
		Capture: func(ctx context.Context, prompt string, store state.Writer, opts kv.StateBlockOptions) (*kv.StateBlockBundle, error) {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
			session, err := model.NewSession()
			if err != nil {
				return nil, err
			}
			defer session.Close()
			if err := session.Prefill(prompt); err != nil {
				return nil, err
			}
			return session.SaveKVBlocksToState(ctx, store, opts)
		},
		Generate: func(ctx context.Context, store state.Store, bundle *kv.StateBlockBundle, prefixTokens int, suffix string) (chaptersmoke.Generation, error) {
			if err := ctx.Err(); err != nil {
				return chaptersmoke.Generation{}, err
			}
			session, err := model.NewSession()
			if err != nil {
				return chaptersmoke.Generation{}, err
			}
			defer session.Close()
			restoreStart := time.Now()
			if err := session.LoadKVPrefixBlocksFromState(ctx, store, bundle, prefixTokens); err != nil {
				return chaptersmoke.Generation{}, err
			}
			restoreDuration := time.Since(restoreStart)
			if err := session.AppendPrompt(suffix); err != nil {
				return chaptersmoke.Generation{}, err
			}
			text, err := session.Generate(genOpts...)
			metrics := model.Metrics()
			return chaptersmoke.Generation{
				Text:                       text,
				DecodeDuration:             metrics.DecodeDuration,
				TotalDuration:              metrics.TotalDuration,
				PromptCacheRestoreDuration: restoreDuration,
			}, err
		},
	}
}

// NewModelMemvidKVChapterRunner builds a chaptersmoke.Runner from a loaded
// Model using the old memvid-named API.
//
// Deprecated: use NewModelStateKVChapterRunner.
func NewModelMemvidKVChapterRunner(model *Model, baseGen GenerateConfig) chaptersmoke.Runner {
	return NewModelStateKVChapterRunner(model, baseGen)
}

// RunModelStateKVChapterSmoke wraps chaptersmoke.Run with a Model-backed
// runner.
//
//	report, err := mlx.RunModelStateKVChapterSmoke(ctx, model, cfg)
func RunModelStateKVChapterSmoke(ctx context.Context, model *Model, cfg chaptersmoke.Config) (*chaptersmoke.Report, error) {
	if model == nil {
		return nil, core.NewError("mlx: model is nil")
	}
	baseGen := chapterGenerateConfig(cfg)
	return chaptersmoke.Run(ctx, NewModelStateKVChapterRunner(model, baseGen), cfg)
}

// RunModelMemvidKVChapterSmoke wraps chaptersmoke.Run with a Model-backed
// runner using the old memvid-named API.
//
// Deprecated: use RunModelStateKVChapterSmoke.
func RunModelMemvidKVChapterSmoke(ctx context.Context, model *Model, cfg chaptersmoke.Config) (*chaptersmoke.Report, error) {
	return RunModelStateKVChapterSmoke(ctx, model, cfg)
}

func chapterGenerateConfig(cfg chaptersmoke.Config) GenerateConfig {
	// gen starts at the zero value, so the previous "only assign if
	// non-zero" guards were equivalent to unconditional assignment —
	// writing zero into a zero field is a no-op. Returning a struct
	// literal lets the compiler skip the local stack copy + branch
	// sequence and emit a single composite-literal store.
	return GenerateConfig{
		MaxTokens:   cfg.AnswerMaxTokens,
		Temperature: cfg.Temperature,
	}
}

func stateKVChapterGenerateOptions(cfg GenerateConfig) []GenerateOption {
	// Collapse the per-field With{MaxTokens,Temperature,TopK,…} closures
	// into one closure that captures the relevant cfg fields as scalar
	// locals and writes them in a single pass against the target. The
	// previous per-field shape allocated up to eight GenerateOption
	// closures (one per WithXxx call, each a 24-byte function value
	// plus the int/float scalar capture) plus the 8-cap option slice.
	// The collapsed form heap-allocates one func value + one 1-cap
	// slice header regardless of how many cfg fields are populated.
	// Bench delta against the typical chapter-runner config (TopK +
	// TopP + RepeatPenalty + StopTokens populated):
	//
	//   typical    7 → 3 allocs   (-4)
	//   full(8)    9 → 3 allocs   (-6)
	//
	// The spine.ApplyGenerateOptions loop tolerates a multi-field closure —
	// it simply calls each option once against the same target — so
	// the consumer contract is preserved. Conditional gating on
	// topK > 0 (etc.) moves inside the closure body so the
	// DefaultGenerateConfig() seed fields stay untouched when the
	// chapter caller leaves them zero.
	//
	// Scalar-local capture (instead of capturing the whole cfg struct)
	// keeps the closure capture set narrow: capturing the full
	// GenerateConfig would pin a heap copy of all 15 fields (~144 B
	// including the Thinking parser.Config + two slice headers + the
	// ProbeSink interface), so for chapter-smoke's common Minimum-form
	// cfg (just MaxTokens + Temperature) the closure heap footprint
	// stays close to the prior pair-of-WithXxx form.
	maxTokens := cfg.MaxTokens
	temperature := cfg.Temperature
	topK := cfg.TopK
	topP := cfg.TopP
	minP := cfg.MinP
	stopTokens := cfg.StopTokens
	minTokensBeforeStop := cfg.MinTokensBeforeStop
	repeatPenalty := cfg.RepeatPenalty
	probeSink := cfg.ProbeSink
	apply := func(c *GenerateConfig) {
		c.MaxTokens = maxTokens
		c.Temperature = temperature
		if topK > 0 {
			c.TopK = topK
		}
		if topP > 0 {
			c.TopP = topP
		}
		if minP > 0 {
			c.MinP = minP
		}
		if len(stopTokens) > 0 {
			// stopTokens captures the caller's slice header
			// directly — the chapter-runner Generate code paths
			// only read from StopTokens, never mutate in place,
			// so aliasing the receiver lifetime is safe.
			c.StopTokens = stopTokens
		}
		if minTokensBeforeStop > 0 {
			c.MinTokensBeforeStop = minTokensBeforeStop
		}
		if repeatPenalty > 0 {
			c.RepeatPenalty = repeatPenalty
		}
		if probeSink != nil {
			c.ProbeSink = probeSink
		}
	}
	return []GenerateOption{apply}
}

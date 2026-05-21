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
			text, err := session.Generate(stateKVChapterGenerateOptions(baseGen)...)
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
	gen := GenerateConfig{}
	if cfg.AnswerMaxTokens > 0 {
		gen.MaxTokens = cfg.AnswerMaxTokens
	}
	if cfg.Temperature != 0 {
		gen.Temperature = cfg.Temperature
	}
	return gen
}

func stateKVChapterGenerateOptions(cfg GenerateConfig) []GenerateOption {
	out := []GenerateOption{
		WithMaxTokens(cfg.MaxTokens),
		WithTemperature(cfg.Temperature),
	}
	if cfg.TopK > 0 {
		out = append(out, WithTopK(cfg.TopK))
	}
	if cfg.TopP > 0 {
		out = append(out, WithTopP(cfg.TopP))
	}
	if cfg.MinP > 0 {
		out = append(out, WithMinP(cfg.MinP))
	}
	if len(cfg.StopTokens) > 0 {
		out = append(out, WithStopTokens(cfg.StopTokens...))
	}
	if cfg.RepeatPenalty > 0 {
		out = append(out, WithRepeatPenalty(cfg.RepeatPenalty))
	}
	if cfg.ProbeSink != nil {
		out = append(out, WithProbeSink(cfg.ProbeSink))
	}
	return out
}

// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"
	"time"

	core "dappco.re/go"
	memvid "dappco.re/go/inference/state"
	"dappco.re/go/mlx/chaptersmoke"
	"dappco.re/go/mlx/kv"
)

// NewModelMemvidKVChapterRunner builds a chaptersmoke.Runner from a loaded
// Model. The Capture/Generate closures own all mlx-specific behaviour;
// chaptersmoke itself never touches mlx types.
//
//	runner := mlx.NewModelMemvidKVChapterRunner(model, baseGen)
//	report, err := chaptersmoke.Run(ctx, runner, chaptersmoke.Config{...})
func NewModelMemvidKVChapterRunner(model *Model, baseGen GenerateConfig) chaptersmoke.Runner {
	return chaptersmoke.Runner{
		Capture: func(ctx context.Context, prompt string, store memvid.Writer, opts kv.MemvidBlockOptions) (*kv.MemvidBlockBundle, error) {
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
			return session.SaveKVBlocksToMemvid(ctx, store, opts)
		},
		Generate: func(ctx context.Context, store memvid.Store, bundle *kv.MemvidBlockBundle, prefixTokens int, suffix string) (chaptersmoke.Generation, error) {
			if err := ctx.Err(); err != nil {
				return chaptersmoke.Generation{}, err
			}
			session, err := model.NewSession()
			if err != nil {
				return chaptersmoke.Generation{}, err
			}
			defer session.Close()
			loadOpts := kv.LoadOptions{}
			if bundle != nil && bundle.KVEncoding == kv.EncodingNative {
				loadOpts.RawKVOnly = true
			}
			restoreStart := time.Now()
			snapshot, err := kv.LoadPrefixFromMemvidBlocksWithOptions(ctx, store, bundle, prefixTokens, loadOpts)
			if err != nil {
				return chaptersmoke.Generation{}, err
			}
			if err := session.RestoreKV(snapshot); err != nil {
				return chaptersmoke.Generation{}, err
			}
			restoreDuration := time.Since(restoreStart)
			if err := session.AppendPrompt(suffix); err != nil {
				return chaptersmoke.Generation{}, err
			}
			text, err := session.Generate(memvidKVChapterGenerateOptions(baseGen)...)
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

// RunModelMemvidKVChapterSmoke wraps chaptersmoke.Run with a Model-backed
// runner.
//
//	report, err := mlx.RunModelMemvidKVChapterSmoke(ctx, model, cfg)
func RunModelMemvidKVChapterSmoke(ctx context.Context, model *Model, cfg chaptersmoke.Config) (*chaptersmoke.Report, error) {
	if model == nil {
		return nil, core.NewError("mlx: model is nil")
	}
	baseGen := chapterGenerateConfig(cfg)
	return chaptersmoke.Run(ctx, NewModelMemvidKVChapterRunner(model, baseGen), cfg)
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

func memvidKVChapterGenerateOptions(cfg GenerateConfig) []GenerateOption {
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

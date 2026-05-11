// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"

	core "dappco.re/go"
	"dappco.re/go/inference/bench"
	"dappco.re/go/mlx/lora"
	"dappco.re/go/mlx/probe"
)

// RunFastEvalBench runs the benchmark harness against a loaded Model.
func RunFastEvalBench(ctx context.Context, model *Model, cfg bench.Config) (*bench.Report, error) {
	if model == nil {
		return nil, core.NewError("mlx: model is nil")
	}
	return RunFastEval(ctx, NewModelFastEvalRunner(model), cfg)
}

// RunFastEval runs a local benchmark/eval suite against the supplied runner.
func RunFastEval(ctx context.Context, runner bench.Runner, cfg bench.Config) (*bench.Report, error) {
	return bench.Run(ctx, runner, cfg)
}

// toBenchGenerateOptions converts bench.GenerateOptions into mlx.GenerateConfig
// for callbacks that hand off to mlx-root generation.
func toBenchGenerateOptions(opts bench.GenerateOptions) GenerateConfig {
	cfg := GenerateConfig{
		MaxTokens:     opts.MaxTokens,
		Temperature:   opts.Temperature,
		TopK:          opts.TopK,
		TopP:          opts.TopP,
		MinP:          opts.MinP,
		StopTokens:    append([]int32(nil), opts.StopTokens...),
		RepeatPenalty: opts.RepeatPenalty,
	}
	if sink, ok := opts.ProbeSink.(probe.Sink); ok {
		cfg.ProbeSink = sink
	}
	return cfg
}

// fromMlxMetrics returns a bench.GenerationMetrics from the mlx-root Metrics.
func fromMlxMetrics(m Metrics) bench.GenerationMetrics {
	return bench.GenerationMetrics{
		PromptTokens:               m.PromptTokens,
		GeneratedTokens:            m.GeneratedTokens,
		PrefillDuration:            m.PrefillDuration,
		DecodeDuration:             m.DecodeDuration,
		TotalDuration:              m.TotalDuration,
		PrefillTokensPerSec:        m.PrefillTokensPerSec,
		DecodeTokensPerSec:         m.DecodeTokensPerSec,
		PeakMemoryBytes:            m.PeakMemoryBytes,
		ActiveMemoryBytes:          m.ActiveMemoryBytes,
		PromptCacheHits:            m.PromptCacheHits,
		PromptCacheMisses:          m.PromptCacheMisses,
		PromptCacheHitTokens:       m.PromptCacheHitTokens,
		PromptCacheMissTokens:      m.PromptCacheMissTokens,
		PromptCacheRestoreDuration: m.PromptCacheRestoreDuration,
	}
}

// modelInfoToBench converts an mlx.ModelInfo into bench.Info.
func modelInfoToBench(info ModelInfo) bench.Info {
	return bench.Info{
		Architecture:  info.Architecture,
		VocabSize:     info.VocabSize,
		NumLayers:     info.NumLayers,
		HiddenSize:    info.HiddenSize,
		QuantBits:     info.QuantBits,
		QuantGroup:    info.QuantGroup,
		ContextLength: info.ContextLength,
		Adapter:       loraToBenchAdapter(info.Adapter),
	}
}

// benchInfoToModel converts back from driver-neutral bench.Info to mlx.ModelInfo.
func benchInfoToModel(info bench.Info) ModelInfo {
	return ModelInfo{
		Architecture:  info.Architecture,
		VocabSize:     info.VocabSize,
		NumLayers:     info.NumLayers,
		HiddenSize:    info.HiddenSize,
		QuantBits:     info.QuantBits,
		QuantGroup:    info.QuantGroup,
		ContextLength: info.ContextLength,
		Adapter:       benchAdapterToLora(info.Adapter),
	}
}

func loraToBenchAdapter(info lora.AdapterInfo) bench.AdapterInfo {
	return bench.AdapterInfo{
		Name:       info.Name,
		Path:       info.Path,
		Hash:       info.Hash,
		Rank:       info.Rank,
		Alpha:      info.Alpha,
		Scale:      info.Scale,
		TargetKeys: append([]string(nil), info.TargetKeys...),
	}
}

func benchAdapterToLora(info bench.AdapterInfo) lora.AdapterInfo {
	return lora.AdapterInfo{
		Name:       info.Name,
		Path:       info.Path,
		Hash:       info.Hash,
		Rank:       info.Rank,
		Alpha:      info.Alpha,
		Scale:      info.Scale,
		TargetKeys: append([]string(nil), info.TargetKeys...),
	}
}

func fastEvalResultError(result core.Result) error {
	if result.OK {
		return nil
	}
	if err, ok := result.Value.(error); ok {
		return err
	}
	return core.NewError("core result failed")
}

// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"
	"math"

	core "dappco.re/go"
	"dappco.re/go/inference/bench"
	"dappco.re/go/mlx/lora"
	"dappco.re/go/mlx/probe"
)

// Per-call sentinel — RunFastEvalBench / RunFastEvalBenchWithDraft are
// the entry points exercised by bench / driver harness loops; sharing
// the existing errMLXModelNil sentinel reuses the alloc declared in
// backend.go for the nil-model guard. errFastEvalSpeculativePairNil
// covers the dedicated SpeculativePair entry; errFastEvalResultFailed
// is the JSON marshal/unmarshal failure fallback used by every bench
// iteration that exercises state-bundle JSON round-trips.
var (
	errFastEvalSpeculativePairNil = core.NewError("mlx: speculative pair is nil")
	errFastEvalResultFailed       = core.NewError("core result failed")
)

// RunFastEvalBench runs the benchmark harness against a loaded Model.
func RunFastEvalBench(ctx context.Context, model *Model, cfg bench.Config) (*bench.Report, error) {
	if model == nil {
		return nil, errMLXModelNil
	}
	return RunFastEval(ctx, NewModelFastEvalRunner(model), cfg)
}

// RunFastEvalBenchWithDraft runs the benchmark harness with an optional draft
// model for speculative decode reporting.
func RunFastEvalBenchWithDraft(ctx context.Context, model, draft *Model, cfg bench.Config) (*bench.Report, error) {
	if model == nil {
		return nil, errMLXModelNil
	}
	return RunFastEval(ctx, NewModelFastEvalRunnerWithDraft(model, draft), cfg)
}

// RunFastEvalBenchWithSpeculativePair runs the benchmark harness against a
// loaded target/draft pair, preserving native assistant-only pair state.
func RunFastEvalBenchWithSpeculativePair(ctx context.Context, pair *SpeculativePair, cfg bench.Config) (*bench.Report, error) {
	if pair == nil || pair.Target == nil {
		return nil, errFastEvalSpeculativePairNil
	}
	return RunFastEval(ctx, NewModelFastEvalRunnerWithSpeculativePair(pair), cfg)
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
		StopTokens:    core.SliceClone(opts.StopTokens),
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
		FirstTokenDuration:         m.FirstTokenDuration,
		PrefillDuration:            m.PrefillDuration,
		DecodeDuration:             m.DecodeDuration,
		TotalDuration:              m.TotalDuration,
		PrefillTokensPerSec:        finiteMetricFloat64(m.PrefillTokensPerSec),
		DecodeTokensPerSec:         finiteMetricFloat64(m.DecodeTokensPerSec),
		PeakMemoryBytes:            m.PeakMemoryBytes,
		ActiveMemoryBytes:          m.ActiveMemoryBytes,
		PromptCacheHits:            m.PromptCacheHits,
		PromptCacheMisses:          m.PromptCacheMisses,
		PromptCacheHitTokens:       m.PromptCacheHitTokens,
		PromptCacheMissTokens:      m.PromptCacheMissTokens,
		PromptCacheRestoreDuration: m.PromptCacheRestoreDuration,
	}
}

func finiteMetricFloat64(value float64) float64 {
	if math.IsNaN(value) || math.IsInf(value, 0) {
		return 0
	}
	return value
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
		TargetKeys: core.SliceClone(info.TargetKeys),
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
		TargetKeys: core.SliceClone(info.TargetKeys),
	}
}

func fastEvalResultError(result core.Result) error {
	if result.OK {
		return nil
	}
	if err, ok := result.Value.(error); ok {
		return err
	}
	return errFastEvalResultFailed
}

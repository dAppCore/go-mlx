// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"
	"math"
	"time"

	core "dappco.re/go"
)

const WorkloadBenchReportVersion = 1

// WorkloadBenchConfig controls the library-first local workload benchmark.
type WorkloadBenchConfig struct {
	FastEval           FastEvalConfig       `json:"fast_eval"`
	AdapterPath        string               `json:"adapter_path,omitempty"`
	IncludeAdapterLoad bool                 `json:"include_adapter_load"`
	IncludeAdapterFuse bool                 `json:"include_adapter_fuse"`
	IncludePerplexity  bool                 `json:"include_perplexity"`
	EvalSamples        []WorkloadEvalSample `json:"eval_samples,omitempty"`
}

// WorkloadEvalSample is one record used by benchmark eval hooks.
type WorkloadEvalSample struct {
	Prompt   string            `json:"prompt,omitempty"`
	Response string            `json:"response,omitempty"`
	Text     string            `json:"text,omitempty"`
	Meta     map[string]string `json:"meta,omitempty"`
}

// WorkloadAdapterInfo identifies a LoRA adapter measured by the benchmark.
type WorkloadAdapterInfo struct {
	Path       string   `json:"path,omitempty"`
	Name       string   `json:"name,omitempty"`
	Hash       string   `json:"hash,omitempty"`
	Rank       int      `json:"rank,omitempty"`
	Alpha      float32  `json:"alpha,omitempty"`
	TargetKeys []string `json:"target_keys,omitempty"`

	adapter *LoRAAdapter
}

// WorkloadEvalMetrics stores perplexity/eval hook output.
type WorkloadEvalMetrics struct {
	Samples    int     `json:"samples,omitempty"`
	Tokens     int     `json:"tokens,omitempty"`
	Loss       float64 `json:"loss,omitempty"`
	Perplexity float64 `json:"perplexity,omitempty"`
}

// WorkloadBenchRunner supplies model operations measured by RunWorkloadBench.
type WorkloadBenchRunner struct {
	FastEval FastEvalRunner

	LoadAdapter func(context.Context, string) (WorkloadAdapterInfo, error)
	FuseAdapter func(context.Context, WorkloadAdapterInfo) error

	EvaluatePerplexity func(context.Context, []WorkloadEvalSample) (WorkloadEvalMetrics, error)
}

// WorkloadBenchReport is a JSON-friendly report for local model workloads.
type WorkloadBenchReport struct {
	Version    int                      `json:"version"`
	FastEval   *FastEvalReport          `json:"fast_eval,omitempty"`
	Adapter    WorkloadAdapterReport    `json:"adapter"`
	Evaluation WorkloadEvaluationReport `json:"evaluation"`
	Summary    WorkloadBenchSummary     `json:"summary"`
}

// WorkloadBenchSummary mirrors the high-signal metrics needed for quick comparisons.
type WorkloadBenchSummary struct {
	PrefillTokensPerSec        float64       `json:"prefill_tokens_per_sec,omitempty"`
	DecodeTokensPerSec         float64       `json:"decode_tokens_per_sec,omitempty"`
	PeakMemoryBytes            uint64        `json:"peak_memory_bytes,omitempty"`
	ActiveMemoryBytes          uint64        `json:"active_memory_bytes,omitempty"`
	PromptCacheHitRate         float64       `json:"prompt_cache_hit_rate,omitempty"`
	PromptCacheHitTokens       int           `json:"prompt_cache_hit_tokens,omitempty"`
	PromptCacheMissTokens      int           `json:"prompt_cache_miss_tokens,omitempty"`
	PromptCacheRestoreDuration time.Duration `json:"prompt_cache_restore_duration,omitempty"`
	KVRestoreDuration          time.Duration `json:"kv_restore_duration,omitempty"`
	AdapterLoadDuration        time.Duration `json:"adapter_load_duration,omitempty"`
	AdapterFuseDuration        time.Duration `json:"adapter_fuse_duration,omitempty"`
	EvalSamples                int           `json:"eval_samples,omitempty"`
	EvalTokens                 int           `json:"eval_tokens,omitempty"`
	EvalLoss                   float64       `json:"eval_loss,omitempty"`
	Perplexity                 float64       `json:"perplexity,omitempty"`
}

// WorkloadAdapterReport records adapter load and fuse timings.
type WorkloadAdapterReport struct {
	Adapter WorkloadAdapterInfo   `json:"adapter,omitempty"`
	Load    WorkloadLatencyReport `json:"load"`
	Fuse    WorkloadLatencyReport `json:"fuse"`
}

// WorkloadLatencyReport records one optional workload latency measurement.
type WorkloadLatencyReport struct {
	Attempted bool          `json:"attempted"`
	Duration  time.Duration `json:"duration,omitempty"`
	Error     string        `json:"error,omitempty"`
}

// WorkloadEvaluationReport records perplexity/eval hook output.
type WorkloadEvaluationReport struct {
	Attempted bool                `json:"attempted"`
	Duration  time.Duration       `json:"duration,omitempty"`
	Metrics   WorkloadEvalMetrics `json:"metrics,omitempty"`
	Error     string              `json:"error,omitempty"`
}

// DefaultWorkloadBenchConfig returns a small laptop-safe workload benchmark config.
func DefaultWorkloadBenchConfig() WorkloadBenchConfig {
	return WorkloadBenchConfig{FastEval: DefaultFastEvalConfig()}
}

// NewModelWorkloadBenchRunner adapts a loaded Model to the workload benchmark.
func NewModelWorkloadBenchRunner(model *Model) WorkloadBenchRunner {
	return WorkloadBenchRunner{
		FastEval: NewModelFastEvalRunner(model),
		LoadAdapter: func(ctx context.Context, path string) (WorkloadAdapterInfo, error) {
			if err := ctx.Err(); err != nil {
				return WorkloadAdapterInfo{}, err
			}
			if model == nil {
				return WorkloadAdapterInfo{}, core.NewError("mlx: model is nil")
			}
			adapter, err := model.LoadLoRA(path)
			if err != nil {
				return WorkloadAdapterInfo{}, err
			}
			return workloadAdapterInfo(path, adapter), nil
		},
		FuseAdapter: func(ctx context.Context, info WorkloadAdapterInfo) error {
			if err := ctx.Err(); err != nil {
				return err
			}
			if model == nil {
				return core.NewError("mlx: model is nil")
			}
			if info.adapter == nil {
				return core.NewError("mlx: workload adapter has no native handle")
			}
			model.MergeLoRA(info.adapter)
			return nil
		},
	}
}

// RunModelWorkloadBench runs the workload benchmark against a loaded Model.
func RunModelWorkloadBench(ctx context.Context, model *Model, cfg WorkloadBenchConfig) (*WorkloadBenchReport, error) {
	if model == nil {
		return nil, core.NewError("mlx: model is nil")
	}
	return RunWorkloadBench(ctx, NewModelWorkloadBenchRunner(model), cfg)
}

// RunWorkloadBench measures local inference, cache, adapter, and eval workload hooks.
func RunWorkloadBench(ctx context.Context, runner WorkloadBenchRunner, cfg WorkloadBenchConfig) (*WorkloadBenchReport, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	cfg = normalizeWorkloadBenchConfig(cfg)
	report := &WorkloadBenchReport{Version: WorkloadBenchReportVersion}

	fastEval, err := RunFastEval(ctx, runner.FastEval, cfg.FastEval)
	if err != nil {
		return nil, err
	}
	report.FastEval = fastEval

	var adapter WorkloadAdapterInfo
	if cfg.IncludeAdapterLoad || cfg.IncludeAdapterFuse {
		adapter = runWorkloadAdapterLoad(ctx, runner, cfg, &report.Adapter)
	}
	if cfg.IncludeAdapterFuse {
		runWorkloadAdapterFuse(ctx, runner, adapter, &report.Adapter)
	}
	if cfg.IncludePerplexity {
		report.Evaluation = runWorkloadEvaluation(ctx, runner, cfg)
	}
	report.Summary = summarizeWorkloadBench(report)
	return report, nil
}

func normalizeWorkloadBenchConfig(cfg WorkloadBenchConfig) WorkloadBenchConfig {
	cfg.FastEval = normalizeFastEvalConfig(cfg.FastEval)
	cfg.EvalSamples = cloneWorkloadEvalSamples(cfg.EvalSamples)
	return cfg
}

func runWorkloadAdapterLoad(ctx context.Context, runner WorkloadBenchRunner, cfg WorkloadBenchConfig, report *WorkloadAdapterReport) WorkloadAdapterInfo {
	if report == nil {
		return WorkloadAdapterInfo{}
	}
	report.Load.Attempted = true
	if cfg.AdapterPath == "" {
		report.Load.Error = "adapter path is required"
		return WorkloadAdapterInfo{}
	}
	if runner.LoadAdapter == nil {
		report.Load.Error = "runner does not support LoRA adapter loading"
		return WorkloadAdapterInfo{}
	}
	start := time.Now()
	adapter, err := runner.LoadAdapter(ctx, cfg.AdapterPath)
	report.Load.Duration = nonZeroDuration(time.Since(start))
	if err != nil {
		report.Load.Error = err.Error()
		return WorkloadAdapterInfo{}
	}
	adapter = cloneWorkloadAdapterInfo(adapter)
	if adapter.Path == "" {
		adapter.Path = cfg.AdapterPath
	}
	if adapter.Name == "" {
		adapter.Name = core.PathBase(adapter.Path)
	}
	report.Adapter = adapter
	return adapter
}

func runWorkloadAdapterFuse(ctx context.Context, runner WorkloadBenchRunner, adapter WorkloadAdapterInfo, report *WorkloadAdapterReport) {
	if report == nil {
		return
	}
	report.Fuse.Attempted = true
	if report.Load.Error != "" {
		report.Fuse.Error = "adapter load failed: " + report.Load.Error
		return
	}
	if adapter.Path == "" {
		report.Fuse.Error = "adapter is required for fuse"
		return
	}
	if runner.FuseAdapter == nil {
		report.Fuse.Error = "runner does not support LoRA adapter fuse"
		return
	}
	start := time.Now()
	err := runner.FuseAdapter(ctx, adapter)
	report.Fuse.Duration = nonZeroDuration(time.Since(start))
	if err != nil {
		report.Fuse.Error = err.Error()
	}
}

func runWorkloadEvaluation(ctx context.Context, runner WorkloadBenchRunner, cfg WorkloadBenchConfig) WorkloadEvaluationReport {
	report := WorkloadEvaluationReport{Attempted: true}
	if runner.EvaluatePerplexity == nil {
		report.Error = "runner does not support perplexity evaluation"
		return report
	}
	if len(cfg.EvalSamples) == 0 {
		report.Error = "no eval samples configured"
		return report
	}
	start := time.Now()
	metrics, err := runner.EvaluatePerplexity(ctx, cloneWorkloadEvalSamples(cfg.EvalSamples))
	report.Duration = nonZeroDuration(time.Since(start))
	if err != nil {
		report.Error = err.Error()
		return report
	}
	if metrics.Samples == 0 {
		metrics.Samples = len(cfg.EvalSamples)
	}
	if metrics.Perplexity == 0 && metrics.Loss > 0 {
		metrics.Perplexity = math.Exp(metrics.Loss)
	}
	report.Metrics = metrics
	return report
}

func summarizeWorkloadBench(report *WorkloadBenchReport) WorkloadBenchSummary {
	var summary WorkloadBenchSummary
	if report == nil {
		return summary
	}
	if report.FastEval != nil {
		summary.PrefillTokensPerSec = report.FastEval.Generation.PrefillTokensPerSec
		summary.DecodeTokensPerSec = report.FastEval.Generation.DecodeTokensPerSec
		summary.PeakMemoryBytes = report.FastEval.Generation.PeakMemoryBytes
		summary.ActiveMemoryBytes = report.FastEval.Generation.ActiveMemoryBytes
		summary.PromptCacheHitRate = report.FastEval.PromptCache.HitRate
		summary.PromptCacheHitTokens = report.FastEval.PromptCache.HitTokens
		summary.PromptCacheMissTokens = report.FastEval.PromptCache.MissTokens
		summary.PromptCacheRestoreDuration = report.FastEval.PromptCache.RestoreDuration
		summary.KVRestoreDuration = report.FastEval.KVRestore.Duration
	}
	summary.AdapterLoadDuration = report.Adapter.Load.Duration
	summary.AdapterFuseDuration = report.Adapter.Fuse.Duration
	summary.EvalSamples = report.Evaluation.Metrics.Samples
	summary.EvalTokens = report.Evaluation.Metrics.Tokens
	summary.EvalLoss = report.Evaluation.Metrics.Loss
	summary.Perplexity = report.Evaluation.Metrics.Perplexity
	return summary
}

func workloadAdapterInfo(path string, adapter *LoRAAdapter) WorkloadAdapterInfo {
	info := WorkloadAdapterInfo{
		Path:    path,
		Name:    core.PathBase(path),
		adapter: adapter,
	}
	if adapter != nil {
		info.Rank = adapter.Config.Rank
		info.Alpha = adapter.Config.Alpha
		info.TargetKeys = append([]string(nil), adapter.Config.TargetKeys...)
	}
	return info
}

func cloneWorkloadAdapterInfo(info WorkloadAdapterInfo) WorkloadAdapterInfo {
	info.TargetKeys = append([]string(nil), info.TargetKeys...)
	return info
}

func cloneWorkloadEvalSamples(samples []WorkloadEvalSample) []WorkloadEvalSample {
	if len(samples) == 0 {
		return nil
	}
	out := make([]WorkloadEvalSample, len(samples))
	for i, sample := range samples {
		out[i] = sample
		if sample.Meta != nil {
			out[i].Meta = make(map[string]string, len(sample.Meta))
			for key, value := range sample.Meta {
				out[i].Meta[key] = value
			}
		}
	}
	return out
}

func nonZeroDuration(duration time.Duration) time.Duration {
	if duration <= 0 {
		return time.Nanosecond
	}
	return duration
}

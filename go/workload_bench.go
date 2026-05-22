// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"
	"dappco.re/go/inference/bench"
	"dappco.re/go/mlx/dataset"
	"math"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference/eval"
	"dappco.re/go/inference/quant/jang"
	"dappco.re/go/mlx/kv"
	"dappco.re/go/mlx/memory"
	"dappco.re/go/mlx/model/minimax/m2"
)

const WorkloadBenchReportVersion = 1

// WorkloadBenchConfig controls the library-first local workload benchmark.
type WorkloadBenchConfig struct {
	FastEval               bench.Config               `json:"fast_eval"`
	Eval                   eval.Config                `json:"eval,omitempty"`
	EvalDataset            dataset.Dataset            `json:"-"`
	AdapterPath            string                     `json:"adapter_path,omitempty"`
	IncludeAdapterLoad     bool                       `json:"include_adapter_load"`
	IncludeAdapterFuse     bool                       `json:"include_adapter_fuse"`
	IncludePerplexity      bool                       `json:"include_perplexity"`
	IncludeKVCacheBench    bool                       `json:"include_kv_cache_bench"`
	IncludeExpertResidency bool                       `json:"include_expert_residency"`
	ExpertResidency        memory.ExpertResidencyPlan `json:"expert_residency,omitempty"`
	QuantizationProfile    *jang.PackedProfile        `json:"quantization_profile,omitempty"`
	EvalSamples            []WorkloadEvalSample       `json:"eval_samples,omitempty"`
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
	FastEval bench.Runner
	Eval     eval.Runner

	LoadAdapter func(context.Context, string) (WorkloadAdapterInfo, error)
	FuseAdapter func(context.Context, WorkloadAdapterInfo) error

	EvaluatePerplexity     func(context.Context, []WorkloadEvalSample) (WorkloadEvalMetrics, error)
	MeasureExpertResidency func(context.Context, memory.ExpertResidencyPlan) (memory.ExpertResidencyStats, error)
}

// WorkloadBenchReport is a JSON-friendly report for local model workloads.
type WorkloadBenchReport struct {
	Version             int                           `json:"version"`
	FastEval            *bench.Report                 `json:"fast_eval,omitempty"`
	KVCache             kv.BenchReport                `json:"kv_cache,omitempty"`
	QuantizationProfile *jang.PackedProfile           `json:"quantization_profile,omitempty"`
	Adapter             WorkloadAdapterReport         `json:"adapter"`
	Evaluation          WorkloadEvaluationReport      `json:"evaluation"`
	ExpertResidency     WorkloadExpertResidencyReport `json:"expert_residency"`
	Summary             WorkloadBenchSummary          `json:"summary"`
}

// WorkloadBenchSummary mirrors the high-signal metrics needed for quick comparisons.
type WorkloadBenchSummary struct {
	PrefillTokensPerSec                  float64       `json:"prefill_tokens_per_sec,omitempty"`
	DecodeTokensPerSec                   float64       `json:"decode_tokens_per_sec,omitempty"`
	PeakMemoryBytes                      uint64        `json:"peak_memory_bytes,omitempty"`
	ActiveMemoryBytes                    uint64        `json:"active_memory_bytes,omitempty"`
	PromptCacheHitRate                   float64       `json:"prompt_cache_hit_rate,omitempty"`
	PromptCacheHitTokens                 int           `json:"prompt_cache_hit_tokens,omitempty"`
	PromptCacheMissTokens                int           `json:"prompt_cache_miss_tokens,omitempty"`
	PromptCacheRestoreDuration           time.Duration `json:"prompt_cache_restore_duration,omitempty"`
	PromptCacheSource                    string        `json:"prompt_cache_source,omitempty"`
	PromptTokensAvoided                  int           `json:"prompt_tokens_avoided,omitempty"`
	PromptCacheReplayTokens              int           `json:"prompt_cache_replay_tokens,omitempty"`
	PromptCacheExactFallbackReplayTokens int           `json:"prompt_cache_exact_fallback_replay_tokens,omitempty"`
	StateKVBlockRestoreDuration          time.Duration `json:"state_kv_block_restore_duration,omitempty"`
	StateKVBlockStorePath                string        `json:"state_kv_block_store_path,omitempty"`
	StateKVBlockStoreBytes               int64         `json:"state_kv_block_store_bytes,omitempty"`
	StateKVBlocksRead                    int           `json:"state_kv_blocks_read,omitempty"`
	StateKVChunksRead                    int           `json:"state_kv_chunks_read,omitempty"`
	StateKVPrefixTokensRestored          int           `json:"state_kv_prefix_tokens_restored,omitempty"`
	KVRestoreDuration                    time.Duration `json:"kv_restore_duration,omitempty"`
	SpeculativeAcceptanceRate            float64       `json:"speculative_acceptance_rate,omitempty"`
	SpeculativeAcceptedTokens            int           `json:"speculative_accepted_tokens,omitempty"`
	SpeculativeRejectedTokens            int           `json:"speculative_rejected_tokens,omitempty"`
	PromptLookupAcceptanceRate           float64       `json:"prompt_lookup_acceptance_rate,omitempty"`
	PromptLookupAcceptedTokens           int           `json:"prompt_lookup_accepted_tokens,omitempty"`
	PromptLookupRejectedTokens           int           `json:"prompt_lookup_rejected_tokens,omitempty"`
	ExpertResidencyResidentExperts       int           `json:"expert_residency_resident_experts,omitempty"`
	ExpertResidencyPeakResidentExperts   int           `json:"expert_residency_peak_resident_experts,omitempty"`
	ExpertResidencyPageIns               int           `json:"expert_residency_page_ins,omitempty"`
	ExpertResidencyPageOuts              int           `json:"expert_residency_page_outs,omitempty"`
	ExpertResidencyLoadedBytes           uint64        `json:"expert_residency_loaded_bytes,omitempty"`
	ExpertResidencyEvictedBytes          uint64        `json:"expert_residency_evicted_bytes,omitempty"`
	ExpertResidencyFirstUseLatency       time.Duration `json:"expert_residency_first_use_latency,omitempty"`
	ExpertResidencyTotalLoadDuration     time.Duration `json:"expert_residency_total_load_duration,omitempty"`
	AdapterLoadDuration                  time.Duration `json:"adapter_load_duration,omitempty"`
	AdapterFuseDuration                  time.Duration `json:"adapter_fuse_duration,omitempty"`
	EvalSamples                          int           `json:"eval_samples,omitempty"`
	EvalTokens                           int           `json:"eval_tokens,omitempty"`
	EvalLoss                             float64       `json:"eval_loss,omitempty"`
	Perplexity                           float64       `json:"perplexity,omitempty"`
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
	Quality   eval.QualityReport  `json:"quality,omitempty"`
	Report    *eval.Report        `json:"report,omitempty"`
	Error     string              `json:"error,omitempty"`
}

// WorkloadExpertResidencyReport records optional lazy expert residency timing.
type WorkloadExpertResidencyReport struct {
	Attempted bool                        `json:"attempted"`
	Duration  time.Duration               `json:"duration,omitempty"`
	Plan      memory.ExpertResidencyPlan  `json:"plan,omitempty"`
	Stats     memory.ExpertResidencyStats `json:"stats,omitempty"`
	Error     string                      `json:"error,omitempty"`
}

// DefaultWorkloadBenchConfig returns a small laptop-safe workload benchmark config.
func DefaultWorkloadBenchConfig() WorkloadBenchConfig {
	return WorkloadBenchConfig{FastEval: bench.DefaultConfig()}
}

// NewModelWorkloadBenchRunner adapts a loaded Model to the workload benchmark.
func NewModelWorkloadBenchRunner(model *Model) WorkloadBenchRunner {
	return WorkloadBenchRunner{
		FastEval: NewModelFastEvalRunner(model),
		Eval:     NewModelEvalRunner(model),
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
	report := &WorkloadBenchReport{
		Version:             WorkloadBenchReportVersion,
		QuantizationProfile: jang.ClonePackedProfile(cfg.QuantizationProfile),
	}

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
	if cfg.IncludeKVCacheBench && report.FastEval != nil {
		report.KVCache = kv.CompareModes(kvBenchConfigFromModelInfo(benchInfoToModel(report.FastEval.ModelInfo)))
	}
	if cfg.IncludeExpertResidency {
		report.ExpertResidency = runWorkloadExpertResidency(ctx, runner, cfg)
	}
	report.Summary = summarizeWorkloadBench(report)
	return report, nil
}

func normalizeWorkloadBenchConfig(cfg WorkloadBenchConfig) WorkloadBenchConfig {
	cfg.Eval = normalizeWorkloadEvalConfig(cfg.Eval)
	cfg.QuantizationProfile = jang.ClonePackedProfile(cfg.QuantizationProfile)
	cfg.EvalSamples = cloneWorkloadEvalSamples(cfg.EvalSamples)
	cfg.ExpertResidency = m2.NormalisePlan(cfg.ExpertResidency)
	return cfg
}

// kvBenchModes is the fixed mode set the workload benchmark compares —
// hoisted out of kvBenchConfigFromModelInfo so we don't re-allocate the
// same 4-element slice literal on every benchmark dispatch. CompareModes
// reads cfg.Modes via range without mutation.
var kvBenchModes = []memory.KVCacheMode{
	memory.KVCacheModeFP16,
	memory.KVCacheModePaged,
	memory.KVCacheModeQ8,
	memory.KVCacheModeKQ8VQ4,
}

func kvBenchConfigFromModelInfo(info ModelInfo) kv.BenchConfig {
	return kv.BenchConfig{
		ContextLength: info.ContextLength,
		NumLayers:     info.NumLayers,
		HiddenSize:    info.HiddenSize,
		Modes:         kvBenchModes,
	}
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
	if cfg.EvalDataset != nil {
		evalCfg := cfg.Eval
		if evalCfg.AdapterPath == "" && !cfg.IncludeAdapterLoad {
			evalCfg.AdapterPath = cfg.AdapterPath
		}
		start := time.Now()
		evalReport, err := eval.RunDataset(ctx, runner.Eval, wrapSFTDataset(cfg.EvalDataset), evalCfg)
		report.Duration = nonZeroDuration(time.Since(start))
		if err != nil {
			report.Error = err.Error()
			return report
		}
		report.Report = evalReport
		report.Quality = evalReport.Quality
		report.Metrics = workloadEvalMetricsFromEval(evalReport.Metrics)
		return report
	}
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

func runWorkloadExpertResidency(ctx context.Context, runner WorkloadBenchRunner, cfg WorkloadBenchConfig) WorkloadExpertResidencyReport {
	report := WorkloadExpertResidencyReport{Attempted: true, Plan: cfg.ExpertResidency}
	if runner.MeasureExpertResidency == nil {
		report.Error = "runner does not support expert residency measurement"
		return report
	}
	start := time.Now()
	stats, err := runner.MeasureExpertResidency(ctx, cfg.ExpertResidency)
	report.Duration = nonZeroDuration(time.Since(start))
	if err != nil {
		report.Error = err.Error()
		return report
	}
	report.Stats = stats
	return report
}

func workloadEvalMetricsFromEval(metrics eval.Metrics) WorkloadEvalMetrics {
	return WorkloadEvalMetrics{
		Samples:    metrics.Samples,
		Tokens:     metrics.Tokens,
		Loss:       metrics.Loss,
		Perplexity: metrics.Perplexity,
	}
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
		if report.FastEval.StateKVBlockWarm.Attempted {
			summary.PromptCacheSource = report.FastEval.StateKVBlockWarm.Source
			summary.PromptTokensAvoided = report.FastEval.StateKVBlockWarm.PromptTokensAvoided
			summary.PromptCacheReplayTokens = report.FastEval.StateKVBlockWarm.ReplayTokens
			summary.PromptCacheExactFallbackReplayTokens = report.FastEval.StateKVBlockWarm.ExactFallbackReplayTokens
			summary.StateKVBlockRestoreDuration = report.FastEval.StateKVBlockWarm.RestoreDuration
			summary.StateKVBlockStorePath = report.FastEval.StateKVBlockWarm.StorePath
			summary.StateKVBlockStoreBytes = report.FastEval.StateKVBlockWarm.StoreBytes
			summary.StateKVBlocksRead = report.FastEval.StateKVBlockWarm.BlocksRead
			summary.StateKVChunksRead = report.FastEval.StateKVBlockWarm.ChunksRead
			summary.StateKVPrefixTokensRestored = report.FastEval.StateKVBlockWarm.PrefixTokensRestored
		}
		summary.KVRestoreDuration = report.FastEval.KVRestore.Duration
		if report.FastEval.SpeculativeDecode.Attempted && report.FastEval.SpeculativeDecode.Error == "" {
			summary.SpeculativeAcceptanceRate = report.FastEval.SpeculativeDecode.Metrics.AcceptanceRate
			summary.SpeculativeAcceptedTokens = report.FastEval.SpeculativeDecode.Metrics.AcceptedTokens
			summary.SpeculativeRejectedTokens = report.FastEval.SpeculativeDecode.Metrics.RejectedTokens
		}
		if report.FastEval.PromptLookupDecode.Attempted && report.FastEval.PromptLookupDecode.Error == "" {
			summary.PromptLookupAcceptanceRate = report.FastEval.PromptLookupDecode.Metrics.AcceptanceRate
			summary.PromptLookupAcceptedTokens = report.FastEval.PromptLookupDecode.Metrics.AcceptedTokens
			summary.PromptLookupRejectedTokens = report.FastEval.PromptLookupDecode.Metrics.RejectedTokens
		}
	}
	summary.AdapterLoadDuration = report.Adapter.Load.Duration
	summary.AdapterFuseDuration = report.Adapter.Fuse.Duration
	if report.ExpertResidency.Attempted && report.ExpertResidency.Error == "" {
		summary.ExpertResidencyResidentExperts = report.ExpertResidency.Stats.ResidentExperts
		summary.ExpertResidencyPeakResidentExperts = report.ExpertResidency.Stats.PeakResidentExperts
		summary.ExpertResidencyPageIns = report.ExpertResidency.Stats.PageIns
		summary.ExpertResidencyPageOuts = report.ExpertResidency.Stats.PageOuts
		summary.ExpertResidencyLoadedBytes = report.ExpertResidency.Stats.LoadedBytes
		summary.ExpertResidencyEvictedBytes = report.ExpertResidency.Stats.EvictedBytes
		summary.ExpertResidencyFirstUseLatency = report.ExpertResidency.Stats.FirstUseLatency
		summary.ExpertResidencyTotalLoadDuration = report.ExpertResidency.Stats.TotalLoadDuration
	}
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
		info.TargetKeys = core.SliceClone(adapter.Config.TargetKeys)
	}
	return info
}

func cloneWorkloadAdapterInfo(info WorkloadAdapterInfo) WorkloadAdapterInfo {
	info.TargetKeys = core.SliceClone(info.TargetKeys)
	return info
}

func cloneWorkloadEvalSamples(samples []WorkloadEvalSample) []WorkloadEvalSample {
	if len(samples) == 0 {
		return nil
	}
	out := make([]WorkloadEvalSample, len(samples))
	for i, sample := range samples {
		out[i] = sample
		out[i].Meta = core.MapClone(sample.Meta)
	}
	return out
}

func nonZeroDuration(duration time.Duration) time.Duration {
	if duration <= 0 {
		return time.Nanosecond
	}
	return duration
}

func normalizeWorkloadEvalConfig(cfg eval.Config) eval.Config {
	if batch, ok := cfg.Batch.(dataset.BatchConfig); ok {
		cfg.Batch = normalizeDatasetBatchConfig(batch)
	}
	cfg.QualityProbes = core.SliceClone(cfg.QualityProbes)
	return cfg
}

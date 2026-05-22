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

// Sentinel errors hoisted from per-call core.NewError sites — the
// "mlx: model is nil" message recurred at four entry points and each
// call allocated a fresh *Err. Sharing one instance keeps the message
// stable for callers comparing via errors.Is and removes the cold-path
// allocation entirely.
var (
	errWorkloadModelNil   = core.NewError("mlx: model is nil")
	errWorkloadAdapterNil = core.NewError("mlx: workload adapter has no native handle")
)

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
				return WorkloadAdapterInfo{}, errWorkloadModelNil
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
				return errWorkloadModelNil
			}
			if info.adapter == nil {
				return errWorkloadAdapterNil
			}
			model.MergeLoRA(info.adapter)
			return nil
		},
	}
}

// RunModelWorkloadBench runs the workload benchmark against a loaded Model.
func RunModelWorkloadBench(ctx context.Context, model *Model, cfg WorkloadBenchConfig) (*WorkloadBenchReport, error) {
	if model == nil {
		return nil, errWorkloadModelNil
	}
	return RunWorkloadBench(ctx, NewModelWorkloadBenchRunner(model), cfg)
}

// RunWorkloadBench measures local inference, cache, adapter, and eval workload hooks.
func RunWorkloadBench(ctx context.Context, runner WorkloadBenchRunner, cfg WorkloadBenchConfig) (*WorkloadBenchReport, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	cfg = normalizeWorkloadBenchConfig(cfg)
	// normalizeWorkloadBenchConfig already produced a fresh clone of the
	// caller's QuantizationProfile and bound it to cfg — cfg is a local
	// value the caller never sees, so the report can take ownership of
	// the same clone instead of paying a second jang.ClonePackedProfile
	// (full struct copy + RoleBits clone) on every dispatch.
	report := &WorkloadBenchReport{
		Version:             WorkloadBenchReportVersion,
		QuantizationProfile: cfg.QuantizationProfile,
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
	// normalizeWorkloadBenchConfig already produced a defensive clone
	// of cfg.EvalSamples (including per-sample Meta map clones) before
	// this helper ran. The slice and its Meta payloads are private to
	// the RunWorkloadBench call frame — we only read its length below —
	// so we hand the same backing slice straight to the user callback
	// instead of paying a second cloneWorkloadEvalSamples (one slice
	// alloc + one map alloc per sample with metadata) on every
	// perplexity-evaluation dispatch.
	metrics, err := runner.EvaluatePerplexity(ctx, cfg.EvalSamples)
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
	// Cache report.FastEval into a local pointer to avoid the ~30
	// re-dereferences the previous body paid through report.FastEval
	// for every field read. The sub-report structs (StateKVBlockWarm,
	// SpeculativeDecode, PromptLookupDecode) are deliberately kept as
	// pointer-deref chains — copying them into locals would clone
	// ~20-field GenerationMetrics blobs we only read a few fields out
	// of.
	if fast := report.FastEval; fast != nil {
		// Cache the Generation + PromptCache sub-block pointers — each
		// is read four times and the chained field-offset compute on
		// every read collapses to a single pointer plus a fixed offset
		// when we hand the compiler a sub-pointer to chase.
		gen := &fast.Generation
		summary.PrefillTokensPerSec = gen.PrefillTokensPerSec
		summary.DecodeTokensPerSec = gen.DecodeTokensPerSec
		summary.PeakMemoryBytes = gen.PeakMemoryBytes
		summary.ActiveMemoryBytes = gen.ActiveMemoryBytes
		pc := &fast.PromptCache
		summary.PromptCacheHitRate = pc.HitRate
		summary.PromptCacheHitTokens = pc.HitTokens
		summary.PromptCacheMissTokens = pc.MissTokens
		summary.PromptCacheRestoreDuration = pc.RestoreDuration
		if kvWarm := &fast.StateKVBlockWarm; kvWarm.Attempted {
			summary.PromptCacheSource = kvWarm.Source
			summary.PromptTokensAvoided = kvWarm.PromptTokensAvoided
			summary.PromptCacheReplayTokens = kvWarm.ReplayTokens
			summary.PromptCacheExactFallbackReplayTokens = kvWarm.ExactFallbackReplayTokens
			summary.StateKVBlockRestoreDuration = kvWarm.RestoreDuration
			summary.StateKVBlockStorePath = kvWarm.StorePath
			summary.StateKVBlockStoreBytes = kvWarm.StoreBytes
			summary.StateKVBlocksRead = kvWarm.BlocksRead
			summary.StateKVChunksRead = kvWarm.ChunksRead
			summary.StateKVPrefixTokensRestored = kvWarm.PrefixTokensRestored
		}
		summary.KVRestoreDuration = fast.KVRestore.Duration
		if spec := &fast.SpeculativeDecode; spec.Attempted && spec.Error == "" {
			m := &spec.Metrics
			summary.SpeculativeAcceptanceRate = m.AcceptanceRate
			summary.SpeculativeAcceptedTokens = m.AcceptedTokens
			summary.SpeculativeRejectedTokens = m.RejectedTokens
		}
		if pl := &fast.PromptLookupDecode; pl.Attempted && pl.Error == "" {
			m := &pl.Metrics
			summary.PromptLookupAcceptanceRate = m.AcceptanceRate
			summary.PromptLookupAcceptedTokens = m.AcceptedTokens
			summary.PromptLookupRejectedTokens = m.RejectedTokens
		}
	}
	summary.AdapterLoadDuration = report.Adapter.Load.Duration
	summary.AdapterFuseDuration = report.Adapter.Fuse.Duration
	// Cache the residency sub-report pointer when reading the Stats
	// block so we don't pay the chained field-offset compute on every
	// summary field — eight stats reads collapse to one cached pointer
	// plus eight fixed-offset loads.
	if er := &report.ExpertResidency; er.Attempted && er.Error == "" {
		stats := &er.Stats
		summary.ExpertResidencyResidentExperts = stats.ResidentExperts
		summary.ExpertResidencyPeakResidentExperts = stats.PeakResidentExperts
		summary.ExpertResidencyPageIns = stats.PageIns
		summary.ExpertResidencyPageOuts = stats.PageOuts
		summary.ExpertResidencyLoadedBytes = stats.LoadedBytes
		summary.ExpertResidencyEvictedBytes = stats.EvictedBytes
		summary.ExpertResidencyFirstUseLatency = stats.FirstUseLatency
		summary.ExpertResidencyTotalLoadDuration = stats.TotalLoadDuration
	}
	// Eval metrics are read four times — cache the sub-block pointer to
	// match the residency pattern.
	em := &report.Evaluation.Metrics
	summary.EvalSamples = em.Samples
	summary.EvalTokens = em.Tokens
	summary.EvalLoss = em.Loss
	summary.Perplexity = em.Perplexity
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
		// Adapters built without an explicit TargetKeys override carry
		// a nil slice — match cloneWorkloadAdapterInfo by guarding the
		// SliceClone behind a len>0 check so the no-targets branch
		// pays only a nil-check instead of the slices.Clone shape.
		if len(adapter.Config.TargetKeys) > 0 {
			info.TargetKeys = core.SliceClone(adapter.Config.TargetKeys)
		}
	}
	return info
}

func cloneWorkloadAdapterInfo(info WorkloadAdapterInfo) WorkloadAdapterInfo {
	// Skip the SliceClone call entirely when TargetKeys is empty —
	// core.SliceClone → slices.Clone hits the make+copy path even for
	// zero-length slices unless the input is nil, and a nil-check here
	// pre-empts the generic call+return-path overhead on the common
	// "adapter has no explicit target overrides" branch.
	if len(info.TargetKeys) > 0 {
		info.TargetKeys = core.SliceClone(info.TargetKeys)
	}
	return info
}

func cloneWorkloadEvalSamples(samples []WorkloadEvalSample) []WorkloadEvalSample {
	if len(samples) == 0 {
		return nil
	}
	// Bulk-copy the sample headers in one shot — the previous loop
	// re-copied the WorkloadEvalSample struct (string headers + map
	// pointer) twice per iteration via `range sample, out[i] = sample`.
	// `copy` is a memmove and lets us index `samples[i].Meta` directly
	// without taking a fresh per-iteration sample copy. The Meta clone
	// is skipped for nil maps so the API/internal "no metadata" path
	// pays only the slice alloc.
	out := make([]WorkloadEvalSample, len(samples))
	copy(out, samples)
	for i := range samples {
		if meta := samples[i].Meta; meta != nil {
			out[i].Meta = core.MapClone(meta)
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

func normalizeWorkloadEvalConfig(cfg eval.Config) eval.Config {
	if batch, ok := cfg.Batch.(dataset.BatchConfig); ok {
		cfg.Batch = normalizeDatasetBatchConfig(batch)
	}
	// QualityProbes defaults to nil for callers that don't wire a
	// custom probe set — guarding the clone keeps the workload bench
	// normalisation hot path (called once per RunWorkloadBench plus
	// every cfg-without-probes dispatch) free of the SliceClone
	// generic-dispatch+append shape on the empty slice.
	if len(cfg.QualityProbes) > 0 {
		cfg.QualityProbes = core.SliceClone(cfg.QualityProbes)
	}
	return cfg
}

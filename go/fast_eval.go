// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"
	"time"

	core "dappco.re/go"
)

const FastEvalReportVersion = 1

// FastEvalConfig controls the first-party local benchmark/eval harness.
type FastEvalConfig struct {
	Model                       string   `json:"model,omitempty"`
	ModelPath                   string   `json:"model_path,omitempty"`
	Prompt                      string   `json:"prompt"`
	CachePrompt                 string   `json:"cache_prompt,omitempty"`
	MaxTokens                   int      `json:"max_tokens"`
	Runs                        int      `json:"runs"`
	Temperature                 float32  `json:"temperature"`
	TopK                        int      `json:"top_k,omitempty"`
	TopP                        float32  `json:"top_p,omitempty"`
	MinP                        float32  `json:"min_p,omitempty"`
	StopTokens                  []int32  `json:"stop_tokens,omitempty"`
	RepeatPenalty               float32  `json:"repeat_penalty,omitempty"`
	IncludePromptCache          bool     `json:"include_prompt_cache"`
	IncludeKVRestore            bool     `json:"include_kv_restore"`
	IncludeStateBundleRoundTrip bool     `json:"include_state_bundle_round_trip"`
	IncludeProbeOverhead        bool     `json:"include_probe_overhead"`
	QualityPrompts              []string `json:"quality_prompts,omitempty"`
}

// DefaultFastEvalConfig returns a short local benchmark suite suitable for a laptop.
func DefaultFastEvalConfig() FastEvalConfig {
	return FastEvalConfig{
		Prompt:                      "Write one precise sentence about local inference.",
		MaxTokens:                   32,
		Runs:                        1,
		Temperature:                 0,
		IncludePromptCache:          true,
		IncludeKVRestore:            true,
		IncludeStateBundleRoundTrip: true,
		IncludeProbeOverhead:        true,
	}
}

// FastEvalRunner is the small model surface required by RunFastEval.
type FastEvalRunner struct {
	Info            func(context.Context) ModelInfo
	Generate        func(context.Context, string, GenerateConfig) (FastEvalGeneration, error)
	WarmPromptCache func(context.Context, string) error
	CaptureKV       func(context.Context, string) (*KVSnapshot, error)
	RestoreKV       func(context.Context, *KVSnapshot) error
}

// FastEvalGeneration is one generation result plus the model metrics it produced.
type FastEvalGeneration struct {
	Text    string  `json:"text,omitempty"`
	Metrics Metrics `json:"metrics"`
}

// FastEvalReport is the JSON-friendly local benchmark/eval result.
type FastEvalReport struct {
	Version     int                       `json:"version"`
	Model       string                    `json:"model,omitempty"`
	ModelPath   string                    `json:"model_path,omitempty"`
	ModelInfo   ModelInfo                 `json:"model_info"`
	Config      FastEvalConfig            `json:"config"`
	Generation  FastEvalGenerationSummary `json:"generation"`
	PromptCache FastEvalPromptCacheReport `json:"prompt_cache"`
	KVRestore   FastEvalLatencyReport     `json:"kv_restore"`
	StateBundle FastEvalStateBundleReport `json:"state_bundle"`
	Probes      FastEvalProbeReport       `json:"probes"`
	Quality     FastEvalQualityReport     `json:"quality"`
}

// FastEvalGenerationSample stores one measured generation pass.
type FastEvalGenerationSample struct {
	Prompt  string        `json:"prompt"`
	Text    string        `json:"text,omitempty"`
	Metrics Metrics       `json:"metrics"`
	Elapsed time.Duration `json:"elapsed"`
}

// FastEvalGenerationSummary aggregates baseline generation passes.
type FastEvalGenerationSummary struct {
	Runs                int                        `json:"runs"`
	PromptTokens        int                        `json:"prompt_tokens"`
	GeneratedTokens     int                        `json:"generated_tokens"`
	PrefillTokensPerSec float64                    `json:"prefill_tokens_per_sec"`
	DecodeTokensPerSec  float64                    `json:"decode_tokens_per_sec"`
	PrefillDuration     time.Duration              `json:"prefill_duration"`
	DecodeDuration      time.Duration              `json:"decode_duration"`
	TotalDuration       time.Duration              `json:"total_duration"`
	PeakMemoryBytes     uint64                     `json:"peak_memory_bytes"`
	ActiveMemoryBytes   uint64                     `json:"active_memory_bytes"`
	Samples             []FastEvalGenerationSample `json:"samples,omitempty"`
}

// FastEvalPromptCacheReport measures warmed prompt-cache reuse.
type FastEvalPromptCacheReport struct {
	Attempted       bool          `json:"attempted"`
	Hits            int           `json:"hits,omitempty"`
	Misses          int           `json:"misses,omitempty"`
	HitRate         float64       `json:"hit_rate,omitempty"`
	HitTokens       int           `json:"hit_tokens,omitempty"`
	MissTokens      int           `json:"miss_tokens,omitempty"`
	WarmDuration    time.Duration `json:"warm_duration,omitempty"`
	RestoreDuration time.Duration `json:"restore_duration,omitempty"`
	Metrics         Metrics       `json:"metrics,omitempty"`
	Error           string        `json:"error,omitempty"`
}

// FastEvalLatencyReport records a best-effort latency measurement.
type FastEvalLatencyReport struct {
	Attempted bool          `json:"attempted"`
	Duration  time.Duration `json:"duration,omitempty"`
	Error     string        `json:"error,omitempty"`
}

// FastEvalStateBundleReport records state-bundle JSON round-trip behavior.
type FastEvalStateBundleReport struct {
	Attempted bool          `json:"attempted"`
	Duration  time.Duration `json:"duration,omitempty"`
	Bytes     int           `json:"bytes,omitempty"`
	Error     string        `json:"error,omitempty"`
}

// FastEvalProbeReport records probe event count and estimated runtime overhead.
type FastEvalProbeReport struct {
	Attempted     bool           `json:"attempted"`
	EventCount    int            `json:"event_count,omitempty"`
	KindCounts    map[string]int `json:"kind_counts,omitempty"`
	Duration      time.Duration  `json:"duration,omitempty"`
	OverheadRatio float64        `json:"overhead_ratio,omitempty"`
	Metrics       Metrics        `json:"metrics,omitempty"`
	Error         string         `json:"error,omitempty"`
	Events        []ProbeEvent   `json:"events,omitempty"`
}

// FastEvalQualityReport contains small deterministic checks over generated text and probes.
type FastEvalQualityReport struct {
	Checks []FastEvalQualityCheck `json:"checks,omitempty"`
}

// FastEvalQualityCheck is a small pass/fail eval item.
type FastEvalQualityCheck struct {
	Name   string  `json:"name"`
	Pass   bool    `json:"pass"`
	Score  float64 `json:"score"`
	Detail string  `json:"detail,omitempty"`
}

// NewModelFastEvalRunner adapts a loaded Model to the benchmark harness.
func NewModelFastEvalRunner(model *Model) FastEvalRunner {
	return FastEvalRunner{
		Info: func(ctx context.Context) ModelInfo {
			if err := ctx.Err(); err != nil {
				return ModelInfo{}
			}
			return model.Info()
		},
		Generate: func(ctx context.Context, prompt string, cfg GenerateConfig) (FastEvalGeneration, error) {
			if err := ctx.Err(); err != nil {
				return FastEvalGeneration{}, err
			}
			text, err := model.Generate(prompt, fastEvalGenerateOptions(cfg)...)
			return FastEvalGeneration{Text: text, Metrics: model.Metrics()}, err
		},
		WarmPromptCache: func(ctx context.Context, prompt string) error {
			if err := ctx.Err(); err != nil {
				return err
			}
			return model.WarmPromptCache(prompt)
		},
		CaptureKV: func(ctx context.Context, prompt string) (*KVSnapshot, error) {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
			return model.CaptureKV(prompt)
		},
		RestoreKV: func(ctx context.Context, snapshot *KVSnapshot) error {
			if err := ctx.Err(); err != nil {
				return err
			}
			session, err := model.NewSessionFromKV(snapshot)
			if err != nil {
				return err
			}
			if session != nil {
				return session.Close()
			}
			return nil
		},
	}
}

// RunFastEvalBench runs the benchmark harness against a loaded Model.
func RunFastEvalBench(ctx context.Context, model *Model, cfg FastEvalConfig) (*FastEvalReport, error) {
	if model == nil {
		return nil, core.NewError("mlx: model is nil")
	}
	return RunFastEval(ctx, NewModelFastEvalRunner(model), cfg)
}

// RunFastEval runs a local benchmark/eval suite against the supplied runner.
func RunFastEval(ctx context.Context, runner FastEvalRunner, cfg FastEvalConfig) (*FastEvalReport, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	cfg = normalizeFastEvalConfig(cfg)
	if runner.Generate == nil {
		return nil, core.NewError("mlx: fast eval runner requires Generate")
	}
	report := &FastEvalReport{
		Version:   FastEvalReportVersion,
		Model:     cfg.Model,
		ModelPath: cfg.ModelPath,
		Config:    cfg,
	}
	if runner.Info != nil {
		report.ModelInfo = runner.Info(ctx)
	}

	var samples []FastEvalGenerationSample
	for range cfg.Runs {
		sample, err := runFastEvalGeneration(ctx, runner, cfg.Prompt, cfg.generateConfig(nil))
		if err != nil {
			return nil, err
		}
		samples = append(samples, sample)
	}
	report.Generation = summarizeFastEvalGenerations(samples)
	report.Quality.Checks = append(report.Quality.Checks, qualityChecks(samples)...)

	var snapshot *KVSnapshot
	if cfg.IncludePromptCache {
		report.PromptCache = runFastEvalPromptCache(ctx, runner, cfg)
	}
	if cfg.IncludeKVRestore || cfg.IncludeStateBundleRoundTrip {
		snapshot = runFastEvalCapture(ctx, runner, cfg)
	}
	if cfg.IncludeKVRestore {
		report.KVRestore = runFastEvalRestore(ctx, runner, snapshot)
	}
	if cfg.IncludeStateBundleRoundTrip {
		report.StateBundle = runFastEvalStateBundle(ctx, snapshot, cfg, report.ModelInfo)
	}
	if cfg.IncludeProbeOverhead {
		report.Probes = runFastEvalProbes(ctx, runner, cfg, report.Generation.TotalDuration)
	}
	return report, nil
}

func normalizeFastEvalConfig(cfg FastEvalConfig) FastEvalConfig {
	def := DefaultFastEvalConfig()
	if fastEvalConfigZero(cfg) {
		return def
	}
	if cfg.Prompt == "" {
		cfg.Prompt = def.Prompt
	}
	if cfg.MaxTokens <= 0 {
		cfg.MaxTokens = def.MaxTokens
	}
	if cfg.Runs <= 0 {
		cfg.Runs = def.Runs
	}
	if cfg.CachePrompt == "" {
		cfg.CachePrompt = cfg.Prompt
	}
	cfg.StopTokens = append([]int32(nil), cfg.StopTokens...)
	cfg.QualityPrompts = append([]string(nil), cfg.QualityPrompts...)
	return cfg
}

func fastEvalConfigZero(cfg FastEvalConfig) bool {
	return cfg.Model == "" &&
		cfg.ModelPath == "" &&
		cfg.Prompt == "" &&
		cfg.CachePrompt == "" &&
		cfg.MaxTokens == 0 &&
		cfg.Runs == 0 &&
		cfg.Temperature == 0 &&
		cfg.TopK == 0 &&
		cfg.TopP == 0 &&
		cfg.MinP == 0 &&
		len(cfg.StopTokens) == 0 &&
		cfg.RepeatPenalty == 0 &&
		!cfg.IncludePromptCache &&
		!cfg.IncludeKVRestore &&
		!cfg.IncludeStateBundleRoundTrip &&
		!cfg.IncludeProbeOverhead &&
		len(cfg.QualityPrompts) == 0
}

func (cfg FastEvalConfig) generateConfig(sink ProbeSink) GenerateConfig {
	return GenerateConfig{
		MaxTokens:     cfg.MaxTokens,
		Temperature:   cfg.Temperature,
		TopK:          cfg.TopK,
		TopP:          cfg.TopP,
		MinP:          cfg.MinP,
		StopTokens:    append([]int32(nil), cfg.StopTokens...),
		RepeatPenalty: cfg.RepeatPenalty,
		ProbeSink:     sink,
	}
}

func fastEvalGenerateOptions(cfg GenerateConfig) []GenerateOption {
	opts := []GenerateOption{
		WithMaxTokens(cfg.MaxTokens),
		WithTemperature(cfg.Temperature),
	}
	if cfg.TopK > 0 {
		opts = append(opts, WithTopK(cfg.TopK))
	}
	if cfg.TopP > 0 {
		opts = append(opts, WithTopP(cfg.TopP))
	}
	if cfg.MinP > 0 {
		opts = append(opts, WithMinP(cfg.MinP))
	}
	if len(cfg.StopTokens) > 0 {
		opts = append(opts, WithStopTokens(cfg.StopTokens...))
	}
	if cfg.RepeatPenalty > 0 {
		opts = append(opts, WithRepeatPenalty(cfg.RepeatPenalty))
	}
	if cfg.ProbeSink != nil {
		opts = append(opts, WithProbeSink(cfg.ProbeSink))
	}
	return opts
}

func runFastEvalGeneration(ctx context.Context, runner FastEvalRunner, prompt string, cfg GenerateConfig) (FastEvalGenerationSample, error) {
	start := time.Now()
	generation, err := runner.Generate(ctx, prompt, cfg)
	elapsed := time.Since(start)
	if err != nil {
		return FastEvalGenerationSample{}, err
	}
	return FastEvalGenerationSample{
		Prompt:  prompt,
		Text:    generation.Text,
		Metrics: generation.Metrics,
		Elapsed: elapsed,
	}, nil
}

func summarizeFastEvalGenerations(samples []FastEvalGenerationSample) FastEvalGenerationSummary {
	summary := FastEvalGenerationSummary{
		Runs:    len(samples),
		Samples: append([]FastEvalGenerationSample(nil), samples...),
	}
	var prefillRateTotal, decodeRateTotal float64
	for _, sample := range samples {
		metrics := sample.Metrics
		summary.PromptTokens += metrics.PromptTokens
		summary.GeneratedTokens += metrics.GeneratedTokens
		summary.PrefillDuration += metrics.PrefillDuration
		summary.DecodeDuration += metrics.DecodeDuration
		if metrics.TotalDuration > 0 {
			summary.TotalDuration += metrics.TotalDuration
		} else {
			summary.TotalDuration += sample.Elapsed
		}
		prefillRateTotal += metrics.PrefillTokensPerSec
		decodeRateTotal += metrics.DecodeTokensPerSec
		if metrics.PeakMemoryBytes > summary.PeakMemoryBytes {
			summary.PeakMemoryBytes = metrics.PeakMemoryBytes
		}
		if metrics.ActiveMemoryBytes > summary.ActiveMemoryBytes {
			summary.ActiveMemoryBytes = metrics.ActiveMemoryBytes
		}
	}
	if len(samples) > 0 {
		summary.PrefillTokensPerSec = prefillRateTotal / float64(len(samples))
		summary.DecodeTokensPerSec = decodeRateTotal / float64(len(samples))
	}
	return summary
}

func runFastEvalPromptCache(ctx context.Context, runner FastEvalRunner, cfg FastEvalConfig) FastEvalPromptCacheReport {
	report := FastEvalPromptCacheReport{Attempted: true}
	if runner.WarmPromptCache == nil {
		report.Error = "runner does not support prompt cache warming"
		return report
	}
	start := time.Now()
	if err := runner.WarmPromptCache(ctx, cfg.CachePrompt); err != nil {
		report.WarmDuration = time.Since(start)
		report.Error = err.Error()
		return report
	}
	report.WarmDuration = time.Since(start)
	sample, err := runFastEvalGeneration(ctx, runner, cfg.CachePrompt, cfg.generateConfig(nil))
	if err != nil {
		report.Error = err.Error()
		return report
	}
	metrics := sample.Metrics
	report.Metrics = metrics
	report.Hits = metrics.PromptCacheHits
	report.Misses = metrics.PromptCacheMisses
	report.HitTokens = metrics.PromptCacheHitTokens
	report.MissTokens = metrics.PromptCacheMissTokens
	report.RestoreDuration = metrics.PromptCacheRestoreDuration
	trials := report.Hits + report.Misses
	if trials == 0 {
		trials = 1
		if report.HitTokens > 0 {
			report.Hits = 1
		} else {
			report.Misses = 1
		}
	}
	report.HitRate = float64(report.Hits) / float64(trials)
	return report
}

func runFastEvalCapture(ctx context.Context, runner FastEvalRunner, cfg FastEvalConfig) *KVSnapshot {
	if runner.CaptureKV == nil {
		return nil
	}
	snapshot, err := runner.CaptureKV(ctx, cfg.CachePrompt)
	if err != nil {
		return nil
	}
	return snapshot
}

func runFastEvalRestore(ctx context.Context, runner FastEvalRunner, snapshot *KVSnapshot) FastEvalLatencyReport {
	report := FastEvalLatencyReport{Attempted: true}
	if snapshot == nil {
		report.Error = "no KV snapshot captured"
		return report
	}
	if runner.RestoreKV == nil {
		report.Error = "runner does not support KV restore"
		return report
	}
	start := time.Now()
	if err := runner.RestoreKV(ctx, snapshot); err != nil {
		report.Duration = time.Since(start)
		report.Error = err.Error()
		return report
	}
	report.Duration = time.Since(start)
	return report
}

func runFastEvalStateBundle(ctx context.Context, snapshot *KVSnapshot, cfg FastEvalConfig, info ModelInfo) FastEvalStateBundleReport {
	report := FastEvalStateBundleReport{Attempted: true}
	if snapshot == nil {
		report.Error = "no KV snapshot captured"
		return report
	}
	start := time.Now()
	bundle, err := NewStateBundle(snapshot, StateBundleOptions{
		Model:     cfg.Model,
		ModelPath: cfg.ModelPath,
		ModelInfo: info,
		Prompt:    cfg.CachePrompt,
		Sampler:   cfg.generateConfig(nil),
	})
	if err != nil {
		report.Duration = time.Since(start)
		report.Error = err.Error()
		return report
	}
	data := core.JSONMarshal(bundle)
	if !data.OK {
		report.Duration = time.Since(start)
		report.Error = fastEvalResultError(data).Error()
		return report
	}
	raw := data.Value.([]byte)
	var decoded StateBundle
	if result := core.JSONUnmarshal(raw, &decoded); !result.OK {
		report.Duration = time.Since(start)
		report.Error = fastEvalResultError(result).Error()
		return report
	}
	if err := decoded.Validate(); err != nil {
		report.Duration = time.Since(start)
		report.Error = err.Error()
		return report
	}
	if _, err := decoded.Snapshot(); err != nil {
		report.Duration = time.Since(start)
		report.Error = err.Error()
		return report
	}
	select {
	case <-ctx.Done():
		report.Duration = time.Since(start)
		report.Error = ctx.Err().Error()
		return report
	default:
	}
	report.Duration = time.Since(start)
	report.Bytes = len(raw)
	return report
}

func runFastEvalProbes(ctx context.Context, runner FastEvalRunner, cfg FastEvalConfig, baseline time.Duration) FastEvalProbeReport {
	report := FastEvalProbeReport{Attempted: true}
	recorder := NewProbeRecorder()
	sample, err := runFastEvalGeneration(ctx, runner, cfg.Prompt, cfg.generateConfig(recorder))
	if err != nil {
		report.Error = err.Error()
		return report
	}
	events := recorder.Events()
	report.EventCount = len(events)
	report.KindCounts = make(map[string]int)
	for _, event := range events {
		report.KindCounts[string(event.Kind)]++
	}
	report.Events = events
	report.Metrics = sample.Metrics
	report.Duration = sample.Metrics.TotalDuration
	if report.Duration == 0 {
		report.Duration = sample.Elapsed
	}
	if baseline > 0 {
		report.OverheadRatio = float64(report.Duration-baseline) / float64(baseline)
	}
	return report
}

func qualityChecks(samples []FastEvalGenerationSample) []FastEvalQualityCheck {
	var checks []FastEvalQualityCheck
	nonEmpty := false
	generatedTokens := 0
	for _, sample := range samples {
		if sample.Text != "" {
			nonEmpty = true
		}
		generatedTokens += sample.Metrics.GeneratedTokens
	}
	checks = append(checks, FastEvalQualityCheck{
		Name:  "non_empty_output",
		Pass:  nonEmpty,
		Score: boolScore(nonEmpty),
	})
	checks = append(checks, FastEvalQualityCheck{
		Name:   "generated_tokens",
		Pass:   generatedTokens > 0,
		Score:  boolScore(generatedTokens > 0),
		Detail: core.Sprintf("%d", generatedTokens),
	})
	return checks
}

func boolScore(pass bool) float64 {
	if pass {
		return 1
	}
	return 0
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

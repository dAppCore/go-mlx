// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"context"
	"flag"
	"io"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference/bench"
	mlx "dappco.re/go/mlx"
	"dappco.re/go/mlx/benchsummary"
)

type benchCommandReport struct {
	*bench.Report
	SpeculativeAssistantLayout *mlx.SpeculativeAssistantLayout `json:"speculative_assistant_layout,omitempty"`
	SpeculativeMTPMetrics      *benchCommandMTPMetrics         `json:"speculative_mtp_metrics,omitempty"`
}

type benchCommandMTPMetrics struct {
	TargetOnlyTokensPerSec float64       `json:"target_only_tokens_per_sec,omitempty"`
	DraftTokenSchedule     []int         `json:"draft_token_schedule,omitempty"`
	ProposedTokens         int           `json:"proposed_tokens,omitempty"`
	AcceptedTokens         int           `json:"accepted_tokens,omitempty"`
	RejectedTokens         int           `json:"rejected_tokens,omitempty"`
	TargetVerifyCalls      int           `json:"target_verify_calls,omitempty"`
	TargetCalls            int           `json:"target_calls,omitempty"`
	DraftCalls             int           `json:"draft_calls,omitempty"`
	AcceptanceRate         float64       `json:"acceptance_rate,omitempty"`
	VisibleTokensPerSec    float64       `json:"visible_tokens_per_sec,omitempty"`
	TargetTokensPerSec     float64       `json:"target_tokens_per_sec,omitempty"`
	WarmDecodeTokensPerSec float64       `json:"warm_decode_tokens_per_sec,omitempty"`
	WallDuration           time.Duration `json:"wall_duration,omitempty"`
	RestoreDuration        time.Duration `json:"restore_duration,omitempty"`
	TargetVerifyDuration   time.Duration `json:"target_verify_duration,omitempty"`
	TargetDuration         time.Duration `json:"target_duration,omitempty"`
	DraftDuration          time.Duration `json:"draft_duration,omitempty"`
	PeakMemoryBytes        uint64        `json:"peak_memory_bytes,omitempty"`
	QualityFlags           []string      `json:"quality_flags"`
}

func runBenchCommand(ctx context.Context, args []string, stdout, stderr io.Writer) int {
	cfg := bench.DefaultConfig()
	fs := flag.NewFlagSet(cliCommandName("bench"), flag.ContinueOnError)
	fs.SetOutput(stderr)
	jsonOut := fs.Bool("json", false, "print JSON report")
	profilePath := fs.String("profile", "", "saved tuning profile to apply before loading the model")
	prompt := fs.String("prompt", cfg.Prompt, "baseline benchmark prompt")
	promptFile := fs.String("prompt-file", "", "read baseline benchmark prompt text from a file")
	promptRepeat := fs.Int("prompt-repeat", 1, "repeat the resolved benchmark prompt N times")
	promptSuffix := fs.String("prompt-suffix", "", "append extra text to the resolved benchmark prompt")
	promptSuffixFile := fs.String("prompt-suffix-file", "", "read prompt suffix text from a file")
	cachePrompt := fs.String("cache-prompt", "", "stable prompt used for prompt-cache and KV restore checks")
	maxTokens := fs.Int("max-tokens", cfg.MaxTokens, "generated tokens per pass")
	runs := fs.Int("runs", cfg.Runs, "baseline generation passes")
	contextLen := fs.Int("context", 0, "override context length")
	prefillChunkSize := fs.Int("prefill-chunk-size", 0, "override long-prompt prefill chunk size in tokens")
	cacheMode := fs.String("cache-mode", "", cacheModeFlagUsage)
	device := fs.String("device", "", "execution device: gpu or cpu")
	fastGemma4Lane := fs.Bool("fast-gemma4-lane", true, "enable the accepted Gemma 4 fast runtime gates by default; set false for baseline diagnostics")
	speculativeDraftModel := fs.String("speculative-draft-model", "", "assistant/draft model path for speculative decode metrics")
	speculativeDraftTokens := fs.Int("speculative-draft-tokens", mlx.ProductionMTPDefaultDraftTokens, "draft tokens proposed per speculative decode pass")
	noCache := fs.Bool("no-cache", false, "skip prompt-cache warm/hit check")
	noRestore := fs.Bool("no-restore", false, "skip KV restore latency check")
	noBundle := fs.Bool("no-bundle", false, "skip state-bundle round trip check")
	noProbes := fs.Bool("no-probes", false, "skip probe overhead check")
	stateKVWarm := fs.Bool("state-kv-warm", false, "include State KV block build, restore, and warmed generation check")
	stateKVBlockSize := fs.Int("state-kv-block-size", 0, "State KV block size in tokens; 0 uses the runtime default")
	stateKVPrefixTokens := fs.Int("state-kv-prefix-tokens", 0, "tokens to restore from State KV blocks; 0 restores the full captured prefix")
	stateKVStore := fs.String("state-kv-store", "", "path for the State KV block store; empty uses a temporary file")
	memvidKVWarm := fs.Bool("memvid-kv-warm", false, "deprecated alias for -state-kv-warm")
	memvidKVBlockSize := fs.Int("memvid-kv-block-size", 0, "deprecated alias for -state-kv-block-size")
	memvidKVPrefixTokens := fs.Int("memvid-kv-prefix-tokens", 0, "deprecated alias for -state-kv-prefix-tokens")
	memvidKVStore := fs.String("memvid-kv-store", "", "deprecated alias for -state-kv-store")
	fs.Usage = func() {
		name := cliName()
		core.WriteString(stderr, core.Sprintf("Usage: %s bench [flags] <model-path>\n", name))
		core.WriteString(stderr, "\n")
		core.WriteString(stderr, "Single-shot benchmark against a model — measures prefill / decode\n")
		core.WriteString(stderr, "throughput, prompt-cache hit rate, KV-restore latency, and state-bundle\n")
		core.WriteString(stderr, "round-trip. Use this to verify a model loads and to compare runtime\n")
		core.WriteString(stderr, "settings before running tune-plan.\n")
		core.WriteString(stderr, "\n")
		core.WriteString(stderr, "Flags:\n")
		fs.VisitAll(func(f *flag.Flag) {
			if f.DefValue == "" {
				core.WriteString(stderr, core.Sprintf("  -%s\n\t%s\n", f.Name, f.Usage))
				return
			}
			core.WriteString(stderr, core.Sprintf("  -%s\n\t%s (default %q)\n", f.Name, f.Usage, f.DefValue))
		})
		core.WriteString(stderr, "\n")
		core.WriteString(stderr, "Examples:\n")
		core.WriteString(stderr, core.Sprintf("  %s bench ~/models/lemer-lite\n", name))
		core.WriteString(stderr, core.Sprintf("    # default 32 tokens, baseline prompt — quick sanity check\n"))
		core.WriteString(stderr, core.Sprintf("  %s bench -max-tokens 256 -runs 3 ~/models/lemer-lite\n", name))
		core.WriteString(stderr, core.Sprintf("    # longer generations, 3-pass average\n"))
		core.WriteString(stderr, core.Sprintf("  %s bench -json -no-bundle ~/models/lemer-lite\n", name))
		core.WriteString(stderr, core.Sprintf("    # machine-readable output, skip state-bundle round trip\n"))
		core.WriteString(stderr, core.Sprintf("  %s bench -profile ~/profiles/lemer-lite-m3ultra.json\n", name))
		core.WriteString(stderr, core.Sprintf("    # apply a saved tune profile (model path embedded in profile)\n"))
	}
	if err := fs.Parse(args); err != nil {
		if core.Is(err, flag.ErrHelp) {
			return 0
		}
		return 2
	}
	visitedFlags := driverProfileVisitedFlags(fs)
	if driverProfileFastGemma4LaneEnabled(*fastGemma4Lane, visitedFlags, *profilePath) {
		for _, restore := range applyGemma4FastLaneDefaults(
			visitedFlags,
			contextLen,
			cacheMode,
			prefillChunkSize,
			nil,
			mlx.ProductionLaneContextLength,
		) {
			defer restore()
		}
	}
	if fs.NArg() > 1 || (fs.NArg() == 0 && core.Trim(*profilePath) == "") {
		core.WriteString(stderr, core.Sprintf("%s bench: expected one model path or -profile\n", cliName()))
		fs.Usage()
		return 2
	}
	if *promptRepeat < 1 {
		core.WriteString(stderr, core.Sprintf("%s bench: prompt repeat must be >= 1\n", cliName()))
		return 2
	}
	if *stateKVBlockSize < 0 || *memvidKVBlockSize < 0 {
		core.WriteString(stderr, core.Sprintf("%s bench: State KV block size must be >= 0\n", cliName()))
		return 2
	}
	if *stateKVPrefixTokens < 0 || *memvidKVPrefixTokens < 0 {
		core.WriteString(stderr, core.Sprintf("%s bench: State KV prefix tokens must be >= 0\n", cliName()))
		return 2
	}
	if *prefillChunkSize < 0 {
		core.WriteString(stderr, core.Sprintf("%s bench: prefill chunk size must be >= 0\n", cliName()))
		return 2
	}
	if core.Trim(*promptFile) != "" {
		read := core.ReadFile(*promptFile)
		if !read.OK {
			core.Print(stderr, "%s bench: prompt file: %v", cliName(), read.Value)
			return 1
		}
		*prompt = string(read.Value.([]byte))
	}
	if core.Trim(*promptSuffixFile) != "" {
		read := core.ReadFile(*promptSuffixFile)
		if !read.OK {
			core.Print(stderr, "%s bench: prompt suffix file: %v", cliName(), read.Value)
			return 1
		}
		*promptSuffix = string(read.Value.([]byte))
	}
	resolvedPrompt := appendDriverProfilePromptSuffix(repeatDriverProfilePrompt(*prompt, *promptRepeat), *promptSuffix)

	modelPath := ""
	loadOptions := []mlx.LoadOption{}
	if core.Trim(*profilePath) != "" {
		report, err := readTuneProfileReport(*profilePath)
		if err != nil {
			core.Print(stderr, "%s bench: profile: %v", cliName(), err)
			return 1
		}
		if report.Profile == nil {
			core.Print(stderr, "%s bench: profile payload missing", cliName())
			return 1
		}
		modelPath = report.ModelPath
		loadOptions = append(loadOptions, mlx.TuningCandidateLoadOptions(report.Profile.Candidate)...)
	}
	if fs.NArg() == 1 {
		modelPath = fs.Arg(0)
	}
	if core.Trim(modelPath) == "" {
		core.WriteString(stderr, core.Sprintf("%s bench: model path missing from profile\n", cliName()))
		fs.Usage()
		return 2
	}
	cfg.Model = core.PathBase(modelPath)
	cfg.ModelPath = modelPath
	cfg.Prompt = resolvedPrompt
	cfg.CachePrompt = *cachePrompt
	cfg.MaxTokens = *maxTokens
	cfg.Runs = *runs
	cfg.IncludePromptCache = !*noCache
	cfg.IncludeKVRestore = !*noRestore
	cfg.IncludeStateBundleRoundTrip = !*noBundle
	cfg.IncludeProbeOverhead = !*noProbes
	if *memvidKVWarm {
		*stateKVWarm = true
	}
	if *stateKVBlockSize == 0 && *memvidKVBlockSize != 0 {
		*stateKVBlockSize = *memvidKVBlockSize
	}
	if *stateKVPrefixTokens == 0 && *memvidKVPrefixTokens != 0 {
		*stateKVPrefixTokens = *memvidKVPrefixTokens
	}
	if core.Trim(*stateKVStore) == "" && core.Trim(*memvidKVStore) != "" {
		*stateKVStore = core.Trim(*memvidKVStore)
	}
	cfg.IncludeStateKVBlockWarm = *stateKVWarm
	cfg.StateKVBlockSize = *stateKVBlockSize
	cfg.StateKVPrefixTokens = *stateKVPrefixTokens
	cfg.StateKVBlockStorePath = core.Trim(*stateKVStore)
	if *speculativeDraftTokens < 0 {
		core.WriteString(stderr, core.Sprintf("%s bench: speculative draft tokens must be >= 0\n", cliName()))
		return 2
	}
	if core.Trim(*speculativeDraftModel) != "" {
		cfg.IncludeSpeculativeDecode = true
		cfg.SpeculativeDraftModelPath = core.Trim(*speculativeDraftModel)
		cfg.SpeculativeDraftTokens = *speculativeDraftTokens
	}

	if *contextLen > 0 {
		loadOptions = append(loadOptions, mlx.WithContextLength(*contextLen))
	}
	if *prefillChunkSize > 0 {
		loadOptions = append(loadOptions, mlx.WithPrefillChunkSize(*prefillChunkSize))
	}
	if mode, ok := parseRuntimeCacheMode(*cacheMode); ok {
		if !isRuntimeCacheMode(mode) {
			core.WriteString(stderr, core.Sprintf("%s bench: unsupported cache mode %q\n", cliName(), string(mode)))
			return 2
		}
		loadOptions = append(loadOptions, mlx.WithKVCacheMode(mode))
	}
	if *device != "" {
		loadOptions = append(loadOptions, mlx.WithDevice(*device))
	}
	if cfg.IncludeSpeculativeDecode {
		pair, err := loadSpeculativePair(modelPath, cfg.SpeculativeDraftModelPath, mlx.SpeculativePairConfig{
			TargetOptions: loadOptions,
			DraftOptions:  loadOptions,
		})
		if err != nil {
			core.Print(stderr, "%s bench: load speculative pair: %v", cliName(), err)
			return 1
		}
		defer pair.Close()
		var report *bench.Report
		if pair.Gemma4Assistant != nil {
			report, err = runBenchReportWithSpeculativePair(ctx, pair, cfg)
		} else {
			report, err = runBenchReportWithDraft(ctx, pair.Target, pair.Draft, cfg)
		}
		if err != nil {
			core.Print(stderr, "%s bench: %v", cliName(), err)
			return 1
		}
		if *jsonOut {
			data := core.JSONMarshalIndent(benchCommandJSONReport(report, pair.Report.AssistantLayout, pair.Metrics()), "", "  ")
			if !data.OK {
				core.Print(stderr, "%s bench: marshal report failed", cliName())
				return 1
			}
			core.WriteString(stdout, string(data.Value.([]byte)))
			core.WriteString(stdout, "\n")
			return 0
		}
		writeBenchCommandSummary(stdout, report, pair.Report.AssistantLayout, pair.Metrics())
		return 0
	}
	model, err := loadBenchModel(modelPath, loadOptions...)
	if err != nil {
		core.Print(stderr, "%s bench: load model: %v", cliName(), err)
		return 1
	}
	defer model.Close()

	report, err := runBenchReport(ctx, model, cfg)
	if err != nil {
		core.Print(stderr, "%s bench: %v", cliName(), err)
		return 1
	}
	if *jsonOut {
		data := core.JSONMarshalIndent(benchCommandJSONReport(report, nil, mlx.Metrics{}), "", "  ")
		if !data.OK {
			core.Print(stderr, "%s bench: marshal report failed", cliName())
			return 1
		}
		core.WriteString(stdout, string(data.Value.([]byte)))
		core.WriteString(stdout, "\n")
		return 0
	}
	writeBenchCommandSummary(stdout, report, nil, mlx.Metrics{})
	return 0
}

func benchCommandJSONReport(report *bench.Report, layout *mlx.SpeculativeAssistantLayout, metrics mlx.Metrics) any {
	mtpMetrics := benchCommandMTPMetricsFromReport(report, metrics.MTP)
	if layout == nil && mtpMetrics == nil {
		return report
	}
	return benchCommandReport{
		Report:                     report,
		SpeculativeAssistantLayout: layout,
		SpeculativeMTPMetrics:      mtpMetrics,
	}
}

func benchCommandMTPMetricsFromReport(report *bench.Report, metrics *mlx.MTPMetrics) *benchCommandMTPMetrics {
	if metrics == nil {
		return nil
	}
	out := &benchCommandMTPMetrics{
		DraftTokenSchedule:     core.SliceClone(metrics.DraftTokenSchedule),
		ProposedTokens:         metrics.ProposedTokens,
		AcceptedTokens:         metrics.AcceptedTokens,
		RejectedTokens:         metrics.RejectedTokens,
		TargetVerifyCalls:      metrics.TargetVerifyCalls,
		TargetCalls:            metrics.TargetCalls,
		DraftCalls:             metrics.DraftCalls,
		AcceptanceRate:         metrics.AcceptanceRate,
		VisibleTokensPerSec:    metrics.VisibleTokensPerSec,
		TargetTokensPerSec:     metrics.TargetTokensPerSec,
		WarmDecodeTokensPerSec: metrics.WarmDecodeTokensPerSec,
		WallDuration:           metrics.WallDuration,
		RestoreDuration:        metrics.RestoreDuration,
		TargetVerifyDuration:   metrics.TargetVerifyDuration,
		TargetDuration:         metrics.TargetDuration,
		DraftDuration:          metrics.DraftDuration,
		PeakMemoryBytes:        metrics.PeakMemoryBytes,
		QualityFlags:           benchCommandQualityFlags(report),
	}
	if report != nil {
		out.TargetOnlyTokensPerSec = report.Generation.DecodeTokensPerSec
	}
	return out
}

func benchCommandQualityFlags(report *bench.Report) []string {
	flags := []string{}
	if report == nil {
		return flags
	}
	for _, check := range report.Quality.Checks {
		if check.Pass {
			continue
		}
		name := core.Trim(check.Name)
		if name == "" {
			name = "quality_check_failed"
		}
		flags = append(flags, name)
	}
	return flags
}

func writeBenchCommandSummary(stdout io.Writer, report *bench.Report, layout *mlx.SpeculativeAssistantLayout, metrics mlx.Metrics) {
	benchsummary.Write(stdout, report)
	if layout == nil {
		return
	}
	core.WriteString(stdout, core.Sprintf("  assistant: %s, ordered embeddings %t, centroids %d, centroid top-k %d, four-layer %t\n",
		layout.Architecture,
		layout.OrderedEmbeddings,
		layout.Centroids,
		layout.CentroidIntermediateTopK,
		layout.FourLayerDrafter,
	))
	if metrics.MTP != nil {
		core.WriteString(stdout, core.Sprintf("    MTP evidence: schedule %v, warm decode %.1f tok/s, restore %s, verify calls %d, peak memory %d MB\n",
			metrics.MTP.DraftTokenSchedule,
			metrics.MTP.WarmDecodeTokensPerSec,
			metrics.MTP.RestoreDuration,
			metrics.MTP.TargetVerifyCalls,
			metrics.MTP.PeakMemoryBytes/1024/1024,
		))
	}
}

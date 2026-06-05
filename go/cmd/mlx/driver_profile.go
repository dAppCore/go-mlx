// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"context"
	"flag"
	"io"
	"iter"
	"sort"
	"sync"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/bench"
	mlx "dappco.re/go/mlx"
	"dappco.re/go/mlx/memory"
	"dappco.re/go/mlx/pkg/metal"
	speculativeprofile "dappco.re/go/mlx/speculative"
)

func runDriverProfileCommand(ctx context.Context, args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet(cliCommandName("driver-profile"), flag.ContinueOnError)
	fs.SetOutput(stderr)
	productionLane := mlx.DefaultProductionLane()
	jsonOut := fs.Bool("json", false, "print JSON driver profile")
	reportFile := fs.String("report-file", "", "write JSON driver profile to a file")
	profilePath := fs.String("profile", "", "saved tuning profile to apply before loading the model")
	prompt := fs.String("prompt", defaultRetainedProfilePrompt, "prompt/question to run")
	promptFile := fs.String("prompt-file", "", "read prompt/question text from a file")
	promptSuffix := fs.String("prompt-suffix", "", "append one final task after any repeated prompt context")
	promptSuffixFile := fs.String("prompt-suffix-file", "", "read final prompt/task suffix text from a file")
	promptChunkBytes := fs.Int("prompt-chunk-bytes", 0, "split prompt or chat message text into bounded byte chunks before tokenisation")
	promptRepeat := fs.Int("prompt-repeat", 1, "repeat the resolved prompt N times before tokenisation")
	maxTokens := fs.Int("max-tokens", productionLane.MaxTokens, "generated tokens per profiling run")
	runs := fs.Int("runs", productionLane.Runs, "profiling runs to execute")
	includeOutput := fs.Bool("include-output", productionLane.IncludeOutput, "include generated text in the report")
	chat := fs.Bool("chat", true, "run the prompt through the model chat template")
	traceTokenPhases := fs.Bool("trace-token-phases", productionLane.TraceTokenPhases, "include per-token native decode phase timings")
	throughputBenchmark := fs.Bool("throughput-benchmark", false, "profile decode throughput through repetitive output by lifting repetition guard ceilings for this driver-profile run")
	temperature := fs.Float64("temperature", driverProfileDefaultTemperature, "sampling temperature for generated tokens")
	topP := fs.Float64("top-p", driverProfileDefaultTopP, "nucleus sampling top-p")
	topK := fs.Int("top-k", driverProfileDefaultTopK, "top-k sampling candidate count")
	repeatPenalty := fs.Float64("repeat-penalty", driverProfileDefaultRepeatPenalty, "repetition penalty; 1 disables penalty")
	speculativeDraftModel := fs.String("speculative-draft-model", "", "assistant/draft model path for attached-assistant MTP profile metrics")
	speculativeDraftTokens := fs.Int("speculative-draft-tokens", mlx.ProductionMTPDefaultDraftTokens, "draft tokens proposed per attached-assistant MTP pass")
	contextLen := fs.Int("context", 0, "override context length")
	prefillChunkSize := fs.Int("prefill-chunk-size", 0, "override long-prompt prefill chunk size in tokens")
	cacheMode := fs.String("cache-mode", "", cacheModeFlagUsage)
	pagedKVPageSize := fs.Int("paged-kv-page-size", 0, "page size for native paged KV caches; 0 leaves the backend default")
	pagedKVPrealloc := fs.Bool("paged-kv-prealloc", false, "use full-page preallocation for native paged KV caches; lowers MLX residency in some runs but is not a default speed path")
	device := fs.String("device", "", "execution device: gpu or cpu")
	estimatePowerWatts := fs.Float64("estimate-power-watts", 0, "record an estimated average active power draw in watts and derive joule deltas")
	fastGemma4Lane := fs.Bool("fast-gemma4-lane", true, "enable the accepted Gemma 4 fast runtime gates by default; set false for baseline diagnostics")
	expertIDMatVec := fs.Bool("expert-id-matvec", false, "enable the opt-in Gemma 4 expert-ID matvec MoE path")
	expertIDFusedActivation := fs.Bool("expert-id-fused-activation", false, "enable fused activation inside the opt-in expert-ID matvec path")
	sortedExpertPrefill := fs.Bool("sorted-expert-prefill", false, "enable the opt-in Gemma 4 sorted expert prefill MoE path")
	pagedDecodeFastConcat := fs.Bool("paged-decode-fast-concat", false, "enable the opt-in Gemma 4 fast-SDPA concat path for multi-page decode")
	nativePagedAttention := fs.Bool("native-paged-attention", false, "enable the opt-in native C++ paged attention reduction path")
	nativeMLPMatVec := fs.Bool("native-mlp-matvec", false, "enable the opt-in native q4/q6/q8 MLP matvec path")
	nativeLinearMatVec := fs.Bool("native-linear-matvec", false, "enable the opt-in native q4/q6/q8 single-token linear matvec path")
	nativeGemma4FFNResidual := fs.Bool("native-gemma4-ffn-residual", false, "enable the opt-in native Gemma 4 MoE FFN residual path")
	nativeGemma4RouterMatVec := fs.Bool("native-gemma4-router-matvec", false, "enable the opt-in native Gemma 4 router quantized matvec path")
	nativeGemma4RouterTopK := fs.Bool("native-gemma4-router-topk", false, "enable the opt-in native Gemma 4 router top-k path")
	nativeGemma4AttentionOMatVec := fs.Bool("native-gemma4-attention-o-matvec", false, "enable the opt-in native Gemma 4 attention output matvec path")
	nativeGemma4ResidualNorm := fs.Bool("native-gemma4-residual-norm", false, "enable the opt-in native Gemma 4 attention residual norm path")
	nativeGemma4Layer := fs.Bool("native-gemma4-layer", false, "enable the opt-in native Gemma 4 one-token decode layer path")
	nativeGemma4MoELayer := fs.Bool("native-gemma4-moe-layer", false, "enable the opt-in native Gemma 4 MoE layer path")
	compiledGemma4Layer := fs.Bool("compiled-gemma4-layer", false, "enable the opt-in compiled Gemma 4 one-token decode layer path")
	fixedGemma4Cache := fs.Bool("fixed-gemma4-cache", false, "enable the opt-in fixed-size Gemma 4 cache path")
	fixedGemma4SlidingCacheBound := fs.Bool("fixed-gemma4-sliding-cache-bound", false, "enable the opt-in fixed Gemma 4 sliding-window cache bound")
	fixedGemma4SharedMask := fs.Bool("fixed-gemma4-shared-mask", false, "enable the opt-in fixed Gemma 4 shared attention mask path")
	fixedGemma4CacheSize := fs.Int("fixed-gemma4-cache-size", 0, "fixed Gemma 4 cache size in tokens; 0 leaves the runtime default")
	nativeFixedSlidingAttention := fs.Bool("native-fixed-sliding-attention", false, "enable the opt-in native fixed sliding-window attention path")
	nativeGemma4FixedOwnerAttention := fs.Bool("native-gemma4-fixed-owner-attention", false, "enable the opt-in native Gemma 4 fixed-owner attention path")
	nativeGemma4FixedOwnerAttentionResidual := fs.Bool("native-gemma4-fixed-owner-attention-residual", false, "enable the opt-in native Gemma 4 fixed-owner attention residual path")
	fixedWideSDPAAttention := fs.Bool("fixed-wide-sdpa-attention", false, "enable the diagnostic fixed-cache wide-head SDPA attention path")
	fixedWideMatmulAttention := fs.Bool("fixed-wide-matmul-attention", false, "enable the diagnostic fixed-cache wide-head matmul attention path")
	fixedRowCacheUpdate := fs.Bool("fixed-row-cache-update", false, "enable the diagnostic fixed-cache row-update path")
	directGreedyToken := fs.Bool("direct-greedy-token", false, "enable the opt-in direct greedy token decode path")
	generationStream := fs.Bool("generation-stream", false, "enable the opt-in dedicated MLX stream for generation")
	generationClearCache := fs.Bool("generation-clear-cache", false, "clear the MLX allocator cache after prefill chunks and periodically during decode")
	generationClearCacheInterval := fs.Int("generation-clear-cache-interval", 0, "decode-token interval for generation clear-cache mode; 0 leaves the backend default")
	maxActiveMemoryBytes := fs.Uint64("max-active-memory-bytes", 0, "abort a run if MLX active memory exceeds this many bytes; 0 derives from the resolved memory limit")
	maxProcessVirtualMemoryBytes := fs.Uint64("max-process-virtual-memory-bytes", 0, "abort a run if process virtual memory exceeds this many bytes; 0 records process virtual memory without a hard cap")
	maxProcessResidentMemoryBytes := fs.Uint64("max-process-resident-memory-bytes", 0, "abort a run if process resident memory exceeds this many bytes; 0 derives from the resolved memory limit")
	repeatedTokenLoopLimit := fs.Int("repeated-token-loop-limit", driverProfileDefaultRepeatedTokenLoopLimit, "abort when this many consecutive sampled tokens have the same token id")
	repeatedLineLoopLimit := fs.Int("repeated-line-loop-limit", profileDefaultRepeatedLineLoopLimit, "abort when this many consecutive visible non-empty lines repeat")
	repeatedSentenceLoopLimit := fs.Int("repeated-sentence-loop-limit", profileDefaultRepeatedSentenceLoopLimit, "abort when the same visible sentence repeats this many times in one output")
	fs.Usage = func() {
		name := cliName()
		core.WriteString(stderr, core.Sprintf("Usage: %s driver-profile [flags] [model-path]\n", name))
		core.WriteString(stderr, "\n")
		core.WriteString(stderr, "Measure end-to-end timings for one prompt: model load, first-token\n")
		core.WriteString(stderr, "latency, decode throughput, optional per-token native phase trace.\n")
		core.WriteString(stderr, "The single-prompt diagnostic for understanding whether a model + cfg\n")
		core.WriteString(stderr, "combination is fast enough; the long list of `-native-*` and\n")
		core.WriteString(stderr, "`-expert-*` flags toggles opt-in runtime gates for A/B testing\n")
		core.WriteString(stderr, "individual fast-path implementations against the baseline.\n")
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
		core.WriteString(stderr, core.Sprintf("  %s driver-profile ~/models/lemer-lite\n", name))
		core.WriteString(stderr, core.Sprintf("    # default prompt, production lane settings\n"))
		core.WriteString(stderr, core.Sprintf("  %s driver-profile -json -trace-token-phases ~/models/lemer-lite\n", name))
		core.WriteString(stderr, core.Sprintf("    # JSON output + per-token phase breakdown (prefetch / sample_eval / forward)\n"))
		core.WriteString(stderr, core.Sprintf("  %s driver-profile -profile ~/profiles/lemer-lite-chat.json -prompt-file ~/test.txt\n", name))
		core.WriteString(stderr, core.Sprintf("    # apply saved tune profile + run a custom prompt file\n"))
	}
	if err := fs.Parse(args); err != nil {
		if core.Is(err, flag.ErrHelp) {
			return 0
		}
		return 2
	}
	visitedFlags := driverProfileVisitedFlags(fs)
	fastLaneEnabled := driverProfileFastGemma4LaneEnabled(*fastGemma4Lane, visitedFlags, *profilePath)
	if fastLaneEnabled {
		for _, restore := range applyGemma4FastLaneDefaults(
			visitedFlags,
			contextLen,
			cacheMode,
			0,
		) {
			defer restore()
		}
	}
	if fs.NArg() > 1 || (fs.NArg() == 0 && core.Trim(*profilePath) == "") {
		core.WriteString(stderr, core.Sprintf("%s driver-profile: expected one model path or -profile\n", cliName()))
		fs.Usage()
		return 2
	}
	if core.Trim(*promptFile) != "" {
		read := core.ReadFile(*promptFile)
		if !read.OK {
			core.Print(stderr, "%s driver-profile: prompt file: %v", cliName(), read.Value)
			return 1
		}
		*prompt = string(read.Value.([]byte))
	}
	if *promptRepeat < 1 {
		core.WriteString(stderr, core.Sprintf("%s driver-profile: prompt repeat must be >= 1\n", cliName()))
		return 2
	}
	if core.Trim(*promptSuffixFile) != "" {
		read := core.ReadFile(*promptSuffixFile)
		if !read.OK {
			core.Print(stderr, "%s driver-profile: prompt suffix file: %v", cliName(), read.Value)
			return 1
		}
		*promptSuffix = string(read.Value.([]byte))
	}
	*prompt = repeatDriverProfilePrompt(*prompt, *promptRepeat)
	*prompt = appendDriverProfilePromptSuffix(*prompt, *promptSuffix)
	if *expertIDMatVec {
		defer setDriverProfileRuntimeGate("GO_MLX_ENABLE_EXPERT_ID_MATVEC", "1")()
	}
	if *expertIDFusedActivation {
		defer setDriverProfileRuntimeGate("GO_MLX_ENABLE_EXPERT_ID_MATVEC", "1")()
		defer setDriverProfileRuntimeGate("GO_MLX_ENABLE_EXPERT_ID_FUSED_ACTIVATION", "1")()
	}
	if *sortedExpertPrefill {
		defer setDriverProfileRuntimeGate("GO_MLX_ENABLE_SORTED_EXPERT_PREFILL", "1")()
	}
	if *pagedDecodeFastConcat {
		defer setDriverProfileRuntimeGate("GO_MLX_ENABLE_PAGED_DECODE_FAST_CONCAT", "1")()
	}
	if *nativePagedAttention {
		defer setDriverProfileRuntimeGate("GO_MLX_ENABLE_NATIVE_PAGED_ATTENTION", "1")()
	}
	if *nativeMLPMatVec {
		defer setDriverProfileRuntimeGate("GO_MLX_ENABLE_NATIVE_MLP_MATVEC", "1")()
	}
	if *nativeLinearMatVec {
		defer setDriverProfileRuntimeGate("GO_MLX_ENABLE_NATIVE_LINEAR_MATVEC", "1")()
	}
	if *nativeGemma4FFNResidual {
		defer setDriverProfileRuntimeGate("GO_MLX_ENABLE_NATIVE_GEMMA4_FFN_RESIDUAL", "1")()
	}
	if *nativeGemma4RouterMatVec {
		defer setDriverProfileRuntimeGate("GO_MLX_ENABLE_NATIVE_GEMMA4_ROUTER_MATVEC", "1")()
	}
	if *nativeGemma4RouterTopK {
		defer setDriverProfileRuntimeGate("GO_MLX_ENABLE_NATIVE_GEMMA4_ROUTER_TOPK", "1")()
	}
	if *nativeGemma4AttentionOMatVec {
		defer setDriverProfileRuntimeGate("GO_MLX_ENABLE_NATIVE_GEMMA4_ATTENTION_O_MATVEC", "1")()
	}
	if *nativeGemma4ResidualNorm {
		defer setDriverProfileRuntimeGate("GO_MLX_ENABLE_NATIVE_GEMMA4_RESIDUAL_NORM", "1")()
	}
	if *nativeGemma4Layer {
		defer setDriverProfileRuntimeGate("GO_MLX_ENABLE_NATIVE_GEMMA4_LAYER", "1")()
	}
	if *nativeGemma4MoELayer {
		defer setDriverProfileRuntimeGate("GO_MLX_ENABLE_NATIVE_GEMMA4_MOE_LAYER", "1")()
	}
	if *compiledGemma4Layer {
		defer setDriverProfileRuntimeGate("GO_MLX_ENABLE_COMPILED_GEMMA4_LAYER", "1")()
	}
	if *fixedGemma4Cache {
		defer setDriverProfileRuntimeGate(mlx.Gemma4FastRuntimeGateFixedGemma4Cache, "1")()
	}
	if *fixedGemma4SlidingCacheBound {
		defer setDriverProfileRuntimeGate(mlx.Gemma4FastRuntimeGateFixedGemma4Sliding, "1")()
	}
	if *fixedGemma4SharedMask {
		defer setDriverProfileRuntimeGate(mlx.Gemma4FastRuntimeGateFixedGemma4SharedMask, "1")()
	}
	if *nativeFixedSlidingAttention {
		defer setDriverProfileRuntimeGate(mlx.Gemma4FastRuntimeGateNativeFixedSliding, "1")()
	}
	if *nativeGemma4FixedOwnerAttention {
		defer setDriverProfileRuntimeGate("GO_MLX_ENABLE_NATIVE_GEMMA4_FIXED_OWNER_ATTENTION", "1")()
	}
	if *nativeGemma4FixedOwnerAttentionResidual {
		defer setDriverProfileRuntimeGate("GO_MLX_ENABLE_NATIVE_GEMMA4_FIXED_OWNER_ATTENTION_RESIDUAL", "1")()
	}
	if *fixedWideSDPAAttention || *fixedWideMatmulAttention || *fixedRowCacheUpdate {
		defer setDriverProfileFixedAttentionDiagnostics(*fixedWideSDPAAttention, *fixedWideMatmulAttention, *fixedRowCacheUpdate)()
	}
	if *directGreedyToken {
		defer setDriverProfileRuntimeGate("GO_MLX_ENABLE_DIRECT_GREEDY_TOKEN", "1")()
	}
	if *generationStream {
		defer setDriverProfileRuntimeGate("GO_MLX_ENABLE_GENERATION_STREAM", "1")()
	}
	modelPath := ""
	loadOptions := []mlx.LoadOption{}
	var loadSettings *tuneProfileLoadSettings
	if core.Trim(*profilePath) != "" {
		report, err := readTuneProfileReport(*profilePath)
		if err != nil {
			core.Print(stderr, "%s driver-profile: profile: %v", cliName(), err)
			return 1
		}
		if report.Profile == nil {
			core.Print(stderr, "%s driver-profile: profile payload missing", cliName())
			return 1
		}
		modelPath = report.ModelPath
		loadOptions = append(loadOptions, mlx.TuningCandidateLoadOptions(report.Profile.Candidate)...)
		load := report.Load
		loadSettings = &load
	}
	if fs.NArg() == 1 {
		modelPath = fs.Arg(0)
	}
	if core.Trim(modelPath) == "" {
		core.WriteString(stderr, core.Sprintf("%s driver-profile: model path missing from profile\n", cliName()))
		fs.Usage()
		return 2
	}
	if *contextLen > 0 {
		loadOptions = append(loadOptions, mlx.WithContextLength(*contextLen))
		if loadSettings == nil {
			loadSettings = &tuneProfileLoadSettings{}
		}
		loadSettings.ContextLength = *contextLen
	}
	if *prefillChunkSize < 0 {
		core.WriteString(stderr, core.Sprintf("%s driver-profile: prefill chunk size must be >= 0\n", cliName()))
		return 2
	}
	if *pagedKVPageSize < 0 {
		core.WriteString(stderr, core.Sprintf("%s driver-profile: paged KV page size must be >= 0\n", cliName()))
		return 2
	}
	if *prefillChunkSize > 0 {
		loadOptions = append(loadOptions, mlx.WithPrefillChunkSize(*prefillChunkSize))
		if loadSettings == nil {
			loadSettings = &tuneProfileLoadSettings{}
		}
		loadSettings.PrefillChunkSize = *prefillChunkSize
	}
	if *estimatePowerWatts < 0 {
		core.WriteString(stderr, core.Sprintf("%s driver-profile: estimated power watts must be >= 0\n", cliName()))
		return 2
	}
	if *promptChunkBytes < 0 {
		core.WriteString(stderr, core.Sprintf("%s driver-profile: prompt chunk bytes must be >= 0\n", cliName()))
		return 2
	}
	if *speculativeDraftTokens < 0 {
		core.WriteString(stderr, core.Sprintf("%s driver-profile: speculative draft tokens must be >= 0\n", cliName()))
		return 2
	}
	if *fixedGemma4CacheSize < 0 {
		core.WriteString(stderr, core.Sprintf("%s driver-profile: fixed Gemma 4 cache size must be >= 0\n", cliName()))
		return 2
	}
	if *generationClearCacheInterval < 0 {
		core.WriteString(stderr, core.Sprintf("%s driver-profile: generation clear-cache interval must be >= 0\n", cliName()))
		return 2
	}
	if *temperature < 0 {
		core.WriteString(stderr, core.Sprintf("%s driver-profile: temperature must be >= 0\n", cliName()))
		return 2
	}
	if *topP < 0 {
		core.WriteString(stderr, core.Sprintf("%s driver-profile: top-p must be >= 0\n", cliName()))
		return 2
	}
	if *topK < 0 {
		core.WriteString(stderr, core.Sprintf("%s driver-profile: top-k must be >= 0\n", cliName()))
		return 2
	}
	if *repeatPenalty < 0 {
		core.WriteString(stderr, core.Sprintf("%s driver-profile: repeat penalty must be >= 0\n", cliName()))
		return 2
	}
	applyDriverProfileThroughputBenchmarkLimits(*throughputBenchmark, visitedFlags, *maxTokens, repeatedTokenLoopLimit, repeatedLineLoopLimit, repeatedSentenceLoopLimit)
	if *repeatedTokenLoopLimit < 1 {
		core.WriteString(stderr, core.Sprintf("%s driver-profile: repeated token loop limit must be >= 1\n", cliName()))
		return 2
	}
	if *repeatedLineLoopLimit < 1 {
		core.WriteString(stderr, core.Sprintf("%s driver-profile: repeated line loop limit must be >= 1\n", cliName()))
		return 2
	}
	if *repeatedSentenceLoopLimit < 1 {
		core.WriteString(stderr, core.Sprintf("%s driver-profile: repeated sentence loop limit must be >= 1\n", cliName()))
		return 2
	}
	if mode, ok := parseRuntimeCacheMode(*cacheMode); ok {
		if !isRuntimeCacheMode(mode) {
			core.WriteString(stderr, core.Sprintf("%s driver-profile: unsupported cache mode %q\n", cliName(), string(mode)))
			return 2
		}
		loadOptions = append(loadOptions, mlx.WithKVCacheMode(mode))
		if loadSettings == nil {
			loadSettings = &tuneProfileLoadSettings{}
		}
		loadSettings.CacheMode = string(mode)
	}
	if *pagedKVPageSize > 0 {
		loadOptions = append(loadOptions, mlx.WithPagedKVPageSize(*pagedKVPageSize))
		if loadSettings == nil {
			loadSettings = &tuneProfileLoadSettings{}
		}
		loadSettings.PagedKVPageSize = *pagedKVPageSize
	}
	if *pagedKVPrealloc {
		loadOptions = append(loadOptions, mlx.WithPagedKVPrealloc(true))
		if loadSettings == nil {
			loadSettings = &tuneProfileLoadSettings{}
		}
		loadSettings.PagedKVPrealloc = true
	}
	if *fixedGemma4CacheSize > 0 {
		loadOptions = append(loadOptions, mlx.WithFixedGemma4CacheSize(*fixedGemma4CacheSize))
		if loadSettings == nil {
			loadSettings = &tuneProfileLoadSettings{}
		}
		loadSettings.FixedGemma4CacheSize = *fixedGemma4CacheSize
	}
	if fastLaneEnabled && *contextLen > mlx.ProductionLaneContextLength {
		loadOptions = append(loadOptions, mlx.WithKVCacheStorageDType(mlx.ProductionLaneRetainedKVCacheDType))
		if loadSettings == nil {
			loadSettings = &tuneProfileLoadSettings{}
		}
		loadSettings.KVCacheStorageDType = mlx.ProductionLaneRetainedKVCacheDType
	}
	if *device != "" {
		loadOptions = append(loadOptions, mlx.WithDevice(*device))
	}
	report, err := runDriverProfileGuarded(ctx, modelPath, loadOptions, driverProfileOptions{
		Prompt:                       *prompt,
		PromptSuffix:                 *promptSuffix,
		PromptChunkBytes:             *promptChunkBytes,
		PromptRepeat:                 *promptRepeat,
		MaxTokens:                    *maxTokens,
		Runs:                         *runs,
		IncludeOutput:                *includeOutput,
		Chat:                         *chat,
		TraceTokenPhases:             *traceTokenPhases,
		ThroughputBenchmark:          *throughputBenchmark,
		Temperature:                  *temperature,
		TopP:                         *topP,
		TopK:                         *topK,
		RepeatPenalty:                *repeatPenalty,
		SpeculativeDraftModelPath:    core.Trim(*speculativeDraftModel),
		SpeculativeDraftTokens:       *speculativeDraftTokens,
		SpeculativeGenerationMode:    driverProfileSpeculativeGenerationMode(core.Trim(*speculativeDraftModel)),
		GenerationClearCache:         *generationClearCache,
		GenerationClearCacheInterval: *generationClearCacheInterval,
		SafetyLimits: driverProfileSafetyLimits{
			MaxActiveMemoryBytes:          *maxActiveMemoryBytes,
			MaxProcessVirtualMemoryBytes:  *maxProcessVirtualMemoryBytes,
			MaxProcessResidentMemoryBytes: *maxProcessResidentMemoryBytes,
			RepeatedTokenLoopLimit:        *repeatedTokenLoopLimit,
			RepeatedLineLoopLimit:         *repeatedLineLoopLimit,
			RepeatedSentenceLoopLimit:     *repeatedSentenceLoopLimit,
		},
		temperatureExplicit:           visitedFlags["temperature"],
		topPExplicit:                  visitedFlags["top-p"],
		topKExplicit:                  visitedFlags["top-k"],
		repeatPenaltyExplicit:         visitedFlags["repeat-penalty"],
		repeatedTokenLimitExplicit:    visitedFlags["repeated-token-loop-limit"],
		repeatedLineLimitExplicit:     visitedFlags["repeated-line-loop-limit"],
		repeatedSentenceLimitExplicit: visitedFlags["repeated-sentence-loop-limit"],
	})
	if report != nil && loadSettings != nil {
		report.Load = mergeDriverProfileLoadSettings(loadSettings, report.Load)
	}
	if report != nil && *estimatePowerWatts > 0 {
		report.EstimatedEnergy = estimateDriverProfileEnergy(report, *estimatePowerWatts)
	}
	reportPath := core.Trim(*reportFile)
	if *jsonOut || reportPath != "" {
		if report == nil {
			report = &driverProfileReport{
				Version:                   1,
				ModelPath:                 modelPath,
				PromptBytes:               len(*prompt),
				PromptSuffixBytes:         len(*promptSuffix),
				MaxTokens:                 *maxTokens,
				RequestedRuns:             *runs,
				PromptRepeat:              driverProfileReportPromptRepeat(*promptRepeat),
				TraceTokenPhases:          *traceTokenPhases,
				ThroughputBenchmark:       *throughputBenchmark,
				Temperature:               *temperature,
				TopP:                      *topP,
				TopK:                      *topK,
				RepeatPenalty:             *repeatPenalty,
				SpeculativeDraftModelPath: core.Trim(*speculativeDraftModel),
				SpeculativeDraftTokens:    *speculativeDraftTokens,
				SpeculativeGenerationMode: driverProfileSpeculativeGenerationMode(core.Trim(*speculativeDraftModel)),
				SafetyLimits: driverProfileSafetyLimits{
					MaxActiveMemoryBytes:          *maxActiveMemoryBytes,
					MaxProcessVirtualMemoryBytes:  *maxProcessVirtualMemoryBytes,
					MaxProcessResidentMemoryBytes: *maxProcessResidentMemoryBytes,
					RepeatedTokenLoopLimit:        *repeatedTokenLoopLimit,
					RepeatedLineLoopLimit:         *repeatedLineLoopLimit,
					RepeatedSentenceLoopLimit:     *repeatedSentenceLoopLimit,
				},
			}
		}
		if report.SpeculativeGenerationMode == "" {
			report.SpeculativeGenerationMode = driverProfileSpeculativeGenerationMode(report.SpeculativeDraftModelPath)
		}
		if err != nil && report.Error == "" {
			report.Error = err.Error()
		}
		data := core.JSONMarshalIndent(report, "", "  ")
		if !data.OK {
			core.Print(stderr, "%s driver-profile: marshal report failed", cliName())
			return 1
		}
		if reportPath != "" {
			if writeErr := writeJSONReportFile(reportPath, data.Value.([]byte)); writeErr != nil {
				core.Print(stderr, "%s driver-profile: write report file: %v", cliName(), writeErr)
				return 1
			}
		}
		if *jsonOut {
			core.WriteString(stdout, string(data.Value.([]byte)))
			core.WriteString(stdout, "\n")
		}
		if err != nil {
			return 1
		}
		if *jsonOut {
			return 0
		}
	}
	if err != nil {
		core.Print(stderr, "%s driver-profile: %v", cliName(), err)
		return 1
	}
	printDriverProfileSummary(stdout, report)
	return 0
}

func driverProfileVisitedFlags(fs *flag.FlagSet) map[string]bool {
	visited := map[string]bool{}
	if fs == nil {
		return visited
	}
	fs.Visit(func(f *flag.Flag) {
		if f != nil {
			visited[f.Name] = true
		}
	})
	return visited
}

func driverProfileFastGemma4LaneEnabled(enabled bool, visited map[string]bool, profilePath string) bool {
	if visited != nil && visited["fast-gemma4-lane"] {
		return enabled
	}
	if core.Trim(profilePath) != "" {
		return false
	}
	return enabled
}

func applyGemma4FastLaneDefaults(
	visited map[string]bool,
	contextLen *int,
	cacheMode *string,
	defaultContextLength int,
) []func() {
	if visited == nil {
		visited = map[string]bool{}
	}
	if contextLen != nil && !visited["context"] {
		*contextLen = defaultContextLength
	}
	if cacheMode != nil && !visited["cache-mode"] {
		*cacheMode = string(memory.KVCacheModePaged)
	}
	resolvedContext := 0
	if contextLen != nil {
		resolvedContext = *contextLen
	}
	gateCount := mlx.DefaultGemma4FastRuntimeGateCount()
	restoreCap := gateCount
	if resolvedContext > mlx.ProductionLaneContextLength {
		restoreCap++
	}
	restores := make([]func(), 0, restoreCap)
	for i := range gateCount {
		gate, ok := mlx.DefaultGemma4FastRuntimeGate(i)
		if !ok {
			continue
		}
		if driverProfileRuntimeGateValue(gate) != "" {
			continue
		}
		restores = append(restores, setDriverProfileRuntimeGate(gate, "1"))
	}
	return restores
}

func applyDriverProfileThroughputBenchmarkLimits(
	enabled bool,
	visited map[string]bool,
	maxTokens int,
	repeatedTokenLoopLimit *int,
	repeatedLineLoopLimit *int,
	repeatedSentenceLoopLimit *int,
) {
	if !enabled {
		return
	}
	if visited == nil {
		visited = map[string]bool{}
	}
	limit := maxTokens + 1
	if limit < 1024 {
		limit = 1024
	}
	if repeatedTokenLoopLimit != nil && !visited["repeated-token-loop-limit"] {
		*repeatedTokenLoopLimit = limit
	}
	if repeatedLineLoopLimit != nil && !visited["repeated-line-loop-limit"] {
		*repeatedLineLoopLimit = limit
	}
	if repeatedSentenceLoopLimit != nil && !visited["repeated-sentence-loop-limit"] {
		*repeatedSentenceLoopLimit = limit
	}
}

func resolveDriverProfileThroughputBenchmarkLimits(opts *driverProfileOptions) {
	if opts == nil || !opts.ThroughputBenchmark {
		return
	}
	limit := opts.GenerationMaxTokens + 1
	if limit < 1024 {
		limit = 1024
	}
	if !opts.repeatedTokenLimitExplicit {
		opts.SafetyLimits.RepeatedTokenLoopLimit = limit
	}
	if !opts.repeatedLineLimitExplicit {
		opts.SafetyLimits.RepeatedLineLoopLimit = limit
	}
	if !opts.repeatedSentenceLimitExplicit {
		opts.SafetyLimits.RepeatedSentenceLoopLimit = limit
	}
}

var runDriverProfile = defaultRunDriverProfile

func runDriverProfileGuarded(ctx context.Context, modelPath string, loadOptions []mlx.LoadOption, opts driverProfileOptions) (report *driverProfileReport, err error) {
	defer func() {
		if recovered := recover(); recovered != nil {
			err = core.NewError(core.Sprintf("driver-profile panic: %v", recovered))
		}
	}()
	return runDriverProfile(ctx, modelPath, loadOptions, opts)
}

func defaultRunDriverProfile(ctx context.Context, modelPath string, loadOptions []mlx.LoadOption, opts driverProfileOptions) (*driverProfileReport, error) {
	opts = normalizeDriverProfileOptions(opts)
	speculativeDraftModelPath := core.Trim(opts.SpeculativeDraftModelPath)
	report := &driverProfileReport{
		Version:                   1,
		ModelPath:                 modelPath,
		PromptBytes:               len(opts.Prompt),
		PromptSuffixBytes:         len(opts.PromptSuffix),
		PromptChunkBytes:          opts.PromptChunkBytes,
		PromptRepeat:              driverProfileReportPromptRepeat(opts.PromptRepeat),
		MaxTokens:                 opts.MaxTokens,
		RequestedRuns:             opts.Runs,
		Chat:                      opts.Chat,
		TraceTokenPhases:          opts.TraceTokenPhases,
		ThroughputBenchmark:       opts.ThroughputBenchmark,
		Temperature:               opts.Temperature,
		TopP:                      opts.TopP,
		TopK:                      opts.TopK,
		RepeatPenalty:             opts.RepeatPenalty,
		SpeculativeDraftModelPath: speculativeDraftModelPath,
		SpeculativeDraftTokens:    driverProfileSpeculativeDraftTokensForReport(speculativeDraftModelPath, opts.SpeculativeDraftTokens),
		SpeculativeGenerationMode: driverProfileSpeculativeGenerationMode(speculativeDraftModelPath),
		SafetyLimits:              opts.SafetyLimits,
		RuntimeGates:              driverProfileRuntimeGates(),
	}
	loadStart := time.Now()
	if speculativeDraftModelPath != "" {
		opts.SpeculativeDraftModelPath = speculativeDraftModelPath
		return defaultRunDriverProfileSpeculative(ctx, modelPath, loadOptions, opts, report, loadStart)
	}
	model, err := loadBenchModel(modelPath, loadOptions...)
	report.LoadDuration = bench.NonZeroDuration(time.Since(loadStart))
	if err != nil {
		report.Error = err.Error()
		return report, err
	}
	if model == nil {
		err := core.NewError("mlx: driver profile loaded nil model")
		report.Error = err.Error()
		return report, err
	}
	report.Load = mergeDriverProfileLoadSettings(report.Load, loadSettingsFromModelInfo(model.Info()))
	opts.GenerationMaxTokens = profileGenerationBudgetTokens(opts.MaxTokens, report.Load)
	report.MaxTokens = opts.MaxTokens
	opts.SafetyLimits = resolveDriverProfileSafetyLimits(opts.SafetyLimits, report.Load)
	resolveDriverProfileThroughputBenchmarkLimits(&opts)
	report.SafetyLimits = opts.SafetyLimits
	if opts.Chat {
		template := chapterProfileTemplate("", model.Info().Architecture)
		stopTokenIDs, suppressTokenIDs := chapterProfileTemplateTokenControls(template, model.Tokenizer())
		opts.StopTokenIDs = stopTokenIDs
		opts.SuppressTokenIDs = suppressTokenIDs
		report.StopTokenIDs = stopTokenIDs
		report.SuppressTokenIDs = suppressTokenIDs
	}
	defer model.Close()
	if err := driverProfileMetricsSafetyError("load", model.Metrics(), opts.SafetyLimits); err != nil {
		report.Error = err.Error()
		return report, err
	}

	var firstErr error
	for i := 0; i < opts.Runs; i++ {
		run := profileLoadedModelGeneration(ctx, model, i+1, opts)
		if run.Error != "" && firstErr == nil {
			firstErr = core.NewError(run.Error)
		}
		report.Runs = append(report.Runs, run)
		mlx.ClearCache()
	}
	report.Summary = summariseDriverProfileRuns(report.Runs)
	if firstErr != nil {
		report.Error = firstErr.Error()
		return report, firstErr
	}
	return report, nil
}

func driverProfileSpeculativeDraftTokensForReport(draftModelPath string, draftTokens int) int {
	if core.Trim(draftModelPath) == "" {
		return 0
	}
	return draftTokens
}

func driverProfileSpeculativeGenerationMode(draftModelPath string) string {
	if core.Trim(draftModelPath) == "" {
		return ""
	}
	return speculativeGenerationModeTargetDraft
}

func stateRampProfileSpeculativeGenerationMode(draftModelPath string) string {
	if core.Trim(draftModelPath) == "" {
		return ""
	}
	return speculativeGenerationModeTargetOnlyRetainedConfig
}

func defaultRunDriverProfileSpeculative(ctx context.Context, modelPath string, loadOptions []mlx.LoadOption, opts driverProfileOptions, report *driverProfileReport, loadStart time.Time) (*driverProfileReport, error) {
	pair, err := loadSpeculativePair(modelPath, opts.SpeculativeDraftModelPath, mlx.SpeculativePairConfig{
		TargetOptions: loadOptions,
		DraftOptions:  loadOptions,
	})
	report.LoadDuration = bench.NonZeroDuration(time.Since(loadStart))
	if err != nil {
		report.Error = err.Error()
		return report, err
	}
	if pair == nil || pair.Target == nil {
		err := core.NewError("mlx: driver profile loaded nil speculative pair")
		report.Error = err.Error()
		return report, err
	}
	defer pair.Close()
	if pair.Report.AssistantLayout != nil {
		layout := *pair.Report.AssistantLayout
		report.SpeculativeAssistantLayout = &layout
	}
	report.Load = mergeDriverProfileLoadSettings(report.Load, loadSettingsFromModelInfo(pair.Target.Info()))
	opts.GenerationMaxTokens = profileGenerationBudgetTokens(opts.MaxTokens, report.Load)
	report.MaxTokens = opts.MaxTokens
	opts.SafetyLimits = resolveDriverProfileSafetyLimits(opts.SafetyLimits, report.Load)
	resolveDriverProfileThroughputBenchmarkLimits(&opts)
	report.SafetyLimits = opts.SafetyLimits
	if opts.Chat {
		template := chapterProfileTemplate("", pair.Target.Info().Architecture)
		stopTokenIDs, suppressTokenIDs := chapterProfileTemplateTokenControls(template, pair.Target.Tokenizer())
		opts.StopTokenIDs = stopTokenIDs
		opts.SuppressTokenIDs = suppressTokenIDs
		report.StopTokenIDs = stopTokenIDs
		report.SuppressTokenIDs = suppressTokenIDs
	}
	if err := driverProfileMetricsSafetyError("load", pair.Target.Metrics(), opts.SafetyLimits); err != nil {
		report.Error = err.Error()
		return report, err
	}

	var firstErr error
	for i := 0; i < opts.Runs; i++ {
		profileRun, runErr := speculativeprofile.RunPairProfile(ctx, pair, speculativeprofile.ProfileConfig{
			Prompt:        opts.Prompt,
			MaxTokens:     opts.MaxTokens,
			DraftTokens:   opts.SpeculativeDraftTokens,
			IncludeOutput: opts.IncludeOutput,
			Chat:          opts.Chat,
			Architecture:  pair.Target.Info().Architecture,
			GenerateConfig: mlx.GenerateConfig{
				StopTokens:     opts.StopTokenIDs,
				SuppressTokens: opts.SuppressTokenIDs,
			},
		})
		run := driverProfileRunFromSpeculativeProfile(i+1, profileRun, opts)
		if runErr != nil && run.Error == "" {
			run.Error = runErr.Error()
		}
		if run.Error == "" {
			if err := driverProfileRunSafetyError(i+1, run, opts.SafetyLimits); err != nil {
				run.Error = err.Error()
			}
		}
		if run.Error != "" && firstErr == nil {
			firstErr = core.NewError(run.Error)
		}
		report.Runs = append(report.Runs, run)
		mlx.ClearCache()
	}
	report.Summary = summariseDriverProfileRuns(report.Runs)
	if firstErr != nil {
		report.Error = firstErr.Error()
		return report, firstErr
	}
	return report, nil
}

func driverProfileRunFromSpeculativeProfile(index int, profileRun speculativeprofile.ProfileRun, opts driverProfileOptions) driverProfileRun {
	outputTokenIDBytes := make([]byte, 0, len(profileRun.Result.Tokens)*4)
	for _, token := range profileRun.Result.Tokens {
		outputTokenIDBytes = appendDriverProfileTokenIDBytes(outputTokenIDBytes, token.ID)
	}
	run := driverProfileRun{
		Index:                  index,
		Duration:               profileRun.Duration,
		RestoreDuration:        profileRun.Metrics.PromptCacheRestoreDuration,
		FirstTokenDuration:     profileRun.Metrics.FirstTokenDuration,
		StreamDuration:         profileRun.Metrics.DecodeDuration,
		DriverOverheadDuration: driverRunOverhead(profileRun.Duration, profileRun.Metrics),
		VisibleTokens:          profileRun.VisibleTokens,
		SampledTokenIDs:        profileRun.SampledTokenIDs,
		SampledTokenTexts:      profileRun.SampledTokenTexts,
		OutputTokenIDSHA256:    driverProfileTokenIDBytesSHA256(outputTokenIDBytes),
		Metrics:                profileRun.Metrics,
	}
	if opts.IncludeOutput {
		run.Output = profileRun.Output
	}
	return run
}

func driverProfileTokenIDSHA256(ids []int32) string {
	if len(ids) == 0 {
		return ""
	}
	data := make([]byte, 0, len(ids)*4)
	for _, id := range ids {
		data = appendDriverProfileTokenIDBytes(data, id)
	}
	return driverProfileTokenIDBytesSHA256(data)
}

func driverProfileTokenIDBytesSHA256(data []byte) string {
	if len(data) == 0 {
		return ""
	}
	return core.SHA256Hex(data)
}

func appendDriverProfileTokenIDBytes(data []byte, id int32) []byte {
	value := uint32(id)
	return append(data, byte(value), byte(value>>8), byte(value>>16), byte(value>>24))
}

var driverProfileRuntimeGateOverrides struct {
	sync.RWMutex
	values map[string]string
}

func setDriverProfileRuntimeGate(name, value string) func() {
	restoreMetal := metal.SetRuntimeGate(name, value)
	name = core.Trim(name)
	value = core.Trim(value)
	if name == "" {
		return restoreMetal
	}
	driverProfileRuntimeGateOverrides.Lock()
	if driverProfileRuntimeGateOverrides.values == nil {
		driverProfileRuntimeGateOverrides.values = map[string]string{}
	}
	previous, hadPrevious := driverProfileRuntimeGateOverrides.values[name]
	if value == "" {
		delete(driverProfileRuntimeGateOverrides.values, name)
	} else {
		driverProfileRuntimeGateOverrides.values[name] = value
	}
	driverProfileRuntimeGateOverrides.Unlock()

	return func() {
		restoreMetal()
		driverProfileRuntimeGateOverrides.Lock()
		defer driverProfileRuntimeGateOverrides.Unlock()
		if driverProfileRuntimeGateOverrides.values == nil {
			driverProfileRuntimeGateOverrides.values = map[string]string{}
		}
		if hadPrevious {
			driverProfileRuntimeGateOverrides.values[name] = previous
			return
		}
		delete(driverProfileRuntimeGateOverrides.values, name)
	}
}

func setDriverProfileFixedAttentionDiagnostics(wideSDPA, wideMatmul, rowCacheUpdate bool) func() {
	restores := []func(){metal.SetFixedAttentionDiagnostics(wideSDPA, wideMatmul, rowCacheUpdate)}
	if wideSDPA {
		restores = append(restores, setDriverProfileRuntimeGate("GO_MLX_ENABLE_FIXED_WIDE_SDPA_ATTENTION", "1"))
	}
	if wideMatmul {
		restores = append(restores, setDriverProfileRuntimeGate("GO_MLX_ENABLE_FIXED_WIDE_MATMUL_ATTENTION", "1"))
	}
	if rowCacheUpdate {
		restores = append(restores, setDriverProfileRuntimeGate("GO_MLX_ENABLE_FIXED_ROW_CACHE_UPDATE", "1"))
	}
	return func() {
		for i := len(restores) - 1; i >= 0; i-- {
			restores[i]()
		}
	}
}

var driverProfileRuntimeGateNameList = []string{
	"GO_MLX_ENABLE_EXPERT_ID_MATVEC",
	"GO_MLX_ENABLE_EXPERT_ID_FUSED_ACTIVATION",
	"GO_MLX_ENABLE_EXPERT_ID_UNROLLED_Q4",
	"GO_MLX_ENABLE_SORTED_EXPERT_PREFILL",
	mlx.Gemma4FastRuntimeGatePagedDecodeFastConcat,
	mlx.Gemma4FastRuntimeGateNativePagedAttention,
	"GO_MLX_ENABLE_NATIVE_GELU_GATE_MUL",
	"GO_MLX_ENABLE_NATIVE_MLP_MATVEC",
	"GO_MLX_ENABLE_NATIVE_LINEAR_MATVEC",
	"GO_MLX_ENABLE_NATIVE_Q6_BITSTREAM_MATVEC",
	"GO_MLX_ENABLE_NATIVE_MLP_GELU",
	"GO_MLX_ENABLE_NATIVE_GEMMA4_FFN_RESIDUAL",
	"GO_MLX_ENABLE_NATIVE_GEMMA4_ROUTER_MATVEC",
	"GO_MLX_ENABLE_NATIVE_GEMMA4_ROUTER_TOPK",
	"GO_MLX_ENABLE_NATIVE_GEMMA4_FIXED_OWNER_ATTENTION",
	"GO_MLX_ENABLE_NATIVE_GEMMA4_FIXED_OWNER_ATTENTION_RESIDUAL",
	"GO_MLX_ENABLE_NATIVE_GEMMA4_ATTENTION_O_MATVEC",
	"GO_MLX_ENABLE_NATIVE_GEMMA4_RESIDUAL_NORM",
	"GO_MLX_ENABLE_NATIVE_GEMMA4_LAYER",
	"GO_MLX_ENABLE_NATIVE_GEMMA4_MOE_LAYER",
	"GO_MLX_ENABLE_COMPILED_GEMMA4_LAYER",
	"GO_MLX_ENABLE_COMPILED_GEMMA4_PER_LAYER_INPUTS",
	"GO_MLX_ENABLE_FIXED_GEMMA4_CACHE",
	"GO_MLX_ENABLE_FIXED_GEMMA4_SLIDING_CACHE_BOUND",
	"GO_MLX_ENABLE_FIXED_GEMMA4_SHARED_MASK",
	"GO_MLX_ENABLE_NATIVE_FIXED_SLIDING_ATTENTION",
	"GO_MLX_ENABLE_FIXED_WIDE_SDPA_ATTENTION",
	"GO_MLX_ENABLE_FIXED_WIDE_MATMUL_ATTENTION",
	"GO_MLX_ENABLE_FIXED_ROW_CACHE_UPDATE",
	"GO_MLX_ENABLE_DIRECT_GREEDY_TOKEN",
	"GO_MLX_ENABLE_GENERATION_STREAM",
	"GO_MLX_ENABLE_ASYNC_DECODE_PREFETCH",
}

func driverProfileRuntimeGateNames() []string {
	return driverProfileRuntimeGateNameList
}

func driverProfileRuntimeGateValue(name string) string {
	name = core.Trim(name)
	if name == "" {
		return ""
	}
	driverProfileRuntimeGateOverrides.RLock()
	if value, ok := driverProfileRuntimeGateOverrides.values[name]; ok {
		driverProfileRuntimeGateOverrides.RUnlock()
		return core.Trim(value)
	}
	driverProfileRuntimeGateOverrides.RUnlock()
	if driverProfileRuntimeGateIgnoresAmbientEnv(name) {
		return ""
	}
	return core.Trim(core.Env(name))
}

func driverProfileRuntimeGateIgnoresAmbientEnv(name string) bool {
	switch name {
	case mlx.Gemma4FastRuntimeGateFixedGemma4Cache,
		mlx.Gemma4FastRuntimeGateFixedGemma4Sliding,
		mlx.Gemma4FastRuntimeGateFixedGemma4SharedMask,
		mlx.Gemma4FastRuntimeGateNativeFixedSliding,
		"GO_MLX_ENABLE_NATIVE_GEMMA4_FIXED_OWNER_ATTENTION",
		"GO_MLX_ENABLE_NATIVE_GEMMA4_FIXED_OWNER_ATTENTION_RESIDUAL",
		"GO_MLX_ENABLE_FIXED_WIDE_SDPA_ATTENTION",
		"GO_MLX_ENABLE_FIXED_WIDE_MATMUL_ATTENTION",
		"GO_MLX_ENABLE_FIXED_ROW_CACHE_UPDATE":
		return true
	default:
		return false
	}
}

func driverProfileRuntimeGates() map[string]string {
	var gates map[string]string
	for _, name := range driverProfileRuntimeGateNames() {
		if value := driverProfileRuntimeGateValue(name); value != "" && value != "0" {
			if gates == nil {
				gates = make(map[string]string, mlx.DefaultGemma4FastRuntimeGateCount()+1)
			}
			gates[name] = value
		}
	}
	return gates
}

func loadSettingsFromModelInfo(info mlx.ModelInfo) *tuneProfileLoadSettings {
	settings := &tuneProfileLoadSettings{
		ContextLength:        info.ContextLength,
		ParallelSlots:        info.ParallelSlots,
		PromptCache:          info.PromptCache,
		PromptCacheMinTokens: info.PromptCacheMinTokens,
		CachePolicy:          string(info.CachePolicy),
		CacheMode:            string(info.CacheMode),
		KVCacheStorageDType:  info.KVCacheStorageDType,
		PagedKVPageSize:      info.PagedKVPageSize,
		PagedKVPrealloc:      info.PagedKVPrealloc,
		FixedGemma4CacheSize: info.FixedGemma4CacheSize,
		BatchSize:            info.BatchSize,
		PrefillChunkSize:     info.PrefillChunkSize,
		ExpectedQuantization: info.ExpectedQuantization,
		MemoryLimitBytes:     info.MemoryLimitBytes,
		CacheLimitBytes:      info.CacheLimitBytes,
		WiredLimitBytes:      info.WiredLimitBytes,
	}
	if *settings == (tuneProfileLoadSettings{}) {
		return nil
	}
	return settings
}

func mergeDriverProfileLoadSettings(primary, resolved *tuneProfileLoadSettings) *tuneProfileLoadSettings {
	if primary == nil {
		return resolved
	}
	if resolved == nil {
		return primary
	}
	merged := *primary
	if merged.ContextLength == 0 {
		merged.ContextLength = resolved.ContextLength
	}
	if merged.ParallelSlots == 0 {
		merged.ParallelSlots = resolved.ParallelSlots
	}
	if !merged.PromptCache {
		merged.PromptCache = resolved.PromptCache
	}
	if merged.PromptCacheMinTokens == 0 {
		merged.PromptCacheMinTokens = resolved.PromptCacheMinTokens
	}
	if merged.CachePolicy == "" {
		merged.CachePolicy = resolved.CachePolicy
	}
	if merged.CacheMode == "" {
		merged.CacheMode = resolved.CacheMode
	}
	if merged.KVCacheStorageDType == "" {
		merged.KVCacheStorageDType = resolved.KVCacheStorageDType
	}
	if merged.PagedKVPageSize == 0 {
		merged.PagedKVPageSize = resolved.PagedKVPageSize
	}
	if !merged.PagedKVPrealloc {
		merged.PagedKVPrealloc = resolved.PagedKVPrealloc
	}
	if merged.FixedGemma4CacheSize == 0 {
		merged.FixedGemma4CacheSize = resolved.FixedGemma4CacheSize
	}
	if merged.BatchSize == 0 {
		merged.BatchSize = resolved.BatchSize
	}
	if merged.PrefillChunkSize == 0 {
		merged.PrefillChunkSize = resolved.PrefillChunkSize
	}
	if merged.ExpectedQuantization == 0 {
		merged.ExpectedQuantization = resolved.ExpectedQuantization
	}
	if merged.MemoryLimitBytes == 0 {
		merged.MemoryLimitBytes = resolved.MemoryLimitBytes
	}
	if merged.CacheLimitBytes == 0 {
		merged.CacheLimitBytes = resolved.CacheLimitBytes
	}
	if merged.WiredLimitBytes == 0 {
		merged.WiredLimitBytes = resolved.WiredLimitBytes
	}
	return &merged
}

func normalizeDriverProfileOptions(opts driverProfileOptions) driverProfileOptions {
	opts.Prompt = core.Trim(opts.Prompt)
	if opts.Prompt == "" {
		opts.Prompt = defaultRetainedProfilePrompt
	}
	if opts.PromptRepeat <= 0 {
		opts.PromptRepeat = 1
	}
	if opts.Runs <= 0 {
		opts.Runs = 1
	}
	if opts.Temperature == 0 && !opts.temperatureExplicit {
		opts.Temperature = driverProfileDefaultTemperature
	}
	if opts.TopP == 0 && !opts.topPExplicit {
		opts.TopP = driverProfileDefaultTopP
	}
	if opts.TopK == 0 && !opts.topKExplicit {
		opts.TopK = driverProfileDefaultTopK
	}
	if opts.RepeatPenalty == 0 && !opts.repeatPenaltyExplicit {
		opts.RepeatPenalty = driverProfileDefaultRepeatPenalty
	}
	if opts.SafetyLimits.RepeatedTokenLoopLimit <= 0 {
		opts.SafetyLimits.RepeatedTokenLoopLimit = driverProfileDefaultRepeatedTokenLoopLimit
	}
	if opts.SafetyLimits.RepeatedLineLoopLimit <= 0 {
		opts.SafetyLimits.RepeatedLineLoopLimit = profileDefaultRepeatedLineLoopLimit
	}
	if opts.SafetyLimits.RepeatedSentenceLoopLimit <= 0 {
		opts.SafetyLimits.RepeatedSentenceLoopLimit = profileDefaultRepeatedSentenceLoopLimit
	}
	return opts
}

func resolveDriverProfileSafetyLimits(limits driverProfileSafetyLimits, load *tuneProfileLoadSettings) driverProfileSafetyLimits {
	if limits.RepeatedTokenLoopLimit <= 0 {
		limits.RepeatedTokenLoopLimit = driverProfileDefaultRepeatedTokenLoopLimit
	}
	if limits.RepeatedLineLoopLimit <= 0 {
		limits.RepeatedLineLoopLimit = profileDefaultRepeatedLineLoopLimit
	}
	if limits.RepeatedSentenceLoopLimit <= 0 {
		limits.RepeatedSentenceLoopLimit = profileDefaultRepeatedSentenceLoopLimit
	}
	memoryLimit := profileResolvedMemoryLimit(load)
	if memoryLimit == 0 {
		return limits
	}
	if limits.MaxActiveMemoryBytes == 0 {
		limits.MaxActiveMemoryBytes = profileDefaultActiveMemoryLimit(memoryLimit)
	}
	if limits.MaxProcessResidentMemoryBytes == 0 {
		limits.MaxProcessResidentMemoryBytes = memoryLimit
	}
	return limits
}

func repeatDriverProfilePrompt(prompt string, repeat int) string {
	if repeat <= 1 || prompt == "" {
		return prompt
	}
	builder := core.NewBuilder()
	for i := range repeat {
		if i > 0 {
			builder.WriteString("\n\n")
		}
		builder.WriteString(prompt)
	}
	return builder.String()
}

func appendDriverProfilePromptSuffix(prompt, suffix string) string {
	suffix = core.Trim(suffix)
	if suffix == "" {
		return prompt
	}
	prompt = core.Trim(prompt)
	if prompt == "" {
		return suffix
	}
	builder := core.NewBuilder()
	builder.WriteString(prompt)
	builder.WriteString("\n\n")
	builder.WriteString(suffix)
	return builder.String()
}

func driverProfileReportPromptRepeat(repeat int) int {
	if repeat <= 1 {
		return 0
	}
	return repeat
}

func promptByteChunks(prompt string, chunkBytes int) iter.Seq[string] {
	return func(yield func(string) bool) {
		if prompt == "" {
			return
		}
		if chunkBytes <= 0 || len(prompt) <= chunkBytes {
			yield(prompt)
			return
		}
		start := 0
		for index := range prompt {
			if index == start || index-start < chunkBytes {
				continue
			}
			if !yield(prompt[start:index]) {
				return
			}
			start = index
		}
		if start < len(prompt) {
			yield(prompt[start:])
		}
	}
}

func driverProfileTokenSeq(ctx context.Context, model driverProfileModel, opts driverProfileOptions, generateOptions []mlx.GenerateOption) iter.Seq[mlx.Token] {
	if opts.PromptChunkBytes > 0 && opts.Chat {
		return model.ChatChunkTokens(ctx, []inference.Message{{Role: "user", Content: opts.Prompt}}, opts.PromptChunkBytes, generateOptions...)
	}
	if opts.PromptChunkBytes > 0 {
		return model.GenerateChunkTokens(ctx, promptByteChunks(opts.Prompt, opts.PromptChunkBytes), generateOptions...)
	}
	if opts.Chat {
		return model.ChatTokens(ctx, []inference.Message{{Role: "user", Content: opts.Prompt}}, generateOptions...)
	}
	return model.GenerateTokens(ctx, opts.Prompt, generateOptions...)
}

func profileLoadedModelGeneration(ctx context.Context, model driverProfileModel, index int, opts driverProfileOptions) (run driverProfileRun) {
	memoryBefore := stateWakeMemoryNow()
	defer func() {
		run.MemoryDelta = stateWakeMemoryDeltaBetween(memoryBefore, stateWakeMemoryNow())
	}()
	start := time.Now()
	builder := core.NewBuilder()
	firstToken := time.Duration(0)
	visibleTokens := 0
	generateOptions := driverProfileGenerateOptions(opts)
	generationCtx := ctx
	if generationCtx == nil {
		generationCtx = context.Background()
	}
	generationCtx, cancelGeneration := context.WithCancel(generationCtx)
	defer cancelGeneration()
	var probeErr error
	sampledTokenIDs := make([]int32, 0, 32)
	sampledTokenTexts := make([]string, 0, 32)
	repeatedTokenID := int32(0)
	repeatedTokenCount := 0
	var lineErr error
	currentLine := ""
	lastLine := ""
	repeatedLineCount := 0
	draining := false
	outputTokenIDByteCapacity := 0
	if opts.MaxTokens > 0 {
		outputTokenIDByteCapacity = opts.MaxTokens * 4
	}
	outputTokenIDBytes := make([]byte, 0, outputTokenIDByteCapacity)
	for token := range driverProfileTokenSeq(generationCtx, model, opts, generateOptions) {
		if draining {
			continue
		}
		if firstToken == 0 {
			firstToken = bench.NonZeroDuration(time.Since(start))
		}
		visibleTokens++
		outputTokenIDBytes = appendDriverProfileTokenIDBytes(outputTokenIDBytes, token.ID)
		if len(sampledTokenIDs) < 32 {
			sampledTokenIDs = append(sampledTokenIDs, token.ID)
			if opts.IncludeOutput {
				sampledTokenTexts = append(sampledTokenTexts, token.Text)
			}
		}
		if probeErr == nil {
			if err := driverProfileMetricsSafetyError(core.Sprintf("run %d stream", index), profileLiveMetrics(), opts.SafetyLimits); err != nil {
				probeErr = err
				cancelGeneration()
				draining = true
				continue
			}
			if opts.SafetyLimits.RepeatedTokenLoopLimit <= 0 {
				repeatedTokenCount = 0
			} else {
				if repeatedTokenCount == 0 || token.ID != repeatedTokenID {
					repeatedTokenID = token.ID
					repeatedTokenCount = 1
				} else {
					repeatedTokenCount++
				}
				if repeatedTokenCount >= opts.SafetyLimits.RepeatedTokenLoopLimit {
					probeErr = core.NewError(core.Sprintf("driver-profile: run %d sampled token %d for %d consecutive tokens", index, token.ID, repeatedTokenCount))
					cancelGeneration()
					draining = true
					continue
				}
			}
		}
		if opts.IncludeOutput {
			builder.WriteString(token.Text)
		}
		if lineErr == nil {
			if line, count, ok := profileObserveRepeatedLineFragment(token.Text, &currentLine, &lastLine, &repeatedLineCount, opts.SafetyLimits.RepeatedLineLoopLimit); ok {
				lineErr = core.NewError(core.Sprintf("driver-profile: run %d repeated visible line %q for %d consecutive lines", index, line, count))
				cancelGeneration()
				draining = true
				continue
			}
		}
	}
	if lineErr == nil {
		if line, count, ok := profileFlushRepeatedLine(&currentLine, &lastLine, &repeatedLineCount, opts.SafetyLimits.RepeatedLineLoopLimit); ok {
			lineErr = core.NewError(core.Sprintf("driver-profile: run %d repeated visible line %q for %d consecutive lines", index, line, count))
		}
	}
	duration := bench.NonZeroDuration(time.Since(start))
	streamDuration := duration
	if firstToken > 0 && duration > firstToken {
		streamDuration = duration - firstToken
	}
	metrics := model.Metrics()
	run = driverProfileRun{
		Index:               index,
		Duration:            duration,
		RestoreDuration:     metrics.PromptCacheRestoreDuration,
		FirstTokenDuration:  firstToken,
		StreamDuration:      streamDuration,
		VisibleTokens:       visibleTokens,
		SampledTokenIDs:     sampledTokenIDs,
		SampledTokenTexts:   sampledTokenTexts,
		OutputTokenIDSHA256: driverProfileTokenIDBytesSHA256(outputTokenIDBytes),
		Metrics:             metrics,
	}
	run.DriverOverheadDuration = driverRunOverhead(run.Duration, run.Metrics)
	if opts.IncludeOutput {
		run.Output = builder.String()
	}
	if probeErr != nil {
		run.Error = probeErr.Error()
		return run
	}
	if lineErr != nil {
		run.Error = lineErr.Error()
		return run
	}
	if err := model.Err(); err != nil {
		run.Error = err.Error()
		return run
	}
	if err := driverProfileRunSafetyError(index, run, opts.SafetyLimits); err != nil {
		run.Error = err.Error()
		return run
	}
	if ctx != nil {
		if err := ctx.Err(); err != nil {
			run.Error = err.Error()
		}
	}
	return run
}

func driverProfileGenerateOptions(opts driverProfileOptions) []mlx.GenerateOption {
	generateOptions := []mlx.GenerateOption{
		mlx.WithTemperature(float32(opts.Temperature)),
		mlx.WithTopP(float32(opts.TopP)),
		mlx.WithTopK(opts.TopK),
		mlx.WithRepeatPenalty(float32(opts.RepeatPenalty)),
	}
	if opts.GenerationMaxTokens > 0 {
		generateOptions = append(generateOptions, mlx.WithMaxTokens(opts.GenerationMaxTokens))
	}
	if opts.TraceTokenPhases {
		if opts.IncludeOutput {
			generateOptions = append(generateOptions, mlx.WithTokenPhaseTraceText())
		} else {
			generateOptions = append(generateOptions, mlx.WithTokenPhaseTrace())
		}
	}
	if opts.GenerationClearCacheInterval > 0 {
		generateOptions = append(generateOptions, mlx.WithGenerationClearCacheInterval(opts.GenerationClearCacheInterval))
	}
	if opts.GenerationClearCache {
		generateOptions = append(generateOptions, mlx.WithGenerationClearCache())
	}
	if len(opts.StopTokenIDs) > 0 {
		generateOptions = append(generateOptions, mlx.WithStopTokens(opts.StopTokenIDs...))
	}
	if len(opts.SuppressTokenIDs) > 0 {
		generateOptions = append(generateOptions, mlx.WithSuppressTokens(opts.SuppressTokenIDs...))
	}
	return generateOptions
}

func driverProfileRunSafetyError(index int, run driverProfileRun, limits driverProfileSafetyLimits) error {
	if err := driverProfileMetricsSafetyError(core.Sprintf("run %d", index), run.Metrics, limits); err != nil {
		return err
	}
	if id, count, ok := driverProfileRepeatedTokenLoop(run.SampledTokenIDs, limits.RepeatedTokenLoopLimit); ok {
		return core.NewError(core.Sprintf("driver-profile: run %d sampled token %d for %d consecutive tokens", index, id, count))
	}
	if line, count, ok := profileRepeatedLineLoop(run.Output, limits.RepeatedLineLoopLimit); ok {
		return core.NewError(core.Sprintf("driver-profile: run %d repeated visible line %q for %d consecutive lines", index, line, count))
	}
	if sentence, count, ok := profileRepeatedSentenceLoop(run.Output, limits.RepeatedSentenceLoopLimit); ok {
		return core.NewError(core.Sprintf("driver-profile: run %d repeated visible sentence %q for %d total occurrences", index, sentence, count))
	}
	if fragments, total, ok := profileFragmentedSentenceOutput(run.Output); ok {
		return core.NewError(core.Sprintf("driver-profile: run %d produced fragmented visible output: %d of %d sentence fragments are too short", index, fragments, total))
	}
	return nil
}

func driverProfileMetricsSafetyError(phase string, metrics mlx.Metrics, limits driverProfileSafetyLimits) error {
	if limits.MaxActiveMemoryBytes > 0 && metrics.ActiveMemoryBytes > limits.MaxActiveMemoryBytes {
		return core.NewError(core.Sprintf("driver-profile: %s exceeded active memory safety limit: %d > %d bytes", phase, metrics.ActiveMemoryBytes, limits.MaxActiveMemoryBytes))
	}
	if limits.MaxProcessVirtualMemoryBytes > 0 && metrics.ProcessVirtualMemoryBytes > limits.MaxProcessVirtualMemoryBytes {
		return core.NewError(core.Sprintf("driver-profile: %s exceeded process virtual memory safety limit: %d > %d bytes", phase, metrics.ProcessVirtualMemoryBytes, limits.MaxProcessVirtualMemoryBytes))
	}
	if limits.MaxProcessResidentMemoryBytes > 0 && metrics.ProcessResidentMemoryBytes > limits.MaxProcessResidentMemoryBytes {
		return core.NewError(core.Sprintf("driver-profile: %s exceeded process resident memory safety limit: %d > %d bytes", phase, metrics.ProcessResidentMemoryBytes, limits.MaxProcessResidentMemoryBytes))
	}
	return nil
}

func driverProfileRepeatedTokenLoop(sampledTokenIDs []int32, limit int) (int32, int, bool) {
	if limit <= 0 || len(sampledTokenIDs) == 0 {
		return 0, 0, false
	}
	last := sampledTokenIDs[0]
	count := 1
	if count >= limit {
		return last, count, true
	}
	for _, id := range sampledTokenIDs[1:] {
		if id != last {
			last = id
			count = 1
		} else {
			count++
		}
		if count >= limit {
			return id, count, true
		}
	}
	return 0, 0, false
}

func profileRepeatedLineLoop(text string, limit int) (string, int, bool) {
	currentLine := ""
	lastLine := ""
	repeatedLineCount := 0
	if line, count, ok := profileObserveRepeatedLineFragment(text, &currentLine, &lastLine, &repeatedLineCount, limit); ok {
		return line, count, ok
	}
	return profileFlushRepeatedLine(&currentLine, &lastLine, &repeatedLineCount, limit)
}

func profileObserveRepeatedLineFragment(fragment string, currentLine, lastLine *string, repeatedLineCount *int, limit int) (string, int, bool) {
	if limit <= 0 || fragment == "" || currentLine == nil || lastLine == nil || repeatedLineCount == nil {
		return "", 0, false
	}
	parts := core.Split(fragment, "\n")
	for i, part := range parts {
		*currentLine += part
		if i == len(parts)-1 {
			continue
		}
		line := core.Trim(*currentLine)
		*currentLine = ""
		if line == "" {
			continue
		}
		if line, count, ok := profileObserveRepeatedLine(line, lastLine, repeatedLineCount, limit); ok {
			return line, count, ok
		}
	}
	return "", 0, false
}

func profileFlushRepeatedLine(currentLine, lastLine *string, repeatedLineCount *int, limit int) (string, int, bool) {
	if limit <= 0 || currentLine == nil || lastLine == nil || repeatedLineCount == nil {
		return "", 0, false
	}
	line := core.Trim(*currentLine)
	*currentLine = ""
	if line == "" {
		return "", 0, false
	}
	return profileObserveRepeatedLine(line, lastLine, repeatedLineCount, limit)
}

func profileObserveRepeatedLine(line string, lastLine *string, repeatedLineCount *int, limit int) (string, int, bool) {
	if limit <= 0 || line == "" || lastLine == nil || repeatedLineCount == nil {
		return "", 0, false
	}
	if line == *lastLine {
		*repeatedLineCount++
	} else {
		*lastLine = line
		*repeatedLineCount = 1
	}
	if *repeatedLineCount >= limit {
		return line, *repeatedLineCount, true
	}
	return "", 0, false
}

type profileRepeatedWordObserver struct {
	current []byte
	last    string
	count   int
}

func profileRepeatedWordLoop(text string, limit int) (string, int, bool) {
	observer := profileRepeatedWordObserver{}
	if word, count, ok := profileObserveRepeatedWordFragment(text, &observer, limit); ok {
		return word, count, ok
	}
	return profileFlushRepeatedWord(&observer, limit)
}

func profileObserveRepeatedWordFragment(fragment string, observer *profileRepeatedWordObserver, limit int) (string, int, bool) {
	if limit <= 0 || fragment == "" || observer == nil {
		return "", 0, false
	}
	for i := 0; i < len(fragment); i++ {
		b := fragment[i]
		switch {
		case b >= 'A' && b <= 'Z':
			observer.current = append(observer.current, b+('a'-'A'))
		case (b >= 'a' && b <= 'z') || (b >= '0' && b <= '9'):
			observer.current = append(observer.current, b)
		default:
			if word, count, ok := profileFlushRepeatedWord(observer, limit); ok {
				return word, count, ok
			}
		}
	}
	return "", 0, false
}

func profileFlushRepeatedWord(observer *profileRepeatedWordObserver, limit int) (string, int, bool) {
	if limit <= 0 || observer == nil || len(observer.current) == 0 {
		return "", 0, false
	}
	word := string(observer.current)
	observer.current = observer.current[:0]
	if len(word) < 2 {
		observer.last = ""
		observer.count = 0
		return "", 0, false
	}
	if word == observer.last {
		observer.count++
	} else {
		observer.last = word
		observer.count = 1
	}
	if observer.count >= limit {
		return word, observer.count, true
	}
	return "", 0, false
}

func profileRepeatedSentenceLoop(text string, limit int) (string, int, bool) {
	if limit <= 0 || text == "" {
		return "", 0, false
	}
	normalised := core.Replace(text, "!", ".")
	normalised = core.Replace(normalised, "?", ".")
	counts := map[string]int{}
	for _, raw := range core.Split(normalised, ".") {
		sentence := profileNormaliseSentence(raw)
		if len(sentence) < 12 {
			continue
		}
		counts[sentence]++
		if counts[sentence] >= limit {
			return sentence, counts[sentence], true
		}
	}
	return "", 0, false
}

func profileNormaliseSentence(raw string) string {
	text := core.Lower(core.Trim(raw))
	text = core.Replace(text, "\n", " ")
	text = core.Replace(text, "\r", " ")
	text = core.Replace(text, "\t", " ")
	for core.Contains(text, "  ") {
		text = core.Replace(text, "  ", " ")
	}
	return core.Trim(text)
}

func profileFragmentedSentenceOutput(text string) (int, int, bool) {
	if text == "" {
		return 0, 0, false
	}
	normalised := core.Replace(text, "!", ".")
	normalised = core.Replace(normalised, "?", ".")
	fragments := 0
	total := 0
	for _, raw := range core.Split(normalised, ".") {
		sentence := profileNormaliseSentence(raw)
		if sentence == "" {
			continue
		}
		total++
		if len(sentence) < 12 {
			fragments++
		}
	}
	if total < profileFragmentedSentenceMinCount {
		return fragments, total, false
	}
	return fragments, total, float64(fragments)/float64(total) >= profileFragmentedSentenceRatio
}

func driverRunOverhead(duration time.Duration, metrics mlx.Metrics) time.Duration {
	if duration <= 0 || metrics.TotalDuration <= 0 || duration <= metrics.TotalDuration {
		return 0
	}
	return duration - metrics.TotalDuration
}

func summariseDriverProfileRuns(runs []driverProfileRun) driverProfileSummary {
	summary := driverProfileSummary{}
	restoreSamples := 0
	firstTokenSamples := 0
	promptSamples := 0
	promptTokens := 0
	prefillSamples := 0
	decodeSamples := 0
	mtpAcceptanceSamples := 0
	mtpVisibleRateSamples := 0
	mtpTargetRateSamples := 0
	mtpWarmRateSamples := 0
	outputTokenIDHashSamples := 0
	outputTokenIDHashConsistent := true
	outputTokenIDHash := ""
	var tokenPhaseIndex map[string]int
	var nativeEventIndex map[string]int
	var nativeEventDetailIndex map[string]int
	traceAggregationInitialised := false
	for _, run := range runs {
		accumulateDriverProfileSummaryMemory(&summary, run.Metrics)
		if run.MemoryDelta != nil {
			summary.GoTotalAllocDeltaBytes += run.MemoryDelta.GoTotalAllocDeltaBytes
			summary.GoMallocsDelta += run.MemoryDelta.GoMallocsDelta
		}
		if run.Error != "" {
			summary.FailedRuns++
			continue
		}
		summary.SuccessfulRuns++
		summary.TotalDuration += run.Duration
		summary.VisibleTokens += run.VisibleTokens
		if run.OutputTokenIDSHA256 == "" {
			outputTokenIDHashConsistent = false
		} else {
			outputTokenIDHashSamples++
			if outputTokenIDHash == "" {
				outputTokenIDHash = run.OutputTokenIDSHA256
			} else if outputTokenIDHash != run.OutputTokenIDSHA256 {
				outputTokenIDHashConsistent = false
			}
		}
		generated := run.Metrics.GeneratedTokens
		if generated == 0 {
			generated = run.VisibleTokens
		}
		summary.GeneratedTokens += generated
		if run.Metrics.PromptTokens > 0 {
			promptSamples++
			promptTokens += run.Metrics.PromptTokens
			if summary.PromptTokensMin == 0 || run.Metrics.PromptTokens < summary.PromptTokensMin {
				summary.PromptTokensMin = run.Metrics.PromptTokens
			}
			if run.Metrics.PromptTokens > summary.PromptTokensMax {
				summary.PromptTokensMax = run.Metrics.PromptTokens
			}
		}
		if run.RestoreDuration > 0 {
			restoreSamples++
			summary.RestoreAvgDuration += run.RestoreDuration
			if summary.RestoreMinDuration == 0 || run.RestoreDuration < summary.RestoreMinDuration {
				summary.RestoreMinDuration = run.RestoreDuration
			}
			if run.RestoreDuration > summary.RestoreMaxDuration {
				summary.RestoreMaxDuration = run.RestoreDuration
			}
		}
		if run.FirstTokenDuration > 0 {
			firstTokenSamples++
			summary.FirstTokenAvgDuration += run.FirstTokenDuration
			if summary.FirstTokenMinDuration == 0 || run.FirstTokenDuration < summary.FirstTokenMinDuration {
				summary.FirstTokenMinDuration = run.FirstTokenDuration
			}
			if run.FirstTokenDuration > summary.FirstTokenMaxDuration {
				summary.FirstTokenMaxDuration = run.FirstTokenDuration
			}
		}
		summary.DriverOverheadAvgDuration += run.DriverOverheadDuration
		if run.Metrics.PrefillTokensPerSec > 0 {
			prefillSamples++
			summary.PrefillTokensPerSecAverage += run.Metrics.PrefillTokensPerSec
		}
		if run.Metrics.DecodeTokensPerSec > 0 {
			decodeSamples++
			summary.DecodeTokensPerSecAverage += run.Metrics.DecodeTokensPerSec
		}
		if mtp := run.Metrics.MTP; mtp != nil {
			summary.MTPProposedTokens += mtp.ProposedTokens
			summary.MTPAcceptedTokens += mtp.AcceptedTokens
			summary.MTPRejectedTokens += mtp.RejectedTokens
			summary.MTPTargetVerifyCalls += mtp.TargetVerifyCalls
			summary.MTPTargetCalls += mtp.TargetCalls
			summary.MTPDraftCalls += mtp.DraftCalls
			if mtp.AcceptanceRate > 0 {
				mtpAcceptanceSamples++
				summary.MTPAcceptanceRateAverage += mtp.AcceptanceRate
			}
			if mtp.VisibleTokensPerSec > 0 {
				mtpVisibleRateSamples++
				summary.MTPVisibleTokensPerSecAverage += mtp.VisibleTokensPerSec
			}
			if mtp.TargetTokensPerSec > 0 {
				mtpTargetRateSamples++
				summary.MTPTargetTokensPerSecAverage += mtp.TargetTokensPerSec
			}
			if mtp.WarmDecodeTokensPerSec > 0 {
				mtpWarmRateSamples++
				summary.MTPWarmDecodeTokensPerSecAverage += mtp.WarmDecodeTokensPerSec
			}
		}
		if run.Metrics.PeakMemoryBytes > summary.PeakMemoryBytes {
			summary.PeakMemoryBytes = run.Metrics.PeakMemoryBytes
		}
		if run.Metrics.ActiveMemoryBytes > summary.ActiveMemoryBytes {
			summary.ActiveMemoryBytes = run.Metrics.ActiveMemoryBytes
		}
		if run.Metrics.CacheMemoryBytes > summary.CacheMemoryBytes {
			summary.CacheMemoryBytes = run.Metrics.CacheMemoryBytes
		}
		if activePlusCache := run.Metrics.ActiveMemoryBytes + run.Metrics.CacheMemoryBytes; activePlusCache > summary.ActivePlusCacheMemoryBytes {
			summary.ActivePlusCacheMemoryBytes = activePlusCache
		}
		if run.Metrics.ProcessVirtualMemoryBytes > summary.ProcessVirtualMemoryBytes {
			summary.ProcessVirtualMemoryBytes = run.Metrics.ProcessVirtualMemoryBytes
		}
		if run.Metrics.ProcessResidentMemoryBytes > summary.ProcessResidentMemoryBytes {
			summary.ProcessResidentMemoryBytes = run.Metrics.ProcessResidentMemoryBytes
		}
		if run.Metrics.ProcessPeakResidentBytes > summary.ProcessPeakResidentBytes {
			summary.ProcessPeakResidentBytes = run.Metrics.ProcessPeakResidentBytes
		}
		summary.TurboQuantKVPayload = driverProfileMaxTurboQuantKVPayload(summary.TurboQuantKVPayload, run.Metrics.TurboQuantKVPayload)
		if len(run.Metrics.TokenPhases) > 0 && !traceAggregationInitialised {
			traceAggregationInitialised = true
			summary.TokenPhases = make([]driverProfileNativeEventSummary, 0, 8)
			summary.NativeEvents = make([]driverProfileNativeEventSummary, 0, 4)
			summary.NativeEventDetails = make([]driverProfileNativeEventSummary, 0, 8)
		}
		for _, phase := range run.Metrics.TokenPhases {
			accumulateDriverProfileTokenPhase(&summary, tokenPhaseIndex, "total", phase.TotalDuration)
			accumulateDriverProfileTokenPhase(&summary, tokenPhaseIndex, "forward", phase.ForwardDuration)
			accumulateDriverProfileTokenPhase(&summary, tokenPhaseIndex, "sample_eval", phase.SampleEvalDuration)
			accumulateDriverProfileTokenPhase(&summary, tokenPhaseIndex, "sample", phase.SampleDuration)
			accumulateDriverProfileTokenPhase(&summary, tokenPhaseIndex, "logits", phase.LogitsDuration)
			accumulateDriverProfileTokenPhase(&summary, tokenPhaseIndex, "token_read", phase.TokenReadDuration)
			accumulateDriverProfileTokenPhase(&summary, tokenPhaseIndex, "decode_text", phase.DecodeTextDuration)
			accumulateDriverProfileTokenPhase(&summary, tokenPhaseIndex, "probe_token", phase.ProbeTokenDuration)
			accumulateDriverProfileTokenPhase(&summary, tokenPhaseIndex, "yield", phase.YieldDuration)
			accumulateDriverProfileTokenPhase(&summary, tokenPhaseIndex, "next_input", phase.NextInputDuration)
			accumulateDriverProfileTokenPhase(&summary, tokenPhaseIndex, "materialize", phase.MaterializeDuration)
			accumulateDriverProfileTokenPhase(&summary, tokenPhaseIndex, "prefetch", phase.PrefetchDuration)
			accumulateDriverProfileTokenPhase(&summary, tokenPhaseIndex, "prefetch_logits", phase.PrefetchLogitsDuration)
			accumulateDriverProfileTokenPhase(&summary, tokenPhaseIndex, "prefetch_cache", phase.PrefetchCacheDuration)
			accumulateDriverProfileTokenPhase(&summary, tokenPhaseIndex, "detach", phase.DetachDuration)
			accumulateDriverProfileTokenPhase(&summary, tokenPhaseIndex, "cache_probe", phase.CacheProbeDuration)
			accumulateDriverProfileTokenPhase(&summary, tokenPhaseIndex, "other", phase.OtherDuration)
			for _, event := range phase.NativeEvents {
				if event.Name == "" || event.Duration <= 0 {
					continue
				}
				name := driverProfileNativeEventBucket(event.Name)
				accumulateDriverProfileNativeEvent(&summary.NativeEvents, nativeEventIndex, name, event)
				accumulateDriverProfileNativeEvent(&summary.NativeEventDetails, nativeEventDetailIndex, event.Name, event)
			}
		}
	}
	if firstTokenSamples > 0 {
		summary.FirstTokenAvgDuration /= time.Duration(firstTokenSamples)
	}
	if restoreSamples > 0 {
		summary.RestoreAvgDuration /= time.Duration(restoreSamples)
	}
	if promptSamples > 0 {
		summary.PromptTokensAverage = float64(promptTokens) / float64(promptSamples)
	}
	if summary.SuccessfulRuns > 0 {
		summary.DriverOverheadAvgDuration /= time.Duration(summary.SuccessfulRuns)
		if outputTokenIDHashSamples > 0 {
			summary.OutputTokenIDSHA256 = outputTokenIDHash
			summary.OutputTokenIDSHA256Consistent = outputTokenIDHashConsistent && outputTokenIDHashSamples == summary.SuccessfulRuns
		}
	}
	if prefillSamples > 0 {
		summary.PrefillTokensPerSecAverage /= float64(prefillSamples)
	}
	if decodeSamples > 0 {
		summary.DecodeTokensPerSecAverage /= float64(decodeSamples)
	}
	if summary.GeneratedTokens > 0 {
		summary.GoBytesPerGeneratedToken = float64(summary.GoTotalAllocDeltaBytes) / float64(summary.GeneratedTokens)
		summary.GoAllocsPerGeneratedToken = float64(summary.GoMallocsDelta) / float64(summary.GeneratedTokens)
	}
	if mtpAcceptanceSamples > 0 {
		summary.MTPAcceptanceRateAverage /= float64(mtpAcceptanceSamples)
	}
	if mtpVisibleRateSamples > 0 {
		summary.MTPVisibleTokensPerSecAverage /= float64(mtpVisibleRateSamples)
	}
	if mtpTargetRateSamples > 0 {
		summary.MTPTargetTokensPerSecAverage /= float64(mtpTargetRateSamples)
	}
	if mtpWarmRateSamples > 0 {
		summary.MTPWarmDecodeTokensPerSecAverage /= float64(mtpWarmRateSamples)
	}
	summary.DecodeBandwidthProxy = estimateDecodeBandwidthProxy(
		summary.DecodeTokensPerSecAverage,
		summary.ActivePlusCacheMemoryBytes,
	)
	for i := range summary.NativeEvents {
		if summary.NativeEvents[i].Count > 0 {
			summary.NativeEvents[i].AverageDuration = summary.NativeEvents[i].Duration / time.Duration(summary.NativeEvents[i].Count)
		}
	}
	for i := range summary.NativeEventDetails {
		if summary.NativeEventDetails[i].Count > 0 {
			summary.NativeEventDetails[i].AverageDuration = summary.NativeEventDetails[i].Duration / time.Duration(summary.NativeEventDetails[i].Count)
		}
	}
	for i := range summary.TokenPhases {
		if summary.TokenPhases[i].Count > 0 {
			summary.TokenPhases[i].AverageDuration = summary.TokenPhases[i].Duration / time.Duration(summary.TokenPhases[i].Count)
		}
	}
	sort.SliceStable(summary.TokenPhases, func(i, j int) bool {
		return summary.TokenPhases[i].Duration > summary.TokenPhases[j].Duration
	})
	sort.SliceStable(summary.NativeEvents, func(i, j int) bool {
		return summary.NativeEvents[i].Duration > summary.NativeEvents[j].Duration
	})
	sort.SliceStable(summary.NativeEventDetails, func(i, j int) bool {
		return summary.NativeEventDetails[i].Duration > summary.NativeEventDetails[j].Duration
	})
	return summary
}

func driverProfileMaxTurboQuantKVPayload(current, candidate *mlx.TurboQuantKVPayloadEstimate) *mlx.TurboQuantKVPayloadEstimate {
	if candidate == nil {
		return current
	}
	if current != nil && current.PaddedPayloadBytes >= candidate.PaddedPayloadBytes {
		return current
	}
	clone := *candidate
	return &clone
}

func accumulateDriverProfileTokenPhase(summary *driverProfileSummary, index map[string]int, name string, duration time.Duration) {
	if summary == nil || duration <= 0 || name == "" {
		return
	}
	idx := -1
	if index != nil {
		if got, ok := index[name]; ok {
			idx = got
		}
	} else {
		for i := range summary.TokenPhases {
			if summary.TokenPhases[i].Name == name {
				idx = i
				break
			}
		}
	}
	if idx < 0 {
		summary.TokenPhases = append(summary.TokenPhases, driverProfileNativeEventSummary{Name: name})
		idx = len(summary.TokenPhases) - 1
		if index != nil {
			index[name] = idx
		}
	}
	summary.TokenPhases[idx].Count++
	summary.TokenPhases[idx].Duration += duration
}

func accumulateDriverProfileNativeEvent(events *[]driverProfileNativeEventSummary, index map[string]int, name string, event mlx.NativePhaseTrace) {
	if events == nil || event.Duration <= 0 || name == "" {
		return
	}
	idx := -1
	if index != nil {
		if got, ok := index[name]; ok {
			idx = got
		}
	} else {
		for i := range *events {
			if (*events)[i].Name == name {
				idx = i
				break
			}
		}
	}
	if idx < 0 {
		*events = append(*events, driverProfileNativeEventSummary{Name: name})
		idx = len(*events) - 1
		if index != nil {
			index[name] = idx
		}
	}
	(*events)[idx].Count++
	(*events)[idx].Duration += event.Duration
	if event.Pages > (*events)[idx].MaxPages {
		(*events)[idx].MaxPages = event.Pages
	}
	if event.Tokens > (*events)[idx].MaxTokens {
		(*events)[idx].MaxTokens = event.Tokens
	}
}

func accumulateDriverProfileSummaryMemory(summary *driverProfileSummary, metrics mlx.Metrics) {
	if summary == nil {
		return
	}
	if metrics.PeakMemoryBytes > summary.PeakMemoryBytes {
		summary.PeakMemoryBytes = metrics.PeakMemoryBytes
	}
	if metrics.ActiveMemoryBytes > summary.ActiveMemoryBytes {
		summary.ActiveMemoryBytes = metrics.ActiveMemoryBytes
	}
	if metrics.CacheMemoryBytes > summary.CacheMemoryBytes {
		summary.CacheMemoryBytes = metrics.CacheMemoryBytes
	}
	if activePlusCache := metrics.ActiveMemoryBytes + metrics.CacheMemoryBytes; activePlusCache > summary.ActivePlusCacheMemoryBytes {
		summary.ActivePlusCacheMemoryBytes = activePlusCache
	}
	if metrics.ProcessVirtualMemoryBytes > summary.ProcessVirtualMemoryBytes {
		summary.ProcessVirtualMemoryBytes = metrics.ProcessVirtualMemoryBytes
	}
	if metrics.ProcessResidentMemoryBytes > summary.ProcessResidentMemoryBytes {
		summary.ProcessResidentMemoryBytes = metrics.ProcessResidentMemoryBytes
	}
	if metrics.ProcessPeakResidentBytes > summary.ProcessPeakResidentBytes {
		summary.ProcessPeakResidentBytes = metrics.ProcessPeakResidentBytes
	}
}

func estimateDecodeBandwidthProxy(decodeTokensPerSec float64, activePlusCacheBytes uint64) *decodeBandwidthProxy {
	if decodeTokensPerSec <= 0 || activePlusCacheBytes == 0 {
		return nil
	}
	const decimalGB = 1000 * 1000 * 1000
	gbPerToken := float64(activePlusCacheBytes) / decimalGB
	return &decodeBandwidthProxy{
		Method:                                       "decode_tokens_per_sec_times_active_plus_cache_residency",
		DecodeTokensPerSec:                           decodeTokensPerSec,
		ActivePlusCacheBytesPerDecodeTokenProxy:      activePlusCacheBytes,
		ActivePlusCacheGBPerDecodeTokenProxy:         gbPerToken,
		ImpliedActivePlusCacheBandwidthGBPerSecProxy: decodeTokensPerSec * gbPerToken,
		Note: "proxy only: active+cache residency is not a hardware bandwidth sampler or exact weight-read counter",
	}
}

func driverProfileNativeEventBucket(name string) string {
	const prefix = "gemma4.layer."
	if !core.HasPrefix(name, prefix) {
		return name
	}
	tail := name[len(prefix):]
	dot := core.Index(tail, ".")
	if dot < 0 {
		return name
	}
	return tail[dot+1:]
}

func estimateDriverProfileEnergy(report *driverProfileReport, powerWatts float64) *driverProfileEnergy {
	if report == nil || powerWatts <= 0 {
		return nil
	}
	estimate := &driverProfileEnergy{
		Method:     "estimated_wall_clock_seconds_times_average_active_watts",
		PowerWatts: powerWatts,
	}
	if report.Summary.TotalDuration > 0 {
		estimate.TotalJoules = durationJoules(report.Summary.TotalDuration, powerWatts)
	}
	if report.Summary.VisibleTokens > 0 && estimate.TotalJoules > 0 {
		estimate.JoulesPerVisibleToken = estimate.TotalJoules / float64(report.Summary.VisibleTokens)
	}

	setup, replay, speedup := driverProfilePromptSetupDurations(report.Runs)
	estimate.PromptSetupDuration = setup
	estimate.PromptSetupJoules = durationJoules(setup, powerWatts)
	estimate.ReplayPromptSetupDuration = replay
	estimate.ReplayPromptSetupJoules = durationJoules(replay, powerWatts)
	if replay > setup {
		estimate.PromptSetupSavedDuration = replay - setup
		estimate.PromptSetupSavedJoules = durationJoules(estimate.PromptSetupSavedDuration, powerWatts)
	}
	estimate.PromptSetupSpeedup = speedup
	return estimate
}

func driverProfilePromptSetupDurations(runs []driverProfileRun) (time.Duration, time.Duration, float64) {
	successfulRuns := 0
	actual := time.Duration(0)
	coldPromptSetup := time.Duration(0)
	for _, run := range runs {
		if run.Error != "" {
			continue
		}
		successfulRuns++
		if run.Metrics.PrefillDuration <= 0 {
			continue
		}
		actual += run.Metrics.PrefillDuration
		if coldPromptSetup == 0 {
			coldPromptSetup = run.Metrics.PrefillDuration
		}
		if run.Metrics.PromptCacheMisses > 0 || run.Metrics.PromptCacheMissTokens > 0 {
			coldPromptSetup = run.Metrics.PrefillDuration
		}
	}
	replay := time.Duration(0)
	if successfulRuns > 0 && coldPromptSetup > 0 {
		replay = coldPromptSetup * time.Duration(successfulRuns)
	}
	speedup := 0.0
	if actual > 0 && replay > 0 {
		speedup = float64(replay) / float64(actual)
	}
	return actual, replay, speedup
}

func durationJoules(duration time.Duration, powerWatts float64) float64 {
	if duration <= 0 || powerWatts <= 0 {
		return 0
	}
	return duration.Seconds() * powerWatts
}

func printDriverProfileSummary(stdout io.Writer, report *driverProfileReport) {
	if report == nil {
		return
	}
	core.WriteString(stdout, core.Sprintf("driver profile: %s\n", report.ModelPath))
	core.WriteString(stdout, core.Sprintf("  load: %s, runs: %d ok / %d failed\n", report.LoadDuration, report.Summary.SuccessfulRuns, report.Summary.FailedRuns))
	if report.Summary.RestoreAvgDuration > 0 {
		core.WriteString(stdout, core.Sprintf("  restore avg: %s\n", report.Summary.RestoreAvgDuration))
	}
	core.WriteString(stdout, core.Sprintf("  first token avg: %s, decode: %.1f tok/s\n", report.Summary.FirstTokenAvgDuration, report.Summary.DecodeTokensPerSecAverage))
	if proxy := report.Summary.DecodeBandwidthProxy; proxy != nil {
		core.WriteString(stdout, core.Sprintf("  bandwidth proxy: %.3f GB/token active+cache -> %.1f GB/s implied\n", proxy.ActivePlusCacheGBPerDecodeTokenProxy, proxy.ImpliedActivePlusCacheBandwidthGBPerSecProxy))
	}
	if report.EstimatedEnergy != nil {
		core.WriteString(stdout, core.Sprintf("  estimated energy: %.1f J at %.1f W", report.EstimatedEnergy.TotalJoules, report.EstimatedEnergy.PowerWatts))
		if report.EstimatedEnergy.PromptSetupSavedJoules > 0 {
			core.WriteString(stdout, core.Sprintf(", setup saved: %.1f J", report.EstimatedEnergy.PromptSetupSavedJoules))
		}
		core.WriteString(stdout, "\n")
	}
	core.WriteString(stdout, core.Sprintf("  generated: %d tokens, peak memory: %d MB, active+cache: %d MB, process virtual: %d MB, process resident: %d MB\n",
		report.Summary.GeneratedTokens,
		report.Summary.PeakMemoryBytes/1024/1024,
		report.Summary.ActivePlusCacheMemoryBytes/1024/1024,
		report.Summary.ProcessVirtualMemoryBytes/1024/1024,
		report.Summary.ProcessResidentMemoryBytes/1024/1024))
	if report.Summary.GoTotalAllocDeltaBytes > 0 || report.Summary.GoMallocsDelta > 0 {
		core.WriteString(stdout, core.Sprintf("  go allocs: %d bytes, %d mallocs, %.1f B/token, %.3f allocs/token\n",
			report.Summary.GoTotalAllocDeltaBytes,
			report.Summary.GoMallocsDelta,
			report.Summary.GoBytesPerGeneratedToken,
			report.Summary.GoAllocsPerGeneratedToken,
		))
	}
}

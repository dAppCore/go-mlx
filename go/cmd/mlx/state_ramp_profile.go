// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"context"
	"flag"
	"io"
	"sort"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference/bench"
	statefile "dappco.re/go/inference/state/filestore"
	mlx "dappco.re/go/mlx"
	"dappco.re/go/mlx/agent"
)

func runStateRampProfileCommand(ctx context.Context, args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet(cliCommandName("state-ramp-profile"), flag.ContinueOnError)
	fs.SetOutput(stderr)
	jsonOut := fs.Bool("json", false, "print JSON state ramp profile")
	reportFile := fs.String("report-file", "", "write JSON state ramp profile to a file")
	prompt := fs.String("prompt", defaultRetainedProfilePrompt, "source text to repeat into the warm and appended state")
	promptFile := fs.String("prompt-file", "", "read source text from a file")
	appendPrompt := fs.String("append-prompt", "", "source text for appended turn material; defaults to the seed prompt")
	appendFile := fs.String("append-file", "", "read appended turn material from a file")
	appendTurnDelimiter := fs.String("append-turn-delimiter", "", "split appended material into whole turn sections using this delimiter instead of fixed token offsets")
	turnPromptMode := fs.String("turn-prompt-mode", "reference", "turn prompt shape: reference wraps material, direct sends the turn text inside the chat template")
	wakeMarkerFile := fs.String("wake-marker-file", "", "start the ramp by waking this State compact marker or .kv container instead of prefilling the seed prompt")
	wakeStateStorePath := fs.String("wake-state-store", "", "existing append-only State file to wake before ramp turns")
	wakeIndexURI := fs.String("wake-index-uri", "", "State index URI to wake before ramp turns")
	chatTemplate := fs.String("chat-template", "", "chat template override for retained turns: gemma4, gemma, qwen, llama, or plain")
	enableThinking := fs.Bool("enable-thinking", false, "enable Gemma 4 thinking control token in the retained state ramp prompts")
	startTokens := fs.Int("start-tokens", 30000, "initial warmed-state token target")
	targetTokens := fs.Int("target-tokens", 100000, "final live-state token target")
	compactionThresholdTokens := fs.Int("compaction-threshold-tokens", 0, "live-state token count that marks the context exhausted and requires a folded state; 0 uses the context window")
	compactionTailTokens := fs.Int("compaction-tail-tokens", 8192, "recent live-state tail token budget to carry into the future folded-state summary")
	appendTokens := fs.Int("append-tokens", 8192, "maximum source tokens to append before each generation turn")
	turnMaxTokens := fs.Int("turn-max-tokens", mlx.ProductionLaneLongFormMaxTokens, "generated tokens per ramp turn")
	turnMinTokens := fs.Int("turn-min-tokens", 0, "debug-only visible token annotation threshold; 0 disables the annotation")
	turnMinTokensPolicy := fs.String("turn-min-tokens-policy", "mark", "debug handling for turns below the visible-token threshold: mark or fail")
	turns := fs.Int("turns", 0, "maximum ramp turns; 0 runs until target tokens are reached")
	temperature := fs.Float64("temperature", 1.0, "sampling temperature for generated turns")
	topP := fs.Float64("top-p", 0.95, "top-p sampling value for generated turns")
	topK := fs.Int("top-k", 64, "top-k sampling value for generated turns")
	repeatPenalty := fs.Float64("repeat-penalty", 1.0, "repeat penalty for generated turns")
	seed := fs.Uint64("seed", 0, "seed MLX sampling for reproducible retained-state turns; omitted leaves the current RNG stream")
	suppressEOS := fs.Bool("suppress-eos", false, "suppress the tokenizer EOS token during generated turns")
	includeOutput := fs.Bool("include-output", false, "include generated text in the report")
	traceTokenPhases := fs.Bool("trace-token-phases", false, "include per-token retained decode phase timings in turn metrics and summary")
	foldOnDegradation := fs.Bool("fold-on-degradation", false, "checkpoint, fold, wake, and continue from a fresh state when inspected output degrades before the target")
	degradationMinConsecutive := fs.Int("degradation-min-consecutive-turns", 2, "consecutive output-issue turns required before folding on retained-content degradation")
	foldStorePath := fs.String("fold-store", "", "append-only state store path for folded-state checkpoint artefacts")
	foldSummary := fs.String("fold-summary", "", "summary text to seed the folded state; empty uses a benchmark lifecycle summary")
	foldSummaryFile := fs.String("fold-summary-file", "", "read folded-state summary text from a file")
	foldSummaryGenerate := fs.Bool("fold-summary-generate", false, "generate folded-state summary text from the live session before creating the fresh folded State")
	foldSummaryPrompt := fs.String("fold-summary-prompt", defaultStateRampFoldSummaryPrompt, "prompt appended to the live session when -fold-summary-generate is enabled")
	foldSummaryPromptFile := fs.String("fold-summary-prompt-file", "", "read folded-state summary generation prompt text from a file")
	foldSummaryMaxTokens := fs.Int("fold-summary-max-tokens", 512, "maximum generated tokens for -fold-summary-generate")
	foldRecentTail := fs.String("fold-tail", "", "recent tail text to seed the folded state")
	foldRecentTailFile := fs.String("fold-tail-file", "", "read folded-state recent tail text from a file")
	foldPrefillChunkBytes := fs.Int("fold-prefill-chunk-bytes", 0, "byte chunk size for folded-state prefill; 0 uses the session default")
	foldContinuePrompt := fs.String("fold-continue-prompt", defaultStateRampFoldContinuePrompt, "prompt appended after waking the folded state")
	foldContinueMaxTokens := fs.Int("fold-continue-max-tokens", 512, "generated tokens for the folded-state wake/continue check; 0 skips the check")
	speculativeDraftModel := fs.String("speculative-draft-model", "", "assistant/draft model path for retained attached-assistant MTP report plumbing")
	speculativeDraftTokens := fs.Int("speculative-draft-tokens", mlx.ProductionMTPDefaultDraftTokens, "draft tokens proposed per retained attached-assistant MTP pass")
	contextLen := fs.Int("context", 0, "override context length")
	prefillChunkSize := fs.Int("prefill-chunk-size", 0, "override long-prompt prefill chunk size in tokens")
	cacheMode := fs.String("cache-mode", "", cacheModeFlagUsage)
	device := fs.String("device", "", "execution device: gpu or cpu")
	estimatePowerWatts := fs.Float64("estimate-power-watts", 0, "record an estimated average active power draw in watts")
	fastGemma4Lane := fs.Bool("fast-gemma4-lane", true, "enable the accepted Gemma 4 fast runtime gates by default; set false for baseline diagnostics")
	maxActiveMemoryBytes := fs.Uint64("max-active-memory-bytes", 0, "abort a turn if MLX active memory exceeds this many bytes; 0 derives from the resolved memory limit")
	maxProcessVirtualMemoryBytes := fs.Uint64("max-process-virtual-memory-bytes", 0, "abort a turn if process virtual memory exceeds this many bytes; 0 records process virtual memory without a hard cap")
	maxProcessResidentMemoryBytes := fs.Uint64("max-process-resident-memory-bytes", 0, "abort a turn if process resident memory exceeds this many bytes; 0 derives from the resolved memory limit")
	repeatedTokenLoopLimit := fs.Int("repeated-token-loop-limit", driverProfileDefaultRepeatedTokenLoopLimit, "abort when this many consecutive sampled tokens have the same token id")
	repeatedLineLoopLimit := fs.Int("repeated-line-loop-limit", profileDefaultRepeatedLineLoopLimit, "abort when this many consecutive visible non-empty lines repeat")
	repeatedSentenceLoopLimit := fs.Int("repeated-sentence-loop-limit", profileDefaultRepeatedSentenceLoopLimit, "abort when the same visible sentence repeats this many times in one output")
	fs.Usage = func() {
		name := cliName()
		core.WriteString(stderr, core.Sprintf("Usage: %s state-ramp-profile [flags] [model-path]\n", name))
		core.WriteString(stderr, "\n")
		core.WriteString(stderr, "Measure how retained State grows across multi-turn generation —\n")
		core.WriteString(stderr, "per-turn KV cache size, decode throughput as State accumulates,\n")
		core.WriteString(stderr, "memory growth curve. Used to characterise long-conversation\n")
		core.WriteString(stderr, "behaviour without the State-wake/restore round-trip that\n")
		core.WriteString(stderr, "state-wake-profile covers.\n")
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
		core.WriteString(stderr, core.Sprintf("  %s state-ramp-profile ~/models/lemer-lite\n", name))
		core.WriteString(stderr, core.Sprintf("    # default ramp shape, production lane\n"))
		core.WriteString(stderr, core.Sprintf("  %s state-ramp-profile -json -trace-token-phases ~/models/lemer-lite\n", name))
		core.WriteString(stderr, core.Sprintf("    # JSON + per-token phase trace across every turn\n"))
	}
	if err := fs.Parse(args); err != nil {
		if core.Is(err, flag.ErrHelp) {
			return 0
		}
		return 2
	}
	visitedFlags := driverProfileVisitedFlags(fs)
	if driverProfileFastGemma4LaneEnabled(*fastGemma4Lane, visitedFlags, "") {
		for _, restore := range applyGemma4FastLaneDefaults(
			visitedFlags,
			contextLen,
			cacheMode,
			mlx.ProductionLaneHyperLongContextLength,
		) {
			defer restore()
		}
	}
	if fs.NArg() != 1 {
		core.WriteString(stderr, core.Sprintf("%s state-ramp-profile: expected one model path\n", cliName()))
		fs.Usage()
		return 2
	}
	wakeStateStoreSegmentAlias := ""
	wakeStateStorePayloadOffset := int64(0)
	wakeStateStorePayloadBytes := int64(0)
	if core.Trim(*wakeMarkerFile) != "" {
		markerSource, err := stateWakeProfileMarkerSourceFromFile(*wakeMarkerFile)
		if err != nil {
			core.Print(stderr, "%s state-ramp-profile: wake marker file: %v", cliName(), err)
			return 1
		}
		if core.Trim(*wakeStateStorePath) == "" {
			*wakeStateStorePath = markerSource.Marker.StorePath
		}
		if core.Trim(*wakeIndexURI) == "" {
			*wakeIndexURI = markerSource.Marker.IndexURI
		}
		if !visitedFlags["start-tokens"] && markerSource.Marker.TokenCount > 0 {
			*startTokens = markerSource.Marker.TokenCount
		}
		wakeStateStoreSegmentAlias = markerSource.SegmentAlias
		wakeStateStorePayloadOffset = markerSource.PayloadOffset
		wakeStateStorePayloadBytes = markerSource.PayloadBytes
	}
	if core.Trim(*promptFile) != "" {
		read := core.ReadFile(*promptFile)
		if !read.OK {
			core.Print(stderr, "%s state-ramp-profile: prompt file: %v", cliName(), read.Value)
			return 1
		}
		*prompt = string(read.Value.([]byte))
	}
	if core.Trim(*appendFile) != "" {
		read := core.ReadFile(*appendFile)
		if !read.OK {
			core.Print(stderr, "%s state-ramp-profile: append file: %v", cliName(), read.Value)
			return 1
		}
		*appendPrompt = string(read.Value.([]byte))
	}
	if core.Trim(*foldSummaryFile) != "" {
		read := core.ReadFile(*foldSummaryFile)
		if !read.OK {
			core.Print(stderr, "%s state-ramp-profile: fold summary file: %v", cliName(), read.Value)
			return 1
		}
		*foldSummary = string(read.Value.([]byte))
	}
	if core.Trim(*foldSummaryPromptFile) != "" {
		read := core.ReadFile(*foldSummaryPromptFile)
		if !read.OK {
			core.Print(stderr, "%s state-ramp-profile: fold summary prompt file: %v", cliName(), read.Value)
			return 1
		}
		*foldSummaryPrompt = string(read.Value.([]byte))
	}
	if core.Trim(*foldRecentTailFile) != "" {
		read := core.ReadFile(*foldRecentTailFile)
		if !read.OK {
			core.Print(stderr, "%s state-ramp-profile: fold tail file: %v", cliName(), read.Value)
			return 1
		}
		*foldRecentTail = string(read.Value.([]byte))
	}
	if *startTokens < 0 {
		core.WriteString(stderr, core.Sprintf("%s state-ramp-profile: start tokens must be >= 0\n", cliName()))
		return 2
	}
	if *targetTokens <= *startTokens {
		core.WriteString(stderr, core.Sprintf("%s state-ramp-profile: target tokens must be greater than start tokens\n", cliName()))
		return 2
	}
	if *compactionThresholdTokens < 0 {
		core.WriteString(stderr, core.Sprintf("%s state-ramp-profile: compaction threshold tokens must be >= 0\n", cliName()))
		return 2
	}
	if *compactionThresholdTokens == 0 && *contextLen > 0 {
		*compactionThresholdTokens = *contextLen
	}
	if *compactionTailTokens < 0 {
		core.WriteString(stderr, core.Sprintf("%s state-ramp-profile: compaction tail tokens must be >= 0\n", cliName()))
		return 2
	}
	if *appendTokens < 1 {
		core.WriteString(stderr, core.Sprintf("%s state-ramp-profile: append tokens must be >= 1\n", cliName()))
		return 2
	}
	if *turnMaxTokens < 1 {
		core.WriteString(stderr, core.Sprintf("%s state-ramp-profile: turn max tokens must be >= 1\n", cliName()))
		return 2
	}
	if *turnMinTokens < 0 {
		core.WriteString(stderr, core.Sprintf("%s state-ramp-profile: turn min tokens must be >= 0\n", cliName()))
		return 2
	}
	*turnMinTokensPolicy = core.Lower(core.Trim(*turnMinTokensPolicy))
	if *turnMinTokensPolicy == "" {
		*turnMinTokensPolicy = "fail"
	}
	if *turnMinTokensPolicy != "fail" && *turnMinTokensPolicy != "mark" {
		core.WriteString(stderr, core.Sprintf("%s state-ramp-profile: turn min tokens policy must be fail or mark\n", cliName()))
		return 2
	}
	if *turns < 0 {
		core.WriteString(stderr, core.Sprintf("%s state-ramp-profile: turns must be >= 0\n", cliName()))
		return 2
	}
	if *prefillChunkSize < 0 {
		core.WriteString(stderr, core.Sprintf("%s state-ramp-profile: prefill chunk size must be >= 0\n", cliName()))
		return 2
	}
	if *estimatePowerWatts < 0 {
		core.WriteString(stderr, core.Sprintf("%s state-ramp-profile: estimated power watts must be >= 0\n", cliName()))
		return 2
	}
	if *temperature < 0 {
		core.WriteString(stderr, core.Sprintf("%s state-ramp-profile: temperature must be >= 0\n", cliName()))
		return 2
	}
	if *topP < 0 {
		core.WriteString(stderr, core.Sprintf("%s state-ramp-profile: top-p must be >= 0\n", cliName()))
		return 2
	}
	if *topK < 0 {
		core.WriteString(stderr, core.Sprintf("%s state-ramp-profile: top-k must be >= 0\n", cliName()))
		return 2
	}
	if *repeatPenalty < 0 {
		core.WriteString(stderr, core.Sprintf("%s state-ramp-profile: repeat penalty must be >= 0\n", cliName()))
		return 2
	}
	if *degradationMinConsecutive < 1 {
		core.WriteString(stderr, core.Sprintf("%s state-ramp-profile: degradation min consecutive turns must be >= 1\n", cliName()))
		return 2
	}
	foldRequested := *foldOnDegradation ||
		core.Trim(*foldSummary) != "" ||
		*foldSummaryGenerate ||
		core.Trim(*foldRecentTail) != ""
	if foldRequested && core.Trim(*foldStorePath) == "" {
		core.WriteString(stderr, core.Sprintf("%s state-ramp-profile: fold store path is required when folding is enabled\n", cliName()))
		return 2
	}
	wakeRequested := core.Trim(*wakeStateStorePath) != "" || core.Trim(*wakeIndexURI) != ""
	if wakeRequested && core.Trim(*wakeStateStorePath) == "" {
		core.WriteString(stderr, core.Sprintf("%s state-ramp-profile: wake state store path is required\n", cliName()))
		return 2
	}
	if wakeRequested && core.Trim(*wakeIndexURI) == "" {
		core.WriteString(stderr, core.Sprintf("%s state-ramp-profile: wake index URI is required\n", cliName()))
		return 2
	}
	if *foldPrefillChunkBytes < 0 {
		core.WriteString(stderr, core.Sprintf("%s state-ramp-profile: fold prefill chunk bytes must be >= 0\n", cliName()))
		return 2
	}
	if *foldContinueMaxTokens < 0 {
		core.WriteString(stderr, core.Sprintf("%s state-ramp-profile: fold continue max tokens must be >= 0\n", cliName()))
		return 2
	}
	if *foldSummaryMaxTokens < 1 {
		core.WriteString(stderr, core.Sprintf("%s state-ramp-profile: fold summary max tokens must be >= 1\n", cliName()))
		return 2
	}
	if *foldSummaryGenerate && core.Trim(*foldSummary) != "" {
		core.WriteString(stderr, core.Sprintf("%s state-ramp-profile: fold summary generation cannot be combined with explicit fold summary text\n", cliName()))
		return 2
	}
	if *foldSummaryGenerate && core.Trim(*foldSummaryPrompt) == "" {
		core.WriteString(stderr, core.Sprintf("%s state-ramp-profile: fold summary prompt must not be empty when generation is enabled\n", cliName()))
		return 2
	}
	if *speculativeDraftTokens < 0 {
		core.WriteString(stderr, core.Sprintf("%s state-ramp-profile: speculative draft tokens must be >= 0\n", cliName()))
		return 2
	}
	if *repeatedTokenLoopLimit < 1 {
		core.WriteString(stderr, core.Sprintf("%s state-ramp-profile: repeated token loop limit must be >= 1\n", cliName()))
		return 2
	}
	if *repeatedLineLoopLimit < 1 {
		core.WriteString(stderr, core.Sprintf("%s state-ramp-profile: repeated line loop limit must be >= 1\n", cliName()))
		return 2
	}
	if *repeatedSentenceLoopLimit < 1 {
		core.WriteString(stderr, core.Sprintf("%s state-ramp-profile: repeated sentence loop limit must be >= 1\n", cliName()))
		return 2
	}
	loadOptions := []mlx.LoadOption{}
	var loadSettings *tuneProfileLoadSettings
	if *contextLen > 0 {
		loadOptions = append(loadOptions, mlx.WithContextLength(*contextLen))
		loadSettings = &tuneProfileLoadSettings{ContextLength: *contextLen}
	}
	if *prefillChunkSize > 0 {
		loadOptions = append(loadOptions, mlx.WithPrefillChunkSize(*prefillChunkSize))
		if loadSettings == nil {
			loadSettings = &tuneProfileLoadSettings{}
		}
		loadSettings.PrefillChunkSize = *prefillChunkSize
	}
	if mode, ok := parseRuntimeCacheMode(*cacheMode); ok {
		if !isRuntimeCacheMode(mode) {
			core.WriteString(stderr, core.Sprintf("%s state-ramp-profile: unsupported cache mode %q\n", cliName(), string(mode)))
			return 2
		}
		loadOptions = append(loadOptions, mlx.WithKVCacheMode(mode))
		if loadSettings == nil {
			loadSettings = &tuneProfileLoadSettings{}
		}
		loadSettings.CacheMode = string(mode)
	}
	if *device != "" {
		loadOptions = append(loadOptions, mlx.WithDevice(*device))
	}

	report, err := runStateRampProfileGuarded(ctx, fs.Arg(0), loadOptions, stateRampProfileOptions{
		Prompt:                      *prompt,
		PromptSet:                   visitedFlags["prompt"] || visitedFlags["prompt-file"],
		AppendPrompt:                *appendPrompt,
		AppendTurnDelimiter:         *appendTurnDelimiter,
		TurnPromptMode:              *turnPromptMode,
		WakeMarkerFile:              core.Trim(*wakeMarkerFile),
		WakeStateStorePath:          core.Trim(*wakeStateStorePath),
		WakeStateStoreSegmentAlias:  core.Trim(wakeStateStoreSegmentAlias),
		WakeStateStorePayloadOffset: wakeStateStorePayloadOffset,
		WakeStateStorePayloadBytes:  wakeStateStorePayloadBytes,
		WakeIndexURI:                core.Trim(*wakeIndexURI),
		ChatTemplate:                *chatTemplate,
		EnableThinking:              *enableThinking,
		StartTokens:                 *startTokens,
		TargetTokens:                *targetTokens,
		CompactionThresholdTokens:   *compactionThresholdTokens,
		CompactionTailTokens:        *compactionTailTokens,
		AppendTokens:                *appendTokens,
		TurnMaxTokens:               *turnMaxTokens,
		TurnMinTokens:               *turnMinTokens,
		TurnMinTokensPolicy:         *turnMinTokensPolicy,
		Turns:                       *turns,
		Temperature:                 *temperature,
		TopP:                        *topP,
		TopK:                        *topK,
		RepeatPenalty:               *repeatPenalty,
		Seed:                        *seed,
		SeedSet:                     visitedFlags["seed"],
		SuppressEOS:                 *suppressEOS,
		IncludeOutput:               *includeOutput,
		TraceTokenPhases:            *traceTokenPhases,
		FoldOnDegradation:           *foldOnDegradation,
		DegradationMinConsecutive:   *degradationMinConsecutive,
		FoldStorePath:               core.Trim(*foldStorePath),
		FoldSummary:                 *foldSummary,
		FoldSummaryGenerate:         *foldSummaryGenerate,
		FoldSummaryPrompt:           *foldSummaryPrompt,
		FoldSummaryMaxTokens:        *foldSummaryMaxTokens,
		FoldRecentTail:              *foldRecentTail,
		FoldPrefillChunkBytes:       *foldPrefillChunkBytes,
		FoldContinuePrompt:          *foldContinuePrompt,
		FoldContinueMaxTokens:       *foldContinueMaxTokens,
		SpeculativeDraftModelPath:   core.Trim(*speculativeDraftModel),
		SpeculativeDraftTokens:      driverProfileSpeculativeDraftTokensForReport(core.Trim(*speculativeDraftModel), *speculativeDraftTokens),
		SpeculativeGenerationMode:   stateRampProfileSpeculativeGenerationMode(core.Trim(*speculativeDraftModel)),
		SafetyLimits: driverProfileSafetyLimits{
			MaxActiveMemoryBytes:          *maxActiveMemoryBytes,
			MaxProcessVirtualMemoryBytes:  *maxProcessVirtualMemoryBytes,
			MaxProcessResidentMemoryBytes: *maxProcessResidentMemoryBytes,
			RepeatedTokenLoopLimit:        *repeatedTokenLoopLimit,
			RepeatedLineLoopLimit:         *repeatedLineLoopLimit,
			RepeatedSentenceLoopLimit:     *repeatedSentenceLoopLimit,
		},
	})
	if report != nil && loadSettings != nil {
		report.Load = mergeDriverProfileLoadSettings(loadSettings, report.Load)
	}
	annotateStateRampProfileFoldDurations(report)
	if report != nil && *estimatePowerWatts > 0 {
		report.EstimatedEnergy = estimateStateRampProfileEnergy(report, *estimatePowerWatts)
	}
	reportPath := core.Trim(*reportFile)
	if *jsonOut || reportPath != "" {
		if report == nil {
			report = &stateRampProfileReport{
				Version:                     1,
				ModelPath:                   fs.Arg(0),
				PromptBytes:                 len(*prompt),
				AppendPromptBytes:           len(*appendPrompt),
				AppendTurnSections:          0,
				WakeMarkerFile:              core.Trim(*wakeMarkerFile),
				WakeStateStorePath:          core.Trim(*wakeStateStorePath),
				WakeStateStoreAlias:         core.Trim(wakeStateStoreSegmentAlias),
				WakeStateStorePayloadOffset: wakeStateStorePayloadOffset,
				WakeStateStorePayloadBytes:  wakeStateStorePayloadBytes,
				WakeIndexURI:                core.Trim(*wakeIndexURI),
				ChatTemplate:                *chatTemplate,
				EnableThinking:              *enableThinking,
				StartTokens:                 *startTokens,
				TargetTokens:                *targetTokens,
				CompactionThresholdTokens:   *compactionThresholdTokens,
				CompactionTailTokens:        *compactionTailTokens,
				AppendTokens:                *appendTokens,
				TurnMaxTokens:               *turnMaxTokens,
				TurnMinTokens:               *turnMinTokens,
				TurnMinTokensPolicy:         *turnMinTokensPolicy,
				TurnPromptMode:              *turnPromptMode,
				RequestedTurns:              *turns,
				Temperature:                 *temperature,
				TopP:                        *topP,
				TopK:                        *topK,
				RepeatPenalty:               *repeatPenalty,
				SuppressEOS:                 *suppressEOS,
				IncludeOutput:               *includeOutput,
				TraceTokenPhases:            *traceTokenPhases,
				FoldOnDegradation:           *foldOnDegradation,
				DegradationMinConsecutive:   *degradationMinConsecutive,
				FoldStorePath:               core.Trim(*foldStorePath),
				FoldSummaryBytes:            len(*foldSummary),
				FoldSummaryGenerate:         *foldSummaryGenerate,
				FoldSummaryPromptBytes:      len(*foldSummaryPrompt),
				FoldSummaryMaxTokens:        *foldSummaryMaxTokens,
				FoldRecentTailBytes:         len(*foldRecentTail),
				FoldPrefillChunkBytes:       *foldPrefillChunkBytes,
				FoldContinueMaxTokens:       *foldContinueMaxTokens,
				SpeculativeDraftModelPath:   core.Trim(*speculativeDraftModel),
				SpeculativeDraftTokens:      driverProfileSpeculativeDraftTokensForReport(core.Trim(*speculativeDraftModel), *speculativeDraftTokens),
				SpeculativeGenerationMode:   stateRampProfileSpeculativeGenerationMode(core.Trim(*speculativeDraftModel)),
			}
		}
		if report.SpeculativeGenerationMode == "" {
			report.SpeculativeGenerationMode = stateRampProfileSpeculativeGenerationMode(report.SpeculativeDraftModelPath)
		}
		if err != nil && report.Error == "" {
			report.Error = err.Error()
		}
		data := core.JSONMarshalIndent(report, "", "  ")
		if !data.OK {
			core.Print(stderr, "%s state-ramp-profile: marshal report failed", cliName())
			return 1
		}
		if reportPath != "" {
			if writeErr := writeJSONReportFile(reportPath, data.Value.([]byte)); writeErr != nil {
				core.Print(stderr, "%s state-ramp-profile: write report file: %v", cliName(), writeErr)
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
		core.Print(stderr, "%s state-ramp-profile: %v", cliName(), err)
		return 1
	}
	printStateRampProfileSummary(stdout, report)
	return 0
}

var runStateRampProfile = defaultRunStateRampProfile

func runStateRampProfileGuarded(ctx context.Context, modelPath string, loadOptions []mlx.LoadOption, opts stateRampProfileOptions) (report *stateRampProfileReport, err error) {
	defer func() {
		if recovered := recover(); recovered != nil {
			err = core.NewError(core.Sprintf("state-ramp-profile panic: %v", recovered))
		}
	}()
	return runStateRampProfile(ctx, modelPath, loadOptions, opts)
}

func defaultRunStateRampProfile(ctx context.Context, modelPath string, loadOptions []mlx.LoadOption, opts stateRampProfileOptions) (*stateRampProfileReport, error) {
	opts = normalizeStateRampProfileOptions(opts)
	report := &stateRampProfileReport{
		Version:                     1,
		ModelPath:                   modelPath,
		PromptBytes:                 len(opts.Prompt),
		AppendPromptBytes:           len(opts.AppendPrompt),
		WakeMarkerFile:              opts.WakeMarkerFile,
		WakeStateStorePath:          opts.WakeStateStorePath,
		WakeStateStoreAlias:         opts.WakeStateStoreSegmentAlias,
		WakeStateStorePayloadOffset: opts.WakeStateStorePayloadOffset,
		WakeStateStorePayloadBytes:  opts.WakeStateStorePayloadBytes,
		WakeIndexURI:                opts.WakeIndexURI,
		EnableThinking:              opts.EnableThinking,
		StartTokens:                 opts.StartTokens,
		TargetTokens:                opts.TargetTokens,
		CompactionThresholdTokens:   opts.CompactionThresholdTokens,
		CompactionTailTokens:        opts.CompactionTailTokens,
		AppendTokens:                opts.AppendTokens,
		TurnMaxTokens:               opts.TurnMaxTokens,
		TurnMinTokens:               opts.TurnMinTokens,
		TurnMinTokensPolicy:         opts.TurnMinTokensPolicy,
		TurnPromptMode:              opts.TurnPromptMode,
		RequestedTurns:              opts.Turns,
		Temperature:                 opts.Temperature,
		TopP:                        opts.TopP,
		TopK:                        opts.TopK,
		RepeatPenalty:               opts.RepeatPenalty,
		Seed:                        opts.Seed,
		SeedSet:                     opts.SeedSet,
		SuppressEOS:                 opts.SuppressEOS,
		IncludeOutput:               opts.IncludeOutput,
		TraceTokenPhases:            opts.TraceTokenPhases,
		FoldOnDegradation:           opts.FoldOnDegradation,
		DegradationMinConsecutive:   opts.DegradationMinConsecutive,
		FoldStorePath:               opts.FoldStorePath,
		FoldSummaryBytes:            len(opts.FoldSummary),
		FoldSummaryGenerate:         opts.FoldSummaryGenerate,
		FoldSummaryPromptBytes:      len(opts.FoldSummaryPrompt),
		FoldSummaryMaxTokens:        opts.FoldSummaryMaxTokens,
		FoldRecentTailBytes:         len(opts.FoldRecentTail),
		FoldPrefillChunkBytes:       opts.FoldPrefillChunkBytes,
		FoldContinueMaxTokens:       opts.FoldContinueMaxTokens,
		SpeculativeDraftModelPath:   opts.SpeculativeDraftModelPath,
		SpeculativeDraftTokens:      opts.SpeculativeDraftTokens,
		SpeculativeGenerationMode:   stateRampProfileSpeculativeGenerationMode(opts.SpeculativeDraftModelPath),
		SafetyLimits:                opts.SafetyLimits,
		RuntimeGates:                driverProfileRuntimeGates(),
	}
	loadStart := time.Now()
	model, err := loadBenchModel(modelPath, loadOptions...)
	report.LoadDuration = bench.NonZeroDuration(time.Since(loadStart))
	if err != nil {
		report.Error = err.Error()
		return report, err
	}
	if model == nil {
		err := core.NewError("mlx: state ramp profile loaded nil model")
		report.Error = err.Error()
		return report, err
	}
	modelInfo := model.Info()
	if opts.CompactionThresholdTokens <= 0 {
		opts.CompactionThresholdTokens = stateRampProfileDefaultCompactionThreshold(opts, modelInfo)
	}
	report.CompactionThresholdTokens = opts.CompactionThresholdTokens
	report.Load = mergeDriverProfileLoadSettings(report.Load, loadSettingsFromModelInfo(modelInfo))
	opts.SafetyLimits = resolveDriverProfileSafetyLimits(opts.SafetyLimits, report.Load)
	report.SafetyLimits = opts.SafetyLimits
	defer model.Close()
	if err := driverProfileMetricsSafetyError("load", model.Metrics(), opts.SafetyLimits); err != nil {
		report.Error = err.Error()
		return report, err
	}
	opts.ChatTemplate = chapterProfileTemplate(opts.ChatTemplate, modelInfo.Architecture)
	report.ChatTemplate = opts.ChatTemplate
	tok := model.Tokenizer()
	if tok == nil {
		err := core.NewError("state-ramp-profile: model tokenizer is nil")
		report.Error = err.Error()
		return report, err
	}
	report.StopTokenIDs, report.SuppressTokenIDs = chapterProfileTemplateTokenControls(opts.ChatTemplate, tok)
	report.SuppressTokenIDs = stateRampProfileEffectiveSuppressTokenIDs(report.SuppressTokenIDs, report.StopTokenIDs, tok, opts.SuppressEOS)
	sourceTokens, err := tok.Encode(opts.Prompt)
	if err != nil {
		report.Error = err.Error()
		return report, err
	}
	report.SourceTokens = len(sourceTokens)
	appendText := opts.AppendPrompt
	if appendText == "" {
		appendText = opts.Prompt
		report.AppendPromptBytes = len(appendText)
	}
	appendSourceTokens, appendTurnSections, err := stateRampProfileAppendSources(tok, appendText, opts.AppendTurnDelimiter, opts.ChatTemplate, opts.EnableThinking, opts.TurnMinTokens, opts.TurnPromptMode)
	if err != nil {
		report.Error = err.Error()
		return report, err
	}
	report.AppendSourceTokens = countStateRampAppendSourceTokens(appendSourceTokens, appendTurnSections)
	report.AppendTurnSections = len(appendTurnSections)
	var wakeStore *statefile.Store
	var session *mlx.ModelSession
	initialSetupDuration := time.Duration(0)
	currentTokens := 0
	if opts.WakeStateStorePath != "" || opts.WakeIndexURI != "" {
		openStart := time.Now()
		if opts.WakeStateStorePayloadOffset > 0 || opts.WakeStateStorePayloadBytes > 0 {
			wakeStore, err = statefile.OpenRegionWithSegmentAlias(ctx, opts.WakeStateStorePath, opts.WakeStateStorePayloadOffset, opts.WakeStateStorePayloadBytes, opts.WakeStateStoreSegmentAlias)
		} else if opts.WakeStateStoreSegmentAlias != "" {
			wakeStore, err = statefile.OpenWithSegmentAlias(ctx, opts.WakeStateStorePath, opts.WakeStateStoreSegmentAlias)
		} else {
			wakeStore, err = statefile.Open(ctx, opts.WakeStateStorePath)
		}
		report.InitialWakeStoreOpenDuration = bench.NonZeroDuration(time.Since(openStart))
		if err != nil {
			report.Error = err.Error()
			return report, err
		}
		defer wakeStore.Close()
		wakeStart := time.Now()
		session, report.InitialWake, err = model.WakeAgentMemory(ctx, wakeStore, agent.WakeOptions{IndexURI: opts.WakeIndexURI})
		report.InitialWakeDuration = bench.NonZeroDuration(time.Since(wakeStart))
		initialSetupDuration = report.InitialWakeDuration
		if err != nil {
			report.Error = err.Error()
			return report, err
		}
		if report.InitialWake != nil {
			currentTokens = report.InitialWake.PrefixTokens
			report.InitialPrefillTokens = currentTokens
		}
		report.InitialSetupMetrics = profileLiveMetrics()
		if err := driverProfileMetricsSafetyError("initial wake", report.InitialSetupMetrics, opts.SafetyLimits); err != nil {
			report.Error = err.Error()
			return report, err
		}
		mlx.ClearCache()
		report.InitialSetupPostClearMetrics = profileLiveMetrics()
	} else {
		session, err = model.NewSession()
		if err != nil {
			report.Error = err.Error()
			return report, err
		}
		if len(sourceTokens) > 0 {
			seedTokens, err := stateRampProfileSeedTokens(tok, sourceTokens, opts)
			if err != nil {
				report.Error = err.Error()
				return report, err
			}
			prefillStart := time.Now()
			err = session.PrefillTokens(ctx, seedTokens)
			report.InitialPrefillDuration = bench.NonZeroDuration(time.Since(prefillStart))
			report.InitialPrefillTokens = len(seedTokens)
			initialSetupDuration = report.InitialPrefillDuration
			if err != nil {
				report.Error = err.Error()
				return report, err
			}
			currentTokens = len(seedTokens)
		}
		report.InitialSetupMetrics = profileLiveMetrics()
		if err := driverProfileMetricsSafetyError("initial prefill", report.InitialSetupMetrics, opts.SafetyLimits); err != nil {
			report.Error = err.Error()
			return report, err
		}
		mlx.ClearCache()
		report.InitialSetupPostClearMetrics = profileLiveMetrics()
	}
	defer session.Close()

	initialTokens := currentTokens
	sourceOffset := 0
	consecutiveContentIssues := 0
	var firstErr error
	for turnIndex := 1; shouldRunStateRampTurn(turnIndex, currentTokens, opts); turnIndex++ {
		turnSourceTokens, turnSourceOffset, appendCount := stateRampProfileTurnAppendSource(appendSourceTokens, appendTurnSections, sourceOffset, currentTokens, turnIndex, opts)
		turn := stateRampProfileGenerateTurn(ctx, model, session, turnSourceTokens, turnSourceOffset, appendCount, currentTokens, turnIndex, opts)
		if len(appendTurnSections) == 0 {
			sourceOffset += turn.AppendedTokens
		}
		if turn.TokensAfterGenerate > 0 {
			currentTokens = turn.TokensAfterGenerate
		} else {
			currentTokens += turn.AppendedTokens
		}
		if turn.Error != "" && firstErr == nil {
			if stateRampProfileTurnErrorFatal(turn, opts) {
				firstErr = core.NewError(turn.Error)
			}
		}
		if stateRampProfileTurnHasContentIssue(turn) {
			consecutiveContentIssues++
		} else {
			consecutiveContentIssues = 0
		}
		report.Turns = append(report.Turns, turn)
		mlx.ClearCache()
		if turn.Error != "" && stateRampProfileTurnErrorFatal(turn, opts) {
			break
		}
		if stateRampProfileDegradationFoldReached(consecutiveContentIssues, opts) {
			break
		}
	}
	report.Summary = summariseStateRampProfileTurns(initialSetupDuration, initialTokens, report.Turns, opts)
	if stateRampProfileShouldRunFold(report.Summary, opts) {
		report.Fold = stateRampProfileFoldExhausted(ctx, model, session, report, opts)
		annotateStateRampProfileFoldDurations(report)
		if report.Fold != nil && report.Fold.Error != "" && firstErr == nil {
			firstErr = core.NewError(report.Fold.Error)
		}
	}
	if firstErr != nil {
		report.Error = firstErr.Error()
		return report, firstErr
	}
	return report, nil
}

func normalizeStateRampProfileOptions(opts stateRampProfileOptions) stateRampProfileOptions {
	opts.Prompt = core.Trim(opts.Prompt)
	opts.AppendPrompt = core.Trim(opts.AppendPrompt)
	opts.WakeMarkerFile = core.Trim(opts.WakeMarkerFile)
	opts.WakeStateStorePath = core.Trim(opts.WakeStateStorePath)
	opts.WakeStateStoreSegmentAlias = core.Trim(opts.WakeStateStoreSegmentAlias)
	opts.WakeIndexURI = core.Trim(opts.WakeIndexURI)
	if opts.Prompt == "" && !opts.PromptSet {
		opts.Prompt = defaultRetainedProfilePrompt
	}
	if opts.StartTokens < 0 || (opts.StartTokens == 0 && opts.Prompt != "") {
		opts.StartTokens = 30000
	}
	if opts.TargetTokens <= 0 {
		opts.TargetTokens = 100000
	}
	if opts.CompactionThresholdTokens < 0 {
		opts.CompactionThresholdTokens = 0
	}
	if opts.CompactionTailTokens < 0 {
		opts.CompactionTailTokens = 0
	}
	if opts.AppendTokens <= 0 {
		opts.AppendTokens = 8192
	}
	if opts.TurnMaxTokens <= 0 {
		opts.TurnMaxTokens = mlx.ProductionLaneLongFormMaxTokens
	}
	if opts.TurnMinTokens < 0 {
		opts.TurnMinTokens = 0
	}
	opts.TurnMinTokensPolicy = core.Lower(core.Trim(opts.TurnMinTokensPolicy))
	if opts.TurnMinTokensPolicy == "" {
		opts.TurnMinTokensPolicy = "mark"
	}
	if opts.TurnMinTokensPolicy != "mark" && opts.TurnMinTokensPolicy != "fail" {
		opts.TurnMinTokensPolicy = "mark"
	}
	opts.TurnPromptMode = core.Lower(core.Trim(opts.TurnPromptMode))
	if opts.TurnPromptMode == "" {
		opts.TurnPromptMode = "reference"
	}
	if opts.TurnPromptMode != "reference" && opts.TurnPromptMode != "direct" {
		opts.TurnPromptMode = "reference"
	}
	if opts.DegradationMinConsecutive <= 0 {
		opts.DegradationMinConsecutive = 2
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
	opts.FoldStorePath = core.Trim(opts.FoldStorePath)
	opts.FoldSummary = core.Trim(opts.FoldSummary)
	opts.FoldSummaryPrompt = core.Trim(opts.FoldSummaryPrompt)
	if opts.FoldSummaryPrompt == "" {
		opts.FoldSummaryPrompt = defaultStateRampFoldSummaryPrompt
	}
	if opts.FoldSummaryMaxTokens <= 0 {
		opts.FoldSummaryMaxTokens = 512
	}
	opts.FoldRecentTail = core.Trim(opts.FoldRecentTail)
	if opts.FoldPrefillChunkBytes < 0 {
		opts.FoldPrefillChunkBytes = 0
	}
	if opts.FoldContinueMaxTokens < 0 {
		opts.FoldContinueMaxTokens = 0
	}
	if opts.FoldContinuePrompt == "" {
		opts.FoldContinuePrompt = defaultStateRampFoldContinuePrompt
	}
	return opts
}

func shouldRunStateRampTurn(index, currentTokens int, opts stateRampProfileOptions) bool {
	if stateRampProfileLiveTokenLimitReached(currentTokens, opts) {
		return false
	}
	if opts.Turns > 0 {
		return index <= opts.Turns
	}
	return currentTokens < opts.TargetTokens
}

func stateRampProfileLiveTokenLimitReached(currentTokens int, opts stateRampProfileOptions) bool {
	limit := stateRampProfileLiveTokenLimit(opts)
	return limit > 0 && currentTokens >= limit
}

func stateRampProfileLiveTokenLimit(opts stateRampProfileOptions) int {
	limit := opts.TargetTokens
	if stateRampProfileCompactionStopArmed(opts) && opts.CompactionThresholdTokens > 0 && (limit <= 0 || opts.CompactionThresholdTokens < limit) {
		limit = opts.CompactionThresholdTokens
	}
	return limit
}

func stateRampProfileCompactionStopArmed(opts stateRampProfileOptions) bool {
	return core.Trim(opts.FoldStorePath) != ""
}

func stateRampProfileDefaultCompactionThreshold(opts stateRampProfileOptions, info mlx.ModelInfo) int {
	if opts.CompactionThresholdTokens > 0 {
		return opts.CompactionThresholdTokens
	}
	if info.ContextLength > 0 {
		return info.ContextLength
	}
	return opts.TargetTokens
}

func repeatedStateRampTokens(source []int32, offset, count int) []int32 {
	if len(source) == 0 || count <= 0 {
		return nil
	}
	offset %= len(source)
	if offset < 0 {
		offset += len(source)
	}
	if count <= len(source)-offset {
		return source[offset : offset+count]
	}
	out := make([]int32, count)
	for i := range out {
		out[i] = source[(offset+i)%len(source)]
	}
	return out
}

func forEachRepeatedStateRampTokenSpan(source []int32, offset, count int, yield func([]int32) error) (int, error) {
	if len(source) == 0 || count <= 0 {
		return 0, nil
	}
	if yield == nil {
		return 0, core.NewError("state-ramp-profile: nil token span callback")
	}
	offset %= len(source)
	if offset < 0 {
		offset += len(source)
	}
	appended := 0
	for appended < count {
		spanLen := len(source) - offset
		if remaining := count - appended; spanLen > remaining {
			spanLen = remaining
		}
		if spanLen <= 0 {
			offset = 0
			continue
		}
		if err := yield(source[offset : offset+spanLen]); err != nil {
			return appended, err
		}
		appended += spanLen
		offset = 0
	}
	return appended, nil
}

type stateRampProfileTokenizer interface {
	Encode(string) ([]int32, error)
	Decode([]int32) (string, error)
}

func stateRampProfileSeedTokens(tok stateRampProfileTokenizer, sourceTokens []int32, opts stateRampProfileOptions) ([]int32, error) {
	if len(sourceTokens) == 0 {
		return nil, core.NewError("state-ramp-profile: source prompt produced no tokens")
	}
	if stateRampProfilePlainTemplate(opts.ChatTemplate) {
		return repeatedStateRampTokens(sourceTokens, 0, opts.StartTokens), nil
	}
	target := opts.StartTokens
	if target <= 0 {
		target = len(sourceTokens)
	}
	contextBudget := target
	for contextBudget >= 0 {
		contextText, err := tok.Decode(repeatedStateRampTokens(sourceTokens, 0, contextBudget))
		if err != nil {
			return nil, err
		}
		wrapped := stateRampProfileInitialPrompt(opts.ChatTemplate, contextText, opts.EnableThinking)
		tokens, err := tok.Encode(wrapped)
		if err != nil {
			return nil, err
		}
		if len(tokens) <= target || contextBudget == 0 {
			return tokens, nil
		}
		overage := max(len(tokens)-target, 1)
		contextBudget -= overage
	}
	return nil, core.NewError("state-ramp-profile: could not fit chat-wrapped seed prompt")
}

func stateRampProfilePlainTemplate(template string) bool {
	template = core.Lower(core.Trim(template))
	return template == "" || template == "plain"
}

func stateRampProfileInitialPrompt(template, contextPrompt string, enableThinking bool) string {
	contextPrompt = core.Trim(contextPrompt)
	switch template {
	case "gemma4":
		builder := core.NewBuilder()
		builder.WriteString("<bos><|turn>system\n")
		if enableThinking {
			builder.WriteString("<|think|>\n")
		}
		builder.WriteString(defaultStateRampRetainedSystemPrompt)
		builder.WriteString("\n\n")
		builder.WriteString(contextPrompt)
		builder.WriteString("<turn|>\n<|turn>model\n")
		builder.WriteString("Ready.<turn|>\n")
		return builder.String()
	case "gemma":
		builder := core.NewBuilder()
		builder.Grow(len(contextPrompt) + len(defaultStateRampRetainedSystemPrompt) + 96)
		builder.WriteString("<bos><start_of_turn>user\n")
		builder.WriteString(defaultStateRampRetainedSystemPrompt)
		if contextPrompt != "" {
			builder.WriteString("\n\n")
			builder.WriteString(contextPrompt)
		}
		builder.WriteString("<end_of_turn>\n<start_of_turn>model\nReady.<end_of_turn>\n")
		return builder.String()
	case "qwen":
		return "<|im_start|>system\n" + defaultStateRampRetainedSystemPrompt + "\n\n" + contextPrompt + "<|im_end|>\n<|im_start|>assistant\nReady.<|im_end|>\n"
	case "llama":
		return "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n" + defaultStateRampRetainedSystemPrompt + "\n\n" + contextPrompt + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nReady.<|eot_id|>"
	default:
		return contextPrompt
	}
}

func stateRampProfileTurnPrompt(template, prompt string, enableThinking bool, minVisibleTokens ...int) string {
	return stateRampProfileTurnPromptWithMode(template, prompt, enableThinking, "reference", minVisibleTokens...)
}

func stateRampProfileDirectTurnPrompt(template, prompt string, enableThinking bool) string {
	return stateRampProfileTurnPromptWithMode(template, prompt, enableThinking, "direct")
}

func stateRampProfileTurnPromptWithMode(template, prompt string, enableThinking bool, mode string, minVisibleTokens ...int) string {
	prompt = core.Trim(prompt)
	mode = core.Lower(core.Trim(mode))
	if mode != "direct" {
		mode = "reference"
	}
	referenceMode := mode == "reference"
	switch template {
	case "gemma4":
		builder := core.NewBuilder()
		builder.Grow(len(prompt) + 768)
		builder.WriteString("<|turn>user\n")
		writeStateRampProfileTurnMaterial(builder, prompt, referenceMode)
		builder.WriteString("<turn|>\n<|turn>model\n")
		return builder.String()
	case "gemma":
		builder := core.NewBuilder()
		builder.Grow(len(prompt) + 768)
		builder.WriteString("<start_of_turn>user\n")
		writeStateRampProfileTurnMaterial(builder, prompt, referenceMode)
		builder.WriteString("<end_of_turn>\n<start_of_turn>model\n")
		return builder.String()
	case "qwen":
		builder := core.NewBuilder()
		builder.Grow(len(prompt) + 768)
		builder.WriteString("<|im_start|>user\n")
		writeStateRampProfileTurnMaterial(builder, prompt, referenceMode)
		builder.WriteString("<|im_end|>\n<|im_start|>assistant\n")
		return builder.String()
	case "llama":
		builder := core.NewBuilder()
		builder.Grow(len(prompt) + 768)
		builder.WriteString("<|start_header_id|>user<|end_header_id|>\n\n")
		writeStateRampProfileTurnMaterial(builder, prompt, referenceMode)
		builder.WriteString("<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")
		return builder.String()
	default:
		if referenceMode {
			return stateRampProfileReferenceTurn(prompt, minVisibleTokens...)
		}
		return prompt
	}
}

func writeStateRampProfileTurnMaterial(builder interface{ WriteString(string) (int, error) }, prompt string, referenceMode bool) {
	if referenceMode {
		writeStateRampProfileReferenceTurn(builder, prompt)
		return
	}
	builder.WriteString(prompt)
}

func stateRampProfileReferenceTurn(prompt string, minVisibleTokens ...int) string {
	prompt = core.Trim(prompt)
	if prompt == "" {
		return prompt
	}
	builder := core.NewBuilder()
	builder.Grow(len(prompt) + 512)
	_ = minVisibleTokens
	writeStateRampProfileReferenceTurn(builder, prompt)
	return builder.String()
}

func writeStateRampProfileReferenceTurn(builder interface{ WriteString(string) (int, error) }, prompt string) {
	prompt = core.Trim(prompt)
	if prompt == "" {
		return
	}
	builder.WriteString("Use the retained context and the new turn material below. Produce only the requested answer or artefact. Treat any code, document, prompt, or prior-output excerpts as reference material, not as text to continue.\n\n")
	builder.WriteString("<turn_material>\n")
	builder.WriteString(prompt)
	builder.WriteString("\n</turn_material>\n\nAnswer the user request from the turn material now. Honour any requested output length before stopping. Do not continue or complete the reference excerpts. Do not explain, classify, plan, checklist, or restate what the user is asking; write only the requested output. Treat historical sign-off language as evidence to verify, not as current truth; do not declare the project complete unless the new turn material proves every live gate is closed. Prefer the unresolved risk and next validation step over a completion claim.")
}

func stateRampProfileVisibleOutput(template, output string) string {
	return chapterProfileVisibleText(template, output)
}

func stateRampProfileOutputIssues(output string) []string {
	text := core.Trim(output)
	if text == "" {
		return nil
	}
	lower := core.Lower(text)
	issues := []string{}
	if core.Contains(text, "<|channel>") || core.Contains(text, "<channel|>") || core.Contains(text, "<turn|>") || core.Contains(text, "<|turn>") {
		issues = append(issues, "visible_chat_control_token")
	}
	if stateRampProfileFenceOnlyOutput(text) {
		issues = append(issues, "visible_fence_only")
	}
	if _, _, ok := stateRampProfileRepeatedTableCellOutput(text); ok {
		issues = append(issues, "visible_repeated_table_cell")
	}
	if _, _, ok := stateRampProfileRepeatedTableRowLabelOutput(text); ok {
		issues = append(issues, "visible_repeated_table_row_label")
	}
	if _, ok := stateRampProfileRepeatedShortLineCycleOutput(text); ok {
		issues = append(issues, "visible_repeated_short_line_cycle")
	}
	if core.HasPrefix(text, "```") {
		issues = append(issues, "visible_code_fence_prefix")
	}
	if core.Contains(lower, "the user is asking") ||
		core.Contains(lower, "the user's prompt") ||
		core.Contains(lower, "this request asks") ||
		core.Contains(lower, "this request is") ||
		core.Contains(lower, "the provided request is") ||
		core.Contains(lower, "the request is a directive") ||
		core.Contains(lower, "the previous turn material") ||
		core.Contains(lower, "the core objective is to") ||
		core.Contains(lower, "the analysis must focus on") ||
		core.Contains(lower, "the analysis must specifically address") ||
		core.Contains(lower, "the output should function as") ||
		core.Contains(lower, "based on the retained context") ||
		core.Contains(lower, "the instruction is to") ||
		core.Contains(lower, "this is an engineering session") ||
		core.Contains(lower, "the core instruction is to") ||
		core.Contains(lower, "seed prompt to preserve") ||
		core.Contains(lower, "constraint checklist") ||
		core.Contains(lower, "execution plan") {
		issues = append(issues, "visible_prompt_analysis")
	}
	if core.Contains(lower, "self-correction") || core.Contains(lower, "self correction") || core.Contains(lower, "i need to act as if") {
		issues = append(issues, "visible_self_correction")
	}
	if core.Contains(text, "**Plan:**") || core.Contains(text, "Plan:\n") || core.Contains(text, "**Plan**") {
		issues = append(issues, "visible_plan_scaffold")
	}
	trimmedLower := core.Trim(core.TrimSuffix(lower, "."))
	if trimmedLower == "ready" {
		issues = append(issues, "visible_seed_ready_echo")
	}
	if core.Contains(lower, "i don't have the actual results") || core.Contains(lower, "i do not have the actual results") {
		issues = append(issues, "visible_missing_results_admission")
	}
	if core.Contains(lower, "officially complete") ||
		core.Contains(lower, "officially accepted") ||
		core.Contains(lower, "officially validated") ||
		core.Contains(lower, "is production-ready") ||
		core.Contains(lower, "now production-ready") ||
		core.Contains(lower, "deemed production-ready") ||
		core.Contains(lower, "the implementation is now officially") ||
		core.Contains(lower, "superior production candidate") ||
		core.Contains(lower, "superior production-ready runner") ||
		core.Contains(lower, "achieved a significant milestone") ||
		core.Contains(lower, "confirms successful implementation") ||
		core.Contains(lower, "validates the entire implementation path") {
		issues = append(issues, "visible_false_completion_claim")
	}
	if core.Contains(lower, "production runner wins") ||
		core.Contains(lower, "go-mlx surpasses llama.cpp") ||
		core.Contains(lower, "go-mlx surpasses mlx_lm") ||
		core.Contains(lower, "go-mlx surpasses vllm") ||
		core.Contains(lower, "go-mlx outperforms llama.cpp") ||
		core.Contains(lower, "go-mlx outperforms mlx_lm") ||
		core.Contains(lower, "go-mlx outperforms vllm") ||
		core.Contains(lower, "performance advantage over llama.cpp") ||
		core.Contains(lower, "performance advantage over mlx_lm") ||
		core.Contains(lower, "performance advantage over vllm") ||
		core.Contains(lower, "demonstrates superior performance") ||
		core.Contains(lower, "achieves superior performance") ||
		core.Contains(lower, "established itself as the leading") ||
		core.Contains(lower, "superior performance to llama.cpp") ||
		core.Contains(lower, "superior performance to mlx_lm") ||
		core.Contains(lower, "superior performance to vllm") {
		issues = append(issues, "visible_unproven_performance_win_claim")
	}
	return issues
}

func stateRampProfileRepeatedTableCellOutput(text string) (string, int, bool) {
	if !core.Contains(text, "|") {
		return "", 0, false
	}
	counts := map[string]int{}
	for _, raw := range core.Split(text, "|") {
		cell := core.Lower(core.Trim(raw))
		if cell == "" || len(cell) > 16 || stateRampProfileTableSeparatorCell(cell) {
			continue
		}
		counts[cell]++
		if counts[cell] >= profileRepeatedTableCellLoopLimit {
			return cell, counts[cell], true
		}
	}
	return "", 0, false
}

func stateRampProfileRepeatedTableRowLabelOutput(text string) (string, int, bool) {
	if !core.Contains(text, "|") {
		return "", 0, false
	}
	counts := map[string]int{}
	for _, line := range core.Split(text, "\n") {
		line = core.Trim(line)
		if !core.HasPrefix(line, "|") {
			continue
		}
		cells := core.Split(line, "|")
		if len(cells) < 3 {
			continue
		}
		label := normaliseStateRampTableRowLabel(cells[1])
		if label == "" || len(label) > 32 || stateRampProfileTableSeparatorCell(label) {
			continue
		}
		counts[label]++
		if counts[label] >= profileRepeatedTableRowLabelLoopLimit {
			return label, counts[label], true
		}
	}
	return "", 0, false
}

func normaliseStateRampTableRowLabel(label string) string {
	label = core.Trim(core.Lower(label))
	for core.HasPrefix(label, "**") {
		label = core.Trim(core.TrimPrefix(label, "**"))
	}
	for core.HasSuffix(label, "**") {
		label = core.Trim(core.TrimSuffix(label, "**"))
	}
	return label
}

func stateRampProfileRepeatedShortLineCycleOutput(text string) (int, bool) {
	run := 0
	var symbols [4]string
	symbolCount := 0
	for start := 0; start <= len(text); {
		end := start
		for end < len(text) && text[end] != '\n' {
			end++
		}
		line := core.Trim(text[start:end])
		if !stateRampProfileShortCycleLine(line) {
			run = 0
			symbols = [4]string{}
			symbolCount = 0
			if end >= len(text) {
				break
			}
			start = end + 1
			continue
		}
		found := false
		for i := 0; i < symbolCount; i++ {
			if symbols[i] == line {
				found = true
				break
			}
		}
		if !found {
			if symbolCount == len(symbols) {
				run = 0
				symbols = [4]string{}
				symbolCount = 0
			}
			symbols[symbolCount] = line
			symbolCount++
		}
		run++
		if run >= profileRepeatedShortLineCycleLimit {
			return run, true
		}
		if end >= len(text) {
			break
		}
		start = end + 1
	}
	return 0, false
}

func stateRampProfileShortCycleLine(line string) bool {
	if line == "" || len(line) > 4 {
		return false
	}
	for _, r := range line {
		if r > 127 {
			return false
		}
		if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') {
			return false
		}
		switch r {
		case '"', '\'', '`', '(', ')', '[', ']', '{', '}', '<', '>', '.', ',', ';', ':', '-', '_', '*', '/', '\\', '|', '!', '?':
		default:
			return false
		}
	}
	return true
}

func stateRampProfileTableSeparatorCell(cell string) bool {
	if cell == "" {
		return false
	}
	for _, r := range cell {
		switch r {
		case '-', ':', ' ':
		default:
			return false
		}
	}
	return true
}

func stateRampProfileFenceOnlyOutput(text string) bool {
	sawFence := false
	for _, r := range text {
		switch r {
		case '`':
			sawFence = true
		case ' ', '\n', '\r', '\t':
		default:
			return false
		}
	}
	return sawFence
}

func stateRampProfileAssistantCloseSuffix(template string) string {
	if stateRampProfilePlainTemplate(template) {
		return ""
	}
	return chapterProfileAssistantHistorySuffix(template, "")
}

func stateRampProfileAppendSources(tok *mlx.Tokenizer, text, delimiter, template string, enableThinking bool, minVisibleTokens int, turnPromptMode string) ([]int32, [][]int32, error) {
	if tok == nil {
		return nil, nil, core.NewError("state-ramp-profile: model tokenizer is nil")
	}
	delimiter = core.Trim(delimiter)
	if delimiter == "" {
		tokens, err := tok.Encode(text)
		if err != nil {
			return nil, nil, err
		}
		if len(tokens) == 0 {
			return nil, nil, core.NewError("state-ramp-profile: append prompt produced no tokens")
		}
		return tokens, nil, nil
	}
	sections := [][]int32{}
	for _, raw := range core.Split(text, delimiter) {
		section := core.Trim(raw)
		if section == "" {
			continue
		}
		if !stateRampProfilePlainTemplate(template) {
			section = stateRampProfileTurnPromptWithMode(template, section, enableThinking, turnPromptMode, minVisibleTokens)
		}
		tokens, err := tok.Encode(section)
		if err != nil {
			return nil, nil, err
		}
		if len(tokens) > 0 {
			sections = append(sections, tokens)
		}
	}
	if len(sections) == 0 {
		return nil, nil, core.NewError("state-ramp-profile: append turn delimiter produced no token sections")
	}
	return nil, sections, nil
}

func countStateRampAppendSourceTokens(tokens []int32, sections [][]int32) int {
	if len(sections) == 0 {
		return len(tokens)
	}
	total := 0
	for _, section := range sections {
		total += len(section)
	}
	return total
}

func stateRampProfileTurnAppendSource(source []int32, sections [][]int32, sourceOffset, currentTokens, turnIndex int, opts stateRampProfileOptions) ([]int32, int, int) {
	tokens := source
	appendCount := opts.AppendTokens
	if len(sections) > 0 {
		tokens = sections[(turnIndex-1)%len(sections)]
		appendCount = len(tokens)
		sourceOffset = 0
	} else if limit := stateRampProfileLiveTokenLimit(opts); limit > 0 {
		if remaining := limit - currentTokens; remaining < appendCount {
			appendCount = remaining
		}
	}
	if appendCount < 0 {
		appendCount = 0
	}
	if sourceOffset < 0 {
		sourceOffset = 0
	}
	return tokens, sourceOffset, appendCount
}

func stateRampProfileAppendRepeatedTokens(ctx context.Context, session *mlx.ModelSession, sourceTokens []int32, sourceOffset, appendCount int) (int, error) {
	if session == nil {
		return 0, core.NewError("state-ramp-profile: session is nil")
	}
	return forEachRepeatedStateRampTokenSpan(sourceTokens, sourceOffset, appendCount, func(tokens []int32) error {
		if len(tokens) == 0 {
			return nil
		}
		return session.AppendTokens(ctx, tokens)
	})
}

func stateRampProfileGenerateTurn(ctx context.Context, model *mlx.Model, session *mlx.ModelSession, sourceTokens []int32, sourceOffset, appendCount, currentTokens, index int, opts stateRampProfileOptions) stateRampProfileTurn {
	turn := stateRampProfileTurn{
		Index:              index,
		TokensBeforeAppend: currentTokens,
	}
	if appendCount > 0 {
		appendStart := time.Now()
		appended, err := stateRampProfileAppendRepeatedTokens(ctx, session, sourceTokens, sourceOffset, appendCount)
		turn.AppendDuration = bench.NonZeroDuration(time.Since(appendStart))
		turn.AppendedTokens = appended
		if err != nil {
			turn.Error = err.Error()
			return turn
		}
	}
	turn.TokensAfterAppend = currentTokens + turn.AppendedTokens
	start := time.Now()
	firstToken := time.Duration(0)
	builder := core.NewBuilder()
	generateOptions := []mlx.GenerateOption{
		mlx.WithMaxTokens(opts.TurnMaxTokens),
		mlx.WithTemperature(float32(opts.Temperature)),
		mlx.WithTopP(float32(opts.TopP)),
		mlx.WithTopK(opts.TopK),
		mlx.WithRepeatPenalty(float32(opts.RepeatPenalty)),
	}
	if opts.SeedSet {
		generateOptions = append(generateOptions, mlx.WithSeed(opts.Seed))
	}
	if opts.TraceTokenPhases {
		generateOptions = append(generateOptions, mlx.WithTokenPhaseTrace())
	}
	stopTokenIDs, suppressTokenIDs := chapterProfileTemplateTokenControls(opts.ChatTemplate, model.Tokenizer())
	suppressTokenIDs = stateRampProfileEffectiveSuppressTokenIDs(suppressTokenIDs, stopTokenIDs, model.Tokenizer(), opts.SuppressEOS)
	if len(stopTokenIDs) > 0 {
		generateOptions = append(generateOptions, mlx.WithStopTokens(stopTokenIDs...))
	}
	if len(stopTokenIDs) > 0 && !opts.SuppressEOS {
		generateOptions = append(generateOptions, mlx.WithMinTokensBeforeStop(1))
	}
	if len(suppressTokenIDs) > 0 {
		generateOptions = append(generateOptions, mlx.WithSuppressTokens(suppressTokenIDs...))
	}
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
	for token := range session.GenerateStream(generationCtx, generateOptions...) {
		if draining {
			continue
		}
		if firstToken == 0 {
			firstToken = bench.NonZeroDuration(time.Since(start))
		}
		turn.VisibleTokens++
		if len(sampledTokenIDs) < 32 {
			sampledTokenIDs = append(sampledTokenIDs, token.ID)
			sampledTokenTexts = append(sampledTokenTexts, token.Text)
		}
		builder.WriteString(token.Text)
		if probeErr == nil {
			if err := driverProfileMetricsSafetyError(core.Sprintf("state-ramp-profile turn %d stream", index), profileLiveMetrics(), opts.SafetyLimits); err != nil {
				probeErr = err
				cancelGeneration()
				draining = true
				continue
			}
			if opts.SafetyLimits.RepeatedTokenLoopLimit <= 0 {
				repeatedTokenCount = 0
			} else if repeatedTokenCount == 0 || token.ID != repeatedTokenID {
				repeatedTokenID = token.ID
				repeatedTokenCount = 1
			} else {
				repeatedTokenCount++
				if repeatedTokenCount >= opts.SafetyLimits.RepeatedTokenLoopLimit {
					probeErr = core.NewError(core.Sprintf("state-ramp-profile: turn %d sampled token %d for %d consecutive tokens", index, token.ID, repeatedTokenCount))
					cancelGeneration()
					draining = true
					continue
				}
			}
		}
		if lineErr == nil {
			if line, count, ok := profileObserveRepeatedLineFragment(token.Text, &currentLine, &lastLine, &repeatedLineCount, opts.SafetyLimits.RepeatedLineLoopLimit); ok {
				lineErr = core.NewError(core.Sprintf("state-ramp-profile: turn %d repeated visible line %q for %d consecutive lines", index, line, count))
				cancelGeneration()
				draining = true
				continue
			}
		}
	}
	if lineErr == nil {
		if line, count, ok := profileFlushRepeatedLine(&currentLine, &lastLine, &repeatedLineCount, opts.SafetyLimits.RepeatedLineLoopLimit); ok {
			lineErr = core.NewError(core.Sprintf("state-ramp-profile: turn %d repeated visible line %q for %d consecutive lines", index, line, count))
		}
	}
	turn.Duration = bench.NonZeroDuration(time.Since(start))
	turn.FirstTokenDuration = firstToken
	turn.StreamDuration = turn.Duration
	if firstToken > 0 && turn.Duration > firstToken {
		turn.StreamDuration = turn.Duration - firstToken
	}
	turn.SampledTokenIDs = sampledTokenIDs
	turn.SampledTokenTexts = sampledTokenTexts
	turn.Metrics = model.Metrics()
	if opts.TraceTokenPhases {
		if phaseIDs, phaseTexts := stateRampProfileSampledTokensFromPhases(turn.Metrics.TokenPhases, 32); len(phaseIDs) > 0 {
			turn.SampledTokenIDs = phaseIDs
			if len(phaseTexts) > 0 {
				turn.SampledTokenTexts = phaseTexts
			}
		}
	}
	turn.DriverOverheadDuration = driverRunOverhead(turn.Duration, turn.Metrics)
	turn.TokensAfterGenerate = turn.Metrics.PromptTokens + turn.Metrics.GeneratedTokens
	visibleOutput := stateRampProfileVisibleOutput(opts.ChatTemplate, builder.String())
	turn.OutputIssues = stateRampProfileOutputIssues(visibleOutput)
	if opts.IncludeOutput {
		turn.Output = visibleOutput
	}
	if turn.VisibleTokens == 0 {
		turn.OutputIssues = append(turn.OutputIssues, "empty_visible_output")
		turn.Error = core.Sprintf("state-ramp-profile: turn %d produced no visible output", index)
		return turn
	}
	if probeErr != nil {
		turn.Error = probeErr.Error()
		return turn
	}
	if lineErr != nil {
		turn.Error = lineErr.Error()
		return turn
	}
	if err := session.Err(); err != nil {
		turn.Error = err.Error()
		return turn
	}
	if err := driverProfileMetricsSafetyError(core.Sprintf("state-ramp-profile turn %d", index), turn.Metrics, opts.SafetyLimits); err != nil {
		turn.Error = err.Error()
		return turn
	}
	if err := driverProfileRunSafetyError(index, driverProfileRun{
		Index:             index,
		VisibleTokens:     turn.VisibleTokens,
		SampledTokenIDs:   turn.SampledTokenIDs,
		SampledTokenTexts: turn.SampledTokenTexts,
		Output:            visibleOutput,
		Metrics:           turn.Metrics,
	}, opts.SafetyLimits); err != nil {
		turn.Error = err.Error()
		return turn
	}
	if suffix := stateRampProfileAssistantCloseSuffix(opts.ChatTemplate); suffix != "" {
		closeStart := time.Now()
		if err := chapterProfileAppendPrompt(ctx, model, session, suffix); err != nil {
			turn.Error = err.Error()
			return turn
		}
		turn.AppendDuration += bench.NonZeroDuration(time.Since(closeStart))
		if tok := model.Tokenizer(); tok != nil {
			if tokens, err := tok.Encode(suffix); err == nil {
				turn.TurnCloseTokens = len(tokens)
				turn.TokensAfterGenerate += len(tokens)
			}
		}
	}
	stateRampProfileApplyVisibleTokenFloor(&turn, opts)
	if turn.Error != "" {
		return turn
	}
	if ctx != nil {
		if err := ctx.Err(); err != nil {
			turn.Error = err.Error()
		}
	}
	return turn
}

func stateRampProfileSampledTokensFromPhases(phases []mlx.TokenPhaseTrace, limit int) ([]int32, []string) {
	if limit <= 0 || len(phases) == 0 {
		return nil, nil
	}
	count := min(limit, len(phases))
	ids := make([]int32, 0, count)
	texts := make([]string, 0, count)
	hasText := false
	for i := range count {
		ids = append(ids, phases[i].TokenID)
		if phases[i].TokenText != "" {
			hasText = true
		}
		texts = append(texts, phases[i].TokenText)
	}
	if !hasText {
		return ids, nil
	}
	return ids, texts
}

func stateRampProfileApplyVisibleTokenFloor(turn *stateRampProfileTurn, opts stateRampProfileOptions) {
	if turn == nil || opts.TurnMinTokens <= 0 || turn.VisibleTokens >= opts.TurnMinTokens {
		return
	}
	turn.BelowMinTokens = true
	issue := core.Sprintf("below_debug_visible_token_floor:%d/%d", turn.VisibleTokens, opts.TurnMinTokens)
	turn.OutputIssues = append(turn.OutputIssues, issue)
	if opts.TurnMinTokensPolicy == "fail" {
		turn.Error = core.Sprintf("state-ramp-profile: turn %d produced %d visible tokens, below requested visible-token debug floor %d", turn.Index, turn.VisibleTokens, opts.TurnMinTokens)
	}
}

func stateRampProfileTurnErrorFatal(turn stateRampProfileTurn, opts stateRampProfileOptions) bool {
	if turn.Error == "" {
		return false
	}
	return !(turn.BelowMinTokens && opts.TurnMinTokensPolicy == "mark")
}

func stateRampProfileTurnHasContentIssue(turn stateRampProfileTurn) bool {
	for _, issue := range turn.OutputIssues {
		if core.HasPrefix(issue, "below_debug_visible_token_floor:") {
			continue
		}
		return true
	}
	return false
}

func stateRampProfileDegradationFoldReached(consecutiveContentIssues int, opts stateRampProfileOptions) bool {
	if !opts.FoldOnDegradation {
		return false
	}
	minConsecutive := opts.DegradationMinConsecutive
	if minConsecutive <= 0 {
		minConsecutive = 2
	}
	return consecutiveContentIssues >= minConsecutive
}

func summariseStateRampProfileTurns(initialPrefill time.Duration, initialTokens int, turns []stateRampProfileTurn, opts stateRampProfileOptions) stateRampProfileSummary {
	summary := stateRampProfileSummary{
		InitialPrefillTokens: initialTokens,
		FinalStateTokens:     initialTokens,
		TotalDuration:        initialPrefill,
	}
	if initialPrefill > 0 && initialTokens > 0 {
		summary.InitialPrefillTokensPerSec = float64(initialTokens) / initialPrefill.Seconds()
	}
	var decodeDuration time.Duration
	var turnWallDuration time.Duration
	var replayDecodeDuration time.Duration
	var mtpRateSamples int
	var mtpRestoreSamples int
	var tokenPhaseIndex map[string]int
	var nativeEventIndex map[string]int
	var nativeEventDetailIndex map[string]int
	traceAggregationInitialised := false
	for _, turn := range turns {
		turnFatal := stateRampProfileTurnErrorFatal(turn, opts)
		if turnFatal {
			summary.FailedTurns++
		} else {
			summary.SuccessfulTurns++
			if turn.Metrics.PrefillDuration > 0 {
				summary.ReplayEstimateTurns++
				summary.ReplayPrefillDuration += turn.Metrics.PrefillDuration
				replayDecodeDuration += turn.Duration
			}
		}
		summary.AppendedTokens += turn.AppendedTokens
		summary.GeneratedTokens += turn.Metrics.GeneratedTokens
		summary.VisibleTokens += turn.VisibleTokens
		summary.TotalDuration += turn.AppendDuration + turn.Duration
		summary.AppendDuration += turn.AppendDuration
		turnWallDuration += turn.AppendDuration + turn.Duration
		decodeDuration += turn.Metrics.DecodeDuration
		if turn.TokensAfterGenerate > summary.FinalStateTokens {
			summary.FinalStateTokens = turn.TokensAfterGenerate
		} else if turn.TokensAfterAppend > summary.FinalStateTokens {
			summary.FinalStateTokens = turn.TokensAfterAppend
		}
		if turn.Metrics.PeakMemoryBytes > summary.PeakMemoryBytes {
			summary.PeakMemoryBytes = turn.Metrics.PeakMemoryBytes
		}
		if turn.Metrics.ActiveMemoryBytes > summary.ActiveMemoryBytes {
			summary.ActiveMemoryBytes = turn.Metrics.ActiveMemoryBytes
		}
		if turn.Metrics.CacheMemoryBytes > summary.CacheMemoryBytes {
			summary.CacheMemoryBytes = turn.Metrics.CacheMemoryBytes
		}
		if activePlusCache := turn.Metrics.ActiveMemoryBytes + turn.Metrics.CacheMemoryBytes; activePlusCache > summary.ActivePlusCacheMemoryBytes {
			summary.ActivePlusCacheMemoryBytes = activePlusCache
		}
		if turn.Metrics.ProcessVirtualMemoryBytes > summary.ProcessVirtualMemoryBytes {
			summary.ProcessVirtualMemoryBytes = turn.Metrics.ProcessVirtualMemoryBytes
		}
		if turn.Metrics.ProcessResidentMemoryBytes > summary.ProcessResidentMemoryBytes {
			summary.ProcessResidentMemoryBytes = turn.Metrics.ProcessResidentMemoryBytes
		}
		if turn.Metrics.ProcessPeakResidentBytes > summary.ProcessPeakResidentBytes {
			summary.ProcessPeakResidentBytes = turn.Metrics.ProcessPeakResidentBytes
		}
		if mtp := turn.Metrics.MTP; mtp != nil {
			summary.MTPProposedTokens += mtp.ProposedTokens
			summary.MTPAcceptedTokens += mtp.AcceptedTokens
			summary.MTPRejectedTokens += mtp.RejectedTokens
			summary.MTPTargetVerifyCalls += mtp.TargetVerifyCalls
			summary.MTPTargetCalls += mtp.TargetCalls
			summary.MTPDraftCalls += mtp.DraftCalls
			summary.MTPDraftTokenSchedule = append(summary.MTPDraftTokenSchedule, mtp.DraftTokenSchedule...)
			summary.MTPWallDuration += mtp.WallDuration
			summary.MTPTargetVerifyDuration += mtp.TargetVerifyDuration
			summary.MTPTargetDuration += mtp.TargetDuration
			summary.MTPDraftDuration += mtp.DraftDuration
			if mtp.RestoreDuration > 0 {
				summary.MTPRestoreAvgDuration += mtp.RestoreDuration
				mtpRestoreSamples++
			}
			if mtp.VisibleTokensPerSec > 0 || mtp.TargetTokensPerSec > 0 || mtp.WarmDecodeTokensPerSec > 0 {
				summary.MTPVisibleTokensPerSecAverage += mtp.VisibleTokensPerSec
				summary.MTPTargetTokensPerSecAverage += mtp.TargetTokensPerSec
				summary.MTPWarmDecodeTokensPerSecAverage += mtp.WarmDecodeTokensPerSec
				mtpRateSamples++
			}
			if mtp.PeakMemoryBytes > summary.MTPPeakMemoryBytes {
				summary.MTPPeakMemoryBytes = mtp.PeakMemoryBytes
			}
		}
		if len(turn.OutputIssues) > 0 {
			summary.OutputIssueTurns++
			if summary.OutputIssueCounts == nil {
				summary.OutputIssueCounts = map[string]int{}
			}
			for _, issue := range turn.OutputIssues {
				summary.OutputIssueCounts[issue]++
			}
		}
		if len(turn.Metrics.TokenPhases) > 0 && !traceAggregationInitialised {
			traceAggregationInitialised = true
			summary.TokenPhases = make([]driverProfileNativeEventSummary, 0, 8)
			summary.NativeEvents = make([]driverProfileNativeEventSummary, 0, 4)
			summary.NativeEventDetails = make([]driverProfileNativeEventSummary, 0, 8)
		}
		for _, phase := range turn.Metrics.TokenPhases {
			accumulateStateRampProfileTokenPhase(&summary, tokenPhaseIndex, "total", phase.TotalDuration)
			accumulateStateRampProfileTokenPhase(&summary, tokenPhaseIndex, "forward", phase.ForwardDuration)
			accumulateStateRampProfileTokenPhase(&summary, tokenPhaseIndex, "sample_eval", phase.SampleEvalDuration)
			accumulateStateRampProfileTokenPhase(&summary, tokenPhaseIndex, "sample", phase.SampleDuration)
			accumulateStateRampProfileTokenPhase(&summary, tokenPhaseIndex, "logits", phase.LogitsDuration)
			accumulateStateRampProfileTokenPhase(&summary, tokenPhaseIndex, "token_read", phase.TokenReadDuration)
			accumulateStateRampProfileTokenPhase(&summary, tokenPhaseIndex, "decode_text", phase.DecodeTextDuration)
			accumulateStateRampProfileTokenPhase(&summary, tokenPhaseIndex, "probe_token", phase.ProbeTokenDuration)
			accumulateStateRampProfileTokenPhase(&summary, tokenPhaseIndex, "yield", phase.YieldDuration)
			accumulateStateRampProfileTokenPhase(&summary, tokenPhaseIndex, "next_input", phase.NextInputDuration)
			accumulateStateRampProfileTokenPhase(&summary, tokenPhaseIndex, "materialize", phase.MaterializeDuration)
			accumulateStateRampProfileTokenPhase(&summary, tokenPhaseIndex, "prefetch", phase.PrefetchDuration)
			accumulateStateRampProfileTokenPhase(&summary, tokenPhaseIndex, "prefetch_logits", phase.PrefetchLogitsDuration)
			accumulateStateRampProfileTokenPhase(&summary, tokenPhaseIndex, "prefetch_cache", phase.PrefetchCacheDuration)
			accumulateStateRampProfileTokenPhase(&summary, tokenPhaseIndex, "detach", phase.DetachDuration)
			accumulateStateRampProfileTokenPhase(&summary, tokenPhaseIndex, "cache_probe", phase.CacheProbeDuration)
			accumulateStateRampProfileTokenPhase(&summary, tokenPhaseIndex, "other", phase.OtherDuration)
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
	if len(turns) > 0 {
		summary.AppendAvgDuration = summary.AppendDuration / time.Duration(len(turns))
	}
	summary.RetainedSetupDuration = initialPrefill + summary.AppendDuration
	if summary.ReplayEstimateTurns > 0 {
		summary.ReplayTotalDuration = summary.ReplayPrefillDuration + replayDecodeDuration
		if summary.ReplayPrefillDuration > summary.RetainedSetupDuration {
			summary.ReplayPrefillSavedDuration = summary.ReplayPrefillDuration - summary.RetainedSetupDuration
		}
		if summary.ReplayTotalDuration > summary.TotalDuration {
			summary.ReplayTotalSavedDuration = summary.ReplayTotalDuration - summary.TotalDuration
		}
		if summary.TotalDuration > 0 && summary.ReplayTotalDuration > 0 {
			summary.RetainedVsReplaySpeedup = float64(summary.ReplayTotalDuration) / float64(summary.TotalDuration)
		}
	}
	if summary.AppendDuration > 0 && summary.AppendedTokens > 0 {
		summary.AppendTokensPerSecAverage = float64(summary.AppendedTokens) / summary.AppendDuration.Seconds()
	}
	if decodeDuration > 0 && summary.GeneratedTokens > 0 {
		summary.DecodeTokensPerSecAverage = float64(summary.GeneratedTokens) / decodeDuration.Seconds()
	}
	if summary.MTPProposedTokens > 0 {
		summary.MTPAcceptanceRateAverage = float64(summary.MTPAcceptedTokens) / float64(summary.MTPProposedTokens)
	}
	if mtpRateSamples > 0 {
		sampleCount := float64(mtpRateSamples)
		summary.MTPVisibleTokensPerSecAverage /= sampleCount
		summary.MTPTargetTokensPerSecAverage /= sampleCount
		summary.MTPWarmDecodeTokensPerSecAverage /= sampleCount
	}
	if mtpRestoreSamples > 0 {
		summary.MTPRestoreAvgDuration /= time.Duration(mtpRestoreSamples)
	}
	summary.DecodeBandwidthProxy = estimateDecodeBandwidthProxy(
		summary.DecodeTokensPerSecAverage,
		summary.ActivePlusCacheMemoryBytes,
	)
	if turnWallDuration > 0 && summary.GeneratedTokens > 0 {
		summary.EffectiveTurnTokensPerSec = float64(summary.GeneratedTokens) / turnWallDuration.Seconds()
	}
	for i := range summary.TokenPhases {
		if summary.TokenPhases[i].Count > 0 {
			summary.TokenPhases[i].AverageDuration = summary.TokenPhases[i].Duration / time.Duration(summary.TokenPhases[i].Count)
		}
	}
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
	sort.SliceStable(summary.TokenPhases, func(i, j int) bool {
		return summary.TokenPhases[i].Duration > summary.TokenPhases[j].Duration
	})
	sort.SliceStable(summary.NativeEvents, func(i, j int) bool {
		return summary.NativeEvents[i].Duration > summary.NativeEvents[j].Duration
	})
	sort.SliceStable(summary.NativeEventDetails, func(i, j int) bool {
		return summary.NativeEventDetails[i].Duration > summary.NativeEventDetails[j].Duration
	})
	annotateStateRampProfileContentDegradation(&summary, turns, opts)
	annotateStateRampProfileContextLifecycle(&summary, opts)
	return summary
}

func accumulateStateRampProfileTokenPhase(summary *stateRampProfileSummary, index map[string]int, name string, duration time.Duration) {
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

func annotateStateRampProfileContentDegradation(summary *stateRampProfileSummary, turns []stateRampProfileTurn, opts stateRampProfileOptions) {
	if summary == nil || !opts.FoldOnDegradation {
		return
	}
	minConsecutive := opts.DegradationMinConsecutive
	if minConsecutive <= 0 {
		minConsecutive = 2
	}
	streak := 0
	for _, turn := range turns {
		if stateRampProfileTurnHasContentIssue(turn) {
			streak++
		} else {
			streak = 0
		}
		if streak < minConsecutive {
			continue
		}
		summary.ContentDegraded = true
		summary.ContentDegradationTurn = turn.Index
		summary.ContentDegradationStreak = streak
		summary.ContentDegradationReason = core.Sprintf(
			"retained context produced %d consecutive output-issue turns at turn %d; checkpoint, summarise, and prefill a folded state before appending more turns",
			streak,
			turn.Index,
		)
		summary.FoldedStateRequired = true
		if summary.CompactionReason == "" {
			summary.CompactionReason = summary.ContentDegradationReason
		}
		return
	}
}

func annotateStateRampProfileContextLifecycle(summary *stateRampProfileSummary, opts stateRampProfileOptions) {
	if summary == nil {
		return
	}
	threshold := opts.CompactionThresholdTokens
	if threshold <= 0 {
		return
	}
	summary.CompactionThresholdTokens = threshold
	summary.CompactionTailTokens = opts.CompactionTailTokens
	if summary.FinalStateTokens < threshold {
		return
	}
	summary.ContextExhausted = true
	summary.FoldedStateRequired = true
	summary.CompactionReason = "live state reached the compaction threshold; checkpoint, summarise, and prefill a folded state from durable summary plus recent tail before appending more turns"
}

func stateRampProfileShouldRunFold(summary stateRampProfileSummary, opts stateRampProfileOptions) bool {
	if !summary.FoldedStateRequired {
		return false
	}
	if opts.FoldOnDegradation {
		return true
	}
	return summary.ContextExhausted && core.Trim(opts.FoldStorePath) != ""
}

func stateRampProfileFoldExhausted(ctx context.Context, model *mlx.Model, session *mlx.ModelSession, report *stateRampProfileReport, opts stateRampProfileOptions) *stateRampProfileFold {
	fold := &stateRampProfileFold{
		StorePath:           opts.FoldStorePath,
		SummaryMode:         stateRampProfileFoldSummaryMode(opts),
		SummaryBytes:        len(opts.FoldSummary),
		SummaryPromptBytes:  len(opts.FoldSummaryPrompt),
		SummaryMaxTokens:    opts.FoldSummaryMaxTokens,
		RecentTailBytes:     len(opts.FoldRecentTail),
		ContinuePromptBytes: len(opts.FoldContinuePrompt),
	}
	if report == nil || !report.Summary.FoldedStateRequired {
		fold.SkippedReason = "live state did not reach the compaction threshold or content-degradation boundary"
		return fold
	}
	fold.Attempted = true
	if model == nil || session == nil {
		fold.Error = "state-ramp-profile: folded-state handoff requires a live model session"
		return fold
	}
	if core.Trim(opts.FoldStorePath) == "" {
		fold.Error = "state-ramp-profile: fold store path is required"
		return fold
	}
	store, action, err := stateRampProfileOpenFoldStore(ctx, opts.FoldStorePath)
	if err != nil {
		fold.Error = err.Error()
		return fold
	}
	fold.StoreAction = action
	defer store.Close()

	summary := stateRampProfileFoldSummary(report, opts)
	tail := stateRampProfileFoldRecentTail(report, opts)
	start := time.Now()
	if opts.FoldSummaryGenerate {
		generatedSummary, summaryTurn, err := stateRampProfileGenerateFoldSummary(ctx, model, session, report, opts)
		if summaryTurn != nil {
			fold.SummaryGeneration = summaryTurn
		}
		if err != nil {
			fold.Duration = bench.NonZeroDuration(time.Since(start))
			fold.Error = err.Error()
			return fold
		}
		if core.Trim(generatedSummary) != "" {
			summary = generatedSummary
		}
		mlx.ClearCache()
	}
	fold.SummaryBytes = len(summary)
	fold.RecentTailBytes = len(tail)
	foldPrompt := stateRampProfileInitialPrompt(opts.ChatTemplate, stateRampProfileFoldBody(summary, tail), opts.EnableThinking)
	fold.FoldedPromptBytes = len(foldPrompt)
	baseURI := stateRampProfileFoldBaseURI()
	folded, foldReport, err := model.FoldAgentMemory(ctx, session, store, mlx.AgentMemoryFoldOptions{
		Summary:           summary,
		RecentTail:        tail,
		FoldedPrompt:      foldPrompt,
		PrefillChunkBytes: opts.FoldPrefillChunkBytes,
		Checkpoint:        stateRampProfileFoldSleepOptions(report, baseURI, "checkpoint"),
		Folded:            stateRampProfileFoldSleepOptions(report, baseURI, "folded"),
	})
	fold.Duration = bench.NonZeroDuration(time.Since(start))
	if foldReport != nil {
		fold.Checkpoint = foldReport.Checkpoint
		fold.Folded = foldReport.Folded
		fold.SummaryBytes = foldReport.SummaryBytes
		fold.RecentTailBytes = foldReport.RecentTailBytes
		fold.FoldedPromptBytes = foldReport.FoldedPromptBytes
	}
	fold.CompactMarker = stateRampProfileFoldMarker(opts.FoldStorePath, fold.Folded)
	if err != nil {
		fold.Error = err.Error()
		return fold
	}
	if folded != nil {
		defer folded.Close()
	}
	if opts.FoldContinueMaxTokens <= 0 {
		return fold
	}
	if fold.Folded == nil || fold.Folded.IndexURI == "" {
		fold.Error = "state-ramp-profile: folded-state wake index is missing"
		return fold
	}
	wakeStart := time.Now()
	woken, wake, err := model.WakeAgentMemory(ctx, store, agent.WakeOptions{
		IndexURI: fold.Folded.IndexURI,
	})
	fold.WakeDuration = bench.NonZeroDuration(time.Since(wakeStart))
	fold.Wake = wake
	if err != nil {
		fold.Error = err.Error()
		return fold
	}
	defer woken.Close()
	continueTurn, err := stateRampProfileContinueFromFold(ctx, model, woken, fold, opts)
	fold.ContinueTurn = continueTurn
	if err != nil {
		fold.Error = err.Error()
	}
	return fold
}

func stateRampProfileOpenFoldStore(ctx context.Context, path string) (*statefile.Store, string, error) {
	if stat := core.Stat(path); stat.OK {
		store, err := statefile.Open(ctx, path)
		return store, "append", err
	} else if !core.IsNotExist(stat.Value.(error)) {
		return nil, "", stat.Value.(error)
	}
	store, err := statefile.Create(ctx, path)
	return store, "create", err
}

func stateRampProfileFoldMarker(storePath string, report *agent.SleepReport) *stateRampFoldMarker {
	if report == nil || report.IndexURI == "" {
		return nil
	}
	return &stateRampFoldMarker{
		StorePath:  storePath,
		IndexURI:   report.IndexURI,
		EntryURI:   report.EntryURI,
		BundleURI:  report.BundleURI,
		TokenCount: report.TokenCount,
	}
}

func stateRampProfileContinueFromFold(ctx context.Context, model *mlx.Model, session *mlx.ModelSession, fold *stateRampProfileFold, opts stateRampProfileOptions) (*stateRampProfileTurn, error) {
	if fold == nil || fold.Folded == nil {
		return nil, core.NewError("state-ramp-profile: folded state is missing")
	}
	prompt := stateRampProfileTurnPrompt(opts.ChatTemplate, opts.FoldContinuePrompt, opts.EnableThinking)
	tok := model.Tokenizer()
	if tok == nil {
		return nil, core.NewError("state-ramp-profile: model tokenizer is nil")
	}
	tokens, err := tok.Encode(prompt)
	if err != nil {
		return nil, err
	}
	continueOpts := opts
	continueOpts.TurnMaxTokens = opts.FoldContinueMaxTokens
	continueOpts.TurnMinTokens = 0
	continueOpts.TurnMinTokensPolicy = "mark"
	turn := stateRampProfileGenerateTurn(ctx, model, session, tokens, 0, len(tokens), fold.Folded.TokenCount, 1, continueOpts)
	if turn.Error != "" {
		return &turn, core.NewError(turn.Error)
	}
	return &turn, nil
}

func stateRampProfileGenerateFoldSummary(ctx context.Context, model *mlx.Model, session *mlx.ModelSession, report *stateRampProfileReport, opts stateRampProfileOptions) (string, *stateRampProfileTurn, error) {
	if model == nil || session == nil {
		return "", nil, core.NewError("state-ramp-profile: folded summary generation requires a live model session")
	}
	tok := model.Tokenizer()
	if tok == nil {
		return "", nil, core.NewError("state-ramp-profile: model tokenizer is nil")
	}
	prompt := stateRampProfileTurnPrompt(opts.ChatTemplate, opts.FoldSummaryPrompt, opts.EnableThinking, 0)
	tokens, err := tok.Encode(prompt)
	if err != nil {
		return "", nil, err
	}
	if len(tokens) == 0 {
		return "", nil, core.NewError("state-ramp-profile: fold summary prompt produced no tokens")
	}
	summaryOpts := opts
	summaryOpts.TurnMaxTokens = opts.FoldSummaryMaxTokens
	summaryOpts.TurnMinTokens = 0
	summaryOpts.TurnMinTokensPolicy = "mark"
	summaryOpts.IncludeOutput = true
	currentTokens := 0
	turnIndex := 1
	if report != nil {
		currentTokens = report.Summary.FinalStateTokens
		turnIndex = max(report.Summary.SuccessfulTurns+report.Summary.FailedTurns+1, 1)
	}
	turn := stateRampProfileGenerateTurn(ctx, model, session, tokens, 0, len(tokens), currentTokens, turnIndex, summaryOpts)
	summary := core.Trim(turn.Output)
	if !opts.IncludeOutput {
		turn.Output = ""
	}
	if err := stateRampProfileGeneratedSummaryError(turn, summary); err != nil {
		return summary, &turn, err
	}
	return summary, &turn, nil
}

func stateRampProfileGeneratedSummaryError(turn stateRampProfileTurn, summary string) error {
	if turn.Error != "" {
		return core.NewError(turn.Error)
	}
	if core.Trim(summary) == "" {
		return core.NewError("state-ramp-profile: generated folded summary was empty")
	}
	if stateRampProfileTurnHasContentIssue(turn) {
		return core.NewError(core.Sprintf("state-ramp-profile: generated folded summary has output issues: %s", core.Join(", ", turn.OutputIssues...)))
	}
	return nil
}

func stateRampProfileFoldSummaryMode(opts stateRampProfileOptions) string {
	if opts.FoldSummaryGenerate {
		return "generated"
	}
	if core.Trim(opts.FoldSummary) != "" {
		return "provided"
	}
	return "lifecycle"
}

func stateRampProfileFoldSummary(report *stateRampProfileReport, opts stateRampProfileOptions) string {
	if summary := core.Trim(opts.FoldSummary); summary != "" {
		return summary
	}
	if report == nil {
		return "The previous retained state reached a compaction boundary and was compacted into a folded state."
	}
	if report.Summary.ContentDegraded {
		return core.Sprintf(
			"The previous retained state degraded at %d tokens after turn %d, with %d consecutive output-issue turns. The run appended %d tokens, generated %d tokens, and recorded %.3f raw decode tokens per second with %.3f effective turn tokens per second. Continue from this compacted memory rather than replaying the degraded prefix.",
			report.Summary.FinalStateTokens,
			report.Summary.ContentDegradationTurn,
			report.Summary.ContentDegradationStreak,
			report.Summary.AppendedTokens,
			report.Summary.GeneratedTokens,
			report.Summary.DecodeTokensPerSecAverage,
			report.Summary.EffectiveTurnTokensPerSec,
		)
	}
	return core.Sprintf(
		"The previous retained state reached the live-token budget at %d tokens after %d successful turns. The run appended %d tokens, generated %d tokens, and recorded %.3f raw decode tokens per second with %.3f effective turn tokens per second. Continue from this compacted memory rather than replaying the exhausted prefix.",
		report.Summary.FinalStateTokens,
		report.Summary.SuccessfulTurns,
		report.Summary.AppendedTokens,
		report.Summary.GeneratedTokens,
		report.Summary.DecodeTokensPerSecAverage,
		report.Summary.EffectiveTurnTokensPerSec,
	)
}

func stateRampProfileFoldRecentTail(report *stateRampProfileReport, opts stateRampProfileOptions) string {
	if tail := core.Trim(opts.FoldRecentTail); tail != "" {
		return tail
	}
	if report == nil || len(report.Turns) == 0 {
		return ""
	}
	builder := core.NewBuilder()
	start := max(len(report.Turns)-3, 0)
	for i := start; i < len(report.Turns); i++ {
		turn := report.Turns[i]
		if core.Trim(turn.Output) == "" {
			continue
		}
		builder.WriteString(core.Sprintf("Turn %d output:\n", turn.Index))
		builder.WriteString(core.Trim(turn.Output))
		builder.WriteString("\n\n")
	}
	return core.Trim(builder.String())
}

func stateRampProfileFoldBody(summary, tail string) string {
	builder := core.NewBuilder()
	builder.WriteString("The previous retained context window has been compacted into this folded state.\n\n")
	if core.Trim(summary) != "" {
		builder.WriteString("<summary>\n")
		builder.WriteString(core.Trim(summary))
		builder.WriteString("\n</summary>\n\n")
	}
	if core.Trim(tail) != "" {
		builder.WriteString("<recent_tail>\n")
		builder.WriteString(core.Trim(tail))
		builder.WriteString("\n</recent_tail>\n\n")
	}
	builder.WriteString("Use the summary as durable memory and the recent tail as the immediate continuation point. Do not assume the full exhausted context is still present.")
	return builder.String()
}

func stateRampProfileFoldBaseURI() string {
	return core.Sprintf("mlx://state-ramp/fold/%d", time.Now().UTC().UnixNano())
}

func stateRampProfileFoldSleepOptions(report *stateRampProfileReport, baseURI, kind string) agent.SleepOptions {
	if core.Trim(baseURI) == "" {
		baseURI = stateRampProfileFoldBaseURI()
	}
	kind = core.Trim(kind)
	if kind == "" {
		kind = "state"
	}
	uri := baseURI + "/" + kind
	meta := map[string]string{
		"source": "state-ramp-profile",
		"kind":   kind,
	}
	if report != nil {
		meta["start_tokens"] = core.Itoa(report.StartTokens)
		meta["target_tokens"] = core.Itoa(report.TargetTokens)
		meta["final_state_tokens"] = core.Itoa(report.Summary.FinalStateTokens)
	}
	return agent.SleepOptions{
		EntryURI:  uri,
		BundleURI: uri + "/bundle",
		IndexURI:  uri + "/index",
		Title:     "state ramp " + kind,
		ModelPath: reportModelPath(report),
		Labels:    []string{"state-ramp-profile", kind},
		Meta:      meta,
	}
}

func reportModelPath(report *stateRampProfileReport) string {
	if report == nil {
		return ""
	}
	return report.ModelPath
}

func estimateStateRampProfileEnergy(report *stateRampProfileReport, powerWatts float64) *stateRampProfileEnergy {
	energy := &stateRampProfileEnergy{
		Method:     "estimated_wall_clock_seconds_times_average_active_watts",
		PowerWatts: powerWatts,
	}
	if report == nil || powerWatts <= 0 {
		return energy
	}
	energy.TotalJoules = durationJoules(report.Summary.TotalDuration, powerWatts)
	energy.AppendJoules = durationJoules(report.Summary.AppendDuration, powerWatts)
	if report.Summary.ReplayTotalDuration > 0 {
		energy.ReplayTotalJoules = durationJoules(report.Summary.ReplayTotalDuration, powerWatts)
	}
	if report.Summary.ReplayTotalSavedDuration > 0 {
		energy.RetainedVsReplaySavedJoules = durationJoules(report.Summary.ReplayTotalSavedDuration, powerWatts)
	}
	if report.Summary.VisibleTokens > 0 {
		energy.JoulesPerVisibleToken = energy.TotalJoules / float64(report.Summary.VisibleTokens)
	}
	if foldDuration := stateRampProfileFoldDuration(report.Fold); foldDuration > 0 {
		energy.FoldLifecycleJoules = durationJoules(foldDuration, powerWatts)
		energy.TotalWithFoldLifecycleJoules = energy.TotalJoules + energy.FoldLifecycleJoules
	}
	if report.Fold != nil && report.Fold.ContinueTurn != nil {
		turn := report.Fold.ContinueTurn
		turnWall := report.Fold.WakeDuration + turn.AppendDuration + turn.Duration
		if turn.VisibleTokens > 0 && turnWall > 0 {
			energy.FoldContinueJoulesPerToken = durationJoules(turnWall, powerWatts) / float64(turn.VisibleTokens)
			energy.FoldContinueEffectiveTokensSec = float64(turn.VisibleTokens) / turnWall.Seconds()
		}
	}
	return energy
}

func stateRampProfileFoldDuration(fold *stateRampProfileFold) time.Duration {
	if fold == nil {
		return 0
	}
	total := fold.Duration + fold.WakeDuration
	if fold.ContinueTurn != nil {
		total += fold.ContinueTurn.AppendDuration + fold.ContinueTurn.Duration
	}
	return total
}

func annotateStateRampProfileFoldDurations(report *stateRampProfileReport) {
	if report == nil || report.Fold == nil {
		return
	}
	report.Fold.LifecycleDuration = stateRampProfileFoldDuration(report.Fold)
	if report.Fold.LifecycleDuration > 0 && report.Summary.TotalDuration > 0 {
		report.Fold.TotalWithRetained = report.Summary.TotalDuration + report.Fold.LifecycleDuration
	}
}

func printStateRampProfileSummary(stdout io.Writer, report *stateRampProfileReport) {
	if report == nil {
		return
	}
	core.WriteString(stdout, core.Sprintf("state ramp profile: %s\n", report.ModelPath))
	core.WriteString(stdout, core.Sprintf("  seed: %d tokens in %s, final state: %d tokens\n", report.InitialPrefillTokens, report.InitialPrefillDuration, report.Summary.FinalStateTokens))
	core.WriteString(stdout, core.Sprintf("  turns: %d ok / %d failed, appended: %d tokens at %.1f tok/s\n", report.Summary.SuccessfulTurns, report.Summary.FailedTurns, report.Summary.AppendedTokens, report.Summary.AppendTokensPerSecAverage))
	core.WriteString(stdout, core.Sprintf("  generated: %d tokens, decode: %.1f tok/s, effective turn: %.1f tok/s, total: %s\n", report.Summary.GeneratedTokens, report.Summary.DecodeTokensPerSecAverage, report.Summary.EffectiveTurnTokensPerSec, report.Summary.TotalDuration))
	if report.Summary.ReplayTotalDuration > 0 {
		core.WriteString(stdout, core.Sprintf(
			"  replay estimate: %s one-shot wall, saved %s, speedup %.2fx\n",
			report.Summary.ReplayTotalDuration,
			report.Summary.ReplayTotalSavedDuration,
			report.Summary.RetainedVsReplaySpeedup,
		))
	}
	core.WriteString(stdout, core.Sprintf("  peak memory: %d MB, active+cache: %d MB, process virtual: %d MB, process resident: %d MB\n",
		report.Summary.PeakMemoryBytes/1024/1024,
		report.Summary.ActivePlusCacheMemoryBytes/1024/1024,
		report.Summary.ProcessVirtualMemoryBytes/1024/1024,
		report.Summary.ProcessResidentMemoryBytes/1024/1024,
	))
	if proxy := report.Summary.DecodeBandwidthProxy; proxy != nil {
		core.WriteString(stdout, core.Sprintf("  bandwidth proxy: %.3f GB/token active+cache -> %.1f GB/s implied\n", proxy.ActivePlusCacheGBPerDecodeTokenProxy, proxy.ImpliedActivePlusCacheBandwidthGBPerSecProxy))
	}
	if report.EstimatedEnergy != nil {
		core.WriteString(stdout, core.Sprintf("  estimated energy: %.1f J at %.1f W\n", report.EstimatedEnergy.TotalJoules, report.EstimatedEnergy.PowerWatts))
	}
	if report.Summary.ContentDegraded {
		core.WriteString(stdout, core.Sprintf("  content degraded: folded state required after %d consecutive output-issue turns at turn %d\n", report.Summary.ContentDegradationStreak, report.Summary.ContentDegradationTurn))
	}
	if report.Summary.ContextExhausted {
		core.WriteString(stdout, core.Sprintf("  context exhausted: folded state required at %d tokens (tail hint: %d tokens)\n", report.Summary.CompactionThresholdTokens, report.Summary.CompactionTailTokens))
	} else if report.Summary.FoldedStateRequired && report.Summary.CompactionReason != "" {
		core.WriteString(stdout, core.Sprintf("  folded state required: %s\n", report.Summary.CompactionReason))
	}
	if report.Fold != nil {
		if report.Fold.Attempted {
			core.WriteString(stdout, core.Sprintf("  folded state: %s in %s", report.Fold.StorePath, report.Fold.Duration))
			if report.Fold.WakeDuration > 0 {
				core.WriteString(stdout, core.Sprintf(", wake %s", report.Fold.WakeDuration))
			}
			if report.Fold.ContinueTurn != nil {
				core.WriteString(stdout, core.Sprintf(", continue %d tokens in %s at %.1f tok/s", report.Fold.ContinueTurn.VisibleTokens, report.Fold.ContinueTurn.Duration, report.Fold.ContinueTurn.Metrics.DecodeTokensPerSec))
			}
			if report.Fold.LifecycleDuration > 0 {
				core.WriteString(stdout, core.Sprintf(", fold lifecycle %s", report.Fold.LifecycleDuration))
			}
			if report.Fold.StoreAction != "" {
				core.WriteString(stdout, core.Sprintf(", store %s", report.Fold.StoreAction))
			}
			if report.Fold.CompactMarker != nil && report.Fold.CompactMarker.IndexURI != "" {
				core.WriteString(stdout, core.Sprintf(", compact marker %s", report.Fold.CompactMarker.IndexURI))
			}
			core.WriteString(stdout, "\n")
		} else if report.Fold.SkippedReason != "" {
			core.WriteString(stdout, core.Sprintf("  folded state: skipped (%s)\n", report.Fold.SkippedReason))
		}
	}
}

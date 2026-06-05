// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"context"
	"flag"
	"io"
	"runtime"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference/bench"
	statefile "dappco.re/go/inference/state/filestore"
	mlx "dappco.re/go/mlx"
	"dappco.re/go/mlx/agent"
	"dappco.re/go/mlx/pkg/metal"
)

func runStateWakeProfileCommand(ctx context.Context, args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet(cliCommandName("state-wake-profile"), flag.ContinueOnError)
	fs.SetOutput(stderr)
	jsonOut := fs.Bool("json", false, "print JSON State wake profile")
	reportFile := fs.String("report-file", "", "write JSON State wake profile to a file")
	markerFile := fs.String("marker-file", "", "read State compact marker from a state-ramp-profile report or marker JSON")
	stateStorePath := fs.String("state-store", "", "existing append-only State file to open")
	indexURI := fs.String("index-uri", "", "State index URI to wake")
	prompt := fs.String("prompt", defaultStateRampFoldContinuePrompt, "prompt appended after waking the selected State")
	promptFile := fs.String("prompt-file", "", "read wake prompt text from a file")
	chatTemplate := fs.String("chat-template", "", "chat template override for the wake prompt: gemma4, gemma, qwen, llama, or plain")
	enableThinking := fs.Bool("enable-thinking", false, "enable Gemma 4 thinking control token in the wake prompt")
	maxTokens := fs.Int("max-tokens", 512, "generated tokens for the wake/continue check")
	temperature := fs.Float64("temperature", 1.0, "sampling temperature for the wake turn")
	topP := fs.Float64("top-p", 0.95, "top-p sampling value for the wake turn")
	topK := fs.Int("top-k", 64, "top-k sampling value for the wake turn")
	repeatPenalty := fs.Float64("repeat-penalty", 1.0, "repeat penalty for the wake turn")
	suppressEOS := fs.Bool("suppress-eos", false, "suppress the tokenizer EOS token during the wake turn")
	includeOutput := fs.Bool("include-output", true, "include generated text in the report")
	contextLen := fs.Int("context", 0, "override context length")
	prefillChunkSize := fs.Int("prefill-chunk-size", 0, "override long-prompt prefill chunk size in tokens")
	cacheMode := fs.String("cache-mode", "", cacheModeFlagUsage)
	device := fs.String("device", "", "execution device: gpu or cpu")
	estimatePowerWatts := fs.Float64("estimate-power-watts", 0, "record an estimated average active power draw in watts")
	fastGemma4Lane := fs.Bool("fast-gemma4-lane", true, "enable the accepted Gemma 4 fast runtime gates by default; set false for baseline diagnostics")
	maxActiveMemoryBytes := fs.Uint64("max-active-memory-bytes", 0, "abort if MLX active memory exceeds this many bytes; 0 derives from the resolved memory limit")
	maxProcessVirtualMemoryBytes := fs.Uint64("max-process-virtual-memory-bytes", 0, "abort if process virtual memory exceeds this many bytes; 0 records process virtual memory without a hard cap")
	maxProcessResidentMemoryBytes := fs.Uint64("max-process-resident-memory-bytes", 0, "abort if process resident memory exceeds this many bytes; 0 derives from the resolved memory limit")
	repeatedTokenLoopLimit := fs.Int("repeated-token-loop-limit", driverProfileDefaultRepeatedTokenLoopLimit, "abort when this many consecutive sampled tokens have the same token id")
	repeatedLineLoopLimit := fs.Int("repeated-line-loop-limit", profileDefaultRepeatedLineLoopLimit, "abort when this many consecutive visible non-empty lines repeat")
	repeatedSentenceLoopLimit := fs.Int("repeated-sentence-loop-limit", profileDefaultRepeatedSentenceLoopLimit, "abort when the same visible sentence repeats this many times in one output")
	fs.Usage = func() {
		name := cliName()
		core.WriteString(stderr, core.Sprintf("Usage: %s state-wake-profile [flags] [model-path]\n", name))
		core.WriteString(stderr, "\n")
		core.WriteString(stderr, "Wake an existing State index (the persisted KV-cache snapshot of a\n")
		core.WriteString(stderr, "prior session) and measure one continuation turn — restore latency,\n")
		core.WriteString(stderr, "first-token-after-wake, decode throughput. Pairs with state-pack\n")
		core.WriteString(stderr, "(which writes the State container being woken).\n")
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
		core.WriteString(stderr, core.Sprintf("  %s state-wake-profile -state-index ~/sessions/session-1/folded ~/models/lemer-lite\n", name))
		core.WriteString(stderr, core.Sprintf("    # wake the folded State + run a continuation turn\n"))
		core.WriteString(stderr, core.Sprintf("  %s state-wake-profile -json -trace-token-phases -state-index ~/sessions/s1/folded ~/models/lemer-lite\n", name))
		core.WriteString(stderr, core.Sprintf("    # JSON + per-token phase trace of the wake + continuation\n"))
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
		core.WriteString(stderr, core.Sprintf("%s state-wake-profile: expected one model path\n", cliName()))
		fs.Usage()
		return 2
	}
	var markerCleanup func()
	stateStoreSegmentAlias := ""
	stateStorePayloadOffset := int64(0)
	stateStorePayloadBytes := int64(0)
	if core.Trim(*markerFile) != "" {
		markerSource, err := stateWakeProfileMarkerSourceFromFile(*markerFile)
		if err != nil {
			core.Print(stderr, "%s state-wake-profile: marker file: %v", cliName(), err)
			return 1
		}
		if markerSource.Cleanup != nil {
			markerCleanup = markerSource.Cleanup
			defer markerCleanup()
		}
		if core.Trim(*stateStorePath) == "" {
			*stateStorePath = markerSource.Marker.StorePath
		}
		if core.Trim(*indexURI) == "" {
			*indexURI = markerSource.Marker.IndexURI
		}
		stateStoreSegmentAlias = markerSource.SegmentAlias
		stateStorePayloadOffset = markerSource.PayloadOffset
		stateStorePayloadBytes = markerSource.PayloadBytes
	}
	if core.Trim(*stateStorePath) == "" {
		core.WriteString(stderr, core.Sprintf("%s state-wake-profile: state store path is required\n", cliName()))
		return 2
	}
	if core.Trim(*indexURI) == "" {
		core.WriteString(stderr, core.Sprintf("%s state-wake-profile: index URI is required\n", cliName()))
		return 2
	}
	if core.Trim(*promptFile) != "" {
		read := core.ReadFile(*promptFile)
		if !read.OK {
			core.Print(stderr, "%s state-wake-profile: prompt file: %v", cliName(), read.Value)
			return 1
		}
		*prompt = string(read.Value.([]byte))
	}
	if *maxTokens < 1 {
		core.WriteString(stderr, core.Sprintf("%s state-wake-profile: max tokens must be >= 1\n", cliName()))
		return 2
	}
	if *prefillChunkSize < 0 {
		core.WriteString(stderr, core.Sprintf("%s state-wake-profile: prefill chunk size must be >= 0\n", cliName()))
		return 2
	}
	if *estimatePowerWatts < 0 {
		core.WriteString(stderr, core.Sprintf("%s state-wake-profile: estimated power watts must be >= 0\n", cliName()))
		return 2
	}
	if *temperature < 0 {
		core.WriteString(stderr, core.Sprintf("%s state-wake-profile: temperature must be >= 0\n", cliName()))
		return 2
	}
	if *topP < 0 {
		core.WriteString(stderr, core.Sprintf("%s state-wake-profile: top-p must be >= 0\n", cliName()))
		return 2
	}
	if *topK < 0 {
		core.WriteString(stderr, core.Sprintf("%s state-wake-profile: top-k must be >= 0\n", cliName()))
		return 2
	}
	if *repeatPenalty < 0 {
		core.WriteString(stderr, core.Sprintf("%s state-wake-profile: repeat penalty must be >= 0\n", cliName()))
		return 2
	}
	if *repeatedTokenLoopLimit < 1 {
		core.WriteString(stderr, core.Sprintf("%s state-wake-profile: repeated token loop limit must be >= 1\n", cliName()))
		return 2
	}
	if *repeatedLineLoopLimit < 1 {
		core.WriteString(stderr, core.Sprintf("%s state-wake-profile: repeated line loop limit must be >= 1\n", cliName()))
		return 2
	}
	if *repeatedSentenceLoopLimit < 1 {
		core.WriteString(stderr, core.Sprintf("%s state-wake-profile: repeated sentence loop limit must be >= 1\n", cliName()))
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
			core.WriteString(stderr, core.Sprintf("%s state-wake-profile: unsupported cache mode %q\n", cliName(), string(mode)))
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

	report, err := runStateWakeProfileGuarded(ctx, fs.Arg(0), loadOptions, stateWakeProfileOptions{
		StateStorePath:          core.Trim(*stateStorePath),
		StateStoreSegmentAlias:  core.Trim(stateStoreSegmentAlias),
		StateStorePayloadOffset: stateStorePayloadOffset,
		StateStorePayloadBytes:  stateStorePayloadBytes,
		IndexURI:                core.Trim(*indexURI),
		Prompt:                  *prompt,
		ChatTemplate:            *chatTemplate,
		EnableThinking:          *enableThinking,
		MaxTokens:               *maxTokens,
		Temperature:             *temperature,
		TopP:                    *topP,
		TopK:                    *topK,
		RepeatPenalty:           *repeatPenalty,
		SuppressEOS:             *suppressEOS,
		IncludeOutput:           *includeOutput,
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
	if report != nil && *estimatePowerWatts > 0 {
		report.EstimatedEnergy = estimateStateWakeProfileEnergy(report, *estimatePowerWatts)
	}
	reportPath := core.Trim(*reportFile)
	if *jsonOut || reportPath != "" {
		if report == nil {
			report = &stateWakeProfileReport{
				Version:                 1,
				ModelPath:               fs.Arg(0),
				StateStorePath:          core.Trim(*stateStorePath),
				StateStoreAlias:         core.Trim(stateStoreSegmentAlias),
				StateStorePayloadOffset: stateStorePayloadOffset,
				StateStorePayloadBytes:  stateStorePayloadBytes,
				IndexURI:                core.Trim(*indexURI),
				PromptBytes:             len(*prompt),
				ChatTemplate:            *chatTemplate,
				EnableThinking:          *enableThinking,
				MaxTokens:               *maxTokens,
				Temperature:             *temperature,
				TopP:                    *topP,
				TopK:                    *topK,
				RepeatPenalty:           *repeatPenalty,
				SuppressEOS:             *suppressEOS,
				IncludeOutput:           *includeOutput,
			}
		}
		if err != nil && report.Error == "" {
			report.Error = err.Error()
		}
		data := core.JSONMarshalIndent(report, "", "  ")
		if !data.OK {
			core.Print(stderr, "%s state-wake-profile: marshal report failed", cliName())
			return 1
		}
		if reportPath != "" {
			if writeErr := writeJSONReportFile(reportPath, data.Value.([]byte)); writeErr != nil {
				core.Print(stderr, "%s state-wake-profile: write report file: %v", cliName(), writeErr)
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
		core.Print(stderr, "%s state-wake-profile: %v", cliName(), err)
		return 1
	}
	printStateWakeProfileSummary(stdout, report)
	return 0
}

type stateWakeProfileMarkerFile struct {
	StorePath string                      `json:"store_path,omitempty"`
	IndexURI  string                      `json:"index_uri,omitempty"`
	EntryURI  string                      `json:"entry_uri,omitempty"`
	BundleURI string                      `json:"bundle_uri,omitempty"`
	Fold      *stateWakeProfileMarkerFold `json:"fold,omitempty"`
}

type stateWakeProfileMarkerFold struct {
	StorePath     string               `json:"store_path,omitempty"`
	CompactMarker *stateRampFoldMarker `json:"compact_marker,omitempty"`
	Folded        *agent.SleepReport   `json:"folded,omitempty"`
}

func stateWakeProfileCompactMarkerFromFile(path string) (stateRampFoldMarker, error) {
	read := core.ReadFile(path)
	if !read.OK {
		return stateRampFoldMarker{}, read.Value.(error)
	}
	var payload stateWakeProfileMarkerFile
	if result := core.JSONUnmarshal(read.Value.([]byte), &payload); !result.OK {
		return stateRampFoldMarker{}, result.Value.(error)
	}
	if marker := stateWakeProfileCompactMarkerFromPayload(payload); marker.IndexURI != "" {
		return marker, nil
	}
	return stateRampFoldMarker{}, core.NewError("State compact marker missing store_path or index_uri")
}

func stateWakeProfileCompactMarkerFromPayload(payload stateWakeProfileMarkerFile) stateRampFoldMarker {
	if payload.IndexURI != "" {
		return stateRampFoldMarker{
			StorePath: payload.StorePath,
			IndexURI:  payload.IndexURI,
			EntryURI:  payload.EntryURI,
			BundleURI: payload.BundleURI,
		}
	}
	if payload.Fold == nil {
		return stateRampFoldMarker{}
	}
	if marker := payload.Fold.CompactMarker; marker != nil && marker.IndexURI != "" {
		return *marker
	}
	if payload.Fold.Folded == nil || payload.Fold.Folded.IndexURI == "" {
		return stateRampFoldMarker{}
	}
	return stateRampFoldMarker{
		StorePath:  payload.Fold.StorePath,
		IndexURI:   payload.Fold.Folded.IndexURI,
		EntryURI:   payload.Fold.Folded.EntryURI,
		BundleURI:  payload.Fold.Folded.BundleURI,
		TokenCount: payload.Fold.Folded.TokenCount,
	}
}

var runStateWakeProfile = defaultRunStateWakeProfile

func runStateWakeProfileGuarded(ctx context.Context, modelPath string, loadOptions []mlx.LoadOption, opts stateWakeProfileOptions) (report *stateWakeProfileReport, err error) {
	defer func() {
		if recovered := recover(); recovered != nil {
			err = core.NewError(core.Sprintf("state-wake-profile panic: %v", recovered))
		}
	}()
	return runStateWakeProfile(ctx, modelPath, loadOptions, opts)
}

func defaultRunStateWakeProfile(ctx context.Context, modelPath string, loadOptions []mlx.LoadOption, opts stateWakeProfileOptions) (*stateWakeProfileReport, error) {
	opts = normalizeStateWakeProfileOptions(opts)
	report := &stateWakeProfileReport{
		Version:                 1,
		ModelPath:               modelPath,
		StateStorePath:          opts.StateStorePath,
		StateStoreAlias:         opts.StateStoreSegmentAlias,
		StateStorePayloadOffset: opts.StateStorePayloadOffset,
		StateStorePayloadBytes:  opts.StateStorePayloadBytes,
		IndexURI:                opts.IndexURI,
		PromptBytes:             len(opts.Prompt),
		EnableThinking:          opts.EnableThinking,
		MaxTokens:               opts.MaxTokens,
		Temperature:             opts.Temperature,
		TopP:                    opts.TopP,
		TopK:                    opts.TopK,
		RepeatPenalty:           opts.RepeatPenalty,
		SuppressEOS:             opts.SuppressEOS,
		IncludeOutput:           opts.IncludeOutput,
		SafetyLimits:            opts.SafetyLimits,
		RuntimeGates:            driverProfileRuntimeGates(),
	}
	loadStart := time.Now()
	model, err := loadBenchModel(modelPath, loadOptions...)
	report.LoadDuration = bench.NonZeroDuration(time.Since(loadStart))
	if err != nil {
		report.Error = err.Error()
		return report, err
	}
	if model == nil {
		err := core.NewError("mlx: state wake profile loaded nil model")
		report.Error = err.Error()
		return report, err
	}
	report.Load = mergeDriverProfileLoadSettings(report.Load, loadSettingsFromModelInfo(model.Info()))
	opts.SafetyLimits = resolveDriverProfileSafetyLimits(opts.SafetyLimits, report.Load)
	report.SafetyLimits = opts.SafetyLimits
	defer model.Close()
	if err := driverProfileMetricsSafetyError("load", model.Metrics(), opts.SafetyLimits); err != nil {
		report.Error = err.Error()
		return report, err
	}
	opts.ChatTemplate = chapterProfileTemplate(opts.ChatTemplate, model.Info().Architecture)
	report.ChatTemplate = opts.ChatTemplate
	tok := model.Tokenizer()
	if tok == nil {
		err := core.NewError("state-wake-profile: model tokenizer is nil")
		report.Error = err.Error()
		return report, err
	}

	openMemory := stateWakeMemoryNow()
	openStart := time.Now()
	var store *statefile.Store
	if opts.StateStorePayloadOffset > 0 || opts.StateStorePayloadBytes > 0 {
		store, err = statefile.OpenRegionWithSegmentAlias(ctx, opts.StateStorePath, opts.StateStorePayloadOffset, opts.StateStorePayloadBytes, opts.StateStoreSegmentAlias)
	} else if opts.StateStoreSegmentAlias != "" {
		store, err = statefile.OpenWithSegmentAlias(ctx, opts.StateStorePath, opts.StateStoreSegmentAlias)
	} else {
		store, err = statefile.Open(ctx, opts.StateStorePath)
	}
	report.StoreOpenDuration = bench.NonZeroDuration(time.Since(openStart))
	report.StoreOpenMemoryDelta = stateWakeMemoryDeltaBetween(openMemory, stateWakeMemoryNow())
	if err != nil {
		report.Error = err.Error()
		return report, err
	}
	defer store.Close()

	wakeMemory := stateWakeMemoryNow()
	wakeStart := time.Now()
	session, wake, err := model.WakeAgentMemory(ctx, store, agent.WakeOptions{IndexURI: opts.IndexURI})
	report.WakeDuration = bench.NonZeroDuration(time.Since(wakeStart))
	report.WakeMemoryDelta = stateWakeMemoryDeltaBetween(wakeMemory, stateWakeMemoryNow())
	report.Wake = wake
	if err != nil {
		report.Error = err.Error()
		return report, err
	}
	defer session.Close()
	if err := driverProfileMetricsSafetyError("wake", model.Metrics(), opts.SafetyLimits); err != nil {
		report.Error = err.Error()
		return report, err
	}

	prompt := stateRampProfileTurnPrompt(opts.ChatTemplate, opts.Prompt, opts.EnableThinking)
	tokens, err := tok.Encode(prompt)
	if err != nil {
		report.Error = err.Error()
		return report, err
	}
	if len(tokens) == 0 {
		err := core.NewError("state-wake-profile: wake prompt produced no tokens")
		report.Error = err.Error()
		return report, err
	}
	report.PromptTokens = len(tokens)
	currentTokens := 0
	if wake != nil {
		currentTokens = wake.PrefixTokens
	}
	turnOpts := stateRampProfileOptions{
		ChatTemplate:   opts.ChatTemplate,
		EnableThinking: opts.EnableThinking,
		TurnMaxTokens:  opts.MaxTokens,
		Temperature:    opts.Temperature,
		TopP:           opts.TopP,
		TopK:           opts.TopK,
		RepeatPenalty:  opts.RepeatPenalty,
		SuppressEOS:    opts.SuppressEOS,
		IncludeOutput:  opts.IncludeOutput,
		SafetyLimits:   opts.SafetyLimits,
	}
	turn := stateRampProfileGenerateTurn(ctx, model, session, tokens, 0, len(tokens), currentTokens, 1, turnOpts)
	report.Turn = &turn
	if turn.Error != "" {
		err := core.NewError(turn.Error)
		report.Error = err.Error()
		return report, err
	}
	return report, nil
}

func normalizeStateWakeProfileOptions(opts stateWakeProfileOptions) stateWakeProfileOptions {
	opts.StateStorePath = core.Trim(opts.StateStorePath)
	opts.IndexURI = core.Trim(opts.IndexURI)
	opts.Prompt = core.Trim(opts.Prompt)
	if opts.Prompt == "" {
		opts.Prompt = defaultStateRampFoldContinuePrompt
	}
	if opts.MaxTokens <= 0 {
		opts.MaxTokens = 512
	}
	if opts.Temperature < 0 {
		opts.Temperature = 0
	}
	if opts.TopP < 0 {
		opts.TopP = 0
	}
	if opts.TopK < 0 {
		opts.TopK = 0
	}
	if opts.RepeatPenalty < 0 {
		opts.RepeatPenalty = 0
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

func estimateStateWakeProfileEnergy(report *stateWakeProfileReport, powerWatts float64) *stateWakeProfileEnergy {
	energy := &stateWakeProfileEnergy{
		Method:     "estimated_wake_append_generate_seconds_times_average_active_watts",
		PowerWatts: powerWatts,
	}
	if report == nil || powerWatts <= 0 {
		return energy
	}
	if report.Turn != nil {
		turnWall := report.WakeDuration + report.Turn.AppendDuration + report.Turn.Duration
		energy.TotalJoules = durationJoules(turnWall, powerWatts)
		energy.AppendJoules = durationJoules(report.Turn.AppendDuration, powerWatts)
		energy.GenerationJoules = durationJoules(report.Turn.Duration, powerWatts)
		if report.Turn.VisibleTokens > 0 && turnWall > 0 {
			energy.JoulesPerVisibleToken = energy.TotalJoules / float64(report.Turn.VisibleTokens)
			energy.EffectiveTokensPerSec = float64(report.Turn.VisibleTokens) / turnWall.Seconds()
		}
		energy.DecodeTokensPerSec = report.Turn.Metrics.DecodeTokensPerSec
		energy.VisibleOutputIssueCount = len(report.Turn.OutputIssues)
	}
	energy.WakeJoules = durationJoules(report.WakeDuration, powerWatts)
	return energy
}

func stateWakeMemoryNow() stateWakeMemorySample {
	var stats runtime.MemStats
	runtime.ReadMemStats(&stats)
	process := metal.GetProcessMemory()
	return stateWakeMemorySample{
		goHeapAllocBytes:     stats.HeapAlloc,
		goHeapObjects:        stats.HeapObjects,
		goTotalAllocBytes:    stats.TotalAlloc,
		goMallocs:            stats.Mallocs,
		goFrees:              stats.Frees,
		activeMemoryBytes:    metal.GetActiveMemory(),
		cacheMemoryBytes:     metal.GetCacheMemory(),
		peakMemoryBytes:      metal.GetPeakMemory(),
		processVirtualBytes:  process.VirtualMemoryBytes,
		processResidentBytes: process.ResidentMemoryBytes,
		processPeakResident:  process.PeakResidentMemoryBytes,
	}
}

func stateWakeMemoryDeltaBetween(before, after stateWakeMemorySample) *stateWakeMemoryDelta {
	return &stateWakeMemoryDelta{
		GoHeapAllocDeltaBytes:         stateWakeSignedDelta(after.goHeapAllocBytes, before.goHeapAllocBytes),
		GoHeapObjectsDelta:            stateWakeSignedDelta(after.goHeapObjects, before.goHeapObjects),
		GoTotalAllocDeltaBytes:        stateWakeUnsignedDelta(after.goTotalAllocBytes, before.goTotalAllocBytes),
		GoMallocsDelta:                stateWakeUnsignedDelta(after.goMallocs, before.goMallocs),
		GoFreesDelta:                  stateWakeUnsignedDelta(after.goFrees, before.goFrees),
		ActiveMemoryDeltaBytes:        stateWakeSignedDelta(after.activeMemoryBytes, before.activeMemoryBytes),
		CacheMemoryDeltaBytes:         stateWakeSignedDelta(after.cacheMemoryBytes, before.cacheMemoryBytes),
		PeakMemoryDeltaBytes:          stateWakeSignedDelta(after.peakMemoryBytes, before.peakMemoryBytes),
		ProcessVirtualDeltaBytes:      stateWakeSignedDelta(after.processVirtualBytes, before.processVirtualBytes),
		ProcessResidentDeltaBytes:     stateWakeSignedDelta(after.processResidentBytes, before.processResidentBytes),
		ProcessPeakResidentDeltaBytes: stateWakeSignedDelta(after.processPeakResident, before.processPeakResident),
	}
}

func stateWakeUnsignedDelta(after, before uint64) uint64 {
	if after < before {
		return 0
	}
	return after - before
}

func stateWakeSignedDelta(after, before uint64) int64 {
	const maxInt64 = uint64(1<<63 - 1)
	if after >= before {
		delta := after - before
		if delta > maxInt64 {
			return int64(maxInt64)
		}
		return int64(delta)
	}
	delta := before - after
	if delta > maxInt64 {
		return -int64(maxInt64)
	}
	return -int64(delta)
}

func printStateWakeProfileSummary(stdout io.Writer, report *stateWakeProfileReport) {
	if report == nil {
		return
	}
	core.WriteString(stdout, core.Sprintf("state wake profile: %s\n", report.ModelPath))
	if report.Wake != nil {
		core.WriteString(stdout, core.Sprintf("  wake: %s, %d prefix tokens via %s\n", report.WakeDuration, report.Wake.PrefixTokens, report.Wake.RestoreStrategy))
	} else {
		core.WriteString(stdout, core.Sprintf("  wake: %s\n", report.WakeDuration))
	}
	if report.Turn != nil {
		core.WriteString(stdout, core.Sprintf("  generated: %d visible tokens, decode: %.1f tok/s, wall: %s\n", report.Turn.VisibleTokens, report.Turn.Metrics.DecodeTokensPerSec, report.Turn.AppendDuration+report.Turn.Duration))
		if len(report.Turn.OutputIssues) > 0 {
			core.WriteString(stdout, core.Sprintf("  output issues: %s\n", core.Join(", ", report.Turn.OutputIssues...)))
		}
		core.WriteString(stdout, core.Sprintf("  peak memory: %d MB, active+cache: %d MB, process resident: %d MB\n",
			report.Turn.Metrics.PeakMemoryBytes/1024/1024,
			(report.Turn.Metrics.ActiveMemoryBytes+report.Turn.Metrics.CacheMemoryBytes)/1024/1024,
			report.Turn.Metrics.ProcessResidentMemoryBytes/1024/1024,
		))
	}
	if report.EstimatedEnergy != nil {
		core.WriteString(stdout, core.Sprintf("  estimated energy: %.1f J at %.1f W\n", report.EstimatedEnergy.TotalJoules, report.EstimatedEnergy.PowerWatts))
	}
}

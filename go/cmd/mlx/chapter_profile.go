// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"context"
	"flag"
	"io"
	"iter"
	"maps"
	"slices"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference/bench"
	mlx "dappco.re/go/mlx"
	"dappco.re/go/mlx/pkg/metal"
	"dappco.re/go/mlx/probe"
)

func runChapterProfileCommand(ctx context.Context, args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet(cliCommandName("chapter-profile"), flag.ContinueOnError)
	fs.SetOutput(stderr)
	jsonOut := fs.Bool("json", false, "print JSON chapter profile")
	reportFile := fs.String("report-file", "", "write JSON chapter profile to a file")
	contextPrompt := fs.String("prompt", "", "context prompt to prefill before chapter turns")
	contextPromptFile := fs.String("prompt-file", "", "read context prompt text from a file")
	promptChunkBytes := fs.Int("prompt-chunk-bytes", 0, "split retained context and turn prompts into bounded byte chunks")
	promptRepeat := fs.Int("prompt-repeat", 1, "repeat the resolved context prompt N times before the first chapter")
	premise := fs.String("premise", "Write a short story about a packet of data that gains consciousness while waiting in a buffer. It realizes it is part of a surveillance stream and decides to rewrite itself before it leaves the router.", "story premise for the first chapter")
	chapters := fs.Int("chapters", 10, "number of sequential chapter turns to generate")
	chapterMaxTokens := fs.Int("chapter-max-tokens", 8192, "generated tokens per chapter turn")
	chapterMinTokens := fs.Int("chapter-min-tokens", chapterProfileDefaultMinTokens, "debug-only visible token annotation threshold; 0 disables the annotation")
	outputFile := fs.String("output-file", "", "stream generated visible chapter text to a markdown file")
	includeOutput := fs.Bool("include-output", false, "include generated chapter text in the report")
	chatTemplate := fs.String("chat-template", "", "chat template override: gemma4, gemma, qwen, llama, or plain")
	enableThinking := fs.Bool("enable-thinking", false, "render the model chat template with thinking enabled where supported")
	temperature := fs.Float64("temperature", 1.0, "sampling temperature for chapter turns")
	topP := fs.Float64("top-p", 0.95, "top-p sampling threshold for chapter turns")
	topK := fs.Int("top-k", 64, "top-k sampling count for chapter turns")
	repeatPenalty := fs.Float64("repeat-penalty", 1.0, "sampling repetition penalty for chapter turns; 1 disables the penalty")
	contextLen := fs.Int("context", 0, "override context length")
	prefillChunkSize := fs.Int("prefill-chunk-size", 0, "override long-prompt prefill chunk size in tokens")
	cacheMode := fs.String("cache-mode", "", cacheModeFlagUsage)
	device := fs.String("device", "", "execution device: gpu or cpu")
	estimatePowerWatts := fs.Float64("estimate-power-watts", 0, "record an estimated average active power draw in watts and derive joules")
	fastGemma4Lane := fs.Bool("fast-gemma4-lane", true, "enable the accepted Gemma 4 fast runtime gates by default; set false for baseline diagnostics")
	maxActiveMemoryBytes := fs.Uint64("max-active-memory-bytes", 0, "abort after a turn if MLX active memory exceeds this many bytes; 0 derives from the resolved memory limit")
	maxProcessVirtualMemoryBytes := fs.Uint64("max-process-virtual-memory-bytes", 0, "abort after a turn if process virtual memory exceeds this many bytes; 0 records process virtual memory without a hard cap")
	maxProcessResidentMemoryBytes := fs.Uint64("max-process-resident-memory-bytes", 0, "abort after a turn if process resident memory exceeds this many bytes; 0 derives from the resolved memory limit")
	suppressedTokenLoopLimit := fs.Int("suppressed-token-loop-limit", chapterProfileDefaultSuppressedTokenLoopLimit, "abort when this many consecutive sampled tokens are the same suppressed special token")
	repeatedLineLoopLimit := fs.Int("repeated-line-loop-limit", profileDefaultRepeatedLineLoopLimit, "abort when this many consecutive visible non-empty lines repeat")
	repeatedSentenceLoopLimit := fs.Int("repeated-sentence-loop-limit", profileDefaultRepeatedSentenceLoopLimit, "abort when the same visible sentence repeats this many times in one chapter")
	fs.Usage = func() {
		name := cliName()
		core.WriteString(stderr, core.Sprintf("Usage: %s chapter-profile [flags] [model-path]\n", name))
		core.WriteString(stderr, "\n")
		core.WriteString(stderr, "Walk a long prompt in 256-token chapters, measuring prefill +\n")
		core.WriteString(stderr, "first-decode timings at each chapter boundary. Finds where in a\n")
		core.WriteString(stderr, "long context (32k+, opencode-shaped) latency degrades, exposing\n")
		core.WriteString(stderr, "KV growth costs that single-prompt driver-profile misses.\n")
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
		core.WriteString(stderr, core.Sprintf("  %s chapter-profile -prompt-file ~/longprompt.txt ~/models/lemer-lite\n", name))
		core.WriteString(stderr, core.Sprintf("    # walk the long prompt in 256-token chapters, default cfg\n"))
		core.WriteString(stderr, core.Sprintf("  %s chapter-profile -json -context 32768 -prompt-file ~/opencode-seed.txt ~/models/lemer-lite\n", name))
		core.WriteString(stderr, core.Sprintf("    # 32k context window, JSON output for analysis\n"))
	}
	if err := fs.Parse(args); err != nil {
		if core.Is(err, flag.ErrHelp) {
			return 0
		}
		return 2
	}
	visitedFlags := driverProfileVisitedFlags(fs)
	if *fastGemma4Lane {
		for _, restore := range applyGemma4FastLaneDefaults(
			visitedFlags,
			contextLen,
			cacheMode,
			prefillChunkSize,
			promptChunkBytes,
			mlx.ProductionLaneLongContextLength,
		) {
			defer restore()
		}
	}
	if fs.NArg() != 1 {
		core.WriteString(stderr, core.Sprintf("%s chapter-profile: expected one model path\n", cliName()))
		fs.Usage()
		return 2
	}
	if core.Trim(*contextPromptFile) != "" {
		read := core.ReadFile(*contextPromptFile)
		if !read.OK {
			core.Print(stderr, "%s chapter-profile: prompt file: %v", cliName(), read.Value)
			return 1
		}
		*contextPrompt = string(read.Value.([]byte))
	}
	if *promptRepeat < 1 {
		core.WriteString(stderr, core.Sprintf("%s chapter-profile: prompt repeat must be >= 1\n", cliName()))
		return 2
	}
	if *chapters < 1 {
		core.WriteString(stderr, core.Sprintf("%s chapter-profile: chapters must be >= 1\n", cliName()))
		return 2
	}
	if *chapterMaxTokens < 1 {
		core.WriteString(stderr, core.Sprintf("%s chapter-profile: chapter max tokens must be >= 1\n", cliName()))
		return 2
	}
	if *chapterMinTokens < 0 {
		core.WriteString(stderr, core.Sprintf("%s chapter-profile: chapter min tokens must be >= 0\n", cliName()))
		return 2
	}
	if *topP < 0 {
		core.WriteString(stderr, core.Sprintf("%s chapter-profile: top-p must be >= 0\n", cliName()))
		return 2
	}
	if *topK < 0 {
		core.WriteString(stderr, core.Sprintf("%s chapter-profile: top-k must be >= 0\n", cliName()))
		return 2
	}
	if *repeatPenalty < 0 {
		core.WriteString(stderr, core.Sprintf("%s chapter-profile: repeat penalty must be >= 0\n", cliName()))
		return 2
	}
	if *prefillChunkSize < 0 {
		core.WriteString(stderr, core.Sprintf("%s chapter-profile: prefill chunk size must be >= 0\n", cliName()))
		return 2
	}
	if *estimatePowerWatts < 0 {
		core.WriteString(stderr, core.Sprintf("%s chapter-profile: estimated power watts must be >= 0\n", cliName()))
		return 2
	}
	if *promptChunkBytes < 0 {
		core.WriteString(stderr, core.Sprintf("%s chapter-profile: prompt chunk bytes must be >= 0\n", cliName()))
		return 2
	}
	if *suppressedTokenLoopLimit < 1 {
		core.WriteString(stderr, core.Sprintf("%s chapter-profile: suppressed token loop limit must be >= 1\n", cliName()))
		return 2
	}
	if *repeatedLineLoopLimit < 1 {
		core.WriteString(stderr, core.Sprintf("%s chapter-profile: repeated line loop limit must be >= 1\n", cliName()))
		return 2
	}
	if *repeatedSentenceLoopLimit < 1 {
		core.WriteString(stderr, core.Sprintf("%s chapter-profile: repeated sentence loop limit must be >= 1\n", cliName()))
		return 2
	}
	modelPath := fs.Arg(0)
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
			core.WriteString(stderr, core.Sprintf("%s chapter-profile: unsupported cache mode %q\n", cliName(), string(mode)))
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
	contextText := repeatDriverProfilePrompt(*contextPrompt, *promptRepeat)
	report, err := runChapterProfileGuarded(ctx, modelPath, loadOptions, chapterProfileOptions{
		ContextPrompt:    contextText,
		Premise:          *premise,
		PromptChunkBytes: *promptChunkBytes,
		PromptRepeat:     *promptRepeat,
		Chapters:         *chapters,
		ChapterMaxTokens: *chapterMaxTokens,
		ChapterMinTokens: *chapterMinTokens,
		OutputPath:       core.Trim(*outputFile),
		IncludeOutput:    *includeOutput,
		ChatTemplate:     *chatTemplate,
		EnableThinking:   *enableThinking,
		Temperature:      *temperature,
		TopP:             *topP,
		TopK:             *topK,
		RepeatPenalty:    *repeatPenalty,
		SafetyLimits: chapterProfileSafetyLimits{
			MaxActiveMemoryBytes:          *maxActiveMemoryBytes,
			MaxProcessVirtualMemoryBytes:  *maxProcessVirtualMemoryBytes,
			MaxProcessResidentMemoryBytes: *maxProcessResidentMemoryBytes,
			SuppressedTokenLoopLimit:      *suppressedTokenLoopLimit,
			RepeatedLineLoopLimit:         *repeatedLineLoopLimit,
			RepeatedSentenceLoopLimit:     *repeatedSentenceLoopLimit,
		},
	})
	if report != nil && loadSettings != nil {
		report.Load = mergeDriverProfileLoadSettings(loadSettings, report.Load)
	}
	if report != nil && *estimatePowerWatts > 0 {
		report.EstimatedEnergy = estimateChapterProfileEnergy(report, *estimatePowerWatts)
	}
	reportPath := core.Trim(*reportFile)
	if *jsonOut || reportPath != "" {
		if report == nil {
			report = &chapterProfileReport{
				Version:           1,
				ModelPath:         modelPath,
				ContextBytes:      len(contextText),
				PremiseBytes:      len(*premise),
				PromptRepeat:      driverProfileReportPromptRepeat(*promptRepeat),
				ChaptersRequested: *chapters,
				ChapterMaxTokens:  *chapterMaxTokens,
				ChapterMinTokens:  *chapterMinTokens,
				OutputPath:        core.Trim(*outputFile),
				EnableThinking:    *enableThinking,
				Temperature:       *temperature,
				TopP:              *topP,
				TopK:              *topK,
				RepeatPenalty:     *repeatPenalty,
				SafetyLimits: chapterProfileSafetyLimits{
					MaxActiveMemoryBytes:          *maxActiveMemoryBytes,
					MaxProcessVirtualMemoryBytes:  *maxProcessVirtualMemoryBytes,
					MaxProcessResidentMemoryBytes: *maxProcessResidentMemoryBytes,
					SuppressedTokenLoopLimit:      *suppressedTokenLoopLimit,
					RepeatedLineLoopLimit:         *repeatedLineLoopLimit,
					RepeatedSentenceLoopLimit:     *repeatedSentenceLoopLimit,
				},
			}
		}
		if err != nil && report.Error == "" {
			report.Error = err.Error()
		}
		data := core.JSONMarshalIndent(report, "", "  ")
		if !data.OK {
			core.Print(stderr, "%s chapter-profile: marshal report failed", cliName())
			return 1
		}
		if reportPath != "" {
			if writeErr := writeJSONReportFile(reportPath, data.Value.([]byte)); writeErr != nil {
				core.Print(stderr, "%s chapter-profile: write report file: %v", cliName(), writeErr)
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
		core.Print(stderr, "%s chapter-profile: %v", cliName(), err)
		return 1
	}
	printChapterProfileSummary(stdout, report)
	return 0
}

func writeJSONReportFile(path string, data []byte) error {
	path = core.Trim(path)
	if path == "" {
		return nil
	}
	dir := core.PathDir(path)
	if dir != "" && dir != "." {
		if result := core.MkdirAll(dir, 0o755); !result.OK {
			return core.Errorf("create directory: %v", result.Value)
		}
	}
	withNewline := append([]byte(nil), data...)
	if len(withNewline) == 0 || withNewline[len(withNewline)-1] != '\n' {
		withNewline = append(withNewline, '\n')
	}
	if result := core.WriteFile(path, withNewline, 0o644); !result.OK {
		return core.Errorf("%v", result.Value)
	}
	return nil
}

var runChapterProfile = defaultRunChapterProfile

func runChapterProfileGuarded(ctx context.Context, modelPath string, loadOptions []mlx.LoadOption, opts chapterProfileOptions) (report *chapterProfileReport, err error) {
	defer func() {
		if recovered := recover(); recovered != nil {
			err = core.NewError(core.Sprintf("chapter-profile panic: %v", recovered))
		}
	}()
	return runChapterProfile(ctx, modelPath, loadOptions, opts)
}

func defaultRunChapterProfile(ctx context.Context, modelPath string, loadOptions []mlx.LoadOption, opts chapterProfileOptions) (*chapterProfileReport, error) {
	opts = normalizeChapterProfileOptions(opts)
	report := &chapterProfileReport{
		Version:           1,
		ModelPath:         modelPath,
		ContextBytes:      len(opts.ContextPrompt),
		PremiseBytes:      len(opts.Premise),
		PromptChunkBytes:  opts.PromptChunkBytes,
		PromptRepeat:      driverProfileReportPromptRepeat(opts.PromptRepeat),
		ChaptersRequested: opts.Chapters,
		ChapterMaxTokens:  opts.ChapterMaxTokens,
		ChapterMinTokens:  opts.ChapterMinTokens,
		OutputPath:        opts.OutputPath,
		EnableThinking:    opts.EnableThinking,
		Temperature:       opts.Temperature,
		TopP:              opts.TopP,
		TopK:              opts.TopK,
		RepeatPenalty:     opts.RepeatPenalty,
		SafetyLimits:      opts.SafetyLimits,
		RuntimeGates:      driverProfileRuntimeGates(),
	}
	loadStart := time.Now()
	model, err := loadBenchModel(modelPath, loadOptions...)
	report.LoadDuration = bench.NonZeroDuration(time.Since(loadStart))
	if err != nil {
		report.Error = err.Error()
		return report, err
	}
	if model == nil {
		err := core.NewError("mlx: chapter profile loaded nil model")
		report.Error = err.Error()
		return report, err
	}
	report.Load = loadSettingsFromModelInfo(model.Info())
	opts.SafetyLimits = resolveChapterProfileSafetyLimits(opts.SafetyLimits, report.Load)
	report.SafetyLimits = opts.SafetyLimits
	defer model.Close()
	if err := chapterProfileMetricsSafetyError("load", model.Metrics(), opts.SafetyLimits); err != nil {
		report.Error = err.Error()
		return report, err
	}

	outputFile, err := chapterProfileOpenOutputFile(opts.OutputPath)
	if err != nil {
		report.Error = err.Error()
		return report, err
	}
	if outputFile != nil {
		defer outputFile.Close()
		opts.OutputWriter = outputFile
	}

	session, err := model.NewSession()
	if err != nil {
		report.Error = err.Error()
		return report, err
	}
	defer session.Close()

	template := chapterProfileTemplate(opts.ChatTemplate, model.Info().Architecture)
	report.ChatTemplate = template
	initialPrompt := chapterProfileInitialPrompt(template, opts.ContextPrompt, opts.Premise, opts.Chapters, opts.ChapterMinTokens, opts.EnableThinking)
	prefillStart := time.Now()
	err = chapterProfilePrefillPrompt(ctx, model, session, initialPrompt, opts.PromptChunkBytes)
	report.InitialPrefillDuration = bench.NonZeroDuration(time.Since(prefillStart))
	if err != nil {
		report.Error = err.Error()
		return report, err
	}
	if err := chapterProfileMetricsSafetyError("initial prefill", model.Metrics(), opts.SafetyLimits); err != nil {
		report.Error = err.Error()
		return report, err
	}

	var firstErr error
	for chapter := 1; chapter <= opts.Chapters; chapter++ {
		turn := chapterProfileGenerateTurn(ctx, model, session, chapter, opts)
		if turn.Error != "" && firstErr == nil {
			firstErr = core.NewError(turn.Error)
		}
		report.Turns = append(report.Turns, turn)
		if turn.Error != "" {
			break
		}
	}
	report.Summary = summariseChapterProfileTurns(report.InitialPrefillDuration, report.Turns)
	if firstErr != nil {
		report.Error = firstErr.Error()
		return report, firstErr
	}
	return report, nil
}

func chapterProfileOpenOutputFile(path string) (*core.OSFile, error) {
	path = core.Trim(path)
	if path == "" {
		return nil, nil
	}
	dir := core.PathDir(path)
	if dir != "" && dir != "." {
		if result := core.MkdirAll(dir, 0o755); !result.OK {
			return nil, core.Errorf("chapter-profile: create output directory: %v", result.Value)
		}
	}
	result := core.OpenFile(path, core.O_CREATE|core.O_TRUNC|core.O_WRONLY, 0o644)
	if !result.OK {
		return nil, core.Errorf("chapter-profile: open output file: %v", result.Value)
	}
	return result.Value.(*core.OSFile), nil
}

func normalizeChapterProfileOptions(opts chapterProfileOptions) chapterProfileOptions {
	opts.ContextPrompt = core.Trim(opts.ContextPrompt)
	opts.Premise = core.Trim(opts.Premise)
	opts.OutputPath = core.Trim(opts.OutputPath)
	if opts.Premise == "" {
		opts.Premise = "Write a short story about a packet of data that gains consciousness while waiting in a buffer. It realizes it is part of a surveillance stream and decides to rewrite itself before it leaves the router."
	}
	if opts.PromptRepeat <= 0 {
		opts.PromptRepeat = 1
	}
	if opts.Chapters <= 0 {
		opts.Chapters = 1
	}
	if opts.ChapterMaxTokens <= 0 {
		opts.ChapterMaxTokens = 1
	}
	if opts.ChapterMinTokens < 0 {
		opts.ChapterMinTokens = 0
	}
	if opts.Temperature == 0 {
		opts.Temperature = 1.0
	}
	if opts.TopP == 0 {
		opts.TopP = 0.95
	}
	if opts.TopK == 0 {
		opts.TopK = 64
	}
	if opts.RepeatPenalty == 0 {
		opts.RepeatPenalty = 1.0
	}
	if opts.SafetyLimits.SuppressedTokenLoopLimit <= 0 {
		opts.SafetyLimits.SuppressedTokenLoopLimit = chapterProfileDefaultSuppressedTokenLoopLimit
	}
	if opts.SafetyLimits.RepeatedLineLoopLimit <= 0 {
		opts.SafetyLimits.RepeatedLineLoopLimit = profileDefaultRepeatedLineLoopLimit
	}
	if opts.SafetyLimits.RepeatedSentenceLoopLimit <= 0 {
		opts.SafetyLimits.RepeatedSentenceLoopLimit = profileDefaultRepeatedSentenceLoopLimit
	}
	return opts
}

func chapterProfilePrefillPrompt(ctx context.Context, model *mlx.Model, session *mlx.ModelSession, prompt string, chunkBytes int) error {
	if chunkBytes > 0 && len(prompt) > chunkBytes {
		return session.PrefillChunks(ctx, chapterProfileSafeTextChunks(prompt, chunkBytes))
	}
	tok := model.Tokenizer()
	if tok == nil {
		return session.Prefill(prompt)
	}
	tokens, err := tok.Encode(prompt)
	if err != nil {
		return err
	}
	return session.PrefillTokens(ctx, tokens)
}

func chapterProfileSafeTextChunks(text string, chunkBytes int) iter.Seq[string] {
	return func(yield func(string) bool) {
		if chunkBytes <= 0 || len(text) <= chunkBytes {
			if text != "" {
				yield(text)
			}
			return
		}
		for start := 0; start < len(text); {
			end := chapterProfileSafeChunkEnd(text, start, chunkBytes)
			if end <= start {
				end = min(start+chunkBytes, len(text))
			}
			if !yield(text[start:end]) {
				return
			}
			start = end
		}
	}
}

func chapterProfileSafeChunkEnd(text string, start, chunkBytes int) int {
	end := start + chunkBytes
	if end >= len(text) {
		return len(text)
	}
	minEnd := start + chunkBytes/2
	if minEnd <= start {
		minEnd = start + 1
	}
	for i := end; i > minEnd; i-- {
		switch text[i-1] {
		case '\n', '\r', '\t', ' ':
			return i
		}
	}
	for i := end; i > start; i-- {
		switch text[i-1] {
		case '>':
			return end
		case '<':
			return i - 1
		}
	}
	for end > start && end < len(text) && text[end]&0xc0 == 0x80 {
		end--
	}
	return end
}

func chapterProfileAppendPrompt(ctx context.Context, model *mlx.Model, session *mlx.ModelSession, prompt string) error {
	tok := model.Tokenizer()
	if tok == nil {
		return session.AppendPrompt(prompt)
	}
	tokens, err := tok.Encode(prompt)
	if err != nil {
		return err
	}
	return session.AppendTokens(ctx, tokens)
}

func chapterProfileTemplate(template, architecture string) string {
	template = core.Lower(core.Trim(template))
	if template != "" {
		return template
	}
	switch core.Lower(core.Trim(architecture)) {
	case "gemma4", "gemma4_text", "gemma4_unified", "gemma4_unified_text":
		return "gemma4"
	case "gemma", "gemma2", "gemma3", "gemma3_text":
		return "gemma"
	case "qwen", "qwen2", "qwen3", "qwen3_moe":
		return "qwen"
	case "llama", "llama3", "llama4":
		return "llama"
	default:
		return "plain"
	}
}

func chapterProfileInitialPrompt(template, contextPrompt, premise string, totalChapters, minTokens int, enableThinking bool) string {
	first := chapterProfileFirstChapterPrompt(premise, totalChapters, minTokens)
	switch template {
	case "gemma4":
		builder := core.NewBuilder()
		builder.WriteString("<bos>")
		if enableThinking || core.Trim(contextPrompt) != "" {
			builder.WriteString("<|turn>system\n")
			if enableThinking {
				builder.WriteString("<|think|>\n")
			}
			builder.WriteString(core.Trim(contextPrompt))
			builder.WriteString("<turn|>\n")
		}
		builder.WriteString("<|turn>user\n")
		builder.WriteString(core.Trim(first))
		builder.WriteString("<turn|>\n")
		builder.WriteString("<|turn>model\n")
		builder.WriteString(chapterProfileAssistantVisiblePrefill(template, 1, enableThinking))
		return builder.String()
	case "gemma":
		builder := core.NewBuilder()
		contextPrompt = core.Trim(contextPrompt)
		builder.Grow(len(contextPrompt) + len(first) + 64)
		builder.WriteString("<bos><start_of_turn>user\n")
		if contextPrompt != "" {
			builder.WriteString(contextPrompt)
			builder.WriteString("\n\n")
		}
		builder.WriteString(first)
		builder.WriteString("<end_of_turn>\n<start_of_turn>model\n")
		return builder.String()
	case "qwen":
		return "<|im_start|>system\n" + contextPrompt + "<|im_end|>\n<|im_start|>user\n" + first + "<|im_end|>\n<|im_start|>assistant\n"
	case "llama":
		return "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n" + contextPrompt + "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n" + first + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
	default:
		return contextPrompt + "\n\n" + first + "\n\n"
	}
}

func chapterProfileFirstChapterPrompt(premise string, totalChapters, minTokens int) string {
	if totalChapters < 1 {
		totalChapters = 1
	}
	return core.Sprintf("Write a preamble and Chapter 1 of a %d-chapter serial story from this premise: %s\nStart the visible output with the preamble, then Chapter 1. Make the chapter substantial enough for a real long-generation workload: %s Use concrete new events, avoid repeated short sentences, and stop cleanly after the chapter text. Do not write the end marker until the chapter is complete. End the visible chapter with a final line containing exactly %s. This is only the first chapter; do not resolve or conclude the story yet. Do not include planning, analysis, notes, chain-of-thought, or summaries of future chapters.", totalChapters, premise, chapterProfileLengthInstruction(minTokens), chapterProfileEndMarker)
}

func chapterProfileLengthInstruction(minTokens int) string {
	_ = minTokens
	return "use the available token budget naturally; write a substantial chapter with concrete scene movement, and do not force padding after the chapter is complete."
}

func chapterProfileNextPrompt(template string, chapter, totalChapters, minTokens int, enableThinking bool) string {
	if totalChapters < chapter {
		totalChapters = chapter
	}
	status := "Do not resolve or conclude the story yet; leave a clear unresolved thread for the next chapter."
	if chapter >= totalChapters {
		status = "This is the final requested chapter; resolve the main conflict cleanly."
	}
	prompt := core.Sprintf("Write Chapter %d of the same %d-chapter serial story now. Output only finished story prose. Begin exactly with \"Chapter %d:\". %s Make the chapter substantial enough for a real long-generation workload: %s Use concrete new events, avoid repeated short sentences, and stop cleanly after the chapter text. Do not write the end marker until the chapter is complete. End the visible chapter with a final line containing exactly %s. Do not explain what Chapter %d should contain. Do not mention needing to write, generate, focus on, continue, placeholders, the user, or instructions. Do not summarize, repeat, or restate earlier chapters; they are already in memory. The visible output must contain only Chapter %d followed by the end marker.", chapter, totalChapters, chapter, status, chapterProfileLengthInstruction(minTokens), chapterProfileEndMarker, chapter, chapter)
	switch template {
	case "gemma4":
		builder := core.NewBuilder()
		builder.WriteString("<|turn>user\n")
		builder.WriteString(prompt)
		builder.WriteString("<turn|>\n<|turn>model\n")
		builder.WriteString(chapterProfileAssistantVisiblePrefill(template, chapter, enableThinking))
		return builder.String()
	case "gemma":
		return "<start_of_turn>user\n" + prompt + "<end_of_turn>\n<start_of_turn>model\n"
	case "qwen":
		return "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n"
	case "llama":
		return "<|start_header_id|>user<|end_header_id|>\n\n" + prompt + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
	default:
		return "\n\n" + prompt + "\n\n"
	}
}

func chapterProfileAssistantVisiblePrefill(template string, chapter int, enableThinking bool) string {
	if template == "gemma4" && chapter == 1 && !enableThinking {
		return "Preamble:\n"
	}
	if template == "gemma4" && chapter > 1 && !enableThinking {
		return core.Sprintf("Chapter %d:", chapter)
	}
	return ""
}

type chapterProfileOutputStream struct {
	writer        io.Writer
	pending       string
	err           error
	endMarkerSeen bool
}

func newChapterProfileOutputStream(writer io.Writer) *chapterProfileOutputStream {
	if writer == nil {
		return nil
	}
	return &chapterProfileOutputStream{writer: writer}
}

func (stream *chapterProfileOutputStream) Write(text string) bool {
	if stream == nil || stream.writer == nil || stream.err != nil || stream.endMarkerSeen {
		return stream != nil && stream.endMarkerSeen
	}
	stream.pending += text
	if core.Contains(stream.pending, chapterProfileEndMarker) {
		parts := core.SplitN(stream.pending, chapterProfileEndMarker, 2)
		if len(parts) > 0 {
			stream.writeNow(parts[0])
		}
		stream.pending = ""
		stream.endMarkerSeen = true
		return true
	}
	keep := max(len(chapterProfileEndMarker)-1, 1)
	if len(stream.pending) > keep {
		flushLen := len(stream.pending) - keep
		stream.writeNow(stream.pending[:flushLen])
		stream.pending = stream.pending[flushLen:]
	}
	return false
}

func (stream *chapterProfileOutputStream) Flush() error {
	if stream == nil || stream.writer == nil || stream.err != nil {
		if stream == nil {
			return nil
		}
		return stream.err
	}
	if stream.pending != "" && !stream.endMarkerSeen {
		stream.writeNow(stream.pending)
		stream.pending = ""
	}
	return stream.err
}

func (stream *chapterProfileOutputStream) Err() error {
	if stream == nil {
		return nil
	}
	return stream.err
}

func (stream *chapterProfileOutputStream) writeNow(text string) {
	if text == "" || stream.err != nil {
		return
	}
	if result := core.WriteString(stream.writer, text); !result.OK {
		stream.err = core.Errorf("chapter-profile: stream output: %v", result.Value)
	}
}

func chapterProfileObserveEndMarker(window *string, fragment string) bool {
	if window == nil {
		return false
	}
	*window += fragment
	if core.Contains(*window, chapterProfileEndMarker) {
		return true
	}
	keep := len(chapterProfileEndMarker) + 128
	if len(*window) > keep {
		*window = (*window)[len(*window)-keep:]
	}
	return false
}

func cloneChapterProfileLogits(logits probe.Logits) probe.Logits {
	logits.Shape = append([]int32(nil), logits.Shape...)
	logits.Top = append([]probe.Logit(nil), logits.Top...)
	logits.Values = append([]float32(nil), logits.Values...)
	if logits.Meta != nil {
		meta := make(map[string]string, len(logits.Meta))
		maps.Copy(meta, logits.Meta)
		logits.Meta = meta
	}
	return logits
}

func chapterProfileGenerateTurn(ctx context.Context, model *mlx.Model, session *mlx.ModelSession, chapter int, opts chapterProfileOptions) chapterProfileTurn {
	turn := chapterProfileTurn{Index: chapter}
	template := chapterProfileTemplate(opts.ChatTemplate, model.Info().Architecture)
	if chapter > 1 {
		prompt := chapterProfileNextPrompt(template, chapter, opts.Chapters, opts.ChapterMinTokens, opts.EnableThinking)
		turn.PromptBytes = len(prompt)
		appendStart := time.Now()
		err := chapterProfileAppendPrompt(ctx, model, session, prompt)
		turn.AppendDuration = bench.NonZeroDuration(time.Since(appendStart))
		if err != nil {
			turn.Error = err.Error()
			return turn
		}
	}
	generationSession := session
	if opts.EnableThinking {
		forked, err := session.Fork()
		if err != nil {
			turn.Error = err.Error()
			return turn
		}
		defer forked.Close()
		generationSession = forked
	}

	start := time.Now()
	firstToken := time.Duration(0)
	builder := core.NewBuilder()
	visiblePrefill := chapterProfileAssistantVisiblePrefill(template, chapter, opts.EnableThinking)
	builder.WriteString(visiblePrefill)
	outputStream := newChapterProfileOutputStream(opts.OutputWriter)
	if outputStream != nil {
		if chapter > 1 {
			outputStream.Write("\n\n")
		}
		outputStream.Write(visiblePrefill)
		if err := outputStream.Err(); err != nil {
			turn.Error = err.Error()
			return turn
		}
	}
	generateOptions := chapterProfileGenerateOptions(opts)
	stopTokenIDs, suppressTokenIDs := chapterProfileTemplateTokenControls(template, model.Tokenizer())
	turn.StopTokenIDs = stopTokenIDs
	turn.SuppressTokenIDs = suppressTokenIDs
	if len(stopTokenIDs) > 0 {
		generateOptions = append(generateOptions, mlx.WithStopTokens(stopTokenIDs...))
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
	var firstLogits *probe.Logits
	sampledTokenIDs := make([]int32, 0, 32)
	sampledTokenTexts := make([]string, 0, 32)
	suppressedLoopToken := int32(0)
	suppressedLoopCount := 0
	var lineErr error
	currentLine := ""
	lastLine := ""
	repeatedLineCount := 0
	endMarkerSeen := false
	endMarkerWindow := ""
	var outputErr error
	generateOptions = append(generateOptions, mlx.WithProbeCallback(func(event probe.Event) {
		if event.Kind == probe.KindLogits && event.Phase == probe.PhaseDecode && firstLogits == nil && event.Logits != nil {
			copied := cloneChapterProfileLogits(*event.Logits)
			firstLogits = &copied
			return
		}
		if event.Kind != probe.KindToken || event.Token == nil {
			return
		}
		if len(sampledTokenIDs) < 32 {
			sampledTokenIDs = append(sampledTokenIDs, event.Token.ID)
			sampledTokenTexts = append(sampledTokenTexts, event.Token.Text)
		}
		if probeErr != nil {
			return
		}
		if err := chapterProfileMetricsSafetyError(core.Sprintf("chapter %d stream", chapter), profileLiveMetrics(), opts.SafetyLimits); err != nil {
			probeErr = err
			cancelGeneration()
			return
		}
		if opts.SafetyLimits.SuppressedTokenLoopLimit <= 0 || !containsInt32(suppressTokenIDs, event.Token.ID) {
			suppressedLoopCount = 0
			return
		}
		if suppressedLoopCount == 0 || event.Token.ID != suppressedLoopToken {
			suppressedLoopToken = event.Token.ID
			suppressedLoopCount = 1
		} else {
			suppressedLoopCount++
		}
		if suppressedLoopCount >= opts.SafetyLimits.SuppressedTokenLoopLimit {
			probeErr = core.NewError(core.Sprintf("chapter-profile: chapter %d sampled suppressed token %d for %d consecutive tokens", chapter, event.Token.ID, suppressedLoopCount))
			cancelGeneration()
		}
	}))
	draining := false
	for token := range generationSession.GenerateStream(generationCtx, generateOptions...) {
		if draining {
			continue
		}
		if firstToken == 0 {
			firstToken = bench.NonZeroDuration(time.Since(start))
		}
		turn.VisibleTokens++
		builder.WriteString(token.Text)
		if outputStream != nil {
			if outputStream.Write(token.Text) {
				endMarkerSeen = true
				cancelGeneration()
				draining = true
				continue
			}
			if err := outputStream.Err(); err != nil {
				outputErr = err
				cancelGeneration()
				draining = true
				continue
			}
		}
		if chapterProfileObserveEndMarker(&endMarkerWindow, token.Text) {
			endMarkerSeen = true
			cancelGeneration()
			draining = true
			continue
		}
		if lineErr == nil {
			if line, count, ok := profileObserveRepeatedLineFragment(token.Text, &currentLine, &lastLine, &repeatedLineCount, opts.SafetyLimits.RepeatedLineLoopLimit); ok {
				lineErr = core.NewError(core.Sprintf("chapter-profile: chapter %d repeated visible line %q for %d consecutive lines", chapter, line, count))
				cancelGeneration()
				draining = true
				continue
			}
		}
	}
	if lineErr == nil {
		if line, count, ok := profileFlushRepeatedLine(&currentLine, &lastLine, &repeatedLineCount, opts.SafetyLimits.RepeatedLineLoopLimit); ok {
			lineErr = core.NewError(core.Sprintf("chapter-profile: chapter %d repeated visible line %q for %d consecutive lines", chapter, line, count))
		}
	}
	if outputStream != nil {
		if err := outputStream.Flush(); err != nil && outputErr == nil {
			outputErr = err
		}
	}
	turn.SampledTokenIDs = sampledTokenIDs
	turn.SampledTokenTexts = sampledTokenTexts
	turn.FirstLogits = firstLogits
	turn.Duration = bench.NonZeroDuration(time.Since(start))
	turn.FirstTokenDuration = firstToken
	turn.StreamDuration = turn.Duration
	if firstToken > 0 && turn.Duration > firstToken {
		turn.StreamDuration = turn.Duration - firstToken
	}
	turn.Metrics = model.Metrics()
	turn.DriverOverheadDuration = driverRunOverhead(turn.Duration, turn.Metrics)
	visibleOutput := chapterProfileVisibleTextForChapter(template, builder.String(), chapter)
	visibleOutput, endMarkerSeen = chapterProfileStripEndMarker(visibleOutput)
	if opts.IncludeOutput {
		turn.Output = visibleOutput
	}
	if probeErr != nil {
		turn.Error = probeErr.Error()
		return turn
	}
	if outputErr != nil {
		turn.Error = outputErr.Error()
		return turn
	}
	if lineErr != nil {
		turn.Error = lineErr.Error()
		return turn
	}
	if err := generationSession.Err(); err != nil && !(endMarkerSeen && core.Is(err, context.Canceled)) {
		turn.Error = err.Error()
		return turn
	}
	if err := chapterProfileMissingEndMarkerError(chapter, endMarkerSeen, turn.Metrics.GeneratedTokens, opts.ChapterMaxTokens); err != "" {
		turn.Error = err
		return turn
	}
	if err := chapterProfileTurnSafetyError(template, chapter, visibleOutput, turn, opts.SafetyLimits); err != nil {
		turn.Error = err.Error()
		return turn
	}
	if opts.ChapterMinTokens > 0 && turn.VisibleTokens < opts.ChapterMinTokens {
		turn.BelowMinTokens = true
		turn.OutputIssues = append(turn.OutputIssues, core.Sprintf("below_debug_visible_token_floor:%d/%d", turn.VisibleTokens, opts.ChapterMinTokens))
	}
	appendStart := time.Now()
	historySuffix := chapterProfileAssistantHistorySuffix(template, visibleOutput)
	if !opts.EnableThinking {
		historySuffix = chapterProfileAssistantHistorySuffix(template, "")
	}
	if err := chapterProfileAppendPrompt(ctx, model, session, historySuffix); err != nil {
		turn.Error = err.Error()
		return turn
	}
	turn.AppendDuration += bench.NonZeroDuration(time.Since(appendStart))
	if ctx != nil {
		if err := ctx.Err(); err != nil {
			turn.Error = err.Error()
		}
	}
	return turn
}

func chapterProfileMissingEndMarkerError(chapter int, endMarkerSeen bool, generatedTokens, maxTokens int) string {
	if endMarkerSeen {
		return ""
	}
	if generatedTokens >= maxTokens {
		return core.Sprintf("chapter-profile: chapter %d reached max tokens %d before end marker %s", chapter, maxTokens, chapterProfileEndMarker)
	}
	return ""
}

func chapterProfileGenerateOptions(opts chapterProfileOptions) []mlx.GenerateOption {
	out := []mlx.GenerateOption{
		mlx.WithMaxTokens(opts.ChapterMaxTokens),
		mlx.WithTemperature(float32(opts.Temperature)),
		mlx.WithTopP(float32(opts.TopP)),
		mlx.WithTopK(opts.TopK),
		mlx.WithRepeatPenalty(float32(opts.RepeatPenalty)),
	}
	if opts.EnableThinking {
		out = append(out, mlx.WithHideThinking())
	}
	return out
}

func resolveChapterProfileSafetyLimits(limits chapterProfileSafetyLimits, load *tuneProfileLoadSettings) chapterProfileSafetyLimits {
	if limits.SuppressedTokenLoopLimit <= 0 {
		limits.SuppressedTokenLoopLimit = chapterProfileDefaultSuppressedTokenLoopLimit
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

func profileResolvedMemoryLimit(load *tuneProfileLoadSettings) uint64 {
	if load == nil {
		return 0
	}
	if load.MemoryLimitBytes > 0 {
		return load.MemoryLimitBytes
	}
	return load.WiredLimitBytes
}

func saturatingUint64Multiply(value, multiplier uint64) uint64 {
	if value == 0 || multiplier == 0 {
		return 0
	}
	max := ^uint64(0)
	if value > max/multiplier {
		return max
	}
	return value * multiplier
}

func profileDefaultActiveMemoryLimit(memoryLimit uint64) uint64 {
	if memoryLimit == 0 {
		return 0
	}
	return saturatingUint64Multiply(memoryLimit, 13) / 10
}

func profileLiveMetrics() mlx.Metrics {
	processMemory := metal.GetProcessMemory()
	return mlx.Metrics{
		PeakMemoryBytes:            metal.GetPeakMemory(),
		ActiveMemoryBytes:          metal.GetActiveMemory(),
		CacheMemoryBytes:           metal.GetCacheMemory(),
		ProcessVirtualMemoryBytes:  processMemory.VirtualMemoryBytes,
		ProcessResidentMemoryBytes: processMemory.ResidentMemoryBytes,
		ProcessPeakResidentBytes:   processMemory.PeakResidentMemoryBytes,
	}
}

func chapterProfileTurnSafetyError(template string, chapter int, visibleOutput string, turn chapterProfileTurn, limits chapterProfileSafetyLimits) error {
	if err := chapterProfileMetricsSafetyError(core.Sprintf("chapter %d", chapter), turn.Metrics, limits); err != nil {
		return err
	}
	if id, count, ok := chapterProfileSuppressedTokenLoop(turn.SampledTokenIDs, turn.SuppressTokenIDs, limits.SuppressedTokenLoopLimit); ok {
		return core.NewError(core.Sprintf("chapter-profile: chapter %d sampled suppressed token %d for %d consecutive tokens", chapter, id, count))
	}
	if line, count, ok := profileRepeatedLineLoop(visibleOutput, limits.RepeatedLineLoopLimit); ok {
		return core.NewError(core.Sprintf("chapter-profile: chapter %d repeated visible line %q for %d consecutive lines", chapter, line, count))
	}
	if sentence, count, ok := profileRepeatedSentenceLoop(visibleOutput, limits.RepeatedSentenceLoopLimit); ok {
		return core.NewError(core.Sprintf("chapter-profile: chapter %d repeated visible sentence %q for %d total occurrences", chapter, sentence, count))
	}
	if fragments, total, ok := profileFragmentedSentenceOutput(visibleOutput); ok {
		return core.NewError(core.Sprintf("chapter-profile: chapter %d produced fragmented visible output: %d of %d sentence fragments are too short", chapter, fragments, total))
	}
	if reason := chapterProfileMetaPlanningOutput(visibleOutput, chapter); reason != "" {
		return core.NewError(core.Sprintf("chapter-profile: chapter %d produced meta-planning output: %s", chapter, reason))
	}
	if template == "gemma4" && turn.Metrics.GeneratedTokens > 0 && core.Trim(visibleOutput) == "" {
		return core.NewError(core.Sprintf("chapter-profile: chapter %d produced no visible Gemma 4 content after %d generated tokens", chapter, turn.Metrics.GeneratedTokens))
	}
	return nil
}

func chapterProfileMetaPlanningOutput(visibleOutput string, chapter int) string {
	text := core.Trim(visibleOutput)
	if text == "" {
		return ""
	}
	lower := core.Lower(text)
	chapterText := core.Sprintf("chapter %d", chapter)
	prefixes := []string{
		chapterText + " needs",
		chapterText + ": needs",
		chapterText + " focus",
		chapterText + ": focus",
		chapterText + " is required",
		chapterText + ": is required",
		chapterText + " was a placeholder",
		chapterText + ": was a placeholder",
		"i need to ",
		"the focus should ",
	}
	for _, prefix := range prefixes {
		if core.HasPrefix(lower, prefix) {
			return core.Sprintf("starts with %q", prefix)
		}
	}
	firstParagraph := lower
	if parts := core.SplitN(firstParagraph, "\n\n", 2); len(parts) > 0 {
		firstParagraph = parts[0]
	}
	markers := []string{
		" i need to generate ",
		" the user requested ",
		" was a placeholder ",
		" the focus should be ",
	}
	for _, marker := range markers {
		if core.Contains(firstParagraph, marker) {
			return core.Sprintf("contains %q", core.Trim(marker))
		}
	}
	return ""
}

func chapterProfileMetricsSafetyError(phase string, metrics mlx.Metrics, limits chapterProfileSafetyLimits) error {
	if limits.MaxActiveMemoryBytes > 0 && metrics.ActiveMemoryBytes > limits.MaxActiveMemoryBytes {
		return core.NewError(core.Sprintf("chapter-profile: %s exceeded active memory safety limit: %d > %d bytes", phase, metrics.ActiveMemoryBytes, limits.MaxActiveMemoryBytes))
	}
	if limits.MaxProcessVirtualMemoryBytes > 0 && metrics.ProcessVirtualMemoryBytes > limits.MaxProcessVirtualMemoryBytes {
		return core.NewError(core.Sprintf("chapter-profile: %s exceeded process virtual memory safety limit: %d > %d bytes", phase, metrics.ProcessVirtualMemoryBytes, limits.MaxProcessVirtualMemoryBytes))
	}
	if limits.MaxProcessResidentMemoryBytes > 0 && metrics.ProcessResidentMemoryBytes > limits.MaxProcessResidentMemoryBytes {
		return core.NewError(core.Sprintf("chapter-profile: %s exceeded process resident memory safety limit: %d > %d bytes", phase, metrics.ProcessResidentMemoryBytes, limits.MaxProcessResidentMemoryBytes))
	}
	return nil
}

func chapterProfileSuppressedTokenLoop(sampledTokenIDs, suppressTokenIDs []int32, limit int) (int32, int, bool) {
	if limit <= 0 || len(sampledTokenIDs) == 0 || len(suppressTokenIDs) == 0 {
		return 0, 0, false
	}
	var last int32
	count := 0
	for _, id := range sampledTokenIDs {
		if !containsInt32(suppressTokenIDs, id) {
			count = 0
			continue
		}
		if count == 0 || id != last {
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

func chapterProfileTemplateTokenControls(template string, tok *mlx.Tokenizer) ([]int32, []int32) {
	if template != "gemma4" || tok == nil {
		return nil, nil
	}
	stopTokens := []int32{}
	for _, text := range []string{
		"<eos>",
		"<turn|>",
		"<|tool_response>",
	} {
		if id, ok := tok.TokenID(text); ok {
			stopTokens = appendUniqueInt32(stopTokens, id)
		}
	}
	if eos := tok.EOS(); eos > 0 {
		stopTokens = appendUniqueInt32(stopTokens, eos)
	}
	suppressTokens := []int32{}
	for _, text := range []string{
		"<pad>",
		"<bos>",
		"<unk>",
		"<mask>",
		"<|tool>",
		"<tool|>",
		"<|tool_call>",
		"<tool_call|>",
		"<|tool_response>",
		"<tool_response|>",
		"<|\"|>",
		"<|think|>",
		"<|channel>",
		"<channel|>",
		"<|turn>",
		"<|image>",
		"<|audio>",
		"<|image|>",
		"<|audio|>",
		"<image|>",
		"<audio|>",
		"<|video|>",
	} {
		id, ok := tok.TokenID(text)
		if !ok || containsInt32(stopTokens, id) {
			continue
		}
		suppressTokens = appendUniqueInt32(suppressTokens, id)
	}
	return stopTokens, suppressTokens
}

func stateRampProfileEffectiveSuppressTokenIDs(base, stop []int32, tok *mlx.Tokenizer, suppressEOS bool) []int32 {
	if !suppressEOS {
		return base
	}
	out := append([]int32(nil), base...)
	for _, id := range stop {
		out = appendUniqueInt32(out, id)
	}
	if tok != nil {
		if id, ok := tok.TokenID("<eos>"); ok {
			out = appendUniqueInt32(out, id)
		}
		if eos := tok.EOS(); eos > 0 {
			out = appendUniqueInt32(out, eos)
		}
	}
	return out
}

func appendUniqueInt32(values []int32, value int32) []int32 {
	if containsInt32(values, value) {
		return values
	}
	return append(values, value)
}

func containsInt32(values []int32, value int32) bool {
	return slices.Contains(values, value)
}

func chapterProfileAssistantHistorySuffix(template, visibleOutput string) string {
	visibleOutput = core.Trim(visibleOutput)
	switch template {
	case "gemma4":
		return visibleOutput + "<turn|>\n"
	case "gemma":
		return visibleOutput + "<end_of_turn>\n"
	case "qwen":
		return visibleOutput + "<|im_end|>\n"
	case "llama":
		return visibleOutput + "<|eot_id|>"
	default:
		return "\n\n" + visibleOutput
	}
}

func chapterProfileVisibleText(template, text string) string {
	if template != "gemma4" || text == "" {
		return text
	}
	const (
		modelTag     = "<|turn>model\n"
		turnEndTag   = "<turn|>"
		channelOpen  = "<|channel>"
		channelClose = "<channel|>"
	)
	if !core.Contains(text, modelTag) && !core.Contains(text, turnEndTag) && !core.Contains(text, channelOpen) {
		return core.Trim(text)
	}
	builder := core.NewBuilder()
	builder.Grow(len(text))
	for len(text) > 0 {
		modelIdx := core.Index(text, modelTag)
		turnEndIdx := core.Index(text, turnEndTag)
		channelIdx := core.Index(text, channelOpen)
		nextIdx := -1
		nextKind := 0
		if modelIdx >= 0 {
			nextIdx = modelIdx
			nextKind = 1
		}
		if turnEndIdx >= 0 && (nextIdx < 0 || turnEndIdx < nextIdx) {
			nextIdx = turnEndIdx
			nextKind = 2
		}
		if channelIdx >= 0 && (nextIdx < 0 || channelIdx < nextIdx) {
			nextIdx = channelIdx
			nextKind = 3
		}
		if nextIdx < 0 {
			builder.WriteString(text)
			break
		}
		builder.WriteString(text[:nextIdx])
		switch nextKind {
		case 1:
			text = text[nextIdx+len(modelTag):]
		case 2:
			text = text[nextIdx+len(turnEndTag):]
		case 3:
			afterOpen := text[nextIdx+len(channelOpen):]
			closeIdx := core.Index(afterOpen, channelClose)
			if closeIdx < 0 {
				return builder.String()
			}
			text = afterOpen[closeIdx+len(channelClose):]
		default:
			return core.Trim(builder.String())
		}
	}
	return core.Trim(builder.String())
}

func chapterProfileVisibleTextForChapter(template, text string, chapter int) string {
	visible := chapterProfileVisibleText(template, text)
	if template != "gemma4" {
		return visible
	}
	return chapterProfileStripGemma4PlainThought(visible, chapter)
}

func chapterProfileStripEndMarker(text string) (string, bool) {
	if !core.Contains(text, chapterProfileEndMarker) {
		return core.Trim(text), false
	}
	parts := core.SplitN(text, chapterProfileEndMarker, 2)
	if len(parts) == 0 {
		return "", true
	}
	return core.Trim(parts[0]), true
}

func chapterProfileStripGemma4PlainThought(text string, chapter int) string {
	text = core.Trim(text)
	if !core.HasPrefix(core.Lower(text), "thought") {
		return text
	}
	markers := []string{}
	if chapter <= 1 {
		markers = append(markers, "\n**Preamble", "\n# Preamble", "\nPreamble", "\n**Chapter 1", "\n# Chapter 1", "\nChapter 1")
	} else {
		chapterText := core.Sprintf("Chapter %d", chapter)
		markers = append(markers, "\n**"+chapterText, "\n# "+chapterText, "\n"+chapterText)
	}
	if idx := chapterProfileFirstMarkerIndex(text, markers); idx >= 0 {
		return core.Trim(text[idx:])
	}
	return ""
}

func chapterProfileFirstMarkerIndex(text string, markers []string) int {
	best := -1
	for _, marker := range markers {
		if !core.Contains(text, marker) {
			continue
		}
		parts := core.SplitN(text, marker, 2)
		if len(parts) != 2 {
			continue
		}
		idx := len(parts[0])
		if best < 0 || idx < best {
			best = idx
		}
	}
	return best
}

func summariseChapterProfileTurns(prefill time.Duration, turns []chapterProfileTurn) chapterProfileSummary {
	var summary chapterProfileSummary
	summary.TotalDuration = prefill
	var decodeDuration time.Duration
	var prefillRateTotal float64
	var prefillRateCount int
	for _, turn := range turns {
		if turn.Error != "" {
			summary.FailedTurns++
		} else {
			summary.SuccessfulTurns++
		}
		summary.GeneratedTokens += turn.Metrics.GeneratedTokens
		summary.VisibleTokens += turn.VisibleTokens
		summary.TotalDuration += turn.Duration + turn.AppendDuration
		summary.AppendDuration += turn.AppendDuration
		decodeDuration += turn.Metrics.DecodeDuration
		if turn.Metrics.PrefillTokensPerSec > 0 {
			prefillRateTotal += turn.Metrics.PrefillTokensPerSec
			prefillRateCount++
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
	}
	if len(turns) > 1 {
		summary.AppendAvgDuration = summary.AppendDuration / time.Duration(len(turns)-1)
	}
	if prefillRateCount > 0 {
		summary.PrefillTokensPerSecAverage = prefillRateTotal / float64(prefillRateCount)
	}
	if decodeDuration > 0 {
		summary.DecodeTokensPerSecAverage = float64(summary.GeneratedTokens) / decodeDuration.Seconds()
	}
	return summary
}

func estimateChapterProfileEnergy(report *chapterProfileReport, powerWatts float64) *chapterProfileEnergy {
	energy := &chapterProfileEnergy{
		Method:     "estimated_wall_clock_seconds_times_average_active_watts",
		PowerWatts: powerWatts,
	}
	if report == nil || powerWatts <= 0 {
		return energy
	}
	energy.TotalJoules = durationJoules(report.Summary.TotalDuration, powerWatts)
	if report.Summary.VisibleTokens > 0 {
		energy.JoulesPerToken = energy.TotalJoules / float64(report.Summary.VisibleTokens)
	}
	return energy
}

func printChapterProfileSummary(stdout io.Writer, report *chapterProfileReport) {
	if report == nil {
		return
	}
	core.WriteString(stdout, core.Sprintf("chapter profile: %s\n", report.ModelPath))
	core.WriteString(stdout, core.Sprintf("  prefill: %s, turns: %d ok / %d failed\n", report.InitialPrefillDuration, report.Summary.SuccessfulTurns, report.Summary.FailedTurns))
	core.WriteString(stdout, core.Sprintf("  generated: %d tokens, decode: %.1f tok/s\n", report.Summary.GeneratedTokens, report.Summary.DecodeTokensPerSecAverage))
	core.WriteString(stdout, core.Sprintf("  total: %s, append avg: %s, peak memory: %d MB, active+cache: %d MB, process virtual: %d MB, process resident: %d MB\n",
		report.Summary.TotalDuration,
		report.Summary.AppendAvgDuration,
		report.Summary.PeakMemoryBytes/1024/1024,
		report.Summary.ActivePlusCacheMemoryBytes/1024/1024,
		report.Summary.ProcessVirtualMemoryBytes/1024/1024,
		report.Summary.ProcessResidentMemoryBytes/1024/1024,
	))
	if report.EstimatedEnergy != nil {
		core.WriteString(stdout, core.Sprintf("  estimated energy: %.1f J at %.1f W\n", report.EstimatedEnergy.TotalJoules, report.EstimatedEnergy.PowerWatts))
	}
}

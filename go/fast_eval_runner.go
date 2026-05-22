// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"
	"dappco.re/go/mlx/blockcache"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference/bench"
	"dappco.re/go/inference/decode"
	state "dappco.re/go/inference/state"
	filestore "dappco.re/go/inference/state/filestore"
	"dappco.re/go/mlx/bundle"
	"dappco.re/go/mlx/kv"
	"dappco.re/go/mlx/probe"
)

// Hoisted package-level sentinel — the decode generate closure runs once
// per bench iteration (target + draft) plus once per speculative decode
// call. Sharing one *Err avoids the per-call core.NewError allocation on
// the otherwise hot path.
var errModelDecodeNil = core.NewError("mlx: bench decode runner has nil model")

// NewModelFastEvalRunner adapts a loaded Model to bench.Runner with
// verb-shaped callbacks for each driver-specific bench section.
func NewModelFastEvalRunner(model *Model) bench.Runner {
	return NewModelFastEvalRunnerWithDraft(model, nil)
}

// NewModelFastEvalRunnerWithDraft adapts a loaded target Model plus an optional
// assistant/draft Model to bench.Runner.
func NewModelFastEvalRunnerWithDraft(model, draft *Model) bench.Runner {
	return bench.Runner{
		Info: func(ctx context.Context) bench.Info {
			if err := ctx.Err(); err != nil || model == nil {
				return bench.Info{}
			}
			return modelInfoToBench(model.Info())
		},
		Generate: func(ctx context.Context, prompt string, opts bench.GenerateOptions) (bench.Generation, error) {
			if err := ctx.Err(); err != nil || model == nil {
				return bench.Generation{}, err
			}
			text, err := model.Generate(prompt, toModelGenerateOptions(opts)...)
			if err != nil {
				return bench.Generation{}, err
			}
			return bench.Generation{Text: text, Metrics: fromMlxMetrics(model.Metrics())}, nil
		},
		BenchPromptCache:        modelBenchPromptCache(model),
		BenchStateKVBlockWarm:   modelBenchStateKVBlockWarm(model),
		BenchKVRestore:          modelBenchKVRestore(model),
		BenchStateBundle:        modelBenchStateBundle(model),
		BenchProbeOverhead:      modelBenchProbeOverhead(model),
		BenchSpeculativeDecode:  modelBenchSpeculativeDecode(model, draft),
		BenchPromptLookupDecode: modelBenchPromptLookupDecode(model),
	}
}

// NewModelFastEvalRunnerWithSpeculativePair adapts a loaded speculative pair
// without dropping assistant-only native state.
func NewModelFastEvalRunnerWithSpeculativePair(pair *SpeculativePair) bench.Runner {
	if pair == nil {
		return NewModelFastEvalRunner(nil)
	}
	runner := NewModelFastEvalRunnerWithDraft(pair.Target, pair.Draft)
	runner.BenchSpeculativeDecode = modelBenchSpeculativePairDecode(pair)
	return runner
}

func toModelGenerateOptions(opts bench.GenerateOptions) []GenerateOption {
	// A single closure capturing opts replaces the per-knob With* allocs.
	// applyGenerateOptions only ever folds the slice into a GenerateConfig,
	// so we can collapse the eight branches into one functional step and
	// still feed the same shape downstream.
	sink, _ := opts.ProbeSink.(probe.Sink)
	return []GenerateOption{func(c *GenerateConfig) {
		c.MaxTokens = opts.MaxTokens
		c.Temperature = opts.Temperature
		if opts.TopK > 0 {
			c.TopK = opts.TopK
		}
		if opts.TopP > 0 {
			c.TopP = opts.TopP
		}
		if opts.MinP > 0 {
			c.MinP = opts.MinP
		}
		if len(opts.StopTokens) > 0 {
			c.StopTokens = opts.StopTokens
		}
		if opts.RepeatPenalty > 0 {
			c.RepeatPenalty = opts.RepeatPenalty
		}
		if sink != nil {
			c.ProbeSink = sink
		}
	}}
}

func modelBenchPromptCache(model *Model) func(context.Context, bench.Config, bench.GenerationSummary) bench.PromptCacheReport {
	return func(ctx context.Context, cfg bench.Config, _ bench.GenerationSummary) bench.PromptCacheReport {
		report := bench.PromptCacheReport{Attempted: true}
		start := time.Now()
		if err := model.WarmPromptCache(cfg.CachePrompt); err != nil {
			report.WarmDuration = time.Since(start)
			report.Error = err.Error()
			return report
		}
		report.WarmDuration = time.Since(start)
		if _, err := model.Generate(cfg.CachePrompt, toModelGenerateOptions(cfg.GenerateOptions(nil))...); err != nil {
			report.Error = err.Error()
			return report
		}
		metrics := fromMlxMetrics(model.Metrics())
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
}

func modelBenchStateKVBlockWarm(model *Model) func(context.Context, bench.Config, bench.GenerationSummary) bench.StateKVBlockWarmReport {
	return func(ctx context.Context, cfg bench.Config, baseline bench.GenerationSummary) bench.StateKVBlockWarmReport {
		report := bench.StateKVBlockWarmReport{
			Attempted: true,
			Source:    filestore.CodecFile,
		}
		blockSize := cfg.StateKVBlockSize
		if blockSize <= 0 {
			blockSize = blockcache.DefaultBlockSize
		}
		prefixTokens := cfg.StateKVPrefixTokens
		report.BlockSize = blockSize
		storePath, err := benchStateStorePath(cfg)
		if err != nil {
			report.Error = err.Error()
			return report
		}
		report.StorePath = storePath
		buildStart := time.Now()
		store, err := filestore.Create(ctx, storePath)
		if err != nil {
			report.BuildDuration = bench.NonZeroDuration(time.Since(buildStart))
			report.Error = err.Error()
			return report
		}
		session, err := model.NewSession()
		if err != nil {
			_ = store.Close()
			report.BuildDuration = bench.NonZeroDuration(time.Since(buildStart))
			report.Error = err.Error()
			return report
		}
		defer session.Close()
		if err := session.Prefill(cfg.CachePrompt); err != nil {
			_ = store.Close()
			report.BuildDuration = bench.NonZeroDuration(time.Since(buildStart))
			report.Error = err.Error()
			return report
		}
		bundle, err := session.SaveKVBlocksToState(ctx, store, kv.StateBlockOptions{
			BlockSize:  blockSize,
			KVEncoding: kv.EncodingNative,
		})
		if err != nil {
			_ = store.Close()
			report.BuildDuration = bench.NonZeroDuration(time.Since(buildStart))
			report.Error = err.Error()
			return report
		}
		if bundle == nil {
			_ = store.Close()
			report.BuildDuration = bench.NonZeroDuration(time.Since(buildStart))
			report.Error = "State KV block capture returned nil bundle"
			return report
		}
		if prefixTokens <= 0 {
			prefixTokens = bundle.TokenCount
		}
		if prefixTokens <= 0 {
			_ = store.Close()
			report.BuildDuration = bench.NonZeroDuration(time.Since(buildStart))
			report.Error = "State KV block bundle has no prefix tokens"
			return report
		}
		if err := store.Close(); err != nil {
			report.BuildDuration = bench.NonZeroDuration(time.Since(buildStart))
			report.Error = err.Error()
			return report
		}
		report.BuildDuration = bench.NonZeroDuration(time.Since(buildStart))
		report.BuildTokens = bundle.TokenCount
		if report.BuildDuration > 0 {
			report.BuildTokensPerSec = float64(report.BuildTokens) / report.BuildDuration.Seconds()
		}
		report.StoreBytes = benchFileSize(storePath)
		report.TotalBlocks = len(bundle.Blocks)
		report.PrefixTokensRestored = prefixTokens

		reader, err := filestore.Open(ctx, storePath)
		if err != nil {
			report.Error = err.Error()
			return report
		}
		defer reader.Close()
		counting := newBenchReadCountingStore(reader)
		restoreStart := time.Now()
		if err := model.WarmPromptCacheFromStateBlocks(ctx, counting, bundle, prefixTokens); err != nil {
			report.RestoreDuration = bench.NonZeroDuration(time.Since(restoreStart))
			report.BlocksRead = counting.UniqueReads()
			report.ChunksRead = counting.Reads()
			report.Error = err.Error()
			return report
		}
		report.RestoreDuration = bench.NonZeroDuration(time.Since(restoreStart))
		report.BlocksRead = counting.UniqueReads()
		report.ChunksRead = counting.Reads()

		generateStart := time.Now()
		if _, err := model.Generate(cfg.CachePrompt, toModelGenerateOptions(cfg.GenerateOptions(nil))...); err != nil {
			report.GenerateDuration = bench.NonZeroDuration(time.Since(generateStart))
			report.Error = err.Error()
			return report
		}
		report.GenerateDuration = bench.NonZeroDuration(time.Since(generateStart))
		metrics := fromMlxMetrics(model.Metrics())
		report.Metrics = metrics
		report.PromptTokensAvoided = metrics.PromptCacheHitTokens
		report.ReplayTokens = metrics.PromptCacheMissTokens
		if metrics.PromptTokens > 0 && prefixTokens >= metrics.PromptTokens && metrics.PromptCacheMissTokens > 0 {
			report.ExactFallbackReplayTokens = metrics.PromptCacheMissTokens
		}
		bench.PopulateStateKVBlockWarmBench(&report, baseline)
		return report
	}
}

func modelBenchKVRestore(model *Model) func(context.Context, bench.Config) bench.LatencyReport {
	return func(ctx context.Context, cfg bench.Config) bench.LatencyReport {
		report := bench.LatencyReport{Attempted: true}
		snapshot, err := model.CaptureKV(cfg.CachePrompt)
		if err != nil {
			report.Error = err.Error()
			return report
		}
		start := time.Now()
		session, err := model.NewSessionFromKV(snapshot)
		report.Duration = time.Since(start)
		if err != nil {
			report.Error = err.Error()
			return report
		}
		if session != nil {
			_ = session.Close()
		}
		return report
	}
}

func modelBenchStateBundle(model *Model) func(context.Context, bench.Config, bench.Info) bench.StateBundleReport {
	return func(ctx context.Context, cfg bench.Config, _ bench.Info) bench.StateBundleReport {
		report := bench.StateBundleReport{Attempted: true}
		snapshot, err := model.CaptureKV(cfg.CachePrompt)
		if err != nil {
			report.Error = err.Error()
			return report
		}
		start := time.Now()
		b, err := bundle.New(snapshot, bundle.Options{
			Model:     cfg.Model,
			ModelPath: cfg.ModelPath,
			Source:    modelInfoToBundle(model.Info()),
			Prompt:    cfg.CachePrompt,
			Sampler:   sampleFromGenerateConfig(toBenchGenerateOptions(cfg.GenerateOptions(nil))),
		})
		if err != nil {
			report.Duration = time.Since(start)
			report.Error = err.Error()
			return report
		}
		data := core.JSONMarshal(b)
		if !data.OK {
			report.Duration = time.Since(start)
			report.Error = fastEvalResultError(data).Error()
			return report
		}
		raw := data.Value.([]byte)
		var decoded bundle.Bundle
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
}

func modelBenchProbeOverhead(model *Model) func(context.Context, bench.Config, time.Duration) bench.ProbeReport {
	return func(ctx context.Context, cfg bench.Config, baseline time.Duration) bench.ProbeReport {
		report := bench.ProbeReport{Attempted: true}
		recorder := probe.NewRecorder()
		opts := cfg.GenerateOptions(recorder)
		start := time.Now()
		if _, err := model.Generate(cfg.Prompt, toModelGenerateOptions(opts)...); err != nil {
			report.Error = err.Error()
			return report
		}
		elapsed := time.Since(start)
		metrics := fromMlxMetrics(model.Metrics())
		events := recorder.Events()
		report.EventCount = len(events)
		report.KindCounts = make(map[string]int)
		report.Events = make([]any, len(events))
		for i, event := range events {
			report.KindCounts[string(event.Kind)]++
			report.Events[i] = event
		}
		report.Metrics = metrics
		if metrics.TotalDuration > 0 {
			report.Duration = metrics.TotalDuration
		} else {
			report.Duration = elapsed
		}
		if baseline > 0 {
			report.OverheadRatio = float64(report.Duration-baseline) / float64(baseline)
		}
		return report
	}
}

func modelBenchSpeculativeDecode(model, draft *Model) func(context.Context, bench.Config) bench.DecodeOptimisationReport {
	draftModel := draft
	if draftModel == nil {
		draftModel = model
	}
	return func(ctx context.Context, cfg bench.Config) bench.DecodeOptimisationReport {
		report := bench.DecodeOptimisationReport{Attempted: true}
		result, err := decode.Speculative(ctx, decode.SpeculativeConfig{
			Prompt:         cfg.Prompt,
			MaxTokens:      cfg.MaxTokens,
			DraftTokens:    cfg.SpeculativeDraftTokens,
			GenerateConfig: decode.GenerateConfig{MaxTokens: cfg.MaxTokens},
			TargetGenerate: benchModelDecodeGenerate(model),
			DraftGenerate:  benchModelDecodeGenerate(draftModel),
		})
		if err != nil {
			report.Error = err.Error()
			return report
		}
		report.Result = decodeResultToBench(result)
		report.Metrics = report.Result.Metrics
		return report
	}
}

func modelBenchSpeculativePairDecode(pair *SpeculativePair) func(context.Context, bench.Config) bench.DecodeOptimisationReport {
	return func(ctx context.Context, cfg bench.Config) bench.DecodeOptimisationReport {
		report := bench.DecodeOptimisationReport{Attempted: true}
		if pair == nil {
			report.Error = "mlx: speculative pair is nil"
			return report
		}
		result, err := pair.Generate(ctx, cfg.Prompt, SpeculativeDecodeConfig{
			MaxTokens:   cfg.MaxTokens,
			DraftTokens: cfg.SpeculativeDraftTokens,
			GenerateConfig: GenerateConfig{
				MaxTokens: cfg.MaxTokens,
			},
		})
		if err != nil {
			report.Error = err.Error()
			return report
		}
		report.Result = decodeResultToBench(result)
		report.Metrics = report.Result.Metrics
		return report
	}
}

func modelBenchPromptLookupDecode(model *Model) func(context.Context, bench.Config) bench.DecodeOptimisationReport {
	return func(ctx context.Context, cfg bench.Config) bench.DecodeOptimisationReport {
		report := bench.DecodeOptimisationReport{Attempted: true}
		if len(cfg.PromptLookupTokens) == 0 {
			report.Error = "prompt lookup tokens are required"
			return report
		}
		lookupTokens := make([]decode.Token, len(cfg.PromptLookupTokens))
		for i, id := range cfg.PromptLookupTokens {
			lookupTokens[i] = decode.Token{ID: id}
		}
		result, err := decode.PromptLookup(ctx, decode.PromptLookupConfig{
			Prompt:         cfg.Prompt,
			MaxTokens:      cfg.MaxTokens,
			GenerateConfig: decode.GenerateConfig{MaxTokens: cfg.MaxTokens},
			TargetGenerate: benchModelDecodeGenerate(model),
			LookupTokens:   lookupTokens,
		})
		if err != nil {
			report.Error = err.Error()
			return report
		}
		report.Result = decodeResultToBench(result)
		report.Metrics = report.Result.Metrics
		return report
	}
}

func decodeResultToBench(result decode.Result) bench.DecodeOptimisationResult {
	tokens := result.Tokens
	tokenIDs := make([]int32, len(tokens))
	// Index iteration avoids the per-step copy of the decode.Token (ID
	// + Text + any future fields) into the loop variable that
	// range-and-copy makes; only the int32 ID actually escapes.
	for i := range tokens {
		tokenIDs[i] = tokens[i].ID
	}
	return bench.DecodeOptimisationResult{
		Mode:   result.Mode,
		Prompt: result.Prompt,
		Text:   result.Text,
		Tokens: tokenIDs,
		Metrics: bench.DecodeOptimisationMetrics{
			TargetTokens:        result.Metrics.TargetTokens,
			DraftTokens:         result.Metrics.DraftTokens,
			LookupTokens:        result.Metrics.LookupTokens,
			AcceptedTokens:      result.Metrics.AcceptedTokens,
			RejectedTokens:      result.Metrics.RejectedTokens,
			EmittedTokens:       result.Metrics.EmittedTokens,
			AcceptanceRate:      result.Metrics.AcceptanceRate,
			TargetCalls:         result.Metrics.TargetCalls,
			DraftCalls:          result.Metrics.DraftCalls,
			Duration:            result.Metrics.Duration,
			TargetDuration:      result.Metrics.TargetDuration,
			DraftDuration:       result.Metrics.DraftDuration,
			VisibleTokensPerSec: decodeTokensPerSecond(result.Metrics.EmittedTokens, result.Metrics.Duration),
			TargetTokensPerSec:  decodeTokensPerSecond(result.Metrics.TargetTokens, result.Metrics.TargetDuration),
			DraftTokensPerSec:   decodeTokensPerSecond(result.Metrics.DraftTokens, result.Metrics.DraftDuration),
		},
	}
}

func decodeTokensPerSecond(tokens int, duration time.Duration) float64 {
	if tokens <= 0 || duration <= 0 {
		return 0
	}
	return float64(tokens) / duration.Seconds()
}

func benchModelDecodeGenerate(model *Model) decode.GenerateFunc {
	return modelDecodeGenerate(model, DefaultGenerateConfig())
}

func modelDecodeGenerate(model *Model, base GenerateConfig) decode.GenerateFunc {
	return func(ctx context.Context, prompt string, cfg decode.GenerateConfig) (decode.Generation, error) {
		if model == nil || model.model == nil {
			return decode.Generation{}, errModelDecodeNil
		}
		generateCfg := base
		if cfg.MaxTokens > 0 {
			generateCfg.MaxTokens = cfg.MaxTokens
		}
		// Pre-size tokens to MaxTokens — speculative/prompt-lookup decode
		// caps emitted tokens at MaxTokens, so a single make() avoids the
		// per-token append-grow doubling on every decoded step.
		tokens := make([]decode.Token, 0, generateCfg.MaxTokens)
		for token := range model.model.Generate(ctx, prompt, toMetalGenerateConfig(generateCfg)) {
			tokens = append(tokens, decode.Token{
				ID:   token.ID,
				Text: token.Text,
			})
		}
		if err := model.model.Err(); err != nil {
			return decode.Generation{}, err
		}
		return decode.Generation{Tokens: tokens, Text: decode.TokensText(tokens)}, nil
	}
}

func benchStateStorePath(cfg bench.Config) (string, error) {
	if path := core.Trim(cfg.StateKVBlockStorePath); path != "" {
		return path, nil
	}
	dirResult := core.MkdirTemp("", "go-mlx-state-kv-*")
	if !dirResult.OK {
		return "", core.E("mlx.benchStateStorePath", "create temp directory", fastEvalResultError(dirResult))
	}
	return core.PathJoin(dirResult.Value.(string), "blocks.mvlog"), nil
}

func benchFileSize(path string) int64 {
	stat := core.Stat(path)
	if !stat.OK {
		return 0
	}
	return stat.Value.(core.FsFileInfo).Size()
}

type benchReadCountingStore struct {
	store  state.Store
	reads  int
	unique map[int]struct{}
}

func newBenchReadCountingStore(store state.Store) *benchReadCountingStore {
	return &benchReadCountingStore{store: store, unique: map[int]struct{}{}}
}

func (s *benchReadCountingStore) Get(ctx context.Context, chunkID int) (string, error) {
	s.record(chunkID)
	return s.store.Get(ctx, chunkID)
}

func (s *benchReadCountingStore) Resolve(ctx context.Context, chunkID int) (state.Chunk, error) {
	s.record(chunkID)
	return state.Resolve(ctx, s.store, chunkID)
}

func (s *benchReadCountingStore) ResolveBytes(ctx context.Context, chunkID int) (state.Chunk, error) {
	s.record(chunkID)
	return state.ResolveBytes(ctx, s.store, chunkID)
}

func (s *benchReadCountingStore) Reads() int {
	if s == nil {
		return 0
	}
	return s.reads
}

func (s *benchReadCountingStore) UniqueReads() int {
	if s == nil {
		return 0
	}
	return len(s.unique)
}

func (s *benchReadCountingStore) record(chunkID int) {
	if s == nil {
		return
	}
	s.reads++
	if s.unique == nil {
		s.unique = map[int]struct{}{}
	}
	s.unique[chunkID] = struct{}{}
}

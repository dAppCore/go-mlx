// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference/bench"
	"dappco.re/go/inference/decode"
	memvid "dappco.re/go/inference/state"
	filestore "dappco.re/go/inference/state/filestore"
	"dappco.re/go/mlx/kv"
	"dappco.re/go/mlx/probe"
)

// NewModelFastEvalRunner adapts a loaded Model to bench.Runner with
// verb-shaped callbacks for each driver-specific bench section.
func NewModelFastEvalRunner(model *Model) bench.Runner {
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
		BenchMemvidKVBlockWarm:  modelBenchMemvidKVBlockWarm(model),
		BenchKVRestore:          modelBenchKVRestore(model),
		BenchStateBundle:        modelBenchStateBundle(model),
		BenchProbeOverhead:      modelBenchProbeOverhead(model),
		BenchSpeculativeDecode:  modelBenchSpeculativeDecode(model),
		BenchPromptLookupDecode: modelBenchPromptLookupDecode(model),
	}
}

func toModelGenerateOptions(opts bench.GenerateOptions) []GenerateOption {
	out := []GenerateOption{
		WithMaxTokens(opts.MaxTokens),
		WithTemperature(opts.Temperature),
	}
	if opts.TopK > 0 {
		out = append(out, WithTopK(opts.TopK))
	}
	if opts.TopP > 0 {
		out = append(out, WithTopP(opts.TopP))
	}
	if opts.MinP > 0 {
		out = append(out, WithMinP(opts.MinP))
	}
	if len(opts.StopTokens) > 0 {
		out = append(out, WithStopTokens(opts.StopTokens...))
	}
	if opts.RepeatPenalty > 0 {
		out = append(out, WithRepeatPenalty(opts.RepeatPenalty))
	}
	if sink, ok := opts.ProbeSink.(probe.Sink); ok && sink != nil {
		out = append(out, WithProbeSink(sink))
	}
	return out
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

func modelBenchMemvidKVBlockWarm(model *Model) func(context.Context, bench.Config, bench.GenerationSummary) bench.MemvidKVBlockWarmReport {
	return func(ctx context.Context, cfg bench.Config, baseline bench.GenerationSummary) bench.MemvidKVBlockWarmReport {
		report := bench.MemvidKVBlockWarmReport{
			Attempted: true,
			Source:    filestore.CodecFile,
		}
		blockSize := cfg.MemvidKVBlockSize
		if blockSize <= 0 {
			blockSize = DefaultCacheBlockSize
		}
		prefixTokens := cfg.MemvidKVPrefixTokens
		report.BlockSize = blockSize
		storePath, err := benchMemvidStorePath(cfg)
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
		bundle, err := session.SaveKVBlocksToMemvid(ctx, store, kv.MemvidBlockOptions{
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
			report.Error = "memvid KV block capture returned nil bundle"
			return report
		}
		if prefixTokens <= 0 {
			prefixTokens = bundle.TokenCount
		}
		if prefixTokens <= 0 {
			_ = store.Close()
			report.BuildDuration = bench.NonZeroDuration(time.Since(buildStart))
			report.Error = "memvid KV block bundle has no prefix tokens"
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
		if err := model.WarmPromptCacheFromMemvidBlocks(ctx, counting, bundle, prefixTokens); err != nil {
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
		bench.PopulateMemvidKVBlockWarmBench(&report, baseline)
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
		bundle, err := NewStateBundle(snapshot, StateBundleOptions{
			Model:     cfg.Model,
			ModelPath: cfg.ModelPath,
			ModelInfo: model.Info(),
			Prompt:    cfg.CachePrompt,
			Sampler:   toBenchGenerateOptions(cfg.GenerateOptions(nil)),
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

func modelBenchSpeculativeDecode(model *Model) func(context.Context, bench.Config) bench.DecodeOptimisationReport {
	return func(ctx context.Context, cfg bench.Config) bench.DecodeOptimisationReport {
		report := bench.DecodeOptimisationReport{Attempted: true}
		result, err := decode.Speculative(ctx, decode.SpeculativeConfig{
			Prompt:         cfg.Prompt,
			MaxTokens:      cfg.MaxTokens,
			DraftTokens:    cfg.SpeculativeDraftTokens,
			GenerateConfig: decode.GenerateConfig{MaxTokens: cfg.MaxTokens},
			TargetGenerate: benchModelDecodeGenerate(model),
			DraftGenerate:  benchModelDecodeGenerate(model),
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
	tokenIDs := make([]int32, len(result.Tokens))
	for i, tok := range result.Tokens {
		tokenIDs[i] = tok.ID
	}
	return bench.DecodeOptimisationResult{
		Mode:   result.Mode,
		Prompt: result.Prompt,
		Text:   result.Text,
		Tokens: tokenIDs,
		Metrics: bench.DecodeOptimisationMetrics{
			TargetTokens:   result.Metrics.TargetTokens,
			DraftTokens:    result.Metrics.DraftTokens,
			LookupTokens:   result.Metrics.LookupTokens,
			AcceptedTokens: result.Metrics.AcceptedTokens,
			RejectedTokens: result.Metrics.RejectedTokens,
			EmittedTokens:  result.Metrics.EmittedTokens,
			AcceptanceRate: result.Metrics.AcceptanceRate,
			TargetCalls:    result.Metrics.TargetCalls,
			DraftCalls:     result.Metrics.DraftCalls,
			Duration:       result.Metrics.Duration,
			TargetDuration: result.Metrics.TargetDuration,
			DraftDuration:  result.Metrics.DraftDuration,
		},
	}
}

func benchModelDecodeGenerate(model *Model) decode.GenerateFunc {
	return func(ctx context.Context, prompt string, cfg decode.GenerateConfig) (decode.Generation, error) {
		if model == nil {
			return decode.Generation{}, core.NewError("mlx: bench decode runner has nil model")
		}
		opts := []GenerateOption{WithMaxTokens(cfg.MaxTokens)}
		text, err := model.Generate(prompt, opts...)
		if err != nil {
			return decode.Generation{}, err
		}
		return decode.Generation{Text: text}, nil
	}
}

func benchMemvidStorePath(cfg bench.Config) (string, error) {
	if path := core.Trim(cfg.MemvidKVBlockStorePath); path != "" {
		return path, nil
	}
	dirResult := core.MkdirTemp("", "go-mlx-memvid-kv-*")
	if !dirResult.OK {
		return "", core.E("mlx.benchMemvidStorePath", "create temp directory", fastEvalResultError(dirResult))
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
	store  memvid.Store
	reads  int
	unique map[int]struct{}
}

func newBenchReadCountingStore(store memvid.Store) *benchReadCountingStore {
	return &benchReadCountingStore{store: store, unique: map[int]struct{}{}}
}

func (s *benchReadCountingStore) Get(ctx context.Context, chunkID int) (string, error) {
	s.record(chunkID)
	return s.store.Get(ctx, chunkID)
}

func (s *benchReadCountingStore) Resolve(ctx context.Context, chunkID int) (memvid.Chunk, error) {
	s.record(chunkID)
	return memvid.Resolve(ctx, s.store, chunkID)
}

func (s *benchReadCountingStore) ResolveBytes(ctx context.Context, chunkID int) (memvid.Chunk, error) {
	s.record(chunkID)
	return memvid.ResolveBytes(ctx, s.store, chunkID)
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

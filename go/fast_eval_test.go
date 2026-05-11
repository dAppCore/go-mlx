// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"
	"testing"
	"time"

	core "dappco.re/go"
	memvid "dappco.re/go/inference/state"
	filestore "dappco.re/go/inference/state/filestore"
	"dappco.re/go/mlx/kv"
	"dappco.re/go/mlx/internal/metal"
)

func TestNewModelFastEvalRunner_ForwardsModelAndCancellation_Good(t *testing.T) {
	native := &fakeNativeModel{
		info:   metal.ModelInfo{Architecture: "qwen3", ContextLength: 1024},
		tokens: []metal.Token{{ID: 1, Text: "ok"}},
		metrics: metal.Metrics{
			PromptTokens:    3,
			GeneratedTokens: 1,
		},
		kvSnapshot: &metal.KVSnapshot{
			Version:      metal.KVSnapshotVersion,
			Architecture: "qwen3",
			Tokens:       []int32{1},
			NumLayers:    1,
			NumHeads:     1,
			SeqLen:       1,
			HeadDim:      1,
			Layers: []metal.KVLayerSnapshot{{
				Layer: 0,
				Heads: []metal.KVHeadSnapshot{{
					Key:        []float32{1},
					Value:      []float32{2},
					KeyBytes:   []byte{1, 2},
					ValueBytes: []byte{3, 4},
					KeyDType:   metal.DTypeFloat16,
					ValueDType: metal.DTypeBFloat16,
				}},
			}},
		},
	}
	model := &Model{model: native}
	runner := NewModelFastEvalRunner(model)

	if info := runner.Info(context.Background()); info.Architecture != "qwen3" || info.ContextLength != 1024 {
		t.Fatalf("Info() = %+v, want qwen3 context", info)
	}
	generation, err := runner.Generate(context.Background(), "prompt", GenerateConfig{MaxTokens: 1})
	if err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	if generation.Text != "ok" || generation.Metrics.PromptTokens != 3 {
		t.Fatalf("generation = %+v, want forwarded text and metrics", generation)
	}
	if err := runner.WarmPromptCache(context.Background(), "stable"); err != nil {
		t.Fatalf("WarmPromptCache() error = %v", err)
	}
	if native.warmPrompt != "stable" {
		t.Fatalf("warmPrompt = %q, want stable", native.warmPrompt)
	}
	snapshot, err := runner.CaptureKV(context.Background(), "prompt")
	if err != nil {
		t.Fatalf("CaptureKV() error = %v", err)
	}
	if snapshot == nil || snapshot.Architecture != "qwen3" || len(snapshot.Layers) != 1 {
		t.Fatalf("snapshot = %+v, want converted KV snapshot", snapshot)
	}
	rawOnly, err := runner.CaptureKVWithOptions(context.Background(), "prompt", kv.CaptureOptions{RawKVOnly: true})
	if err != nil {
		t.Fatalf("CaptureKVWithOptions(raw) error = %v", err)
	}
	head := rawOnly.Layers[0].Heads[0]
	if len(head.Key) != 0 || head.KeyDType != "float16" || len(head.KeyBytes) == 0 {
		t.Fatalf("raw-only head = %+v, want dtype bytes without float32 tensors", head)
	}

	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	if info := runner.Info(cancelled); info.Architecture != "" {
		t.Fatalf("Info(cancelled) = %+v, want zero", info)
	}
	if _, err := runner.Generate(cancelled, "prompt", GenerateConfig{}); err != context.Canceled {
		t.Fatalf("Generate(cancelled) error = %v, want context.Canceled", err)
	}
	if err := runner.WarmPromptCache(cancelled, "prompt"); err != context.Canceled {
		t.Fatalf("WarmPromptCache(cancelled) error = %v, want context.Canceled", err)
	}
	if _, err := runner.CaptureKV(cancelled, "prompt"); err != context.Canceled {
		t.Fatalf("CaptureKV(cancelled) error = %v, want context.Canceled", err)
	}
	if _, err := runner.CaptureKVWithOptions(cancelled, "prompt", kv.CaptureOptions{}); err != context.Canceled {
		t.Fatalf("CaptureKVWithOptions(cancelled) error = %v, want context.Canceled", err)
	}
}

func TestRunFastEval_AggregatesGenerationCacheRestoreAndProbes_Good(t *testing.T) {
	calls := 0
	warmed := false
	restored := false
	runner := FastEvalRunner{
		Info: func(context.Context) ModelInfo {
			return ModelInfo{Architecture: "gemma4_text", NumLayers: 4, QuantBits: 4, ContextLength: 8192}
		},
		Generate: func(_ context.Context, prompt string, cfg GenerateConfig) (FastEvalGeneration, error) {
			calls++
			metrics := Metrics{
				PromptTokens:          10,
				GeneratedTokens:       cfg.MaxTokens,
				PrefillDuration:       100 * time.Millisecond,
				DecodeDuration:        50 * time.Millisecond,
				TotalDuration:         150 * time.Millisecond,
				PrefillTokensPerSec:   100,
				DecodeTokensPerSec:    40,
				PeakMemoryBytes:       2048,
				ActiveMemoryBytes:     1024,
				PromptCacheMisses:     1,
				PromptCacheMissTokens: 10,
			}
			if warmed && prompt == "stable prefix" {
				metrics.PromptCacheHits = 1
				metrics.PromptCacheMisses = 0
				metrics.PromptCacheHitTokens = 10
				metrics.PromptCacheMissTokens = 0
				metrics.PromptCacheRestoreDuration = 2 * time.Millisecond
				metrics.PrefillTokensPerSec = 250
			}
			if cfg.ProbeSink != nil {
				cfg.ProbeSink.EmitProbe(ProbeEvent{Kind: ProbeEventToken, Phase: ProbePhaseDecode, Step: 0})
				cfg.ProbeSink.EmitProbe(ProbeEvent{Kind: ProbeEventMemoryPressure, Phase: ProbePhaseDecode, Step: 0})
			}
			return FastEvalGeneration{Text: "ok", Metrics: metrics}, nil
		},
		WarmPromptCache: func(_ context.Context, prompt string) error {
			if prompt != "stable prefix" {
				t.Fatalf("WarmPromptCache prompt = %q, want stable prefix", prompt)
			}
			warmed = true
			return nil
		},
		CaptureKV: func(_ context.Context, prompt string) (*kv.Snapshot, error) {
			if prompt == "" {
				t.Fatal("CaptureKV received empty prompt")
			}
			return fastEvalTestSnapshot(), nil
		},
		RestoreKV: func(_ context.Context, snapshot *kv.Snapshot) error {
			if snapshot == nil {
				t.Fatal("RestoreKV received nil snapshot")
			}
			restored = true
			return nil
		},
	}

	report, err := RunFastEval(context.Background(), runner, FastEvalConfig{
		Model:                       "demo",
		Prompt:                      "baseline prompt",
		CachePrompt:                 "stable prefix",
		MaxTokens:                   3,
		Runs:                        1,
		IncludePromptCache:          true,
		IncludeKVRestore:            true,
		IncludeStateBundleRoundTrip: true,
		IncludeProbeOverhead:        true,
	})
	if err != nil {
		t.Fatalf("RunFastEval() error = %v", err)
	}
	if report.Model != "demo" || report.ModelInfo.Architecture != "gemma4_text" {
		t.Fatalf("model report = %+v info=%+v", report.Model, report.ModelInfo)
	}
	if report.Generation.PrefillTokensPerSec != 100 || report.Generation.DecodeTokensPerSec != 40 {
		t.Fatalf("generation summary = %+v", report.Generation)
	}
	if report.PromptCache.Hits != 1 || report.PromptCache.HitRate != 1 {
		t.Fatalf("prompt cache report = %+v, want hit rate 1", report.PromptCache)
	}
	if !report.KVRestore.Attempted || !restored {
		t.Fatalf("restore report = %+v restored=%v", report.KVRestore, restored)
	}
	if !report.StateBundle.Attempted || report.StateBundle.Bytes == 0 {
		t.Fatalf("state bundle report = %+v, want round-trip bytes", report.StateBundle)
	}
	if report.Probes.EventCount != 2 {
		t.Fatalf("probe event count = %d, want 2", report.Probes.EventCount)
	}
	if !report.Quality.Checks[0].Pass {
		t.Fatalf("quality checks = %+v, want non-empty output pass", report.Quality.Checks)
	}
	if calls != 3 {
		t.Fatalf("Generate calls = %d, want baseline/cache/probe", calls)
	}
}

func TestRunFastEval_MemvidKVBlockWarmCacheReport_Good(t *testing.T) {
	warmedFromMemvid := false
	rawOnlyCapture := false
	storePath := core.PathJoin(t.TempDir(), "kv-blocks.mvlog")
	runner := FastEvalRunner{
		Generate: func(_ context.Context, prompt string, cfg GenerateConfig) (FastEvalGeneration, error) {
			metrics := Metrics{
				PromptTokens:          3,
				GeneratedTokens:       cfg.MaxTokens,
				PrefillDuration:       100 * time.Millisecond,
				PromptCacheMisses:     1,
				PromptCacheMissTokens: 3,
				PeakMemoryBytes:       2048,
			}
			if warmedFromMemvid && prompt == "stable prefix" {
				metrics.PromptCacheHits = 1
				metrics.PromptCacheMisses = 0
				metrics.PromptCacheHitTokens = 2
				metrics.PromptCacheMissTokens = 1
				metrics.PromptCacheRestoreDuration = time.Millisecond
			}
			return FastEvalGeneration{Text: "ok", Metrics: metrics}, nil
		},
		CaptureKV: func(context.Context, string) (*kv.Snapshot, error) {
			return fastEvalTestSnapshot(), nil
		},
		CaptureKVWithOptions: func(_ context.Context, _ string, opts kv.CaptureOptions) (*kv.Snapshot, error) {
			rawOnlyCapture = opts.RawKVOnly
			return fastEvalTestSnapshot(), nil
		},
		WarmPromptCacheFromMemvidBlocks: func(ctx context.Context, store memvid.Store, bundle *kv.MemvidBlockBundle, prefixTokens int) error {
			if bundle.KVEncoding != kv.EncodingNative {
				t.Fatalf("memvid warm bundle encoding = %q, want native", bundle.KVEncoding)
			}
			snapshot, err := kv.LoadPrefixFromMemvidBlocks(ctx, store, bundle, prefixTokens)
			if err != nil {
				return err
			}
			if snapshot.SeqLen != 3 || len(snapshot.Logits) != 0 {
				t.Fatalf("memvid warm snapshot = %+v, want full three-token no-logit prefix", snapshot)
			}
			warmedFromMemvid = true
			return nil
		},
	}

	report, err := RunFastEval(context.Background(), runner, FastEvalConfig{
		Prompt:                      "baseline prompt",
		CachePrompt:                 "stable prefix",
		MaxTokens:                   2,
		Runs:                        1,
		IncludeMemvidKVBlockWarm:    true,
		MemvidKVBlockSize:           2,
		MemvidKVPrefixTokens:        3,
		MemvidKVBlockStorePath:      storePath,
		IncludePromptCache:          false,
		IncludeKVRestore:            false,
		IncludeStateBundleRoundTrip: false,
		IncludeProbeOverhead:        false,
	})
	if err != nil {
		t.Fatalf("RunFastEval() error = %v", err)
	}
	if !report.MemvidKVBlockWarm.Attempted || report.MemvidKVBlockWarm.Source != filestore.CodecFile {
		t.Fatalf("memvid cache report = %+v, want attempted file source", report.MemvidKVBlockWarm)
	}
	if !rawOnlyCapture {
		t.Fatal("CaptureKVWithOptions RawKVOnly = false, want raw-only memvid capture")
	}
	if report.MemvidKVBlockWarm.StorePath != storePath || report.MemvidKVBlockWarm.StoreBytes <= 0 {
		t.Fatalf("memvid cache store = path %q bytes %d, want file-backed store", report.MemvidKVBlockWarm.StorePath, report.MemvidKVBlockWarm.StoreBytes)
	}
	if report.MemvidKVBlockWarm.BlocksRead != 2 || report.MemvidKVBlockWarm.ChunksRead != 2 {
		t.Fatalf("memvid cache reads = blocks %d chunks %d, want 2/2", report.MemvidKVBlockWarm.BlocksRead, report.MemvidKVBlockWarm.ChunksRead)
	}
	if report.MemvidKVBlockWarm.PrefixTokensRestored != 3 || report.MemvidKVBlockWarm.PromptTokensAvoided != 2 || report.MemvidKVBlockWarm.ExactFallbackReplayTokens != 1 {
		t.Fatalf("memvid cache tokens = %+v, want restored=3 avoided=2 exact-replay=1", report.MemvidKVBlockWarm)
	}
	if report.MemvidKVBlockWarm.RestoreDuration <= 0 || report.MemvidKVBlockWarm.Metrics.PromptCacheHitTokens != 2 {
		t.Fatalf("memvid cache timing/metrics = %+v", report.MemvidKVBlockWarm)
	}
	if report.MemvidKVBlockWarm.BuildDuration <= 0 || report.MemvidKVBlockWarm.BuildTokens != 3 || report.MemvidKVBlockWarm.BuildTokensPerSec <= 0 {
		t.Fatalf("memvid build report = %+v, want build duration/tokens", report.MemvidKVBlockWarm)
	}
	if report.MemvidKVBlockWarm.BaselinePrefillDuration != 100*time.Millisecond || report.MemvidKVBlockWarm.BuildAmortizationQuestions <= 0 || report.MemvidKVBlockWarm.BreakEvenQuestions <= 0 {
		t.Fatalf("memvid amortisation report = %+v, want baseline and break-even questions", report.MemvidKVBlockWarm)
	}
	if report.MemvidKVBlockWarm.RestoreSpeedup <= 0 || report.MemvidKVBlockWarm.MemoryPeakBytes != 2048 {
		t.Fatalf("memvid restore speedup/memory = %+v, want speedup and peak memory", report.MemvidKVBlockWarm)
	}
}

func TestRunFastEval_MemvidKVBlockWarmStreamingCaptureDefaultsPrefix_Good(t *testing.T) {
	streamed := false
	warmedFromMemvid := false
	prefixTokensSeen := 0
	storePath := core.PathJoin(t.TempDir(), "streamed-kv-blocks.mvlog")
	runner := FastEvalRunner{
		Generate: func(_ context.Context, prompt string, cfg GenerateConfig) (FastEvalGeneration, error) {
			metrics := Metrics{PromptTokens: 3, GeneratedTokens: cfg.MaxTokens}
			if warmedFromMemvid && prompt == "stable prefix" {
				metrics.PromptCacheHitTokens = 3
			}
			return FastEvalGeneration{Text: "ok", Metrics: metrics}, nil
		},
		CaptureKV: func(context.Context, string) (*kv.Snapshot, error) {
			t.Fatal("CaptureKV should not run for streaming memvid block capture")
			return nil, nil
		},
		CaptureKVBlocksToMemvid: func(ctx context.Context, _ string, store memvid.Writer, opts kv.MemvidBlockOptions) (*kv.MemvidBlockBundle, error) {
			streamed = true
			return fastEvalTestSnapshot().SaveMemvidBlocks(ctx, store, opts)
		},
		WarmPromptCacheFromMemvidBlocks: func(ctx context.Context, store memvid.Store, bundle *kv.MemvidBlockBundle, prefixTokens int) error {
			prefixTokensSeen = prefixTokens
			snapshot, err := kv.LoadPrefixFromMemvidBlocks(ctx, store, bundle, prefixTokens)
			if err != nil {
				return err
			}
			if snapshot.SeqLen != 3 {
				t.Fatalf("streamed memvid warm snapshot seqLen = %d, want 3", snapshot.SeqLen)
			}
			warmedFromMemvid = true
			return nil
		},
	}

	report, err := RunFastEval(context.Background(), runner, FastEvalConfig{
		Prompt:                   "baseline prompt",
		CachePrompt:              "stable prefix",
		MaxTokens:                2,
		Runs:                     1,
		IncludeMemvidKVBlockWarm: true,
		MemvidKVBlockSize:        2,
		MemvidKVBlockStorePath:   storePath,
	})
	if err != nil {
		t.Fatalf("RunFastEval() error = %v", err)
	}
	if !streamed || !warmedFromMemvid {
		t.Fatalf("streamed=%v warmed=%v, want streaming capture and memvid warm", streamed, warmedFromMemvid)
	}
	if prefixTokensSeen != 3 || report.MemvidKVBlockWarm.PrefixTokensRestored != 3 {
		t.Fatalf("prefix tokens = seen %d report %d, want 3 from streamed bundle", prefixTokensSeen, report.MemvidKVBlockWarm.PrefixTokensRestored)
	}
	if report.MemvidKVBlockWarm.StorePath != storePath || report.MemvidKVBlockWarm.StoreBytes <= 0 {
		t.Fatalf("memvid streaming store = path %q bytes %d, want file-backed store", report.MemvidKVBlockWarm.StorePath, report.MemvidKVBlockWarm.StoreBytes)
	}
}

func TestRunFastEval_MemvidKVBlockWarm_Bad(t *testing.T) {
	cfg := normalizeFastEvalConfig(FastEvalConfig{
		Prompt:                 "baseline prompt",
		CachePrompt:            "stable prefix",
		MaxTokens:              1,
		Runs:                   1,
		MemvidKVBlockStorePath: core.PathJoin(t.TempDir(), "kv-blocks.mvlog"),
	})
	if report := runFastEvalMemvidKVBlockWarm(context.Background(), FastEvalRunner{}, nil, cfg); report.Error == "" {
		t.Fatalf("memvid warm without snapshot report = %+v", report)
	}
	if report := runFastEvalMemvidKVBlockWarm(context.Background(), FastEvalRunner{}, fastEvalTestSnapshot(), cfg); report.Error == "" {
		t.Fatalf("memvid warm unsupported runner report = %+v", report)
	}
	nilBundleRunner := FastEvalRunner{
		CaptureKVBlocksToMemvid: func(context.Context, string, memvid.Writer, kv.MemvidBlockOptions) (*kv.MemvidBlockBundle, error) {
			return nil, nil
		},
		WarmPromptCacheFromMemvidBlocks: func(context.Context, memvid.Store, *kv.MemvidBlockBundle, int) error {
			return nil
		},
	}
	if report := runFastEvalMemvidKVBlockWarm(context.Background(), nilBundleRunner, nil, cfg); report.Error == "" {
		t.Fatalf("memvid warm nil bundle report = %+v", report)
	}
	emptyBundleRunner := nilBundleRunner
	emptyBundleRunner.CaptureKVBlocksToMemvid = func(context.Context, string, memvid.Writer, kv.MemvidBlockOptions) (*kv.MemvidBlockBundle, error) {
		return &kv.MemvidBlockBundle{}, nil
	}
	if report := runFastEvalMemvidKVBlockWarm(context.Background(), emptyBundleRunner, nil, cfg); report.Error == "" {
		t.Fatalf("memvid warm empty bundle report = %+v", report)
	}

	warmErrRunner := FastEvalRunner{
		WarmPromptCacheFromMemvidBlocks: func(context.Context, memvid.Store, *kv.MemvidBlockBundle, int) error {
			return core.NewError("warm failed")
		},
		Generate: func(context.Context, string, GenerateConfig) (FastEvalGeneration, error) {
			return FastEvalGeneration{Text: "unused"}, nil
		},
	}
	if report := runFastEvalMemvidKVBlockWarm(context.Background(), warmErrRunner, fastEvalTestSnapshot(), cfg); report.Error == "" || report.RestoreDuration <= 0 {
		t.Fatalf("memvid warm failure report = %+v", report)
	}

	generateErrRunner := FastEvalRunner{
		WarmPromptCacheFromMemvidBlocks: func(context.Context, memvid.Store, *kv.MemvidBlockBundle, int) error {
			return nil
		},
		Generate: func(context.Context, string, GenerateConfig) (FastEvalGeneration, error) {
			return FastEvalGeneration{}, core.NewError("generate failed")
		},
	}
	if report := runFastEvalMemvidKVBlockWarm(context.Background(), generateErrRunner, fastEvalTestSnapshot(), cfg); report.Error == "" || report.GenerateDuration <= 0 {
		t.Fatalf("memvid warm generate failure report = %+v", report)
	}
}

func TestFastEvalMemvidHelpers_Good(t *testing.T) {
	explicit := core.PathJoin(t.TempDir(), "explicit.mvlog")
	if got, err := fastEvalMemvidKVBlockStorePath(FastEvalConfig{MemvidKVBlockStorePath: " " + explicit + " "}); err != nil || got != explicit {
		t.Fatalf("fastEvalMemvidKVBlockStorePath(explicit) = %q/%v, want %q", got, err, explicit)
	}
	generated, err := fastEvalMemvidKVBlockStorePath(FastEvalConfig{})
	if err != nil {
		t.Fatalf("fastEvalMemvidKVBlockStorePath(temp) error = %v", err)
	}
	if core.PathBase(generated) != "blocks.mvlog" {
		t.Fatalf("generated memvid store path = %q, want blocks.mvlog", generated)
	}
	if fastEvalFileSize(core.PathJoin(t.TempDir(), "missing")) != 0 {
		t.Fatal("fastEvalFileSize(missing) != 0")
	}
	if (&memvidReadCountingStore{}).Reads() != 0 || (&memvidReadCountingStore{}).UniqueReads() != 0 {
		t.Fatal("empty read-counting store returned non-zero counts")
	}
	store := memvid.NewInMemoryStore(map[int]string{1: "one"})
	counting := newMemvidReadCountingStore(store)
	if text, err := counting.Get(context.Background(), 1); err != nil || text != "one" {
		t.Fatalf("counting Get() = %q/%v, want one/nil", text, err)
	}
	if _, err := counting.Resolve(context.Background(), 1); err != nil {
		t.Fatalf("counting Resolve() error = %v", err)
	}
	if counting.Reads() != 2 || counting.UniqueReads() != 1 {
		t.Fatalf("counting reads = %d unique = %d, want 2/1", counting.Reads(), counting.UniqueReads())
	}

	binary := &fastEvalBinaryCountingStore{
		chunk: memvid.Chunk{Ref: memvid.ChunkRef{ChunkID: 7}, Data: []byte{0, 1, 2, 3}},
	}
	counting = newMemvidReadCountingStore(binary)
	chunk, err := counting.ResolveBytes(context.Background(), 7)
	if err != nil {
		t.Fatalf("counting ResolveBytes() error = %v", err)
	}
	if len(chunk.Data) != 4 || binary.binaryReads != 1 || binary.textReads != 0 || binary.resolveReads != 0 {
		t.Fatalf("binary counting chunk=%+v binary=%d text=%d resolve=%d, want direct binary read", chunk, binary.binaryReads, binary.textReads, binary.resolveReads)
	}
	if counting.Reads() != 1 || counting.UniqueReads() != 1 {
		t.Fatalf("binary counting reads = %d unique = %d, want 1/1", counting.Reads(), counting.UniqueReads())
	}
}

func TestRunFastEval_DecodeOptimisationsReport_Good(t *testing.T) {
	runner := FastEvalRunner{
		Generate: func(_ context.Context, _ string, cfg GenerateConfig) (FastEvalGeneration, error) {
			return FastEvalGeneration{
				Tokens: []Token{{ID: 1, Text: "A"}, {ID: 2, Text: "B"}, {ID: 4, Text: "D"}},
				Metrics: Metrics{
					PromptTokens:        2,
					GeneratedTokens:     cfg.MaxTokens,
					PrefillTokensPerSec: 20,
					DecodeTokensPerSec:  10,
				},
			}, nil
		},
		DraftGenerate: func(_ context.Context, _ string, _ GenerateConfig) (FastEvalGeneration, error) {
			return FastEvalGeneration{
				Tokens:  []Token{{ID: 1, Text: "A"}, {ID: 2, Text: "B"}, {ID: 3, Text: "C"}},
				Metrics: Metrics{GeneratedTokens: 3},
			}, nil
		},
	}

	report, err := RunFastEval(context.Background(), runner, FastEvalConfig{
		Prompt:                    "baseline",
		MaxTokens:                 3,
		Runs:                      1,
		IncludeSpeculativeDecode:  true,
		SpeculativeDraftTokens:    3,
		IncludePromptLookupDecode: true,
		PromptLookupTokens:        []Token{{ID: 1, Text: "A"}, {ID: 9, Text: "?"}, {ID: 4, Text: "D"}},
	})
	if err != nil {
		t.Fatalf("RunFastEval() error = %v", err)
	}
	if !report.SpeculativeDecode.Attempted || report.SpeculativeDecode.Metrics.AcceptedTokens != 2 || report.SpeculativeDecode.Metrics.RejectedTokens != 1 {
		t.Fatalf("speculative report = %+v, want attempted 2/1 acceptance", report.SpeculativeDecode)
	}
	if !report.PromptLookupDecode.Attempted || report.PromptLookupDecode.Metrics.AcceptedTokens != 2 || report.PromptLookupDecode.Metrics.RejectedTokens != 1 {
		t.Fatalf("prompt lookup report = %+v, want attempted 2/1 acceptance", report.PromptLookupDecode)
	}
}

func TestRunFastEval_DefaultsAndRequiredRunner_Bad(t *testing.T) {
	_, err := RunFastEval(context.Background(), FastEvalRunner{}, FastEvalConfig{})
	if err == nil {
		t.Fatal("expected missing runner error")
	}
}

func TestRunFastEval_DisabledOptionalSections_Ugly(t *testing.T) {
	runner := FastEvalRunner{
		Generate: func(_ context.Context, _ string, cfg GenerateConfig) (FastEvalGeneration, error) {
			return FastEvalGeneration{
				Text: "ok",
				Metrics: Metrics{
					PromptTokens:        1,
					GeneratedTokens:     cfg.MaxTokens,
					PrefillTokensPerSec: 1,
					DecodeTokensPerSec:  2,
				},
			}, nil
		},
	}

	report, err := RunFastEval(context.Background(), runner, FastEvalConfig{
		Prompt:                      "p",
		IncludePromptCache:          false,
		IncludeKVRestore:            false,
		IncludeStateBundleRoundTrip: false,
		IncludeProbeOverhead:        false,
	})
	if err != nil {
		t.Fatalf("RunFastEval() error = %v", err)
	}
	if report.PromptCache.Attempted || report.KVRestore.Attempted || report.StateBundle.Attempted || report.Probes.Attempted {
		t.Fatalf("optional reports should be disabled: cache=%+v restore=%+v bundle=%+v probes=%+v", report.PromptCache, report.KVRestore, report.StateBundle, report.Probes)
	}
}

func TestFastEval_DefaultFastEvalConfig_Good(t *testing.T) {
	cfg := DefaultFastEvalConfig()
	if cfg.MaxTokens <= 0 || cfg.Runs <= 0 || !cfg.IncludePromptCache || !cfg.IncludeProbeOverhead {
		t.Fatalf("DefaultFastEvalConfig() = %+v, want runnable defaults", cfg)
	}
}

func TestFastEval_RunFastEvalBench_Bad(t *testing.T) {
	_, err := RunFastEvalBench(context.Background(), nil, FastEvalConfig{})
	if err == nil {
		t.Fatal("expected nil model error")
	}
}

func TestFastEval_NewModelFastEvalRunner_Ugly(t *testing.T) {
	runner := NewModelFastEvalRunner(&Model{})
	if runner.Generate == nil || runner.WarmPromptCache == nil || runner.CaptureKV == nil || runner.RestoreKV == nil {
		t.Fatalf("runner = %+v, want complete model adapter", runner)
	}

	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	store := memvid.NewInMemoryStore(nil)
	if _, err := runner.CaptureKVBlocksToMemvid(cancelled, "prompt", store, kv.MemvidBlockOptions{}); err != context.Canceled {
		t.Fatalf("CaptureKVBlocksToMemvid(cancelled) = %v, want context.Canceled", err)
	}
	if _, err := runner.CaptureKVBlocksToMemvid(context.Background(), "prompt", store, kv.MemvidBlockOptions{}); err == nil {
		t.Fatal("expected nil model session error for CaptureKVBlocksToMemvid")
	}
	if err := runner.RestoreKV(cancelled, fastEvalTestSnapshot()); err != context.Canceled {
		t.Fatalf("RestoreKV(cancelled) = %v, want context.Canceled", err)
	}
	if err := runner.RestoreKV(context.Background(), fastEvalTestSnapshot()); err == nil {
		t.Fatal("expected nil model session error for RestoreKV")
	}
	if err := runner.WarmPromptCacheFromMemvidBlocks(cancelled, store, &kv.MemvidBlockBundle{}, 0); err != context.Canceled {
		t.Fatalf("WarmPromptCacheFromMemvidBlocks(cancelled) = %v, want context.Canceled", err)
	}
	if err := runner.WarmPromptCacheFromMemvidBlocks(context.Background(), store, &kv.MemvidBlockBundle{}, 0); err == nil {
		t.Fatal("expected nil model warm memvid error")
	}
	if _, err := runner.GenerateWithMemvidPrefix(cancelled, store, &kv.MemvidBlockBundle{}, 1, "suffix", GenerateConfig{}); err != context.Canceled {
		t.Fatalf("GenerateWithMemvidPrefix(cancelled) = %v, want context.Canceled", err)
	}
	if _, err := runner.GenerateWithMemvidPrefix(context.Background(), store, &kv.MemvidBlockBundle{}, 1, "suffix", GenerateConfig{}); err == nil {
		t.Fatal("expected nil model session error for GenerateWithMemvidPrefix")
	}
}

func TestFastEvalConfigAndOptions_Good(t *testing.T) {
	cfg := normalizeFastEvalConfig(FastEvalConfig{
		Model:         "m",
		Prompt:        "p",
		MaxTokens:     -1,
		Runs:          -1,
		TopK:          20,
		TopP:          0.9,
		MinP:          0.1,
		StopTokens:    []int32{1, 2},
		RepeatPenalty: 1.1,
	})
	if cfg.MaxTokens != DefaultFastEvalConfig().MaxTokens || cfg.Runs != DefaultFastEvalConfig().Runs || cfg.CachePrompt != "p" {
		t.Fatalf("normalizeFastEvalConfig() = %+v", cfg)
	}
	cfg.StopTokens[0] = 9
	normalized := normalizeFastEvalConfig(FastEvalConfig{Prompt: "p", MaxTokens: 1, Runs: 1, StopTokens: []int32{1}})
	if normalized.StopTokens[0] != 1 {
		t.Fatal("normalizeFastEvalConfig did not defensively copy stop tokens")
	}
	opts := fastEvalGenerateOptions(FastEvalConfig{
		MaxTokens:     4,
		Temperature:   0.1,
		TopK:          10,
		TopP:          0.8,
		MinP:          0.05,
		StopTokens:    []int32{2},
		RepeatPenalty: 1.2,
	}.generateConfig(NewProbeRecorder()))
	if len(opts) != 8 {
		t.Fatalf("fastEvalGenerateOptions len = %d, want 8", len(opts))
	}
}

func TestFastEvalOptionalErrorBranches_Bad(t *testing.T) {
	cfg := normalizeFastEvalConfig(FastEvalConfig{Prompt: "p", MaxTokens: 1, Runs: 1})
	if report := runFastEvalPromptCache(context.Background(), FastEvalRunner{}, cfg); !report.Attempted || report.Error == "" {
		t.Fatalf("prompt cache unsupported report = %+v", report)
	}
	wantErr := core.NewError("warm failed")
	runner := FastEvalRunner{
		WarmPromptCache: func(context.Context, string) error { return wantErr },
		Generate: func(context.Context, string, GenerateConfig) (FastEvalGeneration, error) {
			return FastEvalGeneration{}, nil
		},
	}
	if report := runFastEvalPromptCache(context.Background(), runner, cfg); report.Error == "" {
		t.Fatalf("prompt cache warm error report = %+v", report)
	}
	runner.WarmPromptCache = func(context.Context, string) error { return nil }
	runner.Generate = func(context.Context, string, GenerateConfig) (FastEvalGeneration, error) {
		return FastEvalGeneration{}, core.NewError("generate failed")
	}
	if report := runFastEvalPromptCache(context.Background(), runner, cfg); report.Error == "" {
		t.Fatalf("prompt cache generate error report = %+v", report)
	}

	if snapshot := runFastEvalCapture(context.Background(), FastEvalRunner{}, cfg); snapshot != nil {
		t.Fatalf("capture without runner = %+v, want nil", snapshot)
	}
	runner.CaptureKV = func(context.Context, string) (*kv.Snapshot, error) { return nil, core.NewError("capture failed") }
	if snapshot := runFastEvalCapture(context.Background(), runner, cfg); snapshot != nil {
		t.Fatalf("capture error = %+v, want nil", snapshot)
	}
	if report := runFastEvalRestore(context.Background(), FastEvalRunner{}, nil); report.Error == "" {
		t.Fatalf("restore nil report = %+v", report)
	}
	if report := runFastEvalRestore(context.Background(), FastEvalRunner{}, fastEvalTestSnapshot()); report.Error == "" {
		t.Fatalf("restore unsupported report = %+v", report)
	}
	if report := runFastEvalStateBundle(context.Background(), nil, cfg, ModelInfo{}); report.Error == "" {
		t.Fatalf("state bundle nil report = %+v", report)
	}
	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	if report := runFastEvalStateBundle(cancelled, fastEvalTestSnapshot(), cfg, ModelInfo{}); report.Error == "" {
		t.Fatalf("state bundle cancelled report = %+v", report)
	}
}

func TestFastEvalMoreOptionalErrorBranches_Bad(t *testing.T) {
	cfg := normalizeFastEvalConfig(FastEvalConfig{Prompt: "p", MaxTokens: 2, Runs: 1})
	wantErr := core.NewError("forced failure")

	if report := runFastEvalRestore(context.Background(), FastEvalRunner{
		RestoreKV: func(context.Context, *kv.Snapshot) error { return wantErr },
	}, fastEvalTestSnapshot()); report.Error == "" {
		t.Fatalf("restore error report = %+v", report)
	}
	if report := runFastEvalProbes(context.Background(), FastEvalRunner{
		Generate: func(context.Context, string, GenerateConfig) (FastEvalGeneration, error) {
			return FastEvalGeneration{}, wantErr
		},
	}, cfg, time.Millisecond); report.Error == "" {
		t.Fatalf("probe error report = %+v", report)
	}
	if report := runFastEvalSpeculativeDecode(context.Background(), FastEvalRunner{}, cfg); report.Error == "" {
		t.Fatalf("speculative unsupported report = %+v", report)
	}
	if report := runFastEvalSpeculativeDecode(context.Background(), FastEvalRunner{
		Generate: func(context.Context, string, GenerateConfig) (FastEvalGeneration, error) {
			return FastEvalGeneration{}, wantErr
		},
		DraftGenerate: func(context.Context, string, GenerateConfig) (FastEvalGeneration, error) {
			return FastEvalGeneration{Tokens: []Token{{ID: 1, Text: "x"}}}, nil
		},
	}, cfg); report.Error == "" {
		t.Fatalf("speculative generate error report = %+v", report)
	}
	if report := runFastEvalPromptLookupDecode(context.Background(), FastEvalRunner{}, cfg); report.Error == "" {
		t.Fatalf("prompt lookup missing tokens report = %+v", report)
	}
	cfg.PromptLookupTokens = []Token{{ID: 1, Text: "x"}}
	if report := runFastEvalPromptLookupDecode(context.Background(), FastEvalRunner{
		Generate: func(context.Context, string, GenerateConfig) (FastEvalGeneration, error) {
			return FastEvalGeneration{}, wantErr
		},
	}, cfg); report.Error == "" {
		t.Fatalf("prompt lookup generate error report = %+v", report)
	}
	decode, err := fastEvalDecodeGenerate(nil)(context.Background(), "p", GenerateConfig{})
	if err == nil || decode.Text != "" {
		t.Fatalf("fastEvalDecodeGenerate(nil) = %+v/%v, want error", decode, err)
	}
	if err := fastEvalResultError(core.Result{OK: true}); err != nil {
		t.Fatalf("fastEvalResultError(OK) = %v, want nil", err)
	}
	var counting memvidReadCountingStore
	counting.record(42)
	if counting.Reads() != 1 || counting.UniqueReads() != 1 {
		t.Fatalf("manual counting store reads = %d unique = %d, want 1/1", counting.Reads(), counting.UniqueReads())
	}
}

func TestFastEvalSummariesAndResults_Ugly(t *testing.T) {
	summary := summarizeFastEvalGenerations([]FastEvalGenerationSample{
		{
			Text:    "",
			Elapsed: 3 * time.Millisecond,
			Metrics: Metrics{
				PromptTokens:        2,
				GeneratedTokens:     0,
				PrefillTokensPerSec: 4,
				DecodeTokensPerSec:  6,
				PeakMemoryBytes:     10,
				ActiveMemoryBytes:   5,
			},
		},
		{
			Text: "ok",
			Metrics: Metrics{
				PromptTokens:        3,
				GeneratedTokens:     1,
				TotalDuration:       2 * time.Millisecond,
				PrefillTokensPerSec: 8,
				DecodeTokensPerSec:  10,
				PeakMemoryBytes:     8,
				ActiveMemoryBytes:   7,
			},
		},
	})
	if summary.Runs != 2 || summary.PromptTokens != 5 || summary.GeneratedTokens != 1 || summary.PrefillTokensPerSec != 6 || summary.DecodeTokensPerSec != 8 || summary.TotalDuration != 5*time.Millisecond {
		t.Fatalf("summary = %+v", summary)
	}
	checks := qualityChecks([]FastEvalGenerationSample{{Text: "", Metrics: Metrics{GeneratedTokens: 0}}})
	if checks[0].Pass || checks[1].Pass {
		t.Fatalf("empty quality checks = %+v, want failures", checks)
	}
	if got := boolScore(false); got != 0 {
		t.Fatalf("boolScore(false) = %f, want 0", got)
	}
	if err := fastEvalResultError(core.Result{Value: "bad", OK: false}); err == nil || !core.Contains(err.Error(), "core result failed") {
		t.Fatalf("fastEvalResultError(non-error) = %v", err)
	}
}

func fastEvalTestSnapshot() *kv.Snapshot {
	return &kv.Snapshot{
		Version:       kv.SnapshotVersion,
		Architecture:  "gemma4_text",
		Tokens:        []int32{1, 2, 3},
		TokenOffset:   3,
		NumLayers:     1,
		NumHeads:      1,
		SeqLen:        3,
		HeadDim:       2,
		NumQueryHeads: 1,
		Layers: []kv.LayerSnapshot{{
			Layer:      0,
			CacheIndex: 0,
			Heads: []kv.HeadSnapshot{{
				Key:   []float32{0.1, 0.2, 0.3, 0.4, 0.5, 0.6},
				Value: []float32{0.6, 0.5, 0.4, 0.3, 0.2, 0.1},
			}},
		}},
	}
}

type fastEvalBinaryCountingStore struct {
	chunk        memvid.Chunk
	textReads    int
	resolveReads int
	binaryReads  int
}

func (s *fastEvalBinaryCountingStore) Get(context.Context, int) (string, error) {
	s.textReads++
	return string(s.chunk.Data), nil
}

func (s *fastEvalBinaryCountingStore) Resolve(context.Context, int) (memvid.Chunk, error) {
	s.resolveReads++
	chunk := s.chunk
	chunk.Text = string(chunk.Data)
	chunk.Data = nil
	return chunk, nil
}

func (s *fastEvalBinaryCountingStore) ResolveBytes(context.Context, int) (memvid.Chunk, error) {
	s.binaryReads++
	return s.chunk, nil
}

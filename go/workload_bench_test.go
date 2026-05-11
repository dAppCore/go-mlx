// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"
	"testing"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference/quant/jang"
	memvid "dappco.re/go/inference/state"
	filestore "dappco.re/go/inference/state/filestore"
)

func TestRunWorkloadBench_AggregatesFastEvalAdapterAndPerplexity_Good(t *testing.T) {
	loadCalled := false
	fuseCalled := false
	evalCalled := false
	adapter := WorkloadAdapterInfo{
		Path:       "/adapters/qwen-lora",
		Name:       "qwen-lora",
		Rank:       16,
		Alpha:      32,
		TargetKeys: []string{"q_proj", "v_proj"},
	}
	runner := WorkloadBenchRunner{
		FastEval: FastEvalRunner{
			Info: func(context.Context) ModelInfo {
				return ModelInfo{Architecture: "qwen3", NumLayers: 28, HiddenSize: 3072, QuantBits: 4, ContextLength: 32768}
			},
			Generate: func(_ context.Context, _ string, cfg GenerateConfig) (FastEvalGeneration, error) {
				return FastEvalGeneration{
					Text: "ok",
					Metrics: Metrics{
						PromptTokens:         16,
						GeneratedTokens:      cfg.MaxTokens,
						PrefillDuration:      80 * time.Millisecond,
						DecodeDuration:       40 * time.Millisecond,
						TotalDuration:        120 * time.Millisecond,
						PrefillTokensPerSec:  200,
						DecodeTokensPerSec:   75,
						PeakMemoryBytes:      8 << 20,
						ActiveMemoryBytes:    4 << 20,
						PromptCacheHits:      1,
						PromptCacheHitTokens: 16,
					},
				}, nil
			},
			WarmPromptCache: func(context.Context, string) error { return nil },
			CaptureKV: func(context.Context, string) (*KVSnapshot, error) {
				return fastEvalTestSnapshot(), nil
			},
			RestoreKV: func(context.Context, *KVSnapshot) error { return nil },
		},
		LoadAdapter: func(_ context.Context, path string) (WorkloadAdapterInfo, error) {
			if path != adapter.Path {
				t.Fatalf("LoadAdapter path = %q, want %q", path, adapter.Path)
			}
			loadCalled = true
			return adapter, nil
		},
		FuseAdapter: func(_ context.Context, got WorkloadAdapterInfo) error {
			if got.Path != adapter.Path || got.Rank != adapter.Rank {
				t.Fatalf("FuseAdapter adapter = %+v, want %+v", got, adapter)
			}
			fuseCalled = true
			return nil
		},
		EvaluatePerplexity: func(_ context.Context, samples []WorkloadEvalSample) (WorkloadEvalMetrics, error) {
			if len(samples) != 2 {
				t.Fatalf("EvaluatePerplexity samples = %d, want 2", len(samples))
			}
			evalCalled = true
			return WorkloadEvalMetrics{
				Samples:    len(samples),
				Tokens:     42,
				Loss:       1.25,
				Perplexity: 3.49,
			}, nil
		},
	}

	report, err := RunWorkloadBench(context.Background(), runner, WorkloadBenchConfig{
		FastEval: FastEvalConfig{
			Model:                       "qwen",
			Prompt:                      "baseline",
			CachePrompt:                 "stable prefix",
			MaxTokens:                   4,
			Runs:                        1,
			IncludePromptCache:          true,
			IncludeKVRestore:            true,
			IncludeStateBundleRoundTrip: true,
			IncludeProbeOverhead:        false,
		},
		AdapterPath:         adapter.Path,
		IncludeAdapterLoad:  true,
		IncludeAdapterFuse:  true,
		IncludePerplexity:   true,
		IncludeKVCacheBench: true,
		QuantizationProfile: jang.BuildPackedProfile(&jang.Info{
			WeightFormat:     "mxtq",
			Profile:          "JANGTQ",
			Method:           "affine+mxtq",
			GroupSize:        64,
			BitsDefault:      2,
			RoutedExpertBits: 2,
			AttentionBits:    8,
		}),
		EvalSamples: []WorkloadEvalSample{
			{Prompt: "a", Response: "b"},
			{Text: "plain eval text"},
		},
	})
	if err != nil {
		t.Fatalf("RunWorkloadBench() error = %v", err)
	}
	if report.Version != WorkloadBenchReportVersion {
		t.Fatalf("Version = %d, want %d", report.Version, WorkloadBenchReportVersion)
	}
	if report.FastEval == nil || report.FastEval.Generation.PrefillTokensPerSec != 200 {
		t.Fatalf("FastEval = %+v, want populated fast eval report", report.FastEval)
	}
	if !loadCalled || !report.Adapter.Load.Attempted || report.Adapter.Load.Duration <= 0 {
		t.Fatalf("adapter load report = %+v loadCalled=%v", report.Adapter.Load, loadCalled)
	}
	if !fuseCalled || !report.Adapter.Fuse.Attempted || report.Adapter.Fuse.Duration <= 0 {
		t.Fatalf("adapter fuse report = %+v fuseCalled=%v", report.Adapter.Fuse, fuseCalled)
	}
	if report.Adapter.Adapter.Path != adapter.Path || len(report.Adapter.Adapter.TargetKeys) != 2 {
		t.Fatalf("adapter metadata = %+v, want cloned adapter metadata", report.Adapter.Adapter)
	}
	if !evalCalled || !report.Evaluation.Attempted || report.Evaluation.Metrics.Perplexity != 3.49 {
		t.Fatalf("evaluation report = %+v evalCalled=%v", report.Evaluation, evalCalled)
	}
	if report.KVCache.Version != KVCacheBenchReportVersion || report.KVCache.RecommendedMode == "" {
		t.Fatalf("KV cache report = %+v, want populated mode comparison", report.KVCache)
	}
	if report.QuantizationProfile == nil || report.QuantizationProfile.Type != "jangtq" || report.QuantizationProfile.RoleBits[string(jang.TensorRoleRoutedExpert)] != 2 {
		t.Fatalf("quantization profile = %+v, want JANGTQ bench metadata", report.QuantizationProfile)
	}
	if report.Summary.PrefillTokensPerSec != 200 || report.Summary.DecodeTokensPerSec != 75 || report.Summary.PeakMemoryBytes != 8<<20 {
		t.Fatalf("summary = %+v, want fast-eval throughput and memory mirrored", report.Summary)
	}
}

func TestRunWorkloadBench_UsesDatasetEvalReport_Good(t *testing.T) {
	runner := WorkloadBenchRunner{
		FastEval: FastEvalRunner{
			Generate: func(context.Context, string, GenerateConfig) (FastEvalGeneration, error) {
				return FastEvalGeneration{
					Text: "ok",
					Metrics: Metrics{
						PromptTokens:        4,
						GeneratedTokens:     2,
						PrefillTokensPerSec: 40,
						DecodeTokensPerSec:  20,
					},
				}, nil
			},
		},
		Eval: EvalRunner{
			BuildBatches: func(context.Context, SFTDataset, DatasetBatchConfig) ([]SFTBatch, error) {
				return []SFTBatch{{Batch: Batch{Tokens: [][]int{{1, 2, 3}}, LossMask: [][]float32{{1, 1, 1}}}}}, nil
			},
			EvaluateBatch: func(context.Context, SFTBatch) (EvalBatchMetrics, error) {
				return EvalBatchMetrics{Loss: 0.75}, nil
			},
		},
	}

	report, err := RunWorkloadBench(context.Background(), runner, WorkloadBenchConfig{
		FastEval: FastEvalConfig{Prompt: "p", MaxTokens: 2, Runs: 1},
		EvalDataset: NewSFTSliceDataset([]SFTSample{
			{Prompt: "a", Response: "b"},
		}),
		IncludePerplexity: true,
	})
	if err != nil {
		t.Fatalf("RunWorkloadBench() error = %v", err)
	}
	if report.Evaluation.Report == nil {
		t.Fatal("Evaluation.Report = nil, want dataset eval report")
	}
	if report.Evaluation.Metrics.Tokens != 3 || report.Summary.EvalTokens != 3 {
		t.Fatalf("eval metrics = %+v summary=%+v", report.Evaluation.Metrics, report.Summary)
	}
	if !evalQualityPassed(report.Evaluation.Quality, "perplexity_finite") {
		t.Fatalf("quality = %+v", report.Evaluation.Quality.Checks)
	}
}

func TestRunWorkloadBench_SummarizesMemvidKVBlockWarm_Good(t *testing.T) {
	warmed := false
	storePath := core.PathJoin(t.TempDir(), "bench-kv-blocks.mvlog")
	runner := WorkloadBenchRunner{
		FastEval: FastEvalRunner{
			Generate: func(_ context.Context, prompt string, cfg GenerateConfig) (FastEvalGeneration, error) {
				metrics := Metrics{
					PromptTokens:          3,
					GeneratedTokens:       cfg.MaxTokens,
					PromptCacheMisses:     1,
					PromptCacheMissTokens: 3,
				}
				if warmed && prompt == "stable prefix" {
					metrics.PromptCacheHits = 1
					metrics.PromptCacheMisses = 0
					metrics.PromptCacheHitTokens = 2
					metrics.PromptCacheMissTokens = 1
				}
				return FastEvalGeneration{Text: "ok", Metrics: metrics}, nil
			},
			CaptureKV: func(context.Context, string) (*KVSnapshot, error) {
				return fastEvalTestSnapshot(), nil
			},
			WarmPromptCacheFromMemvidBlocks: func(ctx context.Context, store memvid.Store, bundle *KVSnapshotMemvidBlockBundle, prefixTokens int) error {
				if _, err := LoadKVSnapshotPrefixFromMemvidBlocks(ctx, store, bundle, prefixTokens); err != nil {
					return err
				}
				warmed = true
				return nil
			},
		},
	}

	report, err := RunWorkloadBench(context.Background(), runner, WorkloadBenchConfig{
		FastEval: FastEvalConfig{
			Prompt:                      "baseline",
			CachePrompt:                 "stable prefix",
			MaxTokens:                   1,
			Runs:                        1,
			IncludeMemvidKVBlockWarm:    true,
			MemvidKVBlockSize:           2,
			MemvidKVPrefixTokens:        3,
			MemvidKVBlockStorePath:      storePath,
			IncludePromptCache:          false,
			IncludeKVRestore:            false,
			IncludeStateBundleRoundTrip: false,
			IncludeProbeOverhead:        false,
		},
	})
	if err != nil {
		t.Fatalf("RunWorkloadBench() error = %v", err)
	}

	if report.Summary.PromptCacheSource != filestore.CodecFile || report.Summary.MemvidKVBlocksRead != 2 {
		t.Fatalf("summary cache fields = %+v, want memvid source and two blocks read", report.Summary)
	}
	if report.Summary.MemvidKVBlockStorePath != storePath || report.Summary.MemvidKVBlockStoreBytes <= 0 {
		t.Fatalf("summary file store = path %q bytes %d, want file-backed store", report.Summary.MemvidKVBlockStorePath, report.Summary.MemvidKVBlockStoreBytes)
	}
	if report.Summary.PromptTokensAvoided != 2 || report.Summary.PromptCacheReplayTokens != 1 || report.Summary.PromptCacheExactFallbackReplayTokens != 1 {
		t.Fatalf("summary token fields = %+v, want avoided=2 replay=1 exact=1", report.Summary)
	}
	if report.Summary.MemvidKVBlockRestoreDuration <= 0 {
		t.Fatalf("summary restore duration = %v, want measured duration", report.Summary.MemvidKVBlockRestoreDuration)
	}
}

func TestRunWorkloadBench_SummarizesDecodeOptimisations_Good(t *testing.T) {
	runner := WorkloadBenchRunner{
		FastEval: FastEvalRunner{
			Generate: func(context.Context, string, GenerateConfig) (FastEvalGeneration, error) {
				return FastEvalGeneration{
					Tokens:  []Token{{ID: 1, Text: "A"}, {ID: 2, Text: "B"}},
					Metrics: Metrics{GeneratedTokens: 2, DecodeTokensPerSec: 20},
				}, nil
			},
			DraftGenerate: func(context.Context, string, GenerateConfig) (FastEvalGeneration, error) {
				return FastEvalGeneration{Tokens: []Token{{ID: 1, Text: "A"}, {ID: 9, Text: "?"}}}, nil
			},
		},
	}

	report, err := RunWorkloadBench(context.Background(), runner, WorkloadBenchConfig{
		FastEval: FastEvalConfig{
			Prompt:                    "baseline",
			MaxTokens:                 2,
			Runs:                      1,
			IncludeSpeculativeDecode:  true,
			SpeculativeDraftTokens:    2,
			IncludePromptLookupDecode: true,
			PromptLookupTokens:        []Token{{ID: 1, Text: "A"}, {ID: 9, Text: "?"}},
		},
	})
	if err != nil {
		t.Fatalf("RunWorkloadBench() error = %v", err)
	}
	if report.Summary.SpeculativeAcceptedTokens != 1 || report.Summary.SpeculativeAcceptanceRate != 0.5 {
		t.Fatalf("summary speculative = %+v, want one accepted at 0.5", report.Summary)
	}
	if report.Summary.PromptLookupAcceptedTokens != 1 || report.Summary.PromptLookupAcceptanceRate != 0.5 {
		t.Fatalf("summary prompt lookup = %+v, want one accepted at 0.5", report.Summary)
	}
}

func TestRunWorkloadBench_SummarizesExpertResidency_Good(t *testing.T) {
	runner := WorkloadBenchRunner{
		FastEval: FastEvalRunner{
			Generate: func(context.Context, string, GenerateConfig) (FastEvalGeneration, error) {
				return FastEvalGeneration{Text: "ok", Metrics: Metrics{GeneratedTokens: 1, DecodeTokensPerSec: 20}}, nil
			},
		},
		MeasureExpertResidency: func(context.Context, ExpertResidencyPlan) (ExpertResidencyStats, error) {
			return ExpertResidencyStats{
				ResidentExperts:     4,
				PeakResidentExperts: 6,
				PageIns:             3,
				PageOuts:            1,
				LoadedBytes:         2048,
				EvictedBytes:        512,
				FirstUseLatency:     5,
				TotalLoadDuration:   9,
			}, nil
		},
	}

	report, err := RunWorkloadBench(context.Background(), runner, WorkloadBenchConfig{
		FastEval:               FastEvalConfig{Prompt: "baseline", MaxTokens: 1, Runs: 1},
		IncludeExpertResidency: true,
		ExpertResidency: ExpertResidencyPlan{
			Enabled:            true,
			Mode:               ExpertResidencyModeLazy,
			MaxResidentExperts: 8,
		},
	})
	if err != nil {
		t.Fatalf("RunWorkloadBench() error = %v", err)
	}
	if !report.ExpertResidency.Attempted || report.ExpertResidency.Stats.PageIns != 3 {
		t.Fatalf("expert residency report = %+v, want attempted stats", report.ExpertResidency)
	}
	if report.Summary.ExpertResidencyPageIns != 3 || report.Summary.ExpertResidencyFirstUseLatency != 5 || report.Summary.ExpertResidencyLoadedBytes != 2048 {
		t.Fatalf("summary expert residency = %+v, want page-ins/latency/bytes", report.Summary)
	}
}

func TestRunWorkloadBench_RequiresFastEvalRunner_Bad(t *testing.T) {
	_, err := RunWorkloadBench(context.Background(), WorkloadBenchRunner{}, WorkloadBenchConfig{})
	if err == nil {
		t.Fatal("expected missing fast eval generate error")
	}
}

func TestRunWorkloadBench_DisabledOptionalSections_Ugly(t *testing.T) {
	runner := WorkloadBenchRunner{
		FastEval: FastEvalRunner{
			Generate: func(context.Context, string, GenerateConfig) (FastEvalGeneration, error) {
				return FastEvalGeneration{
					Text: "ok",
					Metrics: Metrics{
						PromptTokens:        1,
						GeneratedTokens:     1,
						PrefillTokensPerSec: 10,
						DecodeTokensPerSec:  20,
					},
				}, nil
			},
		},
	}

	report, err := RunWorkloadBench(context.Background(), runner, WorkloadBenchConfig{
		FastEval: FastEvalConfig{
			Prompt:    "p",
			MaxTokens: 1,
			Runs:      1,
		},
	})
	if err != nil {
		t.Fatalf("RunWorkloadBench() error = %v", err)
	}
	if report.Adapter.Load.Attempted || report.Adapter.Fuse.Attempted || report.Evaluation.Attempted {
		t.Fatalf("optional sections should be disabled: adapter=%+v eval=%+v", report.Adapter, report.Evaluation)
	}
	if report.Summary.DecodeTokensPerSec != 20 {
		t.Fatalf("summary = %+v, want decode rate from fast eval", report.Summary)
	}
}

func TestWorkloadBench_DefaultWorkloadBenchConfig_Good(t *testing.T) {
	cfg := DefaultWorkloadBenchConfig()
	if cfg.FastEval.MaxTokens <= 0 || cfg.FastEval.Runs <= 0 || !cfg.FastEval.IncludePromptCache {
		t.Fatalf("DefaultWorkloadBenchConfig() = %+v, want fast-eval defaults", cfg)
	}
}

func TestWorkloadBench_RunModelWorkloadBench_Bad(t *testing.T) {
	_, err := RunModelWorkloadBench(context.Background(), nil, WorkloadBenchConfig{})
	if err == nil {
		t.Fatal("expected nil model error")
	}
}

func TestWorkloadBench_NewModelWorkloadBenchRunner_Ugly(t *testing.T) {
	runner := NewModelWorkloadBenchRunner(&Model{})
	if runner.FastEval.Generate == nil || runner.LoadAdapter == nil || runner.FuseAdapter == nil {
		t.Fatalf("runner = %+v, want fast eval and adapter hooks", runner)
	}
}

func TestWorkloadBenchOptionalErrorBranches_Bad(t *testing.T) {
	var adapterReport WorkloadAdapterReport
	if adapter := runWorkloadAdapterLoad(context.Background(), WorkloadBenchRunner{}, WorkloadBenchConfig{}, &adapterReport); adapter.Path != "" || adapterReport.Load.Error == "" {
		t.Fatalf("adapter load without path = %+v report=%+v, want error", adapter, adapterReport)
	}
	adapterReport = WorkloadAdapterReport{}
	if adapter := runWorkloadAdapterLoad(context.Background(), WorkloadBenchRunner{}, WorkloadBenchConfig{AdapterPath: "/adapters/a"}, &adapterReport); adapter.Path != "" || adapterReport.Load.Error == "" {
		t.Fatalf("adapter load unsupported = %+v report=%+v, want error", adapter, adapterReport)
	}
	adapterReport = WorkloadAdapterReport{}
	adapter := runWorkloadAdapterLoad(context.Background(), WorkloadBenchRunner{
		LoadAdapter: func(context.Context, string) (WorkloadAdapterInfo, error) {
			return WorkloadAdapterInfo{}, core.NewError("load failed")
		},
	}, WorkloadBenchConfig{AdapterPath: "/adapters/a"}, &adapterReport)
	if adapter.Path != "" || adapterReport.Load.Error == "" || adapterReport.Load.Duration <= 0 {
		t.Fatalf("adapter load failure = %+v report=%+v, want timed error", adapter, adapterReport)
	}

	runWorkloadAdapterFuse(context.Background(), WorkloadBenchRunner{}, WorkloadAdapterInfo{}, nil)
	adapterReport = WorkloadAdapterReport{Load: WorkloadLatencyReport{Error: "load failed"}}
	runWorkloadAdapterFuse(context.Background(), WorkloadBenchRunner{}, WorkloadAdapterInfo{}, &adapterReport)
	if adapterReport.Fuse.Error == "" {
		t.Fatalf("fuse after failed load report = %+v, want error", adapterReport)
	}
	adapterReport = WorkloadAdapterReport{}
	runWorkloadAdapterFuse(context.Background(), WorkloadBenchRunner{}, WorkloadAdapterInfo{}, &adapterReport)
	if adapterReport.Fuse.Error == "" {
		t.Fatalf("fuse without adapter report = %+v, want error", adapterReport)
	}
	adapterReport = WorkloadAdapterReport{}
	runWorkloadAdapterFuse(context.Background(), WorkloadBenchRunner{}, WorkloadAdapterInfo{Path: "/adapters/a"}, &adapterReport)
	if adapterReport.Fuse.Error == "" {
		t.Fatalf("fuse unsupported report = %+v, want error", adapterReport)
	}
	adapterReport = WorkloadAdapterReport{}
	runWorkloadAdapterFuse(context.Background(), WorkloadBenchRunner{
		FuseAdapter: func(context.Context, WorkloadAdapterInfo) error {
			return core.NewError("fuse failed")
		},
	}, WorkloadAdapterInfo{Path: "/adapters/a"}, &adapterReport)
	if adapterReport.Fuse.Error == "" || adapterReport.Fuse.Duration <= 0 {
		t.Fatalf("fuse failure report = %+v, want timed error", adapterReport)
	}

	if report := runWorkloadEvaluation(context.Background(), WorkloadBenchRunner{}, WorkloadBenchConfig{IncludePerplexity: true}); report.Error == "" {
		t.Fatalf("perplexity unsupported report = %+v, want error", report)
	}
	if report := runWorkloadEvaluation(context.Background(), WorkloadBenchRunner{
		EvaluatePerplexity: func(context.Context, []WorkloadEvalSample) (WorkloadEvalMetrics, error) {
			return WorkloadEvalMetrics{}, nil
		},
	}, WorkloadBenchConfig{IncludePerplexity: true}); report.Error == "" {
		t.Fatalf("perplexity no samples report = %+v, want error", report)
	}
	if report := runWorkloadEvaluation(context.Background(), WorkloadBenchRunner{
		EvaluatePerplexity: func(context.Context, []WorkloadEvalSample) (WorkloadEvalMetrics, error) {
			return WorkloadEvalMetrics{}, core.NewError("eval failed")
		},
	}, WorkloadBenchConfig{IncludePerplexity: true, EvalSamples: []WorkloadEvalSample{{Text: "sample"}}}); report.Error == "" || report.Duration <= 0 {
		t.Fatalf("perplexity failure report = %+v, want timed error", report)
	}
	if report := runWorkloadExpertResidency(context.Background(), WorkloadBenchRunner{}, WorkloadBenchConfig{IncludeExpertResidency: true}); report.Error == "" {
		t.Fatalf("expert unsupported report = %+v, want error", report)
	}
	if report := runWorkloadExpertResidency(context.Background(), WorkloadBenchRunner{
		MeasureExpertResidency: func(context.Context, ExpertResidencyPlan) (ExpertResidencyStats, error) {
			return ExpertResidencyStats{}, core.NewError("residency failed")
		},
	}, WorkloadBenchConfig{IncludeExpertResidency: true}); report.Error == "" || report.Duration <= 0 {
		t.Fatalf("expert failure report = %+v, want timed error", report)
	}
}

func TestWorkloadBenchHelpers_Good(t *testing.T) {
	if summary := summarizeWorkloadBench(nil); summary != (WorkloadBenchSummary{}) {
		t.Fatalf("summarizeWorkloadBench(nil) = %+v, want zero summary", summary)
	}
	evalMetrics := workloadEvalMetricsFromEval(EvalMetrics{Samples: 2, Tokens: 7, Loss: 1.5, Perplexity: 4.4})
	if evalMetrics.Samples != 2 || evalMetrics.Tokens != 7 || evalMetrics.Perplexity != 4.4 {
		t.Fatalf("workload eval metrics = %+v, want copied metrics", evalMetrics)
	}
	adapter := workloadAdapterInfo("/adapters/domain", &LoRAAdapter{})
	if adapter.Name != "domain" || adapter.Path != "/adapters/domain" {
		t.Fatalf("workload adapter info = %+v, want adapter path/name metadata", adapter)
	}
	cloned := cloneWorkloadAdapterInfo(adapter)
	cloned.TargetKeys = []string{"mutated"}
	if len(adapter.TargetKeys) != 0 {
		t.Fatalf("adapter target keys were aliased: %+v", adapter.TargetKeys)
	}
	samples := []WorkloadEvalSample{{Text: "sample", Meta: map[string]string{"id": "1"}}}
	clonedSamples := cloneWorkloadEvalSamples(samples)
	clonedSamples[0].Meta["id"] = "2"
	if samples[0].Meta["id"] != "1" {
		t.Fatalf("eval sample metadata was aliased: %+v", samples[0].Meta)
	}
	if cloneWorkloadEvalSamples(nil) != nil {
		t.Fatal("cloneWorkloadEvalSamples(nil) != nil")
	}
	if nonZeroDuration(0) <= 0 || nonZeroDuration(time.Millisecond) != time.Millisecond {
		t.Fatal("nonZeroDuration() did not preserve positive durations")
	}

	report := runWorkloadEvaluation(context.Background(), WorkloadBenchRunner{
		EvaluatePerplexity: func(context.Context, []WorkloadEvalSample) (WorkloadEvalMetrics, error) {
			return WorkloadEvalMetrics{Loss: 1}, nil
		},
	}, WorkloadBenchConfig{EvalSamples: []WorkloadEvalSample{{Text: "sample"}}})
	if report.Error != "" || report.Metrics.Samples != 1 || report.Metrics.Perplexity == 0 {
		t.Fatalf("perplexity success report = %+v, want default sample count and exp(loss)", report)
	}
}

// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"
	"testing"
	"time"
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
				return ModelInfo{Architecture: "qwen3", NumLayers: 28, QuantBits: 4, ContextLength: 32768}
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
		AdapterPath:        adapter.Path,
		IncludeAdapterLoad: true,
		IncludeAdapterFuse: true,
		IncludePerplexity:  true,
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
	if report.Summary.PrefillTokensPerSec != 200 || report.Summary.DecodeTokensPerSec != 75 || report.Summary.PeakMemoryBytes != 8<<20 {
		t.Fatalf("summary = %+v, want fast-eval throughput and memory mirrored", report.Summary)
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

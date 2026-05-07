// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"
	"testing"
	"time"

	core "dappco.re/go"
)

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
		CaptureKV: func(_ context.Context, prompt string) (*KVSnapshot, error) {
			if prompt == "" {
				t.Fatal("CaptureKV received empty prompt")
			}
			return fastEvalTestSnapshot(), nil
		},
		RestoreKV: func(_ context.Context, snapshot *KVSnapshot) error {
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
	runner.CaptureKV = func(context.Context, string) (*KVSnapshot, error) { return nil, core.NewError("capture failed") }
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

func fastEvalTestSnapshot() *KVSnapshot {
	return &KVSnapshot{
		Version:       KVSnapshotVersion,
		Architecture:  "gemma4_text",
		Tokens:        []int32{1, 2, 3},
		TokenOffset:   3,
		NumLayers:     1,
		NumHeads:      1,
		SeqLen:        3,
		HeadDim:       2,
		NumQueryHeads: 1,
		Layers: []KVLayerSnapshot{{
			Layer:      0,
			CacheIndex: 0,
			Heads: []KVHeadSnapshot{{
				Key:   []float32{0.1, 0.2, 0.3, 0.4, 0.5, 0.6},
				Value: []float32{0.6, 0.5, 0.4, 0.3, 0.2, 0.1},
			}},
		}},
	}
}

// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"
	"testing"
	"time"
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

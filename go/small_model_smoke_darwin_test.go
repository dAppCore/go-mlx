// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && !nomlx

package mlx

import (
	"dappco.re/go/mlx/memory"
	"context"
	"testing"
	"time"

	"dappco.re/go/mlx/internal/metal"
)

func TestRunSmallModelSmoke_ForwardsBudgetedLoadOptions_Good(t *testing.T) {
	dir := t.TempDir()
	writeGoodSafetensorsPack(t, dir, "gemma4_text")

	originalLoadNativeModel := loadNativeModel
	t.Cleanup(func() { loadNativeModel = originalLoadNativeModel })

	var got metal.LoadConfig
	loadNativeModel = func(modelPath string, cfg metal.LoadConfig) (nativeModel, error) {
		got = cfg
		return &fakeNativeModel{
			info: metal.ModelInfo{
				Architecture:  "gemma4_text",
				ContextLength: 8192,
				NumLayers:     26,
				HiddenSize:    2048,
				QuantBits:     4,
			},
			tokens: []metal.Token{{ID: 1, Text: "ok"}},
			metrics: metal.Metrics{
				PromptTokens:               4,
				GeneratedTokens:            1,
				PrefillTokensPerSec:        200,
				DecodeTokensPerSec:         40,
				TotalDuration:              time.Millisecond,
				PromptCacheHits:            1,
				PromptCacheHitTokens:       4,
				PromptCacheRestoreDuration: time.Millisecond,
			},
		}, nil
	}

	report, err := RunSmallModelSmoke(context.Background(), SmallModelSmokeConfig{
		ModelPath: dir,
		Device: DeviceInfo{
			Architecture:                 "apple9",
			MemorySize:                   96 * memory.GiB,
			MaxRecommendedWorkingSetSize: 90 * memory.GiB,
		},
		Workload: WorkloadBenchConfig{
			FastEval: FastEvalConfig{
				Prompt:             "hi",
				CachePrompt:        "hi",
				MaxTokens:          1,
				Runs:               1,
				IncludePromptCache: true,
			},
		},
	})
	if err != nil {
		t.Fatalf("RunSmallModelSmoke() error = %v", err)
	}
	if report == nil || report.Skipped || report.Bench == nil {
		t.Fatalf("report = %+v, want loaded bench", report)
	}
	if got.ContextLen != 8192 || got.ExpectedQuantization != 4 {
		t.Fatalf("load context/quant = %d/q%d, want 8192/q4", got.ContextLen, got.ExpectedQuantization)
	}
	if got.BatchSize != 1 || got.PrefillChunkSize > 1024 {
		t.Fatalf("load shape = batch:%d prefill:%d, want small smoke shape", got.BatchSize, got.PrefillChunkSize)
	}
	if got.MemoryLimitBytes == 0 || got.CacheLimitBytes == 0 || got.WiredLimitBytes == 0 {
		t.Fatalf("allocator limits not forwarded: %+v", got)
	}
	if report.Bench.Summary.PrefillTokensPerSec != 200 || report.Bench.Summary.DecodeTokensPerSec != 40 {
		t.Fatalf("bench summary = %+v, want fake metrics", report.Bench.Summary)
	}
}

// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"testing"

	"dappco.re/go/inference/bench"
	"dappco.re/go/inference/eval"
	"dappco.re/go/inference/quant/jang"
	"dappco.re/go/mlx/dataset"
	"dappco.re/go/mlx/memory"
)

// Baselines the hot helpers in workload_bench.go so the Wave6 perf
// pass has an apples-to-apples allocation-count comparison after each
// edit.

func benchWorkloadConfig() WorkloadBenchConfig {
	return WorkloadBenchConfig{
		Eval: eval.Config{
			Batch:         dataset.BatchConfig{BatchSize: 0},
			QualityProbes: nil,
		},
		QuantizationProfile: &jang.PackedProfile{
			Type:   "q4",
			Format: "q4",
		},
		EvalSamples: []WorkloadEvalSample{
			{Prompt: "p1", Response: "r1", Text: "t1", Meta: map[string]string{"k": "v"}},
			{Prompt: "p2", Response: "r2", Text: "t2", Meta: map[string]string{"k": "v"}},
			{Prompt: "p3", Response: "r3", Text: "t3"},
		},
		ExpertResidency: memory.ExpertResidencyPlan{
			Enabled:        true,
			HotExpertIDs:   []int{1, 2, 3},
			ExpertsPerToken: 2,
		},
	}
}

func BenchmarkNormalizeWorkloadBenchConfig(b *testing.B) {
	cfg := benchWorkloadConfig()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = normalizeWorkloadBenchConfig(cfg)
	}
}

// BenchmarkNormalizeWorkloadBenchConfig_NoProfile exercises the typical
// caller shape (no quantization profile, no eval samples) where the
// guarded jang.ClonePackedProfile short-circuit pays off.
func BenchmarkNormalizeWorkloadBenchConfig_NoProfile(b *testing.B) {
	cfg := WorkloadBenchConfig{}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = normalizeWorkloadBenchConfig(cfg)
	}
}

func BenchmarkCloneWorkloadEvalSamples(b *testing.B) {
	samples := benchWorkloadConfig().EvalSamples
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = cloneWorkloadEvalSamples(samples)
	}
}

func BenchmarkCloneWorkloadAdapterInfo(b *testing.B) {
	info := WorkloadAdapterInfo{
		Path: "/x/y/z",
		Name: "z",
		TargetKeys: []string{"q_proj", "k_proj", "v_proj", "o_proj"},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = cloneWorkloadAdapterInfo(info)
	}
}

func BenchmarkCloneWorkloadAdapterInfo_NoTargets(b *testing.B) {
	info := WorkloadAdapterInfo{
		Path: "/x/y/z",
		Name: "z",
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = cloneWorkloadAdapterInfo(info)
	}
}

func BenchmarkKvBenchConfigFromModelInfo(b *testing.B) {
	info := ModelInfo{
		ContextLength: 4096,
		NumLayers:     32,
		HiddenSize:    4096,
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = kvBenchConfigFromModelInfo(info)
	}
}

func BenchmarkSummarizeWorkloadBench_Empty(b *testing.B) {
	report := &WorkloadBenchReport{}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = summarizeWorkloadBench(report)
	}
}

func BenchmarkSummarizeWorkloadBench_Full(b *testing.B) {
	report := &WorkloadBenchReport{
		FastEval: &bench.Report{
			Generation: bench.GenerationSummary{
				PrefillTokensPerSec: 1000,
				DecodeTokensPerSec:  500,
				PeakMemoryBytes:     1024 * 1024 * 1024,
				ActiveMemoryBytes:   512 * 1024 * 1024,
			},
			PromptCache: bench.PromptCacheReport{
				HitRate:         0.8,
				HitTokens:       400,
				MissTokens:      100,
				RestoreDuration: 1000,
			},
			StateKVBlockWarm: bench.StateKVBlockWarmReport{
				Attempted:                 true,
				Source:                    "s",
				PromptTokensAvoided:       42,
				ReplayTokens:              10,
				ExactFallbackReplayTokens: 5,
				RestoreDuration:           2000,
				StorePath:                 "/path",
				StoreBytes:                1024,
				BlocksRead:                3,
				ChunksRead:                7,
				PrefixTokensRestored:      11,
			},
			SpeculativeDecode: bench.DecodeOptimisationReport{
				Attempted: true,
			},
			PromptLookupDecode: bench.DecodeOptimisationReport{
				Attempted: true,
			},
		},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = summarizeWorkloadBench(report)
	}
}

func BenchmarkSummarizeWorkloadBench_Residency(b *testing.B) {
	report := &WorkloadBenchReport{
		ExpertResidency: WorkloadExpertResidencyReport{
			Attempted: true,
			Stats: memory.ExpertResidencyStats{
				ResidentExperts:     8,
				PeakResidentExperts: 16,
				PageIns:             100,
				PageOuts:            42,
				LoadedBytes:         1024 * 1024,
				EvictedBytes:        512 * 1024,
			},
		},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = summarizeWorkloadBench(report)
	}
}

func BenchmarkNonZeroDuration(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = nonZeroDuration(0)
	}
}

func BenchmarkWorkloadEvalMetricsFromEval(b *testing.B) {
	m := eval.Metrics{Samples: 32, Tokens: 1024, Loss: 1.5, Perplexity: 4.5}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = workloadEvalMetricsFromEval(m)
	}
}

func BenchmarkNormalizeWorkloadEvalConfig(b *testing.B) {
	cfg := eval.Config{
		Batch:         dataset.BatchConfig{BatchSize: 0},
		QualityProbes: nil,
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = normalizeWorkloadEvalConfig(cfg)
	}
}

// BenchmarkToMetalLoRAConfig_Empty exercises the empty-target path of
// toMetalLoRAConfig — the most common shape when callers leave
// TargetKeys/TargetLayers nil.
func BenchmarkToMetalLoRAConfig_Empty(b *testing.B) {
	cfg := LoRAConfig{Rank: 8, Alpha: 16}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = toMetalLoRAConfig(cfg)
	}
}

// BenchmarkRunWorkloadBench_PreCheck exercises the RunWorkloadBench
// entrypoint shape up to (and excluding) the FastEval body — by passing
// a runner whose FastEval rejects immediately we measure the cfg
// normalisation + report construction overhead, the surface this lane
// is optimising.
func BenchmarkRunWorkloadBench_PreCheck(b *testing.B) {
	cfg := benchWorkloadConfig()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		// Run normalize + the report struct build path that follows it
		// in RunWorkloadBench.
		c := normalizeWorkloadBenchConfig(cfg)
		_ = &WorkloadBenchReport{
			Version:             WorkloadBenchReportVersion,
			QuantizationProfile: c.QuantizationProfile,
		}
	}
}

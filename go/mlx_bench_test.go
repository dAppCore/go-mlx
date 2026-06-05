// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for the root mlx-package CPU-only primitives — option
// builders, default config constructors, LoadConfig validation, and the
// split-inference plan deep clone. Per AX-11 — every Generate/LoadModel
// call walks the option fn stack at least once. applyGenerateOptions runs
// per inference call; normalizeLoadConfig runs once per model load but
// is on the model-startup critical path.
//
// Metal-bound entry points (LoadModel, Model.Generate, GC, SetCacheLimit,
// etc.) are intentionally OUT of scope — those need a GPU and live in
// the model-level benches.
//
// Run:    go test -bench='BenchmarkMlxRoot' -benchmem -run='^$' ./go

package mlx

import (
	"testing"

	"dappco.re/go/inference"
	"dappco.re/go/mlx/memory"
	"dappco.re/go/mlx/probe"
)

// Sinks defeat compiler DCE. Names disjoint from root_bench_test.go's
// rootBench* set so the two files coexist in the same package.
var (
	mlxBenchSinkGenConfig  GenerateConfig
	mlxBenchSinkLoadConfig LoadConfig
	mlxBenchSinkErr        error
	mlxBenchSinkBool       bool
	mlxBenchSinkSplitPlan  *inference.SplitInferencePlan
	mlxBenchSinkGenOption  GenerateOption
	mlxBenchSinkLoadOption LoadOption
)

// --- DefaultGenerateConfig / DefaultLoadConfig — struct construction ---

func BenchmarkMlxRoot_DefaultGenerateConfig(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		mlxBenchSinkGenConfig = DefaultGenerateConfig()
	}
}

func BenchmarkMlxRoot_DefaultLoadConfig(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		mlxBenchSinkLoadConfig = DefaultLoadConfig()
	}
}

// --- Generate option builders — invoked once per option per call site ---

func BenchmarkMlxRoot_WithMaxTokens(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		mlxBenchSinkGenOption = WithMaxTokens(256)
	}
}

func BenchmarkMlxRoot_WithTemperature(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		mlxBenchSinkGenOption = WithTemperature(0.7)
	}
}

func BenchmarkMlxRoot_WithStopTokens_3IDs(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		mlxBenchSinkGenOption = WithStopTokens(int32(1), int32(2), int32(3))
	}
}

func BenchmarkMlxRoot_WithProbeCallback_Nil(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		mlxBenchSinkGenOption = WithProbeCallback(nil)
	}
}

func BenchmarkMlxRoot_WithProbeCallback_NonNil(b *testing.B) {
	callback := func(probe.Event) {}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		mlxBenchSinkGenOption = WithProbeCallback(callback)
	}
}

// No-argument option builders should return a package-init singleton
// closure — measured here so future regressions surface immediately.
func BenchmarkMlxRoot_WithLogits(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		mlxBenchSinkGenOption = WithLogits()
	}
}

func BenchmarkMlxRoot_WithTokenPhaseTrace(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		mlxBenchSinkGenOption = WithTokenPhaseTrace()
	}
}

func BenchmarkMlxRoot_WithTokenPhaseTraceText(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		mlxBenchSinkGenOption = WithTokenPhaseTraceText()
	}
}

// --- applyGenerateOptions — full option stack walk, the hot path ---

// Typical caller: a few options (temp + max_tokens + maybe top_p).
func BenchmarkMlxRoot_ApplyGenerateOptions_Typical(b *testing.B) {
	opts := []GenerateOption{
		WithMaxTokens(256),
		WithTemperature(0.7),
		WithTopP(0.95),
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		mlxBenchSinkGenConfig = applyGenerateOptions(opts)
	}
}

// Heavier: stop-tokens + suppress-tokens + every sampler knob.
func BenchmarkMlxRoot_ApplyGenerateOptions_Heavy(b *testing.B) {
	stop := []int32{1, 2, 3}
	suppress := []int32{100, 200, 300, 400}
	opts := []GenerateOption{
		WithMaxTokens(512),
		WithTemperature(0.8),
		WithTopK(40),
		WithTopP(0.9),
		WithMinP(0.05),
		WithSeed(42),
		WithRepeatPenalty(1.1),
		WithStopTokens(stop...),
		WithSuppressTokens(suppress...),
		WithLogits(),
		WithTokenPhaseTrace(),
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		mlxBenchSinkGenConfig = applyGenerateOptions(opts)
	}
}

// --- Load option builders ---

func BenchmarkMlxRoot_WithContextLength(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		mlxBenchSinkLoadOption = WithContextLength(131072)
	}
}

func BenchmarkMlxRoot_WithMemoryPlan(b *testing.B) {
	plan := memory.Plan{
		ContextLength:        32768,
		ParallelSlots:        1,
		PromptCache:          true,
		PromptCacheMinTokens: 2048,
		BatchSize:            1,
		PrefillChunkSize:     512,
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		mlxBenchSinkLoadOption = WithMemoryPlan(plan)
	}
}

func BenchmarkMlxRoot_WithAllocatorLimits(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		mlxBenchSinkLoadOption = WithAllocatorLimits(32<<30, 4<<30, 24<<30)
	}
}

// --- applyLoadOptions — the model-load stack walk ---

func BenchmarkMlxRoot_ApplyLoadOptions_Typical(b *testing.B) {
	opts := []LoadOption{
		WithContextLength(131072),
		WithBatchSize(1),
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		mlxBenchSinkLoadConfig = applyLoadOptions(opts)
	}
}

func BenchmarkMlxRoot_ApplyLoadOptions_Heavy(b *testing.B) {
	opts := []LoadOption{
		WithContextLength(131072),
		WithParallelSlots(2),
		WithPromptCache(true),
		WithPromptCacheMinTokens(2048),
		WithQuantization(4),
		WithExpectedQuantization(4),
		WithDevice("gpu"),
		WithAdapterPath("/some/adapter"),
		WithAutoMemoryPlan(true),
		WithCachePolicy(memory.KVCacheFull),
		WithKVCacheMode(memory.KVCacheModeFP16),
		WithBatchSize(1),
		WithPrefillChunkSize(512),
		WithAllocatorLimits(32<<30, 4<<30, 24<<30),
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		mlxBenchSinkLoadConfig = applyLoadOptions(opts)
	}
}

// --- normalizeLoadConfig — pre-load validation on the critical path ---

func BenchmarkMlxRoot_NormalizeLoadConfig_Default(b *testing.B) {
	cfg := DefaultLoadConfig()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		mlxBenchSinkLoadConfig, mlxBenchSinkErr = normalizeLoadConfig(cfg)
	}
}

func BenchmarkMlxRoot_NormalizeLoadConfig_DeviceLower(b *testing.B) {
	cfg := DefaultLoadConfig()
	// Force the Lower/Trim branch by passing a noisy device string.
	cfg.Device = "  GPU  "
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		mlxBenchSinkLoadConfig, mlxBenchSinkErr = normalizeLoadConfig(cfg)
	}
}

// With a SplitInference plan attached — exercises the validate+mode branch.
func BenchmarkMlxRoot_NormalizeLoadConfig_WithSplitInference(b *testing.B) {
	cfg := DefaultLoadConfig()
	plan := inference.SplitInferencePlan{
		Mode: inference.SplitInferenceModeLocal,
		LocalSlice: inference.ModelSlicePlan{
			Components: []inference.ModelComponent{
				inference.ModelComponentManifest,
				inference.ModelComponentEmbeddings,
				inference.ModelComponentAttention,
			},
		},
	}
	cfg.SplitInference = &plan
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		mlxBenchSinkLoadConfig, mlxBenchSinkErr = normalizeLoadConfig(cfg)
	}
}

// --- cloneSplitInferencePlan — defensive copy on the load-option path ---

func BenchmarkMlxRoot_CloneSplitInferencePlan_Empty(b *testing.B) {
	plan := inference.SplitInferencePlan{Mode: inference.SplitInferenceModeLocal}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		mlxBenchSinkSplitPlan = cloneSplitInferencePlan(plan)
	}
}

func BenchmarkMlxRoot_CloneSplitInferencePlan_Typical(b *testing.B) {
	plan := inference.SplitInferencePlan{
		Mode: inference.SplitInferenceModeLocal,
		LocalSlice: inference.ModelSlicePlan{
			Components: []inference.ModelComponent{
				inference.ModelComponentManifest,
				inference.ModelComponentTokenizer,
				inference.ModelComponentEmbeddings,
				inference.ModelComponentNorms,
				inference.ModelComponentAttention,
				inference.ModelComponentFFN,
			},
			Notes: []string{"local-only", "no remote endpoints"},
			Labels: map[string]string{
				"profile": "local-workstation",
				"runtime": "metal",
			},
		},
		Labels: map[string]string{
			"plan": "default",
		},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		mlxBenchSinkSplitPlan = cloneSplitInferencePlan(plan)
	}
}

// --- AttentionSnapshot.HasQueries — KV inspection helper ---

func BenchmarkMlxRoot_AttentionSnapshot_HasQueries_Nil(b *testing.B) {
	var snap *AttentionSnapshot
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		mlxBenchSinkBool = snap.HasQueries()
	}
}

func BenchmarkMlxRoot_AttentionSnapshot_HasQueries_Populated(b *testing.B) {
	snap := &AttentionSnapshot{
		Queries: make([][][]float32, 28),
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		mlxBenchSinkBool = snap.HasQueries()
	}
}

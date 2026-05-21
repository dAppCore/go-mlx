// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for memory_plan.go — PlanMemory + the pure helpers
// (deviceInfoToMemory, modelInfoPtrToMemory, minPositive, maxPositive).
// Per AX-11 — PlanMemory fires per LoadModel/PlanModelFit call (the
// inference.ModelFitPlanner surface), so cold-start latency budget
// flows through it. It also fires inside applyMemoryPlanToLoadConfig
// every time a Model is loaded with AutoMemoryPlan=true. Multiple
// hardware/pack shapes exercise the M1/M3-Ultra branches + the M2
// expert-residency overlay.
//
// Run:    go test -bench='BenchmarkMemoryPlan' -benchmem -run='^$' ./go

package mlx

import (
	"testing"

	"dappco.re/go/mlx/memory"
	mp "dappco.re/go/mlx/pack"
)

// Sinks defeat compiler DCE.
var (
	memoryPlanBenchSinkPlan    memory.Plan
	memoryPlanBenchSinkDevice  memory.DeviceInfo
	memoryPlanBenchSinkModel   *memory.ModelInfo
	memoryPlanBenchSinkInt     int
)

// --- PlanMemory ---
// 16GB Apple-silicon class (M1) — the smallest end of the planner
// branch tree. Hits the rotating-cache + 8192 context path.

func BenchmarkMemoryPlan_PlanMemory_Apple16GB(b *testing.B) {
	input := MemoryPlanInput{
		Device: DeviceInfo{
			Architecture:                 "apple7",
			MemorySize:                   16 * memory.GiB,
			MaxRecommendedWorkingSetSize: 14 * memory.GiB,
		},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		memoryPlanBenchSinkPlan = PlanMemory(input)
	}
}

// 96GB Apple-silicon class (M3 Ultra) — the canonical workstation
// shape, paged cache + prompt cache + parallel slots.

func BenchmarkMemoryPlan_PlanMemory_Apple96GB(b *testing.B) {
	input := MemoryPlanInput{
		Device: DeviceInfo{
			Architecture:                 "apple9",
			MemorySize:                   96 * memory.GiB,
			MaxRecommendedWorkingSetSize: 90 * memory.GiB,
		},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		memoryPlanBenchSinkPlan = PlanMemory(input)
	}
}

// Typical inference call shape — DeviceInfo + ModelInfo, no Pack.
// Mirrors the inference.ModelFitPlanner surface.

func BenchmarkMemoryPlan_PlanMemory_WithModelInfo(b *testing.B) {
	model := ModelInfo{
		Architecture:  "qwen3",
		VocabSize:     151936,
		NumLayers:     28,
		HiddenSize:    2048,
		QuantBits:     4,
		QuantGroup:    64,
		ContextLength: 40960,
	}
	input := MemoryPlanInput{
		Device: DeviceInfo{
			Architecture:                 "apple9",
			MemorySize:                   64 * memory.GiB,
			MaxRecommendedWorkingSetSize: 60 * memory.GiB,
		},
		ModelInfo: &model,
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		memoryPlanBenchSinkPlan = PlanMemory(input)
	}
}

// PlanMemory with a ModelPack — the cap-context-to-model branch lights
// up here (plan.ContextLength clamped to pack.ContextLength).

func BenchmarkMemoryPlan_PlanMemory_WithPack(b *testing.B) {
	pack := mp.ModelPack{
		Architecture:  "qwen3_moe",
		ContextLength: 32768,
		NumLayers:     48,
		HiddenSize:    4096,
		QuantBits:     4,
	}
	input := MemoryPlanInput{
		Device: DeviceInfo{
			Architecture:                 "apple7",
			MemorySize:                   16 * memory.GiB,
			MaxRecommendedWorkingSetSize: 13 * memory.GiB,
		},
		Pack: &pack,
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		memoryPlanBenchSinkPlan = PlanMemory(input)
	}
}

// --- deviceInfoToMemory ---
// Pure field shuffle — used inside PlanMemory but also reachable
// independently from other root callers.

func BenchmarkMemoryPlan_DeviceInfoToMemory(b *testing.B) {
	info := DeviceInfo{
		Architecture:                 "apple9",
		MaxBufferLength:              16 * memory.GiB,
		MaxRecommendedWorkingSetSize: 90 * memory.GiB,
		MemorySize:                   96 * memory.GiB,
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		memoryPlanBenchSinkDevice = deviceInfoToMemory(info)
	}
}

// --- modelInfoPtrToMemory ---

func BenchmarkMemoryPlan_ModelInfoPtrToMemory_Nil(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		memoryPlanBenchSinkModel = modelInfoPtrToMemory(nil)
	}
}

func BenchmarkMemoryPlan_ModelInfoPtrToMemory_Populated(b *testing.B) {
	info := &ModelInfo{
		Architecture:  "qwen3",
		VocabSize:     151936,
		NumLayers:     28,
		HiddenSize:    2048,
		QuantBits:     4,
		QuantGroup:    64,
		ContextLength: 40960,
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		memoryPlanBenchSinkModel = modelInfoPtrToMemory(info)
	}
}

// --- minPositive / maxPositive ---
// Tiny but called per-tensor in small_model_smoke.go callers.

func BenchmarkMemoryPlan_MinPositive_BothPositive(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		memoryPlanBenchSinkInt = minPositive(2048, 4096)
	}
}

func BenchmarkMemoryPlan_MinPositive_FirstZero(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		memoryPlanBenchSinkInt = minPositive(0, 4096)
	}
}

func BenchmarkMemoryPlan_MaxPositive(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		memoryPlanBenchSinkInt = maxPositive(2048, 4096)
	}
}

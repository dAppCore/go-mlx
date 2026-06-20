// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for the LoRAConfig <-> metal.LoRAConfig shufflers. These run
// once per adapter setup (train start / adapter load), not per token — a
// low-multiplier path — but the defensive SliceClone on TargetKeys /
// TargetLayers is the only allocating work, guarded behind a len>0 check
// so the common no-overrides adapter shape pays no clone. Benched here to
// confirm the empty-targets fast path stays allocation-free and to pin the
// populated-targets clone cost.
//
// Run:    go test -bench='BenchmarkLoRAConfig_' -benchmem -run='^$' ./spine

package spine

import (
	"testing"

	"dappco.re/go/mlx/pkg/metal"
)

// Sinks defeat compiler DCE.
var (
	loraBenchSinkMetal metal.LoRAConfig
	loraBenchSinkSpine LoRAConfig
)

func BenchmarkLoRAConfig_ToMetal_NoTargets(b *testing.B) {
	// The default-adapter shape: nil TargetKeys/TargetLayers, so both
	// SliceClone guards skip — should be 0 allocs/op.
	cfg := LoRAConfig{Rank: 8, Alpha: 16, Scale: 2, DType: metal.DTypeFloat16}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		loraBenchSinkMetal = ToMetalLoRAConfig(cfg)
	}
}

func BenchmarkLoRAConfig_ToMetal_WithTargets(b *testing.B) {
	// Populated targets — the two defensive SliceClones fire.
	cfg := LoRAConfig{
		Rank:         8,
		Alpha:        16,
		Scale:        2,
		DType:        metal.DTypeFloat16,
		TargetKeys:   []string{"q_proj", "v_proj", "k_proj", "o_proj"},
		TargetLayers: []string{"0", "1", "2", "3"},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		loraBenchSinkMetal = ToMetalLoRAConfig(cfg)
	}
}

func BenchmarkLoRAConfig_FromMetal_NoTargets(b *testing.B) {
	cfg := metal.LoRAConfig{Rank: 8, Alpha: 16, Scale: 2, DType: metal.DTypeFloat16}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		loraBenchSinkSpine = LoRAConfigFromMetal(cfg)
	}
}

func BenchmarkLoRAConfig_FromMetal_WithTargets(b *testing.B) {
	cfg := metal.LoRAConfig{
		Rank:         8,
		Alpha:        16,
		Scale:        2,
		DType:        metal.DTypeFloat16,
		TargetKeys:   []string{"q_proj", "v_proj", "k_proj", "o_proj"},
		TargetLayers: []string{"0", "1", "2", "3"},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		loraBenchSinkSpine = LoRAConfigFromMetal(cfg)
	}
}

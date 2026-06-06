// SPDX-Licence-Identifier: EUPL-1.2

package memory_test

import (
	"testing"

	"dappco.re/go/mlx/memory"
	mp "dappco.re/go/mlx/pack"
)

// TestNewPlan_ContextDerivedFromMemory_Good proves the plan derives context
// length from truth — the model's declared maximum bounded by what the machine
// actually holds — instead of pinning it at a per-RAM-class magic baseline that
// could only ever cap DOWN. A 256K-capable model on a big machine rises toward
// its declared max; the same model on a starved machine is bounded below it by
// the real memory budget.
func TestNewPlan_ContextDerivedFromMemory_Good(t *testing.T) {
	model := func(weight uint64) *mp.ModelPack {
		return &mp.ModelPack{
			Architecture:  "gemma4_text",
			ContextLength: 262144, // model declares 256K
			NumLayers:     28,
			HiddenSize:    2048,
			WeightBytes:   weight,
			QuantBits:     6,
		}
	}

	big := memory.NewPlan(memory.Input{
		Device: memory.DeviceInfo{Architecture: "apple", MemorySize: 512 * memory.GiB, MaxRecommendedWorkingSetSize: 480 * memory.GiB},
		Pack:   model(8 * memory.GiB),
	})
	if big.ContextLength <= 131072 {
		t.Fatalf("big-RAM ContextLength = %d, want > 131072 (must rise above the old RAM-bucket cap toward the model's 256K)", big.ContextLength)
	}
	if big.ContextLength > 262144 {
		t.Fatalf("big-RAM ContextLength = %d, want <= 262144 (never exceed the model's declared maximum)", big.ContextLength)
	}

	small := memory.NewPlan(memory.Input{
		Device: memory.DeviceInfo{Architecture: "apple", MemorySize: 16 * memory.GiB, MaxRecommendedWorkingSetSize: 14 * memory.GiB},
		Pack:   model(4 * memory.GiB),
	})
	if small.ContextLength <= 0 {
		t.Fatalf("small-RAM ContextLength = %d, want > 0", small.ContextLength)
	}
	if small.ContextLength >= big.ContextLength {
		t.Fatalf("small-RAM ContextLength = %d, want < big-RAM %d (context bounded by device memory)", small.ContextLength, big.ContextLength)
	}
}

// TestNewPlan_ContextUsesRealKVWidth_Good proves the derivation sizes the KV
// cache from the model's true grouped-query width (num_kv_heads * head_dim),
// not hidden_size: a model that declares its KV dims fits MORE context than the
// same model where the planner must fall back to the hidden-size over-estimate.
func TestNewPlan_ContextUsesRealKVWidth_Good(t *testing.T) {
	dev := memory.DeviceInfo{Architecture: "apple", MemorySize: 96 * memory.GiB, MaxRecommendedWorkingSetSize: 80 * memory.GiB}
	base := func() *mp.ModelPack {
		return &mp.ModelPack{Architecture: "gemma4_text", ContextLength: 262144, NumLayers: 48, HiddenSize: 5120, WeightBytes: 12 * memory.GiB, QuantBits: 6}
	}

	// No KV dims declared → planner falls back to hidden_size (over-counts KV).
	fallback := memory.NewPlan(memory.Input{Device: dev, Pack: base()})

	// Real GQA width: 8 kv-heads x 256 head_dim = 2048, far below hidden 5120.
	gqa := base()
	gqa.NumKVHeads = 8
	gqa.HeadDim = 256
	real := memory.NewPlan(memory.Input{Device: dev, Pack: gqa})

	if real.ContextLength <= fallback.ContextLength {
		t.Fatalf("real-KV-width ContextLength = %d, want > hidden-fallback %d (GQA KV is smaller, so more context fits)", real.ContextLength, fallback.ContextLength)
	}
}

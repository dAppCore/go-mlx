// SPDX-Licence-Identifier: EUPL-1.2

package kv

import (
	core "dappco.re/go"

	"dappco.re/go/mlx/memory"
)

// ExampleCompareModes estimates KV cache memory tradeoffs for a 32GB-class
// context and reads back the recommended mode plus the quantised row's storage
// ranking. Only discrete fields are printed so the output is deterministic.
func ExampleCompareModes() {
	report := CompareModes(BenchConfig{
		ContextLength: 32768,
		NumLayers:     32,
		HiddenSize:    3072,
		Modes:         []memory.KVCacheMode{memory.KVCacheModeFP16, memory.KVCacheModeQ8, memory.KVCacheModeKQ8VQ4},
	})

	fp16 := report.ByMode(memory.KVCacheModeFP16)
	q8 := report.ByMode(memory.KVCacheModeQ8)
	core.Println("modes:", len(report.Modes))
	core.Println("recommended:", report.RecommendedMode)
	core.Println("q8 saves vs fp16:", q8.StorageBytes < fp16.StorageBytes)
	// Output:
	// modes: 3
	// recommended: q8
	// q8 saves vs fp16: true
}

// ExampleBenchReport_ByMode looks up a single mode's comparison row in a report,
// then shows that an absent mode returns the zero ModeBench.
func ExampleBenchReport_ByMode() {
	report := CompareModes(BenchConfig{
		ContextLength: 8192,
		NumLayers:     8,
		HiddenSize:    1024,
		Modes:         []memory.KVCacheMode{memory.KVCacheModeFP16, memory.KVCacheModeQ8},
	})

	q8 := report.ByMode(memory.KVCacheModeQ8)
	absent := report.ByMode(memory.KVCacheModeKQ8VQ4)
	core.Println("q8 mode:", q8.Mode)
	core.Println("q8 key bits:", q8.KeyBits)
	core.Println("absent storage:", absent.StorageBytes)
	// Output:
	// q8 mode: q8
	// q8 key bits: 8
	// absent storage: 0
}

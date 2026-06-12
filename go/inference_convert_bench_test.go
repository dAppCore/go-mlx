// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for options.go — inferenceGenerateConfigToMetal.
// Per AX-11 — this is the boundary between the shared inference
// surface and the metal-native generate config. It fires once per
// adapter.generateConfig() call which in turn fires on every
// Generate/Chat/Classify request. The reflect-MinP fallback is
// load-bearing for forward compatibility, so its alloc shape needs
// to be visible.
//
// Run:    go test -bench='BenchmarkOptions' -benchmem -run='^$' ./go

package mlx

import (
	"testing"

	"dappco.re/go/inference"
	"dappco.re/go/mlx/pkg/metal"
)

// Sinks defeat compiler DCE.
var (
	optionsBenchSinkMetalCfg metal.GenerateConfig
)

// --- inferenceGenerateConfigToMetal ---
// Minimal config — only MaxTokens + Temperature populated. Mirrors the
// "default-shape generation" request from a basic Generate call.

func BenchmarkInferenceConvert_InferenceGenerateConfigToMetal_Minimal(b *testing.B) {
	cfg := inference.GenerateConfig{
		MaxTokens:   256,
		Temperature: 0.7,
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		optionsBenchSinkMetalCfg = inferenceGenerateConfigToMetal(cfg)
	}
}

// Typical-shape generation — all sampler levers set + stop tokens. The
// StopTokens slice is aliased, not cloned, so allocs should come only
// from the reflect MinP probe.

func BenchmarkInferenceConvert_InferenceGenerateConfigToMetal_Typical(b *testing.B) {
	cfg := inference.GenerateConfig{
		MaxTokens:     2048,
		Temperature:   0.7,
		TopK:          40,
		TopP:          0.9,
		StopTokens:    []int32{1, 2, 3},
		RepeatPenalty: 1.1,
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		optionsBenchSinkMetalCfg = inferenceGenerateConfigToMetal(cfg)
	}
}

// Empty config — the reflect-MinP probe still fires (the FieldByName
// call always runs); this isolates the lookup cost from the populated
// fields.

func BenchmarkInferenceConvert_InferenceGenerateConfigToMetal_ZeroValue(b *testing.B) {
	var cfg inference.GenerateConfig
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		optionsBenchSinkMetalCfg = inferenceGenerateConfigToMetal(cfg)
	}
}

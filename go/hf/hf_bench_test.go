// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for the HuggingFace fit-planning + architecture-name
// classifier surface.
// Per AX-11 — PlanFits is the local-cache walker every "what models do
// I have / can I run" call hits. The architecture classifier fires per
// candidate model (search results return 10s, lists return 100s).
// InferJANG runs on every JANG/JANGTQ pack discovered.
//
// Run:    go test -bench=Benchmark -benchmem -run='^$' ./go/hf

package hf

import (
	"context"
	"testing"
)

// Sinks defeat compiler DCE.
var (
	hfSinkString string
	hfSinkInt    int
	hfSinkBool   bool
	hfSinkFit    *FitReport
	hfSinkErr    error
	hfSinkU64    uint64
)

// --- ModelConfig.architecture / contextLength / quantization helpers ---

func BenchmarkHF_ModelConfig_Architecture_Qwen3(b *testing.B) {
	config := ModelConfig{
		ModelType:     "qwen3",
		Architectures: []string{"Qwen3ForCausalLM"},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		hfSinkString = modelConfigArchitecture(config)
	}
}

func BenchmarkHF_ModelConfig_Architecture_NestedText(b *testing.B) {
	config := ModelConfig{
		ModelType: "qwen3_5",
		TextConfig: &ModelConfig{
			ModelType:     "qwen3_next",
			Architectures: []string{"Qwen3NextForCausalLM"},
		},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		hfSinkString = modelConfigArchitecture(config)
	}
}

func BenchmarkHF_ModelConfig_ContextLength(b *testing.B) {
	config := ModelConfig{
		ContextLength:         0,
		MaxPositionEmbeddings: 40960,
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		hfSinkInt = modelConfigContextLength(config)
	}
}

func BenchmarkHF_ModelConfig_Quantization(b *testing.B) {
	config := ModelConfig{
		QuantizationConfig: &QuantizationConfig{Bits: 4, GroupSize: 64, Type: "affine"},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		bits, group := modelConfigQuantization(config)
		hfSinkInt = bits + group
	}
}

// --- weightFormatAndBytes ---

type benchFitSource struct {
	meta ModelMetadata
}

func (s *benchFitSource) SearchModels(_ context.Context, _ string, _ int) ([]ModelMetadata, error) {
	return []ModelMetadata{s.meta}, nil
}

func (s *benchFitSource) ModelMetadata(_ context.Context, _ string) (ModelMetadata, error) {
	return s.meta, nil
}

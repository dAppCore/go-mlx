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
	"testing"
)

func BenchmarkHF_InferJANG_JANGTQ(b *testing.B) {
	meta := ModelMetadata{
		ID:   "dealignai/MiniMax-M2.7-JANGTQ-CRACK",
		Tags: []string{"mlx", "jang", "jangtq", "minimax_m2"},
		Files: []ModelFile{
			{Name: "model-00001-of-00061.safetensors"},
			{Name: "jangtq_runtime.safetensors"},
		},
		Config: ModelConfig{
			QuantizationConfig: &QuantizationConfig{GroupSize: 64},
		},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		info := InferJANG(meta)
		if info != nil {
			hfSinkString = info.Profile
		}
	}
}

func BenchmarkHF_InferJANG_Miss(b *testing.B) {
	meta := ModelMetadata{
		ID:    "Qwen/Qwen3-0.6B",
		Tags:  []string{"mlx", "text-generation"},
		Files: []ModelFile{{Name: "model.safetensors"}},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		info := InferJANG(meta)
		hfSinkBool = info != nil
	}
}

// --- PlanFits — end-to-end against a fake source (no network) ---

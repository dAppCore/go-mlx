// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for model_slice.go — tensor-name classification. Per AX-11 —
// classifyTensor fires per tensor during model load (a Gemma-class model
// has 1000+ tensor refs). Moved from root_bench_test.go in the orphan sweep.
//
// Run:    go test -bench='BenchmarkModelSlice' -benchmem -run='^$' ./go

package mlx

import "testing"

var rootBenchTensorNames = []string{
	"model.embed_tokens.weight",
	"model.layers.0.input_layernorm.weight",
	"model.layers.0.self_attn.q_proj.weight",
	"model.layers.0.self_attn.k_proj.weight",
	"model.layers.0.self_attn.v_proj.weight",
	"model.layers.0.self_attn.o_proj.weight",
	"model.layers.0.post_attention_layernorm.weight",
	"model.layers.0.mlp.gate_proj.weight",
	"model.layers.0.mlp.up_proj.weight",
	"model.layers.0.mlp.down_proj.weight",
	"model.layers.0.mlp.experts.0.gate_proj.weight",
	"model.layers.0.mlp.experts.0.up_proj.weight",
	"model.layers.0.mlp.experts.0.down_proj.weight",
	"model.layers.0.mlp.gate.weight",
	"model.norm.weight",
	"lm_head.weight",
}

func BenchmarkModelSlice_ClassifyTensor_Embedding(b *testing.B) {
	name := "model.embed_tokens.weight"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rootBenchBool = modelSliceTensorIsEmbedding(name)
	}
}

func BenchmarkModelSlice_ClassifyTensor_Attention(b *testing.B) {
	name := "model.layers.12.self_attn.q_proj.weight"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rootBenchBool = modelSliceTensorIsAttention(name)
	}
}

func BenchmarkModelSlice_ClassifyTensor_FFN(b *testing.B) {
	name := "model.layers.12.mlp.gate_proj.weight"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rootBenchBool = modelSliceTensorIsFFN(name)
	}
}

func BenchmarkModelSlice_ClassifyTensor_Expert(b *testing.B) {
	name := "model.layers.5.mlp.experts.7.down_proj.weight"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rootBenchBool = modelSliceTensorIsExpert(name)
	}
}

// Models with miss-paths (negative result, must scan whole substring set)
// exercise the worst-case branch — every contains/suffix check pays.
func BenchmarkModelSlice_ClassifyTensor_NotAttention(b *testing.B) {
	name := "model.layers.12.mlp.gate_proj.weight"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rootBenchBool = modelSliceTensorIsAttention(name)
	}
}

// Full-pass over the representative name set — proxy for the inner
// loop of SliceModel/inspectModelSliceIfPresent.
func BenchmarkModelSlice_ClassifySweep_AllTensors(b *testing.B) {
	names := rootBenchTensorNames
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for _, name := range names {
			rootBenchBool = modelSliceTensorIsEmbedding(name) ||
				modelSliceTensorIsAttention(name) ||
				modelSliceTensorIsFFN(name) ||
				modelSliceTensorIsGate(name) ||
				modelSliceTensorIsRouter(name) ||
				modelSliceTensorIsExpert(name) ||
				modelSliceTensorIsLMHead(name) ||
				modelSliceTensorIsNorm(name)
		}
	}
}

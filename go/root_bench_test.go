// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for root-package mlx primitives — shape coercion + tensor
// name classifiers. Per AX-11 — both fire per tensor during model
// load (a Gemma-class model has 1000+ tensor refs), so a few hundred
// nanoseconds per call matters.
//
// Run:    go test -bench='BenchmarkShape|BenchmarkModelSlice' -benchmem -run='^$' ./go

package mlx

import "testing"

// Sinks defeat compiler DCE.
var (
	rootBenchShape []int32
	rootBenchInt32 int32
	rootBenchBool  bool
)

// --- Shape normalisation (shape.go) ---

func BenchmarkShape_NormalizeShapeArgs_Empty(b *testing.B) {
	args := []any{}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rootBenchShape = normalizeRootShapeArgs(args)
	}
}

func BenchmarkShape_NormalizeShapeArgs_IntSlice4D(b *testing.B) {
	args := []any{[]int{4, 28, 2048, 64}}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rootBenchShape = normalizeRootShapeArgs(args)
	}
}

// 4D variadic (the common per-tensor call shape).
func BenchmarkShape_NormalizeShapeArgs_Variadic4D(b *testing.B) {
	args := []any{4, 28, 2048, 64}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rootBenchShape = normalizeRootShapeArgs(args)
	}
}

func BenchmarkShape_NormalizeShapeArgs_Int32SliceFastPath(b *testing.B) {
	dims := []int32{4, 28, 2048, 64}
	args := []any{dims}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rootBenchShape = normalizeRootShapeArgs(args)
	}
}

func BenchmarkShape_NormalizeInt32Arg_Int(b *testing.B) {
	value := any(2048)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rootBenchInt32 = normalizeRootInt32Arg("shape", value)
	}
}

func BenchmarkShape_NormalizeInt32Arg_Int64(b *testing.B) {
	value := any(int64(2048))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rootBenchInt32 = normalizeRootInt32Arg("shape", value)
	}
}

// --- Tensor-name classifiers (model_slice.go) ---
// Fired per tensor ref during SliceModel + inspection. With 1000+ refs
// per model the per-call substring scan adds up.

// Names representative of the qwen3/gemma-class checkpoint layout.
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

// SPDX-Licence-Identifier: EUPL-1.2

package m2

import (
	"testing"

	"dappco.re/go/mlx/safetensors"
)

// BenchmarkPackedWeightCandidates covers the per-spec weight-name fan-out
// for the packed projection (canonical + .packed + .qweight variants).
func BenchmarkPackedWeightCandidates(b *testing.B) {
	spec := TensorSpec{
		Name:    "model.layers.0.block_sparse_moe.experts.7.gate_proj.weight",
		Aliases: []string{"model.layers.0.mlp.experts.7.gate_proj.weight"},
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = packedWeightCandidates(&spec)
	}
}

// BenchmarkRouterBiasCandidates covers the per-call layer/prefix string
// build path used when resolving the routing correction bias tensor.
func BenchmarkRouterBiasCandidates(b *testing.B) {
	spec := TensorSpec{
		Name:    "model.layers.0.block_sparse_moe.e_score_correction_bias",
		Aliases: []string{"model.layers.0.mlp.e_score_correction_bias"},
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = routerBiasCandidates(&spec, 17)
	}
}

// BenchmarkFindProjectionBiasRef_Miss measures the inline projection-bias
// walk against the typical case where the optional bias is absent — the
// full fan-out runs but no candidate slice is built. This is the dominant
// shape at MiniMax M2 load (projection bias is rare).
func BenchmarkFindProjectionBiasRef_Miss(b *testing.B) {
	spec := TensorSpec{
		Name:    "model.layers.0.block_sparse_moe.experts.7.gate_proj.weight",
		Aliases: []string{"model.layers.0.mlp.experts.7.gate_proj.weight"},
	}
	weightName := "model.layers.0.block_sparse_moe.experts.7.gate_proj.weight.packed"
	index := safetensors.Index{Tensors: map[string]safetensors.TensorRef{}}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _, _ = findProjectionBiasRef(index, &spec, weightName)
	}
}

// BenchmarkFindPackedWeightRef_Hit measures the inline weight-name walk
// against the canonical-layout case where spec.Name resolves on the very
// first probe. Mirrors BenchmarkFindSidecarRef_Hit for the loader's
// other primary lookup.
func BenchmarkFindPackedWeightRef_Hit(b *testing.B) {
	spec := TensorSpec{
		Name:    "model.layers.0.block_sparse_moe.experts.7.gate_proj.weight",
		Aliases: []string{"model.layers.0.mlp.experts.7.gate_proj.weight"},
	}
	index := safetensors.Index{
		Tensors: map[string]safetensors.TensorRef{
			spec.Name: {Name: spec.Name, DType: "U8"},
		},
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _, _ = findPackedWeightRef(index, &spec)
	}
}

// BenchmarkFindPackedWeightRef_Miss measures the full fan-out worst
// case to confirm the inline pattern stays competitive on total-miss
// searches.
func BenchmarkFindPackedWeightRef_Miss(b *testing.B) {
	spec := TensorSpec{
		Name:    "model.layers.0.block_sparse_moe.experts.7.gate_proj.weight",
		Aliases: []string{"model.layers.0.mlp.experts.7.gate_proj.weight"},
	}
	index := safetensors.Index{Tensors: map[string]safetensors.TensorRef{}}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _, _ = findPackedWeightRef(index, &spec)
	}
}

// BenchmarkFindSidecarRef_Hit measures the inline candidate-walk pattern
// when the canonical weightName+"."+sidecar entry resolves first — the
// production-load shape where checkpoints carry the standard layout. The
// goal is to expose that the inline path avoids the allocation of the
// transient candidate slice when the hit lands on the first probe.
func BenchmarkFindSidecarRef_Hit(b *testing.B) {
	spec := TensorSpec{
		Name:    "model.layers.0.block_sparse_moe.experts.7.gate_proj.weight",
		Aliases: []string{"model.layers.0.mlp.experts.7.gate_proj.weight"},
	}
	weightName := "model.layers.0.block_sparse_moe.experts.7.gate_proj.weight.packed"
	index := safetensors.Index{
		Tensors: map[string]safetensors.TensorRef{
			weightName + ".scales": {Name: weightName + ".scales", DType: "F32"},
		},
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _, _ = findSidecarRef(index, &spec, weightName, "scales")
	}
}

// BenchmarkFindSidecarRef_Miss measures the worst case where every
// candidate fails — exercises the full fan-out to confirm the inline
// pattern doesn't regress against the slice-based predecessor on a
// total-miss search.
func BenchmarkFindSidecarRef_Miss(b *testing.B) {
	spec := TensorSpec{
		Name:    "model.layers.0.block_sparse_moe.experts.7.gate_proj.weight",
		Aliases: []string{"model.layers.0.mlp.experts.7.gate_proj.weight"},
	}
	weightName := "model.layers.0.block_sparse_moe.experts.7.gate_proj.weight.packed"
	index := safetensors.Index{Tensors: map[string]safetensors.TensorRef{}}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _, _ = findSidecarRef(index, &spec, weightName, "scales")
	}
}

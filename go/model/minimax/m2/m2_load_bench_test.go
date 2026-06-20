// SPDX-Licence-Identifier: EUPL-1.2

package m2

import (
	"testing"

	core "dappco.re/go"
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

// --- Expert dequant / load path (per-expert, the MoE-load multiplier) ---

// miniMaxM2ExpertIDs returns [0,n) for benchmark expert fan-out.
func miniMaxM2ExpertIDs(n int) []int {
	ids := make([]int, n)
	for i := range ids {
		ids[i] = i
	}
	return ids
}

// BenchmarkDequantizeJANGPackedProjection drives the single-projection
// JANG dequant expansion over a small affine-packed fixture. The dequant
// output buffer is allocated and retained inside jang.DequantizePackedTensor
// (the projection's dense weight), so this measures the m2-owned wrapper
// overhead (the optional bias clone) on top of the jang-owned output.
func BenchmarkDequantizeJANGPackedProjection(b *testing.B) {
	projection := miniMaxM2PackedProjectionFixture(b, "gate_proj", []uint8{0, 1, 2, 3})
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := DequantizeJANGPackedProjection(projection); err != nil {
			b.Fatalf("DequantizeJANGPackedProjection() error = %v", err)
		}
	}
}

// BenchmarkLazyExpertLoad_DequantizedExperts expands a whole loaded
// expert set (gate/up/down per expert) to dense form. Sub-benched by
// expert count so the per-expert dequant multiplier is visible rather
// than amortised into a single call.
func BenchmarkLazyExpertLoad_DequantizedExperts(b *testing.B) {
	for _, n := range []int{1, 8} {
		experts := make(map[int]PackedExpertWeights, n)
		for e := 0; e < n; e++ {
			experts[e] = miniMaxM2PackedExpertFixture(b,
				[]uint8{0, 1, 2, 3}, []uint8{1, 1, 1, 1}, []uint8{3, 2, 1, 0})
		}
		load := LazyExpertLoad{Experts: experts}
		b.Run(core.Sprintf("experts=%d", n), func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if _, err := load.DequantizedExperts(); err != nil {
					b.Fatalf("DequantizedExperts() error = %v", err)
				}
			}
		})
	}
}

// BenchmarkLoadPackedExperts resolves and reads N routed experts from a
// safetensors fixture. This is the per-expert load hot path: each expert
// re-derives its tensor specs and reads packed bytes + affine sidecars.
// Sub-benched by expert count to expose the per-expert spec/descriptor
// build multiplier.
func BenchmarkLoadPackedExperts(b *testing.B) {
	for _, n := range []int{1, 8} {
		ids := miniMaxM2ExpertIDs(n)
		plan := miniMaxM2MultiExpertPlan(b, n)
		weights := core.PathJoin(b.TempDir(), "model.safetensors")
		writeMiniMaxM2RawSafetensors(b, weights, miniMaxM2MultiExpertFixtureTensors(b, n, ids, []uint8{0, 1, 2, 3}))
		files := []string{weights}
		b.Run(core.Sprintf("experts=%d", n), func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if _, err := LoadPackedExperts(plan, files, 0, ids); err != nil {
					b.Fatalf("LoadPackedExperts() error = %v", err)
				}
			}
		})
	}
}

// BenchmarkLoadPackedExpertsForDecisions drives the decision→expert-id
// reduction wrapper plus the expert load for a routed token batch. Each
// token routes one expert; the wrapper flattens + dedups the ids before
// delegating to LoadPackedExperts.
func BenchmarkLoadPackedExpertsForDecisions(b *testing.B) {
	for _, n := range []int{1, 8} {
		ids := miniMaxM2ExpertIDs(n)
		plan := miniMaxM2MultiExpertPlan(b, n)
		weights := core.PathJoin(b.TempDir(), "model.safetensors")
		writeMiniMaxM2RawSafetensors(b, weights, miniMaxM2MultiExpertFixtureTensors(b, n, ids, []uint8{0, 1, 2, 3}))
		files := []string{weights}
		decisions := make([]RouterDecision, n)
		for t := range decisions {
			decisions[t] = RouterDecision{TokenIndex: t, ExpertIDs: []int{ids[t]}, Weights: []float32{1}}
		}
		b.Run(core.Sprintf("experts=%d", n), func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if _, err := LoadPackedExpertsForDecisions(plan, files, 0, decisions); err != nil {
					b.Fatalf("LoadPackedExpertsForDecisions() error = %v", err)
				}
			}
		})
	}
}

// BenchmarkLoadLazyExpertsForHidden drives the full lazy load: router read,
// score projection, top-k routing, routed-expert load, and probe-event
// emission. The single-expert small plan mirrors the existing functional
// fixture (only expert 2 routes for hidden {1,0}).
func BenchmarkLoadLazyExpertsForHidden(b *testing.B) {
	plan := miniMaxM2SmallJANGTQPlan(b)
	weights := core.PathJoin(b.TempDir(), "model.safetensors")
	writeMiniMaxM2RawSafetensors(b, weights, miniMaxM2LazyExpertFixtureTensors(b, 2, []uint8{0, 1, 2, 3}))
	files := []string{weights}
	hidden := [][]float32{{1, 0}}
	tokenIDs := []int32{42}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := LoadLazyExpertsForHidden(plan, files, 0, hidden, tokenIDs, nil); err != nil {
			b.Fatalf("LoadLazyExpertsForHidden() error = %v", err)
		}
	}
}

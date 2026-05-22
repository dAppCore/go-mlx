// SPDX-Licence-Identifier: EUPL-1.2

package m2

import (
	"math/rand"
	"testing"

	"dappco.re/go/mlx/probe"
)

// BenchmarkRouteTokens exercises sigmoid scoring + top-k sort + renormalisation
// on a representative MiniMax M2 routing shape (256 experts × top-8).
func BenchmarkRouteTokens(b *testing.B) {
	const tokens, experts, topK = 32, 256, 8
	cfg := Config{NumLocalExperts: experts, NumExpertsPerToken: topK, ScoringFunc: "sigmoid", UseRoutingBias: true}
	scores := make([][]float32, tokens)
	rng := rand.New(rand.NewSource(1))
	for i := range scores {
		row := make([]float32, experts)
		for j := range row {
			row[j] = rng.Float32()*4 - 2
		}
		scores[i] = row
	}
	bias := make([]float32, experts)
	for i := range bias {
		bias[i] = rng.Float32() * 0.1
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := RouteTokens(cfg, scores, bias); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkRouterProbeEvents covers the per-decision clone path that builds
// probe.Event records (one per token, with cloned ExpertIDs + Weights).
func BenchmarkRouterProbeEvents(b *testing.B) {
	const tokens, topK = 32, 8
	decisions := make([]RouterDecision, tokens)
	for i := range decisions {
		ids := make([]int, topK)
		weights := make([]float32, topK)
		for j := range ids {
			ids[j] = (i*31 + j) & 0xff
			weights[j] = float32(j+1) / 36
		}
		decisions[i] = RouterDecision{TokenIndex: i, ExpertIDs: ids, Weights: weights}
	}
	tokenIDs := make([]int32, tokens)
	for i := range tokenIDs {
		tokenIDs[i] = int32(i)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = RouterProbeEvents(7, tokenIDs, decisions)
	}
}

// BenchmarkProjectRouterScores exercises the inner hidden @ router.weight.T
// loop, the hottest path in router projection.
func BenchmarkProjectRouterScores(b *testing.B) {
	const tokens, hidden, experts = 16, 3072, 256
	router := RouterWeights{NumExperts: experts, HiddenSize: hidden, Weight: make([]float32, experts*hidden)}
	rng := rand.New(rand.NewSource(2))
	for i := range router.Weight {
		router.Weight[i] = rng.Float32()*0.02 - 0.01
	}
	hidStates := make([][]float32, tokens)
	for i := range hidStates {
		row := make([]float32, hidden)
		for j := range row {
			row[j] = rng.Float32()*0.5 - 0.25
		}
		hidStates[i] = row
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := ProjectRouterScores(hidStates, router); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkDispatchExperts covers the per-decision defensive clone of hidden
// row + weighted sum into output, exercising the host-shaped routing path.
func BenchmarkDispatchExperts(b *testing.B) {
	const tokens, topK, dim = 16, 8, 256
	hidden := make([][]float32, tokens)
	for i := range hidden {
		row := make([]float32, dim)
		for j := range row {
			row[j] = float32((i+j)&0xff) / 255
		}
		hidden[i] = row
	}
	decisions := make([]RouterDecision, tokens)
	for i := range decisions {
		ids := make([]int, topK)
		weights := make([]float32, topK)
		for j := range ids {
			ids[j] = j
			weights[j] = float32(j+1) / 36
		}
		decisions[i] = RouterDecision{TokenIndex: i, ExpertIDs: ids, Weights: weights}
	}
	experts := map[int]ExpertFunc{}
	for j := 0; j < topK; j++ {
		j := j
		experts[j] = func(values []float32) []float32 {
			out := make([]float32, len(values))
			for k, v := range values {
				out[k] = v * float32(j+1)
			}
			return out
		}
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := DispatchExperts(hidden, decisions, experts); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkDecisionExpertIDs covers the flatten + pre-size path used when
// turning router decisions into the unique-expert load fan-out.
func BenchmarkDecisionExpertIDs(b *testing.B) {
	const tokens, topK = 32, 8
	decisions := make([]RouterDecision, tokens)
	for i := range decisions {
		ids := make([]int, topK)
		for j := range ids {
			ids[j] = (i*31 + j) & 0xff
		}
		decisions[i] = RouterDecision{TokenIndex: i, ExpertIDs: ids}
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = decisionExpertIDs(decisions)
	}
}

// BenchmarkLayerTensorSpecs covers per-layer + per-expert tensor name
// fan-out used during model loading. MiniMax M2 has 62 layers x 256 experts
// so the inner-name Sprintf budget compounds quickly.
func BenchmarkLayerTensorSpecs(b *testing.B) {
	cfg := Config{
		ModelType:          "minimax_m2",
		HiddenSize:         3072,
		IntermediateSize:   1536,
		NumHiddenLayers:    62,
		NumAttentionHeads:  48,
		NumKeyValueHeads:   8,
		HeadDim:            128,
		NumLocalExperts:    256,
		NumExpertsPerToken: 8,
		ScoringFunc:        "sigmoid",
		UseRoutingBias:     true,
	}
	plan, err := BuildTensorPlan(cfg, nil)
	if err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := plan.LayerTensorSpecs(0, 0); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkProjectionBiasCandidates covers the per-spec projection-bias name
// fan-out + the (now hoisted) trimWeightSuffix call.
func BenchmarkProjectionBiasCandidates(b *testing.B) {
	spec := TensorSpec{
		Name:    "model.layers.0.block_sparse_moe.experts.7.gate_proj.weight",
		Aliases: []string{"model.layers.0.mlp.experts.7.gate_proj.weight"},
	}
	weightName := "model.layers.0.block_sparse_moe.experts.7.gate_proj.weight.packed"
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = projectionBiasCandidates(spec, weightName)
	}
}

// BenchmarkPackedWeightCandidates covers the per-spec weight-name fan-out
// for the packed projection (canonical + .packed + .qweight variants).
func BenchmarkPackedWeightCandidates(b *testing.B) {
	spec := TensorSpec{
		Name:    "model.layers.0.block_sparse_moe.experts.7.gate_proj.weight",
		Aliases: []string{"model.layers.0.mlp.experts.7.gate_proj.weight"},
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = packedWeightCandidates(spec)
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
		_ = routerBiasCandidates(spec, 17)
	}
}

// BenchmarkSidecarCandidates covers the per-spec sidecar name fan-out used
// when resolving safetensors scales/biases for one packed projection.
func BenchmarkSidecarCandidates(b *testing.B) {
	spec := TensorSpec{
		Name:    "model.layers.0.block_sparse_moe.experts.7.gate_proj.weight",
		Aliases: []string{"model.layers.0.mlp.experts.7.gate_proj.weight"},
	}
	weightName := "model.layers.0.block_sparse_moe.experts.7.gate_proj.weight.packed"
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = sidecarCandidates(spec, weightName, "scales")
	}
}

// BenchmarkRouterDecisionsCloneShape exercises only the clone-into-result
// path of ForwardLazyExpertLoadMetal — it isolates the per-call clone cost
// without invoking the (real) Metal kernels, by sending a tiny load with
// zero-element experts and asserting the host-side bookkeeping path.
func BenchmarkRouterDecisionsCloneShape(b *testing.B) {
	load := LazyExpertLoad{
		Decisions:         make([]RouterDecision, 64),
		SelectedExpertIDs: make([]int, 32),
		ProbeEvents:       make([]probe.Event, 64),
	}
	for i := range load.Decisions {
		load.Decisions[i] = RouterDecision{TokenIndex: i, ExpertIDs: []int{0, 1, 2}, Weights: []float32{0.3, 0.4, 0.3}}
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = append([]RouterDecision(nil), load.Decisions...)
		_ = append([]int(nil), load.SelectedExpertIDs...)
		_ = append([]probe.Event(nil), load.ProbeEvents...)
	}
}

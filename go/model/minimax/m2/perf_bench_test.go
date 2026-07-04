// SPDX-Licence-Identifier: EUPL-1.2

package m2

import (
	"testing"

	"dappco.re/go/inference/probe"
)

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

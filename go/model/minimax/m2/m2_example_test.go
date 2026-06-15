// SPDX-Licence-Identifier: EUPL-1.2

package m2

import (
	"fmt"
)

// ExampleResolvedTensor_EstimatedBytes shows a dense tensor's byte estimate
// from its dtype width and shape.
func ExampleResolvedTensor_EstimatedBytes() {
	tensor := ResolvedTensor{DType: "f32", Shape: []uint64{2, 3}}
	fmt.Println(tensor.EstimatedBytes())
	// Output: 24
}

// ExampleResolvedTensor_EstimatedBytes_packed shows that a packed tensor
// reports its packed byte count and ignores the dtype/shape estimate.
func ExampleResolvedTensor_EstimatedBytes_packed() {
	tensor := ResolvedTensor{DType: "f32", Shape: []uint64{2, 3}, PackedBytes: 7}
	fmt.Println(tensor.EstimatedBytes())
	// Output: 7
}

// ExampleLayerForwardSkeleton_EstimatedBytes sums the router gate, attention
// projections, and optional router bias bytes a first forward pass needs.
func ExampleLayerForwardSkeleton_EstimatedBytes() {
	bias := ResolvedTensor{DType: "f32", Shape: []uint64{3}}
	skeleton := LayerForwardSkeleton{
		RouterGate: ResolvedTensor{DType: "f32", Shape: []uint64{3, 4}},
		Attention: []ResolvedTensor{
			{PackedBytes: 16},
			{PackedBytes: 8},
		},
		RouterBias: &bias,
	}
	fmt.Println(skeleton.EstimatedBytes())
	// Output: 84
}

// ExampleParseConfig parses the MiniMax M2 config subset and shows the
// defaulted scoring function.
func ExampleParseConfig() {
	cfg, err := ParseConfig([]byte(`{
		"model_type": "minimax_m2",
		"hidden_size": 3072,
		"num_local_experts": 256,
		"num_experts_per_tok": 8
	}`))
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Println(cfg.ModelType, cfg.HiddenSize, cfg.NumLocalExperts, cfg.ScoringFunc)
	// Output: minimax_m2 3072 256 sigmoid
}

// ExampleRouteTokens computes a deterministic top-k routing decision: scores
// are sigmoid-normalised and the top-k weights are renormalised to sum to 1.
func ExampleRouteTokens() {
	cfg := Config{NumLocalExperts: 4, NumExpertsPerToken: 2, ScoringFunc: "sigmoid", UseRoutingBias: true}
	decisions, err := RouteTokens(cfg, [][]float32{{0, 2, 1, -1}}, []float32{0, 0, 0, 4})
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Println(decisions[0].ExpertIDs)
	// Output: [3 1]
}

// ExampleDispatchExperts applies weighted expert functions to hidden states.
// Expert 0 multiplies by 10, expert 1 by 2; with weights 0.75/0.25 the
// first token's output is 0.75*[10,20] + 0.25*[2,4] = [8 16].
func ExampleDispatchExperts() {
	hidden := [][]float32{{1, 2}}
	decisions := []RouterDecision{{
		TokenIndex: 0,
		ExpertIDs:  []int{1, 0},
		Weights:    []float32{0.25, 0.75},
	}}
	experts := map[int]ExpertFunc{
		0: func(v []float32) []float32 { return []float32{v[0] * 10, v[1] * 10} },
		1: func(v []float32) []float32 { return []float32{v[0] * 2, v[1] * 2} },
	}
	out, err := DispatchExperts(hidden, decisions, experts)
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Println(out[0])
	// Output: [8 16]
}

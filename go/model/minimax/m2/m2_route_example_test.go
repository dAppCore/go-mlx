// SPDX-Licence-Identifier: EUPL-1.2

package m2

import (
	"fmt"
)

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

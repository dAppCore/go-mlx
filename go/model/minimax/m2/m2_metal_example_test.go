// SPDX-Licence-Identifier: EUPL-1.2

package m2

import (
	"fmt"
)

// ExampleDispatchPackedExpertsMetal shows the pre-dispatch guard: a decision
// whose token index is outside the hidden batch is rejected before any fused
// Metal kernel runs, so the entry point is documented without a device.
func ExampleDispatchPackedExpertsMetal() {
	_, err := DispatchPackedExpertsMetal([][]float32{{1, 2}}, []RouterDecision{{
		TokenIndex: 5,
		ExpertIDs:  []int{0},
		Weights:    []float32{1},
	}}, nil)
	fmt.Println(err)
	// Output: mlx: MiniMax M2 packed dispatch token index 5 out of range
}

// ExampleDispatchPackedExpertsFromSafetensorsMetal shows the loader guard: with
// no safetensors weight files the experts can't be resolved, so the call fails
// before any dispatch — device-free and fixture-free.
func ExampleDispatchPackedExpertsFromSafetensorsMetal() {
	plan, err := BuildTensorPlan(Config{
		ModelType: "minimax_m2", HiddenSize: 2, IntermediateSize: 2,
		NumHiddenLayers: 1, NumAttentionHeads: 1, NumKeyValueHeads: 1,
		HeadDim: 2, NumLocalExperts: 3, NumExpertsPerToken: 1,
	}, nil)
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	_, err = DispatchPackedExpertsFromSafetensorsMetal(plan, nil, 0, [][]float32{{1, 0}}, []RouterDecision{
		{TokenIndex: 0, ExpertIDs: []int{0}, Weights: []float32{1}},
	})
	fmt.Println(err)
	// Output: mlx: MiniMax M2 packed expert loading requires safetensors weight files
}

// ExampleForwardLazyExpertLoadMetal shows the dispatch guard reached through the
// lazy-load forward entry: an expert-id/weight length mismatch is caught before
// any projection kernel.
func ExampleForwardLazyExpertLoadMetal() {
	_, err := ForwardLazyExpertLoadMetal([][]float32{{1, 2}}, LazyExpertLoad{
		Decisions: []RouterDecision{{
			TokenIndex: 0,
			ExpertIDs:  []int{0, 1},
			Weights:    []float32{1},
		}},
	})
	fmt.Println(err)
	// Output: mlx: MiniMax M2 packed dispatch expert/weight length mismatch
}

// ExampleForwardPackedLayerMetal shows the shape pre-check: the hidden batch and
// the router-score batch must have the same number of rows, validated before
// routing or any Metal dispatch.
func ExampleForwardPackedLayerMetal() {
	_, err := ForwardPackedLayerMetal(PackedLayerForwardOptions{
		Hidden:       [][]float32{{1, 2}, {3, 4}},
		RouterScores: [][]float32{{0, 1, 2}},
	})
	fmt.Println(err)
	// Output: mlx: MiniMax M2 packed layer hidden rows 2, router rows 1
}

// ExampleForwardPackedLayerFromSafetensorsMetal shows the lazy no-bias branch
// guard: with no router bias supplied the function loads experts for the hidden
// states, and empty weight files make that load fail before any dispatch.
func ExampleForwardPackedLayerFromSafetensorsMetal() {
	plan, err := BuildTensorPlan(Config{
		ModelType: "minimax_m2", HiddenSize: 2, IntermediateSize: 2,
		NumHiddenLayers: 1, NumAttentionHeads: 1, NumKeyValueHeads: 1,
		HeadDim: 2, NumLocalExperts: 3, NumExpertsPerToken: 1,
	}, nil)
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	_, err = ForwardPackedLayerFromSafetensorsMetal(PackedLayerForwardOptions{
		Plan:   plan,
		Hidden: [][]float32{{1, 0}},
	})
	fmt.Println(err)
	// Output: mlx: MiniMax M2 router loading requires safetensors weight files
}

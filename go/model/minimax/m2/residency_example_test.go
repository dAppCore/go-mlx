// SPDX-Licence-Identifier: EUPL-1.2

package m2

import (
	"fmt"

	"dappco.re/go/inference/quant/jang"
	"dappco.re/go/mlx/memory"
)

// ExamplePlanResidency derives a lazy expert residency policy for a small
// MiniMax M2 plan on a 96 GB Apple machine. Hot expert hints are sorted and
// deduplicated for reproducible state bundles, and a lazy plan pages cold
// experts in on first use rather than loading all experts at startup.
func ExamplePlanResidency() {
	tensorPlan, err := BuildTensorPlan(Config{
		ModelType:          "minimax_m2",
		HiddenSize:         4,
		IntermediateSize:   8,
		NumHiddenLayers:    1,
		NumAttentionHeads:  2,
		NumKeyValueHeads:   1,
		HeadDim:            2,
		NumLocalExperts:    16,
		NumExpertsPerToken: 2,
	}, &jang.Info{
		Profile:          "JANGTQ",
		WeightFormat:     "mxtq",
		Method:           "affine+mxtq",
		GroupSize:        4,
		BitsDefault:      2,
		RoutedExpertBits: 2,
	})
	if err != nil {
		fmt.Println("error:", err)
		return
	}

	plan := PlanResidency(tensorPlan, memory.Plan{
		MachineClass: memory.ClassApple96GB,
	}, []int{5, 3, 5, 1, 9})

	fmt.Println(plan.Mode)
	fmt.Println(plan.MaxResidentExperts)
	fmt.Println(plan.StartupExpertIDs)
	// Output:
	// lazy
	// 8
	// [1 3 5 9]
}

// ExampleTensorPlan_EstimatedPackedExpertBytes shows the per-expert packed
// payload estimate derived from tensor descriptors (sidecars excluded).
func ExampleTensorPlan_EstimatedPackedExpertBytes() {
	plan, err := BuildTensorPlan(Config{
		ModelType:          "minimax_m2",
		HiddenSize:         4,
		IntermediateSize:   8,
		NumHiddenLayers:    1,
		NumAttentionHeads:  2,
		NumKeyValueHeads:   1,
		HeadDim:            2,
		NumLocalExperts:    16,
		NumExpertsPerToken: 2,
	}, &jang.Info{
		Profile:          "JANGTQ",
		WeightFormat:     "mxtq",
		Method:           "affine+mxtq",
		GroupSize:        4,
		BitsDefault:      2,
		RoutedExpertBits: 2,
	})
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Println(plan.EstimatedPackedExpertBytes() > 0)
	// Output: true
}

// SPDX-Licence-Identifier: EUPL-1.2

package m2

import (
	"context"
	"fmt"

	"dappco.re/go/inference/quant/jang"
	"dappco.re/go/mlx/memory"
)

// exampleResidencyExpert is a tiny fixed packed expert payload used by the
// residency examples so they run without a model or device.
func exampleResidencyExpert(expertID int) PackedExpertWeights {
	packed := []byte{byte(expertID)}
	return PackedExpertWeights{
		GateProj: JANGPackedProjectionTensor{Packed: packed},
		UpProj:   JANGPackedProjectionTensor{Packed: packed},
		DownProj: JANGPackedProjectionTensor{Packed: packed},
	}
}

// ExampleNewResidencyManager builds a lazy resident expert set that pre-loads
// its startup (hot) experts immediately via the supplied loader.
func ExampleNewResidencyManager() {
	manager, err := NewResidencyManager(context.Background(), ResidencyConfig{
		Layer: 0,
		Policy: memory.ExpertResidencyPlan{
			Enabled:            true,
			Mode:               memory.ExpertResidencyModeLazy,
			StartupExpertIDs:   []int{3, 1},
			MaxResidentExperts: 4,
		},
		Loader: func(_ context.Context, _ int, expertID int) (PackedExpertWeights, error) {
			return exampleResidencyExpert(expertID), nil
		},
	})
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Println(manager.ResidentExpertIDs())
	// Output: [1 3]
}

// ExampleResidencyManager_EnsureExperts pages a cold expert in on first use and
// reports the resulting page-in count.
func ExampleResidencyManager_EnsureExperts() {
	manager, err := NewResidencyManager(context.Background(), ResidencyConfig{
		Layer: 0,
		Policy: memory.ExpertResidencyPlan{
			Enabled:            true,
			Mode:               memory.ExpertResidencyModeLazy,
			StartupExpertIDs:   []int{1},
			MaxResidentExperts: 4,
		},
		Loader: func(_ context.Context, _ int, expertID int) (PackedExpertWeights, error) {
			return exampleResidencyExpert(expertID), nil
		},
	})
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	experts, stats, err := manager.EnsureExperts(context.Background(), []int{1, 2})
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Println(len(experts), stats.ColdLoads)
	// Output: 2 1
}

// ExampleResidencyManager_ResidentExpertIDs returns the sorted set of experts
// currently held resident.
func ExampleResidencyManager_ResidentExpertIDs() {
	manager, err := NewResidencyManager(context.Background(), ResidencyConfig{
		Layer: 0,
		Policy: memory.ExpertResidencyPlan{
			Enabled:            true,
			Mode:               memory.ExpertResidencyModeLazy,
			StartupExpertIDs:   []int{9, 2, 5},
			MaxResidentExperts: 8,
		},
		Loader: func(_ context.Context, _ int, expertID int) (PackedExpertWeights, error) {
			return exampleResidencyExpert(expertID), nil
		},
	})
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Println(manager.ResidentExpertIDs())
	// Output: [2 5 9]
}

// ExampleNormalisePlan repairs a partially-specified residency plan: hot IDs are
// sorted and deduplicated, the eviction policy defaults to LRU, and the resident
// cap is derived from the startup set.
func ExampleNormalisePlan() {
	plan := NormalisePlan(memory.ExpertResidencyPlan{
		Enabled:          true,
		HotExpertIDs:     []int{4, 1, 4},
		StartupExpertIDs: []int{2, 2, 7},
		ExpertsPerToken:  2,
	})
	fmt.Println(plan.HotExpertIDs)
	fmt.Println(plan.MaxResidentExperts)
	fmt.Println(plan.EvictionPolicy)
	// Output:
	// [1 4]
	// 2
	// lru
}

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

// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

// Runnable examples for the Apple-only metal FFN memory augmenter. The
// constructor and SetClusterIDs examples print deterministic cluster IDs with no
// GPU work; the AugmentFFNMemory example assumes a usable Metal device — the
// same assumption the Apple-only metal package makes everywhere else — and
// prints whether memory was applied.
package memorypretrain_test

import (
	"fmt"

	"dappco.re/go/mlx/memorypretrain"
	"dappco.re/go/mlx/pkg/metal"
)

func exampleAugmenterBank() (*memorypretrain.FFNMemoryBank, error) {
	return memorypretrain.NewFFNMemoryBank(memorypretrain.FFNMemoryConfig{
		HiddenSize:       2,
		Layers:           1,
		MemoryLevels:     []string{"1"},
		FFNMemoryTokens:  []int{1},
		NumClusters:      []int{2},
		AddedGenericSize: 1,
	})
}

// ExampleNewMetalFFNMemoryAugmenter builds a model-facing hook over an FFN
// memory bank. Passing no route selects the generic-memory fallback slot, which
// is the final cluster index at the level (two learned clusters plus one generic
// slot, so index 2).
func ExampleNewMetalFFNMemoryAugmenter() {
	bank, err := exampleAugmenterBank()
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	augmenter, err := memorypretrain.NewMetalFFNMemoryAugmenter(bank, nil)
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Println(augmenter.ClusterIDs)
	// Output: [2]
}

// ExampleMetalFFNMemoryAugmenter_SetClusterIDs swaps the selected route after
// construction, then restores the generic fallback by passing no IDs.
func ExampleMetalFFNMemoryAugmenter_SetClusterIDs() {
	bank, err := exampleAugmenterBank()
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	augmenter, err := memorypretrain.NewMetalFFNMemoryAugmenter(bank, nil)
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	if err := augmenter.SetClusterIDs([]int{1}); err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Println(augmenter.ClusterIDs)
	if err := augmenter.SetClusterIDs(nil); err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Println(augmenter.ClusterIDs)
	// Output:
	// [1]
	// [2]
}

// ExampleMetalFFNMemoryAugmenter_AugmentFFNMemory applies the selected memory to
// a one-token FFN output through the neutral metal hook and reports that the
// augmentation was applied. It assumes a usable Metal device.
func ExampleMetalFFNMemoryAugmenter_AugmentFFNMemory() {
	bank, err := exampleAugmenterBank()
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	augmenter, err := memorypretrain.NewMetalFFNMemoryAugmenter(bank, nil)
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	ffnOutput := metal.FromValues([]float32{5, 7}, 1, 1, 2)
	mlpInput := metal.FromValues([]float32{2, 4}, 1, 1, 2)
	defer metal.Free(ffnOutput, mlpInput)
	got, applied, err := augmenter.AugmentFFNMemory(0, ffnOutput, mlpInput)
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	defer metal.Free(got)
	fmt.Println(applied, got.Shape())
	// Output: true [1 1 2]
}

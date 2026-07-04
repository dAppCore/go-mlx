// SPDX-Licence-Identifier: EUPL-1.2

package memorypretrain_test

import (
	"fmt"

	"dappco.re/go/mlx/memorypretrain"
)

func exampleFFNMemoryBank() (*memorypretrain.FFNMemoryBank, error) {
	return memorypretrain.NewFFNMemoryBank(memorypretrain.FFNMemoryConfig{
		HiddenSize:       2,
		Layers:           1,
		MemoryLevels:     []string{"1", "2"},
		FFNMemoryTokens:  []int{1, 1},
		NumClusters:      []int{2, 3},
		AddedGenericSize: 1,
	})
}

// ExampleNewFFNMemoryBank allocates a hierarchical FFN memory table. A fresh
// bank starts with W3 zeroed, so applying it leaves the FFN output unchanged
// while reporting the route as applied.
func ExampleNewFFNMemoryBank() {
	bank, err := memorypretrain.NewFFNMemoryBank(memorypretrain.FFNMemoryConfig{
		HiddenSize:       2,
		Layers:           1,
		MemoryLevels:     []string{"1"},
		FFNMemoryTokens:  []int{1},
		NumClusters:      []int{2},
		AddedGenericSize: 1,
	})
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Println(bank.HiddenSize, len(bank.Layers), len(bank.Layers[0].Levels))
	// Output: 2 1 1
}

// ExampleFFNMemoryBank_AddToFFNOutput applies the generic memory slot to a
// one-token FFN output. A fresh bank's W3 is zero, so the output is unchanged and
// the stats report one applied level.
func ExampleFFNMemoryBank_AddToFFNOutput() {
	bank, err := memorypretrain.NewFFNMemoryBank(memorypretrain.FFNMemoryConfig{
		HiddenSize:       2,
		Layers:           1,
		MemoryLevels:     []string{"1"},
		FFNMemoryTokens:  []int{1},
		NumClusters:      []int{2},
		AddedGenericSize: 1,
	})
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	out, stats, err := bank.AddToFFNOutput(nil, []float32{1, 2}, []float32{3, 4}, 0, []int{2})
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Println(out, stats.LevelsApplied, stats.Applied)
	// Output: [1 2] 1 true
}

// ExampleFFNMemoryBank_ClusterCounts reports the selectable memory count per
// level, including the generic slot added after the learned clusters.
func ExampleFFNMemoryBank_ClusterCounts() {
	bank, err := exampleFFNMemoryBank()
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Println(bank.ClusterCounts())
	// Output: [3 4]
}

// ExampleFFNMemoryBank_GenericClusterIDs reports the generic-memory cluster IDs:
// the final cluster index at each level.
func ExampleFFNMemoryBank_GenericClusterIDs() {
	bank, err := exampleFFNMemoryBank()
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	ids, err := bank.GenericClusterIDs()
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Println(ids)
	// Output: [2 3]
}

// ExampleFFNMemoryBank_AddGenericToFFNOutput applies the generic fallback at each
// level and reports the selected cluster IDs alongside the applied flag.
func ExampleFFNMemoryBank_AddGenericToFFNOutput() {
	bank, err := exampleFFNMemoryBank()
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	_, ids, stats, err := bank.AddGenericToFFNOutput(nil, []float32{5, 7}, []float32{2, 4}, 0)
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Println(ids, stats.Applied)
	// Output: [2 3] true
}

// ExampleFFNMemoryBank_AddRoutedToFFNOutput routes a query through an offline
// clustering bank and applies the selected hierarchical memories, reporting the
// per-level cluster IDs the route resolved to.
func ExampleFFNMemoryBank_AddRoutedToFFNOutput() {
	router, err := memorypretrain.BuildBank([]memorypretrain.Block{
		{ID: "go-1", Embedding: []float32{1, 0}},
		{ID: "go-2", Embedding: []float32{0.9, 0.1}},
		{ID: "poem-1", Embedding: []float32{0, 1}},
		{ID: "poem-2", Embedding: []float32{0.1, 0.9}},
	}, memorypretrain.BuildConfig{BranchingFactor: 2, MaxDepth: 1, MinClusterSize: 1, KMeansIters: 8})
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	bank, err := memorypretrain.NewFFNMemoryBank(memorypretrain.FFNMemoryConfig{
		HiddenSize:       2,
		Layers:           1,
		MemoryLevels:     []string{"1"},
		FFNMemoryTokens:  []int{1},
		NumClusters:      []int{2},
		AddedGenericSize: 1,
	})
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	_, ids, stats, err := bank.AddRoutedToFFNOutput(nil, []float32{1, 2}, []float32{2, 4}, router, []float32{1, 0}, 0)
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Println(ids, stats.Applied)
	// Output: [0] true
}

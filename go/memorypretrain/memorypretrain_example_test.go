// SPDX-Licence-Identifier: EUPL-1.2

package memorypretrain_test

import (
	"context"
	"fmt"
	"strings"

	core "dappco.re/go"
	"dappco.re/go/mlx/memorypretrain"
)

// ExampleBuildBank builds a two-cluster hierarchical memory bank from labelled
// embeddings and retrieves the nearest block to a query.
func ExampleBuildBank() {
	bank, err := memorypretrain.BuildBank([]memorypretrain.Block{
		{ID: "go-1", Text: "Go memory planning", Embedding: []float32{1, 0}},
		{ID: "go-2", Text: "Go cgo bridge", Embedding: []float32{0.9, 0.1}},
		{ID: "poem-1", Text: "winter proof poem", Embedding: []float32{0, 1}},
		{ID: "poem-2", Text: "autumn prayer", Embedding: []float32{0.1, 0.9}},
	}, memorypretrain.BuildConfig{BranchingFactor: 2, MaxDepth: 1, MinClusterSize: 2, KMeansIters: 8})
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	got, err := bank.Retrieve([]float32{1, 0}, 1)
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Println(bank.Dimension, got[0].BlockID)
	// Output: 2 go-1
}

// ExampleBank_ClusterIDs routes a query through the hierarchy and reports the
// per-level cluster IDs the retriever selected.
func ExampleBank_ClusterIDs() {
	bank, err := memorypretrain.BuildBank([]memorypretrain.Block{
		{ID: "go-1", Embedding: []float32{1, 0}},
		{ID: "go-2", Embedding: []float32{0.9, 0.1}},
		{ID: "poem-1", Embedding: []float32{0, 1}},
		{ID: "poem-2", Embedding: []float32{0.1, 0.9}},
	}, memorypretrain.BuildConfig{BranchingFactor: 2, MaxDepth: 1, MinClusterSize: 1, KMeansIters: 8})
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	ids, err := bank.ClusterIDs([]float32{1, 0})
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Println(len(ids))
	// Output: 1
}

// ExampleGenericClusterIDs returns the generic-memory fallback: the last cluster
// index at each hierarchy level.
func ExampleGenericClusterIDs() {
	ids, err := memorypretrain.GenericClusterIDs([]int{16, 256, 1024})
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Println(ids)
	// Output: [15 255 1023]
}

// ExampleSaveFFNMemoryBank persists an FFN memory bank and reloads it through
// the versioned JSON envelope.
func ExampleSaveFFNMemoryBank() {
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
	dirResult := core.MkdirTemp("", "go-mlx-memorypretrain-example-*")
	if !dirResult.OK {
		fmt.Println("error:", dirResult.Value)
		return
	}
	dir := dirResult.Value.(string)
	defer core.RemoveAll(dir)
	path := core.PathJoin(dir, "ffn.json")
	if err := memorypretrain.SaveFFNMemoryBank(path, bank); err != nil {
		fmt.Println("error:", err)
		return
	}
	loaded, err := memorypretrain.LoadFFNMemoryBank(path)
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Println(loaded.HiddenSize, len(loaded.Layers))
	// Output: 2 1
}

// ExampleAddClusterIDsToJSONL enriches a JSONL row with generic-memory cluster
// IDs when no learned router is supplied.
func ExampleAddClusterIDsToJSONL() {
	raw := `{"id":"a","context":"Go memory planning"}` + "\n"
	out, report, err := memorypretrain.AddClusterIDsToJSONL(context.Background(), raw, nil, nil, memorypretrain.ClusterIDJSONLConfig{
		TaskType:      memorypretrain.ClusterIDTaskLanguageModeling,
		ClusterCounts: []int{3},
	})
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Println(report.Rows, report.GenericRows)
	fmt.Println(strings.Contains(out, `"cluster_ids":[2]`))
	// Output:
	// 1 1
	// true
}

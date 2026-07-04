// SPDX-Licence-Identifier: EUPL-1.2

package memorypretrain_test

import (
	"context"
	"fmt"

	"dappco.re/go/mlx/memorypretrain"
)

func exampleRetrievalBank() (*memorypretrain.Bank, error) {
	return memorypretrain.BuildBank([]memorypretrain.Block{
		{ID: "go-1", Text: "Go memory planning", Embedding: []float32{1, 0}},
		{ID: "go-2", Text: "Go cgo bridge", Embedding: []float32{0.9, 0.1}},
		{ID: "poem-1", Text: "winter proof poem", Embedding: []float32{0, 1}},
		{ID: "poem-2", Text: "autumn prayer", Embedding: []float32{0.1, 0.9}},
	}, memorypretrain.BuildConfig{BranchingFactor: 2, MaxDepth: 1, MinClusterSize: 2, KMeansIters: 8})
}

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

// ExampleEmbedFunc_Embed adapts a closure into an Embedder and calls it through
// the Embed method.
func ExampleEmbedFunc_Embed() {
	embed := memorypretrain.EmbedFunc(func(_ context.Context, text string) ([]float32, error) {
		return []float32{float32(len(text)), 0}, nil
	})
	vec, err := embed.Embed(context.Background(), "hello")
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Println(vec)
	// Output: [5 0]
}

// ExampleBuildBankFromCorpus embeds corpus records with the anchor embedder and
// builds a hierarchical memory bank from the resulting embedded blocks.
func ExampleBuildBankFromCorpus() {
	embed := memorypretrain.EmbedFunc(func(_ context.Context, text string) ([]float32, error) {
		if text == "Go memory planning" {
			return []float32{1, 0}, nil
		}
		return []float32{0, 1}, nil
	})
	bank, err := memorypretrain.BuildBankFromCorpus(context.Background(), embed, []memorypretrain.CorpusRecord{
		{ID: "go", Text: "Go memory planning"},
		{ID: "poem", Text: "winter proof poem"},
	}, memorypretrain.BuildConfig{BranchingFactor: 2, MaxDepth: 1, MinClusterSize: 2, KMeansIters: 8})
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Println(bank.Dimension, len(bank.Blocks))
	// Output: 2 2
}

// ExampleBank_Retrieve returns the top-k nearest blocks to a query from the
// routed leaf cluster.
func ExampleBank_Retrieve() {
	bank, err := exampleRetrievalBank()
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	got, err := bank.Retrieve([]float32{1, 0}, 1)
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Println(got[0].BlockID)
	// Output: go-1
}

// ExampleBank_RetrieveInto reuses a caller-supplied scratch slice across
// retrievals instead of allocating a fresh result slice per query.
func ExampleBank_RetrieveInto() {
	bank, err := exampleRetrievalBank()
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	scratch := make([]memorypretrain.Retrieval, 0, 4)
	got, err := bank.RetrieveInto(scratch, []float32{0, 1}, 1)
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Println(got[0].BlockID)
	// Output: poem-1
}

// ExampleBank_ClusterIDsInto appends the hierarchical cluster IDs for a query to
// a reused destination slice, mirroring RetrieveInto's buffer threading.
func ExampleBank_ClusterIDsInto() {
	bank, err := exampleRetrievalBank()
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	ids, err := bank.ClusterIDsInto(make([]int, 0, 4), []float32{1, 0})
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Println(len(ids))
	// Output: 1
}

// ExampleBank_ClusterAssignments routes a query and records one assignment per
// reached hierarchy level.
func ExampleBank_ClusterAssignments() {
	bank, err := exampleRetrievalBank()
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	assignments, err := bank.ClusterAssignments([]float32{1, 0})
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Println(len(assignments), assignments[0].ClusterID)
	// Output: 1 0
}

// ExampleBank_InjectAdditive retrieves memory for a query and adds its weighted
// embedding into a hidden activation, reporting how many blocks were applied.
func ExampleBank_InjectAdditive() {
	bank, err := exampleRetrievalBank()
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	_, _, stats, err := bank.InjectAdditive(nil, []float32{0.25, 0.5}, []float32{1, 0}, nil, memorypretrain.InjectionConfig{TopK: 1, Scale: 0.5})
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Println(stats.Retrieved, stats.Applied)
	// Output: 1 true
}

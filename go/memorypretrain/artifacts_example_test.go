// SPDX-Licence-Identifier: EUPL-1.2

package memorypretrain_test

import (
	"context"
	"fmt"

	"dappco.re/go/mlx/memorypretrain"
)

// ExampleBuildMemoryPretrainingArtifacts runs the native offline pipeline:
// it embeds a small corpus, builds the hierarchical router, and allocates a
// matching FFN memory table. A deterministic embedder keeps the example
// reproducible; real callers pass the anchor model's embedder.
func ExampleBuildMemoryPretrainingArtifacts() {
	embedder := memorypretrain.EmbedFunc(func(_ context.Context, text string) ([]float32, error) {
		// Two axes: anything mentioning "go" leans on the first, everything else
		// on the second, so the corpus forms two clean clusters.
		if text == "Go memory planning" || text == "Go cgo bridge" {
			return []float32{1, 0}, nil
		}
		return []float32{0, 1}, nil
	})
	artifacts, err := memorypretrain.BuildMemoryPretrainingArtifacts(context.Background(), embedder, []memorypretrain.CorpusRecord{
		{ID: "go-1", Text: "Go memory planning"},
		{ID: "go-2", Text: "Go cgo bridge"},
		{ID: "poem-1", Text: "winter proof poem"},
		{ID: "poem-2", Text: "autumn prayer"},
	}, memorypretrain.MemoryPretrainingArtifactConfig{
		Build: memorypretrain.BuildConfig{BranchingFactor: 2, MaxDepth: 1, MinClusterSize: 1, KMeansIters: 4},
		FFNMemory: memorypretrain.FFNMemoryConfig{
			HiddenSize:      2,
			Layers:          1,
			MemoryLevels:    []string{"1"},
			FFNMemoryTokens: []int{1},
		},
	})
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	// The FFN memory table inherits its cluster count from the router hierarchy.
	fmt.Println(artifacts.Report.CorpusRecords, artifacts.FFNMemory.Config.NumClusters[0])
	// Output: 4 2
}

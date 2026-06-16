// SPDX-Licence-Identifier: EUPL-1.2

package memorypretrain_test

import (
	"context"
	"fmt"

	core "dappco.re/go"
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

// ExampleLoadCorpusRecordsJSONL parses corpus records from a JSONL string. Each
// row accepts id, text, and an optional string-valued meta object.
func ExampleLoadCorpusRecordsJSONL() {
	records, err := memorypretrain.LoadCorpusRecordsJSONL(
		`{"id":"go","text":"Go memory planning","meta":{"source":"docs"}}` + "\n" +
			`{"id":"poem","text":"winter proof poem"}` + "\n")
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Println(len(records), records[0].ID, records[0].Meta["source"])
	// Output: 2 go docs
}

// ExampleLoadCorpusRecordsJSONLFile reads corpus records from a JSONL file on
// disk.
func ExampleLoadCorpusRecordsJSONLFile() {
	dirResult := core.MkdirTemp("", "go-mlx-memorypretrain-corpus-*")
	if !dirResult.OK {
		fmt.Println("error:", dirResult.Value)
		return
	}
	dir := dirResult.Value.(string)
	defer core.RemoveAll(dir)
	path := core.PathJoin(dir, "records.jsonl")
	if result := core.WriteFile(path, []byte(`{"id":"go","text":"Go memory planning"}`+"\n"), 0o644); !result.OK {
		fmt.Println("error:", result.Value)
		return
	}
	records, err := memorypretrain.LoadCorpusRecordsJSONLFile(path)
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Println(len(records), records[0].ID)
	// Output: 1 go
}

// ExampleBuildMemoryPretrainingArtifactsFromFiles loads a corpus JSONL file from
// disk and runs the offline artefact builder over it.
func ExampleBuildMemoryPretrainingArtifactsFromFiles() {
	dirResult := core.MkdirTemp("", "go-mlx-memorypretrain-files-*")
	if !dirResult.OK {
		fmt.Println("error:", dirResult.Value)
		return
	}
	dir := dirResult.Value.(string)
	defer core.RemoveAll(dir)
	corpusPath := core.PathJoin(dir, "records.jsonl")
	if result := core.WriteFile(corpusPath, []byte(
		`{"id":"go-1","text":"Go memory planning"}`+"\n"+
			`{"id":"go-2","text":"Go cgo bridge"}`+"\n"+
			`{"id":"poem-1","text":"winter proof poem"}`+"\n"+
			`{"id":"poem-2","text":"autumn prayer"}`+"\n"), 0o644); !result.OK {
		fmt.Println("error:", result.Value)
		return
	}
	embedder := memorypretrain.EmbedFunc(func(_ context.Context, text string) ([]float32, error) {
		if text == "Go memory planning" || text == "Go cgo bridge" {
			return []float32{1, 0}, nil
		}
		return []float32{0, 1}, nil
	})
	artifacts, err := memorypretrain.BuildMemoryPretrainingArtifactsFromFiles(context.Background(), embedder, memorypretrain.MemoryPretrainingArtifactConfig{
		CorpusPath: corpusPath,
		Build:      memorypretrain.BuildConfig{BranchingFactor: 2, MaxDepth: 1, MinClusterSize: 1, KMeansIters: 4},
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
	fmt.Println(artifacts.Report.CorpusRecords)
	// Output: 4
}

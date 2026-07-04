// SPDX-Licence-Identifier: EUPL-1.2

// Runnable usage-in-situ for the dataset batching/streaming surface. Each
// carries an Output: comment so it executes under `go test` and doubles as the
// usage doc (AX principle 2). The fake tokenizer is deterministic and loads no
// model.
//
// Run:    go test -tags metal_runtime -run='Example' ./go

package train

import (
	core "dappco.re/go"
	"dappco.re/go/mlx/dataset"
	"dappco.re/go/mlx/spine"
)

// ExampleBuildDatasetBatches_packing tokenises a two-row dataset with sequence
// packing on. Each row "hi"/"yes" becomes the virtual sequence [1 2 3 EOS], so
// two rows concatenate into one packed window when MaxSeqLen has room — one
// batch instead of two. Packing keeps the GPU fed without padding waste.
func ExampleBuildDatasetBatches_packing() {
	tok := spine.NewTokenizer(exampleSFTTokenizer{})
	ds := dataset.NewSliceDataset([]dataset.Sample{
		{Prompt: "hi", Response: "yes"},
		{Prompt: "hi", Response: "yes"},
	})

	batches, err := BuildDatasetBatches(tok, ds, dataset.BatchConfig{
		BatchSize:       4,
		MaxSeqLen:       16,
		SequencePacking: true,
	})
	if err != nil {
		core.Println("error:", err)
		return
	}
	core.Println("batches:", len(batches))
	core.Println("rows:", len(batches[0].Batch.Tokens))
	core.Println("packed tokens:", batches[0].Batch.Tokens[0])
	// Output:
	// batches: 1
	// rows: 1
	// packed tokens: [1 2 3 1 2 3]
}

// ExampleBuildDatasetBatches_noPack tokenises the same dataset with packing
// off: each row becomes its own row in the batch, one batch holding both since
// BatchSize is 4.
func ExampleBuildDatasetBatches_noPack() {
	tok := spine.NewTokenizer(exampleSFTTokenizer{})
	ds := dataset.NewSliceDataset([]dataset.Sample{
		{Prompt: "hi", Response: "yes"},
		{Prompt: "hi", Response: "yes"},
	})

	batches, err := BuildDatasetBatches(tok, ds, dataset.BatchConfig{
		BatchSize: 4,
		MaxSeqLen: 16,
	})
	if err != nil {
		core.Println("error:", err)
		return
	}
	core.Println("batches:", len(batches))
	core.Println("rows:", len(batches[0].Batch.Tokens))
	core.Println("row tokens:", batches[0].Batch.Tokens[0])
	// Output:
	// batches: 1
	// rows: 2
	// row tokens: [1 2 3]
}

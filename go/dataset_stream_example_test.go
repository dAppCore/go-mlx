// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	core "dappco.re/go"
	"dappco.re/go/mlx/dataset"
)

func ExampleBuildDatasetBatches() {
	tokenizer := &Tokenizer{tok: fakeSFTTokenizer{
		encoded: map[string][]int32{
			"p1": {1},
			"r1": {2},
			"p2": {3},
			"r2": {4},
		},
		eos: 9,
	}}
	samples := dataset.NewSliceDataset([]dataset.Sample{
		{Prompt: "p1", Response: "r1"},
		{Prompt: "p2", Response: "r2"},
	})

	batches, err := BuildDatasetBatches(tokenizer, samples, dataset.BatchConfig{
		BatchSize:       1,
		MaxSeqLen:       8,
		SequencePacking: true,
	})

	core.Println(err == nil, batches[0].Batch.Tokens[0], batches[0].Targets[0], batches[0].Batch.LossMask[0])
	// Output: true [1 2 3 4] [2 9 4 9] [1 1 1 1]
}

// SPDX-Licence-Identifier: EUPL-1.2

package distill

import (
	mlx "dappco.re/go/mlx"
	"dappco.re/go/mlx/dataset"

	core "dappco.re/go"
)

// ExampleBuildDatasetBatches shows the package's re-bound dataset-batch
// builder turning a one-sample dataset into tokenized SFT batches. The
// builder is the same engine entry point the distillation loop uses to
// tokenize a stream when the runner supplies no BuildBatches hook.
func ExampleBuildDatasetBatches() {
	tok := mlx.NewTokenizer(fakeSFTTokenizer{
		encoded: map[string][]int32{"prompt": {1}, "response": {2}},
		eos:     3,
	})
	ds := dataset.NewSliceDataset([]dataset.Sample{{Prompt: "prompt", Response: "response"}})

	batches, err := BuildDatasetBatches(tok, ds, dataset.BatchConfig{BatchSize: 1})
	if err != nil {
		core.Println(err.Error())
		return
	}
	core.Println(len(batches) >= 1)
	// Output: true
}

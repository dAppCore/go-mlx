// SPDX-Licence-Identifier: EUPL-1.2

// Runnable examples for the SFT public surface — these double as usage docs
// (AX principle 2: comments as usage examples) and execute under `go test`
// because each carries an Output: comment. All are deterministic and load no
// model: synthetic tokenizer + t-free pure helpers only.
//
// Run:    go test -tags metal_runtime -run='Example' ./go

package train

import (
	core "dappco.re/go"
	"dappco.re/go/mlx/dataset"
	"dappco.re/go/mlx/spine"
)

// ExampleBuildSFTBatches tokenises a tiny prompt/response dataset into
// response-masked training batches. The fake tokenizer maps the prompt to
// [1 2] and the response to [3]; with the default EOS the virtual sequence is
// [1 2 3 EOS], so the single example yields inputs [1 2 3], targets [2 3 EOS].
// The mask follows the next-token shift: position promptLen-1 (index 1) is the
// step that predicts the FIRST response token, so the loss is taken from index
// 1 onward — the mask is [0 1 1], training only the response (and EOS).
func ExampleBuildSFTBatches() {
	tok := spine.NewTokenizer(exampleSFTTokenizer{})
	ds := dataset.NewSliceDataset([]dataset.Sample{{Prompt: "hi", Response: "yes"}})

	batches, err := BuildSFTBatches(tok, ds, SFTConfig{BatchSize: 1})
	if err != nil {
		core.Println("error:", err)
		return
	}
	core.Println("batches:", len(batches))
	core.Println("inputs:", batches[0].Batch.Tokens[0])
	core.Println("targets:", batches[0].Targets[0])
	core.Println("mask:", batches[0].Batch.LossMask[0])
	// Output:
	// batches: 1
	// inputs: [1 2 3]
	// targets: [2 3 9]
	// mask: [0 1 1]
}

// SPDX-Licence-Identifier: EUPL-1.2

// Runnable usage-in-situ for the SFT epoch surface. Each carries an Output:
// comment so it executes under `go test` and doubles as the usage doc (AX
// principle 2). No model and no Metal: the epoch example drives the build loop
// over rows that produce no batch, and the cascade example uses the in-memory
// scorer.
//
// Run:    go test -tags metal_runtime -run='Example' ./go

package train

import (
	"context"

	core "dappco.re/go"
	"dappco.re/go/mlx/dataset"
)

// ExampleRunSFTDatasetEpoch walks one epoch over a dataset whose rows produce no
// training target (empty prompt/response under NoEOS). The loop visits every
// row, skips them all, and returns cleanly — no gradient step is reached because
// no batch ever forms, so no model or Metal is needed. Samples stays 0.
func ExampleRunSFTDatasetEpoch() {
	tok := newSFTBatchTestTokenizer()
	cfg := SFTConfig{BatchSize: 2, GradientAccumulationSteps: 2, NoEOS: true}
	ds := dataset.NewSliceDataset([]dataset.Sample{
		{Prompt: "", Response: ""},
		{Prompt: "", Response: ""},
	})
	result := &SFTResult{}

	err := RunSFTDatasetEpoch(context.Background(), nil, tok, ds, nil, nil, cfg, result, 1)
	core.Println("error:", err)
	core.Println("samples:", result.Samples)
	// Output:
	// error: <nil>
	// samples: 0
}

// ExampleFinaliseScoreCascade copies a cascade's verdict onto the result after
// the epoch loop. A single scored pass at step 7 makes that the best step and
// lands its record on the result. The scorer runs in memory — no model.
func ExampleFinaliseScoreCascade() {
	result := &SFTResult{cascade: newSFTScoreCascade("", 0)}
	result.cascade.recordPass(7, []SFTEvalResult{
		{Step: 7, Prompt: "p", Text: "I notice the morning holds, and I want to keep it."},
	})

	FinaliseScoreCascade(result)

	core.Println("best step:", result.BestScoreStep)
	core.Println("records:", len(result.ScoreRecords))
	// Output:
	// best step: 7
	// records: 1
}

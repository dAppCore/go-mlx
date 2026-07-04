// SPDX-Licence-Identifier: EUPL-1.2

// Runnable usage-in-situ for the validation lane. Each carries an Output:
// comment so it executes under `go test` and doubles as the usage doc
// (AX principle 2). The loss function is injected, so the schedule + recording
// machinery runs with no model and no Metal.
//
// Run:    go test -tags metal_runtime -run='Example' ./go

package train

import (
	core "dappco.re/go"
	"dappco.re/go/mlx/dataset"
)

// ExampleArmSFTValidation arms the validation lane with an injected no-grad
// loss function, then drives the optimizer clock by hand. The in-loop gate
// fires only on multiples of the cadence (2): a baseline pass at step 0 plus
// passes at steps 2 and 4 record three points, and the last value is the most
// recent loss. No model is loaded — the lossFn stands in for the adapter's
// forward.
func ExampleArmSFTValidation() {
	losses := []float64{2.0, 1.5, 1.0}
	pass := 0
	result := &SFTResult{}
	ArmSFTValidation(result, []SFTBatch{{}}, 2, func(SFTBatch) (float64, bool) {
		v := losses[pass]
		pass++
		return v, true
	})

	cfg := SFTConfig{}
	// Baseline at step 0 — the curve starts before training moves anything.
	_ = RunSFTValidationPass(cfg, result)
	// Steps 1..4 through the gate: passes land only at 2 and 4.
	for step := 1; step <= 4; step++ {
		result.Steps = step
		_ = maybeRunSFTValidation(cfg, result)
	}

	core.Println("points:", len(result.ValLosses))
	core.Println("steps:", result.ValLosses[0].Step, result.ValLosses[1].Step, result.ValLosses[2].Step)
	core.Println("last:", result.LastValLoss)
	// Output:
	// points: 3
	// steps: 0 2 4
	// last: 1
}

// ExampleSFTValEvery shows the validation-cadence resolution: an explicit
// ValEvery wins, otherwise the eval cadence is reused, otherwise 0 means
// baseline-only.
func ExampleSFTValEvery() {
	core.Println(SFTValEvery(SFTConfig{ValEvery: 10, EvalEvery: 25}))
	core.Println(SFTValEvery(SFTConfig{EvalEvery: 25}))
	core.Println(SFTValEvery(SFTConfig{}))
	// Output:
	// 10
	// 25
	// 0
}

// ExampleBuildSFTValidationBatches drains a tiny fixed validation subset and
// tokenises it. Sequence packing is forced off so the subset stays
// sample-stable, so two prompt/response rows yield two discrete validation
// rows. The fake tokenizer loads no model.
func ExampleBuildSFTValidationBatches() {
	tok := newSFTBatchTestTokenizer()
	ds := dataset.NewSliceDataset([]dataset.Sample{
		{Prompt: "prompt", Response: "response"},
		{Prompt: "prompt", Response: "response"},
	})

	batches, err := BuildSFTValidationBatches(tok, SFTConfig{ValidData: ds, ValidSamples: 2})
	if err != nil {
		core.Println("error:", err)
		return
	}
	rows := 0
	for _, b := range batches {
		rows += len(b.Batch.Tokens)
	}
	core.Println("rows:", rows)
	// Output:
	// rows: 2
}

// ExampleRunSFTValidationPass forwards an armed fixed subset under an injected
// no-grad loss function and records the mean at the result's current step. No
// model is loaded — the lossFn stands in for the adapter's forward.
func ExampleRunSFTValidationPass() {
	result := &SFTResult{Steps: 8}
	ArmSFTValidation(result, []SFTBatch{{}, {}}, 1, func(SFTBatch) (float64, bool) { return 0.75, true })

	_ = RunSFTValidationPass(SFTConfig{}, result)

	core.Println("points:", len(result.ValLosses))
	core.Println("step:", result.ValLosses[0].Step)
	core.Println("loss:", result.LastValLoss)
	// Output:
	// points: 1
	// step: 8
	// loss: 0.75
}

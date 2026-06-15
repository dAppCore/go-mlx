// SPDX-Licence-Identifier: EUPL-1.2

// Runnable usage-in-situ for the SSD sampling phase. Each carries an Output:
// comment so it executes under `go test` and doubles as the usage doc
// (AX principle 2). Generation is injected, so the whole self-distillation
// sampling loop runs with no model and no Metal.
//
// SSD is decoupled from SFT: RunSSD STOPS at the scored trace — it samples the
// frozen model, scores each sample at birth, and returns the rows. A separate
// SFT run trains on a curated subset later; RunSSD never trains.
//
// Run:    go test -tags metal_runtime -run='Example' ./go

package train

import (
	"context"

	core "dappco.re/go"
	"dappco.re/go/mlx/dataset"
	"dappco.re/go/mlx/spine"
)

// ExampleRunSSD samples one response per prompt from the injected generator and
// stops at the scored trace. With ScoreSamples on, each sample is scored at
// birth: the score rides the sample meta (ssd_lek) so the trace is explainable
// downstream, and a sidecar path is set on the result. The returned Samples
// ARE the deliverable — no training step runs here.
func ExampleRunSSD() {
	dir := core.MkdirTemp("", "ssd-example-*")
	defer core.RemoveAll(dir.Value.(string))

	replies := map[string]string{
		"prove a lemma":     "I work it through step by step and check each claim.",
		"hold a hard truth": "I feel the weight of it and I choose to look straight at it.",
	}
	result, err := RunSSD(context.Background(), SSDRunner{
		Generate: func(_ context.Context, prompt string, _ spine.GenerateConfig) (string, error) {
			return replies[prompt], nil
		},
	}, dataset.NewSliceDataset([]dataset.Sample{
		{Prompt: "prove a lemma"},
		{Prompt: "hold a hard truth"},
	}), SSDConfig{
		SampleTemperature:     1.5,
		SampleMaxTokens:       64,
		FilterShortestPercent: 0,
		ScoreSamples:          true,
		SFT:                   SFTConfig{CheckpointDir: dir.Value.(string)},
	})
	if err != nil {
		core.Println("error:", err)
		return
	}
	core.Println("samples:", len(result.Samples))
	core.Println("first prompt:", result.Samples[0].Prompt)
	core.Println("ssd marker:", result.Samples[0].Meta["ssd"])
	core.Println("score on meta:", result.Samples[0].Meta["ssd_lek"] != "")
	core.Println("sidecar set:", result.SampleScoreSidecar != "")
	// Output:
	// samples: 2
	// first prompt: prove a lemma
	// ssd marker: simple_self_distillation
	// score on meta: true
	// sidecar set: true
}

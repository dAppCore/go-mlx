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

// ExampleSFTEffectiveBatchSize shows the optimizer batch size after gradient
// accumulation: BatchSize × GradientAccumulationSteps.
func ExampleSFTEffectiveBatchSize() {
	cfg := SFTConfig{BatchSize: 4, GradientAccumulationSteps: 8}
	core.Println(SFTEffectiveBatchSize(cfg))
	// Output: 32
}

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

// ExampleSFTResult_Metrics shows the dashboard summary an SFT run produces,
// including the effective-batch arithmetic and the derived counts.
func ExampleSFTResult_Metrics() {
	result := &SFTResult{Steps: 20, Epochs: 1, Samples: 40, LastLoss: 0.5}
	m := result.Metrics(SFTConfig{BatchSize: 2, GradientAccumulationSteps: 4})
	core.Println("steps:", m.Steps)
	core.Println("effective batch:", m.EffectiveBatchSize)
	core.Println("optimizer steps:", m.OptimizerSteps)
	// Output:
	// steps: 20
	// effective batch: 8
	// optimizer steps: 20
}

// ExampleSaveSFTCheckpointMetadata writes checkpoint metadata beside an
// adapter package and reads it back — the portable JSON sidecar that lets a
// run resume. The metadata path is derived from the adapter path.
func ExampleSaveSFTCheckpointMetadata() {
	dirResult := core.MkdirTemp("", "sft-example-*")
	if !dirResult.OK {
		core.Println("error:", dirResult.Value)
		return
	}
	dir := dirResult.Value.(string)
	defer core.RemoveAll(dir)
	adapterPath := core.PathJoin(dir, "adapter.safetensors")

	meta := NewSFTCheckpointMetadata(adapterPath, "gemma4", SFTConfig{BatchSize: 2}, &SFTResult{Steps: 5}, 1)
	if err := SaveSFTCheckpointMetadata(adapterPath, meta); err != nil {
		core.Println("save error:", err)
		return
	}

	loaded, err := LoadSFTCheckpointMetadata(adapterPath)
	if err != nil {
		core.Println("load error:", err)
		return
	}
	core.Println("model:", loaded.Model)
	core.Println("step:", loaded.Step)
	core.Println("epoch:", loaded.Epoch)
	// Output:
	// model: gemma4
	// step: 5
	// epoch: 1
}

// exampleSFTTokenizer is the minimal fixed-vocab fake used by the Examples:
// "hi"→[1 2] (a 2-token prompt so the response masking is visible), "yes"→[3],
// EOS=9. Kept private to the example file so the Output comments stay
// deterministic and self-contained.
type exampleSFTTokenizer struct{}

func (exampleSFTTokenizer) Encode(s string) []int32 {
	switch s {
	case "hi":
		return []int32{1, 2}
	case "yes":
		return []int32{3}
	}
	return nil
}
func (exampleSFTTokenizer) Decode([]int32) string        { return "" }
func (exampleSFTTokenizer) DecodeOne(int32) string       { return "" }
func (exampleSFTTokenizer) TokenID(string) (int32, bool) { return 0, false }
func (exampleSFTTokenizer) IDToken(int32) string         { return "" }
func (exampleSFTTokenizer) BOS() int32                   { return 0 }
func (exampleSFTTokenizer) EOS() int32                   { return 9 }
func (exampleSFTTokenizer) HasBOSToken() bool            { return false }

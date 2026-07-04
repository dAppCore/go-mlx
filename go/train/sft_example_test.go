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
	"dappco.re/go/mlx/spine"
)

// ExampleSFTEffectiveBatchSize shows the optimizer batch size after gradient
// accumulation: BatchSize × GradientAccumulationSteps.
func ExampleSFTEffectiveBatchSize() {
	cfg := SFTConfig{BatchSize: 4, GradientAccumulationSteps: 8}
	core.Println(SFTEffectiveBatchSize(cfg))
	// Output: 32
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

// ExampleNormalizeSFTConfigForModel fills the run defaults for a target model.
// A non-gemma4 architecture takes the generic LoRA normalisation, backfilling
// the default adapter identity (rank 8, alpha 16, q_proj/v_proj) and the scalar
// defaults (batch/grad-accum/epochs all 1). No model is loaded — only the
// architecture string drives the branch.
func ExampleNormalizeSFTConfigForModel() {
	cfg := NormalizeSFTConfigForModel(SFTConfig{}, spine.ModelInfo{Architecture: "qwen3"})
	core.Println("batch:", cfg.BatchSize)
	core.Println("epochs:", cfg.Epochs)
	core.Println("rank:", cfg.LoRA.Rank)
	core.Println("keys:", cfg.LoRA.TargetKeys)
	// Output:
	// batch: 1
	// epochs: 1
	// rank: 8
	// keys: [q_proj v_proj]
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

func (exampleSFTTokenizer) Decode([]int32) string { return "" }

func (exampleSFTTokenizer) DecodeOne(int32) string { return "" }

func (exampleSFTTokenizer) TokenID(string) (int32, bool) { return 0, false }

func (exampleSFTTokenizer) IDToken(int32) string { return "" }

func (exampleSFTTokenizer) BOS() int32 { return 0 }

func (exampleSFTTokenizer) EOS() int32 { return 9 }

func (exampleSFTTokenizer) HasBOSToken() bool { return false }

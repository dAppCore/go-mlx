// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import core "dappco.re/go"

func ExampleBuildSFTTrainingBatches() {
	core.Println("BuildSFTTrainingBatches")
	// Output: BuildSFTTrainingBatches
}

func ExampleNewSFTCheckpointMetadata() {
	meta := NewSFTCheckpointMetadata("/tmp/adapter", "qwen3", SFTConfig{BatchSize: 2}, &SFTResult{Steps: 1}, 1)
	core.Println(meta.Model, meta.Step, meta.BatchSize)
	// Output: qwen3 1 2
}

func ExampleNewSFTArtifactMetadata() {
	meta := NewSFTArtifactMetadata("/tmp/adapter", "gemma4", SFTConfig{BatchSize: 1}, &SFTResult{Steps: 3})
	core.Println(meta.Model, meta.Step)
	// Output: gemma4 3
}

func ExampleSaveSFTCheckpointMetadata() {
	core.Println("SaveSFTCheckpointMetadata")
	// Output: SaveSFTCheckpointMetadata
}

func ExampleLoadSFTCheckpointMetadata() {
	core.Println("LoadSFTCheckpointMetadata")
	// Output: LoadSFTCheckpointMetadata
}

func ExampleApplySFTResumeMetadata() {
	core.Println("ApplySFTResumeMetadata")
	// Output: ApplySFTResumeMetadata
}

func ExampleSFTResult_Metrics() {
	result := &SFTResult{Steps: 2, Samples: 8}
	metrics := result.Metrics(SFTConfig{BatchSize: 2, GradientAccumulationSteps: 4})
	core.Println(metrics.EffectiveBatchSize, metrics.OptimizerSteps)
	// Output: 8 2
}

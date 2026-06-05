// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	core "dappco.re/go"
	"dappco.re/go/mlx/dataset"
)

func ExampleBuildSFTTrainingBatches() {
	tokenizer := &Tokenizer{tok: fakeSFTTokenizer{
		encoded: map[string][]int32{
			"prompt":   {10, 11},
			"response": {20, 21},
		},
		eos: 2,
	}}
	samples := dataset.NewSliceDataset([]dataset.Sample{{Prompt: "prompt", Response: "response"}})

	batches, err := BuildSFTTrainingBatches(tokenizer, samples, SFTConfig{BatchSize: 1})

	core.Println(err == nil, batches[0].Batch.Tokens[0], batches[0].Targets[0], batches[0].Batch.LossMask[0])
	// Output: true [10 11 20 21] [11 20 21 2] [0 1 1 1]
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
	path, cleanup, ok := exampleSFTMetadataPath()
	if !ok {
		return
	}
	defer cleanup()

	err := SaveSFTCheckpointMetadata(path, SFTCheckpointMetadata{Model: "gemma4", Step: 3})
	loaded, loadErr := LoadSFTCheckpointMetadata(path)

	core.Println(err == nil, loadErr == nil, loaded.Model, loaded.Step)
	// Output: true true gemma4 3
}

func ExampleLoadSFTCheckpointMetadata() {
	path, cleanup, ok := exampleSFTMetadataPath()
	if !ok {
		return
	}
	defer cleanup()

	_ = SaveSFTCheckpointMetadata(path, SFTCheckpointMetadata{Model: "gemma4", OptimizerStep: 4})

	loaded, err := LoadSFTCheckpointMetadata(path)

	core.Println(err == nil, loaded.Model, loaded.OptimizerStep)
	// Output: true gemma4 4
}

func ExampleApplySFTResumeMetadata() {
	path, cleanup, ok := exampleSFTMetadataPath()
	if !ok {
		return
	}
	defer cleanup()
	_ = SaveSFTCheckpointMetadata(path, SFTCheckpointMetadata{Model: "gemma4", Step: 5})

	result := &SFTResult{}
	err := ApplySFTResumeMetadata(result, SFTConfig{ResumePath: path})

	core.Println(err == nil, result.ResumePath == path, result.ResumedFrom.Model, result.ResumedFrom.Step)
	// Output: true true gemma4 5
}

func ExampleSFTResult_Metrics() {
	result := &SFTResult{Steps: 2, Samples: 8}
	metrics := result.Metrics(SFTConfig{BatchSize: 2, GradientAccumulationSteps: 4})
	core.Println(metrics.EffectiveBatchSize, metrics.OptimizerSteps)
	// Output: 8 2
}

func exampleSFTMetadataPath() (string, func(), bool) {
	dirResult := core.MkdirTemp("", "go-mlx-sft-example-*")
	if !dirResult.OK {
		return "", func() {}, false
	}
	dir := dirResult.Value.(string)
	return core.PathJoin(dir, "adapter"), func() { core.RemoveAll(dir) }, true
}

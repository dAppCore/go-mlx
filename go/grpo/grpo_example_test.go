// SPDX-Licence-Identifier: EUPL-1.2

package grpo

import (
	"context"

	"dappco.re/go/mlx/dataset"

	core "dappco.re/go"
)

// ExampleGRPOSampleFromSFT shows how an SFT/JSONL row is turned into a
// reasoning sample: the prompt is trimmed, the answer is recovered from
// the final line of the response, and the reasoning is the body with the
// answer suffix trimmed off.
func ExampleGRPOSampleFromSFT() {
	sample := GRPOSampleFromSFT(dataset.Sample{
		Prompt:   "  Solve 2+2  ",
		Response: "Add two and two.\n4",
	})
	core.Println(sample.Prompt)
	core.Println(sample.ExpectedAnswer)
	core.Println(sample.Reasoning)
	// Output:
	// Solve 2+2
	// 4
	// Add two and two.
}

// ExampleExtractGRPOExpectedAnswer shows answer recovery from a reasoning
// trace whose final line carries a (case-insensitive) answer prefix.
func ExampleExtractGRPOExpectedAnswer() {
	answer := ExtractGRPOExpectedAnswer(dataset.Sample{
		Response: "First reason about it.\nFinal Answer: 42",
	})
	core.Println(answer)
	// Output: 42
}

// ExampleGRPORewardContainsAnswer scores a rollout that mentions the
// expected answer anywhere in its fragments. The reward carries the
// configured weight when matched.
func ExampleGRPORewardContainsAnswer() {
	reward, err := GRPORewardContainsAnswer(2)(GRPORewardContext{
		Sample:  GRPOSample{ExpectedAnswer: "42"},
		Rollout: GRPORollout{Text: "the answer is 42"},
	})
	core.Println(err == nil)
	core.Println(reward.Name, reward.Detail, reward.Score)
	// Output:
	// true
	// contains_answer matched 2
}

// ExampleGRPORewardExactAnswer scores only when the rollout's answer
// matches the expected answer exactly (after trim + case-fold). A
// non-matching answer scores zero.
func ExampleGRPORewardExactAnswer() {
	fn := GRPORewardExactAnswer(1)
	hit, _ := fn(GRPORewardContext{
		Sample:  GRPOSample{ExpectedAnswer: "Paris"},
		Rollout: GRPORollout{Answer: "paris"},
	})
	miss, _ := fn(GRPORewardContext{
		Sample:  GRPOSample{ExpectedAnswer: "Paris"},
		Rollout: GRPORollout{Answer: "London"},
	})
	core.Println(hit.Detail, hit.Score)
	core.Println(miss.Detail, miss.Score)
	// Output:
	// matched 1
	// missing 0
}

// ExampleRunGRPOReasoningTraining runs a minimal experimental GRPO loop
// over a one-row dataset with a stub rollout. The reported metrics count
// the single grouped step.
func ExampleRunGRPOReasoningTraining() {
	result, err := RunGRPOReasoningTraining(context.Background(), GRPORunner{
		Rollout: func(_ context.Context, req GRPORolloutRequest) ([]GRPORollout, error) {
			return []GRPORollout{
				{Answer: req.Sample.ExpectedAnswer, LogProb: -0.5},
				{Answer: "wrong", LogProb: -1.0},
			}, nil
		},
	}, dataset.NewSliceDataset([]dataset.Sample{
		{Prompt: "What is 2+2?", Response: "4"},
	}), GRPOConfig{
		GroupSize:   2,
		RewardFuncs: []GRPORewardFunc{GRPORewardExactAnswer(1)},
	})
	core.Println(err == nil)
	core.Println(result.Experimental)
	core.Println(result.Metrics.Steps, result.Metrics.Rollouts)
	// Output:
	// true
	// true
	// 1 2
}

// ExampleNewGRPOCheckpointMetadata builds the portable checkpoint sidecar
// from a config + update. Group size defaults are normalised and the
// metadata is stamped experimental at the current version.
func ExampleNewGRPOCheckpointMetadata() {
	meta := NewGRPOCheckpointMetadata(
		"runs/step-000010",
		GRPOConfig{KLCoefficient: 0.1},
		nil,
		GRPOUpdate{Step: 10, Epoch: 1},
	)
	core.Println(meta.Version, meta.Experimental)
	core.Println(meta.Step, meta.GroupSize)
	// Output:
	// 1 true
	// 10 4
}

// ExampleSaveGRPOCheckpointMetadata round-trips checkpoint metadata to a
// temporary directory: Save writes the sidecar JSON and Load reads it
// back, backfilling the version stamp.
func ExampleSaveGRPOCheckpointMetadata() {
	dir := core.PathJoin(core.TempDir(), "grpo_example_ckpt_"+core.Itoa(core.Getpid()))
	defer core.RemoveAll(dir)

	err := SaveGRPOCheckpointMetadata(dir, GRPOCheckpointMetadata{Step: 3, GroupSize: 4})
	core.Println(err == nil)

	loaded, err := LoadGRPOCheckpointMetadata(dir)
	core.Println(err == nil)
	core.Println(loaded.Step, loaded.Version, loaded.Experimental)
	// Output:
	// true
	// true
	// 3 1 true
}

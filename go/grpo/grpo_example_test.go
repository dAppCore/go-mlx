// SPDX-Licence-Identifier: EUPL-1.2

package grpo

import (
	"context"

	"dappco.re/go/mlx/dataset"

	core "dappco.re/go"
)

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

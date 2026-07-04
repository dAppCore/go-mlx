// SPDX-Licence-Identifier: EUPL-1.2

package grpo

import (
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

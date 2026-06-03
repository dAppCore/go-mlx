// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for grpo.go — experimental GRPO reasoning loop.
// Per AX-11 — cloneGRPORollouts fires once per training step (one per
// buildGRPOUpdate call); ExtractGRPOExpectedAnswer + cleanGRPOAnswerLine
// fire per dataset row through GRPOSampleFromSFT. Pinning the alloc
// shape of these hot paths is the load-bearing AX commitment of this
// file.
//
// Run:    go test -bench='BenchmarkGRPO' -benchmem -run='^$' ./go

package mlx

import (
	"testing"

	"dappco.re/go/mlx/dataset"
)

var (
	grpoBenchSinkRollouts []GRPORollout
	grpoBenchSinkString   string
	grpoBenchSinkSample   GRPOSample
	grpoBenchSinkReward   GRPOReward
)

// BenchmarkGRPO_CloneRollouts — per-step rollout snapshot taken at the
// end of buildGRPOUpdate. Sized to a default-ish group: 4 rollouts,
// each with 128 tokens + 1 reward part. Tracks the alloc-count and
// byte-count cost as the per-rollout inner makes are the dominant
// per-step allocator on the GRPO update path.
func BenchmarkGRPO_CloneRollouts(b *testing.B) {
	const (
		group  = 4
		tokens = 128
	)
	rollouts := make([]GRPORollout, group)
	for i := range rollouts {
		ids := make([]int32, tokens)
		for k := range ids {
			ids[k] = int32(k)
		}
		rollouts[i] = GRPORollout{
			TokenIDs: ids,
			RewardParts: []GRPOReward{
				{Name: "contains_answer", Score: 1, Weight: 1, Detail: "matched"},
			},
			Text:      "rollout completion text",
			Answer:    "42",
			Reward:    1.0,
			Advantage: 0.5,
			LogProb:   -0.25,
			KL:        0.0,
		}
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		grpoBenchSinkRollouts = cloneGRPORollouts(rollouts)
	}
}

// BenchmarkGRPO_CloneRolloutsLarge — larger group + larger token count
// (8 rollouts, 512 tokens each, 2 rewards). Tracks behaviour when the
// inner-slice sizes are large enough that the per-rollout SliceClone
// allocations dominate. The flat-backing form should drop alloc count
// from O(group) to O(1) per field.
func BenchmarkGRPO_CloneRolloutsLarge(b *testing.B) {
	const (
		group  = 8
		tokens = 512
	)
	rollouts := make([]GRPORollout, group)
	for i := range rollouts {
		ids := make([]int32, tokens)
		for k := range ids {
			ids[k] = int32(k)
		}
		rollouts[i] = GRPORollout{
			TokenIDs: ids,
			RewardParts: []GRPOReward{
				{Name: "contains_answer", Score: 1, Weight: 1, Detail: "matched"},
				{Name: "exact_answer", Score: 0, Weight: 0.5, Detail: "missing"},
			},
			Text:    "longer rollout completion text spanning multiple sentences",
			Answer:  "42",
			Reward:  1.0,
			LogProb: -1.5,
		}
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		grpoBenchSinkRollouts = cloneGRPORollouts(rollouts)
	}
}

// BenchmarkGRPO_CleanAnswerLine_NoMatch — typical free-form answer line
// that doesn't start with one of the {answer,final answer,solution}
// prefixes. The first-byte switch short-circuits before any allocation.
func BenchmarkGRPO_CleanAnswerLine_NoMatch(b *testing.B) {
	line := "the result is 42"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		grpoBenchSinkString = cleanGRPOAnswerLine(line)
	}
}

// BenchmarkGRPO_CleanAnswerLine_NoMatchAlpha — line starts with 'a' (one
// of the trigger bytes) but has no matching prefix — exercises the
// case-fold compare path that does NOT match. This is the genuine hot
// case where the original form paid for a core.Lower allocation just
// to fail the prefix scan.
func BenchmarkGRPO_CleanAnswerLine_NoMatchAlpha(b *testing.B) {
	line := "addition produces forty two"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		grpoBenchSinkString = cleanGRPOAnswerLine(line)
	}
}

// BenchmarkGRPO_CleanAnswerLine_NoMatchAlphaMixedCase — line starts with
// 'A' (trigger byte) AND has a capital letter, forcing core.Lower to
// allocate a fresh string just to fail the prefix scan. This is the
// path the case-fold compare optimisation targets.
func BenchmarkGRPO_CleanAnswerLine_NoMatchAlphaMixedCase(b *testing.B) {
	line := "Addition Produces Forty Two"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		grpoBenchSinkString = cleanGRPOAnswerLine(line)
	}
}

// BenchmarkGRPO_CleanAnswerLine_Match — "Answer: 42" — a line that
// matches "answer:" via case-insensitive prefix. Exercises the
// matched-prefix path with its trailing Trim allocation.
func BenchmarkGRPO_CleanAnswerLine_Match(b *testing.B) {
	line := "Answer: 42"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		grpoBenchSinkString = cleanGRPOAnswerLine(line)
	}
}

// BenchmarkGRPO_SampleFromSFT — the per-dataset-row entry point. Builds
// the prompt, expected answer, reasoning, and meta clone for one SFT
// sample. Runs once per training row before any rollout fires.
func BenchmarkGRPO_SampleFromSFT(b *testing.B) {
	sample := dataset.Sample{
		Prompt:   "Solve: 17 + 25",
		Response: "Add: seventeen plus twenty five.\nAnswer: 42",
		Meta:     map[string]string{"id": "row-1", "split": "train"},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		grpoBenchSinkSample = GRPOSampleFromSFT(sample)
	}
}

// BenchmarkGRPO_SampleFromSFT_MultiLine — more lines exercise the new
// backward walk path that replaces core.Split with iterative
// LastIndex. Five reasoning lines plus the answer at the tail.
func BenchmarkGRPO_SampleFromSFT_MultiLine(b *testing.B) {
	sample := dataset.Sample{
		Prompt: "Solve: 17 + 25",
		Response: "Let me think.\n" +
			"First add the tens.\n" +
			"Ten plus twenty is thirty.\n" +
			"Then the ones.\n" +
			"Seven plus five is twelve.\n" +
			"Answer: 42",
		Meta: map[string]string{"id": "row-1", "split": "train"},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		grpoBenchSinkSample = GRPOSampleFromSFT(sample)
	}
}

// BenchmarkGRPO_RewardContainsAnswer — exercises the default reward
// closure that scores rollouts for the contains-answer rubric. Runs
// once per rollout (group_size × steps over a training run).
func BenchmarkGRPO_RewardContainsAnswer(b *testing.B) {
	fn := GRPORewardContainsAnswer(1)
	ctx := GRPORewardContext{
		Sample: GRPOSample{ExpectedAnswer: "42"},
		Rollout: GRPORollout{
			Answer:    "42",
			Text:      "The arithmetic produces forty two so the answer is 42",
			Reasoning: "Adding seventeen and twenty five gives forty two",
		},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		grpoBenchSinkReward, _ = fn(ctx)
	}
}

// BenchmarkGRPO_RewardContainsAnswer_MatchInText — match lives in the
// long Text fragment instead of the short Answer field. Exercises the
// linear scan over a representative rollout completion.
func BenchmarkGRPO_RewardContainsAnswer_MatchInText(b *testing.B) {
	fn := GRPORewardContainsAnswer(1)
	ctx := GRPORewardContext{
		Sample: GRPOSample{ExpectedAnswer: "forty two"},
		Rollout: GRPORollout{
			Answer:    "the result follows",
			Text:      "The arithmetic produces forty two so the answer is right",
			Reasoning: "Adding seventeen and twenty five gives the same number",
		},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		grpoBenchSinkReward, _ = fn(ctx)
	}
}

// BenchmarkGRPO_RewardContainsAnswer_NoMatch — expected answer absent
// from all three fragments. Worst-case linear scan over all three
// fragments without a hit.
func BenchmarkGRPO_RewardContainsAnswer_NoMatch(b *testing.B) {
	fn := GRPORewardContainsAnswer(1)
	ctx := GRPORewardContext{
		Sample: GRPOSample{ExpectedAnswer: "1729"},
		Rollout: GRPORollout{
			Answer:    "42",
			Text:      "The arithmetic produces forty two so the answer is 42",
			Reasoning: "Adding seventeen and twenty five gives forty two",
		},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		grpoBenchSinkReward, _ = fn(ctx)
	}
}

// BenchmarkGRPO_RewardContainsAnswer_Unicode — expected answer contains
// a non-ASCII character (an em-dash "—"). Forces the fallback to
// core.Join + core.Lower so we keep visibility on the slower path.
func BenchmarkGRPO_RewardContainsAnswer_Unicode(b *testing.B) {
	fn := GRPORewardContainsAnswer(1)
	ctx := GRPORewardContext{
		Sample: GRPOSample{ExpectedAnswer: "vingt — quatre"},
		Rollout: GRPORollout{
			Answer:    "vingt — quatre",
			Text:      "La réponse est vingt — quatre",
			Reasoning: "L'addition produit vingt — quatre",
		},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		grpoBenchSinkReward, _ = fn(ctx)
	}
}

// BenchmarkGRPO_RewardExactAnswer — sister bench, exercises the
// exact-match scorer.
func BenchmarkGRPO_RewardExactAnswer(b *testing.B) {
	fn := GRPORewardExactAnswer(1)
	ctx := GRPORewardContext{
		Sample:  GRPOSample{ExpectedAnswer: "42"},
		Rollout: GRPORollout{Answer: "42"},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		grpoBenchSinkReward, _ = fn(ctx)
	}
}

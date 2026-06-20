// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for grpo.go — experimental GRPO reasoning loop.
// Per AX-11 — cloneGRPORollouts fires once per training step (one per
// buildGRPOUpdate call); ExtractGRPOExpectedAnswer + cleanGRPOAnswerLine
// fire per dataset row through GRPOSampleFromSFT. Pinning the alloc
// shape of these hot paths is the load-bearing AX commitment of this
// file.
//
// Run:    go test -bench='BenchmarkGRPO' -benchmem -run='^$' ./go

package grpo

import (
	"context"
	"testing"
)

var (
	grpoBenchSinkRollouts []GRPORollout
	grpoBenchSinkString   string
	grpoBenchSinkSample   GRPOSample
	grpoBenchSinkReward   GRPOReward
	grpoBenchSinkUpdate   GRPOUpdate
)

// BenchmarkGRPO_BuildUpdate — the per-step update assembly. Scores a
// group of rollouts against the default reward func, computes
// advantages + loss, and snapshots the rollouts into the update. Runs
// once per training step (the dominant per-step allocator on the GRPO
// path). KL + ApplyUpdate are off (nil runner fields) so the bench
// isolates the reward/score/clone allocation shape, not the model-side
// callbacks. Default-ish group of 4, 128 tokens each.
func BenchmarkGRPO_BuildUpdate(b *testing.B) {
	const (
		group  = 4
		tokens = 128
	)
	ctx := context.Background()
	request := GRPORolloutRequest{
		Step:      1,
		Epoch:     1,
		GroupSize: group,
		Sample:    GRPOSample{Prompt: "Solve: 17 + 25", ExpectedAnswer: "42"},
	}
	cfg := normalizeGRPOConfig(GRPOConfig{GroupSize: group})
	rollouts := make([]GRPORollout, group)
	for i := range rollouts {
		ids := make([]int32, tokens)
		for k := range ids {
			ids[k] = int32(k)
		}
		rollouts[i] = GRPORollout{
			TokenIDs:  ids,
			Text:      "the arithmetic produces forty two so the answer is 42",
			Answer:    "42",
			Reasoning: "adding seventeen and twenty five gives forty two",
			LogProb:   -0.5,
		}
	}
	runner := GRPORunner{}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		grpoBenchSinkUpdate, _ = buildGRPOUpdate(ctx, runner, request, rollouts, cfg)
	}
}

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

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
	"testing"
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

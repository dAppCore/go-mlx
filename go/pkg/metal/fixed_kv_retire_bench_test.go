// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import "testing"

// FixedKVCache retirement-path bench (AX-11).
//
// The pipelined decode loop (session_pipelined.go) drives every layer's
// FixedKVCache through ArmPending → staged adoption → CommitPending on EVERY
// decoded token. Once the conversation passes the sliding-window cap, the
// post-cap lane stages a swap (ReplaceFixedFromNativeBorrowed while armed) and
// CommitPending adopts it — retiring the superseded c.keys/c.values handles via
// retireAfterNextEval, then rotating the retirement generations so the pile
// stays one commit deep (struct doc on `retired`/`retiredPrev`).
//
// The figure of merit is allocs PER COMMIT CYCLE: that is the per-token cost of
// the retirement bookkeeping, multiplied by the sliding-layer count and by every
// token of every generation. This bench isolates that bookkeeping by driving the
// real cache methods in the real steady-state sequence — no model load (AX-11
// synthetic). Each cycle adopts a fresh K/V pair (as a real decode step would),
// which the cache owns and frees through the retirement pipeline; the alloc
// figure therefore includes that small, fixed array churn plus the retire/rotate
// bookkeeping the change targets. The -memprofile -list of CommitPending /
// retireAfterNextEval attributes the bookkeeping share precisely.

// fixedRetireKVPair returns a tiny [1,1,2,2] K/V pair for the retire bench. The
// payload is irrelevant — only the handle bookkeeping is measured.
func fixedRetireKVPair() (*Array, *Array) {
	k := FromValues([]float32{1, 2, 3, 4}, 1, 1, 2, 2)
	v := FromValues([]float32{5, 6, 7, 8}, 1, 1, 2, 2)
	return k, v
}

// drivePastCap advances the cache offset beyond maxSize using committed
// (unarmed) adoptions of fresh pairs, so subsequent armed cycles run the
// post-cap staged-swap retire lane with non-nil storage to retire.
func drivePastCap(c *FixedKVCache) {
	for c.Offset() <= c.MaxSize() {
		k, v := fixedRetireKVPair()
		c.ReplaceFixedFromNativeBorrowed(k, v, 1)
	}
}

// BenchmarkFixedKVCache_RetireCommitCycle_PastCap measures the per-token cost of
// the pipelined retirement bookkeeping in steady state past the sliding cap.
// Each iteration is one decode token's worth of cache work: arm, stage a fresh
// K/V (the model output), commit (swap + retire + rotate). allocs/op is
// allocs/commit-cycle — the quantity that multiplies by sliding-layer count and
// token count on serve.
func BenchmarkFixedKVCache_RetireCommitCycle_PastCap(b *testing.B) {
	c := NewFixedKVCache(8)
	drivePastCap(c)
	defer c.Reset()

	b.ReportAllocs()
	for b.Loop() {
		k, v := fixedRetireKVPair()
		c.ArmPending()
		c.ReplaceFixedFromNativeBorrowed(k, v, 1)
		c.CommitPending()
	}
}

// BenchmarkFixedKVCache_AdoptPair_NoCache is the subtraction baseline: the same
// fresh-pair churn each iteration with NO cache bookkeeping. Subtracting its
// allocs/op from RetireCommitCycle_PastCap isolates the retire/rotate cost the
// change targets, independent of the array-handle allocation both share.
func BenchmarkFixedKVCache_AdoptPair_NoCache(b *testing.B) {
	b.ReportAllocs()
	for b.Loop() {
		k, v := fixedRetireKVPair()
		Free(k, v)
	}
}

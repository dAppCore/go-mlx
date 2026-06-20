// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package moba

import (
	"math"
	"testing"

	metal "dappco.re/go/mlx/pkg/metal"
)

// kernels_cover_test.go pins the kernel branches the happy-path kernel tests in
// moba_test.go never reach because they all use an L exactly divisible by
// blockSize and a non-negative selectK: the partial-trailing-block trimming in
// blockScores, the negative-selectK clamp in blockSelectMask, and the
// PadAxis tail in expandToTokens. Each is also a real assertion, not just a line
// to colour — the trailing partial block must be EXCLUDED from the block reps
// and the keep-mask, and the pad must stay -inf.
//
// These exercise live Metal ops, so they gate on requireMetalRuntime (defined in
// moba_test.go) exactly like the other kernel tests.

// TestBlockScores_PartialBlockExcluded_Good pins the covered != L branch of
// blockScores (L=3, blockSize=2 → 1 whole block, token 2 left over). The leftover
// token is a loud sentinel [99,99]; if blockScores let it leak into the block
// representative the score would explode, so this proves the Slice4 trim drops
// the partial block. rep = mean of rows 0,1 = [3,3]; q rows dot it to 3.
func TestBlockScores_PartialBlockExcluded_Good(t *testing.T) {
	requireMetalRuntime(t)

	// Rows 0,1 form the single whole block (mean [3,3]); row 2 is the partial-
	// block sentinel that must be excluded.
	q := metal.FromValues([]float32{1, 0, 0, 1, 1, 1}, 1, 1, 3, 2)
	k := metal.FromValues([]float32{2, 2, 4, 4, 99, 99}, 1, 1, 3, 2)
	defer metal.Free(q, k)

	scores := blockScores(q, k, 2, 1.0)
	defer metal.Free(scores)

	if got := scores.Dim(3); got != 1 {
		t.Fatalf("blocks = %d, want 1 (the partial trailing block must not add a rep)", got)
	}
	// All three queries score 3 against the single rep [3,3]; none sees the 99
	// sentinel. q row2 [1,1]·[3,3] = 6 — proves the rep is [3,3], not skewed by 99.
	checkMask(t, "blockScoresPartial", scores.Floats(), []float32{3, 3, 6})
}

// TestBlockSelectMask_NegativeK_SelfOnly_Good pins the keepN < 1 clamp in
// blockSelectMask: selectK = -1 makes keepN = selectK+1 = 0, which clamps up to
// 1, so the result is identical to the selectK = 0 self-only mask. Same fixture
// and expectation as TestBlockSelectMask_SelfOnly_Good — the assertion is that a
// negative selectK degrades to self-block-only, never to an empty / invalid TopK.
func TestBlockSelectMask_NegativeK_SelfOnly_Good(t *testing.T) {
	requireMetalRuntime(t)

	scores := metal.FromValues([]float32{
		9, 1,
		9, 1,
		9, 1,
		9, 1,
	}, 1, 1, 4, 2)
	defer metal.Free(scores)

	mask := blockSelectMask(scores, 2, -1, 4) // selectK = -1 → keepN clamps to 1
	defer metal.Free(mask)

	n := float32(math.Inf(-1))
	want := []float32{
		0, n, // t=0: self block0; block1 future
		0, n, // t=1: self block0; block1 future
		n, 0, // t=2: self block1; block0 past, not kept (keepN==1, self wins)
		n, 0, // t=3: self block1; block0 past, not kept
	}
	checkMask(t, "negativeKSelfOnly", mask.Floats(), want)
}

// TestExpandToTokens_PadPath_Good pins the covered < L branch of expandToTokens
// (L=3, blockSize=2 → 1 whole block covers tokens 0,1; token 2 is the
// PadAxis-filled tail beyond `covered`). The block keep-mask has one column
// (block0 kept = 0 for every query); after the broadcast covers tokens 0,1 the
// PadAxis pads column 2, and the causal mask is causalMask(L=3, covered=2) —
// which forbids any key j >= covered. So key 2 (the partial-block token) is
// dropped for EVERY query, even query 2's own diagonal: the assertion is that a
// token outside any whole block never enters the attention as a key.
func TestExpandToTokens_PadPath_Good(t *testing.T) {
	requireMetalRuntime(t)

	// nBlocks = 1, block0 kept (0) for all 3 queries.
	blockMask := metal.FromValues([]float32{
		0,
		0,
		0,
	}, 1, 1, 3, 1)
	defer metal.Free(blockMask)

	tokenMask := expandToTokens(blockMask, 2, 3, 1) // L=3, covered = 1*2 = 2 < 3 → pad
	defer metal.Free(tokenMask)

	n := float32(math.Inf(-1))
	// block0 (keys 0,1) kept causally; key 2 is beyond covered=2 → forbidden for
	// all queries (causalMask's j < covered guard), so column 2 is all -inf.
	want := []float32{
		0, n, n, // t=0: only key0
		0, 0, n, // t=1: keys 0,1
		0, 0, n, // t=2: keys 0,1; key2 (partial-block token) excluded as a key
	}
	checkMask(t, "expandPad", tokenMask.Floats(), want)
}

// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package nsa

import (
	"math"
	"testing"

	metal "dappco.re/go/mlx/pkg/metal"
)

// kernels_coverage_test.go pins the trailing-partial-block paths of the two
// block-grid kernels — the branches that only fire when the sequence length is
// NOT an exact multiple of the block size (L % blockSize != 0). compressKV then
// trims the dropped tail before reshaping; expandBlockMaskToTokens then pads the
// key axis back up to L and lets the causal mask forbid the uncovered tail. Both
// need the Metal runtime (they run metal.Array ops), so they reuse
// requireMetalRuntime / approxEqual from nsa_test.go. Names suffixed _Cov to
// stay distinct from the existing whole-block kernel tests.

// TestCompressKV_PartialBlockTrim_Cov drives compressKV with L=5, blockSize=2 →
// nBlocks=2 (the 5th token is a partial block and must be dropped). The two
// block means must therefore come ONLY from tokens 0-1 and 2-3; the trailing
// token 4 is excluded. This exercises the `nBlocks*blockSize != L` trim+Free
// path that the L-divisible-by-blockSize tests never reach.
func TestCompressKV_PartialBlockTrim_Cov(t *testing.T) {
	requireMetalRuntime(t)

	// 5 tokens, dim 2. Rows: [1,1],[3,3],[5,5],[7,7],[100,100].
	// The 100-row is the dropped partial block — if the trim were missing it
	// would either crash the reshape or contaminate a block mean.
	k := metal.FromValues([]float32{1, 1, 3, 3, 5, 5, 7, 7, 100, 100}, 1, 1, 5, 2)
	v := metal.FromValues([]float32{2, 0, 4, 0, 6, 0, 8, 0, 200, 0}, 1, 1, 5, 2)
	defer metal.Free(k, v)

	kCmp, vCmp := compressKV(k, v, 2)
	defer metal.Free(kCmp, vCmp)

	if got := kCmp.Dim(2); got != 2 {
		t.Fatalf("kCmp blocks = %d, want 2 (partial 5th token dropped)", got)
	}
	gotK := kCmp.Floats()
	wantK := []float32{2, 2, 6, 6} // mean(1,3)=2 ; mean(5,7)=6 — no 100
	for i := range wantK {
		if !approxEqual(gotK[i], wantK[i]) {
			t.Errorf("kCmp[%d] = %f, want %f (partial-block tail leaked?)", i, gotK[i], wantK[i])
		}
	}
	gotV := vCmp.Floats()
	wantV := []float32{3, 0, 7, 0} // mean(2,4)=3 ; mean(6,8)=7 — no 200
	for i := range wantV {
		if !approxEqual(gotV[i], wantV[i]) {
			t.Errorf("vCmp[%d] = %f, want %f (partial-block tail leaked?)", i, gotV[i], wantV[i])
		}
	}
}

// TestExpandBlockMaskToTokens_PadTail_Cov drives expandBlockMaskToTokens with a
// block grid that covers fewer than L tokens: nBlocks=2, blockSize=2 → covered=4,
// but L=5. The kernel must pad the key axis from 4 up to 5 and the causal-over-L
// add must drive the uncovered tail column (key 4) to -inf for EVERY query. This
// exercises the `covered < L` PadAxis+Free branch.
func TestExpandBlockMaskToTokens_PadTail_Cov(t *testing.T) {
	requireMetalRuntime(t)

	n := float32(math.Inf(-1))
	// [B=1,H=1,L=5,nBlocks=2]: every query keeps block0, drops block1.
	blockMask := metal.FromValues([]float32{
		0, n,
		0, n,
		0, n,
		0, n,
		0, n,
	}, 1, 1, 5, 2)
	defer metal.Free(blockMask)

	tokenMask := expandBlockMaskToTokens(blockMask, 2, 5)
	defer metal.Free(tokenMask)

	// Shape must be [1,1,L,L] = [1,1,5,5] (key axis padded 4 → 5).
	if got := tokenMask.Shape(); len(got) != 4 || got[2] != 5 || got[3] != 5 {
		t.Fatalf("tokenMask shape = %v, want [1 1 5 5]", got)
	}
	got := tokenMask.Floats()

	// covered=4. Block0 → keys 0,1 (value 0); block1 → keys 2,3 (-inf). Key 4 is
	// the padded uncovered tail → -inf everywhere. Then causal (j <= t).
	want := []float32{
		0, n, n, n, n, // t=0: key0 kept&causal; key1 future; keys2,3 dropped; key4 pad
		0, 0, n, n, n, // t=1: keys0,1 kept&causal; keys2,3 dropped; key4 pad
		0, 0, n, n, n, // t=2: keys0,1 kept; keys2,3 dropped block; key4 pad
		0, 0, n, n, n, // t=3: keys0,1 kept; keys2,3 dropped block; key4 pad
		0, 0, n, n, n, // t=4: keys0,1 kept; keys2,3 dropped; key4 (self) is pad → -inf
	}
	for i := range want {
		if !approxEqual(got[i], want[i]) {
			t.Errorf("tokenMask[%d] = %f, want %f (uncovered tail not forbidden?)", i, got[i], want[i])
		}
	}
}

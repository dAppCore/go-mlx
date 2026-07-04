// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package moba

import (
	"math"
	"testing"

	metal "dappco.re/go/mlx/pkg/metal"
)

// forward_oracle_test.go pins moba.Mixer.Forward end-to-end against an
// INDEPENDENT pure-Go reference, not against Forward's own output (which would
// be circular). MoBA Forward is exactly causal softmax attention with a learned
// block-sparsity mask (see the package doc), so the oracle is dense causal
// softmax restricted to the tokens the kept blocks expose — with the kept set
// known BY CONSTRUCTION (chosen geometry / scores), never read back from the
// mixer's own blockSelectMask. These cover the Forward-level compositions the
// shape-only loader tests cannot see, including the short-sequence edge
// (L < BlockSize → nBlocks == 0) where a past bug lived.

const oracleTol = 1e-4

// identityLinear builds an n×n identity-weight Linear so a projection passes its
// input through unchanged (Forward computes x·Wᵀ = x·I = x). With identity
// Q/K/V the attention sees Q=K=V=x and the Forward math is determined purely by
// the block selection + softmax — exactly what the oracle recomputes.
func identityLinear(n int32) *metal.Linear {
	data := make([]float32, int(n)*int(n))
	for i := int32(0); i < n; i++ {
		data[int(i)*int(n)+int(i)] = 1
	}
	return metal.NewLinear(metal.FromValues(data, int(n), int(n)), nil)
}

// identityMixer builds a single-head MoBA mixer with identity projections and
// the given block geometry, hidden = HeadDim = dim. Q=K=V=O=identity so the
// only non-trivial transform is the block-sparse attention itself.
func identityMixer(dim, blockSize, topK int32, scale float32) *Mixer {
	return &Mixer{
		QProj: identityLinear(dim), KProj: identityLinear(dim),
		VProj: identityLinear(dim), OProj: identityLinear(dim),
		NumHeads: 1, HeadDim: dim, BlockSize: blockSize, TopK: topK, Scale: scale,
	}
}

func freeMixer(m *Mixer) {
	for _, l := range []*metal.Linear{m.QProj, m.KProj, m.VProj, m.OProj} {
		metal.FreeLinear(l)
	}
}

// denseAttentionMasked is the oracle: single-head softmax attention over rows of
// x (Q=K=V=x), where allow[t][j] reports whether query t may attend to key j.
// Scores = scale·(x_t · x_j); forbidden keys are dropped before softmax; the
// output row is the softmax-weighted sum of value rows. Pure Go — no metal, no
// dependence on the unit under test. x is row-major [L, dim].
func denseAttentionMasked(x [][]float32, allow func(t, j int) bool, scale float64) [][]float32 {
	L := len(x)
	out := make([][]float32, L)
	for t := 0; t < L; t++ {
		// Raw scaled scores for the allowed keys.
		logits := make([]float64, 0, L)
		keys := make([]int, 0, L)
		for j := 0; j < L; j++ {
			if !allow(t, j) {
				continue
			}
			dot := 0.0
			for d := range x[t] {
				dot += float64(x[t][d]) * float64(x[j][d])
			}
			logits = append(logits, scale*dot)
			keys = append(keys, j)
		}
		// Numerically-stable softmax over the allowed keys.
		maxLogit := math.Inf(-1)
		for _, l := range logits {
			if l > maxLogit {
				maxLogit = l
			}
		}
		var sum float64
		weights := make([]float64, len(logits))
		for i, l := range logits {
			weights[i] = math.Exp(l - maxLogit)
			sum += weights[i]
		}
		row := make([]float32, len(x[t]))
		for i, j := range keys {
			w := weights[i] / sum
			for d := range row {
				row[d] += float32(w * float64(x[j][d]))
			}
		}
		out[t] = row
	}
	return out
}

// runForward drives Mixer.Forward on a [1,L,dim] input built from rows and
// returns the output as row-major [L][dim]. MoBA's Forward ignores ctx.Cache
// (it attends within the chunk), so a nil cache is fine.
func runForward(t *testing.T, m *Mixer, rows [][]float32) [][]float32 {
	t.Helper()
	L := len(rows)
	dim := len(rows[0])
	flat := make([]float32, 0, L*dim)
	for _, r := range rows {
		flat = append(flat, r...)
	}
	x := metal.FromValues(flat, 1, L, dim)
	defer metal.Free(x)
	out, _ := m.Forward(x, &metal.MixerCtx{B: 1, L: int32(L)})
	defer metal.Free(out)
	got := out.Floats()
	res := make([][]float32, L)
	for r := 0; r < L; r++ {
		res[r] = got[r*dim : (r+1)*dim]
	}
	return res
}

func checkRows(t *testing.T, name string, got, want [][]float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: rows = %d, want %d", name, len(got), len(want))
	}
	for r := range want {
		if len(got[r]) != len(want[r]) {
			t.Fatalf("%s row %d: width = %d, want %d", name, r, len(got[r]), len(want[r]))
		}
		for c := range want[r] {
			if diff := math.Abs(float64(got[r][c] - want[r][c])); diff > oracleTol {
				t.Errorf("%s[%d][%d] = %f, want %f (diff %g)", name, r, c, got[r][c], want[r][c], diff)
			}
		}
	}
}

// TestForward_ShortSeqIsCausal_Good pins the L < BlockSize edge — the exact
// nBlocks == 0 case a past bug lived in. With L=3 and BlockSize=8 there is no
// whole block to score, so MoBA reduces to plain causal softmax over all L
// tokens. The oracle is dense causal softmax (allow j <= t); if Forward instead
// built a degenerate zero-block grid this would diverge loudly.
func TestForward_ShortSeqIsCausal_Good(t *testing.T) {
	requireMetalRuntime(t)

	m := identityMixer(2, 8, 1, 1.0) // BlockSize 8 > L=3 → short path
	defer freeMixer(m)

	rows := [][]float32{{1, 0}, {0, 1}, {1, 1}}
	got := runForward(t, m, rows)

	want := denseAttentionMasked(rows, func(t, j int) bool { return j <= t }, 1.0)
	checkRows(t, "short-seq causal", got, want)
}

// TestForward_SingleBlockBoundary_Good pins the L == BlockSize boundary
// (nBlocks == 1). One block holds every token, so the only causally-valid block
// is the self-block for all queries and MoBA again reduces to plain causal
// softmax — but now through the block-grid code path (nBlocks == 1, not the
// short-circuit). Same dense causal oracle.
func TestForward_SingleBlockBoundary_Good(t *testing.T) {
	requireMetalRuntime(t)

	m := identityMixer(2, 4, 1, 1.0) // BlockSize 4 == L=4 → 1 block
	defer freeMixer(m)

	rows := [][]float32{{1, 0}, {0, 1}, {1, 1}, {2, 0}}
	got := runForward(t, m, rows)

	want := denseAttentionMasked(rows, func(t, j int) bool { return j <= t }, 1.0)
	checkRows(t, "single-block boundary", got, want)
}

// TestForward_SelfBlockOnly_Good pins the block-sparse Forward with TopK=0:
// every query keeps ONLY its own block (the always-on self-block), so the
// allowed keys are exactly the causally-valid tokens within the query's block.
// The kept set is known BY CONSTRUCTION from the geometry (block = t/BlockSize),
// never read from blockSelectMask. L=4, BlockSize=2 → blocks {0,1},{2,3}: token
// 2 attends only to key 2 (block1, causal), token 3 to keys 2,3. This is the
// test the shape-only loader test cannot make — a wrong block→token expansion
// would let a query leak across the block boundary and diverge here.
func TestForward_SelfBlockOnly_Good(t *testing.T) {
	requireMetalRuntime(t)

	const blockSize = 2
	m := identityMixer(2, blockSize, 0, 1.0) // TopK=0 → self-block only
	defer freeMixer(m)

	rows := [][]float32{{1, 0}, {0, 1}, {2, 0}, {0, 3}}
	got := runForward(t, m, rows)

	// Allowed iff same block (j/blockSize == t/blockSize) AND causal (j <= t).
	allow := func(t, j int) bool {
		return j/blockSize == t/blockSize && j <= t
	}
	want := denseAttentionMasked(rows, allow, 1.0)
	checkRows(t, "self-block only", got, want)
}

// TestForward_TopKKeepsScoredPastBlock_Good pins the selection step at the
// Forward level with the kept set fixed by CONSTRUCTION of the scores, not read
// back from the mixer. Geometry L=4, BlockSize=2, TopK=1, identity Q/K so a
// query's block score for block b is q_t · mean(K_block_b). The value rows are
// engineered so block0's mean key aligns strongly with the late queries and
// block1's does not, making block0 the unambiguous top-1 PAST block for queries
// in block1. The self-block is always kept, so queries 2,3 attend over BOTH
// block0 (selected past) and block1 (self) tokens, causal-masked. The oracle
// recomputes the block scores in pure Go to decide selection independently and
// asserts the divergence-free Forward.
func TestForward_TopKKeepsScoredPastBlock_Good(t *testing.T) {
	requireMetalRuntime(t)

	const blockSize = 2
	const topK = 1
	m := identityMixer(2, blockSize, topK, 1.0)
	defer freeMixer(m)

	// Rows chosen so block0 = {[4,0],[4,0]} (mean key [4,0]) dominates the score
	// for queries pointing along +x, while block1 = {[0,1],[0,1]} (mean [0,1])
	// scores ~0 for them. Queries 2,3 point along +x → block0 is the top-1 past
	// block; block1 is their self-block (always kept).
	rows := [][]float32{{4, 0}, {4, 0}, {3, 1}, {5, 2}}
	got := runForward(t, m, rows)

	// Independent pure-Go selection: for each query, score every block by
	// q · mean(key rows of the block), force-keep the self-block, keep the
	// top-(TopK) of the strictly-past blocks, forbid future blocks. nBlocks=2.
	want := denseAttentionMasked(rows, mobaSelectionOracle(rows, blockSize, topK), 1.0)
	checkRows(t, "topK past block", got, want)
}

// TestForward_TopKDropsLowerPastBlock_Good makes the top-K RANKING load-bearing:
// with three blocks (L=6, BlockSize=2) a query in block2 has TWO strictly-past
// blocks (0 and 1), and TopK=1 must keep only the HIGHER-scoring one and drop the
// other. (TestForward_TopKKeepsScoredPastBlock_Good has a single past block, so
// keep-top-1 and keep-all-past coincide there and the ranking is never tested.)
// The rows are engineered so block0's mean key aligns with the block2 queries and
// block1's does not — block0 wins the single past slot, block1 is dropped despite
// being causally valid. The oracle ranks the past blocks independently; a
// keep-all-past bug (or a wrong rank) diverges because block1's tokens would
// wrongly enter the softmax.
func TestForward_TopKDropsLowerPastBlock_Good(t *testing.T) {
	requireMetalRuntime(t)

	const blockSize = 2
	const topK = 1
	m := identityMixer(2, blockSize, topK, 1.0)
	defer freeMixer(m)

	// Both past blocks align with the +x queries so BOTH would carry real
	// softmax mass if attended — what makes a keep-all-past bug numerically loud.
	// block0 mean key [10,0], block1 mean key [9,0]: block0 ranks just above
	// block1, so top-1 keeps block0 and DROPS block1 (whose tokens, k≈[9,0],
	// would otherwise grab weight comparable to block0's). block2 is self.
	rows := [][]float32{{10, 0}, {10, 0}, {9, 0}, {9, 0}, {5, 1}, {7, 2}}
	got := runForward(t, m, rows)

	want := denseAttentionMasked(rows, mobaSelectionOracle(rows, blockSize, topK), 1.0)
	checkRows(t, "topK drops lower past block", got, want)
}

// mobaSelectionOracle reproduces MoBA's per-query kept-block set in pure Go from
// first principles (q·mean-key block scores, self-block forced in, top-K past
// blocks, future forbidden), then expands it to a per-token causal allow
// predicate. It NEVER calls the mixer's blockSelectMask — the selection is
// computed independently so the Forward test is a real oracle, not a tautology.
func mobaSelectionOracle(rows [][]float32, blockSize, topK int) func(t, j int) bool {
	L := len(rows)
	dim := len(rows[0])
	nBlocks := L / blockSize

	// Block mean keys (K == rows under identity K-proj).
	meanKey := make([][]float64, nBlocks)
	for b := 0; b < nBlocks; b++ {
		mk := make([]float64, dim)
		for off := 0; off < blockSize; off++ {
			for d := 0; d < dim; d++ {
				mk[d] += float64(rows[b*blockSize+off][d])
			}
		}
		for d := range mk {
			mk[d] /= float64(blockSize)
		}
		meanKey[b] = mk
	}

	// Per-query kept-block set.
	kept := make([]map[int]bool, L)
	for q := 0; q < L; q++ {
		self := q / blockSize
		keep := map[int]bool{self: true} // self-block always kept
		// Score strictly-past blocks (b < self) and keep the top-K.
		type bs struct {
			b     int
			score float64
		}
		var past []bs
		for b := 0; b < self; b++ {
			dot := 0.0
			for d := 0; d < dim; d++ {
				dot += float64(rows[q][d]) * meanKey[b][d]
			}
			past = append(past, bs{b, dot})
		}
		// Descending sort by score (small set; simple selection sort).
		for i := 0; i < len(past); i++ {
			for j2 := i + 1; j2 < len(past); j2++ {
				if past[j2].score > past[i].score {
					past[i], past[j2] = past[j2], past[i]
				}
			}
		}
		for i := 0; i < topK && i < len(past); i++ {
			keep[past[i].b] = true
		}
		kept[q] = keep
	}

	return func(t, j int) bool {
		if j > t {
			return false // causal
		}
		return kept[t][j/blockSize]
	}
}

// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package nsa

import (
	"math"
	"testing"

	metal "dappco.re/go/mlx/pkg/metal"
)

// forward_oracle_test.go pins nsa.Mixer.Forward end-to-end against an
// INDEPENDENT pure-Go reference. NSA is not a single masked softmax — it is a
// three-branch (compression / selection / sliding) sigmoid-gated BLEND — so the
// oracle is matched to that mechanism: each branch is reconstructed in pure Go
// and the branches are combined with the EXACT sigmoid gates the test fixes via
// a constant-bias gate projection. The branch references never read back from
// the mixer's own masks (no tautology); the selection branch is exercised in the
// keep-all regime (SelectBlocks >= nBlocks) so it collapses to a clean dense
// causal form rather than requiring a circular top-n re-derivation. Covers the
// Forward-level composition the shape-only loader tests cannot see, including the
// short-sequence edge (L < BlockSize → nBlocks == 0) where a past bug lived: that
// path routes straight to the sliding branch with no gate, so its oracle is a
// plain sliding-window causal softmax.

const oracleTol = 1e-4

// identityLinear builds an n×n identity-weight Linear (Forward computes x·Wᵀ =
// x). Identity Q/K/V/O make the attention see Q=K=V=x, so the Forward math is
// determined purely by the masks + gate the test controls.
func identityLinear(n int32) *metal.Linear {
	data := make([]float32, int(n)*int(n))
	for i := int32(0); i < n; i++ {
		data[int(i)*int(n)+int(i)] = 1
	}
	return metal.NewLinear(metal.FromValues(data, int(n), int(n)), nil)
}

// constGateProj builds a single-head GProj whose output is the CONSTANT logit
// vector [lc, ls, lw] for every token, independent of x: a zero weight matrix
// plus a bias of the three logits. NSA applies sigmoid to these, so the gate
// values are exactly (σ(lc), σ(ls), σ(lw)) at every position — the test then
// weights its branch oracles by those same exact gates. out = heads*3 = 3.
func constGateProj(lc, ls, lw float32) *metal.Linear {
	const hidden = 2
	weight := metal.FromValues(make([]float32, 3*hidden), 3, hidden) // zero → x·0ᵀ = 0
	bias := metal.FromValues([]float32{lc, ls, lw}, 3)
	return metal.NewLinear(weight, bias)
}

func sigmoid(x float64) float64 { return 1.0 / (1.0 + math.Exp(-x)) }

// identityMixer builds a single-head NSA mixer with identity Q/K/V/O projections
// and the given sparse geometry; the gate projection is supplied separately so a
// test can pin the blend weights. hidden = HeadDim = 2.
func identityMixer(gproj *metal.Linear, blockSize, selectBlocks, window int32, scale float32) *Mixer {
	return &Mixer{
		QProj: identityLinear(2), KProj: identityLinear(2),
		VProj: identityLinear(2), GProj: gproj, OProj: identityLinear(2),
		NumHeads: 1, HeadDim: 2,
		BlockSize: blockSize, SelectBlocks: selectBlocks, Window: window, Scale: scale,
	}
}

func freeMixer(m *Mixer) {
	for _, l := range []*metal.Linear{m.QProj, m.KProj, m.VProj, m.GProj, m.OProj} {
		metal.FreeLinear(l)
	}
}

// runForward drives Mixer.Forward on a [1,L,dim] input from rows, returning
// [L][dim]. NSA's Forward ignores ctx.Cache (it attends within the chunk).
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
		for c := range want[r] {
			if diff := math.Abs(float64(got[r][c] - want[r][c])); diff > oracleTol {
				t.Errorf("%s[%d][%d] = %f, want %f (diff %g)", name, r, c, got[r][c], want[r][c], diff)
			}
		}
	}
}

// denseSoftmax is the per-query softmax-weighted sum oracle over a value set:
// for query t, the allowed (logit, valueRow) pairs are softmaxed and combined.
// logit(t,j) and the value rows are supplied by the branch reference. Pure Go.
func denseSoftmax(L, dim int, allowed func(t int) (logits []float64, values [][]float64)) [][]float32 {
	out := make([][]float32, L)
	for t := 0; t < L; t++ {
		logits, values := allowed(t)
		maxL := math.Inf(-1)
		for _, l := range logits {
			if l > maxL {
				maxL = l
			}
		}
		var sum float64
		w := make([]float64, len(logits))
		for i, l := range logits {
			w[i] = math.Exp(l - maxL)
			sum += w[i]
		}
		row := make([]float32, dim)
		for i := range logits {
			f := w[i] / sum
			for d := 0; d < dim; d++ {
				row[d] += float32(f * values[i][d])
			}
		}
		out[t] = row
	}
	return out
}

// rows64 converts the float32 fixture rows to float64 for the oracle math.
func rows64(rows [][]float32) [][]float64 {
	r := make([][]float64, len(rows))
	for i := range rows {
		r[i] = make([]float64, len(rows[i]))
		for j := range rows[i] {
			r[i][j] = float64(rows[i][j])
		}
	}
	return r
}

func dot(a, b []float64) float64 {
	var s float64
	for i := range a {
		s += a[i] * b[i]
	}
	return s
}

// slidingOracle is the sliding-window branch reference: query t attends to key j
// iff j <= t (causal) AND t-j < window (window <= 0 → full causal). Q=K=V=x.
func slidingOracle(rows [][]float32, window int, scale float64) [][]float32 {
	x := rows64(rows)
	dim := len(x[0])
	return denseSoftmax(len(x), dim, func(t int) ([]float64, [][]float64) {
		var lg []float64
		var vs [][]float64
		for j := 0; j <= t; j++ {
			if window > 0 && (t-j) >= window {
				continue
			}
			lg = append(lg, scale*dot(x[t], x[j]))
			vs = append(vs, x[j])
		}
		return lg, vs
	})
}

// compressionOracle is the compression branch reference: query t attends over
// the mean-pooled block summaries (mean key / mean value of each whole block) of
// the blocks whose first token is at or before t (b*blockSize <= t). Only whole
// blocks (nBlocks = L/blockSize) are summarised; a trailing partial block is
// dropped, matching compressKV.
func compressionOracle(rows [][]float32, blockSize int, scale float64) [][]float32 {
	x := rows64(rows)
	L := len(x)
	dim := len(x[0])
	nBlocks := L / blockSize
	meanK := make([][]float64, nBlocks)
	meanV := make([][]float64, nBlocks)
	for b := 0; b < nBlocks; b++ {
		mk := make([]float64, dim)
		mv := make([]float64, dim)
		for off := 0; off < blockSize; off++ {
			for d := 0; d < dim; d++ {
				mk[d] += x[b*blockSize+off][d]
				mv[d] += x[b*blockSize+off][d]
			}
		}
		for d := 0; d < dim; d++ {
			mk[d] /= float64(blockSize)
			mv[d] /= float64(blockSize)
		}
		meanK[b], meanV[b] = mk, mv
	}
	return denseSoftmax(L, dim, func(t int) ([]float64, [][]float64) {
		var lg []float64
		var vs [][]float64
		for b := 0; b < nBlocks; b++ {
			if b*blockSize > t {
				continue // block summarises future tokens
			}
			lg = append(lg, scale*dot(x[t], meanK[b]))
			vs = append(vs, meanV[b])
		}
		return lg, vs
	})
}

// selectionKeepAllOracle is the selection branch reference in the keep-all regime
// (SelectBlocks >= nBlocks): every causally-valid whole block is kept, so the
// branch is dense causal attention over the tokens of the covered whole blocks
// (j <= t AND j < nBlocks*blockSize). This is the clean reproducible form — no
// top-n re-derivation, so no circular dependence on the mixer's selectionMask.
func selectionKeepAllOracle(rows [][]float32, blockSize int, scale float64) [][]float32 {
	x := rows64(rows)
	L := len(x)
	dim := len(x[0])
	covered := (L / blockSize) * blockSize
	return denseSoftmax(L, dim, func(t int) ([]float64, [][]float64) {
		var lg []float64
		var vs [][]float64
		for j := 0; j <= t && j < covered; j++ {
			lg = append(lg, scale*dot(x[t], x[j]))
			vs = append(vs, x[j])
		}
		return lg, vs
	})
}

// blend forms gc·a + gs·b + gw·c row-by-row with the exact scalar gates.
func blend(a, b, c [][]float32, gc, gs, gw float64) [][]float32 {
	out := make([][]float32, len(a))
	for r := range a {
		out[r] = make([]float32, len(a[r]))
		for d := range a[r] {
			out[r][d] = float32(gc*float64(a[r][d]) + gs*float64(b[r][d]) + gw*float64(c[r][d]))
		}
	}
	return out
}

// TestForward_ShortSeqIsSliding_Good pins the L < BlockSize edge — the exact
// nBlocks == 0 case a past bug lived in. NSA Forward routes straight to the
// sliding branch (no gate, no blend) here, so the oracle is a plain sliding-
// window causal softmax. A degenerate zero-block grid or an accidental blend
// would diverge. The gate projection is present but unused on this path.
func TestForward_ShortSeqIsSliding_Good(t *testing.T) {
	requireMetalRuntime(t)

	const window = 2
	m := identityMixer(constGateProj(0, 0, 0), 8, 1, window, 1.0) // BlockSize 8 > L=3
	defer freeMixer(m)

	rows := [][]float32{{1, 0}, {0, 1}, {1, 1}}
	got := runForward(t, m, rows)

	want := slidingOracle(rows, window, 1.0)
	checkRows(t, "short-seq sliding", got, want)
}

// TestForward_SlidingBranchIsolated_Good drives the full three-branch path
// (L >= BlockSize) but gates the blend so ONLY the sliding branch survives
// (gw≈1, gc,gs≈0). The output must equal the sliding-window oracle scaled by the
// exact swa gate. Proves the blend wiring routes the sliding branch to slot 2
// and that the other branches are genuinely gated off (a transposed gate-index
// bug — swa weight landing on cmp/slc — would diverge).
func TestForward_SlidingBranchIsolated_Good(t *testing.T) {
	requireMetalRuntime(t)

	const window = 2
	// Logits: cmp,slc strongly negative → ~0; swa strongly positive → ~1.
	lc, ls, lw := float32(-30), float32(-30), float32(30)
	m := identityMixer(constGateProj(lc, ls, lw), 2, 1, window, 1.0) // L=4 → 2 blocks
	defer freeMixer(m)

	rows := [][]float32{{1, 0}, {0, 1}, {1, 1}, {2, 0}}
	got := runForward(t, m, rows)

	gc, gs, gw := sigmoid(float64(lc)), sigmoid(float64(ls)), sigmoid(float64(lw))
	swa := slidingOracle(rows, window, 1.0)
	// cmp/slc still contribute gc,gs (~1e-13 each) — include them at the exact
	// gates so the 1e-4 bound is honest, with the keep-all selection regime.
	cmp := compressionOracle(rows, 2, 1.0)
	slc := selectionKeepAllOracle(rows, 2, 1.0)
	want := blend(cmp, slc, swa, gc, gs, gw)
	checkRows(t, "sliding isolated", got, want)
}

// TestForward_CompressionBranchIsolated_Good gates the blend so ONLY the
// compression branch survives (gc≈1). The output must equal the dense softmax
// over the mean-pooled block summaries. Pins the compression branch end-to-end
// (mean-pool → block-causal softmax → blend slot 0) — the branch the sliding
// oracle cannot see.
func TestForward_CompressionBranchIsolated_Good(t *testing.T) {
	requireMetalRuntime(t)

	lc, ls, lw := float32(30), float32(-30), float32(-30)
	m := identityMixer(constGateProj(lc, ls, lw), 2, 2, 2, 1.0) // L=4 → 2 blocks
	defer freeMixer(m)

	rows := [][]float32{{2, 0}, {0, 2}, {1, 1}, {3, 1}}
	got := runForward(t, m, rows)

	gc, gs, gw := sigmoid(float64(lc)), sigmoid(float64(ls)), sigmoid(float64(lw))
	cmp := compressionOracle(rows, 2, 1.0)
	slc := selectionKeepAllOracle(rows, 2, 1.0)
	swa := slidingOracle(rows, 2, 1.0)
	want := blend(cmp, slc, swa, gc, gs, gw)
	checkRows(t, "compression isolated", got, want)
}

// TestForward_FullBlend_Good is the full three-branch composition with all gates
// open (each ≈ given sigmoid), exercising compression + selection + sliding
// together. SelectBlocks >= nBlocks puts the selection branch in the keep-all
// regime so all three branches have clean independent references, and the blend
// uses the EXACT sigmoid gates. This is the end-to-end Forward oracle: any
// per-branch wiring error, gate-axis transpose, or blend-arithmetic slip
// diverges.
func TestForward_FullBlend_Good(t *testing.T) {
	requireMetalRuntime(t)

	const window = 2
	lc, ls, lw := float32(0.4), float32(-0.7), float32(1.1) // distinct, all moderate
	m := identityMixer(constGateProj(lc, ls, lw), 2, 2, window, 1.0)
	defer freeMixer(m)

	rows := [][]float32{{1, 0}, {0, 1}, {2, 1}, {1, 2}}
	got := runForward(t, m, rows)

	gc, gs, gw := sigmoid(float64(lc)), sigmoid(float64(ls)), sigmoid(float64(lw))
	cmp := compressionOracle(rows, 2, 1.0)
	slc := selectionKeepAllOracle(rows, 2, 1.0)
	swa := slidingOracle(rows, window, 1.0)
	want := blend(cmp, slc, swa, gc, gs, gw)
	checkRows(t, "full blend", got, want)
}

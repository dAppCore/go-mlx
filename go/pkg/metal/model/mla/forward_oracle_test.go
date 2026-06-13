// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package mla

import (
	"math"
	"testing"

	metal "dappco.re/go/mlx/pkg/metal"
)

// forward_oracle_test.go pins mla.Mixer.Forward end-to-end against an
// INDEPENDENT pure-Go latent-expansion reference: decompress per-head K/V from
// the KV latent via the per-head-interleaved split, run standard causal SDPA
// per head, merge, project out. TestUpProjectKV already pins the split in
// isolation at HeadDim=1; this exercises the split WIDTH flowing through
// splitHeads → attendLatent → OProj at heads=2, HeadDim=2 — the composition the
// kv_b_proj per-head-vs-block-concatenated bug corrupts. A wrong (block) split
// would pair head 0's K with head 1's V and diverge numerically here, where a
// shape-only test would still pass.

const oracleTol = 1e-4

// rowMajor builds a [rows×cols] float32 weight from a nested literal.
func rowMajor(m [][]float32) []float32 {
	out := make([]float32, 0, len(m)*len(m[0]))
	for _, r := range m {
		out = append(out, r...)
	}
	return out
}

// selectorLinear builds an [outDim×inDim] Linear that picks specific input
// columns: row r is a one-hot at cols[r], so Forward(x)[...,r] = x[...,cols[r]].
// Used to project the 8-wide latent down to the 4-wide query without introducing
// opaque arithmetic — the query columns are exactly known input columns.
func selectorLinear(inDim int32, cols []int32) *metal.Linear {
	rows := make([][]float32, len(cols))
	for r, c := range cols {
		row := make([]float32, inDim)
		row[c] = 1
		rows[r] = row
	}
	return metal.NewLinear(metal.FromValues(rowMajor(rows), len(cols), int(inDim)), nil)
}

// identityLinear builds an n×n identity Linear (Forward computes x·Iᵀ = x).
func identityLinear(n int32) *metal.Linear {
	data := make([]float32, int(n)*int(n))
	for i := int32(0); i < n; i++ {
		data[int(i)*int(n)+int(i)] = 1
	}
	return metal.NewLinear(metal.FromValues(data, int(n), int(n)), nil)
}

// causalMaskLLT builds the additive [1,1,L,L] causal mask: query t sees key j
// iff j <= t. Allowed → 0, forbidden → -inf.
func causalMaskLLT(L int) *metal.Array {
	n := float32(math.Inf(-1))
	data := make([]float32, L*L)
	for t := 0; t < L; t++ {
		for j := 0; j < L; j++ {
			if j > t {
				data[t*L+j] = n
			}
		}
	}
	return metal.FromValues(data, 1, 1, L, L)
}

// TestForward_LatentExpansionPerHead_Good pins the full MLA Forward against the
// explicit latent-expansion oracle. Construction (hidden=8, heads=2, HeadDim=2):
//
//   - WDQ = WDKV = identity[8×8] so the query/KV latents are exactly x.
//   - WUK = identity[8×8] so kv == x; the per-head-interleaved split then reads
//     head h's [K(2) | V(2)] from x columns [4h .. 4h+4):
//     head0 K=x[0:2] V=x[2:4]; head1 K=x[4:6] V=x[6:8].
//   - WUQ selects q columns [0,1,4,5] → q_head0 = x[0:2] = K_head0,
//     q_head1 = x[4:6] = K_head1 (a clean, traceable query).
//   - OProj = identity[4×4] so the output is the merged head outputs directly.
//
// The oracle decompresses K/V per head with the SAME per-head-interleaved layout
// and runs causal SDPA per head in pure Go. A block-concatenated split (the bug)
// would set V_head0 = x[4:6] and diverge.
func TestForward_LatentExpansionPerHead_Good(t *testing.T) {
	requireMetalRuntime(t)

	const heads, headDim, hidden, L = 2, 2, 8, 3
	m := &Mixer{
		WDQ:  identityLinear(hidden),
		WUQ:  selectorLinear(hidden, []int32{0, 1, 4, 5}), // q = [x0,x1,x4,x5]
		WDKV: identityLinear(hidden),
		WUK:  identityLinear(hidden), // kv == cKV == x, width 8 = heads*2*headDim
		WUV:  nil,                    // single-WUK per-head-interleaved path
		OProj: identityLinear(heads * headDim),
		NumHeads: heads, HeadDim: headDim, Scale: 1.0,
	}
	defer freeMLAMixer(m)

	// Distinct per-column sentinels so a mis-routed split is numerically loud.
	rows := [][]float32{
		{1, 2, 3, 4, 5, 6, 7, 8},
		{2, 1, 4, 3, 6, 5, 8, 7},
		{0, 1, 1, 0, 2, 0, 0, 3},
	}
	x := metal.FromValues(rowMajor(rows), 1, L, hidden)
	defer metal.Free(x)
	mask := causalMaskLLT(L)
	defer metal.Free(mask)

	out, _ := m.Forward(x, &metal.MixerCtx{B: 1, L: L, Mask: mask})
	defer metal.Free(out)
	got := out.Floats()

	want := mlaLatentOracle(rows, heads, headDim, []int{0, 1, 4, 5}, 1.0)
	for i := range want {
		if diff := math.Abs(float64(got[i] - want[i])); diff > oracleTol {
			t.Errorf("Forward[%d] = %f, want %f (diff %g)", i, got[i], want[i], diff)
		}
	}
}

// mlaLatentOracle is the independent reference: per head h, K/V are read from x
// with the per-head-interleaved layout (head h occupies columns [4h..4h+4): K =
// first headDim, V = last headDim), q is x's qCols. It runs causal SDPA per head
// and concatenates the heads (OProj == identity), returning a flat [L*heads*
// headDim] in [L, head, dim] order to match mergeHeads. Pure Go.
func mlaLatentOracle(rows [][]float32, heads, headDim int, qCols []int, scale float64) []float32 {
	L := len(rows)
	x := make([][]float64, L)
	for i := range rows {
		x[i] = make([]float64, len(rows[i]))
		for j := range rows[i] {
			x[i][j] = float64(rows[i][j])
		}
	}
	perHead := 2 * headDim // [K | V] per head in the interleaved kv vector

	// q[t][h][d] from the selected columns: qCols is laid out head-major
	// ([h0_d0,h0_d1,h1_d0,h1_d1]).
	qOf := func(t, h, d int) float64 { return x[t][qCols[h*headDim+d]] }
	// K/V[t][h][d] from the per-head-interleaved kv (== x): head h at base 4h.
	kOf := func(t, h, d int) float64 { return x[t][h*perHead+d] }
	vOf := func(t, h, d int) float64 { return x[t][h*perHead+headDim+d] }

	out := make([]float32, L*heads*headDim)
	for t := 0; t < L; t++ {
		for h := 0; h < heads; h++ {
			// Causal scores over keys j <= t.
			logits := make([]float64, t+1)
			for j := 0; j <= t; j++ {
				var s float64
				for d := 0; d < headDim; d++ {
					s += qOf(t, h, d) * kOf(j, h, d)
				}
				logits[j] = scale * s
			}
			maxL := math.Inf(-1)
			for _, l := range logits {
				if l > maxL {
					maxL = l
				}
			}
			var sum float64
			w := make([]float64, t+1)
			for j := 0; j <= t; j++ {
				w[j] = math.Exp(logits[j] - maxL)
				sum += w[j]
			}
			for d := 0; d < headDim; d++ {
				var acc float64
				for j := 0; j <= t; j++ {
					acc += (w[j] / sum) * vOf(j, h, d)
				}
				// mergeHeads packs [B,L,H*D] in head-major order: index = t*(H*D)+h*D+d.
				out[t*heads*headDim+h*headDim+d] = float32(acc)
			}
		}
	}
	return out
}

// freeMLAMixer releases the mixer's projection weights (WUV is nil here).
func freeMLAMixer(m *Mixer) {
	for _, l := range []*metal.Linear{m.WDQ, m.WUQ, m.WDKV, m.WUK, m.OProj} {
		metal.FreeLinear(l)
	}
}

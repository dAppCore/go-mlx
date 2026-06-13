// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package rwkv7

import (
	"math"
	"testing"

	metal "dappco.re/go/mlx/pkg/metal"
)

// forward_test.go closes the layout gap the kernel oracles leave open. The WKV7
// recurrence already has an independent float64 oracle (wkv7Reference in
// recurrence_test.go) and the chunked path is checked against the sequential
// kernel (chunk_test.go) — but Mixer.Forward's projection/reshape GLUE has only
// a prefill-vs-decode self-consistency test (TestMixer_Forward_Bad), which runs
// the same glue on both sides and so cannot see a consistent layout bug (a
// transposed projection, a wrong per-head reshape, the r/k/v/a/b ports crossed).
//
// This file adds an INDEPENDENT pure-Go oracle for the whole block: it runs the
// r/k/v/a/b projections + the w = -exp(WProj) log-decay + the per-head reshape
// by hand from the same weights, feeds the post-projection per-head values into
// the trusted wkv7Reference, then applies the head-merge + out-projection — and
// asserts the real chunked Forward matches it. A layout bug shows up as gross
// element-wrong-slot error, well above the float32 tolerance.

// linearForward mirrors metal.Linear's dense path y = x @ W.T for a weight
// stored row-major [out, in]: y[o] = Σ_i x[i]·W[o*in + i]. The mixer's
// projections carry no bias, so this is the whole projection.
func linearForward(x, w []float64, in, out int) []float64 {
	y := make([]float64, out)
	for o := 0; o < out; o++ {
		var acc float64
		for i := 0; i < in; i++ {
			acc += x[i] * w[o*in+i]
		}
		y[o] = acc
	}
	return y
}

// weightFloats reads a metal.Array weight back to a host float64 slice in
// row-major order so the pure-Go oracle multiplies the exact loaded values.
func weightFloats(a *metal.Array) []float64 {
	f := a.Floats()
	out := make([]float64, len(f))
	for i, v := range f {
		out[i] = float64(v)
	}
	return out
}

// to32 narrows the float64 oracle output to float32 for the elementwise compare
// against the kernel's float32 result.
func to32(in []float64) []float32 {
	out := make([]float32, len(in))
	for i, v := range in {
		out[i] = float32(v)
	}
	return out
}

// forwardReference is the independent pure-Go oracle for the full RWKV-7 block.
// It reproduces only the NEW glue Mixer.Forward owns — the r/k/v/a/b
// projections, the w = -exp(WProj·x) log-decay, the per-head split — then defers
// the recurrence to the already-trusted wkv7Reference, and finishes with the
// head-merge + out-projection. xSeq is [L][D]; the return is the flat [L*D]
// out-projected output, matching Forward's [B,L,D] (B=1) flatten.
func forwardReference(m *Mixer, xSeq [][]float64) []float64 {
	cfg := m.W.cfg
	H := int(cfg.NumHeads)
	K := int(cfg.KeyDim)
	Vd := int(cfg.ValueDim)
	D := len(xSeq[0])
	L := len(xSeq)
	hk := H * K
	hv := H * Vd

	rW := weightFloats(m.W.RProj.Weight)
	wW := weightFloats(m.W.WProj.Weight)
	kW := weightFloats(m.W.KProj.Weight)
	vW := weightFloats(m.W.VProj.Weight)
	aW := weightFloats(m.W.AProj.Weight)
	bW := weightFloats(m.W.BProj.Weight)
	oW := weightFloats(m.W.OutProj.Weight)

	// Per-step projections, flattened to the [L,H,K] / [L,H,V] layout
	// wkv7Reference indexes as (t*H+h)*K+c.
	r := make([]float64, L*hk)
	w := make([]float64, L*hk)
	k := make([]float64, L*hk)
	a := make([]float64, L*hk)
	b := make([]float64, L*hk)
	v := make([]float64, L*hv)
	for t := 0; t < L; t++ {
		copy(r[t*hk:(t+1)*hk], linearForward(xSeq[t], rW, D, hk))
		copy(k[t*hk:(t+1)*hk], linearForward(xSeq[t], kW, D, hk))
		copy(a[t*hk:(t+1)*hk], linearForward(xSeq[t], aW, D, hk))
		copy(b[t*hk:(t+1)*hk], linearForward(xSeq[t], bW, D, hk))
		copy(v[t*hv:(t+1)*hv], linearForward(xSeq[t], vW, D, hv))
		// w = -exp(WProj·x): the log-decay the recurrence scales the state by.
		rawW := linearForward(xSeq[t], wW, D, hk)
		for i := 0; i < hk; i++ {
			w[t*hk+i] = -math.Exp(rawW[i])
		}
	}

	// Defer the recurrence to the trusted oracle; o is [L,H,V] flattened
	// (t*H+h)*Vd+vd — the same order Forward's [B,L,H,V]→[B,L,H*V] merge keeps.
	o, _ := wkv7Reference(L, H, K, Vd, r, w, k, v, a, b)

	// Head-merge [L,H,V] → [L, H*V] is identity in this flat order, then
	// out-proj [H*V → D].
	out := make([]float64, L*D)
	for t := 0; t < L; t++ {
		copy(out[t*D:(t+1)*D], linearForward(o[t*hv:(t+1)*hv], oW, hv, D))
	}
	return out
}

// TestForward_Block_Good is the layout gate: the real chunked Forward (prefill,
// L>1) must match the independent pure-Go block oracle within float32 noise. A
// crossed projection port, a transposed per-head reshape, or a head-merge in the
// wrong order produces element-wrong-slot error far above the tolerance.
func TestForward_Block_Good(t *testing.T) {
	m := tinyMixer(t)
	const B, L, D = 1, 3, 4
	xVals := []float32{
		0.5, -0.2, 0.1, 0.3,
		-0.1, 0.4, -0.3, 0.2,
		0.2, 0.1, -0.4, 0.05,
	}
	x := metal.FromValues(xVals, B, L, D)
	defer metal.Free(x)

	rc := metal.NewRecurrentCache()
	out, _ := m.Forward(x, &metal.MixerCtx{Cache: rc, B: B, L: L})
	if out == nil {
		t.Fatal("Forward returned nil output")
	}
	defer metal.Free(out)

	// Build the pure-Go reference from the same weights + the same input.
	xSeq := make([][]float64, L)
	for tIdx := 0; tIdx < L; tIdx++ {
		row := make([]float64, D)
		for c := 0; c < D; c++ {
			row[c] = float64(xVals[tIdx*D+c])
		}
		xSeq[tIdx] = row
	}
	want := forwardReference(m, xSeq)

	closeEnough(t, "forward-block-vs-reference", out.Floats(), to32(want), 2e-3)
}

// TestForward_Block_Bad threads a non-zero prior state through the block oracle
// (the decode-continuation layout): a 1-token Forward carrying a holder state
// must match the pure-Go reference run from the same prior. This pins the prior
// read-out + state write-back routing in the block, not just the zero-state
// prefill path.
func TestForward_Block_Bad(t *testing.T) {
	m := tinyMixer(t)
	cfg := m.W.cfg
	const B, D = 1, 4
	H, K, Vd := int(cfg.NumHeads), int(cfg.KeyDim), int(cfg.ValueDim)

	// Seed the holder with a known prior [B,H,K,V] state.
	priorVals := make([]float32, B*H*K*Vd)
	for i := range priorVals {
		priorVals[i] = 0.1 * float32(i%5-2)
	}
	rc := metal.NewRecurrentCache()
	rc.SetRecurrentState([]*metal.Array{metal.FromValues(priorVals, B, H, K, Vd)})

	xVals := []float32{0.3, -0.1, 0.2, 0.4}
	x := metal.FromValues(xVals, B, 1, D)
	defer metal.Free(x)
	out, _ := m.Forward(x, &metal.MixerCtx{Cache: rc, B: B, L: 1, Step: 0})
	if out == nil {
		t.Fatal("Forward returned nil output")
	}
	defer metal.Free(out)

	// Pure-Go reference: project the single token, then run one WKV7 step from
	// the same prior via wkv7Reference's per-step math (L=1), then out-proj.
	xRow := make([]float64, D)
	for c := 0; c < D; c++ {
		xRow[c] = float64(xVals[c])
	}
	want := forwardReferenceWithPrior(m, [][]float64{xRow}, priorVals)

	closeEnough(t, "forward-decode-vs-reference", out.Floats(), to32(want), 2e-3)
}

// forwardReferenceWithPrior is forwardReference for a single decode step carrying
// a non-zero prior [H][K][V] state (priorFlat is [B,H,K,V] with B=1, row-major).
// It runs the projections by hand, then a single WKV7 step from the seeded state,
// then the out-projection — the independent oracle for the decode-continuation
// layout.
func forwardReferenceWithPrior(m *Mixer, xSeq [][]float64, priorFlat []float32) []float64 {
	cfg := m.W.cfg
	H := int(cfg.NumHeads)
	K := int(cfg.KeyDim)
	Vd := int(cfg.ValueDim)
	D := len(xSeq[0])
	hk := H * K
	hv := H * Vd

	rW := weightFloats(m.W.RProj.Weight)
	wW := weightFloats(m.W.WProj.Weight)
	kW := weightFloats(m.W.KProj.Weight)
	vW := weightFloats(m.W.VProj.Weight)
	aW := weightFloats(m.W.AProj.Weight)
	bW := weightFloats(m.W.BProj.Weight)
	oW := weightFloats(m.W.OutProj.Weight)

	rt := linearForward(xSeq[0], rW, D, hk)
	kt := linearForward(xSeq[0], kW, D, hk)
	at := linearForward(xSeq[0], aW, D, hk)
	bt := linearForward(xSeq[0], bW, D, hk)
	vt := linearForward(xSeq[0], vW, D, hv)
	rawW := linearForward(xSeq[0], wW, D, hk)
	wt := make([]float64, hk)
	for i := 0; i < hk; i++ {
		wt[i] = -math.Exp(rawW[i])
	}

	// Seed state [H][K][V] from the prior, run one WKV7 step by hand (mirrors
	// wkv7Reference's inner body for a single timestep).
	state := make([][][]float64, H)
	for h := 0; h < H; h++ {
		state[h] = make([][]float64, K)
		for kk := 0; kk < K; kk++ {
			state[h][kk] = make([]float64, Vd)
			for vd := 0; vd < Vd; vd++ {
				state[h][kk][vd] = float64(priorFlat[((h*K)+kk)*Vd+vd])
			}
		}
	}
	o := make([]float64, H*Vd)
	for h := 0; h < H; h++ {
		sa := make([]float64, Vd)
		for vd := 0; vd < Vd; vd++ {
			for kk := 0; kk < K; kk++ {
				sa[vd] += at[h*K+kk] * state[h][kk][vd]
			}
		}
		for kk := 0; kk < K; kk++ {
			ew := math.Exp(wt[h*K+kk])
			for vd := 0; vd < Vd; vd++ {
				state[h][kk][vd] = ew*state[h][kk][vd] +
					bt[h*K+kk]*sa[vd] + kt[h*K+kk]*vt[h*Vd+vd]
			}
		}
		for vd := 0; vd < Vd; vd++ {
			var acc float64
			for kk := 0; kk < K; kk++ {
				acc += state[h][kk][vd] * rt[h*K+kk]
			}
			o[h*Vd+vd] = acc
		}
	}

	return linearForward(o, oW, hv, D)
}

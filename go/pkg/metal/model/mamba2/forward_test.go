// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package mamba2

import (
	"math"
	"testing"

	metal "dappco.re/go/mlx/pkg/metal"
)

// forward_test.go closes the layout gap the kernel oracles leave open. The SSD
// recurrence already has an independent float64 oracle (scanReference in
// scan_test.go) and the chunked path is checked against the sequential kernel
// (chunk_test.go) — but Mixer.Forward's block assembly (the in-proj split, the
// causal depthwise conv, the group→heads expansion, the dt activation, the
// gated norm) has only a prefill-vs-decode self-consistency test
// (TestMixer_Forward_Bad). That test runs the same glue on both sides, so a
// CONSISTENT layout bug — a wrong split offset, the B/C ports crossed, a group
// mapped to the wrong heads — passes it silently. That is exactly the bug class
// a sibling lane's sparse mixers shipped.
//
// This file adds an INDEPENDENT pure-Go oracle for the whole prefill block: it
// runs the projection, the causal conv, the splits and the group expansion by
// hand from the same weights — encoding the INTENDED layout with explicit
// indices rather than transcribing the MLX op sequence — defers the recurrence
// to the trusted scanReference, then applies the gated norm + out-projection,
// and asserts the real chunked Forward matches. A layout bug shows up as
// element-wrong-slot error far above the float32 tolerance.
//
// Note on the gated norm: the mixer applies SiLU(z)·RMSNorm(y) (norm-BEFORE-
// gate). The oracle mirrors that form, so this is a layout check, not an
// independent blessing of the gate-order math — see gatedNormReference.

// oracleConfig is a hand-traceable mamba2 geometry whose group count is a
// PROPER divisor of the head count (1 < G < H) so the group→heads expansion is
// exercised in its real repeat mode, not the degenerate G==1 broadcast or G==H
// no-op the package's tinyMixer uses.
type oracleConfig struct {
	D, H, P, N, G, K int
}

func (c oracleConfig) dInner() int  { return c.H * c.P }
func (c oracleConfig) convDim() int { return c.dInner() + 2*c.G*c.N }
func (c oracleConfig) projOut() int { return 2*c.dInner() + 2*c.G*c.N + c.H }

// oracleWeights holds the host-side float64 copies of a fixture's weights so the
// pure-Go oracle multiplies the exact loaded values and the metal Mixer is built
// from the same float32 source.
type oracleWeights struct {
	inProj  []float64 // [projOut, D]
	conv    []float64 // [convDim, 1, K]
	aLog    []float64 // [H]
	dSkip   []float64 // [H]
	dtBias  []float64 // [H]
	norm    []float64 // [dInner]
	outProj []float64 // [D, dInner]
}

// buildOracleFixture builds a deterministic weight set for the given geometry
// and returns both the host float64 copies (for the oracle) and a live *Mixer
// (for the kernel) constructed from the identical float32 values.
func buildOracleFixture(cfg oracleConfig) (oracleWeights, *Mixer) {
	w := oracleWeights{
		inProj:  make([]float64, cfg.projOut()*cfg.D),
		conv:    make([]float64, cfg.convDim()*cfg.K),
		aLog:    make([]float64, cfg.H),
		dSkip:   make([]float64, cfg.H),
		dtBias:  make([]float64, cfg.H),
		norm:    make([]float64, cfg.dInner()),
		outProj: make([]float64, cfg.D*cfg.dInner()),
	}
	// Base in-proj pattern, then a strong PER-GROUP scale on the B and C output
	// rows so the groups are O(1)-separated. Without this, a group→heads
	// mis-routing (the prime layout-bug site) only perturbs the SSD output by a
	// few 1e-4 — under any sane tolerance — and the test would rubber-stamp the
	// bug (the shape-only trap, one rung up). With per-group scaling a swapped
	// group produces gross, unmissable error. The projection's output rows lay
	// out as [z: dInner | x_inner: dInner | B: groupDim | C: groupDim | dt: H];
	// rows in the B/C blocks get the scale of the group they feed.
	dInner := cfg.dInner()
	groupDim := cfg.G * cfg.N
	bcStart := dInner + dInner            // first B row
	bcEnd := bcStart + 2*groupDim         // last C row
	groupScale := func(out int) float64 { // 1× group 0, 6× group 1, 11× …
		if out < bcStart || out >= bcEnd {
			return 1
		}
		g := ((out - bcStart) % groupDim) / cfg.N // which group this B/C channel feeds
		return 1 + 5*float64(g)
	}
	for r := 0; r < cfg.projOut(); r++ {
		for c := 0; c < cfg.D; c++ {
			i := r*cfg.D + c
			w.inProj[i] = math.Sin(float64(i)*0.13) * 0.2 * groupScale(r)
		}
	}
	for i := range w.conv {
		w.conv[i] = 0.1 * float64((i%3)+1) // per-channel taps 0.1,0.2,0.3 cycling
	}
	for h := 0; h < cfg.H; h++ {
		w.aLog[h] = -0.2 - 0.1*float64(h)
		w.dSkip[h] = 0.1 * float64(h+1)
		w.dtBias[h] = -0.5 - 0.1*float64(h)
	}
	for i := range w.norm {
		w.norm[i] = 0.9 + 0.05*float64(i%4)
	}
	for i := range w.outProj {
		w.outProj[i] = math.Cos(float64(i)*0.17) * 0.15
	}

	mcfg := &Config{
		NumHeads: int32(cfg.H), HeadDim: int32(cfg.P), StateDim: int32(cfg.N),
		NumGroups: int32(cfg.G), ConvKernel: int32(cfg.K), RMSNormEps: 1e-5,
	}
	weights := &Weights{
		InProj:     metal.NewLinear(metal.FromValues(to32(w.inProj), cfg.projOut(), cfg.D), nil),
		ConvWeight: metal.FromValues(to32(w.conv), cfg.convDim(), 1, cfg.K),
		ALog:       metal.FromValues(to32(w.aLog), cfg.H),
		D:          metal.FromValues(to32(w.dSkip), cfg.H),
		DtBias:     metal.FromValues(to32(w.dtBias), cfg.H),
		Norm:       metal.FromValues(to32(w.norm), cfg.dInner()),
		OutProj:    metal.NewLinear(metal.FromValues(to32(w.outProj), cfg.D, cfg.dInner()), nil),
	}
	return w, NewMixer(weights, mcfg)
}

// to32 narrows a float64 slice to float32 for the metal fixtures and the
// elementwise compare.
func to32(in []float64) []float32 {
	out := make([]float32, len(in))
	for i, v := range in {
		out[i] = float32(v)
	}
	return out
}

// linearF64 mirrors metal.Linear's dense path y = x @ W.T for a weight stored
// row-major [out, in]: y[o] = Σ_i x[i]·W[o*in + i]. The fixtures carry no bias.
func linearF64(x, w []float64, in, out int) []float64 {
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

// silu is x·sigmoid(x).
func silu(x float64) float64 { return x / (1 + math.Exp(-x)) }

// softplusF64 is log(1+exp(x)) — the dt activation.
func softplusF64(x float64) float64 { return math.Log(1 + math.Exp(x)) }

// forwardReference is the independent pure-Go oracle for the full mamba2 prefill
// block at B=1. xSeq is [L][D]; the return is the flat [L*D] out-projected
// output matching Forward's [B,L,D] flatten. It reproduces the block layout from
// the spec, not the op sequence:
//
//	in-proj → split (z | xBC | dt) → causal conv(xBC) → SiLU
//	→ split (x_inner | B | C) → group→heads → softplus(dt+bias)
//	→ scanReference → SiLU(z)·RMSNorm(y) → out-proj
func forwardReference(w oracleWeights, cfg oracleConfig, xSeq [][]float64) []float64 {
	L, D := len(xSeq), cfg.D
	dInner := cfg.dInner()
	convDim := cfg.convDim()
	groupDim := cfg.G * cfg.N

	// --- in-proj, then split each step's projection into z | xBC | dt --------
	z := make([][]float64, L)     // [L][dInner]
	xBC := make([][]float64, L)   // [L][convDim]
	dtRaw := make([][]float64, L) // [L][H]
	for t := 0; t < L; t++ {
		proj := linearF64(xSeq[t], w.inProj, D, cfg.projOut())
		z[t] = append([]float64(nil), proj[0:dInner]...)
		xBC[t] = append([]float64(nil), proj[dInner:dInner+convDim]...)
		dtRaw[t] = append([]float64(nil), proj[dInner+convDim:dInner+convDim+cfg.H]...)
	}

	// --- causal depthwise conv over xBC, then SiLU ---------------------------
	// Per channel c: conv[t][c] = Σ_{kk} xBCpadded[t+kk][c]·W[c,0,kk], where the
	// sequence is left-padded by K-1 zeros (fresh prefill, no prior ring). Tap
	// W[K-1] multiplies the current token (cross-correlation), W[0] the oldest.
	conv := make([][]float64, L)
	for t := 0; t < L; t++ {
		conv[t] = make([]float64, convDim)
		for c := 0; c < convDim; c++ {
			var acc float64
			for kk := 0; kk < cfg.K; kk++ {
				src := t + kk - (cfg.K - 1) // original-sequence index of this tap
				if src >= 0 {               // src < 0 ⇒ left-pad zero (fresh prefill)
					acc += xBC[src][c] * w.conv[c*cfg.K+kk]
				}
			}
			conv[t][c] = silu(acc)
		}
	}

	// --- split conv output into x_inner | B | C, expand groups to heads ------
	// x_inner [L][dInner] → per-head [L][H][P]; B/C [L][groupDim] → [L][H][N] by
	// repeating each group across the H/G heads that share it (the INTENDED
	// group→heads map, written explicitly so a mis-routed head fails loudly).
	repeat := cfg.H / cfg.G
	xHeads := make([][][]float64, L) // [L][H][P]
	bHeads := make([][][]float64, L) // [L][H][N]
	cHeads := make([][][]float64, L) // [L][H][N]
	for t := 0; t < L; t++ {
		xInner := conv[t][0:dInner]
		bGroup := conv[t][dInner : dInner+groupDim]
		cGroup := conv[t][dInner+groupDim : dInner+2*groupDim]
		xHeads[t] = make([][]float64, cfg.H)
		bHeads[t] = make([][]float64, cfg.H)
		cHeads[t] = make([][]float64, cfg.H)
		for h := 0; h < cfg.H; h++ {
			xHeads[t][h] = append([]float64(nil), xInner[h*cfg.P:(h+1)*cfg.P]...)
			g := h / repeat // head h belongs to group g
			bHeads[t][h] = append([]float64(nil), bGroup[g*cfg.N:(g+1)*cfg.N]...)
			cHeads[t][h] = append([]float64(nil), cGroup[g*cfg.N:(g+1)*cfg.N]...)
		}
	}

	// --- dt = softplus(dt_raw + dt_bias), per head ---------------------------
	dt := make([]float64, L*cfg.H)
	for t := 0; t < L; t++ {
		for h := 0; h < cfg.H; h++ {
			dt[t*cfg.H+h] = softplusF64(dtRaw[t][h] + w.dtBias[h])
		}
	}

	// --- A = -exp(A_log), and the flat [L,H,*] arrays scanReference wants -----
	a := make([]float64, cfg.H)
	for h := 0; h < cfg.H; h++ {
		a[h] = -math.Exp(w.aLog[h])
	}
	xFlat := make([]float64, L*cfg.H*cfg.P) // (t*H+h)*P+p
	bFlat := make([]float64, L*cfg.H*cfg.N)
	cFlat := make([]float64, L*cfg.H*cfg.N)
	for t := 0; t < L; t++ {
		for h := 0; h < cfg.H; h++ {
			copy(xFlat[(t*cfg.H+h)*cfg.P:], xHeads[t][h])
			copy(bFlat[(t*cfg.H+h)*cfg.N:], bHeads[t][h])
			copy(cFlat[(t*cfg.H+h)*cfg.N:], cHeads[t][h])
		}
	}

	// --- defer the recurrence to the trusted oracle --------------------------
	// y is [L,H,P] flattened (t*H+h)*P+p — the same order Forward's
	// [B,L,H,P]→[B,L,dInner] reshape keeps.
	y, _ := scanReference(L, cfg.H, cfg.P, cfg.N, xFlat, dt, a, bFlat, cFlat, w.dSkip)

	// --- gated norm (SiLU(z)·RMSNorm(y)) then out-proj -----------------------
	out := make([]float64, L*D)
	for t := 0; t < L; t++ {
		yt := y[t*dInner : (t+1)*dInner] // [dInner] in h-major order
		gated := gatedNormReference(yt, z[t], w.norm, cfg.RMSNormEps())
		copy(out[t*D:(t+1)*D], linearF64(gated, w.outProj, dInner, D))
	}
	return out
}

// RMSNormEps exposes the fixture eps (oracleConfig carries the geometry; the eps
// is fixed to the metal Config's 1e-5).
func (c oracleConfig) RMSNormEps() float64 { return 1e-5 }

// gatedNormReference mirrors the mixer's gated norm: SiLU(z) · RMSNorm(y), where
// RMSNorm reduces over the WHOLE dInner channel axis (the fused MLX kernel
// normalises over the last axis, and Forward feeds it the flat [dInner] vector).
// This transcribes the code's norm-before-gate form; it pins the y/z channel
// routing, not the gate-order math.
func gatedNormReference(y, z, norm []float64, eps float64) []float64 {
	n := len(y)
	var ms float64
	for _, v := range y {
		ms += v * v
	}
	ms /= float64(n)
	inv := 1.0 / math.Sqrt(ms+eps)
	out := make([]float64, n)
	for i := 0; i < n; i++ {
		out[i] = silu(z[i]) * (y[i] * inv * norm[i])
	}
	return out
}

// TestForward_Block_Good is the layout gate at the degenerate G==1 geometry
// (one B/C shared by every head): the real chunked Forward (prefill, L>1) must
// match the independent pure-Go block oracle within float32 noise. A crossed
// split offset, a transposed reshape, or a y/z swap in the gated norm produces
// element-wrong-slot error far above the tolerance.
func TestForward_Block_Good(t *testing.T) {
	cfg := oracleConfig{D: 4, H: 2, P: 2, N: 2, G: 1, K: 3}
	w, m := buildOracleFixture(cfg)
	defer freeMixer(m)

	const B, L = 1, 4
	xVals := make([]float32, L*cfg.D)
	for i := range xVals {
		xVals[i] = float32(math.Sin(float64(i)*0.37) * 0.5)
	}
	x := metal.FromValues(xVals, B, L, cfg.D)
	defer metal.Free(x)

	rc := metal.NewRecurrentCache()
	out, _ := m.Forward(x, &metal.MixerCtx{Cache: rc, B: B, L: L})
	if out == nil {
		t.Fatal("Forward returned nil output")
	}
	defer metal.Free(out)

	want := forwardReference(w, cfg, rows(xVals, L, cfg.D))
	closeEnough(t, "forward-block-vs-reference (G=1)", out.Floats(), to32(want), 1e-4)
}

// TestForward_Block_Bad is the group→heads routing gate at a PROPER group count
// (H=4, G=2 ⇒ repeat 2: heads 0,1 share group 0, heads 2,3 share group 1). The
// G==1 case cannot see a repeat-mapping bug (it broadcasts one group to all
// heads); this case fails loudly if a group is wired to the wrong heads.
func TestForward_Block_Bad(t *testing.T) {
	cfg := oracleConfig{D: 6, H: 4, P: 2, N: 2, G: 2, K: 3}
	w, m := buildOracleFixture(cfg)
	defer freeMixer(m)

	const B, L = 1, 5
	xVals := make([]float32, L*cfg.D)
	for i := range xVals {
		xVals[i] = float32(math.Cos(float64(i)*0.29) * 0.5)
	}
	x := metal.FromValues(xVals, B, L, cfg.D)
	defer metal.Free(x)

	rc := metal.NewRecurrentCache()
	out, _ := m.Forward(x, &metal.MixerCtx{Cache: rc, B: B, L: L})
	if out == nil {
		t.Fatal("Forward returned nil output")
	}
	defer metal.Free(out)

	want := forwardReference(w, cfg, rows(xVals, L, cfg.D))
	closeEnough(t, "forward-block-vs-reference (H=4,G=2)", out.Floats(), to32(want), 1e-4)
}

// rows reshapes a flat [L*D] float32 input into [L][D] float64 rows for the
// oracle.
func rows(flat []float32, L, D int) [][]float64 {
	out := make([][]float64, L)
	for t := 0; t < L; t++ {
		row := make([]float64, D)
		for c := 0; c < D; c++ {
			row[c] = float64(flat[t*D+c])
		}
		out[t] = row
	}
	return out
}

// freeMixer releases a fixture mixer's metal weights.
func freeMixer(m *Mixer) {
	w := m.W
	metal.Free(w.InProj.Weight, w.ConvWeight, w.ALog, w.D, w.DtBias, w.Norm, w.OutProj.Weight)
}

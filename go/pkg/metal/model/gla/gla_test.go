// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gla

import (
	"math"
	"testing"

	metal "dappco.re/go/mlx/pkg/metal"
	scheme "dappco.re/go/mlx/pkg/scheme"
)

// gla_test.go validates the chunked GatedChunk kernel against an independent,
// plain-Go recurrent reference walking S_t = diag(exp(g_t))·S_{t-1} + k_tᵀv_t;
// o_t = q_t·S_t token by token. That definition is what the q̃=q⊙exp(b),
// k̃=k⊙exp(−b) chunked decomposition must reproduce — proving the per-key-dim
// gate, the causal mask, the cross-chunk read-out, and the advanced state.

// glaReference computes GLA output and final state the slow way for one head /
// one batch. q,k,v are [L][D], g is the per-position per-key-dim LOG-gate [L][Dk]
// (g ≤ 0). prevS is the incoming [Dk][Dv] state (nil ⇒ zero). scale scales q.
func glaReference(q, k, v, g [][]float64, scale float64, prevS [][]float64) (out [][]float64, finalS [][]float64) {
	l := len(q)
	dk := len(k[0])
	dv := len(v[0])
	s := make([][]float64, dk)
	for i := range s {
		s[i] = make([]float64, dv)
		if prevS != nil {
			copy(s[i], prevS[i])
		}
	}
	out = make([][]float64, l)
	for t := range l {
		// S_t = diag(exp(g_t)) · S_{t-1} + k_tᵀ v_t.
		for a := range dk {
			alpha := math.Exp(g[t][a])
			for b := range dv {
				s[a][b] = alpha*s[a][b] + k[t][a]*v[t][b]
			}
		}
		// o_t = (scale·q_t) · S_t.
		o := make([]float64, dv)
		for b := range dv {
			var acc float64
			for a := range dk {
				acc += scale * q[t][a] * s[a][b]
			}
			o[b] = acc
		}
		out[t] = o
	}
	return out, s
}

type glaFixture struct {
	h, l, d int32
}

func (f glaFixture) tensor(seed float32) (*metal.Array, [][]float64) {
	n := int(f.h * f.l * f.d)
	values := make([]float32, n)
	for i := range values {
		values[i] = seed + float32(i%11)*0.1 - float32(i%5)*0.07
	}
	arr := metal.FromValues(values, 1, int(f.h), int(f.l), int(f.d))
	head0 := make([][]float64, f.l)
	for t := range int(f.l) {
		row := make([]float64, f.d)
		for d := range int(f.d) {
			row[d] = float64(values[t*int(f.d)+d])
		}
		head0[t] = row
	}
	return arr, head0
}

// gate builds a fixed NEGATIVE log-gate [1,H,L,Dk] (so α=exp(g) ∈ (0,1)) and
// its head-0 [L][Dk] view.
func (f glaFixture) gate() (*metal.Array, [][]float64) {
	n := int(f.h * f.l * f.d)
	values := make([]float32, n)
	for i := range values {
		// keep gates in a modest negative band so exp(-cumsum) stays well-conditioned.
		values[i] = -0.1 - float32(i%4)*0.05
	}
	arr := metal.FromValues(values, 1, int(f.h), int(f.l), int(f.d))
	head0 := make([][]float64, f.l)
	for t := range int(f.l) {
		row := make([]float64, f.d)
		for d := range int(f.d) {
			row[d] = float64(values[t*int(f.d)+d])
		}
		head0[t] = row
	}
	return arr, head0
}

func evalFloats(t *testing.T, label string, a *metal.Array) []float32 {
	t.Helper()
	if a == nil || !a.Valid() {
		t.Fatalf("%s: invalid array", label)
	}
	if err := metal.Eval(a); err != nil {
		t.Fatalf("%s: Eval: %v", label, err)
	}
	return a.Floats()
}

func approxEqual2D(t *testing.T, label string, got []float32, want [][]float64, dInner int) {
	t.Helper()
	const tol = 2e-3
	for a := range want {
		for b := range want[a] {
			g := float64(got[a*dInner+b])
			w := want[a][b]
			if math.Abs(g-w) > tol {
				t.Fatalf("%s: [%d][%d]: got %g want %g (Δ %g)", label, a, b, g, w, math.Abs(g-w))
			}
		}
	}
}

// TestGla_GatedChunk_Good proves the prefill path: the q̃/k̃ chunked form
// reproduces the per-key-dim-gated recurrence with no prior state.
func TestGla_GatedChunk_Good(t *testing.T) {
	f := glaFixture{h: 2, l: 4, d: 3}
	scale := 0.5

	q, q0 := f.tensor(0.3)
	k, k0 := f.tensor(-0.2)
	v, v0 := f.tensor(0.15)
	g, g0 := f.gate()
	defer metal.Free(q, k, v, g)

	out, newS := GatedChunk(q, k, v, g, nil, float32(scale))
	if out == nil {
		t.Fatal("GatedChunk returned nil output")
	}
	defer metal.Free(out, newS)

	gotOut := evalFloats(t, "out", out)
	wantOut, wantS := glaReference(q0, k0, v0, g0, scale, nil)
	approxEqual2D(t, "out head0", gotOut[:int(f.l)*int(f.d)], wantOut, int(f.d))

	gotS := evalFloats(t, "state", newS)
	approxEqual2D(t, "state head0", gotS[:int(f.d)*int(f.d)], wantS, int(f.d))
}

// TestGla_GatedChunk_Ugly proves the cross-chunk path: feeding a non-zero prior
// state equals continuing the per-key-dim-gated recurrence from that state.
func TestGla_GatedChunk_Ugly(t *testing.T) {
	f := glaFixture{h: 1, l: 3, d: 2}
	scale := 1.0

	q, q0 := f.tensor(0.25)
	k, k0 := f.tensor(-0.3)
	v, v0 := f.tensor(0.4)
	g, g0 := f.gate()
	defer metal.Free(q, k, v, g)

	prevVals := []float32{0.5, -0.25, 0.1, 0.33}
	prev := metal.FromValues(prevVals, 1, 1, int(f.d), int(f.d))
	defer metal.Free(prev)
	prevS := [][]float64{{0.5, -0.25}, {0.1, 0.33}}

	out, newS := GatedChunk(q, k, v, g, prev, float32(scale))
	if out == nil {
		t.Fatal("GatedChunk returned nil output")
	}
	defer metal.Free(out, newS)

	gotOut := evalFloats(t, "out", out)
	wantOut, wantS := glaReference(q0, k0, v0, g0, scale, prevS)
	approxEqual2D(t, "out", gotOut[:int(f.l)*int(f.d)], wantOut, int(f.d))

	gotS := evalFloats(t, "state", newS)
	approxEqual2D(t, "state", gotS[:int(f.d)*int(f.d)], wantS, int(f.d))
}

// TestGla_GatedChunk_Bad proves the kernel refuses malformed input (gate shape
// mismatch, wrong rank) by returning nil rather than miscomputing.
func TestGla_GatedChunk_Bad(t *testing.T) {
	f := glaFixture{h: 2, l: 3, d: 2}
	q, _ := f.tensor(0.1)
	k, _ := f.tensor(0.1)
	v, _ := f.tensor(0.1)
	defer metal.Free(q, k, v)

	// gate with wrong L (2 instead of 3).
	badGate := metal.FromValues(make([]float32, int(f.h*2*f.d)), 1, int(f.h), 2, int(f.d))
	defer metal.Free(badGate)
	if out, st := GatedChunk(q, k, v, badGate, nil, 1); out != nil || st != nil {
		t.Fatal("expected nil result for gate shape mismatch")
	}

	threeD := metal.FromValues([]float32{1, 2, 3, 4}, 1, 2, 2) // rank 3
	defer metal.Free(threeD)
	if out, st := GatedChunk(threeD, threeD, threeD, threeD, nil, 1); out != nil || st != nil {
		t.Fatal("expected nil result for rank-3 input")
	}
}

// longGate builds a STEEP negative log-gate [1,H,L,Dk] whose global cumulative
// sum over L positions drives b well past −88, so exp(−b) on a single global
// cumsum overflows fp32 to +Inf. The chunk-local re-basing must keep each
// window's exp(−b) bounded and still reproduce the recurrence. Returns the
// tensor and its head-0 [L][Dk] view.
func (f glaFixture) longGate() (*metal.Array, [][]float64) {
	n := int(f.h * f.l * f.d)
	values := make([]float32, n)
	for i := range values {
		// ~−0.7 mean: over L=200 the global cumsum reaches ≈−140 ⇒ exp(140)=Inf
		// in fp32 without windowed re-basing.
		values[i] = -0.6 - float32(i%3)*0.1
	}
	arr := metal.FromValues(values, 1, int(f.h), int(f.l), int(f.d))
	head0 := make([][]float64, f.l)
	for t := range int(f.l) {
		row := make([]float64, f.d)
		for d := range int(f.d) {
			row[d] = float64(values[t*int(f.d)+d])
		}
		head0[t] = row
	}
	return arr, head0
}

// TestGla_GatedChunk_LongChunk_Ugly proves the long-chunk hardening: at L well
// past the re-basing window (subChunk=64) with a steep gate that would overflow
// exp(−b) on a single global cumsum, the windowed kernel still matches the exact
// token-by-token recurrence — both output and advanced state. This is the
// stability gate the package doc flags.
func TestGla_GatedChunk_LongChunk_Ugly(t *testing.T) {
	if subChunk != 64 {
		t.Fatalf("test assumes subChunk=64, got %d", subChunk)
	}
	f := glaFixture{h: 1, l: 200, d: 4} // L=200 > 3 windows of 64
	scale := 0.5

	q, q0 := f.tensor(0.2)
	k, k0 := f.tensor(-0.15)
	v, v0 := f.tensor(0.1)
	g, g0 := f.longGate()
	defer metal.Free(q, k, v, g)

	out, newS := GatedChunk(q, k, v, g, nil, float32(scale))
	if out == nil {
		t.Fatal("GatedChunk returned nil output")
	}
	defer metal.Free(out, newS)

	gotOut := evalFloats(t, "out", out)
	// Guard: a +Inf/NaN anywhere means re-basing failed to bound exp(−b).
	for i, x := range gotOut {
		if math.IsInf(float64(x), 0) || math.IsNaN(float64(x)) {
			t.Fatalf("out[%d] is non-finite (%g) — long-chunk re-basing did not bound exp(−b)", i, x)
		}
	}
	wantOut, wantS := glaReference(q0, k0, v0, g0, scale, nil)
	approxEqual2D(t, "out head0", gotOut[:int(f.l)*int(f.d)], wantOut, int(f.d))

	gotS := evalFloats(t, "state", newS)
	approxEqual2D(t, "state head0", gotS[:int(f.d)*int(f.d)], wantS, int(f.d))
}

// TestGla_Mixer_Good proves the family registers and declares the recurrent
// state contract.
func TestGla_Mixer_Good(t *testing.T) {
	m, ok := scheme.MixerFor("gla")
	if !ok {
		t.Fatal("gla not registered in scheme catalogue")
	}
	if m.Kind() != "gla" {
		t.Fatalf("Kind: got %q want gla", m.Kind())
	}
	if m.State() != scheme.StateRecurrent {
		t.Fatalf("State: got %v want StateRecurrent", m.State())
	}
}

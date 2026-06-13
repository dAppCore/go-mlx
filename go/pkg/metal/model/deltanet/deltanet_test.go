// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package deltanet

import (
	"math"
	"testing"

	metal "dappco.re/go/mlx/pkg/metal"
	scheme "dappco.re/go/mlx/pkg/scheme"
)

// deltanet_test.go validates the DeltaRuleChunk kernel against an independent,
// plain-Go reference walking the delta-rule recurrence token by token:
// L2-normalise k, read_t = S_{t-1}ᵀk_t, S_t = S_{t-1} + β_t·k_t(v_t−read_t)ᵀ,
// o_t = q_t·S_t. The kernel composes the same recurrence from metal ops, so the
// reference proves the L2-normalisation, the error-write, and the output read
// are all correct.

// l2norm normalises a vector to unit L2 length (matching the kernel's eps).
func l2norm(x []float64, eps float64) []float64 {
	var ss float64
	for _, v := range x {
		ss += v * v
	}
	inv := 1.0 / math.Sqrt(ss+eps)
	out := make([]float64, len(x))
	for i, v := range x {
		out[i] = v * inv
	}
	return out
}

// deltaReference computes DeltaNet output and final state the slow way for one
// head / one batch. q,k,v are [L][D]; beta is per-token [L]; prevS the incoming
// [Dk][Dv] state (nil ⇒ zero); scale scales q; eps is the key-norm epsilon.
func deltaReference(q, k, v [][]float64, beta []float64, scale, eps float64, prevS [][]float64) (out [][]float64, finalS [][]float64) {
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
		kn := l2norm(k[t], eps)
		// read_t[b] = Σ_a S[a][b]·kn[a].
		read := make([]float64, dv)
		for b := range dv {
			var acc float64
			for a := range dk {
				acc += s[a][b] * kn[a]
			}
			read[b] = acc
		}
		// S_t[a][b] += β_t · kn[a] · (v_t[b] − read[b]).
		for a := range dk {
			for b := range dv {
				s[a][b] += beta[t] * kn[a] * (v[t][b] - read[b])
			}
		}
		// o_t[b] = Σ_a (scale·q_t[a]) · S_t[a][b].
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

type deltaFixture struct {
	h, l, d int32
}

func (f deltaFixture) tensor(seed float32) (*metal.Array, [][]float64) {
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

// beta builds a fixed per-token write strength [1,H,L,1] in (0,1) and its head-0
// [L] view.
func (f deltaFixture) beta() (*metal.Array, []float64) {
	n := int(f.h * f.l)
	values := make([]float32, n)
	for i := range values {
		values[i] = 0.3 + float32(i%3)*0.2 // 0.3, 0.5, 0.7 cycling
	}
	arr := metal.FromValues(values, 1, int(f.h), int(f.l), 1)
	head0 := make([]float64, f.l)
	for t := range int(f.l) {
		head0[t] = float64(values[t])
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

// TestDeltanet_DeltaRuleChunk_Good proves the prefill path: the metal recurrence
// reproduces the token-by-token delta rule with no prior state.
func TestDeltanet_DeltaRuleChunk_Good(t *testing.T) {
	f := deltaFixture{h: 2, l: 4, d: 3}
	scale := 0.5
	eps := float64(defaultNormEps)

	q, q0 := f.tensor(0.3)
	k, k0 := f.tensor(-0.2)
	v, v0 := f.tensor(0.15)
	beta, beta0 := f.beta()
	defer metal.Free(q, k, v, beta)

	out, newS := DeltaRuleChunk(q, k, v, beta, nil, float32(scale), 0)
	if out == nil {
		t.Fatal("DeltaRuleChunk returned nil output")
	}
	defer metal.Free(out, newS)

	gotOut := evalFloats(t, "out", out)
	wantOut, wantS := deltaReference(q0, k0, v0, beta0, scale, eps, nil)
	approxEqual2D(t, "out head0", gotOut[:int(f.l)*int(f.d)], wantOut, int(f.d))

	gotS := evalFloats(t, "state", newS)
	approxEqual2D(t, "state head0", gotS[:int(f.d)*int(f.d)], wantS, int(f.d))
}

// TestDeltanet_DeltaRuleChunk_Ugly proves the cross-chunk path: feeding a
// non-zero prior state equals continuing the delta-rule recurrence from it.
func TestDeltanet_DeltaRuleChunk_Ugly(t *testing.T) {
	f := deltaFixture{h: 1, l: 3, d: 2}
	scale := 1.0
	eps := float64(defaultNormEps)

	q, q0 := f.tensor(0.25)
	k, k0 := f.tensor(-0.3)
	v, v0 := f.tensor(0.4)
	beta, beta0 := f.beta()
	defer metal.Free(q, k, v, beta)

	prevVals := []float32{0.5, -0.25, 0.1, 0.33}
	prev := metal.FromValues(prevVals, 1, 1, int(f.d), int(f.d))
	defer metal.Free(prev)
	prevS := [][]float64{{0.5, -0.25}, {0.1, 0.33}}

	out, newS := DeltaRuleChunk(q, k, v, beta, prev, float32(scale), 0)
	if out == nil {
		t.Fatal("DeltaRuleChunk returned nil output")
	}
	defer metal.Free(out, newS)

	gotOut := evalFloats(t, "out", out)
	wantOut, wantS := deltaReference(q0, k0, v0, beta0, scale, eps, prevS)
	approxEqual2D(t, "out", gotOut[:int(f.l)*int(f.d)], wantOut, int(f.d))

	gotS := evalFloats(t, "state", newS)
	approxEqual2D(t, "state", gotS[:int(f.d)*int(f.d)], wantS, int(f.d))
}

// TestDeltanet_DeltaRuleChunk_Bad proves the kernel refuses malformed input
// (beta shape mismatch, wrong rank) by returning nil.
func TestDeltanet_DeltaRuleChunk_Bad(t *testing.T) {
	f := deltaFixture{h: 2, l: 3, d: 2}
	q, _ := f.tensor(0.1)
	k, _ := f.tensor(0.1)
	v, _ := f.tensor(0.1)
	defer metal.Free(q, k, v)

	// beta with a non-1 last dim.
	badBeta := metal.FromValues(make([]float32, int(f.h*f.l*2)), 1, int(f.h), int(f.l), 2)
	defer metal.Free(badBeta)
	if out, st := DeltaRuleChunk(q, k, v, badBeta, nil, 1, 0); out != nil || st != nil {
		t.Fatal("expected nil result for beta shape mismatch")
	}

	threeD := metal.FromValues([]float32{1, 2, 3, 4}, 1, 2, 2) // rank 3
	defer metal.Free(threeD)
	if out, st := DeltaRuleChunk(threeD, threeD, threeD, threeD, nil, 1, 0); out != nil || st != nil {
		t.Fatal("expected nil result for rank-3 input")
	}
}

// TestDeltanet_Mixer_Good proves the family registers and declares the recurrent
// state contract.
func TestDeltanet_Mixer_Good(t *testing.T) {
	m, ok := scheme.MixerFor("deltanet")
	if !ok {
		t.Fatal("deltanet not registered in scheme catalogue")
	}
	if m.Kind() != "deltanet" {
		t.Fatalf("Kind: got %q want deltanet", m.Kind())
	}
	if m.State() != scheme.StateRecurrent {
		t.Fatalf("State: got %v want StateRecurrent", m.State())
	}
}

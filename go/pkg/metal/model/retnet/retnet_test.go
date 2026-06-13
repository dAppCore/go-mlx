// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package retnet

import (
	"math"
	"testing"

	metal "dappco.re/go/mlx/pkg/metal"
	scheme "dappco.re/go/mlx/pkg/scheme"
)

// retnet_test.go validates the parallel-form RetentionChunk kernel against an
// independent, plain-Go recurrent reference. The reference walks the recurrence
// token by token (S_t = γ·S_{t-1} + k_tᵀv_t; o_t = q_t·S_t), which is the
// definition the chunked matmul form must reproduce — proving the decay mask,
// the cross-chunk read-out, and the advanced state are all correct.

// retentionReference computes RetNet output and final state the slow, obvious
// way: one head, one batch, looping over timesteps. q,k,v are [L][D]; gamma the
// scalar decay; prevS the incoming [Dk][Dv] state (nil ⇒ zero). Returns the
// per-token output [L][Dv] and the final state [Dk][Dv].
func retentionReference(q, k, v [][]float64, gamma float64, scale float64, prevS [][]float64) (out [][]float64, finalS [][]float64) {
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
		// S_t = γ·S_{t-1} + k_tᵀ v_t (outer product accumulated).
		for a := range dk {
			for b := range dv {
				s[a][b] = gamma*s[a][b] + k[t][a]*v[t][b]
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

// retnetFixture builds a small deterministic [1,H,L,D] tensor with a smooth
// value pattern, plus the matching [][] view for one head for the reference.
type retnetFixture struct {
	h, l, d int32
}

func (f retnetFixture) tensor(seed float32) (*metal.Array, [][]float64) {
	n := int(f.h * f.l * f.d)
	values := make([]float32, n)
	for i := range values {
		values[i] = seed + float32(i%11)*0.1 - float32(i%5)*0.07
	}
	arr := metal.FromValues(values, 1, int(f.h), int(f.l), int(f.d))
	// head 0 view as [L][D] for the reference.
	head0 := make([][]float64, f.l)
	for t := range int(f.l) {
		row := make([]float64, f.d)
		for d := range int(f.d) {
			row[d] = float64(values[t*int(f.d)+d]) // head 0 occupies the first L*D block
		}
		head0[t] = row
	}
	return arr, head0
}

// evalFloats materialises an array and reads it back to the host as float32 —
// the model-package equivalent of metal's internal test helper, built on the
// exported Eval + Floats surface.
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

func approxEqual(t *testing.T, label string, got []float32, want [][]float64, dv int) {
	t.Helper()
	const tol = 2e-3
	for tok := range want {
		for b := range want[tok] {
			g := float64(got[tok*dv+b])
			w := want[tok][b]
			if math.Abs(g-w) > tol {
				t.Fatalf("%s: token %d dim %d: got %g want %g (Δ %g)", label, tok, b, g, w, math.Abs(g-w))
			}
		}
	}
}

// TestRetnet_RetentionChunk_Good proves the prefill path (no prior state): the
// parallel decay-masked matmul reproduces the token-by-token recurrence.
func TestRetnet_RetentionChunk_Good(t *testing.T) {
	f := retnetFixture{h: 2, l: 4, d: 3}
	gamma := 0.9
	scale := 0.5
	lnGamma := []float32{float32(math.Log(gamma)), float32(math.Log(gamma))}

	q, q0 := f.tensor(0.3)
	k, k0 := f.tensor(-0.2)
	v, v0 := f.tensor(0.15)
	defer metal.Free(q, k, v)

	out, newS := RetentionChunk(q, k, v, nil, lnGamma, float32(scale))
	if out == nil {
		t.Fatal("RetentionChunk returned nil output")
	}
	defer metal.Free(out, newS)

	gotOut := evalFloats(t, "out", out) // [1,H,L,Dv] flattened
	wantOut, wantS := retentionReference(q0, k0, v0, gamma, scale, nil)

	// head 0 occupies the first L*Dv block of the flattened output.
	approxEqual(t, "out head0", gotOut[:int(f.l)*int(f.d)], wantOut, int(f.d))

	// advanced state head 0: first Dk*Dv block of [1,H,Dk,Dv].
	gotS := evalFloats(t, "state", newS)
	approxEqualState(t, "state head0", gotS[:int(f.d)*int(f.d)], wantS, int(f.d))
}

// TestRetnet_RetentionChunk_Ugly proves the cross-chunk path: feeding a chunk a
// non-zero prior state must equal continuing the recurrence from that state.
func TestRetnet_RetentionChunk_Ugly(t *testing.T) {
	f := retnetFixture{h: 1, l: 3, d: 2}
	gamma := 0.8
	scale := 1.0
	lnGamma := []float32{float32(math.Log(gamma))}

	q, q0 := f.tensor(0.25)
	k, k0 := f.tensor(-0.3)
	v, v0 := f.tensor(0.4)
	defer metal.Free(q, k, v)

	// A fixed non-zero incoming state [1,1,Dk,Dv].
	prevVals := []float32{0.5, -0.25, 0.1, 0.33}
	prev := metal.FromValues(prevVals, 1, 1, int(f.d), int(f.d))
	defer metal.Free(prev)
	prevS := [][]float64{{0.5, -0.25}, {0.1, 0.33}}

	out, newS := RetentionChunk(q, k, v, prev, lnGamma, float32(scale))
	if out == nil {
		t.Fatal("RetentionChunk returned nil output")
	}
	defer metal.Free(out, newS)

	gotOut := evalFloats(t, "out", out)
	wantOut, wantS := retentionReference(q0, k0, v0, gamma, scale, prevS)
	approxEqual(t, "out", gotOut[:int(f.l)*int(f.d)], wantOut, int(f.d))

	gotS := evalFloats(t, "state", newS)
	approxEqualState(t, "state", gotS[:int(f.d)*int(f.d)], wantS, int(f.d))
}

// TestRetnet_RetentionChunk_Bad proves the kernel refuses malformed input
// (rank mismatch, head/decay-length mismatch) by returning nil rather than
// miscomputing.
func TestRetnet_RetentionChunk_Bad(t *testing.T) {
	threeD := metal.FromValues([]float32{1, 2, 3, 4}, 1, 2, 2) // rank 3
	defer metal.Free(threeD)
	if out, st := RetentionChunk(threeD, threeD, threeD, nil, []float32{0}, 1); out != nil || st != nil {
		t.Fatal("expected nil result for rank-3 input")
	}

	f := retnetFixture{h: 2, l: 2, d: 2}
	q, _ := f.tensor(0.1)
	k, _ := f.tensor(0.1)
	v, _ := f.tensor(0.1)
	defer metal.Free(q, k, v)
	// decayLn length 1 but H == 2.
	if out, st := RetentionChunk(q, k, v, nil, []float32{0}, 1); out != nil || st != nil {
		t.Fatal("expected nil result for head/decay-length mismatch")
	}
}

// TestRetnet_Mixer_Good proves the family registers and declares the recurrent
// state contract, and that the scaffold compile-proof holds.
func TestRetnet_Mixer_Good(t *testing.T) {
	m, ok := scheme.MixerFor("retnet")
	if !ok {
		t.Fatal("retnet not registered in scheme catalogue")
	}
	if m.Kind() != "retnet" {
		t.Fatalf("Kind: got %q want retnet", m.Kind())
	}
	if m.State() != scheme.StateRecurrent {
		t.Fatalf("State: got %v want StateRecurrent", m.State())
	}
}

func approxEqualState(t *testing.T, label string, got []float32, want [][]float64, dv int) {
	t.Helper()
	const tol = 2e-3
	for a := range want {
		for b := range want[a] {
			g := float64(got[a*dv+b])
			w := want[a][b]
			if math.Abs(g-w) > tol {
				t.Fatalf("%s: [%d][%d]: got %g want %g (Δ %g)", label, a, b, g, w, math.Abs(g-w))
			}
		}
	}
}

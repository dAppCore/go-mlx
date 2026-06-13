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

// stepSlice extracts time-step t of a [1,H,L,X] tensor as [1,H,1,X].
func stepSlice(a *metal.Array, t int32) *metal.Array {
	h := int32(a.Dim(1))
	x := int32(a.Dim(3))
	return metal.Slice4(a, 0, 0, t, 0, 1, h, t+1, x)
}

// TestRetnet_DecodeStep_Ugly proves the L==1 decode fast path equals the chunk
// path: running the whole sequence token-by-token (each call L=1, threading the
// returned state forward by hand) must reproduce the single full-chunk output
// position for position. This is the decode invariant — chunked == sequential.
func TestRetnet_DecodeStep_Ugly(t *testing.T) {
	f := retnetFixture{h: 2, l: 4, d: 3}
	lnGamma := []float32{float32(math.Log(0.9)), float32(math.Log(0.85))}
	scale := float32(0.5)

	q, _ := f.tensor(0.3)
	k, _ := f.tensor(-0.2)
	v, _ := f.tensor(0.15)
	defer metal.Free(q, k, v)

	// Reference: one full-chunk call.
	chunkOut, chunkState := RetentionChunk(q, k, v, nil, lnGamma, scale)
	defer metal.Free(chunkOut, chunkState)
	wantOut := evalFloats(t, "chunk out", chunkOut)
	wantState := evalFloats(t, "chunk state", chunkState)

	// Token-by-token: each step L=1, carry the state forward.
	var state *metal.Array
	gotOut := make([]float32, 0, len(wantOut))
	for tok := int32(0); tok < f.l; tok++ {
		qt := stepSlice(q, tok)
		kt := stepSlice(k, tok)
		vt := stepSlice(v, tok)
		o, next := RetentionChunk(qt, kt, vt, state, lnGamma, scale)
		metal.Free(qt, kt, vt)
		if state != nil {
			metal.Free(state)
		}
		state = next
		gotOut = append(gotOut, evalFloats(t, "step out", o)...)
		metal.Free(o)
	}
	defer metal.Free(state)

	// The per-token outputs, concatenated over L, must match the chunk output.
	// chunk output is [1,H,L,Dv] (head-major); the step outputs are [1,H,1,Dv]
	// per token, so reassemble into head-major order before comparing.
	const tol = 2e-3
	dv := int(f.d)
	for head := 0; head < int(f.h); head++ {
		for tok := 0; tok < int(f.l); tok++ {
			for b := 0; b < dv; b++ {
				want := wantOut[head*int(f.l)*dv+tok*dv+b]
				got := gotOut[tok*int(f.h)*dv+head*dv+b]
				if math.Abs(float64(want-got)) > tol {
					t.Fatalf("decode≠chunk head %d tok %d dim %d: chunk %g step %g", head, tok, b, want, got)
				}
			}
		}
	}

	gotState := evalFloats(t, "step state", state)
	for i := range wantState {
		if math.Abs(float64(wantState[i]-gotState[i])) > tol {
			t.Fatalf("final state mismatch at %d: chunk %g step %g", i, wantState[i], gotState[i])
		}
	}
}

// TestRetnet_Mixer_StateThreading_Good proves Mixer.Forward threads state
// through a metal.RecurrentCache: two sequential single-token forwards with the
// same holder must equal a two-token chunk forward through one holder. Drives
// Forward with a real holder (no model load), per the Phase-2 gate.
func TestRetnet_Mixer_StateThreading_Good(t *testing.T) {
	const (
		h, d   = 2, 4
		dModel = h * d
	)
	w := &Weights{
		QProj:    identityLinear(t, dModel),
		KProj:    identityLinear(t, dModel),
		VProj:    identityLinear(t, dModel),
		Output:   identityLinear(t, dModel),
		NumHeads: h,
		HeadDim:  d,
		Scale:    0.5,
		DecayLn:  []float32{float32(math.Log(0.9)), float32(math.Log(0.85))},
	}
	m := &Mixer{W: w}

	// Two-token input [1,2,dModel].
	x2 := tokenInput(2, dModel, 0.2)
	defer metal.Free(x2)

	// Path A: one 2-token forward through a holder.
	cacheA := metal.NewRecurrentCache()
	outA, _ := m.Forward(x2, &metal.MixerCtx{Cache: cacheA, B: 1, L: 2})
	defer metal.Free(outA)
	wantOut := evalFloats(t, "chunk forward", outA)

	// Path B: token 0 then token 1, same holder, state carried by Forward itself.
	cacheB := metal.NewRecurrentCache()
	x0 := metal.Slice(x2, []int32{0, 0, 0}, []int32{1, 1, dModel})
	x1 := metal.Slice(x2, []int32{0, 1, 0}, []int32{1, 2, dModel})
	defer metal.Free(x0, x1)
	out0, _ := m.Forward(x0, &metal.MixerCtx{Cache: cacheB, B: 1, L: 1})
	out1, _ := m.Forward(x1, &metal.MixerCtx{Cache: cacheB, B: 1, L: 1})
	defer metal.Free(out0, out1)
	got0 := evalFloats(t, "step0", out0)
	got1 := evalFloats(t, "step1", out1)

	const tol = 2e-3
	// token 0 output (first dModel of the chunk) == out0.
	for i := 0; i < dModel; i++ {
		if math.Abs(float64(wantOut[i]-got0[i])) > tol {
			t.Fatalf("token0 mismatch at %d: chunk %g step %g", i, wantOut[i], got0[i])
		}
	}
	// token 1 output (second dModel of the chunk) == out1.
	for i := 0; i < dModel; i++ {
		if math.Abs(float64(wantOut[dModel+i]-got1[i])) > tol {
			t.Fatalf("token1 mismatch at %d: chunk %g step %g", i, wantOut[dModel+i], got1[i])
		}
	}
}

// identityLinear builds an n×n identity-weight Linear so projections pass the
// hidden state through unchanged — isolates the recurrence math from the
// projection weights in the state-threading test. MLX matmul is x @ Wᵀ, and the
// identity is its own transpose.
func identityLinear(t *testing.T, n int) *metal.Linear {
	t.Helper()
	vals := make([]float32, n*n)
	for i := 0; i < n; i++ {
		vals[i*n+i] = 1
	}
	return metal.NewLinear(metal.FromValues(vals, n, n), nil)
}

// tokenInput builds a deterministic [1,L,D] hidden-state tensor.
func tokenInput(l, d int, seed float32) *metal.Array {
	vals := make([]float32, l*d)
	for i := range vals {
		vals[i] = seed + float32(i%9)*0.1 - float32(i%4)*0.05
	}
	return metal.FromValues(vals, 1, l, d)
}

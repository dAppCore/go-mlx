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

// stepSlice extracts time-step t of a [1,H,L,X] tensor as [1,H,1,X].
func stepSlice(a *metal.Array, t int32) *metal.Array {
	h := int32(a.Dim(1))
	x := int32(a.Dim(3))
	return metal.Slice4(a, 0, 0, t, 0, 1, h, t+1, x)
}

// TestGla_DecodeStep_Ugly proves the L==1 decode fast path equals the chunk
// path: running the sequence token-by-token (each call L=1, carrying the
// returned state by hand) reproduces the single full-chunk output position for
// position. This is the decode invariant — chunked == sequential — and it
// exercises decodeStep against the cumsum/window machinery.
func TestGla_DecodeStep_Ugly(t *testing.T) {
	f := glaFixture{h: 2, l: 4, d: 3}
	scale := float32(0.5)

	q, _ := f.tensor(0.3)
	k, _ := f.tensor(-0.2)
	v, _ := f.tensor(0.15)
	g, _ := f.gate()
	defer metal.Free(q, k, v, g)

	chunkOut, chunkState := GatedChunk(q, k, v, g, nil, scale)
	defer metal.Free(chunkOut, chunkState)
	wantOut := evalFloats(t, "chunk out", chunkOut)
	wantState := evalFloats(t, "chunk state", chunkState)

	var state *metal.Array
	gotOut := make([]float32, 0, len(wantOut))
	for tok := int32(0); tok < f.l; tok++ {
		qt := stepSlice(q, tok)
		kt := stepSlice(k, tok)
		vt := stepSlice(v, tok)
		gt := stepSlice(g, tok)
		o, next := GatedChunk(qt, kt, vt, gt, state, scale)
		metal.Free(qt, kt, vt, gt)
		if state != nil {
			metal.Free(state)
		}
		state = next
		gotOut = append(gotOut, evalFloats(t, "step out", o)...)
		metal.Free(o)
	}
	defer metal.Free(state)

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

// TestGla_Mixer_StateThreading_Good proves Mixer.Forward threads state through a
// metal.RecurrentCache: two sequential single-token forwards with the same
// holder equal a two-token chunk forward through one holder. Drives Forward with
// a real holder and a fixed position-independent gate (no model load).
func TestGla_Mixer_StateThreading_Good(t *testing.T) {
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
	}
	// Position-independent gate: a constant log-gate per call, so a 2-token chunk
	// and two 1-token steps see identical gates — the comparison stays exact.
	gate := func(x *metal.Array, b, l int32) *metal.Array {
		vals := make([]float32, int(b)*h*int(l)*d)
		for i := range vals {
			vals[i] = -0.1 - float32(i%d)*0.05
		}
		return metal.FromValues(vals, int(b), h, int(l), d)
	}
	m := &Mixer{W: w, Gate: gate}

	x2 := tokenInput(2, dModel, 0.2)
	defer metal.Free(x2)

	cacheA := metal.NewRecurrentCache()
	outA, _ := m.Forward(x2, &metal.MixerCtx{Cache: cacheA, B: 1, L: 2})
	defer metal.Free(outA)
	wantOut := evalFloats(t, "chunk forward", outA)

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
	for i := 0; i < dModel; i++ {
		if math.Abs(float64(wantOut[i]-got0[i])) > tol {
			t.Fatalf("token0 mismatch at %d: chunk %g step %g", i, wantOut[i], got0[i])
		}
		if math.Abs(float64(wantOut[dModel+i]-got1[i])) > tol {
			t.Fatalf("token1 mismatch at %d: chunk %g step %g", i, wantOut[dModel+i], got1[i])
		}
	}
}

// TestGla_Mixer_HeadLayout_Good pins the per-head split/merge layout of Forward
// numerically. The recurrence oracle tests feed [B,H,L,D] straight into
// GatedChunk and only check head 0, so they bypass projectHeads (the
// [B,L,H*D]→[B,H,L,D] strided view) and mergeHeads (the inverse transpose); the
// state-threading test drives those paths but with identity weights and the same
// routing on both sides, so a static head-misroute cancels and stays green. This
// test gives each head DISTINCT input channels and a DISTINCT gate, then builds
// the expected output with an INDEPENDENT Go oracle that splits the model
// dimension block-contiguously (head h ⇒ channels [h·D, h·D+D)), runs the gated
// recurrence per head, and re-assembles block-contiguously (head h's read-out ⇒
// output channels [h·Dv, h·Dv+Dv)). Identity Q/K/V/Output projections keep the
// projection a pass-through so only the head routing is under test. If a head is
// swapped, fed the wrong channels, or written to the wrong output block, the
// per-head outputs differ enough that the element-wise compare fails.
func TestGla_Mixer_HeadLayout_Good(t *testing.T) {
	const (
		h, d   = 2, 2
		l      = 3
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
	}
	// Per-head-distinct, deliberately asymmetric input: head 0 occupies the first
	// d channels, head 1 the next d, with a different value band each so swapping
	// heads changes the result. xHead[head][t][channel].
	xHead := [][][]float64{
		{{0.4, -0.3}, {0.2, 0.5}, {-0.1, 0.35}},  // head 0
		{{-0.6, 0.7}, {0.45, -0.2}, {0.3, -0.5}}, // head 1
	}
	// Per-head-distinct log-gate (negative ⇒ α∈(0,1)). gHead[head][t][k].
	gHead := [][][]float64{
		{{-0.1, -0.25}, {-0.3, -0.05}, {-0.15, -0.2}}, // head 0
		{{-0.4, -0.1}, {-0.05, -0.35}, {-0.2, -0.3}},  // head 1
	}

	// Flatten heads block-contiguously into the [1,L,dModel] model input — exactly
	// the layout projectHeads' strided view reads back as [1,H,L,D].
	xVals := make([]float32, l*dModel)
	for head := 0; head < h; head++ {
		for tok := 0; tok < l; tok++ {
			for c := 0; c < d; c++ {
				xVals[tok*dModel+head*d+c] = float32(xHead[head][tok][c])
			}
		}
	}
	x := metal.FromValues(xVals, 1, l, dModel)
	defer metal.Free(x)

	// Gate provider mirrors the same block-contiguous head layout into [1,H,L,Dk].
	gate := func(_ *metal.Array, b, ll int32) *metal.Array {
		vals := make([]float32, int(b)*h*int(ll)*d)
		for head := 0; head < h; head++ {
			for tok := 0; tok < int(ll); tok++ {
				for k := 0; k < d; k++ {
					vals[head*int(ll)*d+tok*d+k] = float32(gHead[head][tok][k])
				}
			}
		}
		return metal.FromValues(vals, int(b), h, int(ll), d)
	}
	m := &Mixer{W: w, Gate: gate}

	out, _ := m.Forward(x, &metal.MixerCtx{B: 1, L: l})
	if out == nil {
		t.Fatal("Forward returned nil")
	}
	defer metal.Free(out)
	got := evalFloats(t, "forward", out) // [1,L,dModel] flattened

	// Independent oracle: run the gated recurrence per head on its own channel
	// block, place each head's read-out in its own output block.
	const tol = 2e-3
	scale := float64(w.Scale)
	for head := 0; head < h; head++ {
		wantOut, _ := glaReference(xHead[head], xHead[head], xHead[head], gHead[head], scale, nil)
		for tok := 0; tok < l; tok++ {
			for dv := 0; dv < d; dv++ {
				g := float64(got[tok*dModel+head*d+dv])
				want := wantOut[tok][dv]
				if math.Abs(g-want) > tol {
					t.Fatalf("head %d tok %d dim %d: got %g want %g (Δ %g) — head split/merge layout drifted",
						head, tok, dv, g, want, math.Abs(g-want))
				}
			}
		}
	}
}

// identityLinear builds an n×n identity-weight Linear so projections pass the
// hidden state through unchanged — isolates the recurrence from the projection
// weights in the state-threading test. MLX matmul is x @ Wᵀ; identity is its
// own transpose.
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

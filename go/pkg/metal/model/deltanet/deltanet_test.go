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

// approxEqual1D fails when two flat float32 slices differ beyond tolerance —
// used to compare the parallel and sequential delta-rule paths element-wise.
func approxEqual1D(t *testing.T, label string, got, want []float32) {
	t.Helper()
	const tol = 2e-3
	if len(got) != len(want) {
		t.Fatalf("%s: length %d, want %d", label, len(got), len(want))
	}
	for i := range want {
		if d := math.Abs(float64(got[i] - want[i])); d > tol {
			t.Fatalf("%s[%d]: got %g want %g (Δ %g)", label, i, got[i], want[i], d)
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

// longTensor builds a [1,H,L,D] fixture and its head-0 [L][D] view for the
// multi-window chunked tests (L can exceed chunkWidth).
func (f deltaFixture) longTensor(seed float32) (*metal.Array, [][]float64) {
	n := int(f.h * f.l * f.d)
	values := make([]float32, n)
	for i := range values {
		values[i] = seed + float32(i%13)*0.08 - float32(i%7)*0.05
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

// longBeta builds a per-token write strength [1,H,L,1] in (0,1) for the
// multi-window tests, plus its head-0 [L] view.
func (f deltaFixture) longBeta() (*metal.Array, []float64) {
	n := int(f.h * f.l)
	values := make([]float32, n)
	for i := range values {
		values[i] = 0.2 + float32(i%5)*0.15 // 0.2..0.8 cycling
	}
	arr := metal.FromValues(values, 1, int(f.h), int(f.l), 1)
	head0 := make([]float64, f.l)
	for t := range int(f.l) {
		head0[t] = float64(values[t])
	}
	return arr, head0
}

// TestDeltanet_ChunkedMatchesSequential_Good is the gate criterion: the
// chunked-PARALLEL WY form (DeltaRuleChunk) equals the EXACT sequential
// recurrence (DeltaRuleChunkSequential) within a single window, both output and
// advanced state. The WY solve is exact, so they must agree to tolerance.
func TestDeltanet_ChunkedMatchesSequential_Good(t *testing.T) {
	f := deltaFixture{h: 2, l: 12, d: 4} // L within one chunkWidth window
	scale := 0.5

	q, _ := f.longTensor(0.3)
	k, _ := f.longTensor(-0.2)
	v, _ := f.longTensor(0.15)
	beta, _ := f.longBeta()
	defer metal.Free(q, k, v, beta)

	par, parS := DeltaRuleChunk(q, k, v, beta, nil, float32(scale), 0)
	if par == nil {
		t.Fatal("parallel DeltaRuleChunk returned nil")
	}
	defer metal.Free(par, parS)
	seq, seqS := DeltaRuleChunkSequential(q, k, v, beta, nil, float32(scale), 0)
	if seq == nil {
		t.Fatal("sequential DeltaRuleChunkSequential returned nil")
	}
	defer metal.Free(seq, seqS)

	gotPar := evalFloats(t, "parallel out", par)
	gotSeq := evalFloats(t, "sequential out", seq)
	approxEqual1D(t, "out", gotPar, gotSeq)

	gotParS := evalFloats(t, "parallel state", parS)
	gotSeqS := evalFloats(t, "sequential state", seqS)
	approxEqual1D(t, "state", gotParS, gotSeqS)
}

// TestDeltanet_ChunkedMultiWindow_Ugly proves the windowing: at L well past
// chunkWidth (so the parallel path threads the recurrent state across several
// ≤64 windows), the chunked output still equals the sequential recurrence end
// to end. This is the cross-window state-threading gate.
func TestDeltanet_ChunkedMultiWindow_Ugly(t *testing.T) {
	if chunkWidth != 64 {
		t.Fatalf("test assumes chunkWidth=64, got %d", chunkWidth)
	}
	f := deltaFixture{h: 1, l: 150, d: 4} // 150 = 64 + 64 + 22 → three windows
	scale := 0.7

	q, _ := f.longTensor(0.2)
	k, _ := f.longTensor(-0.15)
	v, _ := f.longTensor(0.1)
	beta, _ := f.longBeta()
	defer metal.Free(q, k, v, beta)

	par, parS := DeltaRuleChunk(q, k, v, beta, nil, float32(scale), 0)
	if par == nil {
		t.Fatal("parallel DeltaRuleChunk returned nil")
	}
	defer metal.Free(par, parS)
	seq, seqS := DeltaRuleChunkSequential(q, k, v, beta, nil, float32(scale), 0)
	if seq == nil {
		t.Fatal("sequential DeltaRuleChunkSequential returned nil")
	}
	defer metal.Free(seq, seqS)

	gotPar := evalFloats(t, "parallel out", par)
	for i, x := range gotPar {
		if math.IsInf(float64(x), 0) || math.IsNaN(float64(x)) {
			t.Fatalf("parallel out[%d] non-finite (%g)", i, x)
		}
	}
	gotSeq := evalFloats(t, "sequential out", seq)
	approxEqual1D(t, "out", gotPar, gotSeq)

	gotParS := evalFloats(t, "parallel state", parS)
	gotSeqS := evalFloats(t, "sequential state", seqS)
	approxEqual1D(t, "state", gotParS, gotSeqS)
}

// TestDeltanet_ChunkedMultiWindow_WithState_Ugly proves the same multi-window
// equivalence when a non-zero prior state is carried in: the first window must
// fold S₀ into both its read-out and its advanced state correctly.
func TestDeltanet_ChunkedMultiWindow_WithState_Ugly(t *testing.T) {
	f := deltaFixture{h: 1, l: 100, d: 4}
	scale := 0.5

	q, _ := f.longTensor(0.25)
	k, _ := f.longTensor(-0.3)
	v, _ := f.longTensor(0.4)
	beta, _ := f.longBeta()
	defer metal.Free(q, k, v, beta)

	prevVals := make([]float32, int(f.d*f.d))
	for i := range prevVals {
		prevVals[i] = 0.1 - float32(i%5)*0.04
	}
	prev := metal.FromValues(prevVals, 1, 1, int(f.d), int(f.d))
	defer metal.Free(prev)
	prev2 := metal.FromValues(prevVals, 1, 1, int(f.d), int(f.d)) // independent copy per path
	defer metal.Free(prev2)

	par, parS := DeltaRuleChunk(q, k, v, beta, prev, float32(scale), 0)
	if par == nil {
		t.Fatal("parallel DeltaRuleChunk returned nil")
	}
	defer metal.Free(par, parS)
	seq, seqS := DeltaRuleChunkSequential(q, k, v, beta, prev2, float32(scale), 0)
	if seq == nil {
		t.Fatal("sequential DeltaRuleChunkSequential returned nil")
	}
	defer metal.Free(seq, seqS)

	approxEqual1D(t, "out", evalFloats(t, "parallel out", par), evalFloats(t, "sequential out", seq))
	approxEqual1D(t, "state", evalFloats(t, "parallel state", parS), evalFloats(t, "sequential state", seqS))
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

// TestDeltanet_Mixer_StateThreading_Good proves Mixer.Forward threads state
// through a metal.RecurrentCache: two sequential single-token forwards with the
// same holder equal a two-token chunk forward through one holder. Drives Forward
// with a real holder and a fixed position-independent β (no model load). β being
// position-independent keeps the chunk and the per-token runs identical.
func TestDeltanet_Mixer_StateThreading_Good(t *testing.T) {
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
	beta := func(x *metal.Array, b, l int32) *metal.Array {
		vals := make([]float32, int(b)*h*int(l))
		for i := range vals {
			vals[i] = 0.4 // constant write strength per token
		}
		return metal.FromValues(vals, int(b), h, int(l), 1)
	}
	m := &Mixer{W: w, Beta: beta}

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

// TestDeltanet_Mixer_HeadLayout_Good pins the per-head split/merge layout of
// Forward numerically. The DeltaRuleChunk oracle tests feed [B,H,L,D] straight
// in and only assert head 0, bypassing projectHeads (the [B,L,H*D]→[B,H,L,D]
// strided view) and mergeHeads; the state-threading test drives those paths but
// with identity weights and identical routing on both sides, so a static
// head-misroute cancels. This test gives each head DISTINCT input channels and a
// DISTINCT per-token write strength β, then builds the expected output with an
// INDEPENDENT Go oracle (deltaReference, which itself L2-normalises keys exactly
// as the kernel does) that splits the model dimension block-contiguously (head h
// ⇒ channels [h·D, h·D+D)), runs the delta-rule recurrence per head, and
// re-assembles block-contiguously. A head swap or wrong output block fails the
// compare. The keys are L2-normalised over each head's own D-block, so an
// interleaved split feeds a head a key vector built from the wrong channels —
// changing its unit direction and hence the read — which the oracle catches.
func TestDeltanet_Mixer_HeadLayout_Good(t *testing.T) {
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
	// Per-head-distinct, asymmetric input: head 0 the first d channels, head 1 the
	// next d, different value bands so swapping heads changes the result.
	xHead := [][][]float64{
		{{0.4, -0.3}, {0.2, 0.5}, {-0.1, 0.35}},  // head 0
		{{-0.6, 0.7}, {0.45, -0.2}, {0.3, -0.5}}, // head 1
	}
	// Per-head-distinct write strength β ∈ (0,1). betaHead[head][t].
	betaHead := [][]float64{
		{0.3, 0.5, 0.7}, // head 0
		{0.6, 0.2, 0.8}, // head 1
	}
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

	// β provider mirrors the same block-contiguous head layout into [1,H,L,1].
	beta := func(_ *metal.Array, b, ll int32) *metal.Array {
		vals := make([]float32, int(b)*h*int(ll))
		for head := 0; head < h; head++ {
			for tok := 0; tok < int(ll); tok++ {
				vals[head*int(ll)+tok] = float32(betaHead[head][tok])
			}
		}
		return metal.FromValues(vals, int(b), h, int(ll), 1)
	}
	m := &Mixer{W: w, Beta: beta}

	out, _ := m.Forward(x, &metal.MixerCtx{B: 1, L: l})
	if out == nil {
		t.Fatal("Forward returned nil")
	}
	defer metal.Free(out)
	got := evalFloats(t, "forward", out) // [1,L,dModel] flattened

	const tol = 2e-3
	scale := float64(w.Scale)
	eps := float64(defaultNormEps)
	for head := 0; head < h; head++ {
		wantOut, _ := deltaReference(xHead[head], xHead[head], xHead[head], betaHead[head], scale, eps, nil)
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
// weights. MLX matmul is x @ Wᵀ; identity is its own transpose. DeltaNet
// L2-normalises keys inside the kernel, so an identity K projection still yields
// unit-norm keys — the threading invariant holds regardless.
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

// gatedDeltaReference is deltaReference with the gated-delta decay: the state is
// scaled by the per-token α_t before each step's read/write. alpha is per-token
// [L]. The Qwen3.5/3.6 linear-attention recurrence.
func gatedDeltaReference(q, k, v [][]float64, beta, alpha []float64, scale, eps float64, prevS [][]float64) (out [][]float64, finalS [][]float64) {
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
		// gated decay: S ← α_t·S before the read/write.
		for a := range dk {
			for b := range dv {
				s[a][b] *= alpha[t]
			}
		}
		kn := l2norm(k[t], eps)
		read := make([]float64, dv)
		for b := range dv {
			var acc float64
			for a := range dk {
				acc += s[a][b] * kn[a]
			}
			read[b] = acc
		}
		for a := range dk {
			for b := range dv {
				s[a][b] += beta[t] * kn[a] * (v[t][b] - read[b])
			}
		}
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

// alpha builds a fixed per-token state decay [1,H,L,1] in (0,1) and its head-0
// [L] view (0.85, 0.90, 0.95 cycling — a real gated-delta decay band).
func (f deltaFixture) alpha() (*metal.Array, []float64) {
	n := int(f.h * f.l)
	values := make([]float32, n)
	for i := range values {
		values[i] = 0.85 + float32(i%3)*0.05
	}
	arr := metal.FromValues(values, 1, int(f.h), int(f.l), 1)
	head0 := make([]float64, f.l)
	for t := range int(f.l) {
		head0[t] = float64(values[t])
	}
	return arr, head0
}

// TestDeltanet_GatedDeltaRule_AlphaOne_Good proves α ≡ 1 recovers the plain delta
// rule exactly: the gated recurrence with an all-ones decay matches the ungated
// reference, output and final state.
func TestDeltanet_GatedDeltaRule_AlphaOne_Good(t *testing.T) {
	f := deltaFixture{h: 2, l: 4, d: 3}
	scale := 0.5
	eps := float64(defaultNormEps)

	q, q0 := f.tensor(0.3)
	k, k0 := f.tensor(-0.2)
	v, v0 := f.tensor(0.15)
	beta, beta0 := f.beta()
	ones := make([]float32, int(f.h*f.l))
	for i := range ones {
		ones[i] = 1
	}
	alpha := metal.FromValues(ones, 1, int(f.h), int(f.l), 1)
	defer metal.Free(q, k, v, beta, alpha)

	out, newS := GatedDeltaRuleChunkSequential(q, k, v, beta, alpha, nil, float32(scale), 0)
	if out == nil {
		t.Fatal("GatedDeltaRuleChunkSequential returned nil output")
	}
	defer metal.Free(out, newS)

	gotOut := evalFloats(t, "out", out)
	wantOut, wantS := deltaReference(q0, k0, v0, beta0, scale, eps, nil) // ungated
	approxEqual2D(t, "out head0", gotOut[:int(f.l)*int(f.d)], wantOut, int(f.d))
	gotS := evalFloats(t, "state", newS)
	approxEqual2D(t, "state head0", gotS[:int(f.d)*int(f.d)], wantS, int(f.d))
}

// TestDeltanet_GatedDeltaRule_Good proves the decay math: a sub-1 per-token α
// matches the gated reference (prefill, no prior state).
func TestDeltanet_GatedDeltaRule_Good(t *testing.T) {
	f := deltaFixture{h: 2, l: 4, d: 3}
	scale := 0.5
	eps := float64(defaultNormEps)

	q, q0 := f.tensor(0.3)
	k, k0 := f.tensor(-0.2)
	v, v0 := f.tensor(0.15)
	beta, beta0 := f.beta()
	alpha, alpha0 := f.alpha()
	defer metal.Free(q, k, v, beta, alpha)

	out, newS := GatedDeltaRuleChunkSequential(q, k, v, beta, alpha, nil, float32(scale), 0)
	if out == nil {
		t.Fatal("GatedDeltaRuleChunkSequential returned nil output")
	}
	defer metal.Free(out, newS)

	gotOut := evalFloats(t, "out", out)
	wantOut, wantS := gatedDeltaReference(q0, k0, v0, beta0, alpha0, scale, eps, nil)
	approxEqual2D(t, "out head0", gotOut[:int(f.l)*int(f.d)], wantOut, int(f.d))
	gotS := evalFloats(t, "state", newS)
	approxEqual2D(t, "state head0", gotS[:int(f.d)*int(f.d)], wantS, int(f.d))
}

// TestDeltanet_GatedDeltaRule_Ugly proves the cross-chunk gated path: a non-zero
// prior state is itself decayed by α_t on the first step, continuing the gated
// recurrence.
func TestDeltanet_GatedDeltaRule_Ugly(t *testing.T) {
	f := deltaFixture{h: 1, l: 3, d: 2}
	scale := 1.0
	eps := float64(defaultNormEps)

	q, q0 := f.tensor(0.25)
	k, k0 := f.tensor(-0.3)
	v, v0 := f.tensor(0.4)
	beta, beta0 := f.beta()
	alpha, alpha0 := f.alpha()
	prevVals := []float32{0.5, -0.25, 0.1, 0.33}
	prev := metal.FromValues(prevVals, 1, 1, int(f.d), int(f.d))
	prevS := [][]float64{{0.5, -0.25}, {0.1, 0.33}}
	defer metal.Free(q, k, v, beta, alpha, prev)

	out, newS := GatedDeltaRuleChunkSequential(q, k, v, beta, alpha, prev, float32(scale), 0)
	if out == nil {
		t.Fatal("GatedDeltaRuleChunkSequential returned nil output")
	}
	defer metal.Free(out, newS)

	gotOut := evalFloats(t, "out", out)
	wantOut, wantS := gatedDeltaReference(q0, k0, v0, beta0, alpha0, scale, eps, prevS)
	approxEqual2D(t, "out", gotOut[:int(f.l)*int(f.d)], wantOut, int(f.d))
	gotS := evalFloats(t, "state", newS)
	approxEqual2D(t, "state", gotS[:int(f.d)*int(f.d)], wantS, int(f.d))
}

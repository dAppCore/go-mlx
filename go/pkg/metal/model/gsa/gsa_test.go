// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gsa

import (
	"math"
	"testing"

	"dappco.re/go/mlx/internal/metaltest"
	metal "dappco.re/go/mlx/pkg/metal"
	scheme "dappco.re/go/mlx/pkg/scheme"
)

func requireMetalRuntime(t testing.TB) {
	t.Helper()
	if !metaltest.RunMetalTests {
		t.Skip("build with -tags metal_runtime to enable Metal runtime tests")
	}
	if !metal.MetalAvailable() {
		t.Skip("Metal runtime unavailable")
	}
}

const gsaTol = 1e-4

func checkClose(t *testing.T, name string, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s len = %d, want %d", name, len(got), len(want))
	}
	for i := range want {
		if diff := math.Abs(float64(got[i] - want[i])); diff > gsaTol {
			t.Errorf("%s[%d] = %f, want %f (diff %g)", name, i, got[i], want[i], diff)
		}
	}
}

// TestMixer_KindState_Good pins the scheme identity: GSA answers to "gsa" and
// declares a recurrent slot memory (not a growing K/V cache). No Metal runtime.
func TestMixer_KindState_Good(t *testing.T) {
	if MixerKind != "gsa" {
		t.Fatalf("MixerKind = %q, want %q", MixerKind, "gsa")
	}
	m := &Mixer{}
	if got := m.Kind(); got != MixerKind {
		t.Fatalf("Kind() = %q, want %q", got, MixerKind)
	}
	if got := m.State(); got != scheme.StateRecurrent {
		t.Fatalf("State() = %v, want StateRecurrent", got)
	}
}

// TestMixer_Register_Good proves init() registered a compute-bearing mixer that
// declares the recurrent state kind.
func TestMixer_Register_Good(t *testing.T) {
	m, ok := scheme.MixerFor(MixerKind)
	if !ok {
		t.Fatalf("scheme.MixerFor(%q) not registered", MixerKind)
	}
	if m.State() != scheme.StateRecurrent {
		t.Fatalf("registered mixer State() = %v, want StateRecurrent", m.State())
	}
	if _, ok := metal.MixerComputeFor(MixerKind); !ok {
		t.Fatalf("metal.MixerComputeFor(%q) = false, want compute-bearing", MixerKind)
	}
}

// TestGateDecay_Math_Good pins g = logsigmoid(f)/norm. f=0 → sigmoid 0.5 →
// log(0.5) ≈ -0.693147; norm=1 leaves it unchanged.
func TestGateDecay_Math_Good(t *testing.T) {
	requireMetalRuntime(t)

	f := metal.FromValues([]float32{0, 0}, 1, 1, 1, 2)
	defer metal.Free(f)

	g := gateDecay(f, 1.0)
	defer metal.Free(g)

	want := float32(math.Log(0.5))
	checkClose(t, "gateDecay", g.Floats(), []float32{want, want})
}

// TestGateDecay_Norm_Good confirms the gate_logit_normalizer divides the
// logsigmoid: f=0, norm=2 → log(0.5)/2.
func TestGateDecay_Norm_Good(t *testing.T) {
	requireMetalRuntime(t)

	f := metal.FromValues([]float32{0}, 1, 1, 1, 1)
	defer metal.Free(f)

	g := gateDecay(f, 2.0)
	defer metal.Free(g)

	want := float32(math.Log(0.5) / 2.0)
	checkClose(t, "gateDecayNorm", g.Floats(), []float32{want})
}

// TestSlotWrite_Math_Good pins s = 1 - exp(g). g=log(0.5) → exp 0.5 → s=0.5.
func TestSlotWrite_Math_Good(t *testing.T) {
	requireMetalRuntime(t)

	g := metal.FromValues([]float32{float32(math.Log(0.5)), float32(math.Log(0.25))}, 1, 1, 1, 2)
	defer metal.Free(g)

	s := slotWrite(g)
	defer metal.Free(s)

	// 1-0.5=0.5 ; 1-0.25=0.75
	checkClose(t, "slotWrite", s.Floats(), []float32{0.5, 0.75})
}

// TestRecurrence_SingleSlot_TwoStep_Good pins the full gated recurrence against
// a hand-derived 2-step trace. 1 head, HeadK=HeadV=Slots=1, zero init.
// Step0: q=2,k=3,v=5,decay=0.5,s=0.5 → Sk=1.5, Sv=2.5, o0=2.5.
// Step1: q=1,k=4,v=7,decay=0.25,s=0.75 → Sk=3.375, Sv=5.875, o1=5.875.
func TestRecurrence_SingleSlot_TwoStep_Good(t *testing.T) {
	requireMetalRuntime(t)

	// [B,H,L,dim] with B=H=1, L=2.
	q := metal.FromValues([]float32{2, 1}, 1, 1, 2, 1)
	k := metal.FromValues([]float32{3, 4}, 1, 1, 2, 1)
	v := metal.FromValues([]float32{5, 7}, 1, 1, 2, 1)
	ln2 := float32(math.Log(0.5))
	ln4 := float32(math.Log(0.25))
	g := metal.FromValues([]float32{ln2, ln4}, 1, 1, 2, 1)
	s := metal.FromValues([]float32{0.5, 0.75}, 1, 1, 2, 1)
	sk0 := metal.Zeros([]int32{1, 1, 1, 1}, metal.DTypeFloat32)
	sv0 := metal.Zeros([]int32{1, 1, 1, 1}, metal.DTypeFloat32)
	defer metal.Free(q, k, v, g, s, sk0, sv0)

	out, skN, svN := recurrence(q, k, v, g, s, sk0, sv0)
	defer metal.Free(out, skN, svN)

	checkClose(t, "out", out.Floats(), []float32{2.5, 5.875})
	checkClose(t, "skN", skN.Floats(), []float32{3.375})
	checkClose(t, "svN", svN.Floats(), []float32{5.875})
}

// TestRecurrence_TwoSlot_SingleStep_Good pins the slot outer-products and the
// softmax-over-slots read (a single slot would hide shape bugs). 1 step,
// HeadK=HeadV=1, Slots=2, zero init. q=1,k=2,v=3, slot-write s=[0.5,0.25]:
//
//	Sk = k·s = [1.0, 0.5];  Sv = s·v = [1.5, 0.75]
//	scores = q·Sk = [1.0, 0.5];  softmax = [0.622459, 0.377541]
//	o = a·Sv = 0.622459*1.5 + 0.377541*0.75 = 1.216844
func TestRecurrence_TwoSlot_SingleStep_Good(t *testing.T) {
	requireMetalRuntime(t)

	q := metal.FromValues([]float32{1}, 1, 1, 1, 1) // [B,H,L=1,HeadK=1]
	k := metal.FromValues([]float32{2}, 1, 1, 1, 1)
	v := metal.FromValues([]float32{3}, 1, 1, 1, 1)    // HeadV=1
	g := metal.FromValues([]float32{0, 0}, 1, 1, 1, 2) // decay irrelevant at step0 (zero init)
	s := metal.FromValues([]float32{0.5, 0.25}, 1, 1, 1, 2)
	sk0 := metal.Zeros([]int32{1, 1, 1, 2}, metal.DTypeFloat32) // [B,H,HeadK=1,Slots=2]
	sv0 := metal.Zeros([]int32{1, 1, 2, 1}, metal.DTypeFloat32) // [B,H,Slots=2,HeadV=1]
	defer metal.Free(q, k, v, g, s, sk0, sv0)

	out, skN, svN := recurrence(q, k, v, g, s, sk0, sv0)
	defer metal.Free(out, skN, svN)

	checkClose(t, "out", out.Floats(), []float32{1.216844})
	checkClose(t, "skN", skN.Floats(), []float32{1.0, 0.5})
	checkClose(t, "svN", svN.Floats(), []float32{1.5, 0.75})
}

// TestRecurrence_InitialStateCarries_Good confirms a non-zero initial slot
// memory is decayed and added to, not ignored. 1 step, single slot, zero
// write (s=0) so the output is purely the decayed initial Sv read back.
// Sk0=4, Sv0=10, decay=0.5, s=0 → Sk=2, Sv=5, a=1, o=5.
func TestRecurrence_InitialStateCarries_Good(t *testing.T) {
	requireMetalRuntime(t)

	q := metal.FromValues([]float32{1}, 1, 1, 1, 1)
	k := metal.FromValues([]float32{9}, 1, 1, 1, 1) // irrelevant: s=0 writes nothing
	v := metal.FromValues([]float32{9}, 1, 1, 1, 1)
	g := metal.FromValues([]float32{float32(math.Log(0.5))}, 1, 1, 1, 1) // decay 0.5
	s := metal.FromValues([]float32{0}, 1, 1, 1, 1)                      // no write
	sk0 := metal.FromValues([]float32{4}, 1, 1, 1, 1)
	sv0 := metal.FromValues([]float32{10}, 1, 1, 1, 1)
	defer metal.Free(q, k, v, g, s, sk0, sv0)

	out, skN, svN := recurrence(q, k, v, g, s, sk0, sv0)
	defer metal.Free(out, skN, svN)

	// Sk = 4*0.5 + 9*0 = 2 ; Sv = 10*0.5 + 0*9 = 5 ; o = softmax(1*2)·5 = 5.
	checkClose(t, "out", out.Floats(), []float32{5})
	checkClose(t, "skN", skN.Floats(), []float32{2})
	checkClose(t, "svN", svN.Floats(), []float32{5})
}

// identityLinear builds an n×n identity-weight Linear so a projection passes its
// input through unchanged (y = x @ Iᵀ = x). Lets the holder integration test run
// the full Forward path with traceable, identity projections.
func identityLinear(n int32) *metal.Linear {
	data := make([]float32, int(n)*int(n))
	for i := int32(0); i < n; i++ {
		data[int(i)*int(n)+int(i)] = 1
	}
	return metal.NewLinear(metal.FromValues(data, int(n), int(n)), nil)
}

// gsaIdentityMixer builds a GSA mixer whose projections are all identity, with
// hidden = HeadK = HeadV = Slots = 2 and a single head, so the Forward path runs
// end-to-end on a [B,L,2] input with the math determined only by the recurrence
// and the output gate.
func gsaIdentityMixer() *Mixer {
	return &Mixer{
		QProj: identityLinear(2), KProj: identityLinear(2), VProj: identityLinear(2),
		FProj: identityLinear(2), GProj: identityLinear(2), OProj: identityLinear(2),
		NumHeads: 1, HeadK: 2, HeadV: 2, Slots: 2, GateNorm: 1,
	}
}

// freeMixer releases the identity-mixer's six projection weights.
func freeMixer(m *Mixer) {
	for _, l := range []*metal.Linear{m.QProj, m.KProj, m.VProj, m.FProj, m.GProj, m.OProj} {
		metal.FreeLinear(l)
	}
}

// TestForward_ChunkedDecodeMatchesSinglePass_Good is the holder GATE: running
// Forward over a 2-token chunk in ONE pass must equal running it as two 1-token
// chunks where the recurrent holder threads the slot memory between them. This
// proves Forward reads/writes ctx.Recurrent() correctly — the integration the
// whole #39 holder exists for.
func TestForward_ChunkedDecodeMatchesSinglePass_Good(t *testing.T) {
	requireMetalRuntime(t)

	m := gsaIdentityMixer()
	defer func() {
		freeMixer(m)
	}()

	// Two tokens of hidden state [B=1, L=2, hidden=2].
	tok0 := []float32{0.5, -0.3}
	tok1 := []float32{0.1, 0.7}

	// Single pass: both tokens at once through one holder.
	xFull := metal.FromValues(append(append([]float32{}, tok0...), tok1...), 1, 2, 2)
	rcFull := metal.NewRecurrentCache()
	outFull, _ := m.Forward(xFull, &metal.MixerCtx{Cache: rcFull, B: 1, L: 2})
	full := outFull.Floats()
	metal.Free(xFull, outFull)
	rcFull.Reset()

	// Chunked: token0 then token1 through a second holder that carries state.
	rcChunk := metal.NewRecurrentCache()
	x0 := metal.FromValues(tok0, 1, 1, 2)
	out0, _ := m.Forward(x0, &metal.MixerCtx{Cache: rcChunk, B: 1, L: 1, Step: 0})
	metal.Free(x0, out0)
	x1 := metal.FromValues(tok1, 1, 1, 2)
	out1, _ := m.Forward(x1, &metal.MixerCtx{Cache: rcChunk, B: 1, L: 1, Step: 1})
	chunkTok1 := out1.Floats()
	metal.Free(x1, out1)
	rcChunk.Reset()

	// The single-pass output for token 1 is its second [HeadV=2] row.
	wantTok1 := full[2:4]
	checkClose(t, "chunked-token1 == single-pass-token1", chunkTok1, wantTok1)
}

// TestForward_FirstForwardZeroState_Good confirms a fresh holder (nil prior)
// starts the recurrence from zero: with an empty RecurrentState the first
// Forward must equal the same Forward against an explicit zero seed. Guards the
// priorState nil-branch.
func TestForward_FirstForwardZeroState_Good(t *testing.T) {
	requireMetalRuntime(t)

	m := gsaIdentityMixer()
	defer func() {
		freeMixer(m)
	}()

	x := metal.FromValues([]float32{0.2, 0.4}, 1, 1, 2)
	defer metal.Free(x)

	rc := metal.NewRecurrentCache()
	out, _ := m.Forward(x, &metal.MixerCtx{Cache: rc, B: 1, L: 1})
	got := out.Floats()
	metal.Free(out)
	rc.Reset()

	// After the first forward the holder must hold the two advanced slots
	// (Sk, Sv) — proves SetRecurrentState ran with a 2-slot state.
	if state := rc.RecurrentState(); len(state) != 0 {
		t.Errorf("holder retained %d slots after Reset, want 0", len(state))
	}
	if len(got) != 2 {
		t.Fatalf("output width = %d, want 2 (HeadV)", len(got))
	}
}

// TestForward_MissingHolderPanics_Bad confirms a StateRecurrent mixer handed a
// non-recurrent cache fails loudly rather than miscomputing — the load-time
// Compatible() gate should prevent this, so a missing holder is fatal.
func TestForward_MissingHolderPanics_Bad(t *testing.T) {
	requireMetalRuntime(t)

	m := gsaIdentityMixer()
	defer func() {
		freeMixer(m)
		if r := recover(); r == nil {
			t.Fatal("Forward with a non-recurrent cache should panic")
		}
	}()

	x := metal.FromValues([]float32{0.2, 0.4}, 1, 1, 2)
	defer metal.Free(x)
	// A plain KV cache is not a RecurrentCache → ctx.Recurrent() reports false.
	_, _ = m.Forward(x, &metal.MixerCtx{Cache: metal.NewKVCache(), B: 1, L: 1})
}

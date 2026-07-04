// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package mamba2

import (
	"testing"

	metal "dappco.re/go/mlx/pkg/metal"
	scheme "dappco.re/go/mlx/pkg/scheme"
)

// randWeights builds a small deterministic Mamba-2 weight set + config for the
// Forward tests. The projections are fixed (not random) so a failure is
// reproducible; the values are small so softplus/exp stay well-scaled at
// fixture precision.
func tinyMixer(t *testing.T) *Mixer {
	t.Helper()
	const D, H, P, N, G, K = 4, 2, 2, 2, 1, 3
	cfg := &Config{NumHeads: H, HeadDim: P, StateDim: N, NumGroups: G, ConvKernel: K, RMSNormEps: 1e-5}
	dInner := int(cfg.dInner())     // 4
	convDim := int(cfg.convDim())   // dInner + 2*G*N = 4 + 4 = 8
	projOut := 2*dInner + 2*G*N + H // z(4) + xBC(8) + dt(2) = 14

	// in_proj weight [projOut, D] — small structured values (row-scaled).
	inW := make([]float32, projOut*D)
	for r := 0; r < projOut; r++ {
		for c := 0; c < D; c++ {
			inW[r*D+c] = 0.05 * float32((r%3)-1) * float32(c+1)
		}
	}
	inProj := metal.NewLinear(metal.FromValues(inW, projOut, D), nil)

	// depthwise conv1d weight [convDim, 1, K].
	convW := make([]float32, convDim*1*K)
	for c := 0; c < convDim; c++ {
		for k := 0; k < K; k++ {
			convW[c*K+k] = 0.1 * float32(k+1)
		}
	}
	conv := metal.FromValues(convW, convDim, 1, K)

	// A_log [H], D [H], dt_bias [H], norm [dInner], out_proj [D, dInner].
	aLog := metal.FromValues([]float32{0.0, -0.3}, H)
	dSkip := metal.FromValues([]float32{0.1, 0.2}, H)
	dtBias := metal.FromValues([]float32{-0.5, -0.4}, H)
	norm := metal.FromValues([]float32{1.0, 1.1, 0.9, 1.05}, dInner)
	outW := make([]float32, D*dInner)
	for r := 0; r < D; r++ {
		for c := 0; c < dInner; c++ {
			outW[r*dInner+c] = 0.07 * float32((c%2)*2-1) * float32(r+1)
		}
	}
	outProj := metal.NewLinear(metal.FromValues(outW, D, dInner), nil)

	w := &Weights{
		InProj: inProj, ConvWeight: conv, ALog: aLog, D: dSkip,
		DtBias: dtBias, Norm: norm, OutProj: outProj,
	}
	return NewMixer(w, cfg)
}

func closeEnough(t *testing.T, label string, got, want []float32, tol float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: length got %d want %d", label, len(got), len(want))
	}
	for i := range got {
		d := got[i] - want[i]
		if d < 0 {
			d = -d
		}
		if d > tol {
			t.Errorf("%s[%d] = %v, want %v (diff %v)", label, i, got[i], want[i], d)
		}
	}
}

// TestMixer_Forward_Good runs a 2-token prefill in one Forward and asserts it
// produces a finite output of the right shape, and that the holder ends with
// the two advanced state slots (conv-state ring + ssm-state) of the expected
// shapes — the load-time contract the decoder relies on.
func TestMixer_Forward_Good(t *testing.T) {
	m := tinyMixer(t)
	cfg := m.W.cfg
	const B, L, D = 1, 2, 4
	x := metal.FromValues([]float32{
		0.5, -0.2, 0.1, 0.3,
		-0.1, 0.4, -0.3, 0.2,
	}, B, L, D)
	defer metal.Free(x)

	rc := metal.NewRecurrentCache()
	out, _ := m.Forward(x, &metal.MixerCtx{Cache: rc, B: B, L: L})
	if out == nil {
		t.Fatal("Forward returned nil output")
	}
	defer metal.Free(out)

	if got := out.Shape(); len(got) != 3 || got[0] != B || got[1] != L || got[2] != D {
		t.Fatalf("out shape = %v, want [%d %d %d]", got, B, L, D)
	}
	for i, v := range out.Floats() {
		if v != v { // NaN
			t.Fatalf("out[%d] is NaN", i)
		}
	}

	slots := rc.RecurrentState()
	if len(slots) != numSlots {
		t.Fatalf("holder has %d slots, want %d", len(slots), numSlots)
	}
	if got := slots[slotConvState].Shape(); len(got) != 3 || got[0] != B || got[1] != cfg.convDim() || got[2] != cfg.ConvKernel-1 {
		t.Errorf("conv-state shape = %v, want [%d %d %d]", got, B, cfg.convDim(), cfg.ConvKernel-1)
	}
	if got := slots[slotSSMState].Shape(); len(got) != 4 || got[0] != B || got[1] != cfg.NumHeads || got[2] != cfg.HeadDim || got[3] != cfg.StateDim {
		t.Errorf("ssm-state shape = %v, want [%d %d %d %d]", got, B, cfg.NumHeads, cfg.HeadDim, cfg.StateDim)
	}
}

// TestMixer_Forward_Bad is the state-threading GATE: a 2-token prefill in one
// Forward must equal two sequential 1-token Forwards through the same holder.
// This proves the conv-state ring AND the ssm-state carry thread correctly —
// streamed decode is identical to single-shot prefill only if both states are
// read, advanced, and written back exactly. (Mirrors the reference equivalence
// SSDScanChunked already holds vs SSDScan; here it is end-to-end through the block.)
func TestMixer_Forward_Bad(t *testing.T) {
	const B, D = 1, 4
	tok0 := []float32{0.5, -0.2, 0.1, 0.3}
	tok1 := []float32{-0.1, 0.4, -0.3, 0.2}

	// Prefill: both tokens in one Forward.
	mPre := tinyMixer(t)
	xPre := metal.FromValues(append(append([]float32{}, tok0...), tok1...), B, 2, D)
	rcPre := metal.NewRecurrentCache()
	outPre, _ := mPre.Forward(xPre, &metal.MixerCtx{Cache: rcPre, B: B, L: 2})
	metal.Free(xPre)
	if outPre == nil {
		t.Fatal("prefill Forward nil")
	}
	prefill := outPre.Floats() // [1,2,4] flattened = 8
	metal.Free(outPre)

	// Sequential decode: token 0 then token 1 through one holder.
	mDec := tinyMixer(t)
	rcDec := metal.NewRecurrentCache()
	x0 := metal.FromValues(tok0, B, 1, D)
	o0, _ := mDec.Forward(x0, &metal.MixerCtx{Cache: rcDec, B: B, L: 1, Step: 0})
	metal.Free(x0)
	if o0 == nil {
		t.Fatal("decode step 0 nil")
	}
	dec0 := o0.Floats()
	metal.Free(o0)

	x1 := metal.FromValues(tok1, B, 1, D)
	o1, _ := mDec.Forward(x1, &metal.MixerCtx{Cache: rcDec, B: B, L: 1, Step: 1})
	metal.Free(x1)
	if o1 == nil {
		t.Fatal("decode step 1 nil")
	}
	dec1 := o1.Floats()
	metal.Free(o1)

	// prefill token-0 slice == decode step 0; prefill token-1 slice == decode step 1.
	closeEnough(t, "prefill-tok0-vs-decode0", prefill[0:D], dec0, 2e-3)
	closeEnough(t, "prefill-tok1-vs-decode1", prefill[D:2*D], dec1, 2e-3)
}

// TestMixer_Ugly asserts the scheme scaffold: the weightless family seed
// resolves through the catalogue (identity + state only), and a built *Mixer —
// what the loader registers — carries the metal compute surface. The seed
// itself does NOT satisfy metal.MixerCompute (it has no Forward); that is the
// load-time-attaches-compute contract, the same shape as the gla peer.
func TestMixer_Ugly(t *testing.T) {
	m := &Mixer{}
	if m.Kind() != "mamba2" {
		t.Errorf("Kind() = %q, want mamba2", m.Kind())
	}
	if m.State() != scheme.StateRecurrent {
		t.Errorf("State() = %v, want StateRecurrent", m.State())
	}
	// The weightless seed (register.go's familyInfo) resolves with the right
	// state but carries no compute.
	resolved, ok := scheme.MixerFor("mamba2")
	if !ok {
		t.Fatal("scheme.MixerFor(mamba2) not registered")
	}
	if resolved.State() != scheme.StateRecurrent {
		t.Errorf("resolved State() = %v, want StateRecurrent", resolved.State())
	}
	// A built *Mixer is a full metal.MixerCompute — register one (as the loader
	// would) and confirm it resolves with the compute surface attached.
	scheme.RegisterMixer(tinyMixer(t))
	if _, ok := metal.MixerComputeFor("mamba2"); !ok {
		t.Error("metal.MixerComputeFor(mamba2) missing compute surface after registering a built mixer")
	}
}

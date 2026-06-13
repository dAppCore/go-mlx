// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package rwkv7

import (
	"testing"

	metal "dappco.re/go/mlx/pkg/metal"
	scheme "dappco.re/go/mlx/pkg/scheme"
)

// tinyMixer builds a small deterministic RWKV-7 weight set + config for the
// Forward tests. The projections are fixed (not random) so a failure is
// reproducible; values are small so exp/log-decay stay well-scaled at fixture
// precision.
func tinyMixer(t *testing.T) *Mixer {
	t.Helper()
	const D, H, K, V = 4, 2, 2, 2
	cfg := &Config{NumHeads: H, KeyDim: K, ValueDim: V}

	// A [outDim, D] projection with small structured values, parametrised by a
	// per-projection seed so r/w/k/v/a/b differ.
	proj := func(outDim, seed int) *metal.Linear {
		w := make([]float32, outDim*D)
		for r := 0; r < outDim; r++ {
			for c := 0; c < D; c++ {
				w[r*D+c] = 0.04 * float32((r+seed)%3-1) * float32(c+1)
			}
		}
		return metal.NewLinear(metal.FromValues(w, outDim, D), nil)
	}
	hk := H * K
	hv := H * V
	w := &Weights{
		RProj:   proj(hk, 0),
		WProj:   proj(hk, 1),
		KProj:   proj(hk, 2),
		VProj:   proj(hv, 3),
		AProj:   proj(hk, 4),
		BProj:   proj(hk, 5),
		OutProj: proj(D, 6), // [D, H*V]; H*V == D here
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

// TestMixer_Forward_Good runs a 2-token prefill in one Forward and asserts a
// finite output of the right shape and that the holder ends with the single
// advanced [B,H,K,V] state slot — the load-time contract the decoder relies on.
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
		if v != v {
			t.Fatalf("out[%d] is NaN", i)
		}
	}

	slots := rc.RecurrentState()
	if len(slots) != 1 {
		t.Fatalf("holder has %d slots, want 1", len(slots))
	}
	if got := slots[slotWKVState].Shape(); len(got) != 4 || got[0] != B || got[1] != cfg.NumHeads || got[2] != cfg.KeyDim || got[3] != cfg.ValueDim {
		t.Errorf("wkv-state shape = %v, want [%d %d %d %d]", got, B, cfg.NumHeads, cfg.KeyDim, cfg.ValueDim)
	}
}

// TestMixer_Forward_Bad is the state-threading GATE: a 2-token prefill in one
// Forward must equal two sequential 1-token Forwards through the same holder.
// This proves the single [K,V] state threads correctly — streamed decode is
// identical to single-shot prefill only if the prior state is read, advanced,
// and written back exactly each step.
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
	prefill := outPre.Floats()
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

	closeEnough(t, "prefill-tok0-vs-decode0", prefill[0:D], dec0, 2e-3)
	closeEnough(t, "prefill-tok1-vs-decode1", prefill[D:2*D], dec1, 2e-3)
}

// TestMixer_Ugly asserts the scheme scaffold: the weightless family seed
// resolves (identity + state only), and a built *Mixer — what the loader
// registers — carries the metal compute surface. The seed itself does NOT
// satisfy metal.MixerCompute (no Forward); the load-time-attaches-compute
// contract, same shape as the mamba2/gla peers.
func TestMixer_Ugly(t *testing.T) {
	m := &Mixer{}
	if m.Kind() != "rwkv7" {
		t.Errorf("Kind() = %q, want rwkv7", m.Kind())
	}
	if m.State() != scheme.StateRecurrent {
		t.Errorf("State() = %v, want StateRecurrent", m.State())
	}
	resolved, ok := scheme.MixerFor("rwkv7")
	if !ok {
		t.Fatal("scheme.MixerFor(rwkv7) not registered")
	}
	if resolved.State() != scheme.StateRecurrent {
		t.Errorf("resolved State() = %v, want StateRecurrent", resolved.State())
	}
	scheme.RegisterMixer(tinyMixer(t))
	if _, ok := metal.MixerComputeFor("rwkv7"); !ok {
		t.Error("metal.MixerComputeFor(rwkv7) missing compute surface after registering a built mixer")
	}
}

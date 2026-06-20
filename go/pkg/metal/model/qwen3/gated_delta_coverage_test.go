// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && metal_runtime

// Coverage companions for gated_delta_mixer.go + gated_delta_loader.go — they
// drive the mixer's identity/state accessors, its nil-weight and no-recurrent-
// holder guards, and the geometry-derivation error arms of the loader. All
// synthetic (hand-built Linears + Arrays via mkLin/mkArr), no live model.

package qwen3

import (
	"testing"

	metal "dappco.re/go/mlx/pkg/metal"
	scheme "dappco.re/go/mlx/pkg/scheme"
)

// TestQwen3_GatedDeltaMixer_KindState_Good covers the compute mixer's identity
// accessors: Kind() resolves the "gated_deltanet" registry token and State()
// declares the recurrent contract.
func TestQwen3_GatedDeltaMixer_KindState_Good(t *testing.T) {
	m := buildGatedDeltaMixer()
	if got := m.Kind(); got != "gated_deltanet" {
		t.Fatalf("Kind() = %q, want gated_deltanet", got)
	}
	if got := m.State(); got != scheme.StateRecurrent {
		t.Fatalf("State() = %v, want StateRecurrent", got)
	}
}

// TestQwen3_GatedDeltaFamily_State_Good covers the registry-stub family's State()
// (the pure identity registered in init() before a load overwrites it with a
// compute-bearing mixer).
func TestQwen3_GatedDeltaFamily_State_Good(t *testing.T) {
	if got := (gatedDeltaFamily{}).State(); got != scheme.StateRecurrent {
		t.Fatalf("gatedDeltaFamily.State() = %v, want StateRecurrent", got)
	}
}

// TestQwen3_GatedDeltaMixer_NilWeights_Bad covers the nil-weight guard of
// Forward: a mixer with no weight set (W == nil) returns a nil output and the
// zero shared-KV rather than dereferencing.
func TestQwen3_GatedDeltaMixer_NilWeights_Bad(t *testing.T) {
	m := &GatedDeltaMixer{} // W == nil
	out, kv := m.Forward(metal.FromValues([]float32{1, 2, 3}, 1, 1, 3), &metal.MixerCtx{B: 1, L: 1})
	if out != nil {
		t.Fatalf("Forward(nil weights) out = %v, want nil", out)
	}
	if kv.Keys != nil || kv.Values != nil {
		t.Fatalf("Forward(nil weights) kv = %+v, want zero SharedKV", kv)
	}

	// A weight set with a nil cfg takes the same guard.
	mc := NewGatedDeltaMixer(&GatedDeltaWeights{}, nil) // cfg == nil
	if out2, _ := mc.Forward(metal.FromValues([]float32{1}, 1, 1, 1), &metal.MixerCtx{B: 1, L: 1}); out2 != nil {
		t.Fatalf("Forward(nil cfg) out = %v, want nil", out2)
	}
}

// TestQwen3_GatedDeltaMixer_NoRecurrentHolder_Good drives the stateless-prefill
// path: a MixerCtx with no recurrent holder (nil Cache). The first Recurrent()
// probe yields no prior state and the writeback at the end frees the advanced
// state instead of storing it (the else arm the recurrent-cache tests skip). The
// output must still be the right shape and finite.
func TestQwen3_GatedDeltaMixer_NoRecurrentHolder_Good(t *testing.T) {
	m := buildGatedDeltaMixer()
	x := mkArr(0.12, 1, 3, 8) // [B,L,hidden]
	defer metal.Free(x)

	out, kv := m.Forward(x, &metal.MixerCtx{B: 1, L: 3}) // Cache == nil → no holder
	if out == nil {
		t.Fatal("Forward(no holder) returned nil")
	}
	defer metal.Free(out)
	if kv.Keys != nil || kv.Values != nil {
		t.Fatalf("Forward(no holder) kv = %+v, want zero SharedKV", kv)
	}
	if s := out.Shape(); len(s) != 3 || s[0] != 1 || s[1] != 3 || s[2] != 8 {
		t.Fatalf("out shape %v, want [1,3,8]", s)
	}
	if err := metal.Eval(out); err != nil {
		t.Fatalf("eval: %v", err)
	}
}

// TestQwen3_GatedDeltaConfigFromShapes_Bad walks every geometry-derivation
// failure arm: a QKV projection with no weight, a malformed (rank<2) QKV shape,
// and an inconsistent conv width (convDim != the QKV projection out). Each must
// surface an error, not a silently mis-built config.
func TestQwen3_GatedDeltaConfigFromShapes_Bad(t *testing.T) {
	convWeight := mkArr(0.2, 32, 1, 3)
	aLog := mkArr(-0.5, 4)
	norm := mkArr(1.0, 4)
	defer metal.Free(convWeight, aLog, norm)

	t.Run("nil-qkv-weight", func(t *testing.T) {
		qkv := &metal.Linear{} // Weight == nil
		if _, err := gatedDeltaConfigFromShapes(qkv, convWeight, aLog, norm, 1e-6); err == nil {
			t.Fatal("gatedDeltaConfigFromShapes(nil qkv weight) = nil, want error")
		}
	})

	t.Run("malformed-qkv-shape", func(t *testing.T) {
		qkv := metal.NewLinear(mkArr(0.1, 32), nil) // rank-1 weight: len(shape) < 2
		defer metal.FreeLinear(qkv)
		if _, err := gatedDeltaConfigFromShapes(qkv, convWeight, aLog, norm, 1e-6); err == nil {
			t.Fatal("gatedDeltaConfigFromShapes(rank-1 qkv) = nil, want error")
		}
	})

	t.Run("inconsistent-geometry", func(t *testing.T) {
		qkv := mkLin(32, 8) // projection out = 32
		defer metal.FreeLinear(qkv)
		badConv := mkArr(0.2, 30, 1, 3) // convDim 30 != 32 ⇒ geometry inconsistent
		defer metal.Free(badConv)
		if _, err := gatedDeltaConfigFromShapes(qkv, badConv, aLog, norm, 1e-6); err == nil {
			t.Fatal("gatedDeltaConfigFromShapes(convDim != projOut) = nil, want error")
		}
	})
}

// TestQwen3_BuildGatedDelta_BadGeometry_Bad covers buildGatedDelta's "derive
// geometry" error wrap: every required weight is present, but the conv width is
// inconsistent with the QKV projection out, so gatedDeltaConfigFromShapes fails
// and buildGatedDelta wraps it rather than returning a broken mixer.
func TestQwen3_BuildGatedDelta_BadGeometry_Bad(t *testing.T) {
	ctx := gatedDeltaCtx()
	origWt := ctx.Weight
	ctx.Weight = func(n string) *metal.Array {
		if n == "conv1d.weight" {
			return mkArr(0.2, 30, 1, 3) // convDim 30 != projOut 32
		}
		return origWt(n)
	}
	if _, err := buildGatedDelta(ctx); err == nil {
		t.Fatal("buildGatedDelta(inconsistent conv width) = nil, want a derive-geometry error")
	}
}

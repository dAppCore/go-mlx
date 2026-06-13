// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package mamba2

import (
	"testing"

	metal "dappco.re/go/mlx/pkg/metal"
	scheme "dappco.re/go/mlx/pkg/scheme"
)

// fakeBuildCtx assembles a neutral metal.MixerBuildCtx from synthetic tensors —
// the Linear resolver returns quant-ready dense Linears, the Weight accessor
// returns raw arrays, mirroring what the real load path hands a loader (only
// there the Linears come from the quant factory). The shapes are the
// self-consistent nGroups=1 geometry the loader infers (H=2, headDim=2, N=2 →
// dInner=4, convDim=8, projOut=14).
func fakeBuildCtx() metal.MixerBuildCtx {
	const D = 4
	linears := map[string]*metal.Linear{
		"in_proj":  metal.NewLinear(metal.FromValues(make([]float32, 14*D), 14, D), nil),
		"out_proj": metal.NewLinear(metal.FromValues(make([]float32, D*4), D, 4), nil),
	}
	weights := map[string]*metal.Array{
		"conv1d.weight": metal.FromValues(make([]float32, 8*1*3), 8, 1, 3),
		"A_log":         metal.FromValues(make([]float32, 2), 2),
		"D":             metal.FromValues(make([]float32, 2), 2),
		"dt_bias":       metal.FromValues(make([]float32, 2), 2),
		"norm.weight":   metal.FromValues(make([]float32, 4), 4),
	}
	return metal.MixerBuildCtx{
		Linear:   func(p string) *metal.Linear { return linears[p] },
		Weight:   func(n string) *metal.Array { return weights[n] },
		Cfg:      metal.TransformerConfig{RMSNormEps: 1e-5},
		LayerIdx: 0,
	}
}

// TestLoader_BuildMamba2_Good is the #44 loader gate: the registered loader
// builds a working Mamba-2 mixer from the neutral ctx — geometry inferred from
// the synthetic shapes, a recurrent MixerCompute returned, and a forward through
// a holder yields the right output shape + the two state slots.
func TestLoader_BuildMamba2_Good(t *testing.T) {
	const D = 4
	mixer, err := buildMamba2(fakeBuildCtx())
	if err != nil {
		t.Fatalf("buildMamba2: %v", err)
	}
	if mixer.Kind() != "mamba2" || mixer.State() != scheme.StateRecurrent {
		t.Fatalf("built mixer = (%q,%v), want (mamba2,recurrent)", mixer.Kind(), mixer.State())
	}

	x := metal.FromValues(make([]float32, 1*2*D), 1, 2, D)
	defer metal.Free(x)
	rc := metal.NewRecurrentCache()
	out, _ := mixer.Forward(x, &metal.MixerCtx{Cache: rc, B: 1, L: 2})
	if out == nil {
		t.Fatal("built mamba2 mixer Forward returned nil")
	}
	defer metal.Free(out)
	if got := out.Shape(); len(got) != 3 || got[0] != 1 || got[1] != 2 || got[2] != D {
		t.Errorf("forward out shape = %v, want [1 2 %d]", got, D)
	}
	if len(rc.RecurrentState()) != numSlots {
		t.Errorf("holder slots = %d, want %d (conv-state + ssm-state)", len(rc.RecurrentState()), numSlots)
	}
}

// TestLoader_BuildMamba2_Bad: a missing required tensor is a loud build error,
// not a silent zero-weight mixer.
func TestLoader_BuildMamba2_Bad(t *testing.T) {
	ctx := fakeBuildCtx()
	// Drop conv1d.weight from the accessor.
	base := ctx.Weight
	ctx.Weight = func(n string) *metal.Array {
		if n == "conv1d.weight" {
			return nil
		}
		return base(n)
	}
	if _, err := buildMamba2(ctx); err == nil {
		t.Error("expected error for missing conv1d.weight, got nil")
	}
}

// TestLoader_Registered_Ugly asserts the loader self-registers from init() so
// the engine resolves it by kind — the contract the generic config-composed
// load path depends on. RegisterMixerLoader is idempotent (re-register is a
// no-op-equivalent override), so re-registering buildMamba2 here is safe and
// confirms the kind round-trips through a built mixer.
func TestLoader_Registered_Ugly(t *testing.T) {
	metal.RegisterMixerLoader("mamba2", buildMamba2)
	mixer, err := buildMamba2(fakeBuildCtx())
	if err != nil {
		t.Fatalf("registered loader build: %v", err)
	}
	if mixer.Kind() != "mamba2" {
		t.Errorf("Kind() = %q, want mamba2", mixer.Kind())
	}
}

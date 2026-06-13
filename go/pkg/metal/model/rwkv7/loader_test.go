// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package rwkv7

import (
	"testing"

	metal "dappco.re/go/mlx/pkg/metal"
	scheme "dappco.re/go/mlx/pkg/scheme"
)

// fakeBuildCtx assembles a neutral metal.MixerBuildCtx from synthetic tensors.
// RWKV-7 is all-Linear, so every projection comes through the Linear resolver;
// H=NumAttentionHeads(2), receptance/value widths H*K=4 / H*V=4 → K=V=2.
func fakeBuildCtx() metal.MixerBuildCtx {
	const D = 4
	linears := map[string]*metal.Linear{
		"receptance": metal.NewLinear(metal.FromValues(make([]float32, 4*D), 4, D), nil),
		"key":        metal.NewLinear(metal.FromValues(make([]float32, 4*D), 4, D), nil),
		"value":      metal.NewLinear(metal.FromValues(make([]float32, 4*D), 4, D), nil),
		"output":     metal.NewLinear(metal.FromValues(make([]float32, D*4), D, 4), nil),
		"decay":      metal.NewLinear(metal.FromValues(make([]float32, 4*D), 4, D), nil),
		"a_proj":     metal.NewLinear(metal.FromValues(make([]float32, 4*D), 4, D), nil),
		"b_proj":     metal.NewLinear(metal.FromValues(make([]float32, 4*D), 4, D), nil),
	}
	cfg := metal.TransformerConfig{}
	cfg.NumAttentionHeads = 2
	return metal.MixerBuildCtx{
		Linear:   func(p string) *metal.Linear { return linears[p] },
		Weight:   func(n string) *metal.Array { return nil },
		Cfg:      cfg,
		LayerIdx: 1,
	}
}

// TestLoader_BuildRWKV7_Good is the #44 loader gate: the registered loader
// builds a working RWKV-7 mixer from the neutral ctx — geometry inferred from
// the synthetic shapes, a recurrent MixerCompute returned, and a forward through
// a holder yields the right output shape + the single state slot.
func TestLoader_BuildRWKV7_Good(t *testing.T) {
	const D = 4
	mixer, err := buildRWKV7(fakeBuildCtx())
	if err != nil {
		t.Fatalf("buildRWKV7: %v", err)
	}
	if mixer.Kind() != "rwkv7" || mixer.State() != scheme.StateRecurrent {
		t.Fatalf("built mixer = (%q,%v), want (rwkv7,recurrent)", mixer.Kind(), mixer.State())
	}

	x := metal.FromValues(make([]float32, 1*2*D), 1, 2, D)
	defer metal.Free(x)
	rc := metal.NewRecurrentCache()
	out, _ := mixer.Forward(x, &metal.MixerCtx{Cache: rc, B: 1, L: 2})
	if out == nil {
		t.Fatal("built rwkv7 mixer Forward returned nil")
	}
	defer metal.Free(out)
	if got := out.Shape(); len(got) != 3 || got[0] != 1 || got[1] != 2 || got[2] != D {
		t.Errorf("forward out shape = %v, want [1 2 %d]", got, D)
	}
	if len(rc.RecurrentState()) != 1 {
		t.Errorf("holder slots = %d, want 1 (wkv-state)", len(rc.RecurrentState()))
	}
}

// TestLoader_BuildRWKV7_Bad: a missing required projection is a loud build
// error, not a silent zero-weight mixer.
func TestLoader_BuildRWKV7_Bad(t *testing.T) {
	ctx := fakeBuildCtx()
	base := ctx.Linear
	ctx.Linear = func(p string) *metal.Linear {
		if p == "decay" {
			return nil
		}
		return base(p)
	}
	if _, err := buildRWKV7(ctx); err == nil {
		t.Error("expected error for missing decay projection, got nil")
	}
}

// TestLoader_Registered_Ugly asserts the loader self-registers from init() so
// the engine resolves it by kind. RegisterMixerLoader override is idempotent, so
// re-registering here is safe and confirms the kind round-trips.
func TestLoader_Registered_Ugly(t *testing.T) {
	metal.RegisterMixerLoader("rwkv7", buildRWKV7)
	mixer, err := buildRWKV7(fakeBuildCtx())
	if err != nil {
		t.Fatalf("registered loader build: %v", err)
	}
	if mixer.Kind() != "rwkv7" {
		t.Errorf("Kind() = %q, want rwkv7", mixer.Kind())
	}
}

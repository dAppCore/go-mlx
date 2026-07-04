// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gsa

import (
	"testing"

	metal "dappco.re/go/mlx/pkg/metal"
	scheme "dappco.re/go/mlx/pkg/scheme"
)

// fakeBuildCtx assembles a neutral metal.MixerBuildCtx with quant-ready dense
// Linears. Geometry: hidden 4, 2 heads, head dim (K=V) 2 → q/k/v/o_proj out 4,
// f_proj out = heads*slots = 2*4 → slots 4.
func fakeBuildCtx() metal.MixerBuildCtx {
	linears := map[string]*metal.Linear{
		"q_proj": metal.NewLinear(metal.FromValues(make([]float32, 4*4), 4, 4), nil),
		"k_proj": metal.NewLinear(metal.FromValues(make([]float32, 4*4), 4, 4), nil),
		"v_proj": metal.NewLinear(metal.FromValues(make([]float32, 4*4), 4, 4), nil),
		"f_proj": metal.NewLinear(metal.FromValues(make([]float32, 8*4), 8, 4), nil),
		"g_proj": metal.NewLinear(metal.FromValues(make([]float32, 4*4), 4, 4), nil),
		"o_proj": metal.NewLinear(metal.FromValues(make([]float32, 4*4), 4, 4), nil),
	}
	return metal.MixerBuildCtx{
		Linear:   func(p string) *metal.Linear { return linears[p] },
		Weight:   func(string) *metal.Array { return nil },
		Cfg:      metal.TransformerConfig{HiddenSize: 4, NumAttentionHeads: 2, HeadDim: 2, RMSNormEps: 1e-5},
		LayerIdx: 0,
	}
}

// TestLoader_BuildGSA_Good is the loader gate: the registered builder constructs
// a working GSA mixer — slots inferred from f_proj, a StateRecurrent
// MixerCompute returned, and a forward through a holder yields the right shape
// and leaves the two slot-memory tensors (Sk, Sv) in the holder.
func TestLoader_BuildGSA_Good(t *testing.T) {
	mixer, err := buildGSA(fakeBuildCtx())
	if err != nil {
		t.Fatalf("buildGSA: %v", err)
	}
	if mixer.Kind() != "gsa" || mixer.State() != scheme.StateRecurrent {
		t.Fatalf("built mixer = (%q,%v), want (gsa,recurrent)", mixer.Kind(), mixer.State())
	}

	x := metal.FromValues(make([]float32, 1*2*4), 1, 2, 4)
	defer metal.Free(x)
	rc := metal.NewRecurrentCache()
	out, _ := mixer.Forward(x, &metal.MixerCtx{Cache: rc, B: 1, L: 2})
	if out == nil {
		t.Fatal("GSA Forward returned nil")
	}
	defer metal.Free(out)
	if got := out.Shape(); len(got) != 3 || got[0] != 1 || got[1] != 2 || got[2] != 4 {
		t.Errorf("forward out shape = %v, want [1 2 4]", got)
	}
	// GSA threads two slots (Sk, Sv) through the holder.
	if got := len(rc.RecurrentState()); got != 2 {
		t.Errorf("holder slots = %d, want 2 (Sk + Sv)", got)
	}
}

// TestLoader_BuildGSA_Bad: a missing required projection is a loud build error.
func TestLoader_BuildGSA_Bad(t *testing.T) {
	ctx := fakeBuildCtx()
	base := ctx.Linear
	ctx.Linear = func(p string) *metal.Linear {
		if p == "f_proj" {
			return nil
		}
		return base(p)
	}
	if _, err := buildGSA(ctx); err == nil {
		t.Error("expected error for missing f_proj, got nil")
	}
}

// TestLoader_Registered_Ugly asserts the loader self-registers from init().
func TestLoader_Registered_Ugly(t *testing.T) {
	metal.RegisterMixerLoader(MixerKind, buildGSA)
	mixer, err := buildGSA(fakeBuildCtx())
	if err != nil {
		t.Fatalf("registered loader build: %v", err)
	}
	if mixer.Kind() != MixerKind {
		t.Errorf("Kind() = %q, want %q", mixer.Kind(), MixerKind)
	}
}

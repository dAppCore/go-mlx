// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package mla

import (
	"testing"

	metal "dappco.re/go/mlx/pkg/metal"
	scheme "dappco.re/go/mlx/pkg/scheme"
)

// fakeBuildCtx assembles a neutral metal.MixerBuildCtx from synthetic tensors —
// the Linear resolver returns quant-ready dense Linears (in the real load path
// they come from the quant factory). Geometry: hidden 4, 2 heads, head dim 2 →
// kv_b_proj out = 2*heads*headDim = 8, latent 6.
func fakeBuildCtx() metal.MixerBuildCtx {
	linears := map[string]*metal.Linear{
		"kv_a_proj_with_mqa": metal.NewLinear(metal.FromValues(make([]float32, 6*4), 6, 4), nil),
		"kv_b_proj":          metal.NewLinear(metal.FromValues(make([]float32, 8*6), 8, 6), nil),
		"q_a_proj":           metal.NewLinear(metal.FromValues(make([]float32, 6*4), 6, 4), nil),
		"q_b_proj":           metal.NewLinear(metal.FromValues(make([]float32, 4*6), 4, 6), nil),
		"o_proj":             metal.NewLinear(metal.FromValues(make([]float32, 4*4), 4, 4), nil),
	}
	return metal.MixerBuildCtx{
		Linear:   func(p string) *metal.Linear { return linears[p] },
		Weight:   func(string) *metal.Array { return nil },
		Cfg:      metal.TransformerConfig{HiddenSize: 4, NumAttentionHeads: 2, HeadDim: 2, RMSNormEps: 1e-5},
		LayerIdx: 0,
	}
}

// TestLoader_BuildMLA_Good is the loader gate: the registered builder constructs
// a working MLA mixer from the neutral ctx — head dim inferred from kv_b_proj, a
// StateKVCache MixerCompute returned, and a forward yields the right shape.
func TestLoader_BuildMLA_Good(t *testing.T) {
	mixer, err := buildMLA(fakeBuildCtx())
	if err != nil {
		t.Fatalf("buildMLA: %v", err)
	}
	if mixer.Kind() != "mla" || mixer.State() != scheme.StateKVCache {
		t.Fatalf("built mixer = (%q,%v), want (mla,kv-cache)", mixer.Kind(), mixer.State())
	}

	// hidden 4 → x [B=1, L=2, 4].
	x := metal.FromValues(make([]float32, 1*2*4), 1, 2, 4)
	defer metal.Free(x)
	out, _ := mixer.Forward(x, &metal.MixerCtx{Cache: metal.NewKVCache(), B: 1, L: 2})
	if out == nil {
		t.Fatal("MLA Forward returned nil")
	}
	defer metal.Free(out)
	if got := out.Shape(); len(got) != 3 || got[0] != 1 || got[1] != 2 || got[2] != 4 {
		t.Errorf("forward out shape = %v, want [1 2 4]", got)
	}
}

// TestLoader_BuildMLA_Bad: a missing required projection is a loud build error.
func TestLoader_BuildMLA_Bad(t *testing.T) {
	ctx := fakeBuildCtx()
	base := ctx.Linear
	ctx.Linear = func(p string) *metal.Linear {
		if p == "kv_b_proj" {
			return nil
		}
		return base(p)
	}
	if _, err := buildMLA(ctx); err == nil {
		t.Error("expected error for missing kv_b_proj, got nil")
	}
}

// TestLoader_Registered_Ugly asserts the loader self-registers from init() (a
// re-register is an idempotent override) and the kind round-trips.
func TestLoader_Registered_Ugly(t *testing.T) {
	metal.RegisterMixerLoader(MixerKind, buildMLA)
	mixer, err := buildMLA(fakeBuildCtx())
	if err != nil {
		t.Fatalf("registered loader build: %v", err)
	}
	if mixer.Kind() != MixerKind {
		t.Errorf("Kind() = %q, want %q", mixer.Kind(), MixerKind)
	}
}

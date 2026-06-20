// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package mla

import (
	"math"
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

// scalarLinear builds a Linear whose weight is a 0-dim scalar array — Valid()
// but with an empty Shape(), so linearOutDim reads no out-dim from it. The
// builder then has no kv_b_proj-inferred head dim and must fall back to the
// config's HeadDim (or error when that is also unset).
func scalarLinear() *metal.Linear {
	return metal.NewLinear(metal.FromValue(float32(1)), nil)
}

// TestLoader_BuildMLA_HeadDimError_Bad covers the head-dim resolution failure:
// every projection is present (so the missing-projection guard passes) but none
// pins a usable per-head dim — kv_b_proj is a scalar (linearOutDim → 0, so the
// 2*heads divisibility inference is skipped) and the config carries HeadDim 0.
// With neither source yielding a positive head dim the builder errors loudly
// rather than constructing a zero-geometry mixer.
func TestLoader_BuildMLA_HeadDimError_Bad(t *testing.T) {
	requireMetalRuntime(t)

	ctx := metal.MixerBuildCtx{
		Linear: func(string) *metal.Linear { return scalarLinear() },
		Weight: func(string) *metal.Array { return nil },
		Cfg:    metal.TransformerConfig{HiddenSize: 4, NumAttentionHeads: 2, HeadDim: 0},
	}
	if _, err := buildMLA(ctx); err == nil {
		t.Error("expected head-dim error when kv_b_proj and config both fail to pin a head dim, got nil")
	}
}

// TestHeadCount_Fallback_Ugly pins headCount's malformed-config fallback: a
// config with no positive attention-head count resolves to 1 (so downstream
// shape checks fail loudly instead of dividing by zero), while a populated
// config passes its value through.
func TestHeadCount_Fallback_Ugly(t *testing.T) {
	if got := headCount(metal.TransformerConfig{NumAttentionHeads: 0}); got != 1 {
		t.Errorf("headCount(0) = %d, want 1 (fallback)", got)
	}
	if got := headCount(metal.TransformerConfig{NumAttentionHeads: -3}); got != 1 {
		t.Errorf("headCount(-3) = %d, want 1 (fallback)", got)
	}
	if got := headCount(metal.TransformerConfig{NumAttentionHeads: 7}); got != 7 {
		t.Errorf("headCount(7) = %d, want 7 (passthrough)", got)
	}
}

// TestLinearOutDim_Absent_Ugly covers linearOutDim's two zero-returns: a nil
// Linear (or one with a nil/invalid weight) yields 0, and a Valid weight with
// an empty shape (a 0-dim scalar) also yields 0 — neither pins a head dim.
func TestLinearOutDim_Absent_Ugly(t *testing.T) {
	requireMetalRuntime(t)

	if got := linearOutDim(nil); got != 0 {
		t.Errorf("linearOutDim(nil) = %d, want 0", got)
	}
	if got := linearOutDim(metal.NewLinear(nil, nil)); got != 0 {
		t.Errorf("linearOutDim(nil-weight) = %d, want 0", got)
	}

	scalar := scalarLinear()
	defer metal.FreeLinear(scalar)
	if !scalar.Weight.Valid() {
		t.Fatal("scalar weight should be Valid (only its shape is empty)")
	}
	if got := linearOutDim(scalar); got != 0 {
		t.Errorf("linearOutDim(scalar) = %d, want 0 (empty shape)", got)
	}

	// Sanity: a normal [out,in] weight does report its out-dim.
	l := metal.NewLinear(metal.FromValues(make([]float32, 8*6), 8, 6), nil)
	defer metal.FreeLinear(l)
	if got := linearOutDim(l); got != 8 {
		t.Errorf("linearOutDim([8,6]) = %d, want 8", got)
	}
}

// TestAttnScale_Fallback_Ugly pins attnScale's guard: a non-positive head dim
// returns the neutral scale 1 (no 1/sqrt(0) division), while a positive head
// dim returns 1/sqrt(dim).
func TestAttnScale_Fallback_Ugly(t *testing.T) {
	if got := attnScale(0); got != 1 {
		t.Errorf("attnScale(0) = %v, want 1 (guard)", got)
	}
	if got := attnScale(-4); got != 1 {
		t.Errorf("attnScale(-4) = %v, want 1 (guard)", got)
	}
	if got := attnScale(4); math.Abs(float64(got-0.5)) > 1e-6 {
		t.Errorf("attnScale(4) = %v, want 0.5 (1/sqrt(4))", got)
	}
}

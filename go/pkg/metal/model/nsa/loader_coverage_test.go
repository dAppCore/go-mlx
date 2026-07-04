// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package nsa

import (
	"math"
	"testing"

	metal "dappco.re/go/mlx/pkg/metal"
)

// loader_coverage_test.go pins the geometry-inference fallbacks in loader.go
// (headCount, headDimFrom, linearOutDim, attnScale) and the buildNSA error path
// taken when no head dim can be derived. These are pure-Go arithmetic over
// config + Linear metadata, so they need no Metal runtime gate — except where a
// *metal.Array must be constructed, which still works without a live device.
// Names are suffixed _Cov to avoid clashing with the existing loader_test.go
// fixtures (fakeBuildCtx, identityLinear) which these tests reuse.

// TestHeadCount_FallbackToOne_Cov: NumAttentionHeads <= 0 → headCount returns 1
// (the safe single-head default the loader uses when a config omits the field).
func TestHeadCount_FallbackToOne_Cov(t *testing.T) {
	zero := metal.TransformerConfig{NumAttentionHeads: 0}
	if got := headCount(zero); got != 1 {
		t.Errorf("headCount(0 heads) = %d, want 1", got)
	}
	neg := metal.TransformerConfig{NumAttentionHeads: -4}
	if got := headCount(neg); got != 1 {
		t.Errorf("headCount(-4 heads) = %d, want 1", got)
	}
	// And the populated path still wins (guards the fallback didn't shadow it).
	if got := headCount(metal.TransformerConfig{NumAttentionHeads: 7}); got != 7 {
		t.Errorf("headCount(7 heads) = %d, want 7", got)
	}
}

// TestHeadDimFrom_ConfigHeadDimFallback_Cov: when q_proj yields no usable out
// dim (nil proj here), headDimFrom falls back to cfg.HeadDim.
func TestHeadDimFrom_ConfigHeadDimFallback_Cov(t *testing.T) {
	cfg := metal.TransformerConfig{HeadDim: 5, HiddenSize: 64}
	// qProj nil → linearOutDim 0 → skip the q_proj branch → use cfg.HeadDim.
	if got := headDimFrom(cfg, nil, 2); got != 5 {
		t.Errorf("headDimFrom(nil qProj, HeadDim 5) = %d, want 5", got)
	}
}

// TestHeadDimFrom_HiddenOverHeadsFallback_Cov: no q_proj out, no cfg.HeadDim,
// but HiddenSize/heads is valid → headDimFrom returns hidden/heads.
func TestHeadDimFrom_HiddenOverHeadsFallback_Cov(t *testing.T) {
	cfg := metal.TransformerConfig{HeadDim: 0, HiddenSize: 64}
	if got := headDimFrom(cfg, nil, 8); got != 8 { // 64/8
		t.Errorf("headDimFrom(hidden 64 / 8 heads) = %d, want 8", got)
	}
}

// TestHeadDimFrom_AllUnknown_Cov: nothing resolves the head dim (no q_proj out,
// no cfg.HeadDim, no HiddenSize) → headDimFrom returns 0.
func TestHeadDimFrom_AllUnknown_Cov(t *testing.T) {
	cfg := metal.TransformerConfig{HeadDim: 0, HiddenSize: 0}
	if got := headDimFrom(cfg, nil, 2); got != 0 {
		t.Errorf("headDimFrom(all unknown) = %d, want 0", got)
	}
	// heads <= 0 also drives the hidden/heads branch off even with HiddenSize set.
	cfg2 := metal.TransformerConfig{HeadDim: 0, HiddenSize: 64}
	if got := headDimFrom(cfg2, nil, 0); got != 0 {
		t.Errorf("headDimFrom(0 heads, no HeadDim) = %d, want 0", got)
	}
}

// TestHeadDimFrom_QProjWins_Cov: q_proj.out / heads (the primary path) takes
// precedence over both fallbacks when out%heads == 0. Guards that the fallbacks
// added above don't shadow the q_proj inference.
func TestHeadDimFrom_QProjWins_Cov(t *testing.T) {
	q := identityLinear(8) // weight [8,8] → out dim 8
	defer metal.FreeLinear(q)
	cfg := metal.TransformerConfig{HeadDim: 99, HiddenSize: 999} // would mislead if used
	if got := headDimFrom(cfg, q, 4); got != 2 {                 // 8/4, not 99
		t.Errorf("headDimFrom(q_proj out 8 / 4 heads) = %d, want 2", got)
	}
}

// TestLinearOutDim_NilAndInvalid_Cov: a nil Linear, a Linear with a nil Weight,
// all return 0 from linearOutDim (the guard before reading the shape).
func TestLinearOutDim_NilAndInvalid_Cov(t *testing.T) {
	if got := linearOutDim(nil); got != 0 {
		t.Errorf("linearOutDim(nil) = %d, want 0", got)
	}
	nilWeight := metal.NewLinear(nil, nil) // Weight == nil → guard returns 0
	if got := linearOutDim(nilWeight); got != 0 {
		t.Errorf("linearOutDim(nil weight) = %d, want 0", got)
	}
}

// TestLinearOutDim_RankZeroShape_Cov: a Linear whose Weight is a valid rank-0
// scalar array (len(shape) == 0) returns 0 — the empty-shape guard, distinct
// from the nil/invalid guard above.
func TestLinearOutDim_RankZeroShape_Cov(t *testing.T) {
	scalar := metal.FromValue(float32(1)) // valid, rank 0 → Shape() len 0
	defer metal.Free(scalar)
	l := metal.NewLinear(scalar, nil)
	if got := linearOutDim(l); got != 0 {
		t.Errorf("linearOutDim(rank-0 weight) = %d, want 0", got)
	}
}

// TestAttnScale_NonPositiveDim_Cov: headDim <= 0 → attnScale returns 1 (the
// degenerate guard); a positive dim returns 1/sqrt(dim).
func TestAttnScale_NonPositiveDim_Cov(t *testing.T) {
	if got := attnScale(0); got != 1 {
		t.Errorf("attnScale(0) = %f, want 1", got)
	}
	if got := attnScale(-3); got != 1 {
		t.Errorf("attnScale(-3) = %f, want 1", got)
	}
	want := float32(1.0 / math.Sqrt(4))
	if got := attnScale(4); math.Abs(float64(got-want)) > 1e-6 {
		t.Errorf("attnScale(4) = %f, want %f", got, want)
	}
}

// TestBuildNSA_NoHeadDim_Bad: when neither q_proj nor the config can determine a
// head dim, buildNSA returns the "cannot determine NSA head dim" error rather
// than constructing a degenerate mixer. A rank-0 q_proj weight (out dim 0) plus
// a config with no HeadDim / HiddenSize starves every headDimFrom branch.
func TestBuildNSA_NoHeadDim_Bad(t *testing.T) {
	scalar := metal.FromValue(float32(1)) // rank-0 → linearOutDim 0
	linears := map[string]*metal.Linear{
		"q_proj": metal.NewLinear(scalar, nil),
		"k_proj": metal.NewLinear(scalar, nil),
		"v_proj": metal.NewLinear(scalar, nil),
		"g_proj": metal.NewLinear(scalar, nil),
		"o_proj": metal.NewLinear(scalar, nil),
	}
	defer metal.Free(scalar)
	ctx := metal.MixerBuildCtx{
		Linear: func(p string) *metal.Linear { return linears[p] },
		Weight: func(string) *metal.Array { return nil },
		// No HeadDim, no HiddenSize, no heads → headDimFrom returns 0.
		Cfg:      metal.TransformerConfig{},
		LayerIdx: 0,
	}
	if _, err := buildNSA(ctx); err == nil {
		t.Error("expected error when head dim cannot be determined, got nil")
	}
}

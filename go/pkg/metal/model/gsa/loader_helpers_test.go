// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gsa

import (
	"testing"

	metal "dappco.re/go/mlx/pkg/metal"
)

// loader_helpers_test.go pins the geometry-inference helpers that buildGSA leans
// on (headCount, headDimFrom, linearOutDim, gateNorm) and the loud build error
// when the geometry cannot be resolved. These are pure Go config/shape readers —
// no Metal runtime — so they run without the metal_runtime tag, exactly like
// loader_test.go. The fakeBuildCtx fixture in loader_test.go always supplies a
// clean q_proj/f_proj geometry, leaving every config-fallback and degenerate
// branch here uncovered; these tests reach them deterministically.

// invalidWeightLinear returns a Linear whose weight is freed, so Weight.Valid()
// reports false — the linearOutDim guard for a stale/invalid projection weight.
func invalidWeightLinear() *metal.Linear {
	w := metal.FromValues([]float32{1, 2}, 2, 1)
	l := metal.NewLinear(w, nil)
	metal.Free(w)
	return l
}

// scalarWeightLinear returns a Linear whose weight is a valid 0-rank (empty
// shape) array — the linearOutDim guard for a shapeless projection weight.
func scalarWeightLinear() *metal.Linear {
	return metal.NewLinear(metal.Zeros([]int32{}, metal.DTypeFloat32), nil)
}

// TestLinearOutDim_Branches_Good walks every linearOutDim exit: nil Linear, nil
// weight, invalid (freed) weight, empty-shape weight, and the happy path that
// returns shape[0].
func TestLinearOutDim_Branches_Good(t *testing.T) {
	if got := linearOutDim(nil); got != 0 {
		t.Errorf("linearOutDim(nil) = %d, want 0", got)
	}
	if got := linearOutDim(metal.NewLinear(nil, nil)); got != 0 {
		t.Errorf("linearOutDim(nil-weight) = %d, want 0", got)
	}

	inv := invalidWeightLinear()
	if got := linearOutDim(inv); got != 0 {
		t.Errorf("linearOutDim(freed-weight) = %d, want 0", got)
	}

	sc := scalarWeightLinear()
	defer metal.FreeLinear(sc)
	if got := linearOutDim(sc); got != 0 {
		t.Errorf("linearOutDim(empty-shape) = %d, want 0", got)
	}

	ok := metal.NewLinear(metal.FromValues(make([]float32, 6*4), 6, 4), nil)
	defer metal.FreeLinear(ok)
	if got := linearOutDim(ok); got != 6 {
		t.Errorf("linearOutDim([6,4]) = %d, want 6 (out features = shape[0])", got)
	}
}

// TestHeadCount_Fallback_Good pins headCount: a configured head count is honoured
// and a missing one (<= 0) falls back to a single head.
func TestHeadCount_Fallback_Good(t *testing.T) {
	if got := headCount(metal.TransformerConfig{NumAttentionHeads: 8}); got != 8 {
		t.Errorf("headCount(8) = %d, want 8", got)
	}
	if got := headCount(metal.TransformerConfig{NumAttentionHeads: 0}); got != 1 {
		t.Errorf("headCount(unset) = %d, want 1 (single-head fallback)", got)
	}
	if got := headCount(metal.TransformerConfig{NumAttentionHeads: -3}); got != 1 {
		t.Errorf("headCount(negative) = %d, want 1 (single-head fallback)", got)
	}
}

// TestHeadDimFrom_AllPaths_Good walks every headDimFrom resolution rung in order:
// q_proj.out/heads (preferred), then cfg.HeadDim, then cfg.HiddenSize/heads, then
// 0. Each lower rung is reached by starving the ones above it (a nil q_proj kills
// the projection rung; HeadDim=0 kills the HeadDim rung; HiddenSize=0 the last).
func TestHeadDimFrom_AllPaths_Good(t *testing.T) {
	heads := int32(2)

	// Rung 1: q_proj.out (=8) / heads (=2) = 4 — wins over HeadDim/HiddenSize.
	q := metal.NewLinear(metal.FromValues(make([]float32, 8*4), 8, 4), nil)
	defer metal.FreeLinear(q)
	cfg := metal.TransformerConfig{HeadDim: 99, HiddenSize: 64}
	if got := headDimFrom(cfg, q, heads); got != 4 {
		t.Errorf("headDimFrom(q_proj path) = %d, want 4 (q.out/heads)", got)
	}

	// q_proj present but out (=9) not divisible by heads (=2) → skip rung 1,
	// fall to cfg.HeadDim.
	qOdd := metal.NewLinear(metal.FromValues(make([]float32, 9*4), 9, 4), nil)
	defer metal.FreeLinear(qOdd)
	if got := headDimFrom(metal.TransformerConfig{HeadDim: 7}, qOdd, heads); got != 7 {
		t.Errorf("headDimFrom(indivisible q.out → HeadDim) = %d, want 7", got)
	}

	// Rung 2: no usable q_proj (nil) → cfg.HeadDim (=7).
	if got := headDimFrom(metal.TransformerConfig{HeadDim: 7, HiddenSize: 64}, nil, heads); got != 7 {
		t.Errorf("headDimFrom(HeadDim path) = %d, want 7", got)
	}

	// Rung 3: no q_proj, HeadDim=0 → cfg.HiddenSize (=64) / heads (=2) = 32.
	if got := headDimFrom(metal.TransformerConfig{HeadDim: 0, HiddenSize: 64}, nil, heads); got != 32 {
		t.Errorf("headDimFrom(HiddenSize path) = %d, want 32", got)
	}

	// Rung 4: nothing resolvable → 0.
	if got := headDimFrom(metal.TransformerConfig{HeadDim: 0, HiddenSize: 0}, nil, heads); got != 0 {
		t.Errorf("headDimFrom(unresolvable) = %d, want 0", got)
	}

	// HiddenSize set but heads == 0 → the HiddenSize rung guards on heads>0, so
	// it is skipped and the result is 0.
	if got := headDimFrom(metal.TransformerConfig{HiddenSize: 64}, nil, 0); got != 0 {
		t.Errorf("headDimFrom(HiddenSize, heads=0) = %d, want 0", got)
	}
}

// TestGateNorm_Fallback_Good pins gateNorm: a positive slot count is the
// normaliser; a non-positive count falls back to 1 (no divide-by-zero).
func TestGateNorm_Fallback_Good(t *testing.T) {
	if got := gateNorm(4); got != 4 {
		t.Errorf("gateNorm(4) = %g, want 4", got)
	}
	if got := gateNorm(0); got != 1 {
		t.Errorf("gateNorm(0) = %g, want 1 (fallback)", got)
	}
	if got := gateNorm(-2); got != 1 {
		t.Errorf("gateNorm(-2) = %g, want 1 (fallback)", got)
	}
}

// TestBuildGSA_UnresolvableGeometry_Bad drives buildGSA through to its geometry
// guard: all six projections are present (so the missing-projection guard
// passes) but f_proj has a degenerate output that yields slots == 0, so the
// builder cannot determine the slot geometry and must return a loud error rather
// than a half-built mixer.
func TestBuildGSA_UnresolvableGeometry_Bad(t *testing.T) {
	// f_proj out (=3) is not divisible by heads (=2) → slots stays 0; every
	// other projection is well-formed, isolating the geometry-failure exit.
	linears := map[string]*metal.Linear{
		"q_proj": metal.NewLinear(metal.FromValues(make([]float32, 4*4), 4, 4), nil),
		"k_proj": metal.NewLinear(metal.FromValues(make([]float32, 4*4), 4, 4), nil),
		"v_proj": metal.NewLinear(metal.FromValues(make([]float32, 4*4), 4, 4), nil),
		"f_proj": metal.NewLinear(metal.FromValues(make([]float32, 3*4), 3, 4), nil),
		"g_proj": metal.NewLinear(metal.FromValues(make([]float32, 4*4), 4, 4), nil),
		"o_proj": metal.NewLinear(metal.FromValues(make([]float32, 4*4), 4, 4), nil),
	}
	defer func() {
		for _, l := range linears {
			metal.FreeLinear(l)
		}
	}()
	ctx := metal.MixerBuildCtx{
		Linear: func(p string) *metal.Linear { return linears[p] },
		Weight: func(string) *metal.Array { return nil },
		Cfg:    metal.TransformerConfig{HiddenSize: 4, NumAttentionHeads: 2, HeadDim: 2, RMSNormEps: 1e-5},
	}
	mixer, err := buildGSA(ctx)
	if err == nil {
		t.Fatalf("expected geometry error for indivisible f_proj, got mixer %+v", mixer)
	}
}

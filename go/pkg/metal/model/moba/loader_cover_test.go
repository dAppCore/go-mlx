// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package moba

import (
	"math"
	"testing"

	metal "dappco.re/go/mlx/pkg/metal"
)

// loader_cover_test.go pins the pure-Go geometry helpers (headCount, headDimFrom,
// linearOutDim, attnScale) and buildMoBA's head-dim error path on their fallback
// branches — the cases the happy-path fakeBuildCtx never reaches (no q_proj
// shape, no config geometry, degenerate dims). These are config arithmetic, so
// they need no Metal runtime: they construct *metal.Array / *metal.Linear
// metadata and read it, exactly as the un-gated loader_test.go does.

// TestHeadCount_DefaultsToOne_Good pins the headCount fallback: a config with no
// declared attention heads (NumAttentionHeads == 0) yields a single head, never
// 0 (which would make the head-dim division blow up downstream).
func TestHeadCount_DefaultsToOne_Good(t *testing.T) {
	if got := headCount(metal.TransformerConfig{}); got != 1 {
		t.Errorf("headCount(zero cfg) = %d, want 1", got)
	}
	if got := headCount(metal.TransformerConfig{NumAttentionHeads: 8}); got != 8 {
		t.Errorf("headCount(8 heads) = %d, want 8", got)
	}
}

// TestLinearOutDim_Degenerate_Good pins the two guard branches of linearOutDim:
// a nil/invalid Linear yields 0, and a valid-but-scalar weight (0-dim shape)
// also yields 0 rather than indexing a non-existent shape[0]. A scalar built
// with FromValue is Valid() (non-nil ctx) yet has an empty Shape(), so it is the
// exact constructor for the len(shape)==0 guard.
func TestLinearOutDim_Degenerate_Good(t *testing.T) {
	if got := linearOutDim(nil); got != 0 {
		t.Errorf("linearOutDim(nil) = %d, want 0", got)
	}
	if got := linearOutDim(&metal.Linear{}); got != 0 {
		t.Errorf("linearOutDim(empty Linear) = %d, want 0", got)
	}

	scalar := metal.FromValue(float32(0))
	defer metal.Free(scalar)
	if !scalar.Valid() {
		t.Fatal("FromValue scalar is not Valid — cannot exercise the len(shape)==0 guard")
	}
	if got := len(scalar.Shape()); got != 0 {
		t.Fatalf("FromValue scalar Shape() len = %d, want 0 (need a 0-dim array for the guard)", got)
	}
	scalarLinear := metal.NewLinear(scalar, nil)
	if got := linearOutDim(scalarLinear); got != 0 {
		t.Errorf("linearOutDim(scalar-weight Linear) = %d, want 0", got)
	}
}

// TestHeadDimFrom_ConfigFallbacks_Good pins headDimFrom's three fallbacks when
// the q_proj weight gives no usable out-dim (nil q_proj → linearOutDim 0):
//   - HeadDim set on the config wins next,
//   - else HiddenSize / heads,
//   - else 0 (nothing to go on).
func TestHeadDimFrom_ConfigFallbacks_Good(t *testing.T) {
	// q_proj is nil → linearOutDim 0; config HeadDim carries the value.
	if got := headDimFrom(metal.TransformerConfig{HeadDim: 5}, nil, 2); got != 5 {
		t.Errorf("headDimFrom(HeadDim=5) = %d, want 5", got)
	}
	// No HeadDim → HiddenSize / heads.
	if got := headDimFrom(metal.TransformerConfig{HiddenSize: 12}, nil, 3); got != 4 {
		t.Errorf("headDimFrom(HiddenSize=12, heads=3) = %d, want 4", got)
	}
	// Nothing usable → 0.
	if got := headDimFrom(metal.TransformerConfig{}, nil, 0); got != 0 {
		t.Errorf("headDimFrom(empty cfg, 0 heads) = %d, want 0", got)
	}
	// HiddenSize set but heads 0 → division guarded → 0.
	if got := headDimFrom(metal.TransformerConfig{HiddenSize: 12}, nil, 0); got != 0 {
		t.Errorf("headDimFrom(HiddenSize=12, 0 heads) = %d, want 0", got)
	}
}

// TestAttnScale_Degenerate_Good pins attnScale's headDim<=0 guard (returns 1, no
// divide-by-zero / sqrt of non-positive) alongside a normal value.
func TestAttnScale_Degenerate_Good(t *testing.T) {
	if got := attnScale(0); got != 1 {
		t.Errorf("attnScale(0) = %f, want 1", got)
	}
	if got := attnScale(-4); got != 1 {
		t.Errorf("attnScale(-4) = %f, want 1", got)
	}
	if got := attnScale(4); math.Abs(float64(got)-0.5) > 1e-6 {
		t.Errorf("attnScale(4) = %f, want 0.5", got)
	}
}

// TestBuildMoBA_NoHeadDim_Bad pins buildMoBA's headDim<=0 error path: all four
// projections are present (so the missing-projection guard passes) but each has
// a scalar 0-dim weight, so linearOutDim is 0 and the config carries no HeadDim
// / HiddenSize — buildMoBA cannot determine the head dim and must fail loudly.
func TestBuildMoBA_NoHeadDim_Bad(t *testing.T) {
	scalarLinear := func() *metal.Linear { return metal.NewLinear(metal.FromValue(float32(0)), nil) }
	linears := map[string]*metal.Linear{
		"q_proj": scalarLinear(), "k_proj": scalarLinear(),
		"v_proj": scalarLinear(), "o_proj": scalarLinear(),
	}
	ctx := metal.MixerBuildCtx{
		Linear: func(p string) *metal.Linear { return linears[p] },
		Weight: func(string) *metal.Array { return nil },
		Cfg:    metal.TransformerConfig{}, // no HeadDim, no HiddenSize → head dim unknowable
	}
	if _, err := buildMoBA(ctx); err == nil {
		t.Error("expected error when head dim cannot be determined, got nil")
	}
}

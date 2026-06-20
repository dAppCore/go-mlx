// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package flakernel

import (
	"math"
	"testing"

	"dappco.re/go/mlx/internal/metaltest"
	metal "dappco.re/go/mlx/pkg/metal"
)

// flakernel_test.go drives every tensor helper in flakernel.go on the real Metal
// runtime and asserts both the output shape and the host-readback values against
// a CPU reference, so the tests prove the kernels compute the documented quantity
// rather than merely executing. The recurrent-slot helpers are exercised across
// all of their nil/range branches with a live RecurrentCache.

const fkTol = 1e-5

// requireMetalRuntime skips when the binary was not built with -tags
// metal_runtime or no Metal device is present — mirrors the sibling mixer suites.
func requireMetalRuntime(t testing.TB) {
	t.Helper()
	if !metaltest.RunMetalTests {
		t.Skip("build with -tags metal_runtime to enable Metal runtime tests")
	}
	if !metal.MetalAvailable() {
		t.Skip("Metal runtime unavailable")
	}
}

// closeFloats fails when got/want differ in length or any element drifts past tol.
func closeFloats(t *testing.T, name string, got, want []float32, tol float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: length %d, want %d", name, len(got), len(want))
	}
	for i := range want {
		if d := got[i] - want[i]; d > tol || d < -tol {
			t.Fatalf("%s[%d] = %g, want %g (Δ=%g > %g)", name, i, got[i], want[i], d, tol)
		}
	}
}

// TestLowerTriangle_Good proves the [L,L] keep-mask is 1.0 on and below the main
// diagonal and 0.0 strictly above.
func TestLowerTriangle_Good(t *testing.T) {
	requireMetalRuntime(t)
	const l = 4
	keep := LowerTriangle(l)
	defer metal.Free(keep)
	if keep.Dim(0) != l || keep.Dim(1) != l {
		t.Fatalf("shape = [%d,%d], want [%d,%d]", keep.Dim(0), keep.Dim(1), l, l)
	}
	want := make([]float32, l*l)
	for i := 0; i < l; i++ {
		for j := 0; j < l; j++ {
			if i >= j {
				want[i*l+j] = 1
			}
		}
	}
	closeFloats(t, "LowerTriangle", keep.Floats(), want, fkTol)
}

// TestPerHeadDecay_Good proves γ_h^(i-j) on/below the diagonal, 0 above, per head.
func TestPerHeadDecay_Good(t *testing.T) {
	requireMetalRuntime(t)
	const l = 3
	lnGamma := []float32{float32(math.Log(0.5)), float32(math.Log(0.9))}
	h := len(lnGamma)
	d := PerHeadDecay(lnGamma, l)
	defer metal.Free(d)
	if d.Dim(0) != h || d.Dim(1) != l || d.Dim(2) != l {
		t.Fatalf("shape = [%d,%d,%d], want [%d,%d,%d]", d.Dim(0), d.Dim(1), d.Dim(2), h, l, l)
	}
	want := make([]float32, h*l*l)
	for hi := 0; hi < h; hi++ {
		g := math.Exp(float64(lnGamma[hi]))
		for i := 0; i < l; i++ {
			for j := 0; j < l; j++ {
				if i >= j {
					want[hi*l*l+i*l+j] = float32(math.Pow(g, float64(i-j)))
				}
			}
		}
	}
	closeFloats(t, "PerHeadDecay", d.Floats(), want, 1e-4)
}

// TestMulHeadBroadcast_Good proves a [H,L,L] mask multiplies a [B,H,L,L] score
// tensor, broadcasting across the batch.
func TestMulHeadBroadcast_Good(t *testing.T) {
	requireMetalRuntime(t)
	const (
		b = 2
		h = 2
		l = 2
	)
	scoresN := b * h * l * l
	sv := make([]float32, scoresN)
	for i := range sv {
		sv[i] = float32(i) + 1
	}
	scores := metal.FromValues(sv, b, h, l, l)
	defer metal.Free(scores)
	mv := make([]float32, h*l*l)
	for i := range mv {
		mv[i] = 0.5 * float32(i+1)
	}
	mask := metal.FromValues(mv, h, l, l)
	defer metal.Free(mask)

	out := MulHeadBroadcast(scores, mask)
	defer metal.Free(out)
	if out.Dim(0) != b || out.Dim(1) != h || out.Dim(2) != l || out.Dim(3) != l {
		t.Fatalf("shape = %v, want [%d,%d,%d,%d]", out.Shape(), b, h, l, l)
	}
	want := make([]float32, scoresN)
	for bi := 0; bi < b; bi++ {
		for k := 0; k < h*l*l; k++ {
			want[bi*h*l*l+k] = sv[bi*h*l*l+k] * mv[k]
		}
	}
	closeFloats(t, "MulHeadBroadcast", out.Floats(), want, 1e-4)
}

// TestMulCausalBroadcast_Good proves a [L,L] mask multiplies a [B,H,L,L] score
// tensor, broadcasting across both batch and head.
func TestMulCausalBroadcast_Good(t *testing.T) {
	requireMetalRuntime(t)
	const (
		b = 2
		h = 2
		l = 2
	)
	scoresN := b * h * l * l
	sv := make([]float32, scoresN)
	for i := range sv {
		sv[i] = float32(i) + 1
	}
	scores := metal.FromValues(sv, b, h, l, l)
	defer metal.Free(scores)
	mv := []float32{1, 0, 1, 1} // [L,L] causal triangle
	mask := metal.FromValues(mv, l, l)
	defer metal.Free(mask)

	out := MulCausalBroadcast(scores, mask)
	defer metal.Free(out)
	if out.Dim(0) != b || out.Dim(3) != l {
		t.Fatalf("shape = %v, want [%d,%d,%d,%d]", out.Shape(), b, h, l, l)
	}
	want := make([]float32, scoresN)
	for idx := 0; idx < scoresN; idx++ {
		want[idx] = sv[idx] * mv[idx%(l*l)]
	}
	closeFloats(t, "MulCausalBroadcast", out.Floats(), want, 1e-4)
}

// TestL2NormalizeLastAxis_Good proves each last-axis vector is scaled to unit L2
// length (x / sqrt(Σx²+eps)).
func TestL2NormalizeLastAxis_Good(t *testing.T) {
	requireMetalRuntime(t)
	const (
		rows = 2
		dim  = 3
	)
	const eps = 1e-6
	xv := []float32{3, 0, 4, 1, 2, 2} // ‖·‖ = 5 and 3
	x := metal.FromValues(xv, rows, dim)
	defer metal.Free(x)
	out := L2NormalizeLastAxis(x, eps)
	defer metal.Free(out)
	if out.Dim(0) != rows || out.Dim(1) != dim {
		t.Fatalf("shape = %v, want [%d,%d]", out.Shape(), rows, dim)
	}
	want := make([]float32, rows*dim)
	for r := 0; r < rows; r++ {
		var ss float64
		for c := 0; c < dim; c++ {
			ss += float64(xv[r*dim+c]) * float64(xv[r*dim+c])
		}
		inv := 1.0 / math.Sqrt(ss+eps)
		for c := 0; c < dim; c++ {
			want[r*dim+c] = float32(float64(xv[r*dim+c]) * inv)
		}
	}
	closeFloats(t, "L2NormalizeLastAxis", out.Floats(), want, 1e-5)
}

// TestDefaultScale_Good covers both branches: 1/sqrt(headDim) for a positive dim
// and the 1.0 fallback for a non-positive dim.
func TestDefaultScale_Good(t *testing.T) {
	if got := DefaultScale(64); math.Abs(float64(got)-1.0/8.0) > 1e-7 {
		t.Errorf("DefaultScale(64) = %g, want %g", got, 1.0/8.0)
	}
	if got := DefaultScale(0); got != 1 {
		t.Errorf("DefaultScale(0) = %g, want 1", got)
	}
	if got := DefaultScale(-4); got != 1 {
		t.Errorf("DefaultScale(-4) = %g, want 1", got)
	}
}

// TestPriorRecurrentSlot_Branches covers all three exits: no holder, a valid
// slot, and out-of-range indices (negative and past the end).
func TestPriorRecurrentSlot_Branches(t *testing.T) {
	requireMetalRuntime(t)

	// No holder (nil cache → Recurrent() reports false).
	if got := PriorRecurrentSlot(&metal.MixerCtx{}, 0); got != nil {
		t.Error("no-holder ctx returned a non-nil slot")
	}

	cache := metal.NewRecurrentCache()
	ctx := &metal.MixerCtx{Cache: cache}

	// Holder present but empty → any index is out of range.
	if got := PriorRecurrentSlot(ctx, 0); got != nil {
		t.Error("empty holder returned a non-nil slot")
	}

	slot0 := metal.FromValues([]float32{1, 2, 3, 4}, 1, 1, 2, 2)
	cache.SetRecurrentState([]*metal.Array{slot0})

	if got := PriorRecurrentSlot(ctx, 0); got != slot0 {
		t.Error("slot 0 not returned for a populated single-slot holder")
	}
	if got := PriorRecurrentSlot(ctx, -1); got != nil {
		t.Error("negative index did not return nil")
	}
	if got := PriorRecurrentSlot(ctx, 1); got != nil {
		t.Error("past-end index did not return nil")
	}
}

// TestStoreRecurrentState_Branches covers the nil-state no-op, the no-holder Free
// path, and the holder-store path (which adopts the state for the next forward).
func TestStoreRecurrentState_Branches(t *testing.T) {
	requireMetalRuntime(t)

	// nil state → no-op, no holder touched.
	StoreRecurrentState(&metal.MixerCtx{}, nil)

	// Non-nil state, no holder → the helper frees it (cannot be carried). Build a
	// real array so the Free path runs against a valid handle.
	orphan := metal.FromValues([]float32{1, 2}, 2)
	StoreRecurrentState(&metal.MixerCtx{}, orphan)
	if orphan.Valid() {
		t.Error("no-holder store did not free the un-carryable state")
	}

	// Holder present → the state is adopted as slot 0.
	cache := metal.NewRecurrentCache()
	ctx := &metal.MixerCtx{Cache: cache}
	next := metal.FromValues([]float32{5, 6, 7, 8}, 1, 1, 2, 2)
	StoreRecurrentState(ctx, next)
	slots := cache.RecurrentState()
	if len(slots) != 1 || slots[0] != next {
		t.Fatalf("holder carries %d slots; slot 0 == next: %v", len(slots), len(slots) == 1 && slots[0] == next)
	}
	metal.Free(next)
}

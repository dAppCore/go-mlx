// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package rwkv7

import (
	"testing"

	metal "dappco.re/go/mlx/pkg/metal"
)

// coverage_test.go pins the boundary and dtype-edge paths the math-oracle tests
// (chunk_test.go / recurrence_test.go / forward_test.go) don't reach: the
// empty-window state plumbing, the width-default guard, the bf16 LAPACK
// round-trip in solve, and the loader/forward early-return validation. These are
// the not-the-happy-path branches, exercised on synthetic configs (no real
// model), keeping the package's correctness gate honest at the edges.
//
// The bf16 round-trip tests exercise solve's working-dtype != float32 cast-back
// (the castBack branch): with bf16 inputs every per-head op stays bf16 through
// the recurrence, so castBack casts the f32 triangular inverse back to bf16.
// asFloat32's up-cast is covered by the direct helper test (TestCoverage_Dtype-
// Helpers) — it is NOT reachable through solve even on bf16 inputs, because the
// keep-masks are always float32 and mixed-dtype Mul promotes the masked products,
// so `I − A_ab` reaches asFloat32 already float32. The one statement that stays
// dead from solve is the imCast free at the solve call-site: it only frees when
// asFloat32 up-cast inside solve, which (per the mask promotion above) never does.
//
// NOTE on the unreachable defensive guards (tests-only ceiling ~95.5%): solve's
// `metal.TriInv` failure return (and every nil-output cascade fed by it —
// compute's outW==nil, the wkv7ChunkedWindowed / Mixer.Forward o==nil paths), and
// the three metal.Eval error branches (chunk / mixer / recurrence) are guards
// against a C-level MLX fault. MLX aborts the process with an uncaught
// std::runtime_error on a malformed LAPACK input (verified: a non-unit-triangular
// batched matrix calls std::terminate, it does not return rc!=0), and Eval
// likewise crashes rather than returning a non-nil error on a well-formed lazy
// graph. compute's o==nil-after-loop guard is dead too (Concatenate of an empty
// window list returns a non-nil zero-element array). None can be
// driven from synthetic inputs without fault-injecting production code, so they
// stay uncovered by design.

// TestCoverage_WindowDefault asserts wkv7ChunkedWindowed's width<=0 guard falls
// back to chunkWidth — width 0 must behave exactly like the production
// WKV7Chunked (single window for L<=chunkWidth).
func TestCoverage_WindowDefault(t *testing.T) {
	const L, H, K, Vd = 8, 2, 3, 4
	in, free := chunkStepInputs(L, H, K, Vd)
	defer free()

	def, defState := WKV7Chunked(in, nil) // width = chunkWidth
	if def == nil {
		t.Fatal("WKV7Chunked returned nil")
	}
	defer metal.Free(def, defState)

	zero, zeroState := wkv7ChunkedWindowed(in, nil, 0) // width<=0 ⇒ chunkWidth
	if zero == nil {
		t.Fatal("wkv7ChunkedWindowed(width=0) returned nil")
	}
	defer metal.Free(zero, zeroState)

	assertCloseF32(t, "width0-vs-default o", zero.Floats(), def.Floats(), 1e-6)
	assertCloseF32(t, "width0-vs-default state", zeroState.Floats(), defState.Floats(), 1e-6)
}

// TestCoverage_EmptySequence_NilPrior drives compute's L==0 boundary with no
// prior: the window loop never runs, so o is the empty-output guard and the
// returned state is a fresh zero [B,H,K,V] (the state==nil branch). Built with
// metal.Zeros (FromValues can't make a zero-element backing slice).
func TestCoverage_EmptySequence_NilPrior(t *testing.T) {
	const H, K, Vd = 2, 3, 4
	z := func(d3 int32) *metal.Array { return metal.Zeros([]int32{1, 0, H, d3}, metal.DTypeFloat32) }
	in := StepInput{R: z(K), W: z(K), K: z(K), V: z(Vd), A: z(K), B: z(K)}
	defer metal.Free(in.R, in.W, in.K, in.V, in.A, in.B)

	o, state := WKV7Chunked(in, nil)
	if o == nil || state == nil {
		t.Fatalf("L=0 nil-prior: o=%v state=%v, want non-nil (empty o, zero state)", o != nil, state != nil)
	}
	defer metal.Free(o, state)

	// The empty-sequence output carries no elements (MLX reports a zero-element
	// array's shape as empty); what matters is the guard returned a non-nil,
	// element-free output rather than crashing on the empty window list.
	if n := len(o.Floats()); n != 0 {
		t.Errorf("empty o has %d elements, want 0", n)
	}
	if got := state.Shape(); len(got) != 4 || got[0] != 1 || got[1] != H || got[2] != K || got[3] != Vd {
		t.Fatalf("zero state shape = %v, want [1 %d %d %d]", got, H, K, Vd)
	}
	for i, v := range state.Floats() {
		if v != 0 {
			t.Errorf("zero state[%d] = %v, want 0 (fresh state on empty sequence)", i, v)
		}
	}
}

// TestCoverage_EmptySequence_WithPrior drives compute's L==0 boundary WITH a
// prior: the loop never runs so state is still in.prev — compute returns a CLONE
// (the state==in.prev branch) so ownership is uniform (the caller frees it). The
// clone must carry the prior values byte-for-byte.
func TestCoverage_EmptySequence_WithPrior(t *testing.T) {
	const H, K, Vd = 2, 2, 2
	z := func(d3 int32) *metal.Array { return metal.Zeros([]int32{1, 0, H, d3}, metal.DTypeFloat32) }
	in := StepInput{R: z(K), W: z(K), K: z(K), V: z(Vd), A: z(K), B: z(K)}
	defer metal.Free(in.R, in.W, in.K, in.V, in.A, in.B)

	priorVals := []float32{0.1, -0.2, 0.3, 0.4, 0.5, -0.6, 0.7, 0.8}
	prior := metal.FromValues(priorVals, 1, H, K, Vd)
	defer metal.Free(prior)

	o, state := WKV7Chunked(in, prior)
	if o == nil || state == nil {
		t.Fatalf("L=0 with-prior: o=%v state=%v, want non-nil", o != nil, state != nil)
	}
	defer metal.Free(o, state)

	got := state.Floats()
	if len(got) != len(priorVals) {
		t.Fatalf("clone length = %d, want %d", len(got), len(priorVals))
	}
	for i := range got {
		if got[i] != priorVals[i] {
			t.Errorf("clone[%d] = %v, want %v (empty-sequence state is the carried prior, cloned)", i, got[i], priorVals[i])
		}
	}
	// The returned state must be a distinct array from prior (a clone, so the
	// caller can free it without touching the still-owned prior).
	if state == prior {
		t.Error("empty-sequence state aliases prior; want a clone")
	}
}

// TestCoverage_BF16_TriRoundTrip runs the chunked scan on bf16 inputs, exercising
// solve's float32 LAPACK round-trip: asFloat32 casts the bf16 (I − A_ab) up for
// metal.TriInv (the imCast=true free), and castBack casts the inverse back to the
// working dtype. The float32 path (chunk_test.go) skips both casts. The bf16
// result must still track the sequential reference within bf16 tolerance.
func TestCoverage_BF16_TriRoundTrip(t *testing.T) {
	const L, H, K, Vd = 8, 2, 3, 4
	in, free := chunkStepInputs(L, H, K, Vd)
	defer free()

	// float32 reference from the sequential kernel.
	seqO, seqState := WKV7(in, nil)
	defer metal.Free(seqO, seqState)

	bf := StepInput{
		R: metal.AsType(in.R, metal.DTypeBFloat16),
		W: metal.AsType(in.W, metal.DTypeBFloat16),
		K: metal.AsType(in.K, metal.DTypeBFloat16),
		V: metal.AsType(in.V, metal.DTypeBFloat16),
		A: metal.AsType(in.A, metal.DTypeBFloat16),
		B: metal.AsType(in.B, metal.DTypeBFloat16),
	}
	defer metal.Free(bf.R, bf.W, bf.K, bf.V, bf.A, bf.B)

	o, state := WKV7Chunked(bf, nil)
	if o == nil {
		t.Fatal("bf16 WKV7Chunked returned nil")
	}
	defer metal.Free(o, state)

	if got := o.Shape(); len(got) != 4 || got[0] != 1 || got[1] != L || got[2] != H || got[3] != Vd {
		t.Fatalf("bf16 o shape = %v, want [1 %d %d %d]", got, L, H, Vd)
	}
	// bf16 has ~3 significant digits; the round-trip through the f32 inverse keeps
	// the structure, so a loose tolerance still catches a wrong cast / aliasing.
	assertCloseF32(t, "bf16 o vs f32-sequential", o.Floats(), seqO.Floats(), 5e-2)
	assertCloseF32(t, "bf16 state vs f32-sequential", state.Floats(), seqState.Floats(), 5e-2)
}

// TestCoverage_BF16_WindowedTriRoundTrip runs the bf16 cast round-trip across
// MULTIPLE windows (width < L), so the asFloat32/castBack pair fires once per
// window and the bf16 inter-window state carry is exercised too.
func TestCoverage_BF16_WindowedTriRoundTrip(t *testing.T) {
	const L, H, K, Vd = 10, 2, 2, 3
	in, free := chunkStepInputs(L, H, K, Vd)
	defer free()

	seqO, seqState := WKV7(in, nil)
	defer metal.Free(seqO, seqState)

	bf := StepInput{
		R: metal.AsType(in.R, metal.DTypeBFloat16),
		W: metal.AsType(in.W, metal.DTypeBFloat16),
		K: metal.AsType(in.K, metal.DTypeBFloat16),
		V: metal.AsType(in.V, metal.DTypeBFloat16),
		A: metal.AsType(in.A, metal.DTypeBFloat16),
		B: metal.AsType(in.B, metal.DTypeBFloat16),
	}
	defer metal.Free(bf.R, bf.W, bf.K, bf.V, bf.A, bf.B)

	o, state := wkv7ChunkedWindowed(bf, nil, 3) // windows 3,3,3,1
	if o == nil {
		t.Fatal("bf16 windowed WKV7Chunked returned nil")
	}
	defer metal.Free(o, state)

	assertCloseF32(t, "bf16 multi-window o", o.Floats(), seqO.Floats(), 5e-2)
	assertCloseF32(t, "bf16 multi-window state", state.Floats(), seqState.Floats(), 5e-2)
}

// TestCoverage_DtypeHelpers unit-tests asFloat32 / castBack directly — the LAPACK
// dtype-bridge contract. Inside solve, mask promotion always hands asFloat32 a
// float32 (I − A_ab) so its up-cast branch never fires there; called directly on
// a bf16 array it must up-cast and report cast=true (and is a no-op, cast=false,
// on a float32 array). castBack is the inverse: cast a float32 result down to a
// non-float32 target, and pass a float32 target straight through.
func TestCoverage_DtypeHelpers(t *testing.T) {
	f32 := metal.FromValues([]float32{1, 2, 3, 4}, 2, 2)
	defer metal.Free(f32)

	// asFloat32 on an already-float32 array: no cast, same handle, cast=false.
	if got, cast := asFloat32(f32); cast || got != f32 {
		t.Errorf("asFloat32(float32) = (%p,cast=%v), want (same handle, false)", got, cast)
	}

	// asFloat32 on a bf16 array: must up-cast to float32 and report cast=true.
	bf := metal.AsType(f32, metal.DTypeBFloat16)
	defer metal.Free(bf)
	up, cast := asFloat32(bf)
	if !cast {
		t.Error("asFloat32(bf16): cast=false, want true (up-cast happened)")
	}
	if up.Dtype() != metal.DTypeFloat32 {
		t.Errorf("asFloat32(bf16) dtype = %v, want float32", up.Dtype())
	}
	if cast {
		metal.Free(up) // caller owns the up-cast copy
	}

	// castBack to a non-float32 target casts down; to float32 passes through.
	src := metal.FromValues([]float32{0.5, 1.5, 2.5, 3.5}, 2, 2)
	down := castBack(src, metal.DTypeBFloat16) // consumes src, returns bf16
	if down.Dtype() != metal.DTypeBFloat16 {
		t.Errorf("castBack(_, bf16) dtype = %v, want bfloat16", down.Dtype())
	}
	metal.Free(down)

	src2 := metal.FromValues([]float32{1, 1, 1, 1}, 2, 2)
	through := castBack(src2, metal.DTypeFloat32) // float32 target ⇒ same handle
	if through != src2 {
		t.Error("castBack(_, float32) returned a new array, want the input unchanged")
	}
	metal.Free(through)
}

// TestCoverage_Loader_BadGeometry drives buildRWKV7's configFromShapes error
// hand-off: a NumAttentionHeads that doesn't divide the receptance width is a
// loud build error, not a silently mis-split mixer. This covers the
// configFromShapes-failed return in buildRWKV7 and the non-divisible branch in
// configFromShapes.
func TestCoverage_Loader_BadGeometry(t *testing.T) {
	ctx := fakeBuildCtx()
	// fakeBuildCtx wires receptance/value out-width 4. Force a head count that
	// doesn't divide 4 so configFromShapes rejects the geometry.
	ctx.Cfg.NumAttentionHeads = 3
	if _, err := buildRWKV7(ctx); err == nil {
		t.Error("expected build error for non-divisible head count (4 % 3 != 0), got nil")
	}
}

// TestCoverage_ConfigFromShapes_Direct hits configFromShapes' two early-return
// validations directly: a non-positive head count (and nil weights), and the
// non-divisible projection width. The happy path is covered by the loader tests;
// these are the guards a head-count/shape mismatch trips.
func TestCoverage_ConfigFromShapes_Direct(t *testing.T) {
	const D = 4
	rProj := metal.NewLinear(metal.FromValues(make([]float32, 4*D), 4, D), nil)
	vProj := metal.NewLinear(metal.FromValues(make([]float32, 4*D), 4, D), nil)
	defer metal.Free(rProj.Weight, vProj.Weight)

	// Non-positive head count.
	if _, err := configFromShapes(0, rProj, vProj); err == nil {
		t.Error("configFromShapes(H=0): expected error, got nil")
	}
	// Nil receptance weight (the other half of the first guard).
	if _, err := configFromShapes(2, metal.NewLinear(nil, nil), vProj); err == nil {
		t.Error("configFromShapes(nil rProj weight): expected error, got nil")
	}
	// Non-divisible width: rOut=4, H=3 ⇒ 4 % 3 != 0.
	if _, err := configFromShapes(3, rProj, vProj); err == nil {
		t.Error("configFromShapes(H=3, rOut=4): expected non-divisible error, got nil")
	}
	// Sanity: the happy path still derives the right geometry.
	cfg, err := configFromShapes(2, rProj, vProj)
	if err != nil {
		t.Fatalf("configFromShapes happy path: %v", err)
	}
	if cfg.NumHeads != 2 || cfg.KeyDim != 2 || cfg.ValueDim != 2 {
		t.Errorf("cfg = %+v, want {H:2 K:2 V:2}", cfg)
	}
}

// TestCoverage_Forward_NilWeights asserts Mixer.Forward's nil-guard: a mixer
// with no weights (or no config) returns (nil, empty) instead of dereferencing a
// nil Weights — the bare-seed safety the scheme scaffold relies on.
func TestCoverage_Forward_NilWeights(t *testing.T) {
	x := metal.FromValues([]float32{0.1, 0.2, 0.3, 0.4}, 1, 1, 4)
	defer metal.Free(x)
	ctx := &metal.MixerCtx{B: 1, L: 1}

	// No weights at all.
	if out, _ := (&Mixer{}).Forward(x, ctx); out != nil {
		t.Error("Mixer{}.Forward: expected nil output, got non-nil")
		metal.Free(out)
	}
	// Weights present but no config (cfg nil) — same guard.
	if out, _ := (&Mixer{W: &Weights{}}).Forward(x, ctx); out != nil {
		t.Error("Mixer{W:&Weights{}}.Forward: expected nil output (nil cfg), got non-nil")
		metal.Free(out)
	}
}

// TestCoverage_Forward_NoHolder runs Mixer.Forward through a bare MixerCtx (no
// recurrent cache): the holder-less path frees the advanced state instead of
// writing it back (the !hasHolder branch). The output must still be the correct
// shape — a unit test of the mixer with no surrounding decoder state.
func TestCoverage_Forward_NoHolder(t *testing.T) {
	m := tinyMixer(t)
	const B, L, D = 1, 2, 4
	x := metal.FromValues([]float32{
		0.5, -0.2, 0.1, 0.3,
		-0.1, 0.4, -0.3, 0.2,
	}, B, L, D)
	defer metal.Free(x)

	// MixerCtx with no Cache ⇒ ctx.Recurrent() reports no holder.
	out, _ := m.Forward(x, &metal.MixerCtx{B: B, L: L})
	if out == nil {
		t.Fatal("holder-less Forward returned nil")
	}
	defer metal.Free(out)
	if got := out.Shape(); len(got) != 3 || got[0] != B || got[1] != L || got[2] != D {
		t.Errorf("holder-less out shape = %v, want [%d %d %d]", got, B, L, D)
	}
	for i, v := range out.Floats() {
		if v != v { // NaN
			t.Fatalf("holder-less out[%d] is NaN", i)
		}
	}
}

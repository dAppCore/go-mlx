// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma4

import (
	"testing"

	"dappco.re/go/mlx/pkg/metal"
)

// Coverage for two vision-forward seams a loaded SigLIP checkpoint never
// exercises: the non-2D fallback arm of gemma4VisionRoPEAndTranspose, and the
// degenerate guards of the patch embedder's poolKernel accessor. Both are
// reachable only by constructing the inputs the loader never produces, so the
// production behaviour is observable nowhere else.

// TestVisionForward_RoPEAndTranspose_FallbackSmallDim_Good drives the ELSE arm
// of gemma4VisionRoPEAndTranspose. A real checkpoint always carries a head dim
// of at least 4, so gemma4VisionApply2DRoPE's D>=4 guard holds and the 2-D
// rotary arm runs. Here the head dim is 2 (D<4): the 2-D arm returns nil and the
// function falls through to a plain rank-4 transpose followed by metal.RoPE over
// the head dim. The base is non-zero so the fallback rotary is well defined (a
// zero base would signal "no explicit base" to the MLX fast-RoPE optional, a
// separate path). The result keeps the rank-4 shape with axes 1 and 2 swapped:
// [B, L, N, D] -> [B, N, L, D].
func TestVisionForward_RoPEAndTranspose_FallbackSmallDim_Good(t *testing.T) {
	requireMetalRuntime(t)

	const B, L, N, D = int32(1), int32(4), int32(2), int32(2) // D<4 forces fallback
	const gridH, gridW = int32(2), int32(2)                   // gridH*gridW == L
	const base = 10000.0
	x := seqArray(0.05, int(B), int(L), int(N), int(D))
	defer metal.Free(x)

	out := gemma4VisionRoPEAndTranspose(x, gridH, gridW, base, D)
	if out == nil || !out.Valid() {
		t.Fatal("gemma4VisionRoPEAndTranspose(D<4) returned nil, want the transpose+RoPE fallback")
	}
	defer metal.Free(out)
	if err := metal.Eval(out); err != nil {
		t.Fatalf("Eval fallback rope: %v", err)
	}

	// Transpose4(0,2,1,3) swaps L and N: [B, L, N, D] -> [B, N, L, D].
	assertShape(t, "rope fallback", out, []int32{B, N, L, D})

	// Independent value reference: the fallback arm is documented (and its
	// source confirms) as "a plain rank-4 transpose followed by metal.RoPE
	// over the head dim" — reproduce exactly that via the SAME public
	// primitives, authored here rather than read back from the function
	// under test, so a regression that changes the transpose axes, the
	// traditional flag, or the base would fail this (a shape-only check
	// cannot catch that; an identity/no-op "RoPE" would still pass shape).
	wantTransposed := metal.Transpose4(x, 0, 2, 1, 3)
	want := metal.RoPE(wantTransposed, int(D), false, base, 1.0, 0)
	defer metal.Free(wantTransposed, want)
	if err := metal.Eval(want); err != nil {
		t.Fatalf("Eval reference rope: %v", err)
	}
	floatSliceApprox(t, out.Floats(), want.Floats())

	// And prove the rotation is not a no-op relative to a plain transpose
	// (L=4 means 3 of the 4 positions get a non-trivial rotation at base
	// 10000 with this non-zero input).
	plainTransposed := metal.Transpose4(x, 0, 2, 1, 3)
	defer metal.Free(plainTransposed)
	if err := metal.Eval(plainTransposed); err != nil {
		t.Fatalf("Eval plain transpose: %v", err)
	}
	sameAsPlain := true
	outVals, plainVals := out.Floats(), plainTransposed.Floats()
	for i := range outVals {
		diff := outVals[i] - plainVals[i]
		if diff < 0 {
			diff = -diff
		}
		if diff >= floatApproxTol {
			sameAsPlain = false
			break
		}
	}
	if sameAsPlain {
		t.Fatal("fallback rope output matches a plain (unrotated) transpose — the RoPE arm made no observable difference")
	}
}

// TestVisionForward_RoPEAndTranspose_TwoDimPath_Good pins the IF arm for
// contrast: a head dim of 4 with a grid that factors L keeps the 2-D rotary arm
// alive (gemma4VisionApply2DRoPE returns non-nil), and the function transposes
// the rotated tensor. The output shape is the same rank-4 axis swap, which lets
// this test and the fallback test share the post-condition while taking opposite
// branches through the function.
func TestVisionForward_RoPEAndTranspose_TwoDimPath_Good(t *testing.T) {
	requireMetalRuntime(t)

	const B, L, N, D = int32(1), int32(4), int32(2), int32(4) // D>=4 keeps the 2-D arm
	const gridH, gridW = int32(2), int32(2)
	const base = 10000.0
	x := seqArray(0.03, int(B), int(L), int(N), int(D))
	defer metal.Free(x)

	out := gemma4VisionRoPEAndTranspose(x, gridH, gridW, base, D)
	if out == nil || !out.Valid() {
		t.Fatal("gemma4VisionRoPEAndTranspose(D>=4) returned nil, want the 2-D rotary arm")
	}
	defer metal.Free(out)
	if err := metal.Eval(out); err != nil {
		t.Fatalf("Eval 2-D rope: %v", err)
	}

	assertShape(t, "rope 2-D", out, []int32{B, N, L, D})

	// Prove the 2-D rotary arm is not a silent no-op: a plain transpose
	// WITHOUT any rotation must differ from the actual output (any patch
	// position off the grid origin gets a non-trivial 2-D rotation at base
	// 10000 with this non-zero input). This does not re-derive the full 2-D
	// rotation formula (gemma4VisionApply2DRoPE's x/y split + sin/cos mix is
	// substantial machinery of its own, unverified independently anywhere in
	// this cluster) — it catches the coarser but still real regression class
	// "the 2-D arm silently stopped rotating".
	plainTransposed := metal.Transpose4(x, 0, 2, 1, 3)
	defer metal.Free(plainTransposed)
	if err := metal.Eval(plainTransposed); err != nil {
		t.Fatalf("Eval plain transpose: %v", err)
	}
	sameAsPlain := true
	outVals, plainVals := out.Floats(), plainTransposed.Floats()
	for i := range outVals {
		diff := outVals[i] - plainVals[i]
		if diff < 0 {
			diff = -diff
		}
		if diff >= floatApproxTol {
			sameAsPlain = false
			break
		}
	}
	if sameAsPlain {
		t.Fatal("2-D rope output matches a plain (unrotated) transpose — the 2-D rotary arm made no observable difference")
	}
}

// TestVisionForward_Apply2DRoPE_RankTooLow_Ugly pins the rank guard of
// gemma4VisionApply2DRoPE directly: a rank-3 tensor has no head axis, so the
// 2-D rotary returns nil rather than slicing a missing dimension. This is the
// guard the loaded path never trips (vision attention always reshapes to rank-4
// before calling), so it is only observable on a hand-built rank-3 input.
func TestVisionForward_Apply2DRoPE_RankTooLow_Ugly(t *testing.T) {
	requireMetalRuntime(t)

	rank3 := seqArray(0.05, 1, 4, 4) // no head axis
	defer metal.Free(rank3)

	if rotated := gemma4VisionApply2DRoPE(rank3, 2, 2, 10000.0); rotated != nil {
		metal.Free(rotated)
		t.Fatal("gemma4VisionApply2DRoPE(rank-3) returned non-nil, want nil for the rank guard")
	}
}

// TestVisionForward_Apply2DRoPE_ZeroBase_Ugly pins the base guard: a zero base
// disables explicit rotary, so the 2-D arm declines (returns nil) rather than
// dividing by a zero frequency. A loaded vision tower always carries a positive
// RopeBase, so this guard is reachable only synthetically.
func TestVisionForward_Apply2DRoPE_ZeroBase_Ugly(t *testing.T) {
	requireMetalRuntime(t)

	x := seqArray(0.05, 1, 4, 2, 8) // valid rank-4, D>=4
	defer metal.Free(x)

	if rotated := gemma4VisionApply2DRoPE(x, 2, 2, 0); rotated != nil {
		metal.Free(rotated)
		t.Fatal("gemma4VisionApply2DRoPE(base=0) returned non-nil, want nil for the zero-base guard")
	}
}

// TestVisionForward_poolKernel_Good drives the live-value arm of the patch
// embedder's poolKernel accessor: a positive PoolingKernelSize is returned
// verbatim. poolKernel is a pure accessor — no Metal — so it runs in either
// build mode.
func TestVisionForward_poolKernel_Good(t *testing.T) {
	embedder := &Gemma4VisionPatchEmbedder{PoolingKernelSize: 4}
	if got := embedder.poolKernel(); got != 4 {
		t.Fatalf("poolKernel() = %d, want 4", got)
	}
}

// TestVisionForward_poolKernel_NonPositive_Bad pins the clamp: a zero or
// negative kernel size means "no pooling", which the accessor normalises to 1
// so downstream grid maths never divides by a non-positive stride. The loader
// defaults the field, so the non-positive case is only reachable on a directly
// constructed embedder.
func TestVisionForward_poolKernel_NonPositive_Bad(t *testing.T) {
	for _, size := range []int32{0, -1, -8} {
		embedder := &Gemma4VisionPatchEmbedder{PoolingKernelSize: size}
		if got := embedder.poolKernel(); got != 1 {
			t.Fatalf("poolKernel(size=%d) = %d, want 1", size, got)
		}
	}
}

// TestVisionForward_poolKernel_NilReceiver_Ugly pins the nil-receiver guard: a
// nil patch embedder must answer 1 (no pooling) rather than panic on the field
// read. Reached when a caller probes pooling before the embedder is wired.
func TestVisionForward_poolKernel_NilReceiver_Ugly(t *testing.T) {
	var embedder *Gemma4VisionPatchEmbedder
	if got := embedder.poolKernel(); got != 1 {
		t.Fatalf("poolKernel(nil) = %d, want 1", got)
	}
}

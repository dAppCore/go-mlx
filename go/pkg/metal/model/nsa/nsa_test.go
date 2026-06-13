// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package nsa

import (
	"math"
	"testing"

	"dappco.re/go/mlx/internal/metaltest"
	metal "dappco.re/go/mlx/pkg/metal"
	scheme "dappco.re/go/mlx/pkg/scheme"
)

func requireMetalRuntime(t testing.TB) {
	t.Helper()
	if !metaltest.RunMetalTests {
		t.Skip("build with -tags metal_runtime to enable Metal runtime tests")
	}
	if !metal.MetalAvailable() {
		t.Skip("Metal runtime unavailable")
	}
}

const nsaTol = 1e-4

func isNegInf(f float32) bool { return math.IsInf(float64(f), -1) }

// approxEqual compares a float32 to a reference, treating -inf specially so the
// additive-mask assertions can check the forbidden fill exactly.
func approxEqual(got, want float32) bool {
	if isNegInf(want) {
		return isNegInf(got)
	}
	return math.Abs(float64(got-want)) <= nsaTol
}

// TestMixer_KindState_Good pins the scheme identity (no Metal runtime needed).
func TestMixer_KindState_Good(t *testing.T) {
	if MixerKind != "nsa" {
		t.Fatalf("MixerKind = %q, want %q", MixerKind, "nsa")
	}
	m := &Mixer{}
	if got := m.Kind(); got != MixerKind {
		t.Fatalf("Kind() = %q, want %q", got, MixerKind)
	}
	if got := m.State(); got != scheme.StateKVCache {
		t.Fatalf("State() = %v, want StateKVCache", got)
	}
}

// TestMixer_Register_Good proves init() registered a compute-bearing mixer.
func TestMixer_Register_Good(t *testing.T) {
	if _, ok := scheme.MixerFor(MixerKind); !ok {
		t.Fatalf("scheme.MixerFor(%q) not registered", MixerKind)
	}
	if _, ok := metal.MixerComputeFor(MixerKind); !ok {
		t.Fatalf("metal.MixerComputeFor(%q) = false, want compute-bearing", MixerKind)
	}
}

// TestCompressKV_Math_Good pins the block mean-pool. L=4, blockSize=2 →
// 2 blocks. K rows [1,1],[3,3],[5,5],[7,7] → block means [2,2],[6,6].
func TestCompressKV_Math_Good(t *testing.T) {
	requireMetalRuntime(t)

	k := metal.FromValues([]float32{1, 1, 3, 3, 5, 5, 7, 7}, 1, 1, 4, 2)
	v := metal.FromValues([]float32{2, 0, 4, 0, 6, 0, 8, 0}, 1, 1, 4, 2)
	defer metal.Free(k, v)

	kCmp, vCmp := compressKV(k, v, 2)
	defer metal.Free(kCmp, vCmp)

	if got := kCmp.Dim(2); got != 2 {
		t.Fatalf("kCmp blocks = %d, want 2", got)
	}
	gotK := kCmp.Floats()
	wantK := []float32{2, 2, 6, 6}
	for i := range wantK {
		if !approxEqual(gotK[i], wantK[i]) {
			t.Errorf("kCmp[%d] = %f, want %f", i, gotK[i], wantK[i])
		}
	}
	gotV := vCmp.Floats()
	wantV := []float32{3, 0, 7, 0} // mean([2,0],[4,0])=[3,0]; mean([6,0],[8,0])=[7,0]
	for i := range wantV {
		if !approxEqual(gotV[i], wantV[i]) {
			t.Errorf("vCmp[%d] = %f, want %f", i, gotV[i], wantV[i])
		}
	}
}

// TestSlidingMask_Math_Good pins the sliding-window additive mask. L=4,
// window=2: query t sees keys [t-1, t]; everything else -inf.
func TestSlidingMask_Math_Good(t *testing.T) {
	requireMetalRuntime(t)

	mask := slidingMask(4, 2)
	defer metal.Free(mask)
	got := mask.Floats()

	n := float32(math.Inf(-1))
	want := []float32{
		0, n, n, n, // t=0: only key 0
		0, 0, n, n, // t=1: keys 0,1
		n, 0, 0, n, // t=2: keys 1,2 (key0 outside window)
		n, n, 0, 0, // t=3: keys 2,3
	}
	for i := range want {
		if !approxEqual(got[i], want[i]) {
			t.Errorf("slidingMask[%d] = %f, want %f", i, got[i], want[i])
		}
	}
}

// TestBlockCausalMask_Math_Good pins the compression-branch mask. L=4,
// nBlocks=2, blockSize=2: query t sees block b iff b*2 <= t.
func TestBlockCausalMask_Math_Good(t *testing.T) {
	requireMetalRuntime(t)

	mask := blockCausalMask(4, 2, 2)
	defer metal.Free(mask)
	got := mask.Floats()

	n := float32(math.Inf(-1))
	want := []float32{
		0, n, // t=0: block0 (starts@0) ok, block1 (starts@2) future
		0, n, // t=1: same
		0, 0, // t=2: both ok
		0, 0, // t=3: both ok
	}
	for i := range want {
		if !approxEqual(got[i], want[i]) {
			t.Errorf("blockCausalMask[%d] = %f, want %f", i, got[i], want[i])
		}
	}
}

// TestSelectionMask_TopN_Good pins the top-n block keep logic. One query row,
// 3 blocks, scores [3,1,2], all causally valid, selectBlocks=2 → keep blocks
// with score >= 2 (the 2nd-largest), i.e. blocks 0 and 2; block 1 → -inf.
func TestSelectionMask_TopN_Good(t *testing.T) {
	requireMetalRuntime(t)

	scores := metal.FromValues([]float32{3, 1, 2}, 1, 1, 1, 3)
	causal := metal.FromValues([]float32{0, 0, 0}, 1, 1, 1, 3) // all valid
	defer metal.Free(scores, causal)

	mask := selectionMask(scores, causal, 2)
	defer metal.Free(mask)
	got := mask.Floats()

	n := float32(math.Inf(-1))
	want := []float32{0, n, 0} // block0 kept, block1 dropped, block2 kept
	for i := range want {
		if !approxEqual(got[i], want[i]) {
			t.Errorf("selectionMask[%d] = %f, want %f", i, got[i], want[i])
		}
	}
}

// TestSelectionMask_CausalDropsFuture_Good confirms a future block (-inf in the
// causal mask) is never selected even if its raw score is the largest.
func TestSelectionMask_CausalDropsFuture_Good(t *testing.T) {
	requireMetalRuntime(t)

	scores := metal.FromValues([]float32{1, 9, 2}, 1, 1, 1, 3) // block1 highest raw
	n := float32(math.Inf(-1))
	causal := metal.FromValues([]float32{0, n, 0}, 1, 1, 1, 3) // block1 is future
	defer metal.Free(scores, causal)

	mask := selectionMask(scores, causal, 1) // keep top-1
	defer metal.Free(mask)
	got := mask.Floats()

	// block1 forbidden by causal → top-1 of the valid {1@b0, 2@b2} is b2.
	want := []float32{n, n, 0}
	for i := range want {
		if !approxEqual(got[i], want[i]) {
			t.Errorf("selectionMask[%d] = %f, want %f", i, got[i], want[i])
		}
	}
}

// TestAttend_Math_Good pins the shared softmax-attention kernel against a
// hand-derived causal reference (same construction as the MLA kernel test).
func TestAttend_Math_Good(t *testing.T) {
	requireMetalRuntime(t)

	q := metal.FromValues([]float32{1, 0, 0, 1}, 1, 1, 2, 2)
	k := metal.FromValues([]float32{1, 0, 0, 1}, 1, 1, 2, 2)
	v := metal.FromValues([]float32{1, 2, 3, 4}, 1, 1, 2, 2)
	nInf := float32(math.Inf(-1))
	mask := metal.FromValues([]float32{0, nInf, 0, 0}, 1, 1, 2, 2)
	defer metal.Free(q, k, v, mask)

	out := attend(q, k, v, mask, 1.0)
	defer metal.Free(out)

	got := out.Floats()
	w0 := 1.0 / (1.0 + math.E)
	w1 := math.E / (1.0 + math.E)
	want := []float32{1, 2, float32(w0*1 + w1*3), float32(w0*2 + w1*4)}
	for i := range want {
		if !approxEqual(got[i], want[i]) {
			t.Errorf("attend[%d] = %f, want %f", i, got[i], want[i])
		}
	}
}

// TestExpandBlockMaskToTokens_Good pins the block→token mask broadcast. Two
// blocks of size 2 over L=4; block mask [0, -inf] (keep block0, drop block1).
// Each block value repeats across its 2 tokens, then causal masking applies.
func TestExpandBlockMaskToTokens_Good(t *testing.T) {
	requireMetalRuntime(t)

	n := float32(math.Inf(-1))
	// [B=1,H=1,L=4,nBlocks=2]: every query keeps block0, drops block1.
	blockMask := metal.FromValues([]float32{
		0, n,
		0, n,
		0, n,
		0, n,
	}, 1, 1, 4, 2)
	defer metal.Free(blockMask)

	tokenMask := expandBlockMaskToTokens(blockMask, 2, 4)
	defer metal.Free(tokenMask)
	got := tokenMask.Floats()

	// covered=4=L. Block0 → tokens 0,1 (value 0); block1 → tokens 2,3 (-inf).
	// Then causal: query t sees key j iff j<=t. Combined (0 + causal):
	want := []float32{
		0, n, n, n, // t=0: key0 kept&causal; key1 kept but future; keys2,3 dropped
		0, 0, n, n, // t=1: keys0,1 kept&causal; keys2,3 dropped block
		0, 0, n, n, // t=2: keys0,1 kept&causal; keys2,3 dropped block (even though causal-ok)
		0, 0, n, n, // t=3: keys0,1 kept; keys2,3 dropped block
	}
	for i := range want {
		if !approxEqual(got[i], want[i]) {
			t.Errorf("tokenMask[%d] = %f, want %f", i, got[i], want[i])
		}
	}
}

// TestBlendBranches_Math_Good pins the gate blend g_cmp·o_cmp + g_slc·o_slc +
// g_swa·o_swa. B=1,H=1,L=1,D=2. Synthetic branch outputs and gates so the
// arithmetic is exact and independent of the attention kernels.
func TestBlendBranches_Math_Good(t *testing.T) {
	requireMetalRuntime(t)

	m := &Mixer{NumHeads: 1, HeadDim: 2}
	oCmp := metal.FromValues([]float32{1, 1}, 1, 1, 1, 2)
	oSlc := metal.FromValues([]float32{2, 2}, 1, 1, 1, 2)
	oSwa := metal.FromValues([]float32{4, 4}, 1, 1, 1, 2)
	// gates [B,L,H*3] = [g_cmp, g_slc, g_swa] = [0.5, 0.25, 0.1].
	gates := metal.FromValues([]float32{0.5, 0.25, 0.1}, 1, 1, 3)
	defer metal.Free(oCmp, oSlc, oSwa, gates)

	out := m.blendBranches(oCmp, oSlc, oSwa, gates, 1, 1)
	defer metal.Free(out)
	got := out.Floats()

	// 0.5*1 + 0.25*2 + 0.1*4 = 0.5 + 0.5 + 0.4 = 1.4 for each of the 2 dims.
	want := []float32{1.4, 1.4}
	for i := range want {
		if !approxEqual(got[i], want[i]) {
			t.Errorf("blend[%d] = %f, want %f", i, got[i], want[i])
		}
	}
}

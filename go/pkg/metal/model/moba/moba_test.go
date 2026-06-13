// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package moba

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

const mobaTol = 1e-4

func isNegInf(f float32) bool { return math.IsInf(float64(f), -1) }

func approxEqual(got, want float32) bool {
	if isNegInf(want) {
		return isNegInf(got)
	}
	return math.Abs(float64(got-want)) <= mobaTol
}

func checkMask(t *testing.T, name string, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s len = %d, want %d", name, len(got), len(want))
	}
	for i := range want {
		if !approxEqual(got[i], want[i]) {
			t.Errorf("%s[%d] = %f, want %f", name, i, got[i], want[i])
		}
	}
}

// TestMixer_KindState_Good pins the scheme identity (no Metal runtime needed).
func TestMixer_KindState_Good(t *testing.T) {
	if MixerKind != "moba" {
		t.Fatalf("MixerKind = %q, want %q", MixerKind, "moba")
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

// TestBlockScores_Math_Good pins q·mean_pool(K_block). L=2, D=2, blockSize=2 →
// 1 block. K=[[2,2],[4,4]] → rep [3,3]. q=[[1,0],[0,1]] → both rows score 3.
func TestBlockScores_Math_Good(t *testing.T) {
	requireMetalRuntime(t)

	q := metal.FromValues([]float32{1, 0, 0, 1}, 1, 1, 2, 2)
	k := metal.FromValues([]float32{2, 2, 4, 4}, 1, 1, 2, 2)
	defer metal.Free(q, k)

	scores := blockScores(q, k, 2, 1.0)
	defer metal.Free(scores)

	if got := scores.Dim(3); got != 1 {
		t.Fatalf("blocks = %d, want 1", got)
	}
	checkMask(t, "blockScores", scores.Floats(), []float32{3, 3})
}

// TestBlockSelectMask_SelfOnly_Good pins selectK=0: every query keeps only its
// self-block; past and future blocks are dropped. L=4, blockSize=2, 2 blocks.
func TestBlockSelectMask_SelfOnly_Good(t *testing.T) {
	requireMetalRuntime(t)

	// Scores favour block0 everywhere, to prove the self-block rule overrides
	// raw score (t=2,3 still pick block1, their self-block, not block0).
	scores := metal.FromValues([]float32{
		9, 1,
		9, 1,
		9, 1,
		9, 1,
	}, 1, 1, 4, 2)
	defer metal.Free(scores)

	mask := blockSelectMask(scores, 2, 0, 4)
	defer metal.Free(mask)

	n := float32(math.Inf(-1))
	want := []float32{
		0, n, // t=0: self block0; block1 future
		0, n, // t=1: self block0; block1 future
		n, 0, // t=2: self block1; block0 past, not in top-1
		n, 0, // t=3: self block1; block0 past, not in top-1
	}
	checkMask(t, "selfOnly", mask.Floats(), want)
}

// TestBlockSelectMask_TopKKeepsPast_Good pins selectK=1: a high-scoring PAST
// block is kept alongside the always-on self-block. At t=3 (self=block1),
// block0 scores high → both blocks kept.
func TestBlockSelectMask_TopKKeepsPast_Good(t *testing.T) {
	requireMetalRuntime(t)

	scores := metal.FromValues([]float32{
		5, 0, // t=0: self block0 (block1 future)
		5, 0, // t=1
		5, 0, // t=2: self block1, block0 past low? here block0=5 high
		5, 1, // t=3: self block1; block0=5 high past
	}, 1, 1, 4, 2)
	defer metal.Free(scores)

	mask := blockSelectMask(scores, 2, 1, 4)
	defer metal.Free(mask)

	n := float32(math.Inf(-1))
	// selectK=1 → keepN=2=nBlocks → threshold = the finite block0 score, so any
	// non-future block at/above it is kept. t=0,1: block1 still future → dropped.
	// t=2,3: block0 (past, high) + block1 (self) both kept.
	want := []float32{
		0, n, // t=0
		0, n, // t=1
		0, 0, // t=2: block0 past kept, block1 self kept
		0, 0, // t=3: block0 past kept, block1 self kept
	}
	checkMask(t, "topKPast", mask.Floats(), want)
}

// TestExpandToTokens_Causal_Good pins the block→token expansion + tril causal.
// blockMask keeps block0 (0) and block1 (0) for every query; blockSize=2, L=4.
// After expansion the causal mask makes it a plain lower-triangular mask, and
// the self-block is causal-masked within itself.
func TestExpandToTokens_Causal_Good(t *testing.T) {
	requireMetalRuntime(t)

	// Both blocks kept for all queries (0 everywhere).
	blockMask := metal.FromValues([]float32{
		0, 0,
		0, 0,
		0, 0,
		0, 0,
	}, 1, 1, 4, 2)
	defer metal.Free(blockMask)

	tokenMask := expandToTokens(blockMask, 2, 4, 2)
	defer metal.Free(tokenMask)

	n := float32(math.Inf(-1))
	// All blocks kept → mask is the pure causal tril (query t sees keys <= t).
	want := []float32{
		0, n, n, n,
		0, 0, n, n,
		0, 0, 0, n,
		0, 0, 0, 0,
	}
	checkMask(t, "expandCausal", tokenMask.Floats(), want)
}

// TestExpandToTokens_DropsBlock_Good confirms a dropped block (-inf) zeroes out
// all of its tokens after expansion, even causally-valid ones. block0 dropped,
// block1 kept; at t=3 keys 0,1 (block0) stay forbidden, keys 2,3 (block1) open.
func TestExpandToTokens_DropsBlock_Good(t *testing.T) {
	requireMetalRuntime(t)

	n := float32(math.Inf(-1))
	blockMask := metal.FromValues([]float32{
		n, n, // t=0,1 only see their own (future block1 also -inf here)
		n, n,
		n, 0, // t=2: block0 dropped, block1 kept (self)
		n, 0, // t=3: same
	}, 1, 1, 4, 2)
	defer metal.Free(blockMask)

	tokenMask := expandToTokens(blockMask, 2, 4, 2)
	defer metal.Free(tokenMask)

	want := []float32{
		n, n, n, n, // t=0: block0 dropped → keys0,1 forbidden; keys2,3 future block
		n, n, n, n, // t=1
		n, n, 0, n, // t=2: keys0,1 (block0) dropped; key2 (block1) causal-ok; key3 future
		n, n, 0, 0, // t=3: keys0,1 dropped; keys2,3 (block1) open
	}
	checkMask(t, "expandDrop", tokenMask.Floats(), want)
}

// TestAttend_Math_Good pins the shared attention kernel against the same
// hand-derived causal reference as the MLA/NSA kernel tests.
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

	w0 := 1.0 / (1.0 + math.E)
	w1 := math.E / (1.0 + math.E)
	want := []float32{1, 2, float32(w0*1 + w1*3), float32(w0*2 + w1*4)}
	checkMask(t, "attend", out.Floats(), want)
}

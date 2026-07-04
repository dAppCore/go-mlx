// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import "testing"

// TestSampleTokenFromLogits_Good covers the exported drafter-proposal entry
// (the gemma4 assistant's temperature>0 sampling step). The synthetic suite
// drives the unexported samplingDistribution / sampleTokenFromProbs directly,
// so the public passthrough was dark. A one-hot logit + a mid-coin draw must
// sample the dominant token; suppression of that token forces the runner-up.
func TestSampleTokenFromLogits_Good(t *testing.T) {
	requireMetalRuntime(t)

	mid := func() float32 { return 0.5 }
	// Token 2 dominates → sampled even off the greedy fast-path.
	logits := FromValues([]float32{0, 0, 10, 0}, 1, 4)
	defer Free(logits)
	if got := SampleTokenFromLogits(logits, 1.0, 0, 0, 0, nil, mid); got != 2 {
		t.Fatalf("SampleTokenFromLogits = %d, want 2 (dominant token)", got)
	}

	// Suppress the dominant token → its probability is zeroed, so the draw lands
	// on a different id.
	if got := SampleTokenFromLogits(logits, 1.0, 0, 0, 0, []int32{2}, mid); got == 2 {
		t.Fatal("SampleTokenFromLogits returned the suppressed token 2")
	}
}

// TestSpeculativeVerifyDecision_Good covers the exported verifier entry the
// gemma4 assistant calls on the temperature>0 path. It delegates to
// sampledVerifyDecision (tested directly elsewhere); this exercises the public
// wrapper. Target and drafter agreeing on the drafted token → accept the whole
// single-token block.
func TestSpeculativeVerifyDecision_Good(t *testing.T) {
	requireMetalRuntime(t)

	cfg := GenerateConfig{Temperature: 1}
	mid := func() float32 { return 0.5 }

	// Agree on token 0 → accepted, single-token block → allAccepted.
	tgt := FromValues([]float32{10, 0}, 1, 2)
	drf := FromValues([]float32{10, 0}, 1, 2)
	defer Free(tgt, drf)
	acc, _, all, err := SpeculativeVerifyDecision([]*Array{tgt}, []*Array{drf}, []int32{0}, cfg, mid, nil)
	if err != nil {
		t.Fatalf("SpeculativeVerifyDecision: %v", err)
	}
	if !all || len(acc) != 1 || acc[0] != 0 {
		t.Fatalf("accept case: acc=%v all=%v, want acc=[0] all=true", acc, all)
	}

	// Reject: drafter proposes 1 but the target overwhelmingly favours 0 → reject,
	// residual replacement = 0.
	tgt2 := FromValues([]float32{10, 0}, 1, 2)
	drf2 := FromValues([]float32{0, 10}, 1, 2)
	defer Free(tgt2, drf2)
	acc2, repl2, all2, err := SpeculativeVerifyDecision([]*Array{tgt2}, []*Array{drf2}, []int32{1}, cfg, mid, nil)
	if err != nil {
		t.Fatalf("SpeculativeVerifyDecision reject: %v", err)
	}
	if all2 || len(acc2) != 0 || repl2 != 0 {
		t.Fatalf("reject case: acc=%v repl=%d all=%v, want acc=[] repl=0 all=false", acc2, repl2, all2)
	}
}

// TestArgmaxFloats_Good covers the pure-Go argmax helper used by the pipelined
// decode loop's greedy step (dark because the synthetic decode tests drive the
// device argmax, not this CPU fallback).
func TestArgmaxFloats_Good(t *testing.T) {
	cases := []struct {
		name string
		in   []float32
		want int
	}{
		{"firstMax", []float32{3, 1, 2}, 0},
		{"midMax", []float32{1, 5, 2}, 1},
		{"lastMax", []float32{1, 2, 9}, 2},
		{"single", []float32{42}, 0},
		{"tieKeepsFirst", []float32{5, 5, 1}, 0},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := argmaxFloats(tc.in); got != tc.want {
				t.Fatalf("argmaxFloats(%v) = %d, want %d", tc.in, got, tc.want)
			}
		})
	}
}

// TestRandomUniformWithKey_Good covers the keyed uniform draw: a given PRNG key
// is deterministic (same key → same draw), a nil key falls back to the
// per-graph default, and every value lands in [low, high). The synthetic suite
// drives RandomUniform / RandomCategorical but not the keyed-uniform entry.
func TestRandomUniformWithKey_Good(t *testing.T) {
	requireMetalRuntime(t)

	shape := []int32{8}
	keyA := RandomKey(12345)
	defer Free(keyA)

	a := RandomUniformWithKey(0, 1, shape, DTypeFloat32, keyA)
	defer Free(a)
	va := a.Floats()
	if len(va) != 8 {
		t.Fatalf("RandomUniformWithKey produced %d values, want 8", len(va))
	}
	for i, v := range va {
		if v < 0 || v >= 1 {
			t.Fatalf("value[%d] = %v out of [0,1)", i, v)
		}
	}

	// Same key → identical draw (deterministic under an explicit key).
	keyB := RandomKey(12345)
	defer Free(keyB)
	b := RandomUniformWithKey(0, 1, shape, DTypeFloat32, keyB)
	defer Free(b)
	vb := b.Floats()
	for i := range va {
		if va[i] != vb[i] {
			t.Fatalf("same-key draw diverged at %d: %v vs %v", i, va[i], vb[i])
		}
	}

	// Nil key → the RandomUniform fallback branch, still a valid array.
	c := RandomUniformWithKey(2, 5, shape, DTypeFloat32, nil)
	defer Free(c)
	vc := c.Floats()
	if len(vc) != 8 {
		t.Fatalf("nil-key fallback produced %d values, want 8", len(vc))
	}
	for i, v := range vc {
		if v < 2 || v >= 5 {
			t.Fatalf("nil-key value[%d] = %v out of [2,5)", i, v)
		}
	}
}

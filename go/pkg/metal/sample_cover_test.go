// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import "testing"

// sample_cover_test.go covers the sampler-construction dispatch and the
// suppression-guard machinery the core sampling suite leaves dark: the
// NewSamplerWithSuppressionKeyed config arms (suppressed-greedy, fused
// top-k/top-p+suppress, min-p, plain chain), the uncompiled top-k/top-p lane,
// the multi-stage SampleTokenIDWithSuppressionGuard, and the host greedy
// fallback. Synthetic logits tensors, no model.

func sampleLogits(vals ...float32) *Array {
	return FromValues(vals, 1, len(vals))
}

// NewSamplerWithSuppressionKeyed routes config to distinct sampler types.
func TestSample_SuppressionKeyed_ConfigArms_Good(t *testing.T) {
	requireMetalRuntime(t)

	// Pure suppression with no temp/topP/minP/topK → suppressedGreedy. Logits
	// favour index 0, but it is suppressed → must pick the next-best (index 1).
	sg := NewSamplerWithSuppressionKeyed(0, 0, 0, 0, []int32{0}, nil)
	if _, ok := sg.(suppressedGreedy); !ok {
		t.Fatalf("pure-suppression sampler = %T, want suppressedGreedy", sg)
	}
	tok := sg.Sample(sampleLogits(100, 50, -10, -10))
	Materialize(tok)
	if tok.Int() != 1 {
		t.Errorf("suppressedGreedy picked %d, want 1 (0 suppressed)", tok.Int())
	}
	Free(tok)

	// topK>0 + topP in (0,1) + minP<=0 + no temp + suppression → fused
	// topKTopPChain carrying the suppress sampler.
	fused := NewSamplerWithSuppressionKeyed(0, 0.9, 0, 8, []int32{0}, nil)
	fusedChain, ok := fused.(*topKTopPChain)
	if !ok {
		t.Fatalf("fused sampler = %T, want *topKTopPChain", fused)
	}
	if fusedChain.suppress == nil {
		t.Error("fused topKTopPChain dropped its suppress sampler")
	}
	CloseSampler(fused)

	// minP>0 → MinPSampler appended into a chain.
	mp := NewSamplerWithSuppressionKeyed(0, 0, 0.05, 0, nil, nil)
	if _, ok := mp.(chain); !ok {
		t.Fatalf("min-p sampler = %T, want chain", mp)
	}
	CloseSampler(mp)

	// No constraints at all → Greedy.
	g := NewSamplerWithSuppressionKeyed(0, 0, 0, 0, nil, nil)
	if _, ok := g.(Greedy); !ok {
		t.Fatalf("unconstrained sampler = %T, want Greedy", g)
	}

	// temp + topP-only (topP<1) builds a chain with Temperature + TopP.
	tc := NewSamplerWithSuppressionKeyed(0.7, 0.9, 0, 0, nil, nil)
	if _, ok := tc.(chain); !ok {
		t.Fatalf("temp+topP sampler = %T, want chain", tc)
	}
	CloseSampler(tc)
}

// The uncompiled top-k/top-p lane runs when topK >= vocab (the compiled path is
// skipped). Drive it both with and without a fused suppress sampler.
func TestSample_TopKTopPUncompiled_Good(t *testing.T) {
	requireMetalRuntime(t)

	// topK (100) >= vocab (4) forces sampleTopKTopPTokenUncompiled.
	plain := NewSamplerWithSuppressionKeyed(0, 0.9, 0, 100, nil, nil)
	defer CloseSampler(plain)
	tok := plain.Sample(sampleLogits(-100, 100, -100, -100))
	Materialize(tok)
	if tok.Int() != 1 {
		t.Errorf("uncompiled top-k/top-p picked %d, want 1 (clear winner)", tok.Int())
	}
	Free(tok)

	// Same, but with a fused suppress on the winner → must avoid index 1.
	sup := NewSamplerWithSuppressionKeyed(0, 0.9, 0, 100, []int32{1}, nil)
	defer CloseSampler(sup)
	tok2 := sup.Sample(sampleLogits(-100, 100, 50, -100))
	Materialize(tok2)
	if tok2.Int() == 1 {
		t.Errorf("uncompiled suppressed lane picked the suppressed index 1")
	}
	Free(tok2)
}

// suppressTokenLogits handles nil logits, out-of-range / duplicate ids, and the
// all-filtered-out fast path.
func TestSample_SuppressTokenLogits_Edges_Good(t *testing.T) {
	requireMetalRuntime(t)

	if got := suppressTokenLogits(nil, []int32{1}); got != nil {
		t.Error("nil logits suppress should return nil")
	}

	logits := sampleLogits(1, 2, 3, 4)
	defer Free(logits)

	// Empty id list → a clone (no suppression).
	clone := suppressTokenLogits(logits, nil)
	Materialize(clone)
	Free(clone)

	// All ids out of range → clone (nothing valid to suppress).
	oor := suppressTokenLogits(logits, []int32{-1, 99, 4})
	Materialize(oor)
	Free(oor)

	// Mixed valid + duplicate + out-of-range: index 2 suppressed once.
	mixed := suppressTokenLogits(logits, []int32{2, 2, -5, 100})
	Materialize(mixed)
	got := mixed.Floats()
	if got[2] > -1e8 {
		t.Errorf("suppressed index 2 logit = %v, want very negative", got[2])
	}
	Free(mixed)
}

// SampleTokenIDWithSuppressionGuard: the clean path returns the sampler's token;
// when the sampler would pick a suppressed token, the guard re-suppresses and
// falls back to greedy.
func TestSample_SuppressionGuard_Paths_Good(t *testing.T) {
	requireMetalRuntime(t)

	// Clean path: greedy picks index 2, not suppressed → returned directly.
	logits := sampleLogits(-10, -10, 100, -10)
	defer Free(logits)
	next, id, _, err := SampleTokenIDWithSuppressionGuard(logits, Greedy{}, []int32{0}, false)
	if err != nil {
		t.Fatalf("clean guard path error: %v", err)
	}
	if id != 2 {
		t.Errorf("clean guard id = %d, want 2", id)
	}
	Free(next)

	// Fallback path: greedy would pick index 2, but 2 is suppressed → the guard
	// re-suppresses and greedily picks the next-best (index 0 here).
	next2, id2, _, err := SampleTokenIDWithSuppressionGuard(logits, Greedy{}, []int32{2}, true)
	if err != nil {
		t.Fatalf("fallback guard path error: %v", err)
	}
	if id2 == 2 {
		t.Errorf("fallback guard returned the suppressed token 2")
	}
	Free(next2)
}

// hostUnsuppressedGreedyToken errors on empty logits and argmaxes over the
// unsuppressed positions otherwise.
func TestSample_HostUnsuppressedGreedy_Good(t *testing.T) {
	requireMetalRuntime(t)

	if _, err := hostUnsuppressedGreedyToken(nil, nil); err == nil {
		t.Error("nil logits host-greedy returned nil error")
	}

	logits := sampleLogits(10, 100, 5, 50)
	defer Free(logits)
	// Suppress the true max (index 1) → must pick the next-best (index 3).
	tok, err := hostUnsuppressedGreedyToken(logits, []int32{1})
	if err != nil {
		t.Fatalf("host-greedy error: %v", err)
	}
	Materialize(tok)
	if tok.Int() != 3 {
		t.Errorf("host-greedy picked %d, want 3 (max suppressed)", tok.Int())
	}
	Free(tok)
}

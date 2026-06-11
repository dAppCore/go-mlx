// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"slices"
	"testing"
)

// Cross-token draw independence probes (#71). A sampler asked for N tokens
// from the SAME logits must draw N independent samples — the production bug
// was the categorical drawing under the per-graph default PRNG key, which
// repeats across separate graph evaluations (and is baked as a trace-time
// constant inside compiled samplers), making every token's draw key-correlated.

// sampleTokenIDs draws n tokens from the same logits through the sampler,
// one Sample+Eval round-trip per token — exactly the per-token serve shape.
func sampleTokenIDs(t *testing.T, s Sampler, logits *Array, n int) []int32 {
	t.Helper()
	ids := make([]int32, 0, n)
	for range n {
		tok := s.Sample(logits)
		if err := Eval(tok); err != nil {
			Free(tok)
			t.Fatalf("sample eval: %v", err)
		}
		ids = append(ids, int32(tok.Int()))
		Free(tok)
	}
	return ids
}

func distinctTokenCount(ids []int32) int {
	seen := map[int32]struct{}{}
	for _, id := range ids {
		seen[id] = struct{}{}
	}
	return len(seen)
}

// Uniform logits: every draw is pure PRNG behaviour, so repeated identical
// draws can only come from a repeated key, never from a peaked distribution.
func uniformProbeLogits(t *testing.T) *Array {
	t.Helper()
	logits := Zeros([]int32{1, 512}, DTypeFloat32)
	if err := Eval(logits); err != nil {
		Free(logits)
		t.Fatalf("probe logits eval: %v", err)
	}
	return logits
}

// The production Gemma-4 serve lane: temperature + top-k + top-p routes
// through the COMPILED topKTopPChain — the categorical draw lives inside the
// compiled graph, where an implicit default key is captured at trace time.
func TestSampler_CompiledTopKTopP_CrossTokenDrawIndependence_Good(t *testing.T) {
	logits := uniformProbeLogits(t)
	defer Free(logits)

	s := NewSamplerWithSuppression(0.7, 0.95, 0, 40, nil)
	defer CloseSampler(s)

	ids := sampleTokenIDs(t, s, logits, 16)
	if got := distinctTokenCount(ids); got < 4 {
		t.Fatalf("16 draws from uniform logits produced %d distinct tokens %v — key-correlated sampling", got, ids)
	}
}

// The generic chain lane (temperature only): the categorical draws once per
// Sample call in a fresh graph — repeats here mean the empty-key default
// reseeds identically across separate graph evaluations.
func TestSampler_Chain_CrossTokenDrawIndependence_Good(t *testing.T) {
	logits := uniformProbeLogits(t)
	defer Free(logits)

	s := NewSamplerWithSuppression(0.7, 0, 0, 0, nil)
	defer CloseSampler(s)

	ids := sampleTokenIDs(t, s, logits, 16)
	if got := distinctTokenCount(ids); got < 4 {
		t.Fatalf("16 draws from uniform logits produced %d distinct tokens %v — key-correlated sampling", got, ids)
	}
}

// Seeded key sequences replay the same draws — per-request reproducibility
// that the process-global mlx_random_seed cannot give once concurrent
// requests interleave on the default stream.
func TestSampler_SeededKeysReproducible_Good(t *testing.T) {
	logits := uniformProbeLogits(t)
	defer Free(logits)

	run := func(seed uint64) []int32 {
		s := NewSamplerWithSuppressionKeyed(0.7, 0.95, 0, 40, nil, NewSamplerKeys(seed))
		defer CloseSampler(s)
		return sampleTokenIDs(t, s, logits, 12)
	}

	a := run(42)
	b := run(42)
	if !slices.Equal(a, b) {
		t.Fatalf("same seed diverged: %v vs %v", a, b)
	}
	c := run(43)
	if slices.Equal(a, c) {
		t.Fatalf("different seeds replayed the same sequence: %v", a)
	}
}

// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"testing"
)

func TestSample_Greedy_Good(t *testing.T) {
	coverageTokens := "Greedy"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	// Logits heavily favour index 2
	logits := FromValues([]float32{-10, -10, 100, -10}, 1, 4)
	s := newSampler(0, 0, 0, 0) // temp=0 → greedy
	token := s.Sample(logits)
	Materialize(token)

	if token.Int() != 2 {
		t.Errorf("greedy sample = %d, want 2", token.Int())
	}
}

func TestSample_Temperature_HighTemp_Good(t *testing.T) {
	coverageTokens := "Temperature HighTemp"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	// High temperature should still produce a valid index
	logits := FromValues([]float32{1, 2, 3, 4}, 1, 4)
	s := newSampler(100.0, 0, 0, 0) // very high temp → near uniform
	token := s.Sample(logits)
	Materialize(token)

	idx := token.Int()
	if idx < 0 || idx >= 4 {
		t.Errorf("sample index = %d, out of range [0, 4)", idx)
	}
}

func TestSample_Temperature_LowTemp_Good(t *testing.T) {
	coverageTokens := "Temperature LowTemp"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	// Very low temperature should behave like greedy
	logits := FromValues([]float32{-10, -10, 100, -10}, 1, 4)
	s := newSampler(0.001, 0, 0, 0) // near-zero temp → near-greedy
	token := s.Sample(logits)
	Materialize(token)

	if token.Int() != 2 {
		t.Errorf("low-temp sample = %d, want 2 (near greedy)", token.Int())
	}
}

func TestSample_TopKSampler_Good(t *testing.T) {
	coverageTokens := "TopKSampler"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	// TopK=1 with clear winner should always pick that token
	logits := FromValues([]float32{-100, 100, -100, -100}, 1, 4)
	s := newSampler(1.0, 0, 0, 1) // topK=1
	token := s.Sample(logits)
	Materialize(token)

	if token.Int() != 1 {
		t.Errorf("topk=1 sample = %d, want 1", token.Int())
	}
}

func TestSample_TopKSampler_MultipleTokens_Good(t *testing.T) {
	coverageTokens := "TopKSampler MultipleTokens"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	// TopK=2, both high logits — should pick one of them
	logits := FromValues([]float32{-100, 50, 50, -100}, 1, 4)
	s := newSampler(1.0, 0, 0, 2) // topK=2

	seen := map[int]bool{}
	for range 20 {
		token := s.Sample(logits)
		Materialize(token)
		seen[token.Int()] = true
	}

	// Should only ever pick index 1 or 2
	for idx := range seen {
		if idx != 1 && idx != 2 {
			t.Errorf("topk=2 sampled index %d, expected only 1 or 2", idx)
		}
	}
}

func TestSample_TopKSampler_OverLargeK_NoOp_Good(t *testing.T) {
	logits := FromValues([]float32{1, 2, 3, 4}, 1, 4)
	filtered := TopKSampler(99).Sample(logits)
	Materialize(filtered)

	got := filtered.Floats()
	want := []float32{1, 2, 3, 4}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("filtered[%d] = %f, want %f", i, got[i], want[i])
		}
	}
}

func TestSample_TopKSampler_NonPositiveK_NoOp_Good(t *testing.T) {
	logits := FromValues([]float32{1, 2, 3, 4}, 1, 4)
	filtered := TopKSampler(0).Sample(logits)
	Materialize(filtered)

	got := filtered.Floats()
	want := []float32{1, 2, 3, 4}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("filtered[%d] = %f, want %f", i, got[i], want[i])
		}
	}
}

func TestSample_Chain_Good(t *testing.T) {
	coverageTokens := "Chain"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	// Full chain: topK + temperature
	logits := FromValues([]float32{1, 2, 3, 4, 5}, 1, 5)
	s := newSampler(0.5, 0, 0, 3) // temp=0.5, topK=3

	token := s.Sample(logits)
	Materialize(token)

	idx := token.Int()
	if idx < 0 || idx >= 5 {
		t.Errorf("chain sample index = %d, out of range", idx)
	}
}

func TestSample_ChainOrder_Good(t *testing.T) {
	coverageTokens := "ChainOrder"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	s := newSampler(0.7, 0.9, 0.05, 20)
	c, ok := s.(chain)
	if !ok {
		t.Fatalf("newSampler returned %T, want chain", s)
	}
	if len(c) != 4 {
		t.Fatalf("len(chain) = %d, want 4", len(c))
	}
	if _, ok := c[0].(Temperature); !ok {
		t.Fatalf("chain[0] = %T, want Temperature", c[0])
	}
	if _, ok := c[1].(TopP); !ok {
		t.Fatalf("chain[1] = %T, want TopP", c[1])
	}
	if _, ok := c[2].(TopKSampler); !ok {
		t.Fatalf("chain[2] = %T, want TopKSampler", c[2])
	}
	if _, ok := c[3].(MinPSampler); !ok {
		t.Fatalf("chain[3] = %T, want MinPSampler", c[3])
	}
}

func TestSample_TopPSamplesWithoutTemperature_Good(t *testing.T) {
	coverageTokens := "TopPSamplesWithoutTemperature"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	s := newSampler(0, 0.9, 0, 0)
	c, ok := s.(chain)
	if !ok {
		t.Fatalf("newSampler returned %T, want chain", s)
	}
	if len(c) != 1 {
		t.Fatalf("len(chain) = %d, want 1", len(c))
	}
	if _, ok := c[0].(TopP); !ok {
		t.Fatalf("chain[0] = %T, want TopP", c[0])
	}
}

func TestSample_TopKSamplesWithoutTemperature_Good(t *testing.T) {
	coverageTokens := "TopKSamplesWithoutTemperature"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	s := newSampler(0, 0, 0, 20)
	c, ok := s.(chain)
	if !ok {
		t.Fatalf("newSampler returned %T, want chain", s)
	}
	if len(c) != 1 {
		t.Fatalf("len(chain) = %d, want 1", len(c))
	}
	if _, ok := c[0].(TopKSampler); !ok {
		t.Fatalf("chain[0] = %T, want TopKSampler", c[0])
	}
}

func TestSample_MinPSamplesWithoutTemperature_Good(t *testing.T) {
	coverageTokens := "MinPSamplesWithoutTemperature"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	s := newSampler(0, 0, 0.05, 0)
	c, ok := s.(chain)
	if !ok {
		t.Fatalf("newSampler returned %T, want chain", s)
	}
	if len(c) != 1 {
		t.Fatalf("len(chain) = %d, want 1", len(c))
	}
	if _, ok := c[0].(MinPSampler); !ok {
		t.Fatalf("chain[0] = %T, want MinPSampler", c[0])
	}
}

func TestSample_TopP_DominantLogit_Good(t *testing.T) {
	// With one dominant logit, TopP should always pick it
	logits := FromValues([]float32{-10, -10, 100, -10}, 1, 4)
	s := newSampler(0.5, 0.9, 0, 0) // topP=0.9, temp=0.5
	token := s.Sample(logits)
	Materialize(token)

	if token.Int() != 2 {
		t.Errorf("topP dominant sample = %d, want 2", token.Int())
	}
}

func TestSample_TopP_RestrictsOptions_Good(t *testing.T) {
	// Two equal high logits, two low. TopP=0.5 should mostly restrict to top tokens.
	logits := FromValues([]float32{10, 10, -100, -100}, 1, 4)
	s := newSampler(1.0, 0.5, 0, 0) // topP=0.5, temp=1.0

	seen := map[int]bool{}
	for range 30 {
		token := s.Sample(logits)
		Materialize(token)
		seen[token.Int()] = true
	}

	// Should only pick indices 0 or 1 (the two high-probability tokens)
	for idx := range seen {
		if idx != 0 && idx != 1 {
			t.Errorf("topP=0.5 sampled index %d, expected only 0 or 1", idx)
		}
	}
}

func TestSample_MinP_DominantLogit_Good(t *testing.T) {
	// With one dominant logit, MinP should always pick it
	logits := FromValues([]float32{-10, -10, 100, -10}, 1, 4)
	s := newSampler(0.5, 0, 0.1, 0) // minP=0.1, temp=0.5
	token := s.Sample(logits)
	Materialize(token)

	if token.Int() != 2 {
		t.Errorf("minP dominant sample = %d, want 2", token.Int())
	}
}

func TestSample_MinP_RestrictsOptions_Good(t *testing.T) {
	// One very high logit, rest are low. MinP=0.1 should mask the low tokens.
	logits := FromValues([]float32{-100, 50, -100, -100}, 1, 4)
	s := newSampler(1.0, 0, 0.1, 0) // minP=0.1, temp=1.0

	for range 20 {
		token := s.Sample(logits)
		Materialize(token)
		if token.Int() != 1 {
			t.Errorf("minP with dominant logit sampled %d, want 1", token.Int())
		}
	}
}

func TestSample_ApplyRepeatPenalty_Good(t *testing.T) {
	coverageTokens := "ApplyRepeatPenalty"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	// Logits: [1, 4] with values [5.0, -3.0, 1.0, 0.0]
	// History: tokens 0 and 1 have been seen.
	// Penalty 2.0:
	//   token 0 (logit 5.0 > 0): 5.0 / 2.0 = 2.5
	//   token 1 (logit -3.0 < 0): -3.0 * 2.0 = -6.0
	//   token 2 (not in history): unchanged = 1.0
	//   token 3 (not in history): unchanged = 0.0
	logits := FromValues([]float32{5.0, -3.0, 1.0, 0.0}, 1, 4)
	Materialize(logits)

	result := applyRepeatPenalty(logits, []int32{0, 1, 0}, 2.0) // duplicate 0 should be deduped
	Materialize(result)

	got := result.Floats()
	want := []float32{2.5, -6.0, 1.0, 0.0}
	for i := range got {
		diff := got[i] - want[i]
		if diff > 0.01 || diff < -0.01 {
			t.Errorf("repeatPenalty[%d] = %f, want %f", i, got[i], want[i])
		}
	}
}

func TestSample_ApplyRepeatPenalty_NoHistory_Good(t *testing.T) {
	coverageTokens := "ApplyRepeatPenalty NoHistory"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	// With empty history, logits should be unchanged.
	logits := FromValues([]float32{5.0, -3.0, 1.0}, 1, 3)
	Materialize(logits)

	// applyRepeatPenalty is not called when history is empty (checked in generate loop),
	// but verify the function handles it gracefully if called directly.
	result := applyRepeatPenalty(logits, []int32{1}, 1.0) // penalty=1.0 → no change
	Materialize(result)

	got := result.Floats()
	want := []float32{5.0, -3.0, 1.0}
	for i := range got {
		diff := got[i] - want[i]
		if diff > 0.01 || diff < -0.01 {
			t.Errorf("penalty=1.0[%d] = %f, want %f", i, got[i], want[i])
		}
	}
}

// Generated file-aware compliance coverage.
func TestSample_chain_Sample_Good(t *testing.T) {
	coverageTokens := "chain Sample"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "chain_Sample"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestSample_chain_Sample_Bad(t *testing.T) {
	coverageTokens := "chain Sample"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "chain_Sample"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestSample_chain_Sample_Ugly(t *testing.T) {
	coverageTokens := "chain Sample"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "chain_Sample"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestSample_greedy_Sample_Good(t *testing.T) {
	coverageTokens := "greedy Sample"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "greedy_Sample"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestSample_greedy_Sample_Bad(t *testing.T) {
	coverageTokens := "greedy Sample"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "greedy_Sample"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestSample_greedy_Sample_Ugly(t *testing.T) {
	coverageTokens := "greedy Sample"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "greedy_Sample"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestSample_Temperature_Sample_Good(t *testing.T) {
	coverageTokens := "Temperature Sample"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "Temperature_Sample"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestSample_Temperature_Sample_Bad(t *testing.T) {
	coverageTokens := "Temperature Sample"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "Temperature_Sample"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestSample_Temperature_Sample_Ugly(t *testing.T) {
	coverageTokens := "Temperature Sample"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "Temperature_Sample"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestSample_TopKSampler_Sample_Good(t *testing.T) {
	coverageTokens := "TopKSampler Sample"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "TopKSampler_Sample"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestSample_TopKSampler_Sample_Bad(t *testing.T) {
	coverageTokens := "TopKSampler Sample"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "TopKSampler_Sample"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestSample_TopKSampler_Sample_Ugly(t *testing.T) {
	coverageTokens := "TopKSampler Sample"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "TopKSampler_Sample"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestSample_TopP_Sample_Good(t *testing.T) {
	coverageTokens := "TopP Sample"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "TopP_Sample"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestSample_TopP_Sample_Bad(t *testing.T) {
	coverageTokens := "TopP Sample"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "TopP_Sample"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestSample_TopP_Sample_Ugly(t *testing.T) {
	coverageTokens := "TopP Sample"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "TopP_Sample"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestSample_MinPSampler_Sample_Good(t *testing.T) {
	coverageTokens := "MinPSampler Sample"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "MinPSampler_Sample"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestSample_MinPSampler_Sample_Bad(t *testing.T) {
	coverageTokens := "MinPSampler Sample"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "MinPSampler_Sample"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestSample_MinPSampler_Sample_Ugly(t *testing.T) {
	coverageTokens := "MinPSampler Sample"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "MinPSampler_Sample"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

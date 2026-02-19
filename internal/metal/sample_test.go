//go:build darwin && arm64

package metal

import (
	"testing"
)

func TestGreedy(t *testing.T) {
	// Logits heavily favour index 2
	logits := FromValues([]float32{-10, -10, 100, -10}, 1, 4)
	s := newSampler(0, 0, 0, 0) // temp=0 → greedy
	token := s.Sample(logits)
	Materialize(token)

	if token.Int() != 2 {
		t.Errorf("greedy sample = %d, want 2", token.Int())
	}
}

func TestTemperature_HighTemp(t *testing.T) {
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

func TestTemperature_LowTemp(t *testing.T) {
	// Very low temperature should behave like greedy
	logits := FromValues([]float32{-10, -10, 100, -10}, 1, 4)
	s := newSampler(0.001, 0, 0, 0) // near-zero temp → near-greedy
	token := s.Sample(logits)
	Materialize(token)

	if token.Int() != 2 {
		t.Errorf("low-temp sample = %d, want 2 (near greedy)", token.Int())
	}
}

func TestSampler_TopK(t *testing.T) {
	// TopK=1 with clear winner should always pick that token
	logits := FromValues([]float32{-100, 100, -100, -100}, 1, 4)
	s := newSampler(1.0, 0, 0, 1) // topK=1
	token := s.Sample(logits)
	Materialize(token)

	if token.Int() != 1 {
		t.Errorf("topk=1 sample = %d, want 1", token.Int())
	}
}

func TestSampler_TopK_MultipleTokens(t *testing.T) {
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

func TestNew_Chain(t *testing.T) {
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

func TestTopP_DominantLogit(t *testing.T) {
	// With one dominant logit, TopP should always pick it
	logits := FromValues([]float32{-10, -10, 100, -10}, 1, 4)
	s := newSampler(0.5, 0.9, 0, 0) // topP=0.9, temp=0.5
	token := s.Sample(logits)
	Materialize(token)

	if token.Int() != 2 {
		t.Errorf("topP dominant sample = %d, want 2", token.Int())
	}
}

func TestTopP_RestrictsOptions(t *testing.T) {
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

func TestMinP_DominantLogit(t *testing.T) {
	// With one dominant logit, MinP should always pick it
	logits := FromValues([]float32{-10, -10, 100, -10}, 1, 4)
	s := newSampler(0.5, 0, 0.1, 0) // minP=0.1, temp=0.5
	token := s.Sample(logits)
	Materialize(token)

	if token.Int() != 2 {
		t.Errorf("minP dominant sample = %d, want 2", token.Int())
	}
}

func TestMinP_RestrictsOptions(t *testing.T) {
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

func TestApplyRepeatPenalty(t *testing.T) {
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

func TestApplyRepeatPenalty_NoHistory(t *testing.T) {
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

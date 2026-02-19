//go:build darwin && arm64

package sample

import (
	"testing"

	"forge.lthn.ai/core/go-mlx"
)

func TestGreedy(t *testing.T) {
	// Logits heavily favour index 2
	logits := mlx.FromValues([]float32{-10, -10, 100, -10}, 1, 4)
	s := New(0, 0, 0, 0) // temp=0 → greedy
	token := s.Sample(logits)
	mlx.Materialize(token)

	if token.Int() != 2 {
		t.Errorf("greedy sample = %d, want 2", token.Int())
	}
}

func TestTemperature_HighTemp(t *testing.T) {
	// High temperature should still produce a valid index
	logits := mlx.FromValues([]float32{1, 2, 3, 4}, 1, 4)
	s := New(100.0, 0, 0, 0) // very high temp → near uniform
	token := s.Sample(logits)
	mlx.Materialize(token)

	idx := token.Int()
	if idx < 0 || idx >= 4 {
		t.Errorf("sample index = %d, out of range [0, 4)", idx)
	}
}

func TestTemperature_LowTemp(t *testing.T) {
	// Very low temperature should behave like greedy
	logits := mlx.FromValues([]float32{-10, -10, 100, -10}, 1, 4)
	s := New(0.001, 0, 0, 0) // near-zero temp → near-greedy
	token := s.Sample(logits)
	mlx.Materialize(token)

	if token.Int() != 2 {
		t.Errorf("low-temp sample = %d, want 2 (near greedy)", token.Int())
	}
}

func TestTopK(t *testing.T) {
	// TopK=1 with clear winner should always pick that token
	logits := mlx.FromValues([]float32{-100, 100, -100, -100}, 1, 4)
	s := New(1.0, 0, 0, 1) // topK=1
	token := s.Sample(logits)
	mlx.Materialize(token)

	if token.Int() != 1 {
		t.Errorf("topk=1 sample = %d, want 1", token.Int())
	}
}

func TestTopK_MultipleTokens(t *testing.T) {
	// TopK=2, both high logits — should pick one of them
	logits := mlx.FromValues([]float32{-100, 50, 50, -100}, 1, 4)
	s := New(1.0, 0, 0, 2) // topK=2

	seen := map[int]bool{}
	for range 20 {
		token := s.Sample(logits)
		mlx.Materialize(token)
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
	logits := mlx.FromValues([]float32{1, 2, 3, 4, 5}, 1, 5)
	s := New(0.5, 0, 0, 3) // temp=0.5, topK=3

	token := s.Sample(logits)
	mlx.Materialize(token)

	idx := token.Int()
	if idx < 0 || idx >= 5 {
		t.Errorf("chain sample index = %d, out of range", idx)
	}
}

func TestTopP_PassThrough(t *testing.T) {
	// TopP is currently a stub — verify it doesn't break the chain
	logits := mlx.FromValues([]float32{-10, -10, 100, -10}, 1, 4)
	s := New(0.5, 0.9, 0, 0) // topP=0.9 (stub), temp=0.5
	token := s.Sample(logits)
	mlx.Materialize(token)

	if token.Int() != 2 {
		t.Errorf("topP stub + temp sample = %d, want 2", token.Int())
	}
}

func TestMinP_PassThrough(t *testing.T) {
	// MinP is currently a stub — verify it doesn't break the chain
	logits := mlx.FromValues([]float32{-10, -10, 100, -10}, 1, 4)
	s := New(0.5, 0, 0.1, 0) // minP=0.1 (stub), temp=0.5
	token := s.Sample(logits)
	mlx.Materialize(token)

	if token.Int() != 2 {
		t.Errorf("minP stub + temp sample = %d, want 2", token.Int())
	}
}

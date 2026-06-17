// SPDX-Licence-Identifier: EUPL-1.2

package model

import (
	"math"
	"testing"
)

// bf16Bytes encodes float32s to bf16 bytes (round-to-nearest-even), the dtype the LM
// head emits — for building test logits.
func bf16Bytes(vals []float32) []byte {
	out := make([]byte, len(vals)*bf16Size)
	for i, v := range vals {
		bits := math.Float32bits(v)
		var h uint16
		if bits&0x7fffffff > 0x7f800000 {
			h = uint16(bits>>16) | 0x0040
		} else {
			h = uint16((bits + ((bits>>16)&1 + 0x7fff)) >> 16)
		}
		out[i*bf16Size] = byte(h)
		out[i*bf16Size+1] = byte(h >> 8)
	}
	return out
}

func TestGreedy(t *testing.T) {
	logits := bf16Bytes([]float32{0.1, 0.5, -0.3, 0.5, 0.2}) // max 0.5 first at index 1
	got, err := Greedy(logits, 5)
	if err != nil {
		t.Fatalf("Greedy: %v", err)
	}
	if got != 1 {
		t.Fatalf("Greedy: got %d, want 1 (lowest-index of the tied max)", got)
	}
	if _, err := Greedy(logits, 4); err == nil {
		t.Fatal("expected a length-mismatch error")
	}
	t.Logf("greedy: argmax with lowest-index tie-break, length validated")
}

func TestSample(t *testing.T) {
	const vocab = 12
	// a spread distribution so stochastic draws actually vary.
	spread := make([]float32, vocab)
	for i := range spread {
		spread[i] = float32(i) * 0.4
	}
	spreadLogits := bf16Bytes(spread)
	argmax, _ := Greedy(spreadLogits, vocab) // = vocab-1 (largest)

	// temp <= 0 → greedy (matches Greedy, no RNG perturbation).
	g, err := NewSampler(1).Sample(spreadLogits, vocab, SampleParams{Temperature: 0})
	if err != nil {
		t.Fatalf("Sample temp0: %v", err)
	}
	if g != argmax {
		t.Fatalf("Sample temp0: got %d, want greedy %d", g, argmax)
	}

	// reproducible: two samplers, same seed, identical sequences.
	a, b := NewSampler(42), NewSampler(42)
	for i := 0; i < 32; i++ {
		ta, _ := a.Sample(spreadLogits, vocab, SampleParams{Temperature: 1})
		tb, _ := b.Sample(spreadLogits, vocab, SampleParams{Temperature: 1})
		if ta != tb {
			t.Fatalf("same seed diverged at draw %d: %d vs %d", i, ta, tb)
		}
	}

	// the RNG must advance (a single seed gives a VARYING sequence, not one repeated token).
	s := NewSampler(7)
	seen := map[int32]int{}
	for i := 0; i < 64; i++ {
		tok, _ := s.Sample(spreadLogits, vocab, SampleParams{Temperature: 1.5})
		seen[tok]++
	}
	if len(seen) < 2 {
		t.Fatalf("temperature sampling produced only %d distinct token(s) over 64 draws — RNG not advancing", len(seen))
	}

	// peaked distribution → temperature sampling lands on the peak ~always.
	peak := make([]float32, vocab)
	peak[5] = 30
	peakLogits := bf16Bytes(peak)
	ps := NewSampler(3)
	hits := 0
	for i := 0; i < 200; i++ {
		if tok, _ := ps.Sample(peakLogits, vocab, SampleParams{Temperature: 1}); tok == 5 {
			hits++
		}
	}
	if hits < 195 {
		t.Fatalf("peaked sampling hit the peak only %d/200 times", hits)
	}

	// top-k = 1 → always the argmax, regardless of temperature.
	ks := NewSampler(9)
	for i := 0; i < 32; i++ {
		if tok, _ := ks.Sample(spreadLogits, vocab, SampleParams{Temperature: 2, TopK: 1}); tok != argmax {
			t.Fatalf("top-k=1 draw %d returned %d, want argmax %d", i, tok, argmax)
		}
	}

	// tiny top-p → nucleus is just the top token → argmax.
	pp := NewSampler(11)
	for i := 0; i < 32; i++ {
		if tok, _ := pp.Sample(spreadLogits, vocab, SampleParams{Temperature: 1, TopP: 0.001}); tok != argmax {
			t.Fatalf("top-p=0.001 draw %d returned %d, want argmax %d", i, tok, argmax)
		}
	}

	if _, err := NewSampler(1).Sample(spreadLogits, vocab+1, SampleParams{Temperature: 1}); err == nil {
		t.Fatal("expected a length-mismatch error")
	}
	t.Logf("sample: temp0=greedy; same-seed reproducible; RNG advances (%d distinct/64); peaked→peak 195+/200; top-k=1 and tiny top-p →argmax", len(seen))
}

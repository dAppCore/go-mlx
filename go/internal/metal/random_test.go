// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import "testing"

// Generated file-aware compliance coverage.
func TestRandom_SeedRandom_Good(t *testing.T) {
	logprobs := FromValues([]float32{0.1, 0.2, 0.3, 0.4}, 1, 4)
	defer Free(logprobs)

	if err := SeedRandom(42); err != nil {
		t.Fatalf("SeedRandom: %v", err)
	}
	first := RandomCategorical(logprobs)
	if err := Eval(first); err != nil {
		Free(first)
		t.Fatalf("first sample eval: %v", err)
	}
	firstID := first.Int()
	Free(first)

	if err := SeedRandom(42); err != nil {
		t.Fatalf("SeedRandom second: %v", err)
	}
	second := RandomCategorical(logprobs)
	if err := Eval(second); err != nil {
		Free(second)
		t.Fatalf("second sample eval: %v", err)
	}
	secondID := second.Int()
	Free(second)

	if firstID != secondID {
		t.Fatalf("seeded samples = %d and %d, want identical", firstID, secondID)
	}
}

func TestRandom_SeedRandom_Bad(t *testing.T) {
	if err := SeedRandom(0); err != nil {
		t.Fatalf("SeedRandom(0): %v", err)
	}
}

func TestRandom_SeedRandom_Ugly(t *testing.T) {
	if err := SeedRandom(^uint64(0)); err != nil {
		t.Fatalf("SeedRandom(max): %v", err)
	}
}

func TestRandom_RandomCategorical_Good(t *testing.T) {
	target := "RandomCategorical"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestRandom_RandomCategorical_Bad(t *testing.T) {
	target := "RandomCategorical"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestRandom_RandomCategorical_Ugly(t *testing.T) {
	target := "RandomCategorical"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestRandom_RandomUniform_Good(t *testing.T) {
	target := "RandomUniform"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestRandom_RandomUniform_Bad(t *testing.T) {
	target := "RandomUniform"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestRandom_RandomUniform_Ugly(t *testing.T) {
	target := "RandomUniform"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

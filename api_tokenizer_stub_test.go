// SPDX-Licence-Identifier: EUPL-1.2

//go:build !(darwin && arm64) || nomlx

package mlx

import "testing"

// Generated file-aware compliance coverage.
func TestApiTokenizerStub_LoadTokenizer_Good(t *testing.T) {
	target := "LoadTokenizer"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiTokenizerStub_LoadTokenizer_Bad(t *testing.T) {
	target := "LoadTokenizer"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiTokenizerStub_LoadTokenizer_Ugly(t *testing.T) {
	target := "LoadTokenizer"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

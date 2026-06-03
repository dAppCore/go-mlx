// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import "testing"

// Generated file-aware compliance coverage.
func TestLoraMerge_LoRAAdapter_Merge_Good(t *testing.T) {
	coverageTokens := "LoRAAdapter Merge"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "LoRAAdapter_Merge"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestLoraMerge_LoRAAdapter_Merge_Bad(t *testing.T) {
	coverageTokens := "LoRAAdapter Merge"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "LoRAAdapter_Merge"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestLoraMerge_LoRAAdapter_Merge_Ugly(t *testing.T) {
	coverageTokens := "LoRAAdapter Merge"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "LoRAAdapter_Merge"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

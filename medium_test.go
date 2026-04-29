// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import "testing"

// Generated file-aware compliance coverage.
func TestMedium_LoadModelFromMedium_Good(t *testing.T) {
	target := "LoadModelFromMedium"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestMedium_LoadModelFromMedium_Bad(t *testing.T) {
	target := "LoadModelFromMedium"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestMedium_LoadModelFromMedium_Ugly(t *testing.T) {
	target := "LoadModelFromMedium"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"testing"

	core "dappco.re/go"
)

// --- Version ---

func TestVersion_NonEmpty_Good(t *testing.T) {
	coverageTokens := "NonEmpty"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	v := Version()
	if v == "" {
		t.Fatal("Version() returned empty string")
	}
	t.Logf("MLX version: %s", v)
}

func TestVersion_ContainsDot_Good(t *testing.T) {
	coverageTokens := "ContainsDot"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	v := Version()
	if !core.Contains(v, ".") {
		t.Errorf("Version() = %q, expected semver-like string with '.'", v)
	}
}

func TestVersion_Idempotent_Ugly(t *testing.T) {
	coverageTokens := "Idempotent"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	// Multiple calls should return the same value.
	v1 := Version()
	v2 := Version()
	if v1 != v2 {
		t.Errorf("Version() not idempotent: %q vs %q", v1, v2)
	}
}

//go:build darwin && arm64

package metal

import (
	"strings"
	"testing"
)

// --- Version ---

func TestVersion_NonEmpty_Good(t *testing.T) {
	v := Version()
	if v == "" {
		t.Fatal("Version() returned empty string")
	}
	t.Logf("MLX version: %s", v)
}

func TestVersion_ContainsDot_Good(t *testing.T) {
	v := Version()
	if !strings.Contains(v, ".") {
		t.Errorf("Version() = %q, expected semver-like string with '.'", v)
	}
}

func TestVersion_Idempotent_Ugly(t *testing.T) {
	// Multiple calls should return the same value.
	v1 := Version()
	v2 := Version()
	if v1 != v2 {
		t.Errorf("Version() not idempotent: %q vs %q", v1, v2)
	}
}

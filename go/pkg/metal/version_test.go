// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"testing"

	core "dappco.re/go"
)

func TestVersion(t *testing.T) {
	v := Version()
	if v == "" {
		t.Fatal("Version() returned empty string")
	}
	if !core.Contains(v, ".") {
		t.Errorf("Version() = %q, expected semver-like string with '.'", v)
	}
	if v2 := Version(); v != v2 {
		t.Errorf("Version() not idempotent: %q vs %q", v, v2)
	}
	t.Logf("MLX version: %s", v)
}

// SPDX-Licence-Identifier: EUPL-1.2

package grpo

import (
	"testing"

	core "dappco.re/go"
)

// writeModelPackFile is a small test helper carried with the package on
// extraction (it was an unexported root helper in distill_test.go, not
// importable across the package boundary).
func writeModelPackFile(t *testing.T, path string, data string) {
	t.Helper()
	if result := core.WriteFile(path, []byte(data), 0o644); !result.OK {
		t.Fatalf("write %s: %v", path, result.Value)
	}
}

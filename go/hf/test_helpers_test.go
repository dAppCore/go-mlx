// SPDX-Licence-Identifier: EUPL-1.2

package hf

import (
	"testing"

	core "dappco.re/go"
)

func writeModelPackFile(t *testing.T, path string, data string) {
	t.Helper()
	if result := core.WriteFile(path, []byte(data), 0o644); !result.OK {
		t.Fatalf("write %s: %v", path, result.Value)
	}
}

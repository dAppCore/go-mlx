// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"testing"

	core "dappco.re/go"
)

// writeModelPackFile writes a file into a test's temp model pack. Shared by
// several root tests (lora_fuse, model_slice, split_cpu_ffn); it previously
// lived in distill_test.go, which moved out to the distill subpackage, so its
// shared root home is here.
func writeModelPackFile(t *testing.T, path string, data string) {
	t.Helper()
	if result := core.WriteFile(path, []byte(data), 0o644); !result.OK {
		t.Fatalf("write %s: %v", path, result.Value)
	}
}

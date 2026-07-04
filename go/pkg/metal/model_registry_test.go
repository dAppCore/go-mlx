// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import "testing"

// TestModelRegistry exercises the registry-driven loader dispatch (go-mlx #45):
// every architecture the old central loadModel switch handled has a registered
// loader, while nested-config-only and unknown archs have none (loadModel routes
// those to the "unsupported architecture" error).
func TestModelRegistry(t *testing.T) {
	for _, arch := range []string{
		"mixtral", "gpt_oss", "kimi",
		"gemma3", "gemma3_text", "gemma2", "gemma4_text", "gemma4", "gemma4_unified",
	} {
		if lookupModelLoader(arch) == nil {
			t.Errorf("no model loader registered for %q", arch)
		}
	}
	for _, arch := range []string{"gemma4_unified_text", "totally-unknown-architecture"} {
		if lookupModelLoader(arch) != nil {
			t.Errorf("%q should have no standalone loader", arch)
		}
	}
}

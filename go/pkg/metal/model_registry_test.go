// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import "testing"

// TestModelRegistry_AllArchitecturesRegistered_Good pins that every architecture
// the old central loadModel switch handled has a registered loader, so the
// registry-driven dispatch is behaviour-preserving. go-mlx #45 (loader registry).
func TestModelRegistry_AllArchitecturesRegistered_Good(t *testing.T) {
	archs := []string{
		"mixtral", "gpt_oss", "kimi",
		"gemma3", "gemma3_text", "gemma2", "gemma4_text", "gemma4", "gemma4_unified",
	}
	for _, arch := range archs {
		if lookupModelLoader(arch) == nil {
			t.Errorf("no model loader registered for %q", arch)
		}
	}
}

func TestModelRegistry_Gemma4UnifiedTextIsNestedConfigOnly_Bad(t *testing.T) {
	if lookupModelLoader("gemma4_unified_text") != nil {
		t.Fatal("gemma4_unified_text should not be a standalone loader; load the outer gemma4_unified pack")
	}
}

// TestModelRegistry_UnknownArchitecture_Bad confirms an unregistered arch has no
// loader (loadModel routes it to the "unsupported architecture" error).
func TestModelRegistry_UnknownArchitecture_Bad(t *testing.T) {
	if lookupModelLoader("totally-unknown-architecture") != nil {
		t.Fatal("unknown architecture should have no registered loader")
	}
}

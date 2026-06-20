// SPDX-Licence-Identifier: EUPL-1.2

package gguf

import (
	"strings"
	"testing"
)

// TestInferLayerCount_OverLongArchitecture_Good is the regression test for the
// stack-scratch overrun fixed in inferLayerCount. The architecture string is
// untrusted GGUF metadata; before the fix, inferLayerCount copied it into a
// 128-byte stack array with no length guard, so an architecture name longer
// than 128 bytes panicked inside ReadInfo. The guard now routes over-long
// names through the alloc-allowed concat path (the same discipline as
// metadataIntForSuffix), so these calls must return cleanly without panicking.
func TestInferLayerCount_OverLongArchitecture_Good(t *testing.T) {
	longArch := strings.Repeat("z", 130) // > the 128-byte scratch

	// Metadata hit via the over-long concat fallback: "<longArch>.block_count"
	// is present, so the count is returned without touching the stack scratch.
	if got := inferLayerCount(map[string]any{longArch + ".block_count": uint32(42)}, nil, longArch); got != 42 {
		t.Fatalf("inferLayerCount(long arch, metadata hit) = %d, want 42", got)
	}

	// No layer-count metadata for the over-long arch: fall through to the
	// tensor-name scan (max layer index + 1) — must not panic.
	tensors := []ggufTensorInfo{
		{Name: "model.layers.0.attn.weight"},
		{Name: "model.layers.3.attn.weight"},
	}
	if got := inferLayerCount(map[string]any{}, tensors, longArch); got != 4 {
		t.Fatalf("inferLayerCount(long arch, tensor scan) = %d, want 4", got)
	}

	// Over-long arch, no metadata, no layer-indexed tensors -> 0, no panic.
	if got := inferLayerCount(map[string]any{}, []ggufTensorInfo{{Name: "token_embd.weight"}}, longArch); got != 0 {
		t.Fatalf("inferLayerCount(long arch, no layers) = %d, want 0", got)
	}

	// Boundary: architecture exactly at the scratch length (128) must also be
	// safe — the prefix ("<arch>.") no longer fits, so it takes the fallback.
	edgeArch := strings.Repeat("e", 128)
	if got := inferLayerCount(map[string]any{edgeArch + ".n_layer": uint32(7)}, nil, edgeArch); got != 7 {
		t.Fatalf("inferLayerCount(128-char arch, metadata hit) = %d, want 7", got)
	}
}

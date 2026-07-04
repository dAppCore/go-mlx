// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"testing"

	scheme "dappco.re/go/mlx/pkg/scheme"
)

// The quant + cache driver-compute schemes resolve from the registry and carry
// the metal compute surface — the contract #32/#33 (quant) and #34 (cache)
// implement against. Pure resolution, no Metal device work.
func TestSchemeComputeResolve_Good(t *testing.T) {
	q, ok := QuantComputeFor("affine")
	if !ok {
		t.Fatal("affine quant compute did not resolve")
	}
	if q.Kind() != "affine" {
		t.Errorf("quant kind = %q, want affine", q.Kind())
	}

	for _, mode := range []string{"default", "fixed", "paged", "q8", "k-q8-v-q4"} {
		c, ok := CacheComputeFor(mode)
		if !ok {
			t.Errorf("cache compute %q did not resolve", mode)
			continue
		}
		if c.Serves() != scheme.StateKVCache {
			t.Errorf("cache %q serves %v, want kv-cache", mode, c.Serves())
		}
	}
}

// An unregistered cache mode resolves to nothing — the engine refuses it
// cleanly rather than calling a stub. ("recurrent" used to sit here as a
// metadata-only seed; #39 attached its compute — see cache_recurrent_test.go.)
func TestSchemeComputeUnregistered_Bad(t *testing.T) {
	if _, ok := CacheComputeFor("no-such-cache-mode"); ok {
		t.Error("an unregistered cache mode must not resolve to a compute scheme")
	}
}

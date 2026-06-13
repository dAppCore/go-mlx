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

// A metadata-only catalogue entry (no compute attached) resolves in pkg/scheme
// but reports (nil,false) here, so the engine refuses it cleanly.
func TestSchemeComputeMetadataOnly_Bad(t *testing.T) {
	// "recurrent" is registered as metadata in pkg/scheme but has no metal
	// CacheCompute yet — it must not pass the compute assertion.
	if _, ok := CacheComputeFor("recurrent"); ok {
		t.Error("recurrent should have no metal CacheCompute yet")
	}
}

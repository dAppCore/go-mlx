// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"testing"

	scheme "dappco.re/go/mlx/pkg/scheme"
)

// TestSparseCacheSchemes_Resolve_Good proves all three sparse KV-cache modes
// resolve through CacheComputeFor as compute-bearing schemes that Serve
// StateKVCache (so scheme.Compatible pairs them with their StateKVCache mixers).
func TestSparseCacheSchemes_Resolve_Good(t *testing.T) {
	for _, mode := range []string{"mla-latent", "nsa", "moba"} {
		cc, ok := CacheComputeFor(mode)
		if !ok {
			t.Errorf("CacheComputeFor(%q) = false, want a compute-bearing scheme", mode)
			continue
		}
		if cc.Serves() != scheme.StateKVCache {
			t.Errorf("%q Serves() = %v, want StateKVCache", mode, cc.Serves())
		}
		if cc.Mode() != mode {
			t.Errorf("scheme Mode() = %q, want %q", cc.Mode(), mode)
		}
	}
}

// TestSparseCacheSchemes_Unbounded_Good confirms MaxSize 0 builds a growing
// cache and an Update round-trips: appending 2 tokens then 1 more yields a
// cache of length 3 with the appended K/V readable.
func TestSparseCacheSchemes_Unbounded_Good(t *testing.T) {
	for _, mode := range []string{"mla-latent", "nsa", "moba"} {
		cc, _ := CacheComputeFor(mode)
		c := cc.NewCache(CacheParams{}) // MaxSize 0 → growing

		// [B=1, H=1, L=2, D=2] then one more token.
		k0 := FromValues([]float32{1, 2, 3, 4}, 1, 1, 2, 2)
		v0 := FromValues([]float32{5, 6, 7, 8}, 1, 1, 2, 2)
		c.Update(k0, v0, 2)
		k1 := FromValues([]float32{9, 10}, 1, 1, 1, 2)
		v1 := FromValues([]float32{11, 12}, 1, 1, 1, 2)
		fullK, _ := c.Update(k1, v1, 1)

		if got := c.Offset(); got != 3 {
			t.Errorf("%q cache Offset after 2+1 tokens = %d, want 3", mode, got)
		}
		// The growing cache pre-allocates in chunks; the live K spans at least the
		// 3 appended tokens. Read the third token's K row back.
		if fullK == nil || fullK.Dim(2) < 3 {
			t.Errorf("%q cache K seq dim = %v, want >= 3", mode, fullK.Dim(2))
		} else {
			row := SliceAxis(fullK, 2, 2, 3) // token index 2 (the appended one)
			got := row.Floats()
			if len(got) >= 2 && (got[0] != 9 || got[1] != 10) {
				t.Errorf("%q appended K row = %v, want [9 10]", mode, got[:2])
			}
			Free(row)
		}
		c.Reset()
	}
}

// TestSparseCacheSchemes_Bounded_Good confirms MaxSize > 0 builds a rotating
// cache that caps its retained window — the bounded-build path the schemes take
// for a sliding / footprint-limited decode.
func TestSparseCacheSchemes_Bounded_Good(t *testing.T) {
	for _, mode := range []string{"mla-latent", "nsa", "moba"} {
		cc, _ := CacheComputeFor(mode)
		c := cc.NewCache(CacheParams{MaxSize: 2})
		if _, ok := c.(*RotatingKVCache); !ok {
			t.Errorf("%q with MaxSize>0 built %T, want *RotatingKVCache", mode, c)
		}
		c.Reset()
	}
}

// TestMLALatentCache_StoresLatentWidth_Good confirms the mla-latent scheme is
// width-agnostic: it stores whatever last dimension it is handed (the small
// latent), not a fixed full-K/V width. Feeding a narrow latent [B,1,L,3]
// round-trips at width 3.
func TestMLALatentCache_StoresLatentWidth_Good(t *testing.T) {
	cc, _ := CacheComputeFor("mla-latent")
	c := cc.NewCache(CacheParams{})

	// Latent width 3 (not a full head dim) — the compressed c_kv.
	latent := FromValues([]float32{1, 2, 3}, 1, 1, 1, 3)
	val := FromValues([]float32{4, 5, 6}, 1, 1, 1, 3)
	fullK, _ := c.Update(latent, val, 1)

	if fullK == nil || fullK.Dim(3) != 3 {
		t.Fatalf("mla-latent stored width = %v, want 3 (the latent width)", fullK.Dim(3))
	}
	row := SliceAxis(fullK, 2, 0, 1)
	got := row.Floats()
	if len(got) != 3 || got[0] != 1 || got[1] != 2 || got[2] != 3 {
		t.Errorf("mla-latent stored latent = %v, want [1 2 3]", got)
	}
	Free(row)
	c.Reset()
}

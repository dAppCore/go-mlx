// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"testing"

	scheme "dappco.re/go/mlx/pkg/scheme"
)

// TestMLALatentCacheScheme_Resolve_Good proves the mla-latent mode resolves
// through CacheComputeFor as a compute-bearing scheme that Serves StateKVCache
// (so scheme.Compatible pairs it with the MLA mixer).
func TestMLALatentCacheScheme_Resolve_Good(t *testing.T) {
	cc, ok := CacheComputeFor("mla-latent")
	if !ok {
		t.Fatal("CacheComputeFor(mla-latent) = false, want a compute-bearing scheme")
	}
	if cc.Serves() != scheme.StateKVCache {
		t.Errorf("Serves() = %v, want StateKVCache", cc.Serves())
	}
	if cc.Mode() != "mla-latent" {
		t.Errorf("Mode() = %q, want mla-latent", cc.Mode())
	}
}

// TestSparseCacheSchemes_NoRedundantModes_Good guards Cladius's decision that
// NSA / MoBA get NO bespoke cache scheme (they reuse the existing growing /
// rotating KV cache; their decode work is mixer-side). Registering "nsa" /
// "moba" modes would be redundant catalogue pollution — assert they are absent.
func TestSparseCacheSchemes_NoRedundantModes_Good(t *testing.T) {
	for _, mode := range []string{"nsa", "moba"} {
		if _, ok := CacheComputeFor(mode); ok {
			t.Errorf("CacheComputeFor(%q) resolved — NSA/MoBA must NOT register a cache scheme (use the existing KV cache)", mode)
		}
	}
}

// TestMLALatentCache_Unbounded_Good confirms MaxSize 0 builds a growing cache
// and an Update round-trips: appending 2 tokens then 1 more yields offset 3 with
// the appended latent readable.
func TestMLALatentCache_Unbounded_Good(t *testing.T) {
	cc, _ := CacheComputeFor("mla-latent")
	c := cc.NewCache(CacheParams{}) // MaxSize 0 → growing

	// Latent stored as [B=1, L, latentDim=2] — the shape the MLA mixer hands the
	// cache (WDKV.Forward(x): sequence on axis 1; v is unused).
	k0 := FromValues([]float32{1, 2, 3, 4}, 1, 2, 2)
	v0 := FromValues([]float32{5, 6, 7, 8}, 1, 2, 2)
	c.Update(k0, v0, 2)
	k1 := FromValues([]float32{9, 10}, 1, 1, 2)
	v1 := FromValues([]float32{11, 12}, 1, 1, 2)
	fullK, _ := c.Update(k1, v1, 1)

	if got := c.Offset(); got != 3 {
		t.Errorf("cache Offset after 2+1 tokens = %d, want 3", got)
	}
	if fullK == nil || fullK.Dim(1) < 3 {
		t.Fatalf("cache K seq dim = %v, want >= 3", fullK.Dim(1))
	}
	row := SliceAxis(fullK, 1, 2, 3) // the appended token
	got := row.Floats()
	if len(got) >= 2 && (got[0] != 9 || got[1] != 10) {
		t.Errorf("appended latent row = %v, want [9 10]", got[:2])
	}
	Free(row)
	c.Reset()
}

// TestMLALatentCache_Bounded_Good confirms MaxSize > 0 builds the bounded
// rotatingLatentKVCache — the footprint-limited decode path — and that its
// window caps the stored latents at maxSize while Offset keeps the true count.
func TestMLALatentCache_Bounded_Good(t *testing.T) {
	cc, _ := CacheComputeFor("mla-latent")
	c := cc.NewCache(CacheParams{MaxSize: 2})
	if _, ok := c.(*rotatingLatentKVCache); !ok {
		t.Fatalf("mla-latent with MaxSize>0 built %T, want *rotatingLatentKVCache", c)
	}

	// Feed 3 single-token latents [B=1, L=1, latentDim=2] into a window of 2.
	for i := range 3 {
		k := FromValues([]float32{float32(i), float32(i)}, 1, 1, 2)
		c.Update(k, k, 1)
	}
	if got := c.Offset(); got != 3 {
		t.Errorf("rotating latent Offset after 3 tokens = %d, want 3 (monotonic)", got)
	}
	if got := c.Len(); got != 2 {
		t.Errorf("rotating latent Len after 3 tokens (maxSize 2) = %d, want 2 (windowed)", got)
	}
	if st := c.State(); len(st) != 1 || st[0].Dim(1) != 2 {
		t.Errorf("rotating latent stored seq dim = %v, want a single [_,2,_] tensor", st)
	}
	c.Reset()
	if got := c.Offset(); got != 0 {
		t.Errorf("Offset after Reset = %d, want 0", got)
	}
}

// TestMLALatentCache_StoresLatentWidth_Good confirms the scheme is width-
// agnostic: it stores whatever latent width it is handed (the small latent), not
// a fixed full-K/V width. Feeding a narrow latent [B, L, 3] round-trips at
// width 3.
func TestMLALatentCache_StoresLatentWidth_Good(t *testing.T) {
	cc, _ := CacheComputeFor("mla-latent")
	c := cc.NewCache(CacheParams{})

	latent := FromValues([]float32{1, 2, 3}, 1, 1, 3) // [B=1, L=1, latentDim=3]
	val := FromValues([]float32{4, 5, 6}, 1, 1, 3)
	fullK, _ := c.Update(latent, val, 1)

	if fullK == nil || fullK.Dim(2) != 3 {
		t.Fatalf("mla-latent stored width = %v, want 3 (the latent width)", fullK.Dim(2))
	}
	row := SliceAxis(fullK, 1, 0, 1)
	got := row.Floats()
	if len(got) != 3 || got[0] != 1 || got[1] != 2 || got[2] != 3 {
		t.Errorf("mla-latent stored latent = %v, want [1 2 3]", got)
	}
	Free(row)
	c.Reset()
}

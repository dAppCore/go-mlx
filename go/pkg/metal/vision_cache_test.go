// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import "testing"

func visionCacheTestFeatures(t *testing.T, value float32) *Array {
	t.Helper()
	return FromValues([]float32{value}, 1, 1)
}

// Hit-path safety: the hit hands back an independent handle on the same
// immutable buffer — freeing the request's clone (or evicting the cache's
// handle) never touches the other side.
func TestVisionFeatureCache_CloneOnHit_Good(t *testing.T) {
	requireMetalRuntime(t)
	c := newVisionFeatureCache(2)
	key := visionFeatureCacheKey([]byte("image-bytes"))
	c.put(key, visionCacheTestFeatures(t, 7), 42)

	first, softTokens, ok := c.get(key)
	if !ok || softTokens != 42 {
		t.Fatalf("get = ok=%v soft=%d, want hit with 42", ok, softTokens)
	}
	Free(first) // request done — cache entry must survive

	second, _, ok := c.get(key)
	if !ok {
		t.Fatal("entry vanished after a request freed its clone")
	}
	Materialize(second)
	if got := second.Float(); got != 7 {
		t.Fatalf("cached features corrupted: %v", got)
	}
	Free(second)

	hits, misses := c.stats()
	if hits != 2 || misses != 0 {
		t.Fatalf("stats = %d/%d, want 2 hits 0 misses", hits, misses)
	}
}

// LRU order: filling past capacity evicts the least-recently-USED entry —
// a get refreshes recency, a put of a new key drops the stale tail.
func TestVisionFeatureCache_LRUEviction_Good(t *testing.T) {
	requireMetalRuntime(t)
	c := newVisionFeatureCache(2)
	keyA := visionFeatureCacheKey([]byte("a"))
	keyB := visionFeatureCacheKey([]byte("b"))
	keyC := visionFeatureCacheKey([]byte("c"))
	c.put(keyA, visionCacheTestFeatures(t, 1), 1)
	c.put(keyB, visionCacheTestFeatures(t, 2), 2)

	// Touch A so B becomes the tail, then insert C.
	if f, _, ok := c.get(keyA); !ok {
		t.Fatal("A missing before eviction round")
	} else {
		Free(f)
	}
	c.put(keyC, visionCacheTestFeatures(t, 3), 3)

	if _, _, ok := c.get(keyB); ok {
		t.Fatal("B survived — eviction is not least-recently-used")
	}
	for _, key := range []visionFeatureKey{keyA, keyC} {
		f, _, ok := c.get(key)
		if !ok {
			t.Fatal("a recent entry was evicted")
		}
		Free(f)
	}
	if len(c.entries) != 2 || len(c.order) != 2 {
		t.Fatalf("cache size = %d/%d entries/order, want 2/2", len(c.entries), len(c.order))
	}
}

// Replacing a key frees the old handle and keeps exactly one entry; freeAll
// empties without breaking later use.
func TestVisionFeatureCache_ReplaceAndFreeAll_Good(t *testing.T) {
	requireMetalRuntime(t)
	c := newVisionFeatureCache(2)
	key := visionFeatureCacheKey([]byte("img"))
	c.put(key, visionCacheTestFeatures(t, 1), 10)
	c.put(key, visionCacheTestFeatures(t, 9), 20)
	if len(c.entries) != 1 {
		t.Fatalf("entries = %d after replace, want 1", len(c.entries))
	}
	f, softTokens, ok := c.get(key)
	if !ok || softTokens != 20 {
		t.Fatalf("replaced entry = ok=%v soft=%d, want 20", ok, softTokens)
	}
	Materialize(f)
	if f.Float() != 9 {
		t.Fatal("replace kept the old features")
	}
	Free(f)

	c.freeAll()
	if _, _, ok := c.get(key); ok {
		t.Fatal("entry survived freeAll")
	}
	c.put(key, visionCacheTestFeatures(t, 4), 5)
	if _, _, ok := c.get(key); !ok {
		t.Fatal("cache unusable after freeAll")
	}
	c.freeAll()
}

// Nil and invalid shapes no-op without panics.
func TestVisionFeatureCache_NilSafety_Ugly(t *testing.T) {
	var nilCache *visionFeatureCache
	if _, _, ok := nilCache.get(visionFeatureKey{}); ok {
		t.Fatal("nil cache must miss")
	}
	nilCache.put(visionFeatureKey{}, nil, 0)
	nilCache.freeAll()
	if h, m := nilCache.stats(); h != 0 || m != 0 {
		t.Fatal("nil cache stats must be zero")
	}
	c := newVisionFeatureCache(0)
	if c.capacity != defaultVisionFeatureCacheEntries {
		t.Fatalf("default capacity = %d, want %d", c.capacity, defaultVisionFeatureCacheEntries)
	}
	c.put(visionFeatureKey{1}, nil, 3) // invalid features refused
	if _, _, ok := c.get(visionFeatureKey{1}); ok {
		t.Fatal("nil features must not be cached")
	}
}

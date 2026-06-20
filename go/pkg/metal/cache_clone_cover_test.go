// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"errors"
	"testing"
)

// cache_clone_cover_test.go fills out CloneCachePrefix beyond the single paged
// arm the relocated test covers: the empty-cache fast clone for every concrete
// type, the populated KV / rotating / fixed / quantized arms, the multi-cache
// CloneCachePrefixes wrapper, and the nil-cache guard. Synthetic K/V tensors.

func TestCloneCachePrefix_NilGuard_Bad(t *testing.T) {
	if _, err := CloneCachePrefix(nil); !errors.Is(err, errCloneCacheNil) {
		t.Fatalf("nil cache → %v, want errCloneCacheNil", err)
	}
}

// Every concrete cache type, while still empty (Len == 0), clones to a fresh
// cache of the same type and geometry via the empty-clone switch.
func TestCloneCachePrefix_EmptyArms_Good(t *testing.T) {
	requireMetalRuntime(t)

	cases := []struct {
		name  string
		cache Cache
	}{
		{"kv", NewKVCache()},
		{"rotating", NewRotatingKVCache(512)},
		{"fixed", NewFixedKVCache(512)},
		{"paged", NewPagedKVCache(0, 4)},
		{"paged-with-dtype", NewPagedKVCacheWithDTypeAndPrealloc(0, 4, DTypeFloat16, true)},
		{"quantized", NewQuantizedKVCache(512, 8, 4)},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			clone, err := CloneCachePrefix(tc.cache)
			if err != nil {
				t.Fatalf("empty %s clone: %v", tc.name, err)
			}
			defer FreeCaches([]Cache{clone})
			if clone.Len() != 0 {
				t.Errorf("empty clone Len = %d, want 0", clone.Len())
			}
		})
	}
}

func TestCloneCachePrefix_PopulatedKVCache_Good(t *testing.T) {
	requireMetalRuntime(t)

	c := NewKVCache()
	k, v := makeKV(3)
	c.Update(k, v, 3)
	Free(k, v)
	defer FreeCaches([]Cache{c})

	clone, err := CloneCachePrefix(c)
	if err != nil {
		t.Fatalf("populated KVCache clone: %v", err)
	}
	defer FreeCaches([]Cache{clone})
	kv, ok := clone.(*KVCache)
	if !ok {
		t.Fatalf("clone = %T, want *KVCache", clone)
	}
	if kv.Len() != 3 {
		t.Errorf("clone Len = %d, want 3", kv.Len())
	}
	// The clone is independent: extending it leaves the original untouched.
	k2, v2 := makeKV(1)
	kv.Update(k2, v2, 1)
	Free(k2, v2)
	if c.Len() != 3 {
		t.Errorf("original Len = %d after extending clone, want 3 (independent)", c.Len())
	}
}

func TestCloneCachePrefix_PopulatedRotating_Good(t *testing.T) {
	requireMetalRuntime(t)

	c := NewRotatingKVCache(512)
	k, v := makeKV(4)
	c.Update(k, v, 4)
	Free(k, v)
	defer FreeCaches([]Cache{c})

	clone, err := CloneCachePrefix(c)
	if err != nil {
		t.Fatalf("populated rotating clone: %v", err)
	}
	defer FreeCaches([]Cache{clone})
	if _, ok := clone.(*RotatingKVCache); !ok {
		t.Fatalf("clone = %T, want *RotatingKVCache", clone)
	}
	if clone.Len() != 4 {
		t.Errorf("rotating clone Len = %d, want 4", clone.Len())
	}
}

func TestCloneCachePrefix_PopulatedFixed_Good(t *testing.T) {
	requireMetalRuntime(t)

	c := NewFixedKVCache(512)
	k, v := makeKV(2)
	c.Update(k, v, 2)
	Free(k, v)
	defer FreeCaches([]Cache{c})

	clone, err := CloneCachePrefix(c)
	if err != nil {
		t.Fatalf("populated fixed clone: %v", err)
	}
	defer FreeCaches([]Cache{clone})
	if _, ok := clone.(*FixedKVCache); !ok {
		t.Fatalf("clone = %T, want *FixedKVCache", clone)
	}
}

func TestCloneCachePrefix_PopulatedQuantized_Good(t *testing.T) {
	requireMetalRuntime(t)

	c := NewQuantizedKVCache(512, 8, 4)
	k, v := makeKV(3)
	c.Update(k, v, 3)
	Free(k, v)
	defer FreeCaches([]Cache{c})

	clone, err := CloneCachePrefix(c)
	if err != nil {
		t.Fatalf("populated quantized clone: %v", err)
	}
	defer FreeCaches([]Cache{clone})
	qc, ok := clone.(*QuantizedKVCache)
	if !ok {
		t.Fatalf("clone = %T, want *QuantizedKVCache", clone)
	}
	if qc.keyBits != 8 || qc.valueBits != 4 {
		t.Errorf("clone bits = %d/%d, want 8/4", qc.keyBits, qc.valueBits)
	}
}

// CloneCachePrefixes clones a whole slice, preserving order and independence.
func TestCloneCachePrefixes_Slice_Good(t *testing.T) {
	requireMetalRuntime(t)

	a := NewKVCache()
	b := NewFixedKVCache(512)
	k, v := makeKV(2)
	a.Update(k, v, 2)
	Free(k, v)
	originals := []Cache{a, b}
	defer FreeCaches(originals)

	clones, err := CloneCachePrefixes(originals)
	if err != nil {
		t.Fatalf("CloneCachePrefixes: %v", err)
	}
	defer FreeCaches(clones)
	if len(clones) != 2 {
		t.Fatalf("cloned %d caches, want 2", len(clones))
	}
	if clones[0].Len() != 2 {
		t.Errorf("first clone Len = %d, want 2", clones[0].Len())
	}
	if clones[1].Len() != 0 {
		t.Errorf("second clone Len = %d, want 0", clones[1].Len())
	}
}

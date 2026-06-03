// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"testing"

	core "dappco.re/go"
)

func requireCacheCloneMetalRuntime(t testing.TB) {
	t.Helper()
	if core.Getenv("GO_MLX_RUN_METAL_TESTS") != "1" {
		t.Skip("set GO_MLX_RUN_METAL_TESTS=1 to enable Metal runtime tests")
	}
	if !MetalAvailable() {
		t.Skip("Metal runtime unavailable")
	}
}

// TestCloneCachePrefix_PagedKeepsPageLens checks that CloneCachePrefix copies a
// paged cache's page-length bookkeeping so the clone keeps appending correctly.
// Relocated from the gemma4 assistant tests when the clone logic moved into
// metal; it asserts metal-private paged-cache state, so it lives here.
func TestCloneCachePrefix_PagedKeepsPageLens(t *testing.T) {
	requireCacheCloneMetalRuntime(t)

	cache := NewPagedKVCache(0, 4)
	k := FromValues([]float32{1, 2, 3, 4}, 1, 1, 2, 2)
	v := FromValues([]float32{5, 6, 7, 8}, 1, 1, 2, 2)
	cache.UpdatePages(k, v, 2).Free()
	Free(k, v)
	defer FreeCaches([]Cache{cache})

	clonedCache, err := CloneCachePrefix(cache)
	if err != nil {
		t.Fatalf("CloneCachePrefix: %v", err)
	}
	defer FreeCaches([]Cache{clonedCache})
	cloned, ok := clonedCache.(*PagedKVCache)
	if !ok {
		t.Fatalf("cloned cache = %T, want *PagedKVCache", clonedCache)
	}
	if len(cloned.pageLens) != len(cloned.kPages) || cloned.pageLen(0) != 2 {
		t.Fatalf("cloned page lens = %v for %d pages, want [2]", cloned.pageLens, len(cloned.kPages))
	}

	nextK := FromValues([]float32{9, 10}, 1, 1, 1, 2)
	nextV := FromValues([]float32{11, 12}, 1, 1, 1, 2)
	cloned.UpdatePages(nextK, nextV, 1).Free()
	Free(nextK, nextV)
	if cloned.Len() != 3 || cloned.pageLen(0) != 3 {
		t.Fatalf("cloned cache len/page = %d/%d, want 3/3", cloned.Len(), cloned.pageLen(0))
	}
}

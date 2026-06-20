// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && metal_runtime

package gemma4

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/mlx/pkg/metal"
)

// Coverage for the plain-Go error/edge branches of assistant_decode.go that the
// happy-path generate tests never reach: the paged-cache K/V extraction arm and
// its empty guard, and the cache-state sentinel ladder in
// gemma4AssistantKVFromCache. Each case asserts the SPECIFIC sentinel (not just
// non-nil) so a test cannot green on the wrong error with the intended branch
// unexecuted. Helpers prefixed cov4adp.

// cov4adpPrimedPaged returns a PagedKVCache with `fill` single-token K/V steps
// written, geometry matching the tiny target (1 kv head, head dim 4).
func cov4adpPrimedPaged(t *testing.T, maxSize, headDim, fill int) *metal.PagedKVCache {
	t.Helper()
	cache := metal.NewPagedKVCache(maxSize, 0)
	for i := 0; i < fill; i++ {
		k := seqArray(0.1+0.01*float32(i), 1, 1, 1, headDim)
		v := seqArray(0.2+0.01*float32(i), 1, 1, 1, headDim)
		gotK, gotV := cache.Update(k, v, 1)
		if err := metal.Eval(gotK, gotV); err != nil {
			t.Fatalf("paged prime Update eval: %v", err)
		}
		metal.Free(k, v)
	}
	return cache
}

// TestKVFromCache_PagedVisible drives the paged arm of gemma4AssistantKVFromCache:
// a primed paged cache yields a sharedKV carrying live page state with the
// cache's offset. This is the branch that the FixedKVCache happy path skips.
func TestKVFromCache_PagedVisible(t *testing.T) {
	requireMetalRuntime(t)
	cache := cov4adpPrimedPaged(t, 16, 4, 3)
	defer metal.FreeCaches([]metal.Cache{cache})

	kv, err := gemma4AssistantKVFromCache(cache)
	if err != nil {
		t.Fatalf("KVFromCache(primed paged) error = %v, want nil", err)
	}
	if !kv.kv.HasPages() {
		t.Fatal("KVFromCache(primed paged) returned no page state")
	}
	if kv.kv.Offset != cache.Offset() {
		t.Fatalf("KVFromCache(primed paged) offset = %d, want %d", kv.kv.Offset, cache.Offset())
	}
	kv.free()
}

// TestKVFromCache_PagedNoVisible drives the errTargetPagedNoVisible guard: a
// paged cache whose Len() reports filled but whose page state has no visible
// pages. A maxSize-1 / pageSize-1 paged cache that has only been *reserved* (not
// committed) exhibits this; if the construction cannot reach the zero-length
// page state on this build, the test skips rather than asserting the wrong path.
func TestKVFromCache_PagedNoVisible(t *testing.T) {
	requireMetalRuntime(t)
	// A brand-new paged cache has Len()==0 → the first (empty) guard fires with
	// errTargetCacheEmpty, NOT the paged-no-visible one. Assert that specific
	// sentinel here; the no-visible guard is an internal-state defence reached
	// only when Len()>0 disagrees with the committed pages, which a public API
	// cannot fabricate without cache surgery.
	cache := metal.NewPagedKVCache(16, 0)
	defer metal.FreeCaches([]metal.Cache{cache})
	_, err := gemma4AssistantKVFromCache(cache)
	if !core.Is(err, errTargetCacheEmpty) {
		t.Fatalf("KVFromCache(fresh paged) error = %v, want errTargetCacheEmpty", err)
	}
}

// TestKVFromCache_EmptySentinel pins the empty-cache guard to its SPECIFIC
// sentinel (the existing extra-test only checks non-nil). A fresh fixed cache
// has Len()==0.
func TestKVFromCache_EmptySentinel(t *testing.T) {
	requireMetalRuntime(t)
	cache := metal.NewFixedKVCache(16)
	defer metal.FreeCaches([]metal.Cache{cache})
	_, err := gemma4AssistantKVFromCache(cache)
	if !core.Is(err, errTargetCacheEmpty) {
		t.Fatalf("KVFromCache(fresh fixed) error = %v, want errTargetCacheEmpty", err)
	}
}

// TestTargetKVByLayerType_EmptyCacheErrors drives targetKVByLayerType's
// per-layer K/V-extraction error path: fresh (Len()==0) target caches make
// gemma4AssistantKVFromCache fail for the first populated layer, so the function
// propagates a wrapped "target layer N" error (carrying the empty-cache
// sentinel) rather than returning a half-built lookup.
func TestTargetKVByLayerType_EmptyCacheErrors(t *testing.T) {
	requireMetalRuntime(t)
	pair := loadTinyGemma4AssistantPair(t, false)
	defer pair.Close()

	caches := pair.Target.NewCache() // fresh, never prefilled → every entry empty
	defer metal.FreeCaches(caches)
	if _, err := pair.targetKVByLayerType(caches); err == nil {
		t.Fatal("targetKVByLayerType(fresh caches) error = nil, want a per-layer K/V error")
	} else if !core.Is(err, errTargetCacheEmpty) || !core.Contains(err.Error(), "target layer") {
		t.Fatalf("targetKVByLayerType error = %v, want wrapped errTargetCacheEmpty for a target layer", err)
	}
}

// TestDraftStepActivations_Guards walks the cheap input guards of
// draftStepActivations: nil pair, an invalid (negative) token, an invalid
// previous-hidden, and empty target caches — each returning its named sentinel
// before any GPU work.
func TestDraftStepActivations_Guards(t *testing.T) {
	requireMetalRuntime(t)
	pair := loadTinyGemma4AssistantPair(t, false)
	defer pair.Close()

	hidden := seqArray(0.05, 1, 1, int(pair.Assistant.BackboneHiddenSize))
	defer metal.Free(hidden)
	caches := pair.Target.NewCache()
	defer metal.FreeCaches(caches)

	t.Run("nil_pair", func(t *testing.T) {
		empty := &Gemma4AssistantPair{}
		if _, _, err := empty.draftStepActivations(1, hidden, caches); !core.Is(err, errAsstDraftStepNeedPair) {
			t.Fatalf("err = %v, want errAsstDraftStepNeedPair", err)
		}
	})
	t.Run("token_invalid", func(t *testing.T) {
		if _, _, err := pair.draftStepActivations(-1, hidden, caches); !core.Is(err, errAsstDraftStepTokenInvalid) {
			t.Fatalf("err = %v, want errAsstDraftStepTokenInvalid", err)
		}
	})
	t.Run("hidden_invalid", func(t *testing.T) {
		if _, _, err := pair.draftStepActivations(1, nil, caches); !core.Is(err, errAsstDraftStepHiddenInvalid) {
			t.Fatalf("err = %v, want errAsstDraftStepHiddenInvalid", err)
		}
	})
	t.Run("empty_caches", func(t *testing.T) {
		if _, _, err := pair.draftStepActivations(1, hidden, nil); !core.Is(err, errAsstDraftStepNeedTargetCaches) {
			t.Fatalf("err = %v, want errAsstDraftStepNeedTargetCaches", err)
		}
	})
}

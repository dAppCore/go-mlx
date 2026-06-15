// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"testing"
)

func TestDetachEvalState_DetachesCaches_Good(t *testing.T) {
	first := &fakeDetachCache{}
	second := &fakeDetachCache{}

	detachEvalState(nil, []Cache{first, nil, second})

	if first.detachCalls != 1 {
		t.Fatalf("first cache detach calls = %d, want 1", first.detachCalls)
	}
	if second.detachCalls != 1 {
		t.Fatalf("second cache detach calls = %d, want 1", second.detachCalls)
	}
}

func TestModel_NewCaches_ShrinksOversizedRotatingCache_Good(t *testing.T) {
	model := &Model{
		model: &fakeRotatingModel{
			caches: []Cache{
				NewRotatingKVCache(4096),
				NewRotatingKVCache(256),
			},
		},
		contextLen: 1024,
	}

	caches := model.newCaches()
	if len(caches) != 2 {
		t.Fatalf("len(caches) = %d, want 2", len(caches))
	}

	first, ok := caches[0].(*RotatingKVCache)
	if !ok {
		t.Fatalf("cache[0] = %T, want *RotatingKVCache", caches[0])
	}
	if first.maxSize != 1024 {
		t.Fatalf("cache[0].maxSize = %d, want 1024", first.maxSize)
	}

	second, ok := caches[1].(*RotatingKVCache)
	if !ok {
		t.Fatalf("cache[1] = %T, want *RotatingKVCache", caches[1])
	}
	if second.maxSize != 256 {
		t.Fatalf("cache[1].maxSize = %d, want 256", second.maxSize)
	}
}

func TestModel_NewCaches_PagedPreservesRotatingCacheBound_Good(t *testing.T) {
	model := &Model{
		model: &fakeRotatingModel{
			caches: []Cache{
				NewKVCache(),
				NewRotatingKVCache(1024),
			},
		},
		contextLen: 4096,
		cacheMode:  string(KVCacheModePaged),
	}

	caches := model.newCaches()
	full, ok := caches[0].(*PagedKVCache)
	if !ok {
		t.Fatalf("cache[0] = %T, want *PagedKVCache", caches[0])
	}
	if full.maxSize != 4096 {
		t.Fatalf("cache[0].maxSize = %d, want 4096", full.maxSize)
	}

	sliding, ok := caches[1].(*PagedKVCache)
	if !ok {
		t.Fatalf("cache[1] = %T, want *PagedKVCache", caches[1])
	}
	if sliding.maxSize != 1024 {
		t.Fatalf("cache[1].maxSize = %d, want inherited sliding bound 1024", sliding.maxSize)
	}
}

func TestModel_NewCaches_PagedPageSizeConfigValue_Good(t *testing.T) {
	model := &Model{
		model: &fakeRotatingModel{
			caches: []Cache{
				NewKVCache(),
				NewRotatingKVCache(512),
			},
		},
		contextLen:      131072,
		cacheMode:       string(KVCacheModePaged),
		pagedKVPageSize: 1024,
	}

	caches := model.newCaches()
	full, ok := caches[0].(*PagedKVCache)
	if !ok {
		t.Fatalf("cache[0] = %T, want *PagedKVCache", caches[0])
	}
	if full.pageSize != 1024 {
		t.Fatalf("cache[0].pageSize = %d, want config page size 1024", full.pageSize)
	}
	sliding, ok := caches[1].(*PagedKVCache)
	if !ok {
		t.Fatalf("cache[1] = %T, want *PagedKVCache", caches[1])
	}
	if sliding.maxSize != 512 || sliding.pageSize != 512 {
		t.Fatalf("sliding cache max/page = %d/%d, want 512/512 capped env size", sliding.maxSize, sliding.pageSize)
	}
}

func TestModel_NewCaches_PagedStorageDTypeConfigValue_Good(t *testing.T) {
	model := &Model{
		model: &fakeRotatingModel{
			caches: []Cache{
				NewKVCache(),
				NewRotatingKVCache(512),
			},
		},
		contextLen:          131072,
		cacheMode:           string(KVCacheModePaged),
		kvCacheStorageDType: "bf16",
	}

	caches := model.newCaches()
	full, ok := caches[0].(*PagedKVCache)
	if !ok {
		t.Fatalf("cache[0] = %T, want *PagedKVCache", caches[0])
	}
	if !full.hasStorageDType || full.storageDType != DTypeBFloat16 {
		t.Fatalf("full storage dtype = %v/%v, want bf16 enabled", full.hasStorageDType, full.storageDType)
	}
	sliding, ok := caches[1].(*PagedKVCache)
	if !ok {
		t.Fatalf("cache[1] = %T, want *PagedKVCache", caches[1])
	}
	if !sliding.hasStorageDType || sliding.storageDType != DTypeBFloat16 {
		t.Fatalf("sliding storage dtype = %v/%v, want bf16 enabled", sliding.hasStorageDType, sliding.storageDType)
	}
}

func TestModel_NewCaches_FixedPagedStorageDTypeConfigValue_Good(t *testing.T) {
	t.Cleanup(SetRuntimeGate(GateFixedSlidingCache, true))
	t.Cleanup(SetRuntimeGate(GateFixedSlidingCacheBound, true))
	model := &Model{
		model: &fakeRotatingModel{
			usesFixedCache: true,
			caches: []Cache{
				NewKVCache(),
				NewRotatingKVCache(512),
			},
		},
		modelType:           "gemma4",
		contextLen:          32768,
		cacheMode:           string(KVCacheModePaged),
		kvCacheStorageDType: "bf16",
	}

	caches := model.newCaches()
	full, ok := caches[0].(*FixedKVCache)
	if !ok {
		t.Fatalf("cache[0] = %T, want *FixedKVCache", caches[0])
	}
	if !full.hasStorageDType || full.storageDType != DTypeBFloat16 {
		t.Fatalf("full fixed storage dtype = %v/%v, want bf16 enabled", full.hasStorageDType, full.storageDType)
	}
	sliding, ok := caches[1].(*FixedKVCache)
	if !ok {
		t.Fatalf("cache[1] = %T, want *FixedKVCache", caches[1])
	}
	if sliding.maxSize != 512 || !sliding.hasStorageDType || sliding.storageDType != DTypeBFloat16 {
		t.Fatalf("sliding fixed max/storage = %d/%v/%v, want 512 bf16", sliding.maxSize, sliding.hasStorageDType, sliding.storageDType)
	}
}

func TestModel_NewCaches_FixedGemma4UsesUniformContextBound_Good(t *testing.T) {
	t.Cleanup(SetRuntimeGate(GateFixedSlidingCache, true))

	model := &Model{
		model: &fakeRotatingModel{
			usesFixedCache: true,
			caches: []Cache{
				NewKVCache(),
				NewRotatingKVCache(1024),
			},
		},
		modelType:  "gemma4_text",
		contextLen: 4096,
		cacheMode:  string(KVCacheModePaged),
	}

	caches := model.newCaches()
	full, ok := caches[0].(*FixedKVCache)
	if !ok {
		t.Fatalf("cache[0] = %T, want *FixedKVCache", caches[0])
	}
	if full.maxSize != 4096 {
		t.Fatalf("cache[0].maxSize = %d, want 4096", full.maxSize)
	}

	sliding, ok := caches[1].(*FixedKVCache)
	if !ok {
		t.Fatalf("cache[1] = %T, want *FixedKVCache", caches[1])
	}
	if sliding.maxSize != 4096 {
		t.Fatalf("cache[1].maxSize = %d, want uniform context bound 4096", sliding.maxSize)
	}
}

func TestModel_NewCaches_FixedGemma4UsesConfiguredSize_Good(t *testing.T) {
	t.Cleanup(SetRuntimeGate(GateFixedSlidingCache, true))

	model := &Model{
		model:                 &fakeModel{numLayers: 1, usesFixedCache: true},
		modelType:             "gemma4_text",
		contextLen:            4096,
		cacheMode:             string(KVCacheModePaged),
		fixedSlidingCacheSize: 2048,
	}

	caches := model.newCaches()
	cache, ok := caches[0].(*FixedKVCache)
	if !ok {
		t.Fatalf("cache[0] = %T, want *FixedKVCache", caches[0])
	}
	if cache.maxSize != 2048 {
		t.Fatalf("cache.maxSize = %d, want configured fixed size 2048", cache.maxSize)
	}
}

func TestModel_NewGenerationCaches_FixedGemma4RightSizesRequest_Good(t *testing.T) {
	t.Cleanup(SetRuntimeGate(GateFixedSlidingCache, true))

	model := &Model{
		model:      &fakeModel{numLayers: 1, usesFixedCache: true},
		modelType:  "gemma4_text",
		contextLen: 4096,
		cacheMode:  string(KVCacheModePaged),
	}

	caches := model.newGenerationCaches(2204, GenerateConfig{MaxTokens: 128})
	cache, ok := caches[0].(*FixedKVCache)
	if !ok {
		t.Fatalf("cache[0] = %T, want *FixedKVCache", caches[0])
	}
	if cache.maxSize != 2336 {
		t.Fatalf("cache.maxSize = %d, want prompt+decode rounded to 2336", cache.maxSize)
	}
}

func TestModel_NewGenerationCaches_FixedGemma4UnifiedRightSizesRequest_Good(t *testing.T) {
	t.Cleanup(SetRuntimeGate(GateFixedSlidingCache, true))

	model := &Model{
		model:      &fakeModel{numLayers: 1, usesFixedCache: true},
		modelType:  "gemma4_unified",
		contextLen: 262144,
		cacheMode:  string(KVCacheModePaged),
	}

	caches := model.newGenerationCaches(4096, GenerateConfig{MaxTokens: 192})
	cache, ok := caches[0].(*FixedKVCache)
	if !ok {
		t.Fatalf("cache[0] = %T, want *FixedKVCache", caches[0])
	}
	if cache.maxSize != 4288 {
		t.Fatalf("cache.maxSize = %d, want 12B Unified prompt+decode rounded to 4288", cache.maxSize)
	}
}

func TestModel_NewGenerationCaches_FixedGemma4KeepsUniformRequestSize_Good(t *testing.T) {
	t.Cleanup(SetRuntimeGate(GateFixedSlidingCache, true))

	model := &Model{
		model: &fakeRotatingModel{
			usesFixedCache: true,
			caches: []Cache{
				NewKVCache(),
				NewRotatingKVCache(1024),
			},
		},
		modelType:  "gemma4_text",
		contextLen: 4096,
		cacheMode:  string(KVCacheModePaged),
	}

	caches := model.newGenerationCaches(2204, GenerateConfig{MaxTokens: 128})
	full, ok := caches[0].(*FixedKVCache)
	if !ok {
		t.Fatalf("cache[0] = %T, want *FixedKVCache", caches[0])
	}
	if full.maxSize != 2336 {
		t.Fatalf("cache[0].maxSize = %d, want request-sized fixed bound 2336", full.maxSize)
	}
	sliding, ok := caches[1].(*FixedKVCache)
	if !ok {
		t.Fatalf("cache[1] = %T, want *FixedKVCache", caches[1])
	}
	if sliding.maxSize != 2336 {
		t.Fatalf("cache[1].maxSize = %d, want request-sized fixed bound 2336", sliding.maxSize)
	}
}

func TestModel_NewGenerationCaches_FixedGemma4SlidingBoundGate_Good(t *testing.T) {
	t.Cleanup(SetRuntimeGate(GateFixedSlidingCache, true))
	restore := SetRuntimeGate(GateFixedSlidingCacheBound, true)
	t.Cleanup(restore)

	model := &Model{
		model: &fakeRotatingModel{
			usesFixedCache: true,
			caches: []Cache{
				NewKVCache(),
				NewRotatingKVCache(1024),
			},
		},
		modelType:  "gemma4_text",
		contextLen: 32768,
		cacheMode:  string(KVCacheModePaged),
	}

	caches := model.newGenerationCaches(28637, GenerateConfig{MaxTokens: 128})
	full, ok := caches[0].(*FixedKVCache)
	if !ok {
		t.Fatalf("cache[0] = %T, want *FixedKVCache", caches[0])
	}
	if full.maxSize != 28768 {
		t.Fatalf("cache[0].maxSize = %d, want request-sized fixed bound 28768", full.maxSize)
	}
	sliding, ok := caches[1].(*FixedKVCache)
	if !ok {
		t.Fatalf("cache[1] = %T, want *FixedKVCache", caches[1])
	}
	if sliding.maxSize != 1024 {
		t.Fatalf("cache[1].maxSize = %d, want sliding fixed bound 1024", sliding.maxSize)
	}
}

func TestModel_FixedSlidingCacheDispatchesOnCapability_Good(t *testing.T) {
	if !modelUsesFixedSlidingCache(&fakeModel{usesFixedCache: true}) {
		t.Fatal("modelUsesFixedSlidingCache = false, want true for a model declaring it")
	}
	if modelUsesFixedSlidingCache(&fakeModel{}) {
		t.Fatal("modelUsesFixedSlidingCache = true, want false for a model that does not declare it")
	}
}

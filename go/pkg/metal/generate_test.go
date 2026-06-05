// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"context"
	"iter"
	"reflect"
	"testing"

	"dappco.re/go"
)

type fakeDetachCache struct {
	detachCalls int
}

func (f *fakeDetachCache) Update(_ *Array, _ *Array, _ int) (*Array, *Array) { return nil, nil }
func (f *fakeDetachCache) Offset() int                                       { return 0 }
func (f *fakeDetachCache) Len() int                                          { return 0 }
func (f *fakeDetachCache) State() []*Array                                   { return nil }
func (f *fakeDetachCache) Reset()                                            {}
func (f *fakeDetachCache) Detach()                                           { f.detachCalls++ }

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

func TestModel_AcquireSlot_ReleasesCapacity_Good(t *testing.T) {
	model := &Model{parallelSlots: make(chan struct{}, 1)}

	release, err := model.acquireSlot(context.Background())
	if err != nil {
		t.Fatalf("acquireSlot: %v", err)
	}
	if len(model.parallelSlots) != 1 {
		t.Fatalf("parallelSlots occupancy = %d, want 1", len(model.parallelSlots))
	}

	release()
	if len(model.parallelSlots) != 0 {
		t.Fatalf("parallelSlots occupancy after release = %d, want 0", len(model.parallelSlots))
	}
}

func TestModel_AcquireSlot_ContextCancelled_Bad(t *testing.T) {
	model := &Model{parallelSlots: make(chan struct{}, 1)}

	release, err := model.acquireSlot(context.Background())
	if err != nil {
		t.Fatalf("acquireSlot first slot: %v", err)
	}
	defer release()

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err = model.acquireSlot(ctx)
	if err == nil {
		t.Fatal("expected context cancellation while waiting for slot")
	}
}

func TestModel_AcquireSlot_ContextCancelledBeforeOpenSlot_Bad(t *testing.T) {
	model := &Model{parallelSlots: make(chan struct{}, 1)}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	for range 100 {
		release, err := model.acquireSlot(ctx)
		if err == nil {
			release()
			t.Fatal("expected cancelled context to win before taking an open slot")
		}
	}
}

func TestModel_AcquireSlot_DefaultIsUnlimited_Ugly(t *testing.T) {
	model := &Model{}

	release, err := model.acquireSlot(context.Background())
	if err != nil {
		t.Fatalf("acquireSlot with nil limiter: %v", err)
	}
	release()
}

func TestPromptCache_LongestTokenPrefix_Good(t *testing.T) {
	got := longestTokenPrefix([]int32{1, 2, 3, 9}, []int32{1, 2, 3, 4})
	if got != 3 {
		t.Fatalf("longestTokenPrefix = %d, want 3", got)
	}
}

func TestModel_PromptCacheMatch_UsesLongStablePrefix_Good(t *testing.T) {
	model := &Model{
		promptCacheEnabled:   true,
		promptCacheMinTokens: 3,
		promptCache: &PromptCacheEntry{
			tokens:          []int32{1, 2, 3, 4},
			cacheableTokens: 4,
		},
	}

	entry, prefixLen := model.promptCacheMatch([]int32{1, 2, 3, 9})
	if entry == nil {
		t.Fatal("expected prompt cache match")
	}
	if prefixLen != 3 {
		t.Fatalf("prefixLen = %d, want 3", prefixLen)
	}
}

func TestModel_PromptCacheMatch_RejectsShortPrefix_Bad(t *testing.T) {
	model := &Model{
		promptCacheEnabled:   true,
		promptCacheMinTokens: 3,
		promptCache: &PromptCacheEntry{
			tokens:          []int32{1, 2, 3, 4},
			cacheableTokens: 4,
		},
	}

	entry, prefixLen := model.promptCacheMatch([]int32{1, 2, 9, 9})
	if entry != nil || prefixLen != 0 {
		t.Fatalf("promptCacheMatch = (%v, %d), want no match", entry, prefixLen)
	}
}

func TestModel_PromptCacheMatch_RejectsShorterPromptWithoutExactLogits_Ugly(t *testing.T) {
	model := &Model{
		promptCacheEnabled:   true,
		promptCacheMinTokens: 2,
		promptCache: &PromptCacheEntry{
			tokens:          []int32{1, 2, 3, 4},
			cacheableTokens: 4,
		},
	}

	entry, prefixLen := model.promptCacheMatch([]int32{1, 2, 3})
	if entry != nil || prefixLen != 0 {
		t.Fatalf("promptCacheMatch = (%v, %d), want no match", entry, prefixLen)
	}
}

func TestModel_PromptCacheMatch_RejectsAdapterMismatch_Ugly(t *testing.T) {
	model := &Model{
		promptCacheEnabled:   true,
		promptCacheMinTokens: 2,
		adapterInfo:          AdapterInfo{Hash: "live-adapter"},
		promptCache: &PromptCacheEntry{
			tokens:          []int32{1, 2, 3},
			cacheableTokens: 3,
			adapterHash:     "old-adapter",
		},
	}

	entry, prefixLen := model.promptCacheMatch([]int32{1, 2, 3, 4})
	if entry != nil || prefixLen != 0 {
		t.Fatalf("promptCacheMatch = (%v, %d), want adapter mismatch miss", entry, prefixLen)
	}
}

func TestPromptCache_RestoresShorterKVPrefix_Good(t *testing.T) {
	cache := NewKVCache()
	k := FromValues([]float32{1, 2, 3, 4}, 1, 1, 4, 1)
	v := FromValues([]float32{5, 6, 7, 8}, 1, 1, 4, 1)
	fullK, fullV := cache.Update(k, v, 4)
	if err := Eval(fullK, fullV); err != nil {
		t.Fatalf("Eval cache update: %v", err)
	}
	Free(k, v, fullK, fullV)
	defer FreeCaches([]Cache{cache})

	logits := FromValues([]float32{42}, 1)
	defer Free(logits)
	entry, err := newPromptCacheEntry([]int32{1, 2, 3, 4}, []Cache{cache}, logits)
	if err != nil {
		t.Fatalf("newPromptCacheEntry: %v", err)
	}
	if entry == nil {
		t.Fatal("expected prompt cache entry")
	}
	defer entry.free()

	restored, err := restorePromptCaches(entry.caches, 3)
	if err != nil {
		t.Fatalf("restorePromptCaches: %v", err)
	}
	defer FreeCaches(restored)
	if len(restored) != 1 {
		t.Fatalf("restored len = %d, want 1", len(restored))
	}
	if restored[0].Offset() != 3 || restored[0].Len() != 3 {
		t.Fatalf("restored cache offset/len = %d/%d, want 3/3", restored[0].Offset(), restored[0].Len())
	}
	state := restored[0].State()
	if state == nil || len(state) < 2 {
		t.Fatal("restored cache missing state")
	}
	if got := state[0].Shape()[2]; got != 3 {
		t.Fatalf("restored key length = %d, want 3", got)
	}
}

func TestPromptCache_MatchesExactNoLogitsByReplayingFinalToken_Good(t *testing.T) {
	model := &Model{
		promptCacheEnabled:   true,
		promptCacheMinTokens: 2,
		promptCache: &PromptCacheEntry{
			tokens:          []int32{1, 2, 3},
			cacheableTokens: 3,
		},
	}

	entry, prefixLen := model.promptCacheMatch([]int32{1, 2, 3})

	if entry == nil || prefixLen != 2 {
		t.Fatalf("promptCacheMatch exact no-logits = (%v, %d), want entry with prefix 2", entry, prefixLen)
	}
}

func TestPromptCache_RestoreFromKVSnapshotWithoutLogits_Good(t *testing.T) {
	model := &Model{
		model:                &fakeModel{numLayers: 1},
		modelType:            "gemma4_text",
		promptCacheEnabled:   true,
		promptCacheMinTokens: 1,
	}
	defer model.clearPromptCache()
	snapshot := &KVSnapshot{
		Version:      KVSnapshotVersion,
		Architecture: "gemma4_text",
		Tokens:       []int32{1, 2},
		TokenOffset:  2,
		SeqLen:       2,
		HeadDim:      2,
		Layers: []KVLayerSnapshot{{
			Layer:      0,
			CacheIndex: 0,
			Heads: []KVHeadSnapshot{{
				Key:   []float32{1, 2, 3, 4},
				Value: []float32{5, 6, 7, 8},
			}},
		}},
	}

	if err := model.RestorePromptCacheFromKV(context.Background(), snapshot); err != nil {
		t.Fatalf("RestorePromptCacheFromKV() error = %v", err)
	}

	if model.promptCache == nil {
		t.Fatal("promptCache = nil, want installed entry")
	}
	if model.promptCache.logits != nil {
		t.Fatalf("promptCache.logits = %v, want nil prefix logits", model.promptCache.logits)
	}
	if model.promptCache.cacheableTokens != 2 || len(model.promptCache.tokens) != 2 {
		t.Fatalf("promptCache metadata = %+v, want two-token prefix", model.promptCache)
	}
	if len(model.promptCache.caches) != 1 || model.promptCache.caches[0].keys == nil || model.promptCache.caches[0].values == nil {
		t.Fatalf("promptCache caches = %+v, want restored KV tensors", model.promptCache.caches)
	}
}

func TestPromptCache_SkipsWrappedRotatingCache_Bad(t *testing.T) {
	cache := NewRotatingKVCache(2)
	k := FromValues([]float32{1, 2, 3, 4}, 1, 1, 4, 1)
	v := FromValues([]float32{5, 6, 7, 8}, 1, 1, 4, 1)
	fullK, fullV := cache.Update(k, v, 4)
	if err := Eval(fullK, fullV); err != nil {
		t.Fatalf("Eval rotating cache update: %v", err)
	}
	Free(k, v, fullK, fullV)
	defer FreeCaches([]Cache{cache})

	logits := FromValues([]float32{42}, 1)
	defer Free(logits)
	entry, err := newPromptCacheEntry([]int32{1, 2, 3, 4}, []Cache{cache}, logits)
	if err != nil {
		t.Fatalf("newPromptCacheEntry: %v", err)
	}
	if entry != nil {
		entry.free()
		t.Fatal("expected wrapped rotating cache to be skipped")
	}
}

func TestKVCacheSnapshot_ExtractsKeysAndValues_Good(t *testing.T) {
	cache := NewKVCache()
	k := FromValues([]float32{1, 2, 3, 4}, 1, 1, 2, 2)
	v := FromValues([]float32{5, 6, 7, 8}, 1, 1, 2, 2)
	fullK, fullV := cache.Update(k, v, 2)
	if err := Eval(fullK, fullV); err != nil {
		t.Fatalf("Eval cache update: %v", err)
	}
	Free(k, v, fullK, fullV)
	defer FreeCaches([]Cache{cache})

	snapshot, ok := inspectKVCache(cache, 2)

	if !ok {
		t.Fatal("inspectKVCache() ok = false, want true")
	}
	if snapshot.NumHeads != 1 || snapshot.HeadDim != 2 || len(snapshot.Heads) != 1 {
		t.Fatalf("snapshot metadata = %+v", snapshot)
	}
	if snapshot.Heads[0].Key[3] != 4 || snapshot.Heads[0].Value[0] != 5 {
		t.Fatalf("snapshot head = %+v", snapshot.Heads[0])
	}
}

func TestKVCacheSnapshot_MissingValue_Bad(t *testing.T) {
	cache := &fakeDetachCache{}

	_, ok := inspectKVCache(cache, 2)

	if ok {
		t.Fatal("inspectKVCache() ok = true, want false for missing state")
	}
}

func TestAttentionCacheIndexByLayer_DefaultModel_Good(t *testing.T) {
	got := attentionCacheIndexByLayer(&fakeModel{numLayers: 4}, 4, 4)
	want := []int{0, 1, 2, 3}
	for i, wantIdx := range want {
		if got[i] != wantIdx {
			t.Fatalf("cache index for layer %d = %d, want %d", i, got[i], wantIdx)
		}
	}
}

type fakeRotatingModel struct {
	caches []Cache
}

func (f *fakeRotatingModel) Forward(_ *Array, _ []Cache) *Array                 { return nil }
func (f *fakeRotatingModel) ForwardMasked(_ *Array, _ *Array, _ []Cache) *Array { return nil }
func (f *fakeRotatingModel) NewCache() []Cache                                  { return append([]Cache(nil), f.caches...) }
func (f *fakeRotatingModel) NumLayers() int                                     { return len(f.caches) }
func (f *fakeRotatingModel) Tokenizer() *Tokenizer                              { return nil }
func (f *fakeRotatingModel) ModelType() string                                  { return "fake" }
func (f *fakeRotatingModel) ApplyLoRA(_ LoRAConfig) *LoRAAdapter                { return nil }

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
	t.Cleanup(SetRuntimeGate("GO_MLX_ENABLE_FIXED_GEMMA4_CACHE", "1"))
	t.Cleanup(SetRuntimeGate("GO_MLX_ENABLE_FIXED_GEMMA4_SLIDING_CACHE_BOUND", "1"))
	model := &Model{
		model: &fakeRotatingModel{
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

func TestPagedKVCache_RequestedPageSizeCapsToMax_Good(t *testing.T) {
	cache := NewPagedKVCache(512, 8192)

	if cache.pageSize != 512 {
		t.Fatalf("cache.pageSize = %d, want capped max size 512", cache.pageSize)
	}
}

func TestModel_NewCaches_FixedGemma4UsesUniformContextBound_Good(t *testing.T) {
	old := enableFixedGemma4Cache
	enableFixedGemma4Cache = true
	t.Cleanup(func() { enableFixedGemma4Cache = old })

	model := &Model{
		model: &fakeRotatingModel{
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
	old := enableFixedGemma4Cache
	enableFixedGemma4Cache = true
	t.Cleanup(func() { enableFixedGemma4Cache = old })

	model := &Model{
		model:                &fakeModel{numLayers: 1},
		modelType:            "gemma4_text",
		contextLen:           4096,
		cacheMode:            string(KVCacheModePaged),
		fixedGemma4CacheSize: 2048,
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
	old := enableFixedGemma4Cache
	enableFixedGemma4Cache = true
	t.Cleanup(func() { enableFixedGemma4Cache = old })

	model := &Model{
		model:      &fakeModel{numLayers: 1},
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
	old := enableFixedGemma4Cache
	enableFixedGemma4Cache = true
	t.Cleanup(func() { enableFixedGemma4Cache = old })

	model := &Model{
		model:      &fakeModel{numLayers: 1},
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
	old := enableFixedGemma4Cache
	enableFixedGemma4Cache = true
	t.Cleanup(func() { enableFixedGemma4Cache = old })

	model := &Model{
		model: &fakeRotatingModel{
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
	old := enableFixedGemma4Cache
	enableFixedGemma4Cache = true
	t.Cleanup(func() { enableFixedGemma4Cache = old })
	restore := SetRuntimeGate("GO_MLX_ENABLE_FIXED_GEMMA4_SLIDING_CACHE_BOUND", "1")
	t.Cleanup(restore)

	model := &Model{
		model: &fakeRotatingModel{
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

type chunkedPrefillModel struct {
	seqLens []int
}

func (m *chunkedPrefillModel) Forward(tokens *Array, _ []Cache) *Array {
	seqLen := tokens.Dim(1)
	m.seqLens = append(m.seqLens, seqLen)
	return Zeros([]int32{1, int32(seqLen), 2}, DTypeFloat32)
}

func (m *chunkedPrefillModel) ForwardMasked(tokens *Array, _ *Array, caches []Cache) *Array {
	return m.Forward(tokens, caches)
}
func (m *chunkedPrefillModel) NewCache() []Cache                   { return nil }
func (m *chunkedPrefillModel) NumLayers() int                      { return 0 }
func (m *chunkedPrefillModel) Tokenizer() *Tokenizer               { return nil }
func (m *chunkedPrefillModel) ModelType() string                   { return "chunked-prefill-test" }
func (m *chunkedPrefillModel) ApplyLoRA(_ LoRAConfig) *LoRAAdapter { return nil }

type lastLogitsPrefillModel struct {
	fullCalls int
	lastLens  []int
	invalid   bool
}

func (m *lastLogitsPrefillModel) Forward(tokens *Array, _ []Cache) *Array {
	m.fullCalls++
	seqLen := tokens.Dim(1)
	return Zeros([]int32{1, int32(seqLen), 64}, DTypeFloat32)
}

func (m *lastLogitsPrefillModel) ForwardMasked(tokens *Array, _ *Array, caches []Cache) *Array {
	return m.Forward(tokens, caches)
}

func (m *lastLogitsPrefillModel) ForwardLastTokenLogits(tokens *Array, _ *Array, _ []Cache) *Array {
	seqLen := tokens.Dim(1)
	m.lastLens = append(m.lastLens, seqLen)
	if m.invalid {
		return &Array{}
	}
	return Zeros([]int32{1, 1, 2}, DTypeFloat32)
}

func (m *lastLogitsPrefillModel) NewCache() []Cache                   { return nil }
func (m *lastLogitsPrefillModel) NumLayers() int                      { return 0 }
func (m *lastLogitsPrefillModel) Tokenizer() *Tokenizer               { return nil }
func (m *lastLogitsPrefillModel) ModelType() string                   { return "last-logits-prefill-test" }
func (m *lastLogitsPrefillModel) ApplyLoRA(_ LoRAConfig) *LoRAAdapter { return nil }

type cacheOnlyChunkPrefillModel struct {
	fullLens []int
	lastLens []int
}

func (m *cacheOnlyChunkPrefillModel) Forward(tokens *Array, caches []Cache) *Array {
	seqLen := int(tokens.Dim(1))
	m.fullLens = append(m.fullLens, seqLen)
	m.updateCache(seqLen, caches)
	return Zeros([]int32{1, int32(seqLen), 64}, DTypeFloat32)
}

func (m *cacheOnlyChunkPrefillModel) ForwardMasked(tokens *Array, _ *Array, caches []Cache) *Array {
	return m.Forward(tokens, caches)
}

func (m *cacheOnlyChunkPrefillModel) ForwardLastTokenLogits(tokens *Array, _ *Array, caches []Cache) *Array {
	seqLen := int(tokens.Dim(1))
	m.lastLens = append(m.lastLens, seqLen)
	m.updateCache(seqLen, caches)
	return Zeros([]int32{1, 1, 2}, DTypeFloat32)
}

func (m *cacheOnlyChunkPrefillModel) updateCache(seqLen int, caches []Cache) {
	if len(caches) == 0 || caches[0] == nil {
		return
	}
	k := Zeros([]int32{1, 1, int32(seqLen), 1}, DTypeFloat32)
	v := Zeros([]int32{1, 1, int32(seqLen), 1}, DTypeFloat32)
	fullK, fullV := caches[0].Update(k, v, seqLen)
	Free(fullK, fullV)
}

func (m *cacheOnlyChunkPrefillModel) NewCache() []Cache                   { return []Cache{NewKVCache()} }
func (m *cacheOnlyChunkPrefillModel) NumLayers() int                      { return 1 }
func (m *cacheOnlyChunkPrefillModel) Tokenizer() *Tokenizer               { return nil }
func (m *cacheOnlyChunkPrefillModel) ModelType() string                   { return "cache-only-chunk-prefill-test" }
func (m *cacheOnlyChunkPrefillModel) ApplyLoRA(_ LoRAConfig) *LoRAAdapter { return nil }

func testTokenIDs(n int) []int32 {
	tokens := make([]int32, n)
	for i := range tokens {
		tokens[i] = int32(i + 1)
	}
	return tokens
}

type boundedGenerateModel struct {
	forwardCalls int
}

func (m *boundedGenerateModel) Forward(tokens *Array, _ []Cache) *Array {
	m.forwardCalls++
	seqLen := tokens.Dim(1)
	return Zeros([]int32{1, int32(seqLen), 2}, DTypeFloat32)
}

func (m *boundedGenerateModel) ForwardMasked(tokens *Array, _ *Array, caches []Cache) *Array {
	return m.Forward(tokens, caches)
}
func (m *boundedGenerateModel) NewCache() []Cache                   { return nil }
func (m *boundedGenerateModel) NumLayers() int                      { return 0 }
func (m *boundedGenerateModel) Tokenizer() *Tokenizer               { return nil }
func (m *boundedGenerateModel) ModelType() string                   { return "bounded-generate-test" }
func (m *boundedGenerateModel) ApplyLoRA(_ LoRAConfig) *LoRAAdapter { return nil }

type directGreedyGenerateModel struct {
	forwardCalls          int
	greedyCalls           int
	suppressedGreedyCalls int
}

func (m *directGreedyGenerateModel) Forward(tokens *Array, _ []Cache) *Array {
	m.forwardCalls++
	seqLen := tokens.Dim(1)
	data := make([]float32, int(seqLen)*2)
	for i := range seqLen {
		data[int(i)*2+1] = 1
	}
	return FromValues(data, 1, int(seqLen), 2)
}

func (m *directGreedyGenerateModel) ForwardMasked(tokens *Array, _ *Array, caches []Cache) *Array {
	return m.Forward(tokens, caches)
}

func (m *directGreedyGenerateModel) ForwardGreedyToken(_ *Array, _ *Array, _ []Cache) *Array {
	m.greedyCalls++
	return FromValues([]int32{0}, 1)
}

func (m *directGreedyGenerateModel) ForwardGreedyTokenWithSuppression(_ *Array, _ *Array, _ []Cache, _ []int32) *Array {
	m.suppressedGreedyCalls++
	return FromValues([]int32{1}, 1)
}

func (m *directGreedyGenerateModel) NewCache() []Cache                   { return nil }
func (m *directGreedyGenerateModel) NumLayers() int                      { return 0 }
func (m *directGreedyGenerateModel) Tokenizer() *Tokenizer               { return nil }
func (m *directGreedyGenerateModel) ModelType() string                   { return "direct-Greedy-generate-test" }
func (m *directGreedyGenerateModel) ApplyLoRA(_ LoRAConfig) *LoRAAdapter { return nil }

type borrowedSuppressedGreedyGenerateModel struct {
	directGreedyGenerateModel
	borrowedSuppressedGreedyCalls int
	borrowedSuppress              *Array
	borrowedSuppressReused        bool
}

func (m *borrowedSuppressedGreedyGenerateModel) forwardGreedyTokenWithSuppressionArray(_ *Array, _ *Array, _ []Cache, _ []int32, suppress *Array) *Array {
	m.borrowedSuppressedGreedyCalls++
	if suppress != nil && suppress.Valid() {
		if m.borrowedSuppress == nil {
			m.borrowedSuppress = suppress
			m.borrowedSuppressReused = true
		} else if m.borrowedSuppress != suppress {
			m.borrowedSuppressReused = false
		}
	}
	return FromValues([]int32{1}, 1)
}

func TestModel_PrefillTokenBlock_ChunksByPlanner_Good(t *testing.T) {
	requireMetalRuntime(t)

	inner := &chunkedPrefillModel{}
	model := &Model{model: inner, prefillChunkSize: 2}
	logits, err := model.prefillTokenBlock(t.Context(), []int32{1, 2, 3, 4, 5}, nil)
	if err != nil {
		t.Fatalf("prefillTokenBlock() error = %v", err)
	}
	defer Free(logits)

	want := []int{2, 2, 1}
	if len(inner.seqLens) != len(want) {
		t.Fatalf("seqLens = %v, want %v", inner.seqLens, want)
	}
	for i := range want {
		if inner.seqLens[i] != want[i] {
			t.Fatalf("seqLens = %v, want %v", inner.seqLens, want)
		}
	}
	if got := logits.Shape(); len(got) != 2 || got[0] != 1 || got[1] != 2 {
		t.Fatalf("last logits shape = %v, want [1 2]", got)
	}
}

func TestModel_PrefillTokenBlock_UsesLastTokenLogitsModel_Good(t *testing.T) {
	requireMetalRuntime(t)

	inner := &lastLogitsPrefillModel{}
	model := &Model{model: inner}
	logits, err := model.prefillTokenBlock(t.Context(), testTokenIDs(defaultLastTokenPrefillMinTokens), nil)
	if err != nil {
		t.Fatalf("prefillTokenBlock() error = %v", err)
	}
	defer Free(logits)

	if inner.fullCalls != 0 {
		t.Fatalf("full forward calls = %d, want 0", inner.fullCalls)
	}
	if got, want := inner.lastLens, []int{defaultLastTokenPrefillMinTokens}; !reflect.DeepEqual(got, want) {
		t.Fatalf("lastLens = %v, want %v", got, want)
	}
	if got := logits.Shape(); len(got) != 2 || got[0] != 1 || got[1] != 2 {
		t.Fatalf("logits shape = %v, want [1 2]", got)
	}
}

func TestModel_PrefillTokenBlock_EvaluatesIntermediateChunksCacheOnly_Good(t *testing.T) {
	requireMetalRuntime(t)
	restoreCacheOnly := SetRuntimeGate("GO_MLX_ENABLE_CACHE_ONLY_CHUNK_PREFILL", "1")
	t.Cleanup(restoreCacheOnly)

	inner := &cacheOnlyChunkPrefillModel{}
	caches := inner.NewCache()
	model := &Model{model: inner, prefillChunkSize: 2}
	logits, err := model.prefillTokenBlock(t.Context(), []int32{1, 2, 3, 4, 5}, caches)
	if err != nil {
		t.Fatalf("prefillTokenBlock() error = %v", err)
	}
	defer Free(logits)
	defer FreeCaches(caches)

	if got, want := inner.fullLens, []int{2, 2, 1}; !reflect.DeepEqual(got, want) {
		t.Fatalf("full forward chunk lengths = %v, want %v", got, want)
	}
	if got, want := inner.lastLens, []int(nil); !reflect.DeepEqual(got, want) {
		t.Fatalf("last-logits chunk lengths = %v, want %v", got, want)
	}
	if caches[0].Offset() != 5 {
		t.Fatalf("cache offset = %d, want 5", caches[0].Offset())
	}
	if got := logits.Shape(); len(got) != 2 || got[0] != 1 || got[1] != 2 {
		t.Fatalf("logits shape = %v, want [1 2]", got)
	}
}

func TestModel_PrefillTokenBlock_UsesFullForwardForShortCachedChunks_Good(t *testing.T) {
	requireMetalRuntime(t)

	inner := &cacheOnlyChunkPrefillModel{}
	caches := inner.NewCache()
	model := &Model{model: inner, prefillChunkSize: 2}
	logits, err := model.prefillTokenBlock(t.Context(), []int32{1, 2, 3, 4, 5}, caches)
	if err != nil {
		t.Fatalf("prefillTokenBlock() error = %v", err)
	}
	defer Free(logits)
	defer FreeCaches(caches)

	if got, want := inner.fullLens, []int{2, 2, 1}; !reflect.DeepEqual(got, want) {
		t.Fatalf("full forward chunk lengths = %v, want %v", got, want)
	}
	if got, want := inner.lastLens, []int(nil); !reflect.DeepEqual(got, want) {
		t.Fatalf("last-logits chunk lengths = %v, want %v", got, want)
	}
	if caches[0].Offset() != 5 {
		t.Fatalf("cache offset = %d, want 5", caches[0].Offset())
	}
}

// TestModel_EffectivePrefillChunkSizeCapsFixedSlidingCache_Good pins the
// metal-side cap logic: effectivePrefillChunkSize takes the min of the model's
// prefill chunk size and the FixedSlidingPrefillLimiter limit. It uses
// fakeCapModel (limit fed by prefillLimit) rather than a concrete *Gemma4Model
// so it stays in package metal. The Gemma 4 limit computation itself
// (sliding-window/fixed-cache min) is pinned by gemma4's methods_test.go.
func TestModel_EffectivePrefillChunkSizeCapsFixedSlidingCache_Good(t *testing.T) {
	model := &Model{
		model:            &fakeCapModel{prefillLimit: 512},
		prefillChunkSize: 4096,
	}
	// gemma4FixedSlidingPrefillChunkLimit short-circuits on an empty cache slice,
	// so a non-empty slice is needed to reach the limiter dispatch.
	caches := []Cache{NewFixedKVCache(512), NewKVCache()}
	if got := model.effectivePrefillChunkSize(caches); got != 512 {
		t.Fatalf("effectivePrefillChunkSize = %d, want capped to limit 512", got)
	}
	model.prefillChunkSize = 0
	if got := model.effectivePrefillChunkSize(caches); got != 512 {
		t.Fatalf("effectivePrefillChunkSize(default) = %d, want limit 512", got)
	}
	model.prefillChunkSize = 256
	if got := model.effectivePrefillChunkSize(caches); got != 256 {
		t.Fatalf("effectivePrefillChunkSize(small explicit) = %d, want 256 (below limit)", got)
	}
}

func TestModel_PrefillTokenBlock_AutoUsesLastTokenForLongPrompt_Good(t *testing.T) {
	requireMetalRuntime(t)

	inner := &lastLogitsPrefillModel{}
	model := &Model{model: inner}
	logits, err := model.prefillTokenBlock(t.Context(), testTokenIDs(defaultLastTokenPrefillMinTokens), nil)
	if err != nil {
		t.Fatalf("prefillTokenBlock() error = %v", err)
	}
	defer Free(logits)

	if inner.fullCalls != 0 {
		t.Fatalf("full forward calls = %d, want 0", inner.fullCalls)
	}
	if len(inner.lastLens) != 1 || inner.lastLens[0] != defaultLastTokenPrefillMinTokens {
		t.Fatalf("lastLens = %v, want [%d]", inner.lastLens, defaultLastTokenPrefillMinTokens)
	}
	if got := logits.Shape(); len(got) != 2 || got[0] != 1 || got[1] != 2 {
		t.Fatalf("logits shape = %v, want [1 2]", got)
	}
}

func TestModel_PrefillTokenBlock_AutoKeepsShortPromptOnFullPath_Bad(t *testing.T) {
	requireMetalRuntime(t)

	inner := &lastLogitsPrefillModel{}
	model := &Model{model: inner}
	logits, err := model.prefillTokenBlock(t.Context(), []int32{1, 2, 3}, nil)
	if err != nil {
		t.Fatalf("prefillTokenBlock() error = %v", err)
	}
	defer Free(logits)

	if inner.fullCalls != 1 {
		t.Fatalf("full forward calls = %d, want 1", inner.fullCalls)
	}
	if len(inner.lastLens) != 0 {
		t.Fatalf("lastLens = %v, want none", inner.lastLens)
	}
	if got := logits.Shape(); len(got) != 2 || got[0] != 1 || got[1] != 64 {
		t.Fatalf("logits shape = %v, want [1 64]", got)
	}
}

func TestModel_PrefillTokenBlock_FallsBackWhenLastTokenLogitsInvalid_Good(t *testing.T) {
	requireMetalRuntime(t)

	inner := &lastLogitsPrefillModel{invalid: true}
	model := &Model{model: inner}
	logits, err := model.prefillTokenBlock(t.Context(), testTokenIDs(defaultLastTokenPrefillMinTokens), nil)
	if err != nil {
		t.Fatalf("prefillTokenBlock() error = %v", err)
	}
	defer Free(logits)

	if inner.fullCalls != 1 {
		t.Fatalf("full forward calls = %d, want 1", inner.fullCalls)
	}
	if len(inner.lastLens) != 1 {
		t.Fatalf("last logits attempts = %d, want 1", len(inner.lastLens))
	}
	if got := logits.Shape(); len(got) != 2 || got[0] != 1 || got[1] != 64 {
		t.Fatalf("fallback logits shape = %v, want [1 64]", got)
	}
}

func TestModel_Generate_DoesNotForwardAfterFinalToken_Good(t *testing.T) {
	requireMetalRuntime(t)

	inner := &boundedGenerateModel{}
	model := &Model{
		model:     inner,
		tokenizer: &Tokenizer{invVocab: map[int32]string{0: "x"}},
	}
	var got []Token
	for token := range model.generateTokens(context.Background(), []int32{1}, GenerateConfig{MaxTokens: 1}) {
		got = append(got, token)
	}
	if model.Err() != nil {
		t.Fatalf("Generate() error = %v", model.Err())
	}
	if len(got) != 1 {
		t.Fatalf("generated tokens = %d, want 1", len(got))
	}
	if inner.forwardCalls != 1 {
		t.Fatalf("Forward calls = %d, want only the prompt prefill", inner.forwardCalls)
	}
}

func TestModel_Generate_TraceTokenPhases_Good(t *testing.T) {
	requireMetalRuntime(t)

	inner := &boundedGenerateModel{}
	model := &Model{
		model:     inner,
		tokenizer: &Tokenizer{invVocab: map[int32]string{0: "x"}},
	}
	for range model.generateTokens(context.Background(), []int32{1}, GenerateConfig{MaxTokens: 2, TraceTokenPhases: true, TraceTokenText: true}) {
	}
	if model.Err() != nil {
		t.Fatalf("Generate() error = %v", model.Err())
	}
	phases := model.LastMetrics().TokenPhases
	if len(phases) != 2 {
		t.Fatalf("TokenPhases length = %d, want 2; phases=%+v", len(phases), phases)
	}
	if phases[0].Step != 0 || phases[1].Step != 1 {
		t.Fatalf("phase steps = %+v, want ordered step traces", phases)
	}
	if phases[0].TokenID != 0 || phases[0].TokenText != "x" || phases[1].TokenID != 0 || phases[1].TokenText != "x" {
		t.Fatalf("phase sampled tokens = %+v, want token id/text captured", phases)
	}
	if phases[0].ForwardDuration <= 0 {
		t.Fatalf("first phase forward duration = %s, want next-token forward timing", phases[0].ForwardDuration)
	}
	if !phases[1].FinalToken || phases[1].ForwardDuration != 0 {
		t.Fatalf("final phase = %+v, want final token with no forward timing", phases[1])
	}
	if phases[0].TotalDuration <= 0 || phases[1].TotalDuration <= 0 {
		t.Fatalf("phase totals = %+v, want positive token timings", phases)
	}
}

func TestModel_Generate_TraceTokenPhasesNoProbeSink_Good(t *testing.T) {
	requireMetalRuntime(t)

	inner := &boundedGenerateModel{}
	model := &Model{
		model:     inner,
		tokenizer: &Tokenizer{invVocab: map[int32]string{0: "x"}},
	}
	for range model.generateTokens(context.Background(), []int32{1}, GenerateConfig{MaxTokens: 2, TraceTokenPhases: true}) {
	}
	if model.Err() != nil {
		t.Fatalf("Generate() error = %v", model.Err())
	}
	for _, phase := range model.LastMetrics().TokenPhases {
		if phase.CacheProbeDuration != 0 {
			t.Fatalf("phase %d cache probe duration = %s, want zero without a probe sink", phase.Step, phase.CacheProbeDuration)
		}
		if phase.TokenText != "" {
			t.Fatalf("phase %d token text = %q, want text omitted unless TraceTokenText is enabled", phase.Step, phase.TokenText)
		}
	}
}

func TestModel_Generate_KeepsDecodeLogitsLazyBetweenTokens_Good(t *testing.T) {
	requireMetalRuntime(t)

	inner := &boundedGenerateModel{}
	model := &Model{
		model:     inner,
		tokenizer: &Tokenizer{invVocab: map[int32]string{0: "x"}},
	}
	for range model.generateTokens(context.Background(), []int32{1}, GenerateConfig{MaxTokens: 2, TraceTokenPhases: true}) {
	}
	if model.Err() != nil {
		t.Fatalf("Generate() error = %v", model.Err())
	}
	phases := model.LastMetrics().TokenPhases
	if len(phases) != 2 {
		t.Fatalf("TokenPhases length = %d, want 2; phases=%+v", len(phases), phases)
	}
	if phases[0].MaterializeDuration != 0 {
		t.Fatalf("first phase materialize duration = %s, want lazy next-token logits", phases[0].MaterializeDuration)
	}
}

func TestModel_Generate_AsyncDecodePrefetch_Good(t *testing.T) {
	requireMetalRuntime(t)
	t.Cleanup(SetRuntimeGate("GO_MLX_ENABLE_ASYNC_DECODE_PREFETCH", "1"))

	out := Zeros([]int32{1, 1, 2}, DTypeFloat32)
	defer Free(out)
	if err := asyncDecodePrefetch(0, "test", out); err != nil {
		t.Fatalf("asyncDecodePrefetch() error = %v", err)
	}
	if err := Eval(out); err != nil {
		t.Fatalf("Eval after asyncDecodePrefetch() error = %v", err)
	}

	cache := NewPagedKVCache(0, 2)
	defer cache.Reset()
	k, v := makeSingleTokenKV(1)
	defer Free(k, v)
	state := cache.UpdateBorrowedPages(k, v, 1)
	state.Free()
	timings, err := asyncDecodePrefetchWithCachesTrace("Model.Generate", 0, "test split", out, []Cache{cache})
	if err != nil {
		t.Fatalf("asyncDecodePrefetchWithCachesTrace() error = %v", err)
	}
	if timings.Logits <= 0 || timings.Cache != 0 {
		t.Fatalf("async prefetch timings = %+v, want production-shaped combined logits timing", timings)
	}
	splitTimings, err := asyncDecodePrefetchWithCachesTraceSplit("Model.Generate", 0, "test split", out, []Cache{cache})
	if err != nil {
		t.Fatalf("asyncDecodePrefetchWithCachesTraceSplit() error = %v", err)
	}
	if splitTimings.Logits <= 0 || splitTimings.Cache <= 0 {
		t.Fatalf("async split prefetch timings = %+v, want diagnostic logits and dirty-cache timing", splitTimings)
	}

	inner := &boundedGenerateModel{}
	model := &Model{
		model:     inner,
		tokenizer: &Tokenizer{invVocab: map[int32]string{0: "x"}},
	}
	for range model.generateTokens(context.Background(), []int32{1}, GenerateConfig{MaxTokens: 2, TraceTokenPhases: true}) {
	}
	if model.Err() != nil {
		t.Fatalf("Generate() error = %v", model.Err())
	}
	phases := model.LastMetrics().TokenPhases
	if len(phases) != 2 || phases[0].PrefetchDuration <= 0 {
		t.Fatalf("TokenPhases = %+v, want async next-token prefetch duration", phases)
	}
	if phases[0].PrefetchLogitsDuration <= 0 || phases[0].PrefetchCacheDuration != 0 {
		t.Fatalf("first phase prefetch split = %+v, want logits-only split for cacheless model", phases[0])
	}
}

func TestModel_Generate_AsyncDecodePrefetchRuntimeGate_Good(t *testing.T) {
	restoreOff := SetRuntimeGate("GO_MLX_ENABLE_ASYNC_DECODE_PREFETCH", "0")
	t.Cleanup(restoreOff)
	if asyncDecodePrefetchEnabled() {
		t.Fatal("asyncDecodePrefetchEnabled() = true, want runtime gate off")
	}
	restoreOn := SetRuntimeGate("GO_MLX_ENABLE_ASYNC_DECODE_PREFETCH", "1")
	t.Cleanup(restoreOn)
	if !asyncDecodePrefetchEnabled() {
		t.Fatal("asyncDecodePrefetchEnabled() = false, want runtime gate on")
	}
}

func TestModel_Generate_AsyncDecodePrefetch_Bad(t *testing.T) {
	t.Cleanup(SetRuntimeGate("GO_MLX_ENABLE_ASYNC_DECODE_PREFETCH", "1"))

	if err := asyncDecodePrefetch(0, "nil", nil); err != nil {
		t.Fatalf("asyncDecodePrefetch(nil) error = %v", err)
	}
}

func TestModel_Generate_GenerationStream_Good(t *testing.T) {
	requireMetalRuntime(t)
	t.Cleanup(SetRuntimeGate("GO_MLX_ENABLE_GENERATION_STREAM", "1"))

	model := &Model{device: DeviceGPU}
	if err := model.withGenerationStream(func() {
		out := Zeros([]int32{1}, DTypeFloat32)
		defer Free(out)
		if evalErr := Eval(out); evalErr != nil {
			t.Fatalf("Eval under generation stream: %v", evalErr)
		}
	}); err != nil {
		t.Fatalf("withGenerationStream() error = %v", err)
	}
}

func TestModel_Generate_GenerationStream_Bad(t *testing.T) {
	restore := SetRuntimeGate("GO_MLX_ENABLE_GENERATION_STREAM", "0")
	t.Cleanup(restore)

	called := false
	model := &Model{device: DeviceGPU}
	if err := model.withGenerationStream(func() { called = true }); err != nil {
		t.Fatalf("withGenerationStream() gate off error = %v", err)
	}
	if !called {
		t.Fatal("withGenerationStream() did not call function with gate off")
	}
}

func TestModel_Generate_GenerationClearCacheIntervalConfig_Good(t *testing.T) {
	if got := generationClearCacheInterval(GenerateConfig{ClearCacheInterval: 64}); got != 64 {
		t.Fatalf("generationClearCacheInterval() = %d, want 64", got)
	}
}

func TestModel_Generate_GenerationClearCacheIntervalDefault_Bad(t *testing.T) {
	if got := generationClearCacheInterval(GenerateConfig{ClearCacheInterval: 0}); got != defaultGenerationClearCacheInterval {
		t.Fatalf("generationClearCacheInterval() = %d, want default %d", got, defaultGenerationClearCacheInterval)
	}
}

func TestModel_Generate_UsesDirectGreedyToken_Good(t *testing.T) {
	requireMetalRuntime(t)
	t.Cleanup(SetRuntimeGate("GO_MLX_ENABLE_DIRECT_GREEDY_TOKEN", "1"))

	inner := &directGreedyGenerateModel{}
	model := &Model{
		model:     inner,
		tokenizer: &Tokenizer{invVocab: map[int32]string{0: "x", 1: "y"}},
	}
	var got []Token
	for token := range model.generateTokens(context.Background(), []int32{1}, GenerateConfig{MaxTokens: 2, TraceTokenPhases: true}) {
		got = append(got, token)
	}
	if model.Err() != nil {
		t.Fatalf("Generate() error = %v", model.Err())
	}
	if len(got) != 2 || got[0].ID != 1 || got[1].ID != 0 {
		t.Fatalf("tokens = %+v, want IDs [1 0]", got)
	}
	if inner.forwardCalls != 1 {
		t.Fatalf("Forward calls = %d, want only prompt prefill", inner.forwardCalls)
	}
	if inner.greedyCalls != 1 {
		t.Fatalf("ForwardGreedyToken calls = %d, want one direct decode call", inner.greedyCalls)
	}
	phases := model.LastMetrics().TokenPhases
	if len(phases) != 2 || phases[0].ForwardDuration <= 0 || phases[1].ForwardDuration != 0 {
		t.Fatalf("phases = %+v, want direct Greedy forward on first step only", phases)
	}
}

func TestModel_Generate_UsesSuppressedDirectGreedyToken_Good(t *testing.T) {
	requireMetalRuntime(t)
	t.Cleanup(SetRuntimeGate("GO_MLX_ENABLE_DIRECT_GREEDY_TOKEN", "1"))

	inner := &directGreedyGenerateModel{}
	model := &Model{
		model:     inner,
		tokenizer: &Tokenizer{invVocab: map[int32]string{0: "x", 1: "y"}},
	}
	var got []Token
	for token := range model.generateTokens(context.Background(), []int32{1}, GenerateConfig{
		MaxTokens:        2,
		SuppressTokens:   []int32{0},
		TraceTokenPhases: true,
	}) {
		got = append(got, token)
	}
	if model.Err() != nil {
		t.Fatalf("Generate() error = %v", model.Err())
	}
	if len(got) != 2 || got[0].ID != 1 || got[1].ID != 1 {
		t.Fatalf("tokens = %+v, want IDs [1 1]", got)
	}
	if inner.forwardCalls != 1 {
		t.Fatalf("Forward calls = %d, want only prompt prefill", inner.forwardCalls)
	}
	if inner.greedyCalls != 0 {
		t.Fatalf("ForwardGreedyToken calls = %d, want suppression-aware path instead", inner.greedyCalls)
	}
	if inner.suppressedGreedyCalls != 1 {
		t.Fatalf("ForwardGreedyTokenWithSuppression calls = %d, want one direct decode call", inner.suppressedGreedyCalls)
	}
}

func TestModel_Generate_UsesBorrowedSuppressionArray_Good(t *testing.T) {
	requireMetalRuntime(t)
	t.Cleanup(SetRuntimeGate("GO_MLX_ENABLE_DIRECT_GREEDY_TOKEN", "1"))

	inner := &borrowedSuppressedGreedyGenerateModel{}
	model := &Model{
		model:     inner,
		tokenizer: &Tokenizer{invVocab: map[int32]string{0: "x", 1: "y"}},
	}
	var got []Token
	for token := range model.generateTokens(context.Background(), []int32{1}, GenerateConfig{
		MaxTokens:      3,
		SuppressTokens: []int32{0},
	}) {
		got = append(got, token)
	}
	if model.Err() != nil {
		t.Fatalf("Generate() error = %v", model.Err())
	}
	if len(got) != 3 || got[0].ID != 1 || got[1].ID != 1 || got[2].ID != 1 {
		t.Fatalf("tokens = %+v, want IDs [1 1 1]", got)
	}
	if inner.borrowedSuppressedGreedyCalls != 2 {
		t.Fatalf("borrowed suppression calls = %d, want two direct decode calls", inner.borrowedSuppressedGreedyCalls)
	}
	if inner.borrowedSuppress == nil || !inner.borrowedSuppressReused {
		t.Fatalf("borrowed suppress array reused = %v ptr=%p, want one valid reused array", inner.borrowedSuppressReused, inner.borrowedSuppress)
	}
	if inner.suppressedGreedyCalls != 0 {
		t.Fatalf("ForwardGreedyTokenWithSuppression calls = %d, want borrowed array path", inner.suppressedGreedyCalls)
	}
}

func TestModel_Generate_DirectGreedyRejectsRepeatPenalty_Bad(t *testing.T) {
	requireMetalRuntime(t)
	t.Cleanup(SetRuntimeGate("GO_MLX_ENABLE_DIRECT_GREEDY_TOKEN", "1"))

	inner := &directGreedyGenerateModel{}
	model := &Model{
		model:     inner,
		tokenizer: &Tokenizer{invVocab: map[int32]string{0: "x", 1: "y"}},
	}
	for range model.generateTokens(context.Background(), []int32{1}, GenerateConfig{MaxTokens: 2, RepeatPenalty: 1.1}) {
	}
	if model.Err() != nil {
		t.Fatalf("Generate() error = %v", model.Err())
	}
	if inner.greedyCalls != 0 {
		t.Fatalf("ForwardGreedyToken calls = %d, want disabled when repeat penalty needs logits history", inner.greedyCalls)
	}
	if inner.forwardCalls != 2 {
		t.Fatalf("Forward calls = %d, want prompt plus logits decode fallback", inner.forwardCalls)
	}
}

func TestModel_FormatChat_Gemma2UsesGemmaTemplate_Good(t *testing.T) {
	model := &Model{modelType: "gemma2"}

	got := model.formatChat([]ChatMessage{
		{Role: "user", Content: "Hello"},
		{Role: "assistant", Content: "Hi"},
	})

	want := "<bos><start_of_turn>user\nHello<end_of_turn>\n" +
		"<start_of_turn>model\nHi<end_of_turn>\n" +
		"<start_of_turn>model\n"
	if got != want {
		t.Fatalf("formatChat() = %q, want %q", got, want)
	}
}

func TestModel_FormatChat_GemmaFoldsSystemIntoFirstUser_Good(t *testing.T) {
	model := &Model{modelType: "gemma3_text"}

	got := model.formatChat([]ChatMessage{
		{Role: "system", Content: " sys "},
		{Role: "user", Content: " hi "},
	})
	want := "<bos><start_of_turn>user\nsys\n\nhi<end_of_turn>\n<start_of_turn>model\n"
	if got != want {
		t.Fatalf("formatChat() = %q, want %q", got, want)
	}
}

func TestModel_FormatChatChunks_GemmaMatchesFormattedPrompt_Good(t *testing.T) {
	model := &Model{modelType: "gemma3_text"}
	messages := []ChatMessage{
		{Role: "system", Content: "abc"},
		{Role: "user", Content: "defghi"},
		{Role: "assistant", Content: "jkl"},
	}

	got := core.Join("", collectChatChunks(model.formatChatChunks(messages, 3))...)
	want := model.formatChat(messages)
	if got != want {
		t.Fatalf("joined gemma chat chunks = %q, want %q", got, want)
	}
}

func TestModel_FormatChat_Gemma4UsesModelTemplate_Good(t *testing.T) {
	model := &Model{modelType: "gemma4_text"}

	got := model.formatChat([]ChatMessage{
		{Role: "system", Content: " be brief "},
		{Role: "user", Content: "Hello"},
		{Role: "assistant", Content: "Hi"},
		{Role: "user", Content: "Again"},
	})

	want := "<bos><|turn>system\n<|think|>\nbe brief<turn|>\n" +
		"<|turn>user\nHello<turn|>\n" +
		"<|turn>model\nHi<turn|>\n" +
		"<|turn>user\nAgain<turn|>\n" +
		"<|turn>model\n"
	if got != want {
		t.Fatalf("formatChat() = %q, want %q", got, want)
	}
}

func TestModel_FormatChat_Gemma4UnifiedUsesModelTemplate_Good(t *testing.T) {
	model := &Model{modelType: "gemma4_unified"}

	got := model.formatChat([]ChatMessage{
		{Role: "system", Content: " be brief "},
		{Role: "user", Content: "Hello"},
	})

	want := "<bos><|turn>system\n<|think|>\nbe brief<turn|>\n" +
		"<|turn>user\nHello<turn|>\n" +
		"<|turn>model\n"
	if got != want {
		t.Fatalf("formatChat(gemma4_unified) = %q, want %q", got, want)
	}
}

func TestModel_FormatChat_Gemma4StripsAssistantThoughtHistory_Good(t *testing.T) {
	model := &Model{modelType: "gemma4_text"}

	got := model.formatChat([]ChatMessage{
		{Role: "user", Content: "Hello"},
		{Role: "assistant", Content: "<|channel>thought\nprivate<channel|>Visible"},
	})
	want := "<bos><|turn>system\n<|think|>\n<turn|>\n<|turn>user\nHello<turn|>\n<|turn>model\nVisible<turn|>\n<|turn>model\n"
	if got != want {
		t.Fatalf("formatChat() = %q, want %q", got, want)
	}
}

func TestFormatGemma4Chat_ThinkingOffSmall_Good(t *testing.T) {
	messages := []ChatMessage{{Role: "user", Content: "Hello"}}
	got := formatGemma4Chat(messages, false, false)
	// E2B/E4B thinking-off: plain template, no <|think|>, no thought channel.
	want := "<bos><|turn>user\nHello<turn|>\n<|turn>model\n"
	if got != want {
		t.Fatalf("thinking-off small = %q, want %q", got, want)
	}
}

func TestFormatGemma4Chat_ThinkingOffLargeStabiliser_Good(t *testing.T) {
	messages := []ChatMessage{{Role: "user", Content: "Hello"}}
	got := formatGemma4Chat(messages, false, true)
	// 26B/31B ghost an empty thought channel when thinking is off; the empty
	// <|channel>thought\n<channel|> suppressor makes them answer directly.
	want := "<bos><|turn>user\nHello<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"
	if got != want {
		t.Fatalf("thinking-off large = %q, want %q", got, want)
	}
}

func TestFormatGemma4Chat_ThinkingOn_Good(t *testing.T) {
	messages := []ChatMessage{{Role: "user", Content: "Hello"}}
	got := formatGemma4Chat(messages, true, false)
	// Thinking on: standalone <|think|>\n system turn (jinja-faithful, via chat.Format).
	want := "<bos><|turn>system\n<|think|>\n<turn|>\n<|turn>user\nHello<turn|>\n<|turn>model\n"
	if got != want {
		t.Fatalf("thinking-on = %q, want %q", got, want)
	}
}

func TestModel_FormatChat_Gemma4ThinkingOff_Good(t *testing.T) {
	model := &Model{modelType: "gemma4_text"} // bare model → not large → small OFF template
	disabled := false
	got := model.formatChat([]ChatMessage{{Role: "user", Content: "Hello"}}, GenerateConfig{EnableThinking: &disabled})
	want := "<bos><|turn>user\nHello<turn|>\n<|turn>model\n"
	if got != want {
		t.Fatalf("formatChat thinking-off = %q, want %q", got, want)
	}
}

func TestModel_FormatChatChunks_Gemma4MatchesFormattedPrompt_Good(t *testing.T) {
	model := &Model{modelType: "gemma4_text"}
	messages := []ChatMessage{
		{Role: "system", Content: " be brief "},
		{Role: "user", Content: "abcdef"},
		{Role: "assistant", Content: "Hi"},
	}

	chunks := collectChatChunks(model.formatChatChunks(messages, 2))
	got := core.Join("", chunks...)
	want := model.formatChat(messages)

	if got != want {
		t.Fatalf("joined chat chunks = %q, want %q", got, want)
	}
	if len(chunks) <= len(messages) {
		t.Fatalf("chunks = %#v, want bounded content chunks plus template chunks", chunks)
	}
}

func TestModel_FormatChatChunks_QwenMatchesFormattedPrompt_Good(t *testing.T) {
	model := &Model{modelType: "qwen3"}
	messages := []ChatMessage{
		{Role: "system", Content: "abc"},
		{Role: "user", Content: "defghi"},
	}

	got := core.Join("", collectChatChunks(model.formatChatChunks(messages, 3))...)
	want := model.formatChat(messages)

	if got != want {
		t.Fatalf("joined qwen chat chunks = %q, want %q", got, want)
	}
}

func collectChatChunks(chunks iter.Seq[string]) []string {
	out := []string{}
	for chunk := range chunks {
		out = append(out, chunk)
	}
	return out
}

func TestGenerate_Model_StagedMiniMaxReturnsDecodeError_Bad(t *testing.T) {
	model := &Model{
		model:     stagedDecodeUnavailableModel{modelType: "minimax_m2", message: "minimax_m2 staged loader has no native decode kernels yet"},
		modelType: "minimax_m2",
	}

	tokenCount := 0
	for range model.Generate(context.Background(), "hello", GenerateConfig{MaxTokens: 1}) {
		tokenCount++
	}
	if tokenCount != 0 {
		t.Fatalf("generated %d token(s), want none before MiniMax decode kernels are linked", tokenCount)
	}
	if err := model.Err(); err == nil || !core.Contains(err.Error(), "minimax_m2") || !core.Contains(err.Error(), "decode") {
		t.Fatalf("Err() = %v, want minimax_m2 decode diagnostic", err)
	}
}

type stagedDecodeUnavailableModel struct {
	modelType string
	message   string
}

func (s stagedDecodeUnavailableModel) Forward(_ *Array, _ []Cache) *Array                 { return nil }
func (s stagedDecodeUnavailableModel) ForwardMasked(_ *Array, _ *Array, _ []Cache) *Array { return nil }
func (s stagedDecodeUnavailableModel) NewCache() []Cache                                  { return nil }
func (s stagedDecodeUnavailableModel) NumLayers() int                                     { return 0 }
func (s stagedDecodeUnavailableModel) Tokenizer() *Tokenizer                              { return nil }
func (s stagedDecodeUnavailableModel) ModelType() string                                  { return s.modelType }
func (s stagedDecodeUnavailableModel) ApplyLoRA(_ LoRAConfig) *LoRAAdapter                { return nil }
func (s stagedDecodeUnavailableModel) DecodeUnavailableError(operation string) error {
	return core.NewError(operation + ": " + s.message)
}

type moeTextUnavailableModel struct {
	stagedDecodeUnavailableModel
}

func (m moeTextUnavailableModel) MoETextRuntimeAvailable() bool { return false }
func (m moeTextUnavailableModel) MoETextDecodeFamily() string   { return m.modelType }

func TestGenerate_Model_StagedQwen36ReturnsDecodeError_Bad(t *testing.T) {
	model := &Model{
		model:     stagedDecodeUnavailableModel{modelType: "qwen3_6", message: "qwen3_6 staged loader has no native hybrid linear-attention decode kernels yet"},
		modelType: "qwen3_6",
	}

	tokenCount := 0
	for range model.Generate(context.Background(), "hello", GenerateConfig{MaxTokens: 1}) {
		tokenCount++
	}
	if tokenCount != 0 {
		t.Fatalf("generated %d token(s), want none before Qwen3.6 linear-attention decode kernels are linked", tokenCount)
	}
	if err := model.Err(); err == nil || !core.Contains(err.Error(), "qwen3_6") || !core.Contains(err.Error(), "linear-attention") {
		t.Fatalf("Err() = %v, want qwen3_6 linear-attention decode diagnostic", err)
	}
}

func TestGenerate_Model_StagedQwen3MoEReturnsDecodeError_Bad(t *testing.T) {
	model := &Model{
		model: moeTextUnavailableModel{
			stagedDecodeUnavailableModel: stagedDecodeUnavailableModel{modelType: "qwen3_moe"},
		},
		modelType: "qwen3_moe",
	}

	tokenCount := 0
	for range model.Generate(context.Background(), "hello", GenerateConfig{MaxTokens: 1}) {
		tokenCount++
	}
	if tokenCount != 0 {
		t.Fatalf("generated %d token(s), want none before Qwen3 MoE sparse-expert decode kernels are linked", tokenCount)
	}
	if err := model.Err(); err == nil || !core.Contains(err.Error(), "qwen3_moe") || !core.Contains(err.Error(), "sparse-expert") {
		t.Fatalf("Err() = %v, want qwen3_moe sparse-expert decode diagnostic", err)
	}
}

func TestGenerate_Model_StagedBERTReturnsDecodeError_Bad(t *testing.T) {
	model := &Model{
		model:     stagedDecodeUnavailableModel{modelType: "bert", message: "bert staged loader has no native text decode kernels; use the encoder/rerank API once scorer kernels land"},
		modelType: "bert",
	}

	tokenCount := 0
	for range model.Generate(context.Background(), "hello", GenerateConfig{MaxTokens: 1}) {
		tokenCount++
	}
	if tokenCount != 0 {
		t.Fatalf("generated %d token(s), want none before BERT encoder kernels are linked", tokenCount)
	}
	if err := model.Err(); err == nil || !core.Contains(err.Error(), "bert") || !core.Contains(err.Error(), "encoder/rerank") {
		t.Fatalf("Err() = %v, want bert staged encoder/rerank diagnostic", err)
	}
}

func TestGenerate_LastTokenLogits_Good(t *testing.T) {
	oneDim := FromValues([]float32{1, 2, 3}, 3)
	oneRow := FromValues([]float32{1, 2, 3}, 1, 3)
	twoDim := FromValues([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	singleStep := FromValues([]float32{1, 2, 3}, 1, 1, 3)
	threeDim := FromValues([]float32{1, 2, 3, 4, 5, 6}, 1, 2, 3)
	defer Free(oneDim, oneRow, twoDim, singleStep, threeDim)

	for name, logits := range map[string]*Array{
		"one":         oneDim,
		"one-row":     oneRow,
		"two":         twoDim,
		"single-step": singleStep,
		"three":       threeDim,
	} {
		last, err := lastTokenLogits(logits)
		if err != nil {
			t.Fatalf("%s lastTokenLogits: %v", name, err)
		}
		if err := Eval(last); err != nil {
			Free(last)
			t.Fatalf("%s Eval(last): %v", name, err)
		}
		if last.NumDims() != 2 || last.Dim(0) != 1 || last.Dim(1) != 3 {
			t.Fatalf("%s last shape = %v, want [1 3]", name, last.Shape())
		}
		Free(last)
	}
}

func TestGenerate_Model_StagedMoEReturnsDecodeError_Bad(t *testing.T) {
	cases := []struct {
		name      string
		modelType string
		model     InternalModel
		want      []string
	}{
		{
			name:      "mixtral",
			modelType: "mixtral",
			model:     moeTextUnavailableModel{stagedDecodeUnavailableModel: stagedDecodeUnavailableModel{modelType: "mixtral"}},
			want:      []string{"mixtral", "sparse-expert"},
		},
		{
			name:      "deepseek",
			modelType: "deepseek",
			model:     moeTextUnavailableModel{stagedDecodeUnavailableModel: stagedDecodeUnavailableModel{modelType: "deepseek"}},
			want:      []string{"deepseek", "sparse-expert"},
		},
		{
			name:      "gpt_oss",
			modelType: "gpt_oss",
			model:     moeTextUnavailableModel{stagedDecodeUnavailableModel: stagedDecodeUnavailableModel{modelType: "gpt_oss"}},
			want:      []string{"gpt_oss", "sparse-expert"},
		},
		{
			name:      "kimi",
			modelType: "kimi",
			model:     moeTextUnavailableModel{stagedDecodeUnavailableModel: stagedDecodeUnavailableModel{modelType: "kimi"}},
			want:      []string{"kimi", "sparse-expert"},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			model := &Model{
				model:     tc.model,
				modelType: tc.modelType,
			}
			tokenCount := 0
			for range model.Generate(context.Background(), "hello", GenerateConfig{MaxTokens: 1}) {
				tokenCount++
			}
			if tokenCount != 0 {
				t.Fatalf("generated %d token(s), want none before %s decode kernels are linked", tokenCount, tc.name)
			}
			for _, want := range tc.want {
				if err := model.Err(); err == nil || !core.Contains(err.Error(), want) {
					t.Fatalf("Err() = %v, want %q in error", err, want)
				}
			}
		})
	}
}

func TestGenerate_Model_StagedQwen36MoEReturnsDecodeError_Bad(t *testing.T) {
	model := &Model{
		model:     stagedDecodeUnavailableModel{modelType: "qwen3_6_moe", message: "qwen3_6_moe staged loader has no native hybrid linear-attention and sparse-expert decode kernels yet"},
		modelType: "qwen3_6_moe",
	}

	tokenCount := 0
	for range model.Generate(context.Background(), "hello", GenerateConfig{MaxTokens: 1}) {
		tokenCount++
	}
	if tokenCount != 0 {
		t.Fatalf("generated %d token(s), want none before qwen3_6_moe decode kernels are linked", tokenCount)
	}
	if err := model.Err(); err == nil || !core.Contains(err.Error(), "qwen3_6_moe") || !core.Contains(err.Error(), "linear-attention") {
		t.Fatalf("Err() = %v, want qwen3_6_moe hybrid linear-attention decode diagnostic", err)
	}
}

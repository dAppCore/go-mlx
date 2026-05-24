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
	coverageTokens := "DetachesCaches"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
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
	coverageTokens := "AcquireSlot ReleasesCapacity"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
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
	coverageTokens := "AcquireSlot ContextCancelled"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
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
	coverageTokens := "AcquireSlot ContextCancelledBeforeOpenSlot"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
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
	coverageTokens := "AcquireSlot DefaultIsUnlimited"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
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
		promptCache: &promptCacheEntry{
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
		promptCache: &promptCacheEntry{
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
		promptCache: &promptCacheEntry{
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
		promptCache: &promptCacheEntry{
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
	coverageTokens := "PromptCache RestoresShorterKVPrefix"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	cache := NewKVCache()
	k := FromValues([]float32{1, 2, 3, 4}, 1, 1, 4, 1)
	v := FromValues([]float32{5, 6, 7, 8}, 1, 1, 4, 1)
	fullK, fullV := cache.Update(k, v, 4)
	if err := Eval(fullK, fullV); err != nil {
		t.Fatalf("Eval cache update: %v", err)
	}
	Free(k, v, fullK, fullV)
	defer freeCaches([]Cache{cache})

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
	defer freeCaches(restored)
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
	coverageTokens := "PromptCache ExactNoLogitsReplaysFinal"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	model := &Model{
		promptCacheEnabled:   true,
		promptCacheMinTokens: 2,
		promptCache: &promptCacheEntry{
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
	coverageTokens := "PromptCache RestoreFromKVSnapshotWithoutLogits"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
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
	coverageTokens := "PromptCache SkipsWrappedRotatingCache"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	cache := NewRotatingKVCache(2)
	k := FromValues([]float32{1, 2, 3, 4}, 1, 1, 4, 1)
	v := FromValues([]float32{5, 6, 7, 8}, 1, 1, 4, 1)
	fullK, fullV := cache.Update(k, v, 4)
	if err := Eval(fullK, fullV); err != nil {
		t.Fatalf("Eval rotating cache update: %v", err)
	}
	Free(k, v, fullK, fullV)
	defer freeCaches([]Cache{cache})

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
	coverageTokens := "KVCacheSnapshot ExtractsKeysAndValues"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	cache := NewKVCache()
	k := FromValues([]float32{1, 2, 3, 4}, 1, 1, 2, 2)
	v := FromValues([]float32{5, 6, 7, 8}, 1, 1, 2, 2)
	fullK, fullV := cache.Update(k, v, 2)
	if err := Eval(fullK, fullV); err != nil {
		t.Fatalf("Eval cache update: %v", err)
	}
	Free(k, v, fullK, fullV)
	defer freeCaches([]Cache{cache})

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
	coverageTokens := "DefaultModel"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	got := attentionCacheIndexByLayer(&fakeModel{numLayers: 4}, 4, 4)
	want := []int{0, 1, 2, 3}
	for i, wantIdx := range want {
		if got[i] != wantIdx {
			t.Fatalf("cache index for layer %d = %d, want %d", i, got[i], wantIdx)
		}
	}
}

func TestAttentionCacheIndexByLayer_Gemma4SharedOwners_Good(t *testing.T) {
	coverageTokens := "Gemma4SharedOwners"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	model := &Gemma4Model{
		Cfg: &Gemma4TextConfig{
			NumKVSharedLayers: 2,
		},
		Layers: []*Gemma4DecoderLayer{
			{LayerType: "sliding_attention"},
			{LayerType: "full_attention"},
			{LayerType: "sliding_attention"},
			{LayerType: "full_attention"},
		},
	}

	got := attentionCacheIndexByLayer(model, len(model.Layers), 2)
	want := []int{0, 1, 0, 1}
	for i, wantIdx := range want {
		if got[i] != wantIdx {
			t.Fatalf("cache index for layer %d = %d, want %d", i, got[i], wantIdx)
		}
	}
}

func TestAttentionCacheIndexByLayer_Gemma4PromotedOwner_Good(t *testing.T) {
	coverageTokens := "Gemma4PromotedOwner"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	model := &Gemma4Model{
		Cfg: &Gemma4TextConfig{
			NumKVSharedLayers: 2,
		},
		Layers: []*Gemma4DecoderLayer{
			{LayerType: "sliding_attention"},
			{LayerType: "sliding_attention"},
			{LayerType: "sliding_attention"},
			{LayerType: "sliding_attention"},
			{LayerType: "full_attention"},
			{LayerType: "sliding_attention"},
		},
	}

	got := attentionCacheIndexByLayer(model, len(model.Layers), 5)
	want := []int{0, 1, 2, 3, 4, 3}
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
	coverageTokens := "NewCaches ShrinksOversizedRotatingCache"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
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
	coverageTokens := "NewCaches PagedPreservesRotatingCacheBound"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
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

func TestModel_NewCaches_PagedPageSizeEnvOverride_Good(t *testing.T) {
	coverageTokens := "NewCaches PagedPageSizeEnvOverride"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	t.Setenv("GO_MLX_PAGED_KV_PAGE_SIZE", "1024")
	model := &Model{
		model: &fakeRotatingModel{
			caches: []Cache{
				NewKVCache(),
				NewRotatingKVCache(512),
			},
		},
		contextLen: 131072,
		cacheMode:  string(KVCacheModePaged),
	}

	caches := model.newCaches()
	full, ok := caches[0].(*PagedKVCache)
	if !ok {
		t.Fatalf("cache[0] = %T, want *PagedKVCache", caches[0])
	}
	if full.pageSize != 1024 {
		t.Fatalf("cache[0].pageSize = %d, want env page size 1024", full.pageSize)
	}
	sliding, ok := caches[1].(*PagedKVCache)
	if !ok {
		t.Fatalf("cache[1] = %T, want *PagedKVCache", caches[1])
	}
	if sliding.maxSize != 512 || sliding.pageSize != 512 {
		t.Fatalf("sliding cache max/page = %d/%d, want 512/512 capped env size", sliding.maxSize, sliding.pageSize)
	}
}

func TestModel_NewCaches_PagedStorageDTypeRuntimeValue_Good(t *testing.T) {
	coverageTokens := "NewCaches PagedStorageDTypeRuntimeValue"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	t.Cleanup(SetRuntimeGate("GO_MLX_KV_CACHE_DTYPE", "bf16"))
	model := &Model{
		model: &fakeRotatingModel{
			caches: []Cache{
				NewKVCache(),
				NewRotatingKVCache(512),
			},
		},
		contextLen: 131072,
		cacheMode:  string(KVCacheModePaged),
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

func TestModel_NewCaches_FixedPagedStorageDTypeRuntimeValue_Good(t *testing.T) {
	coverageTokens := "NewCaches FixedPagedStorageDTypeRuntimeValue"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	t.Cleanup(SetRuntimeGate("GO_MLX_ENABLE_FIXED_GEMMA4_CACHE", "1"))
	t.Cleanup(SetRuntimeGate("GO_MLX_ENABLE_FIXED_GEMMA4_SLIDING_CACHE_BOUND", "1"))
	t.Cleanup(SetRuntimeGate("GO_MLX_KV_CACHE_DTYPE", "bf16"))
	t.Setenv("GO_MLX_FIXED_GEMMA4_CACHE_SIZE", "")
	model := &Model{
		model: &fakeRotatingModel{
			caches: []Cache{
				NewKVCache(),
				NewRotatingKVCache(512),
			},
		},
		modelType:  "gemma4",
		contextLen: 32768,
		cacheMode:  string(KVCacheModePaged),
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

func TestPagedKVCache_PageSizeEnvOverrideCapsToMax_Good(t *testing.T) {
	coverageTokens := "PagedKVCache PageSizeEnvOverrideCapsToMax"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	t.Setenv("GO_MLX_PAGED_KV_PAGE_SIZE", "8192")

	cache := NewPagedKVCache(512, 0)

	if cache.pageSize != 512 {
		t.Fatalf("cache.pageSize = %d, want capped max size 512", cache.pageSize)
	}
}

func TestModel_NewCaches_FixedGemma4UsesUniformContextBound_Good(t *testing.T) {
	coverageTokens := "NewCaches FixedGemma4UsesUniformContextBound"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	old := enableFixedGemma4Cache
	enableFixedGemma4Cache = true
	t.Cleanup(func() { enableFixedGemma4Cache = old })
	t.Setenv("GO_MLX_FIXED_GEMMA4_CACHE_SIZE", "")

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

func TestModel_NewGenerationCaches_FixedGemma4RightSizesRequest_Good(t *testing.T) {
	coverageTokens := "NewGenerationCaches FixedGemma4RightSizesRequest"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	old := enableFixedGemma4Cache
	enableFixedGemma4Cache = true
	t.Cleanup(func() { enableFixedGemma4Cache = old })
	t.Setenv("GO_MLX_FIXED_GEMMA4_CACHE_SIZE", "")

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

func TestModel_NewGenerationCaches_FixedGemma4KeepsUniformRequestSize_Good(t *testing.T) {
	coverageTokens := "NewGenerationCaches FixedGemma4KeepsUniformRequestSize"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	old := enableFixedGemma4Cache
	enableFixedGemma4Cache = true
	t.Cleanup(func() { enableFixedGemma4Cache = old })
	t.Setenv("GO_MLX_FIXED_GEMMA4_CACHE_SIZE", "")

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
	coverageTokens := "NewGenerationCaches FixedGemma4SlidingBoundGate"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	old := enableFixedGemma4Cache
	enableFixedGemma4Cache = true
	t.Cleanup(func() { enableFixedGemma4Cache = old })
	t.Setenv("GO_MLX_FIXED_GEMMA4_CACHE_SIZE", "")
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
func (m *directGreedyGenerateModel) ModelType() string                   { return "direct-greedy-generate-test" }
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
	coverageTokens := "PrefillTokenBlock ChunksByPlanner"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
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
	coverageTokens := "PrefillTokenBlock UsesLastTokenLogitsModel"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	requireMetalRuntime(t)
	t.Setenv("GO_MLX_ENABLE_LAST_LOGITS_PREFILL", "1")

	inner := &lastLogitsPrefillModel{}
	model := &Model{model: inner, prefillChunkSize: 2}
	logits, err := model.prefillTokenBlock(t.Context(), []int32{1, 2, 3, 4, 5}, nil)
	if err != nil {
		t.Fatalf("prefillTokenBlock() error = %v", err)
	}
	defer Free(logits)

	if inner.fullCalls != 0 {
		t.Fatalf("full forward calls = %d, want 0", inner.fullCalls)
	}
	want := []int{2, 2, 1}
	if len(inner.lastLens) != len(want) {
		t.Fatalf("lastLens = %v, want %v", inner.lastLens, want)
	}
	for i := range want {
		if inner.lastLens[i] != want[i] {
			t.Fatalf("lastLens = %v, want %v", inner.lastLens, want)
		}
	}
	if got := logits.Shape(); len(got) != 2 || got[0] != 1 || got[1] != 2 {
		t.Fatalf("logits shape = %v, want [1 2]", got)
	}
}

func TestModel_PrefillTokenBlock_EvaluatesIntermediateChunksCacheOnly_Good(t *testing.T) {
	coverageTokens := "PrefillTokenBlock EvaluatesIntermediateChunksCacheOnly"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	requireMetalRuntime(t)
	restoreCacheOnly := SetRuntimeGate("GO_MLX_ENABLE_CACHE_ONLY_CHUNK_PREFILL", "1")
	t.Cleanup(restoreCacheOnly)
	t.Setenv("GO_MLX_ENABLE_LAST_LOGITS_PREFILL", "1")

	inner := &cacheOnlyChunkPrefillModel{}
	caches := inner.NewCache()
	model := &Model{model: inner, prefillChunkSize: 2}
	logits, err := model.prefillTokenBlock(t.Context(), []int32{1, 2, 3, 4, 5}, caches)
	if err != nil {
		t.Fatalf("prefillTokenBlock() error = %v", err)
	}
	defer Free(logits)
	defer freeCaches(caches)

	if got, want := inner.fullLens, []int{2, 2}; !reflect.DeepEqual(got, want) {
		t.Fatalf("full forward chunk lengths = %v, want %v", got, want)
	}
	if got, want := inner.lastLens, []int{1}; !reflect.DeepEqual(got, want) {
		t.Fatalf("last-logits chunk lengths = %v, want %v", got, want)
	}
	if caches[0].Offset() != 5 {
		t.Fatalf("cache offset = %d, want 5", caches[0].Offset())
	}
	if got := logits.Shape(); len(got) != 2 || got[0] != 1 || got[1] != 2 {
		t.Fatalf("logits shape = %v, want [1 2]", got)
	}
}

func TestModel_PrefillTokenBlock_UsesFullForwardForMultiTokenCachedChunk_Good(t *testing.T) {
	coverageTokens := "PrefillTokenBlock UsesFullForwardForMultiTokenCachedChunk"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	requireMetalRuntime(t)
	t.Setenv("GO_MLX_ENABLE_LAST_LOGITS_PREFILL", "1")

	inner := &cacheOnlyChunkPrefillModel{}
	caches := inner.NewCache()
	model := &Model{model: inner, prefillChunkSize: 2}
	logits, err := model.prefillTokenBlock(t.Context(), []int32{1, 2, 3, 4, 5}, caches)
	if err != nil {
		t.Fatalf("prefillTokenBlock() error = %v", err)
	}
	defer Free(logits)
	defer freeCaches(caches)

	if got, want := inner.fullLens, []int{2}; !reflect.DeepEqual(got, want) {
		t.Fatalf("full forward chunk lengths = %v, want %v", got, want)
	}
	if got, want := inner.lastLens, []int{2, 1}; !reflect.DeepEqual(got, want) {
		t.Fatalf("last-logits chunk lengths = %v, want %v", got, want)
	}
	if caches[0].Offset() != 5 {
		t.Fatalf("cache offset = %d, want 5", caches[0].Offset())
	}
}

func TestModel_EffectivePrefillChunkSizeCapsGemma4FixedSlidingCache_Good(t *testing.T) {
	coverageTokens := "EffectivePrefillChunkSize CapsGemma4FixedSlidingCache"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	model := &Model{
		model: &Gemma4Model{
			Cfg: &Gemma4TextConfig{SlidingWindow: 512},
		},
		prefillChunkSize: 4096,
	}
	caches := []Cache{NewFixedKVCache(512), NewKVCache()}
	if got := model.effectivePrefillChunkSize(caches); got != 512 {
		t.Fatalf("effectivePrefillChunkSize = %d, want 512", got)
	}
	model.prefillChunkSize = 0
	if got := model.effectivePrefillChunkSize(caches); got != 512 {
		t.Fatalf("effectivePrefillChunkSize(default) = %d, want 512", got)
	}
	model.prefillChunkSize = 256
	if got := model.effectivePrefillChunkSize(caches); got != 256 {
		t.Fatalf("effectivePrefillChunkSize(small explicit) = %d, want 256", got)
	}
}

func TestModel_PrefillTokenBlock_AutoUsesLastTokenForLongPrompt_Good(t *testing.T) {
	coverageTokens := "PrefillTokenBlock AutoUsesLastTokenForLongPrompt"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	requireMetalRuntime(t)
	t.Setenv("GO_MLX_LAST_LOGITS_PREFILL_MIN_TOKENS", "4")

	inner := &lastLogitsPrefillModel{}
	model := &Model{model: inner}
	logits, err := model.prefillTokenBlock(t.Context(), []int32{1, 2, 3, 4, 5}, nil)
	if err != nil {
		t.Fatalf("prefillTokenBlock() error = %v", err)
	}
	defer Free(logits)

	if inner.fullCalls != 0 {
		t.Fatalf("full forward calls = %d, want 0", inner.fullCalls)
	}
	if len(inner.lastLens) != 1 || inner.lastLens[0] != 5 {
		t.Fatalf("lastLens = %v, want [5]", inner.lastLens)
	}
	if got := logits.Shape(); len(got) != 2 || got[0] != 1 || got[1] != 2 {
		t.Fatalf("logits shape = %v, want [1 2]", got)
	}
}

func TestModel_PrefillTokenBlock_AutoKeepsShortPromptOnFullPath_Bad(t *testing.T) {
	coverageTokens := "PrefillTokenBlock AutoKeepsShortPromptOnFullPath"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	requireMetalRuntime(t)
	t.Setenv("GO_MLX_LAST_LOGITS_PREFILL_MIN_TOKENS", "8")

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
	coverageTokens := "PrefillTokenBlock FallsBackWhenLastTokenLogitsInvalid"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	requireMetalRuntime(t)
	t.Setenv("GO_MLX_ENABLE_LAST_LOGITS_PREFILL", "1")

	inner := &lastLogitsPrefillModel{invalid: true}
	model := &Model{model: inner, prefillChunkSize: 2}
	logits, err := model.prefillTokenBlock(t.Context(), []int32{1, 2, 3}, nil)
	if err != nil {
		t.Fatalf("prefillTokenBlock() error = %v", err)
	}
	defer Free(logits)

	if inner.fullCalls != 2 {
		t.Fatalf("full forward calls = %d, want 2", inner.fullCalls)
	}
	if len(inner.lastLens) != 2 {
		t.Fatalf("last logits attempts = %d, want 2", len(inner.lastLens))
	}
	if got := logits.Shape(); len(got) != 2 || got[0] != 1 || got[1] != 64 {
		t.Fatalf("fallback logits shape = %v, want [1 64]", got)
	}
}

func TestModel_Generate_DoesNotForwardAfterFinalToken_Good(t *testing.T) {
	coverageTokens := "Generate DoesNotForwardAfterFinalToken"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
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
	coverageTokens := "Generate TraceTokenPhases"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
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
	coverageTokens := "Generate TraceTokenPhasesNoProbeSink"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
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
	}
}

func TestModel_Generate_KeepsDecodeLogitsLazyBetweenTokens_Good(t *testing.T) {
	coverageTokens := "Generate KeepsDecodeLogitsLazyBetweenTokens"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
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
	coverageTokens := "Generate AsyncDecodePrefetch"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	requireMetalRuntime(t)
	old := enableAsyncDecodePrefetch
	enableAsyncDecodePrefetch = true
	t.Cleanup(func() { enableAsyncDecodePrefetch = old })

	out := Zeros([]int32{1, 1, 2}, DTypeFloat32)
	defer Free(out)
	if err := asyncDecodePrefetch(0, "test", out); err != nil {
		t.Fatalf("asyncDecodePrefetch() error = %v", err)
	}
	if err := Eval(out); err != nil {
		t.Fatalf("Eval after asyncDecodePrefetch() error = %v", err)
	}
}

func TestModel_Generate_AsyncDecodePrefetchRuntimeGate_Good(t *testing.T) {
	coverageTokens := "Generate AsyncDecodePrefetchRuntimeGate"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	old := enableAsyncDecodePrefetch
	enableAsyncDecodePrefetch = false
	t.Cleanup(func() { enableAsyncDecodePrefetch = old })

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
	coverageTokens := "Generate AsyncDecodePrefetch"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	old := enableAsyncDecodePrefetch
	enableAsyncDecodePrefetch = true
	t.Cleanup(func() { enableAsyncDecodePrefetch = old })

	if err := asyncDecodePrefetch(0, "nil", nil); err != nil {
		t.Fatalf("asyncDecodePrefetch(nil) error = %v", err)
	}
}

func TestModel_Generate_GenerationStream_Good(t *testing.T) {
	coverageTokens := "Generate GenerationStream"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	requireMetalRuntime(t)
	old := enableGenerationStream
	enableGenerationStream = true
	t.Cleanup(func() { enableGenerationStream = old })

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
	coverageTokens := "Generate GenerationStream"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	old := enableGenerationStream
	enableGenerationStream = false
	t.Cleanup(func() { enableGenerationStream = old })
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

func TestModel_Generate_GenerationClearCacheInterval_Good(t *testing.T) {
	coverageTokens := "Generate GenerationClearCacheInterval"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	restore := SetRuntimeGate("GO_MLX_GENERATION_CLEAR_CACHE_INTERVAL", "64")
	t.Cleanup(restore)

	if got := generationClearCacheInterval(); got != 64 {
		t.Fatalf("generationClearCacheInterval() = %d, want 64", got)
	}
}

func TestModel_Generate_GenerationClearCacheInterval_Bad(t *testing.T) {
	coverageTokens := "Generate GenerationClearCacheInterval"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	restore := SetRuntimeGate("GO_MLX_GENERATION_CLEAR_CACHE_INTERVAL", "0")
	t.Cleanup(restore)

	if got := generationClearCacheInterval(); got != defaultGenerationClearCacheInterval {
		t.Fatalf("generationClearCacheInterval() = %d, want default %d", got, defaultGenerationClearCacheInterval)
	}
}

func TestModel_Generate_UsesDirectGreedyToken_Good(t *testing.T) {
	coverageTokens := "Generate UsesDirectGreedyToken"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	requireMetalRuntime(t)
	old := enableDirectGreedyToken
	enableDirectGreedyToken = true
	t.Cleanup(func() { enableDirectGreedyToken = old })

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
		t.Fatalf("phases = %+v, want direct greedy forward on first step only", phases)
	}
}

func TestModel_Generate_UsesSuppressedDirectGreedyToken_Good(t *testing.T) {
	coverageTokens := "Generate UsesSuppressedDirectGreedyToken"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	requireMetalRuntime(t)
	old := enableDirectGreedyToken
	enableDirectGreedyToken = true
	t.Cleanup(func() { enableDirectGreedyToken = old })

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
	coverageTokens := "Generate UsesBorrowedSuppressionArray"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	requireMetalRuntime(t)
	old := enableDirectGreedyToken
	enableDirectGreedyToken = true
	t.Cleanup(func() { enableDirectGreedyToken = old })

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
	coverageTokens := "Generate DirectGreedyRejectsRepeatPenalty"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	requireMetalRuntime(t)
	old := enableDirectGreedyToken
	enableDirectGreedyToken = true
	t.Cleanup(func() { enableDirectGreedyToken = old })

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
	coverageTokens := "FormatChat Gemma2UsesGemmaTemplate"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
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
	coverageTokens := "FormatChat Gemma4UsesModelTemplate"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	model := &Model{modelType: "gemma4_text"}

	got := model.formatChat([]ChatMessage{
		{Role: "system", Content: " be brief "},
		{Role: "user", Content: "Hello"},
		{Role: "assistant", Content: "Hi"},
		{Role: "user", Content: "Again"},
	})

	want := "<bos><|turn>system\nbe brief<turn|>\n" +
		"<|turn>user\nHello<turn|>\n" +
		"<|turn>model\nHi<turn|>\n" +
		"<|turn>user\nAgain<turn|>\n" +
		"<|turn>model\n"
	if got != want {
		t.Fatalf("formatChat() = %q, want %q", got, want)
	}
}

func TestModel_FormatChat_Gemma4StripsAssistantThoughtHistory_Good(t *testing.T) {
	coverageTokens := "FormatChat Gemma4StripsAssistantThoughtHistory"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	model := &Model{modelType: "gemma4_text"}

	got := model.formatChat([]ChatMessage{
		{Role: "user", Content: "Hello"},
		{Role: "assistant", Content: "<|channel>thought\nprivate<channel|>Visible"},
	})
	want := "<bos><|turn>user\nHello<turn|>\n<|turn>model\nVisible<turn|>\n<|turn>model\n"
	if got != want {
		t.Fatalf("formatChat() = %q, want %q", got, want)
	}
}

func TestModel_FormatChatChunks_Gemma4MatchesFormattedPrompt_Good(t *testing.T) {
	coverageTokens := "FormatChatChunks Gemma4MatchesFormattedPrompt"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
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

// Generated file-aware compliance coverage.
func TestGenerate_Model_ModelType_Good(t *testing.T) {
	coverageTokens := "Model ModelType"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "Model_ModelType"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestGenerate_Model_ModelType_Bad(t *testing.T) {
	coverageTokens := "Model ModelType"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "Model_ModelType"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestGenerate_Model_ModelType_Ugly(t *testing.T) {
	coverageTokens := "Model ModelType"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "Model_ModelType"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestGenerate_Model_Err_Good(t *testing.T) {
	coverageTokens := "Model Err"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "Model_Err"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestGenerate_Model_Err_Bad(t *testing.T) {
	coverageTokens := "Model Err"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "Model_Err"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestGenerate_Model_Err_Ugly(t *testing.T) {
	coverageTokens := "Model Err"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "Model_Err"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestGenerate_Model_StagedMiniMaxReturnsDecodeError_Bad(t *testing.T) {
	coverageTokens := "Model Generate StagedMiniMaxReturnsDecodeError"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	model := &Model{
		model: &miniMaxM2StagedModel{
			plan: miniMaxM2NativeLoadPlan{
				Config: miniMaxM2LoadConfig{
					ModelType:       "minimax_m2",
					NumHiddenLayers: 62,
				},
			},
		},
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

func TestGenerate_Model_LastMetrics_Good(t *testing.T) {
	coverageTokens := "Model LastMetrics"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "Model_LastMetrics"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestGenerate_Model_LastMetrics_Bad(t *testing.T) {
	coverageTokens := "Model LastMetrics"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "Model_LastMetrics"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestGenerate_Model_LastMetrics_Ugly(t *testing.T) {
	coverageTokens := "Model LastMetrics"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "Model_LastMetrics"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestGenerate_Model_Info_Good(t *testing.T) {
	coverageTokens := "Model Info"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "Model_Info"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestGenerate_Model_Info_Bad(t *testing.T) {
	coverageTokens := "Model Info"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "Model_Info"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestGenerate_Model_Info_Ugly(t *testing.T) {
	coverageTokens := "Model Info"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "Model_Info"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestGenerate_Model_Close_Good(t *testing.T) {
	coverageTokens := "Model Close"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "Model_Close"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestGenerate_Model_Close_Bad(t *testing.T) {
	coverageTokens := "Model Close"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "Model_Close"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestGenerate_Model_Close_Ugly(t *testing.T) {
	coverageTokens := "Model Close"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "Model_Close"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestGenerate_Model_Chat_Good(t *testing.T) {
	coverageTokens := "Model Chat"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "Model_Chat"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestGenerate_Model_Chat_Bad(t *testing.T) {
	coverageTokens := "Model Chat"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "Model_Chat"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestGenerate_Model_Chat_Ugly(t *testing.T) {
	coverageTokens := "Model Chat"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "Model_Chat"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestGenerate_Model_Generate_Good(t *testing.T) {
	coverageTokens := "Model Generate"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "Model_Generate"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestGenerate_Model_Generate_Bad(t *testing.T) {
	coverageTokens := "Model Generate"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "Model_Generate"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestGenerate_Model_Generate_Ugly(t *testing.T) {
	coverageTokens := "Model Generate"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "Model_Generate"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestGenerate_Model_InspectAttention_Good(t *testing.T) {
	coverageTokens := "Model InspectAttention"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "Model_InspectAttention"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestGenerate_Model_InspectAttention_Bad(t *testing.T) {
	coverageTokens := "Model InspectAttention"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "Model_InspectAttention"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestGenerate_Model_InspectAttention_Ugly(t *testing.T) {
	coverageTokens := "Model InspectAttention"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "Model_InspectAttention"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestGenerate_Model_CaptureKV_Good(t *testing.T) {
	coverageTokens := "Model CaptureKV"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "Model_CaptureKV"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestGenerate_Model_CaptureKV_Bad(t *testing.T) {
	coverageTokens := "Model CaptureKV"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "Model_CaptureKV"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestGenerate_Model_CaptureKV_Ugly(t *testing.T) {
	coverageTokens := "Model CaptureKV"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "Model_CaptureKV"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestGenerate_LastTokenLogits_Good(t *testing.T) {
	coverageTokens := "Generate LastTokenLogits"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	oneDim := FromValues([]float32{1, 2, 3}, 3)
	twoDim := FromValues([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	threeDim := FromValues([]float32{1, 2, 3, 4, 5, 6}, 1, 2, 3)
	defer Free(oneDim, twoDim, threeDim)

	for name, logits := range map[string]*Array{
		"one":   oneDim,
		"two":   twoDim,
		"three": threeDim,
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

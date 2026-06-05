// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"context"
	"encoding/binary"
	"math"
	"reflect"
	"testing"

	"dappco.re/go"
)

func TestPromptCache_PagedKVCacheSnapshotIsEvaluable_Good(t *testing.T) {
	requireMetalRuntime(t)

	cache := NewPagedKVCache(8, 2)
	k, v := makeKV(3)
	defer Free(k, v)

	outK, outV := cache.Update(k, v, 3)
	logits := Add(outK, outV)
	defer Free(outK, outV, logits)
	if err := Eval(logits); err != nil {
		t.Fatalf("Eval logits: %v", err)
	}
	detachEvalState(logits, []Cache{cache})
	defer cache.Reset()

	entry, err := newPromptCacheEntry([]int32{1, 2, 3}, []Cache{cache}, logits)
	if err != nil {
		t.Fatalf("newPromptCacheEntry() error = %v", err)
	}
	defer entry.free()

	if len(entry.caches) != 1 || entry.cacheableTokens != 3 {
		t.Fatalf("entry cache shape = len %d cacheable %d, want 1/3", len(entry.caches), entry.cacheableTokens)
	}
}

func TestPromptCache_PagedKVCacheSnapshotsTransformedPages_Good(t *testing.T) {
	requireMetalRuntime(t)

	cache := NewPagedKVCache(8, 2)
	kBase := seqArray(0.10, 1, 3, 2, 4)
	vBase := seqArray(0.20, 1, 3, 2, 4)
	kBFloat := AsType(kBase, DTypeBFloat16)
	vBFloat := AsType(vBase, DTypeBFloat16)
	kStrided := AsStrided(kBFloat, []int32{1, 2, 3, 4}, []int64{24, 4, 8, 1}, 0)
	vStrided := AsStrided(vBFloat, []int32{1, 2, 3, 4}, []int64{24, 4, 8, 1}, 0)
	kNormed := RMSNormNoScale(kStrided, 1e-6)
	vNormed := RMSNormNoScale(vStrided, 1e-6)
	k := RoPE(kNormed, 4, false, 10000, 1, 0)
	v := vNormed
	defer Free(kBase, vBase, kBFloat, vBFloat, kStrided, vStrided, kNormed, vNormed, k)

	outK, outV := cache.Update(k, v, 3)
	logits := Add(outK, outV)
	defer Free(outK, outV, logits)
	if err := Eval(logits); err != nil {
		t.Fatalf("Eval logits: %v", err)
	}
	detachEvalState(logits, []Cache{cache})
	defer cache.Reset()

	entry, err := newPromptCacheEntry([]int32{1, 2, 3}, []Cache{cache}, logits)
	if err != nil {
		t.Fatalf("newPromptCacheEntry() error = %v", err)
	}
	defer entry.free()
}

func TestPromptCache_EvalCachesBeforeDetachSkipsPagedCaches_Good(t *testing.T) {
	requireMetalRuntime(t)

	kvCache := NewKVCache()
	pagedCache := NewPagedKVCache(8, 2)
	k, v := makeKV(2)
	defer Free(k, v)
	kvK, kvV := kvCache.Update(k, v, 2)
	pagedK, pagedV := pagedCache.Update(k, v, 2)
	defer Free(kvK, kvV, pagedK, pagedV)
	defer kvCache.Reset()
	defer pagedCache.Reset()

	state := cacheStateArraysForDetach([]Cache{kvCache, pagedCache})
	if len(state) != 2 {
		t.Fatalf("cacheStateArraysForDetach len = %d, want only KVCache K/V state", len(state))
	}
	if state[0] != kvCache.keys || state[1] != kvCache.values {
		t.Fatal("cacheStateArraysForDetach should include contiguous KVCache state and skip paged pages")
	}
	if err := evalCachesBeforeDetach([]Cache{kvCache, pagedCache}); err != nil {
		t.Fatalf("evalCachesBeforeDetach: %v", err)
	}
}

func TestPromptCache_EvalCachesBeforeDetachKeepsChunkedKVCacheEvaluable_Good(t *testing.T) {
	requireMetalRuntime(t)

	cache := NewKVCache()
	defer cache.Reset()

	k1 := FromValues([]float32{1, 2}, 1, 1, 2, 1)
	v1 := FromValues([]float32{10, 20}, 1, 1, 2, 1)
	defer Free(k1, v1)
	firstK, firstV := cache.Update(k1, v1, 2)
	logits := Add(firstK, firstV)
	if err := Eval(logits); err != nil {
		t.Fatalf("Eval first logits: %v", err)
	}
	if err := evalCachesBeforeDetach([]Cache{cache}); err != nil {
		t.Fatalf("evalCachesBeforeDetach first chunk: %v", err)
	}
	DetachCaches([]Cache{cache})
	Free(firstK, firstV, logits)

	k2 := FromValues([]float32{3, 4}, 1, 1, 2, 1)
	v2 := FromValues([]float32{30, 40}, 1, 1, 2, 1)
	defer Free(k2, v2)
	gotK, gotV := cache.Update(k2, v2, 2)
	defer Free(gotK, gotV)
	if err := Eval(gotK, gotV); err != nil {
		t.Fatalf("Eval second chunk cache: %v", err)
	}
	floatSliceApprox(t, gotK.Floats(), []float32{1, 2, 3, 4})
	floatSliceApprox(t, gotV.Floats(), []float32{10, 20, 30, 40})
}

func TestPromptCache_RestoresQuantizedQ8Prefix_Good(t *testing.T) {
	requireMetalRuntime(t)

	cache := NewQuantizedKVCache(0, 8, 8)
	k := FromValues([]float32{1, 2, 3, 4}, 1, 1, 4, 1)
	v := FromValues([]float32{5, 6, 7, 8}, 1, 1, 4, 1)
	fullK, fullV := cache.Update(k, v, 4)
	if err := Eval(fullK, fullV); err != nil {
		t.Fatalf("Eval quantized cache update: %v", err)
	}
	Free(k, v, fullK, fullV)
	defer FreeCaches([]Cache{cache})

	snapshot, ok, err := snapshotCache(cache, 4)
	if err != nil {
		t.Fatalf("snapshotCache() error = %v", err)
	}
	if !ok {
		t.Fatal("snapshotCache() ok = false, want true")
	}
	defer freeCacheSnapshots([]cacheSnapshot{snapshot})
	if snapshot.mode != KVCacheModeQ8 {
		t.Fatalf("snapshot mode = %q, want q8", snapshot.mode)
	}

	restored, err := restorePromptCaches([]cacheSnapshot{snapshot}, 2)
	if err != nil {
		t.Fatalf("restorePromptCaches() error = %v", err)
	}
	defer FreeCaches(restored)
	restoredCache, ok := restored[0].(*QuantizedKVCache)
	if !ok {
		t.Fatalf("restored cache = %T, want *QuantizedKVCache", restored[0])
	}
	if restoredCache.Len() != 2 || restoredCache.Offset() != 2 {
		t.Fatalf("restored len/offset = %d/%d, want 2/2", restoredCache.Len(), restoredCache.Offset())
	}
	state, owned := restoredCache.ReadState()
	defer Free(owned...)
	if len(state) != 2 || state[0].Shape()[2] != 2 {
		t.Fatalf("restored state shape = %v, want prefix length 2", state)
	}
}

func TestPromptCache_RestoresPagedPrefix_Good(t *testing.T) {
	requireMetalRuntime(t)

	cache := NewPagedKVCache(0, 2)
	k := FromValues([]float32{1, 2, 3, 4, 5}, 1, 1, 5, 1)
	v := FromValues([]float32{6, 7, 8, 9, 10}, 1, 1, 5, 1)
	fullK, fullV := cache.Update(k, v, 5)
	if err := Eval(fullK, fullV); err != nil {
		t.Fatalf("Eval paged cache update: %v", err)
	}
	Free(k, v, fullK, fullV)
	defer FreeCaches([]Cache{cache})

	snapshot, ok, err := snapshotCache(cache, 5)
	if err != nil {
		t.Fatalf("snapshotCache() error = %v", err)
	}
	if !ok {
		t.Fatal("snapshotCache() ok = false, want true")
	}
	defer freeCacheSnapshots([]cacheSnapshot{snapshot})
	if snapshot.mode != KVCacheModePaged || len(snapshot.kPages) != 3 {
		t.Fatalf("snapshot mode/pages = %q/%d, want paged physical state", snapshot.mode, len(snapshot.kPages))
	}

	restored, err := restorePromptCaches([]cacheSnapshot{snapshot}, 3)
	if err != nil {
		t.Fatalf("restorePromptCaches() error = %v", err)
	}
	defer FreeCaches(restored)
	restoredCache, ok := restored[0].(*PagedKVCache)
	if !ok {
		t.Fatalf("restored cache = %T, want *PagedKVCache", restored[0])
	}
	if restoredCache.Len() != 3 || restoredCache.Offset() != 3 || len(restoredCache.kPages) != 2 {
		t.Fatalf("restored len/offset/pages = %d/%d/%d, want 3/3/2", restoredCache.Len(), restoredCache.Offset(), len(restoredCache.kPages))
	}
}

func TestPromptCache_RestoresSlidingPagedTail_Good(t *testing.T) {
	requireMetalRuntime(t)

	cache := NewPagedKVCache(2, 2)
	k := FromValues([]float32{1, 2, 3, 4}, 1, 1, 4, 1)
	v := FromValues([]float32{5, 6, 7, 8}, 1, 1, 4, 1)
	fullK, fullV := cache.Update(k, v, 4)
	if err := Eval(fullK, fullV); err != nil {
		t.Fatalf("Eval paged cache update: %v", err)
	}
	Free(k, v, fullK, fullV)
	defer FreeCaches([]Cache{cache})

	snapshot, ok, err := snapshotCache(cache, 4)
	if err != nil {
		t.Fatalf("snapshotCache() error = %v", err)
	}
	if !ok {
		t.Fatal("snapshotCache() ok = false, want true")
	}
	defer freeCacheSnapshots([]cacheSnapshot{snapshot})
	if snapshot.mode != KVCacheModePaged || snapshot.maxSize != 2 || snapshot.length != 2 || snapshot.offset != 4 {
		t.Fatalf("snapshot mode/max/length/offset = %q/%d/%d/%d, want paged/2/2/4", snapshot.mode, snapshot.maxSize, snapshot.length, snapshot.offset)
	}

	restored, err := restorePromptCaches([]cacheSnapshot{snapshot}, 4)
	if err != nil {
		t.Fatalf("restorePromptCaches() error = %v", err)
	}
	defer FreeCaches(restored)
	restoredCache, ok := restored[0].(*PagedKVCache)
	if !ok {
		t.Fatalf("restored cache = %T, want *PagedKVCache", restored[0])
	}
	if restoredCache.Len() != 2 || restoredCache.Offset() != 4 || restoredCache.maxSize != 2 {
		t.Fatalf("restored len/offset/max = %d/%d/%d, want 2/4/2", restoredCache.Len(), restoredCache.Offset(), restoredCache.maxSize)
	}
}

func TestPromptCache_RestoresFixedPrefix_Good(t *testing.T) {
	requireMetalRuntime(t)

	cache := NewFixedKVCache(6)
	k := FromValues([]float32{1, 2, 3, 4}, 1, 1, 4, 1)
	v := FromValues([]float32{5, 6, 7, 8}, 1, 1, 4, 1)
	fullK, fullV := cache.Update(k, v, 4)
	if err := Eval(fullK, fullV); err != nil {
		t.Fatalf("Eval fixed cache update: %v", err)
	}
	Free(k, v, fullK, fullV)
	defer FreeCaches([]Cache{cache})

	snapshot, ok, err := snapshotCache(cache, 4)
	if err != nil {
		t.Fatalf("snapshotCache() error = %v", err)
	}
	if !ok {
		t.Fatal("snapshotCache() ok = false, want true")
	}
	defer freeCacheSnapshots([]cacheSnapshot{snapshot})
	if snapshot.mode != KVCacheModeFixed || snapshot.maxSize != 6 {
		t.Fatalf("snapshot mode/maxSize = %q/%d, want fixed/6", snapshot.mode, snapshot.maxSize)
	}

	restored, err := RestorePromptCachesWithRequestFixedSize([]cacheSnapshot{snapshot}, 3, 8)
	if err != nil {
		t.Fatalf("RestorePromptCachesWithRequestFixedSize() error = %v", err)
	}
	defer FreeCaches(restored)
	restoredCache, ok := restored[0].(*FixedKVCache)
	if !ok {
		t.Fatalf("restored cache = %T, want *FixedKVCache", restored[0])
	}
	if restoredCache.Len() != 3 || restoredCache.Offset() != 3 || restoredCache.maxSize != 8 {
		t.Fatalf("restored len/offset/max = %d/%d/%d, want 3/3/8", restoredCache.Len(), restoredCache.Offset(), restoredCache.maxSize)
	}
	state := restoredCache.State()
	if len(state) != 2 || state[0].Shape()[2] != 8 {
		t.Fatalf("fixed backing shape = %v, want capacity 8", state)
	}
	readState, owned := restoredCache.ReadState()
	defer Free(owned...)
	if len(readState) != 2 || readState[0].Shape()[2] != 3 {
		t.Fatalf("readable fixed prefix shape = %v, want length 3", readState)
	}
}

func TestPromptCache_RestoresSlidingFixedTail_Good(t *testing.T) {
	requireMetalRuntime(t)
	restoreGate := SetRuntimeGate("GO_MLX_ENABLE_FIXED_GEMMA4_SLIDING_CACHE_BOUND", "1")
	t.Cleanup(restoreGate)

	cache := NewFixedKVCache(2)
	k := FromValues([]float32{1, 2, 3, 4}, 1, 1, 4, 1)
	v := FromValues([]float32{5, 6, 7, 8}, 1, 1, 4, 1)
	fullK, fullV := cache.Update(k, v, 4)
	if err := Eval(fullK, fullV); err != nil {
		t.Fatalf("Eval fixed cache update: %v", err)
	}
	Free(k, v, fullK, fullV)
	defer FreeCaches([]Cache{cache})

	snapshot, ok, err := snapshotCache(cache, 4)
	if err != nil {
		t.Fatalf("snapshotCache() error = %v", err)
	}
	if !ok {
		t.Fatal("snapshotCache() ok = false, want true")
	}
	defer freeCacheSnapshots([]cacheSnapshot{snapshot})
	if snapshot.mode != KVCacheModeFixed || snapshot.maxSize != 2 || snapshot.length != 2 || snapshot.offset != 4 {
		t.Fatalf("snapshot mode/max/length/offset = %q/%d/%d/%d, want fixed/2/2/4", snapshot.mode, snapshot.maxSize, snapshot.length, snapshot.offset)
	}

	restored, err := RestorePromptCachesWithRequestFixedSize([]cacheSnapshot{snapshot}, 4, 8)
	if err != nil {
		t.Fatalf("RestorePromptCachesWithRequestFixedSize() error = %v", err)
	}
	defer FreeCaches(restored)
	restoredCache, ok := restored[0].(*FixedKVCache)
	if !ok {
		t.Fatalf("restored cache = %T, want *FixedKVCache", restored[0])
	}
	if restoredCache.Len() != 2 || restoredCache.Offset() != 4 || restoredCache.maxSize != 2 {
		t.Fatalf("restored len/offset/max = %d/%d/%d, want 2/4/2", restoredCache.Len(), restoredCache.Offset(), restoredCache.maxSize)
	}
}

func TestPromptCache_RestoreTurboQuantReferencePayload_Good(t *testing.T) {
	requireMetalRuntime(t)
	cache := NewTurboQuantKVCache(0, 8)
	k, v := makeKV(3)
	fullK, fullV := cache.Update(k, v, 3)
	if err := Eval(fullK, fullV); err != nil {
		t.Fatalf("Eval TurboQuant cache update: %v", err)
	}
	defer FreeCaches([]Cache{cache})

	snapshot, ok, err := snapshotCache(cache, 3)
	if err != nil {
		t.Fatalf("snapshotCache(turboquant) error = %v", err)
	}
	if !ok {
		t.Fatal("snapshotCache(turboquant) ok = false, want true")
	}
	if snapshot.mode != KVCacheModeTurboQuant || len(snapshot.turboPayloads) != 1 {
		t.Fatalf("snapshot mode/pages = %q/%d, want turboquant with one payload page", snapshot.mode, len(snapshot.turboPayloads))
	}

	restored, err := restorePromptCaches([]cacheSnapshot{snapshot}, 3)
	if err != nil {
		t.Fatalf("restorePromptCaches(turboquant) error = %v, want nil", err)
	}
	defer FreeCaches(restored)
	restoredCache, ok := restored[0].(*TurboQuantKVCache)
	if !ok {
		t.Fatalf("restored cache = %T, want *TurboQuantKVCache", restored[0])
	}
	if restoredCache.Len() != 3 || restoredCache.Offset() != 3 {
		t.Fatalf("restored len/offset = %d/%d, want 3/3", restoredCache.Len(), restoredCache.Offset())
	}
	state := restoredCache.State()
	if len(state) != 2 {
		t.Fatalf("restored state arrays = %d, want K/V", len(state))
	}
	if got := cosineSimilarity(k.Floats(), state[0].Floats()); got < 0.98 {
		t.Fatalf("restored key cosine = %.6f, want >= 0.98", got)
	}
	if got := cosineSimilarity(v.Floats(), state[1].Floats()); got < 0.98 {
		t.Fatalf("restored value cosine = %.6f, want >= 0.98", got)
	}
}

func TestPromptCache_RestoreFromKVBlocksStreamsPagedPages_Good(t *testing.T) {
	requireMetalRuntime(t)

	model := &Model{
		model:                &fakePagedModel{numLayers: 1, pageSize: 2},
		modelType:            "fake",
		promptCacheEnabled:   true,
		promptCacheMinTokens: 1,
		cacheMode:            string(KVCacheModePaged),
	}
	source := KVSnapshotBlockSource{
		TokenCount:   4,
		PrefixTokens: 4,
		BlockCount:   2,
		Load: func(_ context.Context, index int) (KVSnapshotBlock, error) {
			switch index {
			case 0:
				return KVSnapshotBlock{Index: 0, TokenStart: 0, TokenCount: 2, Snapshot: kvSnapshotBlockTestSnapshot(0, []int32{1, 2})}, nil
			case 1:
				return KVSnapshotBlock{Index: 1, TokenStart: 2, TokenCount: 2, Snapshot: kvSnapshotBlockTestSnapshot(2, []int32{3, 4})}, nil
			default:
				return KVSnapshotBlock{}, core.NewError("unexpected block")
			}
		},
	}

	if err := model.RestorePromptCacheFromKVBlocks(context.Background(), source); err != nil {
		t.Fatalf("RestorePromptCacheFromKVBlocks() error = %v", err)
	}
	defer model.ClearPromptCache()
	if model.promptCache == nil {
		t.Fatal("promptCache = nil, want restored block cache")
	}
	if got := model.promptCache.tokens; !reflect.DeepEqual(got, []int32{1, 2, 3, 4}) {
		t.Fatalf("prompt cache tokens = %v, want [1 2 3 4]", got)
	}
	cache := model.promptCache.caches[0]
	if cache.mode != KVCacheModePaged || cache.keys != nil || cache.values != nil {
		t.Fatalf("cache snapshot mode/contiguous = %q/%v/%v, want paged without full contiguous arrays", cache.mode, cache.keys, cache.values)
	}
	if cache.length != 4 || cache.offset != 4 || len(cache.kPages) != 2 || len(cache.vPages) != 2 {
		t.Fatalf("cache length/offset/pages = %d/%d/%d/%d, want 4/4/2/2", cache.length, cache.offset, len(cache.kPages), len(cache.vPages))
	}
}

func TestPromptCache_RestoreFromKVBlocksUsesFixedGenerationCache_Good(t *testing.T) {
	requireMetalRuntime(t)
	t.Cleanup(SetRuntimeGate("GO_MLX_ENABLE_FIXED_GEMMA4_CACHE", "1"))

	native := &fakePagedModel{numLayers: 1, pageSize: 2}
	model := &Model{
		model:                native,
		modelType:            "gemma4_text",
		promptCacheEnabled:   true,
		promptCacheMinTokens: 1,
		cacheMode:            string(KVCacheModePaged),
		contextLen:           64,
	}
	source := KVSnapshotBlockSource{
		TokenCount:   4,
		PrefixTokens: 4,
		BlockCount:   2,
		Load: func(_ context.Context, index int) (KVSnapshotBlock, error) {
			switch index {
			case 0:
				return KVSnapshotBlock{Index: 0, TokenStart: 0, TokenCount: 2, Snapshot: kvSnapshotBlockTestSnapshotForArchitecture("gemma4_text", 0, []int32{1, 2})}, nil
			case 1:
				return KVSnapshotBlock{Index: 1, TokenStart: 2, TokenCount: 2, Snapshot: kvSnapshotBlockTestSnapshotForArchitecture("gemma4_text", 2, []int32{3, 4})}, nil
			default:
				return KVSnapshotBlock{}, core.NewError("unexpected block")
			}
		},
	}

	if err := model.RestorePromptCacheFromKVBlocks(context.Background(), source); err != nil {
		t.Fatalf("RestorePromptCacheFromKVBlocks() error = %v", err)
	}
	defer model.ClearPromptCache()
	if model.promptCache == nil || len(model.promptCache.caches) != 1 {
		t.Fatal("promptCache = nil, want fixed restored block cache")
	}
	if cache := model.promptCache.caches[0]; cache.mode != KVCacheModeFixed || cache.maxSize != 64 {
		t.Fatalf("restored cache mode/max = %q/%d, want fixed/64", cache.mode, cache.maxSize)
	}

	prep, err := model.preparePrompt(context.Background(), []int32{1, 2, 3, 4}, GenerateConfig{MaxTokens: 2})
	if err != nil {
		t.Fatalf("preparePrompt() error = %v", err)
	}
	defer Free(prep.Logits)
	defer FreeCaches(prep.Caches)
	if !prep.CacheHit || prep.CacheHitTokens != 3 || prep.CacheMissTokens != 1 {
		t.Fatalf("preparePrompt cache hit/miss = %v/%d/%d, want hit 3/1", prep.CacheHit, prep.CacheHitTokens, prep.CacheMissTokens)
	}
	restoredCache, ok := prep.Caches[0].(*FixedKVCache)
	if !ok {
		t.Fatalf("preparePrompt cache = %T, want *FixedKVCache", prep.Caches[0])
	}
	if restoredCache.maxSize != 32 {
		t.Fatalf("preparePrompt fixed maxSize = %d, want request-sized 32", restoredCache.maxSize)
	}
	if native.forwardCalls != 1 {
		t.Fatalf("Forward calls = %d, want replay of final prompt token only", native.forwardCalls)
	}
}

func TestPromptCache_RestoreFromKVBlocksReplaysExactHitWithoutLogits_Good(t *testing.T) {
	requireMetalRuntime(t)

	native := &fakePagedModel{numLayers: 1, pageSize: 2}
	model := &Model{
		model:                native,
		modelType:            "fake",
		promptCacheEnabled:   true,
		promptCacheMinTokens: 1,
		cacheMode:            string(KVCacheModePaged),
	}
	source := KVSnapshotBlockSource{
		TokenCount:   4,
		PrefixTokens: 4,
		BlockCount:   2,
		Load: func(_ context.Context, index int) (KVSnapshotBlock, error) {
			switch index {
			case 0:
				return KVSnapshotBlock{Index: 0, TokenStart: 0, TokenCount: 2, Snapshot: kvSnapshotBlockTestSnapshot(0, []int32{1, 2})}, nil
			case 1:
				return KVSnapshotBlock{Index: 1, TokenStart: 2, TokenCount: 2, Snapshot: kvSnapshotBlockTestSnapshot(2, []int32{3, 4})}, nil
			default:
				return KVSnapshotBlock{}, core.NewError("unexpected block")
			}
		},
	}
	if err := model.RestorePromptCacheFromKVBlocks(context.Background(), source); err != nil {
		t.Fatalf("RestorePromptCacheFromKVBlocks() error = %v", err)
	}
	defer model.ClearPromptCache()

	prep, err := model.preparePrompt(context.Background(), []int32{1, 2, 3, 4}, GenerateConfig{MaxTokens: 1})
	if err != nil {
		t.Fatalf("preparePrompt() error = %v", err)
	}
	defer Free(prep.Logits)
	defer FreeCaches(prep.Caches)
	if !prep.CacheHit || prep.CacheHitTokens != 3 || prep.CacheMissTokens != 1 {
		t.Fatalf("preparePrompt cache hit/miss = %v/%d/%d, want hit 3/1", prep.CacheHit, prep.CacheHitTokens, prep.CacheMissTokens)
	}
	if native.forwardCalls != 1 {
		t.Fatalf("Forward calls = %d, want replay of final prompt token", native.forwardCalls)
	}
	if prep.Logits == nil || !prep.Logits.Valid() {
		t.Fatal("preparePrompt logits invalid after replay")
	}
}

func TestPromptCache_RestoreFromKVBlocksPreservesNativeDType_Good(t *testing.T) {
	requireMetalRuntime(t)

	model := &Model{
		model:                &fakePagedModel{numLayers: 1, pageSize: 2},
		modelType:            "fake",
		promptCacheEnabled:   true,
		promptCacheMinTokens: 1,
		cacheMode:            string(KVCacheModePaged),
	}
	source := KVSnapshotBlockSource{
		TokenCount:   2,
		PrefixTokens: 2,
		BlockCount:   1,
		Load: func(_ context.Context, index int) (KVSnapshotBlock, error) {
			if index != 0 {
				return KVSnapshotBlock{}, core.NewError("unexpected block")
			}
			snapshot := kvSnapshotBlockTestSnapshot(0, []int32{1, 2})
			head := &snapshot.Layers[0].Heads[0]
			head.KeyDType = DTypeBFloat16
			head.ValueDType = DTypeBFloat16
			head.KeyBytes = bf16Bytes(head.Key)
			head.ValueBytes = bf16Bytes(head.Value)
			return KVSnapshotBlock{Index: 0, TokenStart: 0, TokenCount: 2, Snapshot: snapshot}, nil
		},
	}

	if err := model.RestorePromptCacheFromKVBlocks(context.Background(), source); err != nil {
		t.Fatalf("RestorePromptCacheFromKVBlocks() error = %v", err)
	}
	defer model.ClearPromptCache()
	cache := model.promptCache.caches[0]
	if cache.mode != KVCacheModePaged || len(cache.kPages) != 1 || cache.kPages[0].Dtype() != DTypeBFloat16 {
		t.Fatalf("restored cache mode/pages/dtype = %q/%d/%v, want paged bf16", cache.mode, len(cache.kPages), cache.kPages[0].Dtype())
	}
}

func TestPromptCache_RestorePagedCacheKeepsStorageDType_Good(t *testing.T) {
	requireMetalRuntime(t)

	cache := NewPagedKVCacheWithDType(8, 2, DTypeBFloat16)
	defer cache.Reset()
	k, v := makeKV(2)
	defer Free(k, v)
	state := cache.UpdateBorrowedPages(k, v, 2)
	state.Free()

	snapshot, ok, err := snapshotPagedCache(cache, 2, 2)
	if err != nil {
		t.Fatalf("snapshotPagedCache() error = %v", err)
	}
	if !ok {
		t.Fatal("snapshotPagedCache() ok = false")
	}
	defer freeCacheSnapshot(snapshot)

	restored, err := RestorePromptCachesWithRequestFixedSize([]cacheSnapshot{snapshot}, 2, 0)
	if err != nil {
		t.Fatalf("RestorePromptCachesWithRequestFixedSize() error = %v", err)
	}
	defer FreeCaches(restored)
	paged, ok := restored[0].(*PagedKVCache)
	if !ok {
		t.Fatalf("restored cache = %T, want *PagedKVCache", restored[0])
	}
	if !paged.hasStorageDType || paged.storageDType != DTypeBFloat16 {
		t.Fatalf("restored storage dtype = %v/%v, want bf16 enabled", paged.hasStorageDType, paged.storageDType)
	}

	kNext, vNext := makeKV(1)
	defer Free(kNext, vNext)
	next := paged.UpdateBorrowedPages(kNext, vNext, 1)
	defer next.Free()
	for i, page := range next.Keys {
		if page.Dtype() != DTypeBFloat16 || next.Values[i].Dtype() != DTypeBFloat16 {
			t.Fatalf("restored page %d dtypes = %v/%v, want bf16/bf16", i, page.Dtype(), next.Values[i].Dtype())
		}
	}
}

func TestPromptCache_RestoreFixedCacheKeepsStorageDType_Good(t *testing.T) {
	requireMetalRuntime(t)

	cache := NewFixedKVCacheWithDType(4, DTypeBFloat16)
	defer cache.Reset()
	k, v := makeKV(2)
	defer Free(k, v)
	stateK, stateV := cache.Update(k, v, 2)
	Free(stateK, stateV)

	snapshot, ok, err := snapshotFixedCache(cache, 2)
	if err != nil {
		t.Fatalf("snapshotFixedCache() error = %v", err)
	}
	if !ok {
		t.Fatal("snapshotFixedCache() ok = false")
	}
	defer freeCacheSnapshot(snapshot)

	restored, arrays, err := restoreFixedCacheSnapshot(snapshot, 2, 2, 0)
	if err != nil {
		t.Fatalf("restoreFixedCacheSnapshot() error = %v", err)
	}
	defer FreeCaches([]Cache{restored})
	if err := Eval(arrays...); err != nil {
		t.Fatalf("Eval restored fixed cache: %v", err)
	}
	fixed, ok := restored.(*FixedKVCache)
	if !ok {
		t.Fatalf("restored cache = %T, want *FixedKVCache", restored)
	}
	if !fixed.hasStorageDType || fixed.storageDType != DTypeBFloat16 {
		t.Fatalf("restored fixed storage dtype = %v/%v, want bf16 enabled", fixed.hasStorageDType, fixed.storageDType)
	}

	kNext, vNext := makeKV(1)
	defer Free(kNext, vNext)
	nextK, nextV := fixed.Update(kNext, vNext, 1)
	defer Free(nextK, nextV)
	if nextK.Dtype() != DTypeBFloat16 || nextV.Dtype() != DTypeBFloat16 {
		t.Fatalf("restored fixed dtypes after append = %v/%v, want bf16/bf16", nextK.Dtype(), nextV.Dtype())
	}
}

func TestPromptCache_RestoreFromKVBlocksAcceptsNativeRawOnly_Good(t *testing.T) {
	requireMetalRuntime(t)

	model := &Model{
		model:                &fakePagedModel{numLayers: 1, pageSize: 2},
		modelType:            "fake",
		promptCacheEnabled:   true,
		promptCacheMinTokens: 1,
		cacheMode:            string(KVCacheModePaged),
	}
	source := KVSnapshotBlockSource{
		TokenCount:   2,
		PrefixTokens: 2,
		BlockCount:   1,
		Load: func(_ context.Context, index int) (KVSnapshotBlock, error) {
			if index != 0 {
				return KVSnapshotBlock{}, core.NewError("unexpected block")
			}
			snapshot := kvSnapshotBlockTestSnapshot(0, []int32{1, 2})
			head := &snapshot.Layers[0].Heads[0]
			head.KeyDType = DTypeBFloat16
			head.ValueDType = DTypeBFloat16
			head.KeyBytes = bf16Bytes(head.Key)
			head.ValueBytes = bf16Bytes(head.Value)
			head.Key = nil
			head.Value = nil
			return KVSnapshotBlock{Index: 0, TokenStart: 0, TokenCount: 2, Snapshot: snapshot}, nil
		},
	}

	if err := model.RestorePromptCacheFromKVBlocks(context.Background(), source); err != nil {
		t.Fatalf("RestorePromptCacheFromKVBlocks(raw-only) error = %v", err)
	}
	defer model.ClearPromptCache()
	cache := model.promptCache.caches[0]
	if cache.mode != KVCacheModePaged || len(cache.kPages) != 1 || cache.kPages[0].Dtype() != DTypeBFloat16 {
		t.Fatalf("restored cache mode/pages/dtype = %q/%d/%v, want paged bf16", cache.mode, len(cache.kPages), cache.kPages[0].Dtype())
	}
}

func TestPromptCache_RestoreFromKVBlocksAcceptsNativeLayerRawOnly_Good(t *testing.T) {
	requireMetalRuntime(t)

	model := &Model{
		model:                &fakePagedModel{numLayers: 1, pageSize: 2},
		modelType:            "fake",
		promptCacheEnabled:   true,
		promptCacheMinTokens: 1,
		cacheMode:            string(KVCacheModePaged),
	}
	source := KVSnapshotBlockSource{
		TokenCount:   2,
		PrefixTokens: 2,
		BlockCount:   1,
		Load: func(_ context.Context, index int) (KVSnapshotBlock, error) {
			if index != 0 {
				return KVSnapshotBlock{}, core.NewError("unexpected block")
			}
			snapshot := kvSnapshotBlockTestSnapshot(0, []int32{1, 2})
			snapshot.NumHeads = 2
			snapshot.HeadDim = 1
			snapshot.Layers[0].KeyDType = DTypeFloat32
			snapshot.Layers[0].KeyBytes = f32Bytes([]float32{1, 2, 3, 4})
			snapshot.Layers[0].KeyShape = []int32{1, 2, 2, 1}
			snapshot.Layers[0].ValueDType = DTypeFloat32
			snapshot.Layers[0].ValueBytes = f32Bytes([]float32{5, 6, 7, 8})
			snapshot.Layers[0].ValueShape = []int32{1, 2, 2, 1}
			snapshot.Layers[0].Heads = make([]KVHeadSnapshot, 2)
			return KVSnapshotBlock{Index: 0, TokenStart: 0, TokenCount: 2, Snapshot: snapshot}, nil
		},
	}

	if err := model.RestorePromptCacheFromKVBlocks(context.Background(), source); err != nil {
		t.Fatalf("RestorePromptCacheFromKVBlocks(layer raw-only) error = %v", err)
	}
	defer model.ClearPromptCache()
	cache := model.promptCache.caches[0]
	if cache.mode != KVCacheModePaged || len(cache.kPages) != 1 || cache.kPages[0].Dtype() != DTypeFloat32 {
		t.Fatalf("restored cache mode/pages/dtype = %q/%d/%v, want paged f32", cache.mode, len(cache.kPages), cache.kPages[0].Dtype())
	}
	keys, values, err := cacheSnapshotFloatArrays(cache)
	if err != nil {
		t.Fatalf("cacheSnapshotFloatArrays() error = %v", err)
	}
	defer Free(keys, values)
	if err := Eval(keys, values); err != nil {
		t.Fatalf("Eval layer raw cache: %v", err)
	}
	if got := keys.Floats(); !reflect.DeepEqual(got, []float32{1, 2, 3, 4}) {
		t.Fatalf("layer raw keys = %v, want [1 2 3 4]", got)
	}
	if got := values.Floats(); !reflect.DeepEqual(got, []float32{5, 6, 7, 8}) {
		t.Fatalf("layer raw values = %v, want [5 6 7 8]", got)
	}
}

func TestPromptCache_RestoreFromKVBlocksTransfersPagedPages_Good(t *testing.T) {
	requireMetalRuntime(t)

	model := &Model{
		model:                &fakePagedModel{numLayers: 1, pageSize: 4},
		modelType:            "fake",
		promptCacheEnabled:   true,
		promptCacheMinTokens: 1,
	}
	source := KVSnapshotBlockSource{
		TokenCount:   4,
		PrefixTokens: 4,
		BlockCount:   2,
		Load: func(_ context.Context, index int) (KVSnapshotBlock, error) {
			if index < 0 || index > 1 {
				return KVSnapshotBlock{}, core.NewError("unexpected block")
			}
			tokens := []int32{int32(index*2 + 1), int32(index*2 + 2)}
			snapshot := kvSnapshotBlockTestSnapshot(index*2, tokens)
			return KVSnapshotBlock{Index: index, TokenStart: index * 2, TokenCount: 2, Snapshot: snapshot}, nil
		},
	}

	if err := model.RestorePromptCacheFromKVBlocks(context.Background(), source); err != nil {
		t.Fatalf("RestorePromptCacheFromKVBlocks() error = %v", err)
	}
	defer model.ClearPromptCache()
	cache := model.promptCache.caches[0]
	if cache.mode != KVCacheModePaged || len(cache.kPages) != 2 {
		t.Fatalf("restored cache mode/pages = %q/%d, want paged transferred pages", cache.mode, len(cache.kPages))
	}
	if got := PagedArrayLen(cache.kPages[0]); got != 2 {
		t.Fatalf("first transferred page length = %d, want 2", got)
	}
	keys, values, err := cacheSnapshotFloatArrays(cache)
	if err != nil {
		t.Fatalf("cacheSnapshotFloatArrays() error = %v", err)
	}
	defer Free(keys, values)
	if err := Eval(keys, values); err != nil {
		t.Fatalf("Eval transferred cache: %v", err)
	}
	if got := keys.Floats(); !reflect.DeepEqual(got, []float32{1, 2, 3, 4}) {
		t.Fatalf("transferred keys = %v, want [1 2 3 4]", got)
	}
	if got := values.Floats(); !reflect.DeepEqual(got, []float32{1, 2, 3, 4}) {
		t.Fatalf("transferred values = %v, want [1 2 3 4]", got)
	}
}

func TestPromptCache_RestoreFromKVBlocksZeroCopyPagedRestore_Good(t *testing.T) {
	requireMetalRuntime(t)

	model := &Model{
		model:                &fakePagedModel{numLayers: 1, pageSize: 4},
		modelType:            "fake",
		promptCacheEnabled:   true,
		promptCacheMinTokens: 1,
	}
	source := KVSnapshotBlockSource{
		TokenCount:   4,
		PrefixTokens: 4,
		BlockCount:   2,
		Load: func(_ context.Context, index int) (KVSnapshotBlock, error) {
			if index < 0 || index > 1 {
				return KVSnapshotBlock{}, core.NewError("unexpected block")
			}
			tokens := []int32{int32(index*2 + 1), int32(index*2 + 2)}
			snapshot := kvSnapshotBlockTestSnapshot(index*2, tokens)
			return KVSnapshotBlock{Index: index, TokenStart: index * 2, TokenCount: 2, Snapshot: snapshot}, nil
		},
	}

	if err := model.RestorePromptCacheFromKVBlocks(context.Background(), source); err != nil {
		t.Fatalf("RestorePromptCacheFromKVBlocks() error = %v", err)
	}
	defer model.ClearPromptCache()
	cache := model.promptCache.caches[0]
	if cache.mode != KVCacheModePaged || len(cache.kPages) != 2 {
		t.Fatalf("restored cache mode/pages = %q/%d, want zero-copy paged block pages", cache.mode, len(cache.kPages))
	}
	if got := PagedArrayLen(cache.kPages[0]); got != 2 {
		t.Fatalf("first restored page length = %d, want block length 2", got)
	}
	keys, values, err := cacheSnapshotFloatArrays(cache)
	if err != nil {
		t.Fatalf("cacheSnapshotFloatArrays() error = %v", err)
	}
	defer Free(keys, values)
	if err := Eval(keys, values); err != nil {
		t.Fatalf("Eval zero-copy paged cache: %v", err)
	}
	if got := keys.Floats(); !reflect.DeepEqual(got, []float32{1, 2, 3, 4}) {
		t.Fatalf("zero-copy keys = %v, want [1 2 3 4]", got)
	}
	if got := values.Floats(); !reflect.DeepEqual(got, []float32{1, 2, 3, 4}) {
		t.Fatalf("zero-copy values = %v, want [1 2 3 4]", got)
	}
}

func TestPromptCache_RestoreFromKVBlocksSkipsDuplicateCacheIndexPerBlock_Good(t *testing.T) {
	requireMetalRuntime(t)

	model := &Model{
		model:                &fakePagedModel{numLayers: 1, pageSize: 4},
		modelType:            "fake",
		promptCacheEnabled:   true,
		promptCacheMinTokens: 1,
	}
	source := KVSnapshotBlockSource{
		TokenCount:   4,
		PrefixTokens: 4,
		BlockCount:   2,
		Load: func(_ context.Context, index int) (KVSnapshotBlock, error) {
			if index < 0 || index > 1 {
				return KVSnapshotBlock{}, core.NewError("unexpected block")
			}
			tokens := []int32{int32(index*2 + 1), int32(index*2 + 2)}
			snapshot := kvSnapshotBlockTestSnapshot(index*2, tokens)
			duplicate := snapshot.Layers[0]
			duplicate.Layer = 1
			duplicate.CacheIndex = 0
			duplicate.Heads = cloneKVSnapshotHeads(duplicate.Heads)
			snapshot.Layers = append(snapshot.Layers, duplicate)
			return KVSnapshotBlock{Index: index, TokenStart: index * 2, TokenCount: 2, Snapshot: snapshot}, nil
		},
	}

	if err := model.RestorePromptCacheFromKVBlocks(context.Background(), source); err != nil {
		t.Fatalf("RestorePromptCacheFromKVBlocks() error = %v", err)
	}
	defer model.ClearPromptCache()
	cache := model.promptCache.caches[0]
	if cache.length != 4 || cache.offset != 4 {
		t.Fatalf("cache length/offset = %d/%d, want 4/4", cache.length, cache.offset)
	}
	keys, values, err := cacheSnapshotFloatArrays(cache)
	if err != nil {
		t.Fatalf("cacheSnapshotFloatArrays() error = %v", err)
	}
	defer Free(keys, values)
	if err := Eval(keys, values); err != nil {
		t.Fatalf("Eval duplicate cache: %v", err)
	}
	if got := keys.Floats(); !reflect.DeepEqual(got, []float32{1, 2, 3, 4}) {
		t.Fatalf("deduped keys = %v, want [1 2 3 4]", got)
	}
	if got := values.Floats(); !reflect.DeepEqual(got, []float32{1, 2, 3, 4}) {
		t.Fatalf("deduped values = %v, want [1 2 3 4]", got)
	}
}

type fakePagedModel struct {
	numLayers    int
	pageSize     int
	forwardCalls int
}

func (f *fakePagedModel) Forward(_ *Array, _ []Cache) *Array {
	f.forwardCalls++
	return Zeros([]int32{1, 1, 8}, DTypeFloat32)
}
func (f *fakePagedModel) ForwardMasked(_ *Array, _ *Array, _ []Cache) *Array { return nil }
func (f *fakePagedModel) NewCache() []Cache {
	caches := make([]Cache, f.numLayers)
	for i := range caches {
		caches[i] = NewPagedKVCache(0, f.pageSize)
	}
	return caches
}
func (f *fakePagedModel) NumLayers() int                      { return f.numLayers }
func (f *fakePagedModel) Tokenizer() *Tokenizer               { return nil }
func (f *fakePagedModel) ModelType() string                   { return "fake" }
func (f *fakePagedModel) ApplyLoRA(_ LoRAConfig) *LoRAAdapter { return nil }

func kvSnapshotBlockTestSnapshot(tokenStart int, tokens []int32) *KVSnapshot {
	return kvSnapshotBlockTestSnapshotForArchitecture("fake", tokenStart, tokens)
}

func kvSnapshotBlockTestSnapshotForArchitecture(architecture string, tokenStart int, tokens []int32) *KVSnapshot {
	values := make([]float32, len(tokens))
	for i := range tokens {
		values[i] = float32(tokenStart + i + 1)
	}
	return &KVSnapshot{
		Version:      KVSnapshotVersion,
		Architecture: architecture,
		Tokens:       append([]int32(nil), tokens...),
		TokenOffset:  tokenStart + len(tokens),
		NumLayers:    1,
		NumHeads:     1,
		SeqLen:       len(tokens),
		HeadDim:      1,
		Layers: []KVLayerSnapshot{{
			Layer:      0,
			CacheIndex: 0,
			Heads: []KVHeadSnapshot{{
				Key:   append([]float32(nil), values...),
				Value: append([]float32(nil), values...),
			}},
		}},
	}
}

func bf16Bytes(values []float32) []byte {
	out := make([]byte, 0, len(values)*2)
	var buf [2]byte
	for _, value := range values {
		binary.LittleEndian.PutUint16(buf[:], uint16(math.Float32bits(value)>>16))
		out = append(out, buf[:]...)
	}
	return out
}

func f32Bytes(values []float32) []byte {
	out := make([]byte, 0, len(values)*4)
	var buf [4]byte
	for _, value := range values {
		binary.LittleEndian.PutUint32(buf[:], math.Float32bits(value))
		out = append(out, buf[:]...)
	}
	return out
}

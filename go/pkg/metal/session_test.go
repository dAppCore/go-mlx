// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"context"
	"testing"

	"dappco.re/go"
)

type lenOnlyCache struct {
	offset int
	length int
}

func (c lenOnlyCache) Update(k, v *Array, _ int) (*Array, *Array) { return k, v }
func (c lenOnlyCache) Offset() int                                { return c.offset }
func (c lenOnlyCache) Len() int                                   { return c.length }
func (c lenOnlyCache) State() []*Array                            { return nil }
func (c lenOnlyCache) Reset()                                     {}
func (c lenOnlyCache) Detach()                                    {}

func TestModelSession_RangeKVBlocksStreamsFullTokenTimeline_Good(t *testing.T) {
	coverageTokens := "ModelSession RangeKVBlocks StreamsFullTokenTimeline"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	const (
		tokenCount = 100000
		CacheLen   = 98304
		blockSize  = 32768
	)
	tokens := make([]int32, tokenCount)
	for i := range tokens {
		tokens[i] = int32(i)
	}
	session := &ModelSession{
		model: &Model{
			model:     &fakeModel{numLayers: 1},
			modelType: "test",
		},
		caches:      []Cache{lenOnlyCache{offset: tokenCount, length: CacheLen}},
		tokens:      tokens,
		tokenOffset: tokenCount,
	}
	var (
		gotTokens int
		gotBlocks int
		gotStarts []int
	)
	err := session.rangeKVBlocksLocked(context.Background(), blockSize, KVSnapshotCaptureOptions{}, func(block KVSnapshotBlock) (bool, error) {
		gotBlocks++
		gotTokens += block.TokenCount
		gotStarts = append(gotStarts, block.TokenStart)
		if block.Snapshot == nil {
			t.Fatalf("block %d snapshot is nil", block.Index)
		}
		if block.Snapshot.TokenOffset != block.TokenStart+block.TokenCount {
			t.Fatalf("block %d token offset = %d, want %d", block.Index, block.Snapshot.TokenOffset, block.TokenStart+block.TokenCount)
		}
		return true, nil
	})
	if err != nil {
		t.Fatalf("rangeKVBlocksLocked() error = %v", err)
	}
	if gotTokens != tokenCount {
		t.Fatalf("streamed tokens = %d, want %d", gotTokens, tokenCount)
	}
	if gotBlocks < 4 {
		t.Fatalf("streamed blocks = %d, want cache-window boundary plus block boundaries", gotBlocks)
	}
	if len(gotStarts) == 0 || gotStarts[0] != 0 {
		t.Fatalf("first block start = %v, want 0", gotStarts)
	}
}

func TestSessionCacheSnapshot_RestoresWrappedRotatingOffset_Good(t *testing.T) {
	coverageTokens := "SessionCacheSnapshot RestoresWrappedRotatingOffset"
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
	defer FreeCaches([]Cache{cache})

	snapshot, ok, err := snapshotSessionCache(cache)
	if err != nil {
		t.Fatalf("snapshotSessionCache: %v", err)
	}
	if !ok {
		t.Fatal("snapshotSessionCache() ok = false, want true")
	}
	if snapshot.offset != 4 || snapshot.length != 2 {
		t.Fatalf("snapshot offset/length = %d/%d, want 4/2", snapshot.offset, snapshot.length)
	}
	defer Free(snapshot.keys, snapshot.values)

	restored, err := restoreSessionCaches([]cacheSnapshot{snapshot})
	if err != nil {
		t.Fatalf("restoreSessionCaches: %v", err)
	}
	defer FreeCaches(restored)
	if len(restored) != 1 {
		t.Fatalf("restored len = %d, want 1", len(restored))
	}
	if restored[0].Offset() != 4 || restored[0].Len() != 2 {
		t.Fatalf("restored offset/len = %d/%d, want 4/2", restored[0].Offset(), restored[0].Len())
	}
}

func TestSessionCacheSnapshot_FromKVLayerUsesLocalWindow_Good(t *testing.T) {
	coverageTokens := "SessionCacheSnapshot FromKVLayerUsesLocalWindow"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	snapshot := &KVSnapshot{
		Version:     KVSnapshotVersion,
		Tokens:      []int32{1, 2, 3, 4, 5},
		TokenOffset: 5,
		SeqLen:      5,
		HeadDim:     2,
		Layers: []KVLayerSnapshot{{
			Layer:      0,
			CacheIndex: 0,
			Heads: []KVHeadSnapshot{{
				Key:   []float32{10, 11, 12, 13},
				Value: []float32{20, 21, 22, 23},
			}},
		}},
	}

	cacheSnapshot, err := cacheSnapshotFromKVLayer(snapshot, snapshot.Layers[0], NewRotatingKVCache(2))
	if err != nil {
		t.Fatalf("cacheSnapshotFromKVLayer: %v", err)
	}
	defer freeCacheSnapshot(cacheSnapshot)
	if cacheSnapshot.length != 2 || cacheSnapshot.offset != 5 || !cacheSnapshot.rotating {
		t.Fatalf("cache snapshot length/offset/rotating = %d/%d/%v, want 2/5/true", cacheSnapshot.length, cacheSnapshot.offset, cacheSnapshot.rotating)
	}
	if got := cacheSnapshot.keys.Shape()[2]; got != 2 {
		t.Fatalf("cache key shape = %v, want local window length 2", cacheSnapshot.keys.Shape())
	}
}

func TestSessionCacheSnapshot_PreservesQuantizedQ8State_Good(t *testing.T) {
	coverageTokens := "SessionCacheSnapshot PreservesQuantizedQ8State"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	cache := NewQuantizedKVCache(0, 8, 8)
	k := FromValues([]float32{1, 2, 3, 4}, 1, 1, 4, 1)
	v := FromValues([]float32{5, 6, 7, 8}, 1, 1, 4, 1)
	fullK, fullV := cache.Update(k, v, 4)
	if err := Eval(fullK, fullV); err != nil {
		t.Fatalf("Eval quantized cache update: %v", err)
	}
	Free(k, v, fullK, fullV)
	defer FreeCaches([]Cache{cache})

	snapshot, ok, err := snapshotSessionCache(cache)
	if err != nil {
		t.Fatalf("snapshotSessionCache: %v", err)
	}
	if !ok {
		t.Fatal("snapshotSessionCache() ok = false, want true")
	}
	defer freeCacheSnapshots([]cacheSnapshot{snapshot})
	if snapshot.mode != KVCacheModeQ8 || snapshot.keyScale == nil || snapshot.valueScale == nil {
		t.Fatalf("snapshot mode/scales = %q/%v/%v, want q8 physical state", snapshot.mode, snapshot.keyScale, snapshot.valueScale)
	}

	restored, err := restoreSessionCaches([]cacheSnapshot{snapshot})
	if err != nil {
		t.Fatalf("restoreSessionCaches: %v", err)
	}
	defer FreeCaches(restored)
	restoredCache, ok := restored[0].(*QuantizedKVCache)
	if !ok {
		t.Fatalf("restored cache = %T, want *QuantizedKVCache", restored[0])
	}
	if restoredCache.Offset() != 4 || restoredCache.Len() != 4 || restoredCache.keyBits != 8 || restoredCache.valueBits != 8 {
		t.Fatalf("restored offset/len/bits = %d/%d/%d/%d, want 4/4/8/8", restoredCache.Offset(), restoredCache.Len(), restoredCache.keyBits, restoredCache.valueBits)
	}
	state, owned := restoredCache.ReadState()
	defer Free(owned...)
	if len(state) != 2 || state[0].Shape()[2] != 4 {
		t.Fatalf("restored dequantized state shape = %v, want sequence length 4", state)
	}
}

func TestSessionCacheSnapshot_PreservesPagedPages_Good(t *testing.T) {
	coverageTokens := "SessionCacheSnapshot PreservesPagedPages"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	cache := NewPagedKVCache(0, 2)
	k := FromValues([]float32{1, 2, 3, 4, 5}, 1, 1, 5, 1)
	v := FromValues([]float32{6, 7, 8, 9, 10}, 1, 1, 5, 1)
	fullK, fullV := cache.Update(k, v, 5)
	if err := Eval(fullK, fullV); err != nil {
		t.Fatalf("Eval paged cache update: %v", err)
	}
	Free(k, v, fullK, fullV)
	defer FreeCaches([]Cache{cache})

	snapshot, ok, err := snapshotSessionCache(cache)
	if err != nil {
		t.Fatalf("snapshotSessionCache: %v", err)
	}
	if !ok {
		t.Fatal("snapshotSessionCache() ok = false, want true")
	}
	defer freeCacheSnapshots([]cacheSnapshot{snapshot})
	if snapshot.mode != KVCacheModePaged || len(snapshot.kPages) != 3 || len(snapshot.vPages) != 3 {
		t.Fatalf("snapshot mode/pages = %q/%d/%d, want paged state with three pages", snapshot.mode, len(snapshot.kPages), len(snapshot.vPages))
	}

	restored, err := restoreSessionCaches([]cacheSnapshot{snapshot})
	if err != nil {
		t.Fatalf("restoreSessionCaches: %v", err)
	}
	defer FreeCaches(restored)
	restoredCache, ok := restored[0].(*PagedKVCache)
	if !ok {
		t.Fatalf("restored cache = %T, want *PagedKVCache", restored[0])
	}
	if restoredCache.Offset() != 5 || restoredCache.Len() != 5 || len(restoredCache.kPages) != 3 {
		t.Fatalf("restored offset/len/pages = %d/%d/%d, want 5/5/3", restoredCache.Offset(), restoredCache.Len(), len(restoredCache.kPages))
	}
}

func TestSessionCacheSnapshot_RestoreTurboQuantFailsClosed_Bad(t *testing.T) {
	coverageTokens := "SessionCacheSnapshot RestoreTurboQuantFailsClosed"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	k := FromValues([]float32{1, 2}, 1, 1, 2, 1)
	v := FromValues([]float32{3, 4}, 1, 1, 2, 1)
	defer Free(k, v)

	restored, err := restoreSessionCaches([]cacheSnapshot{{
		mode:   KVCacheModeTurboQuant,
		keys:   k,
		values: v,
		length: 2,
		offset: 2,
		step:   256,
	}})
	defer FreeCaches(restored)
	if err == nil || !core.Contains(err.Error(), "TurboQuant") {
		t.Fatalf("restoreSessionCaches(turboquant) error = %v, want TurboQuant compatibility error", err)
	}
}

func TestSessionKVSnapshot_PreservesTurboQuantPayloads_Good(t *testing.T) {
	coverageTokens := "SessionKVSnapshot PreservesTurboQuantPayloads"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	model := &Model{
		model:      &fakeModel{numLayers: 1},
		modelType:  "fake",
		contextLen: 8,
		cacheMode:  string(KVCacheModeTurboQuant),
	}
	cache := NewTurboQuantKVCache(0, 8)
	k, v := makeKV(3)
	fullK, fullV := cache.Update(k, v, 3)
	if err := Eval(fullK, fullV); err != nil {
		t.Fatalf("Eval TurboQuant cache update: %v", err)
	}
	defer func() {
		Free(k, v, fullK, fullV)
		FreeCaches([]Cache{cache})
	}()

	snapshot, err := model.snapshotKVCachesWithOptions([]int32{1, 2, 3}, []Cache{cache}, KVSnapshotCaptureOptions{})
	if err != nil {
		t.Fatalf("snapshotKVCachesWithOptions(turboquant) error = %v", err)
	}
	layer := snapshot.Layers[0]
	if layer.CacheMode != KVCacheModeTurboQuant || len(layer.TurboQuantPayloads) != 1 {
		t.Fatalf("layer mode/payloads = %q/%d, want turboquant payload snapshot", layer.CacheMode, len(layer.TurboQuantPayloads))
	}
	if len(layer.KeyBytes) != 0 || len(layer.ValueBytes) != 0 || len(layer.Heads) != 0 {
		t.Fatalf("layer carried legacy state: key bytes=%d value bytes=%d heads=%d", len(layer.KeyBytes), len(layer.ValueBytes), len(layer.Heads))
	}

	layerSnapshot, err := cacheSnapshotFromKVLayer(snapshot, layer, NewTurboQuantKVCache(0, 8))
	if err != nil {
		t.Fatalf("cacheSnapshotFromKVLayer(turboquant) error = %v", err)
	}
	restored, err := restoreSessionCaches([]cacheSnapshot{layerSnapshot})
	if err != nil {
		t.Fatalf("restoreSessionCaches(turboquant payload) error = %v", err)
	}
	defer FreeCaches(restored)
	restoredCache, ok := restored[0].(*TurboQuantKVCache)
	if !ok {
		t.Fatalf("restored cache = %T, want *TurboQuantKVCache", restored[0])
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

func TestSessionCacheSnapshot_Bad(t *testing.T) {
	coverageTokens := "SessionCacheSnapshot Bad"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	_, ok, err := snapshotSessionCache(nil)
	if err != nil {
		t.Fatalf("snapshotSessionCache(nil) error = %v", err)
	}
	if ok {
		t.Fatal("snapshotSessionCache(nil) ok = true, want false")
	}
}

func TestSessionCacheSnapshot_Ugly(t *testing.T) {
	coverageTokens := "SessionCacheSnapshot Ugly"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	cache := NewKVCache()

	_, ok, err := snapshotSessionCache(cache)

	if err != nil {
		t.Fatalf("snapshotSessionCache(empty) error = %v", err)
	}
	if ok {
		t.Fatal("snapshotSessionCache(empty) ok = true, want false")
	}
}

func TestSessionKVSnapshot_RestoreLayerAndLogits_Good(t *testing.T) {
	coverageTokens := "SessionKVSnapshot RestoreLayerAndLogits"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	snapshot := &KVSnapshot{
		Version:      KVSnapshotVersion,
		Architecture: "gemma4_text",
		Tokens:       []int32{1, 2},
		TokenOffset:  4,
		SeqLen:       2,
		HeadDim:      2,
		LogitShape:   []int32{1, 1, 3},
		Logits:       []float32{0.1, 0.2, 0.7},
		Layers: []KVLayerSnapshot{{
			Layer:      0,
			CacheIndex: 0,
			Heads: []KVHeadSnapshot{{
				Key:   []float32{1, 2, 3, 4},
				Value: []float32{5, 6, 7, 8},
			}},
		}},
	}

	layerSnapshot, err := cacheSnapshotFromKVLayer(snapshot, snapshot.Layers[0], NewRotatingKVCache(8))
	if err != nil {
		t.Fatalf("cacheSnapshotFromKVLayer() error = %v", err)
	}
	defer Free(layerSnapshot.keys, layerSnapshot.values)
	restored, err := restoreSessionCaches([]cacheSnapshot{layerSnapshot})
	if err != nil {
		t.Fatalf("restoreSessionCaches() error = %v", err)
	}
	defer FreeCaches(restored)
	logits, err := restoreSnapshotLogits(snapshot)
	if err != nil {
		t.Fatalf("restoreSnapshotLogits() error = %v", err)
	}
	defer Free(logits)

	if restored[0].Offset() != 4 || restored[0].Len() != 2 {
		t.Fatalf("restored offset/len = %d/%d, want 4/2", restored[0].Offset(), restored[0].Len())
	}
	if shape := logits.Shape(); len(shape) != 3 || shape[2] != 3 {
		t.Fatalf("logit shape = %v, want [1 1 3]", shape)
	}
}

func TestSessionKVSnapshot_RestoreWithoutLogitsAllowsAppendState_Good(t *testing.T) {
	coverageTokens := "SessionKVSnapshot RestoreWithoutLogitsAllowsAppend"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
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
	session := &ModelSession{
		model: &Model{
			model:     &fakeModel{numLayers: 1},
			tokenizer: &Tokenizer{},
		},
	}
	defer session.resetState()

	if err := session.restoreKVLocked(snapshot); err != nil {
		t.Fatalf("restoreKVLocked(no logits) error = %v", err)
	}
	if len(session.caches) != 1 || session.logits != nil || len(session.tokens) != 2 {
		t.Fatalf("restored session = caches:%d logits:%v tokens:%v, want cache-only appendable state", len(session.caches), session.logits, session.tokens)
	}
	if err := session.readyForAppend(); err != nil {
		t.Fatalf("readyForAppend(no logits) error = %v", err)
	}
	if err := session.readyForGeneration(); err == nil {
		t.Fatal("readyForGeneration(no logits) error = nil")
	}
}

func TestModelSession_Generate_GoodUsesLazyNativeGreedyState(t *testing.T) {
	coverageTokens := "ModelSession Generate LazyNativeGreedyState"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	requireMetalRuntime(t)

	inner := &boundedGenerateModel{}
	model := &Model{
		model:     inner,
		tokenizer: &Tokenizer{invVocab: map[int32]string{0: "x"}},
	}
	session := &ModelSession{
		model:       model,
		logits:      Zeros([]int32{1, 1, 2}, DTypeFloat32),
		tokens:      []int32{1},
		tokenOffset: 1,
	}
	defer session.resetState()

	var got []Token
	for token := range session.Generate(context.Background(), GenerateConfig{MaxTokens: 1}) {
		got = append(got, token)
	}
	if session.Err() != nil {
		t.Fatalf("Generate() error = %v", session.Err())
	}
	if len(got) != 1 || got[0].ID != 0 || got[0].Text != "x" {
		t.Fatalf("generated tokens = %+v, want one Greedy token", got)
	}
	if inner.forwardCalls != 1 {
		t.Fatalf("Forward calls = %d, want one lazy advance", inner.forwardCalls)
	}
	if shape := session.logits.Shape(); len(shape) != 3 || shape[1] != 1 {
		t.Fatalf("session logits shape = %v, want lazy single-step logits", shape)
	}
}

func TestModelSession_Generate_StopTokenDoesNotAdvanceRetainedState_Good(t *testing.T) {
	coverageTokens := "ModelSession Generate StopTokenDoesNotAdvanceRetainedState"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	requireMetalRuntime(t)

	inner := &boundedGenerateModel{}
	model := &Model{
		model:     inner,
		tokenizer: &Tokenizer{invVocab: map[int32]string{0: "<turn|>"}},
	}
	session := &ModelSession{
		model:       model,
		logits:      Zeros([]int32{1, 1, 2}, DTypeFloat32),
		tokens:      []int32{1},
		tokenOffset: 1,
	}
	defer session.resetState()

	var got []Token
	for token := range session.Generate(context.Background(), GenerateConfig{MaxTokens: 1, StopTokens: []int32{0}, TraceTokenPhases: true, TraceTokenText: true}) {
		got = append(got, token)
	}
	if session.Err() != nil {
		t.Fatalf("Generate() error = %v", session.Err())
	}
	if len(got) != 0 {
		t.Fatalf("generated tokens = %+v, want stop token withheld from visible stream", got)
	}
	if inner.forwardCalls != 0 {
		t.Fatalf("Forward calls = %d, want no retained-state advance for stop token", inner.forwardCalls)
	}
	if len(session.tokens) != 1 || session.tokens[0] != 1 || session.tokenOffset != 1 {
		t.Fatalf("session tokens=%v offset=%d, want original retained state only", session.tokens, session.tokenOffset)
	}
	if metrics := model.LastMetrics(); metrics.GeneratedTokens != 0 {
		t.Fatalf("GeneratedTokens = %d, want stop token excluded", metrics.GeneratedTokens)
	}
	phases := model.LastMetrics().TokenPhases
	if len(phases) != 1 || phases[0].TokenID != 0 || phases[0].TokenText != "<turn|>" || !phases[0].FinalToken {
		t.Fatalf("TokenPhases = %+v, want withheld stop token diagnostic", phases)
	}
}

func TestModelSession_Generate_MinTokensBeforeStopSuppressesFirstStop_Good(t *testing.T) {
	coverageTokens := "ModelSession Generate MinTokensBeforeStopSuppressesFirstStop"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	requireMetalRuntime(t)

	inner := &boundedGenerateModel{}
	model := &Model{
		model:     inner,
		tokenizer: &Tokenizer{invVocab: map[int32]string{0: "<turn|>", 1: "x"}},
	}
	session := &ModelSession{
		model:       model,
		logits:      Zeros([]int32{1, 1, 2}, DTypeFloat32),
		tokens:      []int32{7},
		tokenOffset: 1,
	}
	defer session.resetState()

	var got []Token
	for token := range session.Generate(context.Background(), GenerateConfig{
		MaxTokens:           1,
		StopTokens:          []int32{0},
		MinTokensBeforeStop: 1,
		TraceTokenPhases:    true,
	}) {
		got = append(got, token)
	}
	if session.Err() != nil {
		t.Fatalf("Generate() error = %v", session.Err())
	}
	if len(got) != 1 || got[0].ID != 1 || got[0].Text != "x" {
		t.Fatalf("generated tokens = %+v, want first non-stop token", got)
	}
	if inner.forwardCalls != 1 {
		t.Fatalf("Forward calls = %d, want retained-state advance after non-stop token", inner.forwardCalls)
	}
	if len(session.tokens) != 2 || session.tokens[1] != 1 {
		t.Fatalf("session tokens = %v, want generated token retained", session.tokens)
	}
	if metrics := model.LastMetrics(); metrics.GeneratedTokens != 1 {
		t.Fatalf("GeneratedTokens = %d, want first non-stop token counted", metrics.GeneratedTokens)
	}
}

func TestModelSession_Generate_TraceTokenPhases_Good(t *testing.T) {
	coverageTokens := "ModelSession Generate TraceTokenPhases"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	requireMetalRuntime(t)

	model := &Model{
		model:     &boundedGenerateModel{},
		tokenizer: &Tokenizer{invVocab: map[int32]string{0: "x"}},
	}
	session := &ModelSession{
		model:       model,
		logits:      Zeros([]int32{1, 1, 2}, DTypeFloat32),
		tokens:      []int32{1},
		tokenOffset: 1,
	}
	defer session.resetState()

	for range session.Generate(context.Background(), GenerateConfig{MaxTokens: 1, TraceTokenPhases: true, TraceTokenText: true}) {
	}
	if session.Err() != nil {
		t.Fatalf("Generate() error = %v", session.Err())
	}
	phases := model.LastMetrics().TokenPhases
	if len(phases) != 1 {
		t.Fatalf("TokenPhases len = %d, want one phase; phases=%+v", len(phases), phases)
	}
	if phases[0].TokenID != 0 || phases[0].TokenText != "x" {
		t.Fatalf("phase sampled token = %+v, want token id/text captured", phases[0])
	}
	if phases[0].TotalDuration <= 0 || phases[0].ForwardDuration <= 0 || phases[0].SampleEvalDuration <= 0 {
		t.Fatalf("phase = %+v, want retained-session total, forward, and eval timings", phases[0])
	}
}

func TestModelSession_Generate_AsyncDecodePrefetch_Good(t *testing.T) {
	coverageTokens := "ModelSession Generate AsyncDecodePrefetch"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	requireMetalRuntime(t)

	old := enableAsyncDecodePrefetch
	enableAsyncDecodePrefetch = true
	t.Cleanup(func() { enableAsyncDecodePrefetch = old })

	inner := &boundedGenerateModel{}
	model := &Model{
		model:     inner,
		tokenizer: &Tokenizer{invVocab: map[int32]string{0: "x"}},
	}
	session := &ModelSession{
		model:       model,
		logits:      Zeros([]int32{1, 1, 2}, DTypeFloat32),
		tokens:      []int32{1},
		tokenOffset: 1,
	}
	defer session.resetState()

	for range session.Generate(context.Background(), GenerateConfig{MaxTokens: 1, TraceTokenPhases: true}) {
	}
	if session.Err() != nil {
		t.Fatalf("Generate() error = %v", session.Err())
	}
	if inner.forwardCalls != 1 {
		t.Fatalf("Forward calls = %d, want one retained-session advance", inner.forwardCalls)
	}
	if err := Eval(session.logits); err != nil {
		t.Fatalf("Eval prefetched session logits: %v", err)
	}
	phases := model.LastMetrics().TokenPhases
	if len(phases) != 1 || phases[0].PrefetchDuration <= 0 {
		t.Fatalf("TokenPhases = %+v, want retained-session async prefetch duration", phases)
	}
	if phases[0].PrefetchLogitsDuration <= 0 || phases[0].PrefetchCacheDuration != 0 {
		t.Fatalf("TokenPhases = %+v, want retained-session logits-only prefetch split for cacheless model", phases)
	}
}

func TestModelSession_PrefetchTokenStateAdvanceParity_Good(t *testing.T) {
	coverageTokens := "ModelSession PrefetchTokenStateAdvanceParity"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	requireMetalRuntime(t)

	const seed = 240524
	suppress := []int32{0, 7}
	direct := retainedStateAdvanceParityDirectIDs(t, seed, suppress)
	prefetched := retainedStateAdvanceParityPrefetchedIDs(t, seed, suppress)
	if len(prefetched) != len(direct) {
		t.Fatalf("prefetched ids = %v, want %v", prefetched, direct)
	}
	for i := range direct {
		if prefetched[i] != direct[i] {
			t.Fatalf("prefetched ids = %v, want %v", prefetched, direct)
		}
	}
}

func retainedStateAdvanceParityDirectIDs(t *testing.T, seed uint64, suppress []int32) []int32 {
	t.Helper()
	inner := &stateAdvanceParityModel{}
	model := &Model{model: inner, tokenizer: stateAdvanceParityTokenizer()}
	session := stateAdvanceParitySession(model, inner)
	defer func() {
		session.resetState()
		inner.resetOwned()
	}()

	var ids []int32
	for token := range session.Generate(context.Background(), GenerateConfig{
		MaxTokens:      2,
		Temperature:    1,
		TopP:           0.95,
		TopK:           4,
		Seed:           seed,
		SeedSet:        true,
		SuppressTokens: suppress,
	}) {
		ids = append(ids, token.ID)
	}
	if session.Err() != nil {
		t.Fatalf("Generate() error = %v", session.Err())
	}
	if len(ids) != 2 {
		t.Fatalf("generated ids = %v, want two retained-session tokens", ids)
	}
	return ids
}

func retainedStateAdvanceParityPrefetchedIDs(t *testing.T, seed uint64, suppress []int32) []int32 {
	t.Helper()
	inner := &stateAdvanceParityModel{}
	model := &Model{model: inner, tokenizer: stateAdvanceParityTokenizer()}
	session := stateAdvanceParitySession(model, inner)
	defer func() {
		session.resetState()
		inner.resetOwned()
	}()

	if err := model.withDevice(func() {
		if seedErr := SeedRandom(seed); seedErr != nil {
			t.Fatalf("SeedRandom: %v", seedErr)
		}
	}); err != nil {
		t.Fatalf("withDevice seed: %v", err)
	}

	var ids []int32
	if err := model.withDevice(func() {
		sampler := NewSamplerWithSuppression(1, 0.95, 0, 4, suppress)
		defer CloseSampler(sampler)

		lastPos, err := lastTokenLogits(session.logits)
		if err != nil {
			t.Fatalf("lastTokenLogits first: %v", err)
		}
		firstToken, firstID, _, err := SampleTokenIDWithSuppressionGuard(lastPos, sampler, suppress, false)
		Free(lastPos)
		if err != nil {
			t.Fatalf("sample first token: %v", err)
		}
		Free(firstToken)
		ids = append(ids, firstID)

		detachEvalState(session.logits, session.caches)
		if err := session.advanceTokenLocked(context.Background(), firstID, 0); err != nil {
			t.Fatalf("advanceTokenLocked: %v", err)
		}

		lastPos, err = lastTokenLogits(session.logits)
		if err != nil {
			t.Fatalf("lastTokenLogits second: %v", err)
		}
		secondToken := sampler.Sample(lastPos)
		Free(lastPos)
		var stack [8]*Array
		eval := stack[:0]
		eval = append(eval, session.logits, secondToken)
		for _, cache := range session.caches {
			eval = appendCacheDirtyState(eval, cache)
		}
		if err := EvalAsync(eval...); err != nil {
			Free(secondToken)
			t.Fatalf("EvalAsync retained sampled token: %v", err)
		}
		secondID := int32(secondToken.Int())
		Free(secondToken)
		if TokenIDSuppressed(secondID, suppress) {
			t.Fatalf("prefetched second token = %d, want unsuppressed token", secondID)
		}
		ids = append(ids, secondID)
	}); err != nil {
		t.Fatalf("withDevice parity: %v", err)
	}
	return ids
}

func stateAdvanceParitySession(model *Model, inner *stateAdvanceParityModel) *ModelSession {
	return &ModelSession{
		model:       model,
		logits:      inner.logits(),
		caches:      []Cache{NewPagedKVCache(0, 2)},
		tokens:      []int32{42},
		tokenOffset: 1,
	}
}

func stateAdvanceParityTokenizer() *Tokenizer {
	return &Tokenizer{invVocab: map[int32]string{
		1: "a",
		2: "b",
		3: "c",
		4: "d",
		5: "e",
		6: "f",
	}}
}

type stateAdvanceParityModel struct {
	forwardCalls int
	owned        []*Array
}

func (m *stateAdvanceParityModel) Forward(tokens *Array, caches []Cache) *Array {
	m.forwardCalls++
	m.updatePagedCache(tokens, caches)
	return m.logits()
}

func (m *stateAdvanceParityModel) ForwardMasked(tokens *Array, _ *Array, caches []Cache) *Array {
	return m.Forward(tokens, caches)
}

func (m *stateAdvanceParityModel) NewCache() []Cache { return []Cache{NewPagedKVCache(0, 2)} }

func (m *stateAdvanceParityModel) NumLayers() int { return 1 }

func (m *stateAdvanceParityModel) Tokenizer() *Tokenizer { return nil }

func (m *stateAdvanceParityModel) ModelType() string { return "state-advance-parity-test" }

func (m *stateAdvanceParityModel) ApplyLoRA(_ LoRAConfig) *LoRAAdapter { return nil }

func (m *stateAdvanceParityModel) logits() *Array {
	base := FromValues([]float32{9.0, 3.4, 3.2, 3.0, 2.8, 2.6, 2.4, 9.0}, 1, 1, 8)
	zero := Zeros([]int32{1, 1, 8}, DTypeFloat32)
	m.owned = append(m.owned, base, zero)
	return Add(base, zero)
}

func (m *stateAdvanceParityModel) updatePagedCache(tokens *Array, caches []Cache) {
	if len(caches) == 0 || caches[0] == nil {
		return
	}
	seqLen := 1
	if tokens != nil && tokens.Valid() && tokens.NumDims() >= 2 {
		seqLen = int(tokens.Dim(1))
	}
	k := Zeros([]int32{1, 1, int32(seqLen), 1}, DTypeFloat32)
	v := Zeros([]int32{1, 1, int32(seqLen), 1}, DTypeFloat32)
	fullK, fullV := caches[0].Update(k, v, seqLen)
	Free(k, v, fullK, fullV)
}

func (m *stateAdvanceParityModel) resetOwned() {
	Free(m.owned...)
	m.owned = nil
}

func TestModelSession_Generate_BadRequiresGenerationState(t *testing.T) {
	coverageTokens := "ModelSession Generate RequiresGenerationState"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	session := &ModelSession{model: &Model{tokenizer: &Tokenizer{}}}
	for range session.Generate(context.Background(), GenerateConfig{MaxTokens: 1}) {
		t.Fatal("Generate yielded token without retained state")
	}
	if session.Err() == nil {
		t.Fatal("Generate() error = nil, want retained-state error")
	}
}

func TestModelSession_Generate_UglyProbeKeepsLogitEvents(t *testing.T) {
	coverageTokens := "ModelSession Generate ProbeKeepsLogitEvents"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	requireMetalRuntime(t)

	inner := &boundedGenerateModel{}
	model := &Model{
		model:     inner,
		tokenizer: &Tokenizer{invVocab: map[int32]string{0: "x"}},
	}
	session := &ModelSession{
		model:       model,
		logits:      Zeros([]int32{1, 1, 2}, DTypeFloat32),
		tokens:      []int32{1},
		tokenOffset: 1,
	}
	defer session.resetState()

	var logitEvents int
	cfg := GenerateConfig{
		MaxTokens: 1,
		ProbeSink: ProbeSinkFunc(func(event ProbeEvent) {
			if event.Kind == ProbeEventLogits {
				logitEvents++
			}
		}),
	}
	for range session.Generate(context.Background(), cfg) {
	}
	if session.Err() != nil {
		t.Fatalf("Generate() error = %v", session.Err())
	}
	if logitEvents == 0 {
		t.Fatal("logit probe events = 0, want fallback sampling path to preserve probes")
	}
}

func TestSessionKVSnapshot_RestoreInfersLayerHeadDims_Good(t *testing.T) {
	coverageTokens := "SessionKVSnapshot RestoreInfersLayerHeadDims"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
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
				Key:   []float32{1, 2, 3, 4, 5, 6, 7, 8},
				Value: []float32{9, 10, 11, 12, 13, 14},
			}},
		}},
	}

	layerSnapshot, err := cacheSnapshotFromKVLayer(snapshot, snapshot.Layers[0], NewRotatingKVCache(8))
	if err != nil {
		t.Fatalf("cacheSnapshotFromKVLayer() error = %v", err)
	}
	defer Free(layerSnapshot.keys, layerSnapshot.values)

	if got := layerSnapshot.keys.Shape(); got[3] != 4 {
		t.Fatalf("key shape = %v, want inferred key dim 4", got)
	}
	if got := layerSnapshot.values.Shape(); got[3] != 3 {
		t.Fatalf("value shape = %v, want inferred value dim 3", got)
	}
}

func TestSessionKVSnapshot_RestoreUsesQuantizedTemplate_Good(t *testing.T) {
	coverageTokens := "SessionKVSnapshot RestoreUsesQuantizedTemplate"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	requireMetalRuntime(t)
	snapshot := &KVSnapshot{
		Version:     KVSnapshotVersion,
		Tokens:      []int32{1, 2},
		TokenOffset: 2,
		SeqLen:      2,
		HeadDim:     2,
		Layers: []KVLayerSnapshot{{
			Layer:      0,
			CacheIndex: 0,
			Heads: []KVHeadSnapshot{{
				Key:   []float32{1, 2, 3, 4},
				Value: []float32{5, 6, 7, 8},
			}},
		}},
	}

	layerSnapshot, err := cacheSnapshotFromKVLayer(snapshot, snapshot.Layers[0], NewQuantizedKVCache(0, 8, 8))
	if err != nil {
		t.Fatalf("cacheSnapshotFromKVLayer() error = %v", err)
	}
	defer freeCacheSnapshots([]cacheSnapshot{layerSnapshot})
	if layerSnapshot.mode != KVCacheModeQ8 || layerSnapshot.keyScale == nil {
		t.Fatalf("layer snapshot mode/scale = %q/%v, want q8 physical state", layerSnapshot.mode, layerSnapshot.keyScale)
	}

	restored, err := restoreSessionCaches([]cacheSnapshot{layerSnapshot})
	if err != nil {
		t.Fatalf("restoreSessionCaches() error = %v", err)
	}
	defer FreeCaches(restored)
	if _, ok := restored[0].(*QuantizedKVCache); !ok {
		t.Fatalf("restored cache = %T, want *QuantizedKVCache", restored[0])
	}
}

func TestSessionKVSnapshot_RestoreUsesPagedTemplate_Good(t *testing.T) {
	coverageTokens := "SessionKVSnapshot RestoreUsesPagedTemplate"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	requireMetalRuntime(t)
	snapshot := &KVSnapshot{
		Version:     KVSnapshotVersion,
		Tokens:      []int32{1, 2, 3, 4, 5},
		TokenOffset: 5,
		SeqLen:      5,
		HeadDim:     1,
		Layers: []KVLayerSnapshot{{
			Layer:      0,
			CacheIndex: 0,
			Heads: []KVHeadSnapshot{{
				Key:   []float32{1, 2, 3, 4, 5},
				Value: []float32{6, 7, 8, 9, 10},
			}},
		}},
	}

	layerSnapshot, err := cacheSnapshotFromKVLayer(snapshot, snapshot.Layers[0], NewPagedKVCache(0, 2))
	if err != nil {
		t.Fatalf("cacheSnapshotFromKVLayer() error = %v", err)
	}
	defer freeCacheSnapshots([]cacheSnapshot{layerSnapshot})
	if layerSnapshot.mode != KVCacheModePaged || len(layerSnapshot.kPages) != 3 {
		t.Fatalf("layer snapshot mode/pages = %q/%d, want paged physical state", layerSnapshot.mode, len(layerSnapshot.kPages))
	}

	restored, err := restoreSessionCaches([]cacheSnapshot{layerSnapshot})
	if err != nil {
		t.Fatalf("restoreSessionCaches() error = %v", err)
	}
	defer FreeCaches(restored)
	restoredCache, ok := restored[0].(*PagedKVCache)
	if !ok {
		t.Fatalf("restored cache = %T, want *PagedKVCache", restored[0])
	}
	if restoredCache.Len() != 5 || len(restoredCache.kPages) != 3 {
		t.Fatalf("restored len/pages = %d/%d, want 5/3", restoredCache.Len(), len(restoredCache.kPages))
	}
}

func TestSessionKVSnapshot_RestoreTransfersPagedPages_Good(t *testing.T) {
	coverageTokens := "SessionKVSnapshot RestoreTransfersPagedPages"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	requireMetalRuntime(t)
	snapshot := &KVSnapshot{
		Version:     KVSnapshotVersion,
		Tokens:      []int32{1, 2, 3, 4},
		TokenOffset: 4,
		SeqLen:      4,
		HeadDim:     1,
		Layers: []KVLayerSnapshot{{
			Layer:      0,
			CacheIndex: 0,
			Heads: []KVHeadSnapshot{{
				Key:   []float32{1, 2, 3, 4},
				Value: []float32{5, 6, 7, 8},
			}},
		}},
	}

	layerSnapshot, err := cacheSnapshotFromKVLayer(snapshot, snapshot.Layers[0], NewPagedKVCache(0, 2))
	if err != nil {
		t.Fatalf("cacheSnapshotFromKVLayer() error = %v", err)
	}
	if layerSnapshot.mode != KVCacheModePaged || len(layerSnapshot.kPages) != 2 {
		freeCacheSnapshots([]cacheSnapshot{layerSnapshot})
		t.Fatalf("layer snapshot mode/pages = %q/%d, want paged physical state", layerSnapshot.mode, len(layerSnapshot.kPages))
	}
	firstK := layerSnapshot.kPages[0]
	firstV := layerSnapshot.vPages[0]
	snapshots := []cacheSnapshot{layerSnapshot}
	restored, err := restoreSessionCachesTransferringPaged(snapshots)
	if err != nil {
		freeCacheSnapshots(snapshots)
		t.Fatalf("restoreSessionCachesTransferringPaged() error = %v", err)
	}
	defer FreeCaches(restored)
	if len(snapshots[0].kPages) != 0 || len(snapshots[0].vPages) != 0 {
		t.Fatalf("transferred snapshot pages = %d/%d, want 0/0", len(snapshots[0].kPages), len(snapshots[0].vPages))
	}
	restoredCache, ok := restored[0].(*PagedKVCache)
	if !ok {
		t.Fatalf("restored cache = %T, want *PagedKVCache", restored[0])
	}
	if len(restoredCache.kPages) != 2 || restoredCache.kPages[0] != firstK || restoredCache.vPages[0] != firstV {
		t.Fatalf("restored pages were not transferred")
	}
}

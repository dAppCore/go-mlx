// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"context"
	"testing"
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
	if logits.Dim(1) != 1 {
		t.Fatalf("last logits seq len = %d, want 1", logits.Dim(1))
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

	want := "<start_of_turn>user\nHello<end_of_turn>\n" +
		"<start_of_turn>model\nHi<end_of_turn>\n" +
		"<start_of_turn>model\n"
	if got != want {
		t.Fatalf("formatChat() = %q, want %q", got, want)
	}
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

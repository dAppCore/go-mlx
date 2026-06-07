// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"testing"
)

func TestMetalEval_AddsValues(t *testing.T) {
	a := FromValues([]float32{1, 2, 3}, 3)
	b := FromValues([]float32{4, 5, 6}, 3)
	c := Add(a, b)

	if err := Eval(c); err != nil {
		t.Fatalf("Eval should succeed: %v", err)
	}

	got := c.Floats()
	want := []float32{5, 7, 9}
	for i := range got {
		if got[i] != want[i] {
			t.Errorf("got[%d] = %f, want %f", i, got[i], want[i])
		}
	}
}

func TestMetal_Eval_NilArray_Good(t *testing.T) {
	// Eval should handle nil arrays gracefully.
	if err := Eval(nil); err != nil {
		t.Fatalf("Eval(nil) should not error: %v", err)
	}
}

func TestMetal_LastError_NoError_Good(t *testing.T) {
	// When no error has occurred, LastError should return nil.
	if err := LastError(); err != nil {
		t.Errorf("LastError should be nil when no error occurred, got: %v", err)
	}
}

func TestMetal_NewCaches_ContextLen_Good(t *testing.T) {
	// When contextLen is set, unbounded KVCaches should become RotatingKVCaches.
	m := &Model{
		model: &fakeModel{numLayers: 4},
	}

	// Without contextLen — should get plain KVCaches.
	caches := m.newCaches()
	for i, c := range caches {
		if _, ok := c.(*KVCache); !ok {
			t.Errorf("cache[%d] without contextLen: got %T, want *KVCache", i, c)
		}
	}

	// With contextLen — should get RotatingKVCaches.
	m.contextLen = 2048
	caches = m.newCaches()
	for i, c := range caches {
		if _, ok := c.(*RotatingKVCache); !ok {
			t.Errorf("cache[%d] with contextLen=2048: got %T, want *RotatingKVCache", i, c)
		}
	}
}

func TestMetal_NewCaches_KVCacheModeQ8_Good(t *testing.T) {
	m := &Model{
		model:      &fakeModel{numLayers: 2},
		contextLen: 2048,
		cacheMode:  string(KVCacheModeQ8),
	}

	caches := m.newCaches()
	for i, c := range caches {
		cache, ok := c.(*QuantizedKVCache)
		if !ok {
			t.Fatalf("cache[%d] = %T, want *QuantizedKVCache", i, c)
		}
		if cache.keyBits != 8 || cache.valueBits != 8 || cache.maxSize != 2048 {
			t.Fatalf("cache[%d] bits/max = %d/%d/%d, want 8/8/2048", i, cache.keyBits, cache.valueBits, cache.maxSize)
		}
	}
}

func TestMetal_NewCaches_KVCacheModeAsymmetric_Good(t *testing.T) {
	m := &Model{
		model:      &fakeModel{numLayers: 1},
		contextLen: 1024,
		cacheMode:  string(KVCacheModeKQ8VQ4),
	}

	caches := m.newCaches()
	cache, ok := caches[0].(*QuantizedKVCache)
	if !ok {
		t.Fatalf("cache[0] = %T, want *QuantizedKVCache", caches[0])
	}
	if cache.keyBits != 8 || cache.valueBits != 4 {
		t.Fatalf("bits = %d/%d, want K@q8,V@q4", cache.keyBits, cache.valueBits)
	}
}

func TestMetal_NewCaches_KVCacheModePaged_Good(t *testing.T) {
	m := &Model{
		model:      &fakeModel{numLayers: 1},
		contextLen: 4096,
		cacheMode:  string(KVCacheModePaged),
	}

	caches := m.newCaches()
	cache, ok := caches[0].(*PagedKVCache)
	if !ok {
		t.Fatalf("cache[0] = %T, want *PagedKVCache", caches[0])
	}
	if cache.maxSize != 4096 || cache.pageSize == 0 {
		t.Fatalf("paged cache max/page = %d/%d, want bounded non-zero page", cache.maxSize, cache.pageSize)
	}
}

func TestMetal_NewCaches_KVCacheModeTurboQuant_Good(t *testing.T) {
	m := &Model{
		model:      &fakeModel{numLayers: 1},
		contextLen: 4096,
		cacheMode:  string(KVCacheModeTurboQuant),
	}

	caches := m.newCaches()
	cache, ok := caches[0].(*TurboQuantKVCache)
	if !ok {
		t.Fatalf("cache[0] = %T, want *TurboQuantKVCache", caches[0])
	}
	if cache.maxSize != 4096 || cache.pageSize == 0 {
		t.Fatalf("turboquant cache max/page = %d/%d, want bounded non-zero page", cache.maxSize, cache.pageSize)
	}
}

func TestMetal_NewCaches_KVCacheModePagedFixedGemma4_Good(t *testing.T) {
	t.Cleanup(SetRuntimeGate(GateFixedSlidingCache, true))

	m := &Model{
		model:                 &fakeModel{numLayers: 1, usesFixedCache: true},
		modelType:             "gemma4",
		contextLen:            4096,
		cacheMode:             string(KVCacheModePaged),
		fixedSlidingCacheSize: 256,
	}

	caches := m.newCaches()
	cache, ok := caches[0].(*FixedKVCache)
	if !ok {
		t.Fatalf("cache[0] = %T, want *FixedKVCache behind Gemma4 fixed-cache env gate", caches[0])
	}
	if cache.maxSize != 256 {
		t.Fatalf("fixed cache max = %d, want 256 from model config", cache.maxSize)
	}
}

func TestMetal_NewCaches_KVCacheModePagedFixedGemma4RuntimeGate_Good(t *testing.T) {
	t.Cleanup(SetRuntimeGate(GateFixedSlidingCache, false))
	t.Cleanup(SetRuntimeGate(GateFixedSlidingCache, true))

	m := &Model{
		model:                 &fakeModel{numLayers: 1, usesFixedCache: true},
		modelType:             "gemma4",
		contextLen:            4096,
		cacheMode:             string(KVCacheModePaged),
		fixedSlidingCacheSize: 256,
	}

	caches := m.newCaches()
	cache, ok := caches[0].(*FixedKVCache)
	if !ok {
		t.Fatalf("cache[0] = %T, want *FixedKVCache behind Gemma4 fixed-cache runtime gate", caches[0])
	}
	if cache.maxSize != 256 {
		t.Fatalf("fixed cache max = %d, want 256 from model config", cache.maxSize)
	}
}

func TestMetal_NewPromptSnapshotCaches_UsesSnapshotSafePhysicalModes_Good(t *testing.T) {
	cases := map[KVCacheMode]any{
		KVCacheModeQ8:     (*QuantizedKVCache)(nil),
		KVCacheModePaged:  (*PagedKVCache)(nil),
		KVCacheModeKQ8VQ4: (*RotatingKVCache)(nil),
	}
	for mode, want := range cases {
		model := &Model{
			model:      &fakeModel{numLayers: 1},
			contextLen: 4096,
			cacheMode:  string(mode),
		}

		caches := model.newPromptSnapshotCaches()
		switch want.(type) {
		case *QuantizedKVCache:
			if _, ok := caches[0].(*QuantizedKVCache); !ok {
				t.Fatalf("mode %q cache[0] = %T, want *QuantizedKVCache", mode, caches[0])
			}
		case *PagedKVCache:
			if _, ok := caches[0].(*PagedKVCache); !ok {
				t.Fatalf("mode %q cache[0] = %T, want *PagedKVCache", mode, caches[0])
			}
		case *RotatingKVCache:
			if _, ok := caches[0].(*RotatingKVCache); !ok {
				t.Fatalf("mode %q cache[0] = %T, want *RotatingKVCache fallback", mode, caches[0])
			}
		}
	}
}

func TestMetal_RuntimeCachesSnapshotSafe_FlagsPhysicalModes_Good(t *testing.T) {
	for _, mode := range []KVCacheMode{KVCacheModeQ8, KVCacheModePaged} {
		m := &Model{cacheMode: string(mode)}
		if !m.runtimeCachesSnapshotSafe() {
			t.Fatalf("mode %q runtimeCachesSnapshotSafe = false, want true", mode)
		}
	}
	if (&Model{cacheMode: string(KVCacheModeKQ8VQ4)}).runtimeCachesSnapshotSafe() {
		t.Fatal("k-q8-v-q4 runtimeCachesSnapshotSafe = true, want false until q4 prefix slicing lands")
	}
	if !(&Model{}).runtimeCachesSnapshotSafe() {
		t.Fatal("default runtimeCachesSnapshotSafe = false, want true")
	}
}

// fakeModel is a minimal InternalModel for testing cache creation. usesFixedCache
// and suppressor opt into the engine cache + prompt capabilities the dispatch
// helpers assert on (FixedSlidingCacheModel / ThoughtChannelSuppressorModel).
type fakeModel struct {
	numLayers      int
	usesFixedCache bool
	suppressor     bool
}

func (f *fakeModel) Forward(_ *Array, _ []Cache) *Array                 { return nil }
func (f *fakeModel) ForwardMasked(_ *Array, _ *Array, _ []Cache) *Array { return nil }
func (f *fakeModel) NewCache() []Cache {
	caches := make([]Cache, f.numLayers)
	for i := range caches {
		caches[i] = NewKVCache()
	}
	return caches
}
func (f *fakeModel) NumLayers() int                      { return f.numLayers }
func (f *fakeModel) Tokenizer() *Tokenizer               { return nil }
func (f *fakeModel) ModelType() string                   { return "fake" }
func (f *fakeModel) ApplyLoRA(_ LoRAConfig) *LoRAAdapter { return nil }
func (f *fakeModel) UsesFixedSlidingCache() bool         { return f.usesFixedCache }
func (f *fakeModel) NeedsThoughtChannelSuppressor() bool { return f.suppressor }

func TestMetal_LoadAllSafetensors_MissingFile_Bad(t *testing.T) {
	_, err := LoadAllSafetensors("/nonexistent/path/model.safetensors")
	if err == nil {
		t.Fatal("LoadAllSafetensors should fail for missing file")
	}
}

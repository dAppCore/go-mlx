// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import "testing"

// The zero-flag fixed-cache regime (#72): a model that declares the
// fixed-sliding cache gets sized FixedKVCaches in the DEFAULT cache mode —
// no -kv-cache paged, no -context — while explicit modes and non-declaring
// models keep their semantics. These tests pin the selection matrix.

// fakeContextModel adds a declared context length to fakeModel.
type fakeContextModel struct {
	fakeModel
	contextLength int
}

func (f *fakeContextModel) FillModelInfo(info *ModelInfo) {
	info.ContextLength = f.contextLength
}

// fakeHybridModel returns gemma4-shaped templates: rotating (window) caches
// on even layers, unbounded KV caches on odd layers.
type fakeHybridModel struct {
	fakeContextModel
	window int
}

func (f *fakeHybridModel) NewCache() []Cache {
	caches := make([]Cache, f.numLayers)
	for i := range caches {
		if i%2 == 0 {
			caches[i] = NewRotatingKVCache(f.window)
		} else {
			caches[i] = NewKVCache()
		}
	}
	return caches
}

func fixedRegimeGatesOn(t *testing.T) {
	t.Helper()
	restore := EngineFeatures{FixedSlidingCache: true, FixedSlidingCacheBound: true}.Apply()
	t.Cleanup(restore)
}

func fixedCapacities(t *testing.T, caches []Cache) []int {
	t.Helper()
	sizes := make([]int, len(caches))
	for i, c := range caches {
		fixed, ok := c.(*FixedKVCache)
		if !ok {
			t.Fatalf("cache %d is %T, want *FixedKVCache", i, c)
		}
		sizes[i] = fixed.maxSize
	}
	return sizes
}

func TestFixedRegime_DefaultMode_ZeroFlag_Good(t *testing.T) {
	fixedRegimeGatesOn(t)
	m := &Model{model: &fakeHybridModel{
		fakeContextModel: fakeContextModel{
			fakeModel:     fakeModel{numLayers: 4, usesFixedCache: true},
			contextLength: 131072,
		},
		window: 512,
	}}
	caches := m.newCachesWithRequestFixedSize(0)
	defer FreeCaches(caches)
	sizes := fixedCapacities(t, caches)
	// Sliding templates clamp to their window; globals carry the zero-flag
	// bound (model context clamped to DefaultFixedCacheBound).
	want := []int{512, DefaultFixedCacheBound, 512, DefaultFixedCacheBound}
	for i := range want {
		if sizes[i] != want[i] {
			t.Fatalf("cache %d capacity = %d, want %d", i, sizes[i], want[i])
		}
	}
}

func TestFixedRegime_DefaultMode_RequestSized_Good(t *testing.T) {
	fixedRegimeGatesOn(t)
	m := &Model{model: &fakeHybridModel{
		fakeContextModel: fakeContextModel{
			fakeModel:     fakeModel{numLayers: 2, usesFixedCache: true},
			contextLength: 131072,
		},
		window: 512,
	}}
	caches := m.newCachesWithRequestFixedSize(4096)
	defer FreeCaches(caches)
	sizes := fixedCapacities(t, caches)
	if sizes[0] != 512 || sizes[1] != 4096 {
		t.Fatalf("capacities = %v, want [512 4096]", sizes)
	}
}

func TestFixedRegime_SmallModelContext_Clamps_Good(t *testing.T) {
	fixedRegimeGatesOn(t)
	m := &Model{model: &fakeContextModel{
		fakeModel:     fakeModel{numLayers: 2, usesFixedCache: true},
		contextLength: 8192,
	}}
	caches := m.newCachesWithRequestFixedSize(0)
	defer FreeCaches(caches)
	sizes := fixedCapacities(t, caches)
	if sizes[0] != 8192 || sizes[1] != 8192 {
		t.Fatalf("capacities = %v, want clamp to the model's 8192 context", sizes)
	}
}

func TestFixedRegime_NotDeclared_Unchanged_Good(t *testing.T) {
	fixedRegimeGatesOn(t)
	m := &Model{model: &fakeModel{numLayers: 2, usesFixedCache: false}}
	caches := m.newCachesWithRequestFixedSize(0)
	defer FreeCaches(caches)
	for i, c := range caches {
		if _, ok := c.(*KVCache); !ok {
			t.Fatalf("cache %d is %T, want plain *KVCache (no regime)", i, c)
		}
	}
}

func TestFixedRegime_GatesOff_Unchanged_Good(t *testing.T) {
	restore := EngineFeatures{}.Apply() // all gates off
	t.Cleanup(restore)
	m := &Model{model: &fakeModel{numLayers: 2, usesFixedCache: true}}
	caches := m.newCachesWithRequestFixedSize(0)
	defer FreeCaches(caches)
	for i, c := range caches {
		if _, ok := c.(*KVCache); !ok {
			t.Fatalf("cache %d is %T, want plain *KVCache (gates off)", i, c)
		}
	}
}

func TestFixedRegime_FullCachePolicy_Declines_Good(t *testing.T) {
	fixedRegimeGatesOn(t)
	m := &Model{
		model:       &fakeModel{numLayers: 2, usesFixedCache: true},
		cachePolicy: "full",
	}
	caches := m.newCachesWithRequestFixedSize(0)
	defer FreeCaches(caches)
	for i, c := range caches {
		if _, ok := c.(*KVCache); !ok {
			t.Fatalf("cache %d is %T, want plain *KVCache (policy full)", i, c)
		}
	}
}

func TestFixedRegime_PagedWithoutContext_KeepsPaged_Good(t *testing.T) {
	fixedRegimeGatesOn(t)
	m := &Model{
		model:     &fakeModel{numLayers: 2, usesFixedCache: true},
		cacheMode: string(KVCacheModePaged),
	}
	caches := m.newCachesWithRequestFixedSize(0)
	defer FreeCaches(caches)
	for i, c := range caches {
		if _, ok := c.(*PagedKVCache); !ok {
			t.Fatalf("cache %d is %T, want *PagedKVCache (explicit paged, no context)", i, c)
		}
	}
}

func TestFixedRegime_PagedWithContext_Fixed_Good(t *testing.T) {
	fixedRegimeGatesOn(t)
	m := &Model{
		model:      &fakeModel{numLayers: 2, usesFixedCache: true},
		cacheMode:  string(KVCacheModePaged),
		contextLen: 16384,
	}
	caches := m.newCachesWithRequestFixedSize(0)
	defer FreeCaches(caches)
	sizes := fixedCapacities(t, caches)
	if sizes[0] != 16384 || sizes[1] != 16384 {
		t.Fatalf("capacities = %v, want the explicit 16384 regime", sizes)
	}
}

func TestFixedRegime_RegimeActive_Matrix_Good(t *testing.T) {
	declared := &fakeModel{numLayers: 1, usesFixedCache: true}
	cases := []struct {
		name string
		m    *Model
		want bool
	}{
		{"default+declared", &Model{model: declared}, true},
		{"default+undeclared", &Model{model: &fakeModel{numLayers: 1}}, false},
		{"default+policyfull", &Model{model: declared, cachePolicy: "full"}, false},
		{"paged+context", &Model{model: declared, cacheMode: "paged", contextLen: 8192}, true},
		{"paged+nocontext", &Model{model: declared, cacheMode: "paged"}, false},
		{"q8", &Model{model: declared, cacheMode: "q8", contextLen: 8192}, false},
	}
	for _, tc := range cases {
		if got := tc.m.fixedCacheRegimeActive(); got != tc.want {
			t.Fatalf("%s: fixedCacheRegimeActive = %v, want %v", tc.name, got, tc.want)
		}
	}
}

// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import "testing"

func TestCacheProfile_Gemma4LocalWindowBounded_Good(t *testing.T) {
	coverageTokens := "CacheProfile Gemma4LocalWindowBounded"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	model := cacheProfileGemma4TestModel(512)
	caches := []Cache{
		&FixedKVCache{maxSize: 512, length: 512, offset: 2048},
		&FixedKVCache{maxSize: 512, length: 512, offset: 2048},
		&FixedKVCache{maxSize: 512, length: 512, offset: 2048},
		&FixedKVCache{maxSize: 512, length: 512, offset: 2048},
		&FixedKVCache{maxSize: 512, length: 512, offset: 2048},
		&FixedKVCache{maxSize: 71040, length: 4000, offset: 4000},
	}

	profile := modelCacheProfile(model, caches)

	if profile == nil {
		t.Fatal("CacheProfile = nil, want populated Gemma 4 topology")
	}
	if profile.LocalCaches != 5 || profile.GlobalCaches != 1 || profile.SharedLayers != 2 {
		t.Fatalf("topology = local:%d global:%d shared:%d, want 5/1/2", profile.LocalCaches, profile.GlobalCaches, profile.SharedLayers)
	}
	if profile.LocalWindowTokens != 512 || profile.MaxLocalTokens != 512 || profile.MaxLocalCapacity != 512 {
		t.Fatalf("local profile = %+v, want window/tokens/capacity capped at 512", profile)
	}
	if profile.MaxGlobalTokens != 4000 || profile.MaxGlobalCapacity != 71040 || profile.MaxProcessedTokens != 4000 {
		t.Fatalf("global profile = %+v, want retained global cache shape", profile)
	}
	if profile.LocalWindowLeaked {
		t.Fatalf("LocalWindowLeaked = true for bounded local caches: %+v", profile)
	}
}

func TestCacheProfile_Gemma4LocalWindowLeak_Ugly(t *testing.T) {
	coverageTokens := "CacheProfile Gemma4LocalWindowLeak"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	model := cacheProfileGemma4TestModel(512)
	caches := []Cache{
		&FixedKVCache{maxSize: 71040, length: 2048, offset: 2048},
		&FixedKVCache{maxSize: 512, length: 512, offset: 2048},
		&FixedKVCache{maxSize: 512, length: 512, offset: 2048},
		&FixedKVCache{maxSize: 512, length: 512, offset: 2048},
		&FixedKVCache{maxSize: 512, length: 512, offset: 2048},
		&FixedKVCache{maxSize: 71040, length: 4000, offset: 4000},
	}

	profile := modelCacheProfile(model, caches)

	if profile == nil || !profile.LocalWindowLeaked {
		t.Fatalf("CacheProfile = %+v, want local-window leak flagged", profile)
	}
	if profile.MaxLocalTokens != 2048 || profile.MaxLocalCapacity != 71040 {
		t.Fatalf("local profile = %+v, want oversized local cache recorded", profile)
	}
}

func TestCacheProfile_GenericCaches_Bad(t *testing.T) {
	coverageTokens := "CacheProfile GenericCaches"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	profile := modelCacheProfile(nil, []Cache{&KVCache{offset: 8}, &RotatingKVCache{maxSize: 4, offset: 10, idx: 4}})

	if profile == nil {
		t.Fatal("CacheProfile = nil, want generic cache profile")
	}
	if profile.TotalCaches != 2 || profile.FullCaches != 1 || profile.RotatingCaches != 1 {
		t.Fatalf("cache counts = %+v, want full + rotating", profile)
	}
	if profile.UnboundedCaches != 1 || profile.MaxCacheTokens != 8 || profile.MaxCacheCapacity != 4 || profile.MaxProcessedTokens != 10 {
		t.Fatalf("cache profile = %+v, want generic cache bounds", profile)
	}
}

func TestCacheProfile_Qwen36HybridRecordsCachelessLayers_Good(t *testing.T) {
	coverageTokens := "CacheProfile Qwen36Hybrid CachelessLayers"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	model := &qwen36StagedModel{
		config: qwen36StagedConfig{
			ModelType:       "qwen3_6",
			NumHiddenLayers: 4,
			LayerTypes:      []string{"linear_attention", "full_attention"},
		},
	}
	caches := model.NewCache()
	defer freeCaches(caches)
	if len(caches) != 2 {
		t.Fatalf("NewCache() length = %d, want 2 full-attention caches", len(caches))
	}
	caches[0] = &KVCache{offset: 128}
	caches[1] = &KVCache{offset: 256}

	profile := modelCacheProfile(model, caches)

	if profile == nil {
		t.Fatal("CacheProfile = nil, want Qwen 3.6 hybrid topology")
	}
	if profile.Architecture != "qwen3_6" || profile.CachelessLayers != 2 || profile.GlobalCaches != 2 || profile.LocalCaches != 0 {
		t.Fatalf("CacheProfile = %+v, want 2 cacheless linear layers and 2 global caches", profile)
	}
	if profile.MaxGlobalTokens != 256 || profile.MaxProcessedTokens != 256 {
		t.Fatalf("CacheProfile = %+v, want max global/processed tokens from full-attention caches", profile)
	}
}

func cacheProfileGemma4TestModel(slidingWindow int32) *Gemma4Model {
	return &Gemma4Model{
		Cfg: &Gemma4TextConfig{
			SlidingWindow:     slidingWindow,
			NumKVSharedLayers: 2,
		},
		Layers: []*Gemma4DecoderLayer{
			{LayerType: "sliding_attention"},
			{LayerType: "sliding_attention"},
			{LayerType: "sliding_attention"},
			{LayerType: "sliding_attention"},
			{LayerType: "sliding_attention"},
			{LayerType: "full_attention"},
			{LayerType: "sliding_attention"},
			{LayerType: "full_attention"},
		},
		modelType: "gemma4_text",
	}
}

var cacheProfileBenchSink *CacheProfile

func BenchmarkCacheProfile_Gemma4FixedTopology(b *testing.B) {
	model := cacheProfileGemma4TestModel(512)
	caches := []Cache{
		&FixedKVCache{maxSize: 512, length: 512, offset: 2048},
		&FixedKVCache{maxSize: 512, length: 512, offset: 2048},
		&FixedKVCache{maxSize: 512, length: 512, offset: 2048},
		&FixedKVCache{maxSize: 512, length: 512, offset: 2048},
		&FixedKVCache{maxSize: 512, length: 512, offset: 2048},
		&FixedKVCache{maxSize: 71040, length: 4000, offset: 4000},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cacheProfileBenchSink = modelCacheProfile(model, caches)
	}
}

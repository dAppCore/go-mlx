// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma4

import (
	"testing"

	"dappco.re/go/mlx/pkg/metal"
)

// These tests pin Gemma4Model.RecordCacheTopology — the architecture-specific
// half of a CacheProfile: how Gemma 4's local/global sliding-window layout maps
// caches to local/global/shared buckets and flags a leaked local window. They
// were relocated here from package metal's cache_profile_test.go by the gemma4
// extraction (the model type now lives here). The metal-side glue —
// modelCacheProfile dispatching to the CacheTopologyRecorder capability and
// running the generic per-cache pass that fills MaxProcessedTokens etc. — stays
// pinned by metal's model_dispatch_test.go (TestModelCacheProfile_*) and
// cache_profile_test.go (generic + Qwen 3.6 paths).

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

// fixed builds a fixed-capacity cache with pre-set offset/length counters
// (maxSize, length, offset), matching the metal-internal FixedKVCache literals
// the original metal test used.
func fixed(maxSize, length, offset int) metal.Cache {
	return metal.NewFixedKVCacheAtOffset(maxSize, offset, length)
}

func gemma4CacheTopology(model *Gemma4Model, caches []metal.Cache) *metal.CacheProfile {
	profile := &metal.CacheProfile{TotalCaches: len(caches)}
	if model != nil {
		profile.Architecture = model.ModelType()
	}
	model.RecordCacheTopology(profile, caches)
	return profile
}

func TestCacheProfile_Gemma4LocalWindowBounded_Good(t *testing.T) {
	coverageTokens := "CacheProfile Gemma4LocalWindowBounded"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	model := cacheProfileGemma4TestModel(512)
	caches := []metal.Cache{
		fixed(512, 512, 2048),
		fixed(512, 512, 2048),
		fixed(512, 512, 2048),
		fixed(512, 512, 2048),
		fixed(512, 512, 2048),
		fixed(71040, 4000, 4000),
	}

	profile := gemma4CacheTopology(model, caches)

	if profile == nil {
		t.Fatal("CacheProfile = nil, want populated Gemma 4 topology")
	}
	if profile.LocalCaches != 5 || profile.GlobalCaches != 1 || profile.SharedLayers != 2 {
		t.Fatalf("topology = local:%d global:%d shared:%d, want 5/1/2", profile.LocalCaches, profile.GlobalCaches, profile.SharedLayers)
	}
	if profile.LocalWindowTokens != 512 || profile.MaxLocalTokens != 512 || profile.MaxLocalCapacity != 512 {
		t.Fatalf("local profile = %+v, want window/tokens/capacity capped at 512", profile)
	}
	if profile.MaxGlobalTokens != 4000 || profile.MaxGlobalCapacity != 71040 {
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
	caches := []metal.Cache{
		fixed(71040, 2048, 2048),
		fixed(512, 512, 2048),
		fixed(512, 512, 2048),
		fixed(512, 512, 2048),
		fixed(512, 512, 2048),
		fixed(71040, 4000, 4000),
	}

	profile := gemma4CacheTopology(model, caches)

	if profile == nil || !profile.LocalWindowLeaked {
		t.Fatalf("CacheProfile = %+v, want local-window leak flagged", profile)
	}
	if profile.MaxLocalTokens != 2048 || profile.MaxLocalCapacity != 71040 {
		t.Fatalf("local profile = %+v, want oversized local cache recorded", profile)
	}
}

var cacheProfileBenchSink *metal.CacheProfile

func BenchmarkCacheProfile_Gemma4FixedTopology(b *testing.B) {
	model := cacheProfileGemma4TestModel(512)
	caches := []metal.Cache{
		fixed(512, 512, 2048),
		fixed(512, 512, 2048),
		fixed(512, 512, 2048),
		fixed(512, 512, 2048),
		fixed(512, 512, 2048),
		fixed(71040, 4000, 4000),
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cacheProfileBenchSink = gemma4CacheTopology(model, caches)
	}
}

// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import "testing"

// Gemma 4-specific cache-topology assertions (local/global/shared/leak) are
// computed by gemma4.Gemma4Model.RecordCacheTopology and live in that package's
// cache_profile_test.go. The metal-side glue — that modelCacheProfile dispatches
// to the CacheTopologyRecorder capability and runs the generic per-cache pass —
// is pinned by model_dispatch_test.go (TestModelCacheProfile_*). These tests
// cover the generic + Qwen 3.6 hybrid paths that stay entirely in package metal.

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
	defer FreeCaches(caches)
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

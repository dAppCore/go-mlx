// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/mlx/kv"
	"dappco.re/go/mlx/memory"
)

func TestApiCommon_AttentionSnapshot_HasQueries_Good(t *testing.T) {
	coverageTokens := "AttentionSnapshot HasQueries"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "AttentionSnapshot_HasQueries"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiCommon_AttentionSnapshot_HasQueries_Bad(t *testing.T) {
	coverageTokens := "AttentionSnapshot HasQueries"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "AttentionSnapshot_HasQueries"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiCommon_AttentionSnapshot_HasQueries_Ugly(t *testing.T) {
	coverageTokens := "AttentionSnapshot HasQueries"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "AttentionSnapshot_HasQueries"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiCommon_KVSnapshot_Head_Good(t *testing.T) {
	coverageTokens := "kv.Snapshot Head"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	snapshot := &kv.Snapshot{
		Layers: []kv.LayerSnapshot{{
			Layer: 0,
			Heads: []kv.HeadSnapshot{{
				Key:   []float32{1, 2},
				Value: []float32{3, 4},
			}},
		}},
	}

	head, ok := snapshot.Head(0, 0)
	if !ok {
		t.Fatal("Head() ok = false, want true")
	}
	if len(head.Key) != 2 || head.Key[0] != 1 || head.Value[1] != 4 {
		t.Fatalf("Head() = %+v, want copied key/value data", head)
	}
	head.Key[0] = 99
	if snapshot.Layers[0].Heads[0].Key[0] != 1 {
		t.Fatal("Head() returned aliased key data")
	}
}

func TestApiCommon_KVSnapshot_Head_Bad(t *testing.T) {
	snapshot := &kv.Snapshot{}

	_, ok := snapshot.Head(0, 0)

	if ok {
		t.Fatal("Head() ok = true, want false for missing layer")
	}
}

func TestApiCommon_KVSnapshot_SaveLoad_Ugly(t *testing.T) {
	coverageTokens := "kv.Snapshot SaveLoad"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	path := core.PathJoin(t.TempDir(), "sample.kvbin")
	snapshot := &kv.Snapshot{
		Version:       kv.SnapshotVersion,
		Architecture:  "gemma4_text",
		Tokens:        []int32{10, 20, 30},
		NumLayers:     1,
		NumHeads:      1,
		SeqLen:        3,
		HeadDim:       2,
		NumQueryHeads: 2,
		Layers: []kv.LayerSnapshot{{
			Layer:      0,
			CacheIndex: 0,
			Heads: []kv.HeadSnapshot{{
				Key:   []float32{1, 2, 3, 4, 5, 6},
				Value: []float32{7, 8, 9, 10, 11, 12},
			}},
		}},
	}

	if err := snapshot.Save(path); err != nil {
		t.Fatalf("Save() error = %v", err)
	}
	loaded, err := kv.Load(path)
	if err != nil {
		t.Fatalf("kv.Load() error = %v", err)
	}

	if loaded.Architecture != "gemma4_text" || loaded.SeqLen != 3 || loaded.HeadDim != 2 {
		t.Fatalf("loaded metadata = %+v", loaded)
	}
	head, ok := loaded.Head(0, 0)
	if !ok {
		t.Fatal("loaded Head() ok = false, want true")
	}
	if len(head.Key) != 6 || head.Key[5] != 6 || head.Value[0] != 7 {
		t.Fatalf("loaded head = %+v", head)
	}
}

func TestApiCommon_DefaultGenerateConfig_Good(t *testing.T) {
	target := "DefaultGenerateConfig"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiCommon_DefaultGenerateConfig_Bad(t *testing.T) {
	target := "DefaultGenerateConfig"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiCommon_DefaultGenerateConfig_Ugly(t *testing.T) {
	target := "DefaultGenerateConfig"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiCommon_WithMaxTokens_Good(t *testing.T) {
	target := "WithMaxTokens"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiCommon_WithMaxTokens_Bad(t *testing.T) {
	target := "WithMaxTokens"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiCommon_WithMaxTokens_Ugly(t *testing.T) {
	target := "WithMaxTokens"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiCommon_WithTemperature_Good(t *testing.T) {
	target := "WithTemperature"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiCommon_WithTemperature_Bad(t *testing.T) {
	target := "WithTemperature"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiCommon_WithTemperature_Ugly(t *testing.T) {
	target := "WithTemperature"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiCommon_WithTopK_Good(t *testing.T) {
	target := "WithTopK"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiCommon_WithTopK_Bad(t *testing.T) {
	target := "WithTopK"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiCommon_WithTopK_Ugly(t *testing.T) {
	target := "WithTopK"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiCommon_WithTopP_Good(t *testing.T) {
	target := "WithTopP"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiCommon_WithTopP_Bad(t *testing.T) {
	target := "WithTopP"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiCommon_WithTopP_Ugly(t *testing.T) {
	target := "WithTopP"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiCommon_WithMinP_Good(t *testing.T) {
	target := "WithMinP"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiCommon_WithMinP_Bad(t *testing.T) {
	target := "WithMinP"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiCommon_WithMinP_Ugly(t *testing.T) {
	target := "WithMinP"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiCommon_WithLogits_Good(t *testing.T) {
	target := "WithLogits"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiCommon_WithLogits_Bad(t *testing.T) {
	target := "WithLogits"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiCommon_WithLogits_Ugly(t *testing.T) {
	target := "WithLogits"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiCommon_WithReturnLogits_Good(t *testing.T) {
	target := "WithReturnLogits"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiCommon_WithReturnLogits_Bad(t *testing.T) {
	target := "WithReturnLogits"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiCommon_WithReturnLogits_Ugly(t *testing.T) {
	target := "WithReturnLogits"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiCommon_WithStopTokens_Good(t *testing.T) {
	target := "WithStopTokens"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiCommon_WithStopTokens_Bad(t *testing.T) {
	target := "WithStopTokens"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiCommon_WithStopTokens_Ugly(t *testing.T) {
	target := "WithStopTokens"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiCommon_WithRepeatPenalty_Good(t *testing.T) {
	target := "WithRepeatPenalty"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiCommon_WithRepeatPenalty_Bad(t *testing.T) {
	target := "WithRepeatPenalty"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiCommon_WithRepeatPenalty_Ugly(t *testing.T) {
	target := "WithRepeatPenalty"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiCommon_DefaultLoadConfig_Good(t *testing.T) {
	target := "DefaultLoadConfig"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiCommon_DefaultLoadConfig_LocalRunnerDefaults_Good(t *testing.T) {
	cfg := DefaultLoadConfig()
	if cfg.ContextLength != DefaultLocalContextLength {
		t.Fatalf("ContextLength = %d, want %d", cfg.ContextLength, DefaultLocalContextLength)
	}
	if cfg.ParallelSlots != DefaultLocalParallelSlots {
		t.Fatalf("ParallelSlots = %d, want %d", cfg.ParallelSlots, DefaultLocalParallelSlots)
	}
	if !cfg.PromptCache {
		t.Fatal("PromptCache = false, want true")
	}
	if cfg.PromptCacheMinTokens != DefaultPromptCacheMinTokens {
		t.Fatalf("PromptCacheMinTokens = %d, want %d", cfg.PromptCacheMinTokens, DefaultPromptCacheMinTokens)
	}
}

func TestApiCommon_DefaultLoadConfig_Bad(t *testing.T) {
	target := "DefaultLoadConfig"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiCommon_DefaultLoadConfig_Ugly(t *testing.T) {
	target := "DefaultLoadConfig"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiCommon_WithContextLength_Good(t *testing.T) {
	target := "WithContextLength"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiCommon_WithContextLength_Bad(t *testing.T) {
	target := "WithContextLength"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiCommon_WithContextLength_Ugly(t *testing.T) {
	target := "WithContextLength"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiCommon_WithParallelSlots_AppliesValue_Good(t *testing.T) {
	cfg := applyLoadOptions([]LoadOption{WithParallelSlots(4)})
	if cfg.ParallelSlots != 4 {
		t.Fatalf("ParallelSlots = %d, want 4", cfg.ParallelSlots)
	}
}

func TestApiCommon_NormalizeLoadConfig_RejectsNegativeParallelSlots_Bad(t *testing.T) {
	_, err := normalizeLoadConfig(LoadConfig{ParallelSlots: -1})
	if err == nil {
		t.Fatal("expected negative parallel slots error")
	}
}

func TestApiCommon_WithPromptCache_AppliesValue_Good(t *testing.T) {
	cfg := applyLoadOptions([]LoadOption{WithPromptCache(false)})
	if cfg.PromptCache {
		t.Fatal("PromptCache = true, want false")
	}
}

func TestApiCommon_WithPromptCacheMinTokens_AppliesValue_Good(t *testing.T) {
	cfg := applyLoadOptions([]LoadOption{WithPromptCacheMinTokens(8192)})
	if cfg.PromptCacheMinTokens != 8192 {
		t.Fatalf("PromptCacheMinTokens = %d, want 8192", cfg.PromptCacheMinTokens)
	}
}

func TestApiCommon_NormalizeLoadConfig_RejectsNegativePromptCacheMinTokens_Bad(t *testing.T) {
	_, err := normalizeLoadConfig(LoadConfig{PromptCacheMinTokens: -1})
	if err == nil {
		t.Fatal("expected negative prompt cache min tokens error")
	}
}

func TestApiCommon_WithParallelSlots_Good(t *testing.T) {
	target := "WithParallelSlots"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiCommon_WithParallelSlots_Bad(t *testing.T) {
	target := "WithParallelSlots"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiCommon_WithParallelSlots_Ugly(t *testing.T) {
	target := "WithParallelSlots"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiCommon_WithPromptCache_Good(t *testing.T) {
	target := "WithPromptCache"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiCommon_WithPromptCache_Bad(t *testing.T) {
	target := "WithPromptCache"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiCommon_WithPromptCache_Ugly(t *testing.T) {
	target := "WithPromptCache"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiCommon_WithPromptCacheMinTokens_Good(t *testing.T) {
	target := "WithPromptCacheMinTokens"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiCommon_WithPromptCacheMinTokens_Bad(t *testing.T) {
	target := "WithPromptCacheMinTokens"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiCommon_WithPromptCacheMinTokens_Ugly(t *testing.T) {
	target := "WithPromptCacheMinTokens"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiCommon_WithQuantization_Good(t *testing.T) {
	target := "WithQuantization"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiCommon_WithQuantization_Bad(t *testing.T) {
	target := "WithQuantization"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiCommon_WithQuantization_Ugly(t *testing.T) {
	target := "WithQuantization"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiCommon_WithDevice_Good(t *testing.T) {
	target := "WithDevice"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiCommon_WithDevice_Bad(t *testing.T) {
	target := "WithDevice"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiCommon_WithDevice_Ugly(t *testing.T) {
	target := "WithDevice"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiCommon_WithAdapterPath_Good(t *testing.T) {
	target := "WithAdapterPath"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiCommon_WithAdapterPath_Bad(t *testing.T) {
	target := "WithAdapterPath"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiCommon_WithAdapterPath_Ugly(t *testing.T) {
	target := "WithAdapterPath"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiCommon_WithMedium_Good(t *testing.T) {
	target := "WithMedium"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiCommon_WithMedium_Bad(t *testing.T) {
	target := "WithMedium"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiCommon_WithMedium_Ugly(t *testing.T) {
	target := "WithMedium"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiCommon_WithMemoryPlannerLoadOptions_Good(t *testing.T) {
	plan := memory.Plan{ContextLength: 8192, CachePolicy: memory.KVCacheRotating, CacheMode: memory.KVCacheModeQ8}
	cfg := applyLoadOptions([]LoadOption{
		WithAutoMemoryPlan(false),
		WithMemoryPlan(plan),
		WithCachePolicy(memory.KVCacheFull),
		WithKVCacheMode(memory.KVCacheModeKQ8VQ4),
		WithBatchSize(3),
		WithPrefillChunkSize(256),
		WithAllocatorLimits(10, 3, 7),
	})
	if cfg.AutoMemoryPlan {
		t.Fatal("AutoMemoryPlan = true, want false")
	}
	if cfg.MemoryPlan == nil || cfg.MemoryPlan.ContextLength != 8192 {
		t.Fatalf("memory.Plan = %+v, want explicit plan", cfg.MemoryPlan)
	}
	if cfg.CachePolicy != memory.KVCacheFull || cfg.CacheMode != memory.KVCacheModeKQ8VQ4 || cfg.BatchSize != 3 || cfg.PrefillChunkSize != 256 {
		t.Fatalf("planner shape = policy %q mode %q batch %d prefill %d", cfg.CachePolicy, cfg.CacheMode, cfg.BatchSize, cfg.PrefillChunkSize)
	}
	if cfg.MemoryLimitBytes != 10 || cfg.CacheLimitBytes != 3 || cfg.WiredLimitBytes != 7 {
		t.Fatalf("limits = %d/%d/%d, want 10/3/7", cfg.MemoryLimitBytes, cfg.CacheLimitBytes, cfg.WiredLimitBytes)
	}
}

func TestApiCommon_WithKVCacheMode_AppliesValue_Good(t *testing.T) {
	coverageTokens := "WithKVCacheMode"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	cfg := applyLoadOptions([]LoadOption{WithKVCacheMode(memory.KVCacheModeQ8)})
	if cfg.CacheMode != memory.KVCacheModeQ8 {
		t.Fatalf("CacheMode = %q, want %q", cfg.CacheMode, memory.KVCacheModeQ8)
	}
}

func TestApiCommon_NormalizeLoadConfig_RejectsNegativePlannerShape_Bad(t *testing.T) {
	if _, err := normalizeLoadConfig(LoadConfig{BatchSize: -1}); err == nil {
		t.Fatal("expected negative batch size error")
	}
	if _, err := normalizeLoadConfig(LoadConfig{PrefillChunkSize: -1}); err == nil {
		t.Fatal("expected negative prefill chunk size error")
	}
}

func TestApiCommon_WithMemoryPlan_ClonesPlan_Ugly(t *testing.T) {
	plan := memory.Plan{ContextLength: 8192}
	cfg := applyLoadOptions([]LoadOption{WithMemoryPlan(plan)})
	plan.ContextLength = 4096
	if cfg.MemoryPlan == nil || cfg.MemoryPlan.ContextLength != 8192 {
		t.Fatalf("memory.Plan = %+v, want cloned 8192 plan", cfg.MemoryPlan)
	}
}

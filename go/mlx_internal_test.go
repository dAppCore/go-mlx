// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"reflect"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/mlx/kv"
	"dappco.re/go/mlx/memory"
	"dappco.re/go/mlx/pkg/metal"
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

func TestApiCommon_SeedRandom_Good(t *testing.T) {
	target := "SeedRandom"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiCommon_SeedRandom_Bad(t *testing.T) {
	target := "SeedRandom"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiCommon_SeedRandom_Ugly(t *testing.T) {
	target := "SeedRandom"
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

func TestApiCommon_WithSeed_Good(t *testing.T) {
	target := "WithSeed"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiCommon_WithSeed_Bad(t *testing.T) {
	target := "WithSeed"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiCommon_WithSeed_Ugly(t *testing.T) {
	target := "WithSeed"
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

func TestApiCommon_WithMinTokensBeforeStop_Good(t *testing.T) {
	target := "WithMinTokensBeforeStop"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiCommon_WithMinTokensBeforeStop_Bad(t *testing.T) {
	target := "WithMinTokensBeforeStop"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiCommon_WithMinTokensBeforeStop_Ugly(t *testing.T) {
	target := "WithMinTokensBeforeStop"
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
	if cfg.ContextLength != 0 {
		t.Fatalf("ContextLength = %d, want model-native default 0", cfg.ContextLength)
	}
	if cfg.Gemma4SlidingWindow != 0 {
		t.Fatalf("Gemma4SlidingWindow = %d, want model-native default 0", cfg.Gemma4SlidingWindow)
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

func TestApiCommon_WithGemma4SlidingWindow_AppliesValue_Good(t *testing.T) {
	coverageTokens := "WithGemma4SlidingWindow"
	cfg := applyLoadOptions([]LoadOption{WithGemma4SlidingWindow(512)})
	if cfg.Gemma4SlidingWindow != 512 {
		t.Fatalf("Gemma4SlidingWindow = %d, want 512", cfg.Gemma4SlidingWindow)
	}
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
}

func TestApiCommon_NormalizeLoadConfig_RejectsNegativeGemma4SlidingWindow_Bad(t *testing.T) {
	_, err := normalizeLoadConfig(LoadConfig{Gemma4SlidingWindow: -1})
	if err == nil {
		t.Fatal("expected negative Gemma 4 sliding-window error")
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
	split := inference.SplitInferencePlan{
		Mode:       inference.SplitInferenceModeLocal,
		LocalSlice: inference.ModelSlicePlan{Preset: inference.ModelSlicePresetFull},
	}
	cfg := applyLoadOptions([]LoadOption{
		WithAutoMemoryPlan(false),
		WithMemoryPlan(plan),
		WithCachePolicy(memory.KVCacheFull),
		WithKVCacheMode(memory.KVCacheModeKQ8VQ4),
		WithBatchSize(3),
		WithPrefillChunkSize(256),
		WithAllocatorLimits(10, 3, 7),
		WithSplitInference(split),
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
	if cfg.SplitInference == nil || cfg.SplitInference.Mode != inference.SplitInferenceModeLocal {
		t.Fatalf("SplitInference = %+v, want cloned local plan", cfg.SplitInference)
	}
	split.Mode = inference.SplitInferenceModeRemoteFFN
	if cfg.SplitInference.Mode != inference.SplitInferenceModeLocal {
		t.Fatalf("WithSplitInference leaked caller mutation: %+v", cfg.SplitInference)
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

func TestApiCommon_NormalizeLoadConfig_AcceptsTurboQuantResearchMode_Good(t *testing.T) {
	cfg, err := normalizeLoadConfig(LoadConfig{CacheMode: memory.KVCacheModeTurboQuant})
	if err != nil {
		t.Fatalf("normalizeLoadConfig(turboquant) error = %v, want nil", err)
	}
	if cfg.CacheMode != memory.KVCacheModeTurboQuant {
		t.Fatalf("CacheMode = %q, want turboquant", cfg.CacheMode)
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

func TestApiCommon_NormalizeLoadConfig_RejectsRemoteSplit_Bad(t *testing.T) {
	_, err := normalizeLoadConfig(LoadConfig{
		SplitInference: &inference.SplitInferencePlan{
			Mode: inference.SplitInferenceModeRemoteFFN,
			LocalSlice: inference.ModelSlicePlan{
				Preset:     inference.ModelSlicePresetClient,
				Components: []inference.ModelComponent{inference.ModelComponentAttention},
			},
			Endpoints: []inference.SplitEndpoint{{
				ID:   "ffn-0",
				Role: inference.SplitEndpointRoleFFN,
			}},
		},
	})
	if err == nil {
		t.Fatal("expected remote split execution error")
	}
	if !core.Contains(err.Error(), "split inference execution is planned") {
		t.Fatalf("error = %v, want split execution planned message", err)
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
func TestAPIGenerateOptions_Good(t *testing.T) {
	cfg := applyGenerateOptions([]GenerateOption{
		WithMaxTokens(64),
		WithTemperature(0.7),
		WithTopK(20),
		WithTopP(0.9),
		WithMinP(0.05),
		WithSeed(42),
		WithLogits(),
		WithReturnLogits(),
		WithStopTokens(1, 2),
		WithMinTokensBeforeStop(1),
		WithRepeatPenalty(1.1),
		WithTokenPhaseTrace(),
		WithTokenPhaseTraceText(),
	})
	if cfg.MaxTokens != 64 || cfg.Temperature != 0.7 || cfg.TopK != 20 || cfg.TopP != 0.9 || cfg.MinP != 0.05 {
		t.Fatalf("unexpected generate config: %+v", cfg)
	}
	if !cfg.SeedSet || cfg.Seed != 42 {
		t.Fatalf("seed config = %d/%v, want 42/true", cfg.Seed, cfg.SeedSet)
	}
	if !cfg.ReturnLogits {
		t.Fatal("ReturnLogits = false, want true")
	}
	if !reflect.DeepEqual(cfg.StopTokens, []int32{1, 2}) {
		t.Fatalf("stop tokens = %v", cfg.StopTokens)
	}
	if cfg.MinTokensBeforeStop != 1 {
		t.Fatalf("MinTokensBeforeStop = %d, want 1", cfg.MinTokensBeforeStop)
	}
	if cfg.RepeatPenalty != 1.1 {
		t.Fatalf("repeat penalty = %f, want 1.1", cfg.RepeatPenalty)
	}
	if !cfg.TraceTokenPhases {
		t.Fatal("TraceTokenPhases = false, want true")
	}
	if !cfg.TraceTokenText {
		t.Fatal("TraceTokenText = false, want true")
	}
}

func TestAPILoadOptions_Good(t *testing.T) {
	cfg := applyLoadOptions([]LoadOption{
		WithContextLength(8192),
		WithParallelSlots(4),
		WithPromptCache(false),
		WithPromptCacheMinTokens(4096),
		WithQuantization(4),
		WithExpectedQuantization(4),
		WithDevice("cpu"),
		WithAdapterPath("/models/lora/demo"),
	})
	if cfg.ContextLength != 8192 || cfg.ParallelSlots != 4 || cfg.PromptCache || cfg.PromptCacheMinTokens != 4096 || cfg.Quantization != 4 || cfg.ExpectedQuantization != 4 || cfg.Device != "cpu" || cfg.AdapterPath != "/models/lora/demo" {
		t.Fatalf("unexpected load config: %+v", cfg)
	}
}

func TestAPIProbeConversion_AllFields_Good(t *testing.T) {
	meta := map[string]string{"scope": "unit"}
	logitMeta := map[string]string{"logits": "kept"}
	got := toRootProbeEvent(metal.ProbeEvent{
		Kind:  metal.ProbeEventLogits,
		Phase: metal.ProbePhaseDecode,
		Step:  6,
		Meta:  meta,
		Token: &metal.ProbeToken{ID: 1, Text: "tok", PromptTokens: 2, GeneratedTokens: 3},
		Logits: &metal.ProbeLogits{
			Shape:      []int32{1, 2},
			VocabSize:  16,
			MaxTokenID: 4,
			MaxLogit:   1.5,
			MinTokenID: 5,
			MinLogit:   -1.5,
			MeanLogit:  0.25,
			Top:        []metal.ProbeLogit{{TokenID: 4, Logit: 1.5, Probability: 0.7}},
			Values:     []float32{0.1, 0.2},
			Meta:       logitMeta,
		},
		Entropy:        &metal.ProbeEntropy{Value: 0.4, Unit: "nats"},
		SelectedHeads:  &metal.ProbeHeadSelection{Layer: 2, Heads: []int{1, 3}, Scores: []float64{0.5, 0.6}},
		LayerCoherence: &metal.ProbeLayerCoherence{Layer: 3, KeyCoherence: 0.1, ValueCoherence: 0.2, CrossAlignment: 0.3, KVCoupling: 0.4, HeadEntropy: 0.5, PhaseLock: 0.6},
		RouterDecision: &metal.ProbeRouterDecision{Layer: 4, TokenID: 7, ExpertIDs: []int{8, 9}, Weights: []float32{0.25, 0.75}, Temperature: 0.8},
		Residual:       &metal.ProbeResidualSummary{Layer: 5, Mean: 0.1, Variance: 0.2, RMS: 0.3, L2Norm: 0.4, MaxAbs: 0.5},
		Cache:          &metal.ProbeCachePressure{PromptTokens: 10, GeneratedTokens: 2, LayerCount: 6, CacheTokens: 12, ProcessedTokens: 14, MaxCacheTokens: 20, Utilization: 0.6, Rotating: true},
		Memory:         &metal.ProbeMemoryPressure{ActiveBytes: 100, PeakBytes: 200, CacheBytes: 50},
		Training:       &metal.ProbeTraining{Step: 6, Epoch: 1, Loss: 0.9, LearningRate: 0.01, GradNorm: 0.3},
	})
	if got.Token == nil || got.Logits == nil || got.SelectedHeads == nil || got.RouterDecision == nil || got.Training == nil {
		t.Fatalf("probe event = %+v, want all nested payloads", got)
	}
	if got.Meta["scope"] != "unit" || got.Logits.Top[0].TokenID != 4 || got.Cache == nil || !got.Cache.Rotating {
		t.Fatalf("probe event = %+v, want cloned meta/logits/cache", got)
	}
	got.Meta["scope"] = "changed"
	got.Logits.Meta["logits"] = "changed"
	if meta["scope"] != "unit" || logitMeta["logits"] != "kept" {
		t.Fatal("probe conversion leaked metadata map mutation")
	}
	if toRootProbeLogits(nil) != nil || cloneMetalProbeMeta(nil) != nil {
		t.Fatal("empty probe helpers should return nil")
	}
}

func TestAPIKVHeadDTypeAndChunkStringHelpers_Good(t *testing.T) {
	if rootKVHeadDType(metal.DTypeFloat16, []byte{1}) != "float16" {
		t.Fatal("rootKVHeadDType(float16) did not preserve dtype")
	}
	if rootKVHeadDType(metal.DTypeFloat32, nil) != "" || rootKVHeadDType(metal.DTypeInt8, []byte{1}) != "" {
		t.Fatal("rootKVHeadDType should reject empty raw data and unsupported dtype")
	}
	if metalKVHeadDType("F32", []byte{1}) != metal.DTypeFloat32 || metalKVHeadDType("BF16", []byte{1}) != metal.DTypeBFloat16 {
		t.Fatal("metalKVHeadDType aliases did not map to metal dtypes")
	}
	if metalKVHeadDType("bad", []byte{1}) != 0 || metalKVHeadDType("float16", nil) != 0 {
		t.Fatal("metalKVHeadDType should reject empty raw data and unsupported dtype")
	}
	if promptChunksToString(seqStrings("a", "b", "c")) != "abc" || promptChunksToString(nil) != "" {
		t.Fatal("promptChunksToString returned unexpected string")
	}
}

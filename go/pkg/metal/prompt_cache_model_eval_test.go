// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"context"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/mlx/internal/metaltest"
)

// TestPromptCache_StoreMatchRestore_Eval drives the exact-prefix prompt cache on
// a real model: a first generation stores it (storePromptCache), and a second
// prompt that shares the stored prefix then diverges hits it (promptCacheMatch +
// restoreFixedCacheSnapshot). The synthetic suite never replays a prefix, so this
// lifecycle is otherwise dark. The store threshold is lowered from its 2048-token
// default so the logic can be exercised without a multi-thousand-token prompt —
// the cache code paths are identical regardless of the threshold value. Run:
//
//	GO_MLX_BENCH_MODEL=google/gemma-4-e2b-it go test \
//	  -tags 'metal_runtime model_eval' -run TestPromptCache_StoreMatchRestore_Eval ./pkg/metal/
func TestPromptCache_StoreMatchRestore_Eval(t *testing.T) {
	if !metaltest.RunModelEvalTests {
		t.Skip("model-eval test; build with -tags model_eval")
	}
	isolateRuntimeGates(t)
	restore := DefaultEngineFeatures().Apply()
	defer restore()

	repo := core.Getenv("GO_MLX_BENCH_MODEL")
	if repo == "" {
		repo = "google/gemma-4-e2b-it"
	}
	dir := metaltest.HFModelPath(t, repo)
	model, err := LoadAndInit(dir, LoadConfig{ContextLen: 4096})
	if err != nil {
		t.Fatalf("LoadAndInit: %v", err)
	}
	defer model.Close()
	if !model.PromptCacheEnabled() {
		t.Fatal("prompt cache should be enabled by default")
	}
	// Lower the 2048-token store floor so short test prompts exercise the cache;
	// the store/match/restore code paths are unchanged by the threshold.
	model.promptCacheMinTokens = 8

	ctx := context.Background()
	base := "The capital city of France is Paris, a place with a long and storied " +
		"history of art, science, philosophy, and culture spanning many centuries."
	cfg := GenerateConfig{MaxTokens: 16}

	// First generation: cold — stores base + its continuation as the cache entry.
	for range model.Generate(ctx, base, cfg) {
	}
	if err := model.Err(); err != nil {
		t.Fatalf("generation 1: %v", err)
	}
	if hits := model.LastMetrics().PromptCacheHits; hits != 0 {
		t.Fatalf("generation 1 PromptCacheHits = %d, want 0 (cold cache)", hits)
	}

	// Second generation: shares `base` as a strict prefix, then diverges — so the
	// stored prefix is reused (match at the divergence point, not the whole thing).
	for range model.Generate(ctx, base+" Tell me much more about its famous museums and landmarks.", cfg) {
	}
	if err := model.Err(); err != nil {
		t.Fatalf("generation 2: %v", err)
	}
	m := model.LastMetrics()
	if m.PromptCacheHits != 1 || m.PromptCacheHitTokens <= 0 {
		t.Fatalf("generation 2 cache = %d hit / %d tokens, want a prefix hit reusing the stored base",
			m.PromptCacheHits, m.PromptCacheHitTokens)
	}

	// Third generation: a different extension of the same base → another prefix hit
	// against the entry re-stored by generation 2.
	for range model.Generate(ctx, base+" Describe the river that runs through it.", cfg) {
	}
	if err := model.Err(); err != nil {
		t.Fatalf("generation 3: %v", err)
	}
	if m := model.LastMetrics(); m.PromptCacheHitTokens <= 0 {
		t.Fatalf("generation 3 should reuse the shared base prefix (got %d hit tokens)", m.PromptCacheHitTokens)
	}
}

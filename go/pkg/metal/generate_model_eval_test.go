// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"context"
	"slices"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/mlx/internal/metaltest"
)

// TestGenerate_EntrypointsAndDirectPath_Eval drives the model-level generation
// entrypoints the synthetic suite leaves dark: Chat / ChatChunks (chat-template
// formatting), GenerateChunks (chunked prompt), WarmPromptCache, and the direct
// generate() path — the default route goes through generateViaSession, so the
// non-session branch (sessionRouteEligible == !cfg.ClearCache) only runs when
// ClearCache is set. Run:
//
//	GO_MLX_BENCH_MODEL=google/gemma-4-e2b-it go test \
//	  -tags 'metal_runtime model_eval' -run TestGenerate_EntrypointsAndDirectPath_Eval ./pkg/metal/
func TestGenerate_EntrypointsAndDirectPath_Eval(t *testing.T) {
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
	model.promptCacheMinTokens = 8 // let WarmPromptCache actually store a short prompt

	ctx := context.Background()
	open := "Write a long, detailed story about a lighthouse keeper and the deep ocean."
	cfg := GenerateConfig{MaxTokens: 16}
	msgs := []ChatMessage{{Role: "user", Content: "Tell me a short story about the sea."}}

	t.Run("Chat", func(t *testing.T) {
		n := 0
		for range model.Chat(ctx, msgs, cfg) {
			n++
		}
		if err := model.Err(); err != nil {
			t.Fatalf("Chat: %v", err)
		}
		if n == 0 {
			t.Fatal("Chat produced no tokens")
		}
	})

	t.Run("ChatChunks", func(t *testing.T) {
		n := 0
		for range model.ChatChunks(ctx, msgs, 64, cfg) {
			n++
		}
		if err := model.Err(); err != nil {
			t.Fatalf("ChatChunks: %v", err)
		}
		if n == 0 {
			t.Fatal("ChatChunks produced no tokens")
		}
	})

	t.Run("GenerateChunks", func(t *testing.T) {
		chunks := slices.Values([]string{"Write a long story ", "about a lighthouse keeper ", "and the deep ocean."})
		n := 0
		for range model.GenerateChunks(ctx, chunks, cfg) {
			n++
		}
		if err := model.Err(); err != nil {
			t.Fatalf("GenerateChunks: %v", err)
		}
		if n == 0 {
			t.Fatal("GenerateChunks produced no tokens")
		}
	})

	t.Run("WarmPromptCache", func(t *testing.T) {
		if err := model.WarmPromptCache(ctx, open); err != nil {
			t.Fatalf("WarmPromptCache: %v", err)
		}
	})

	// ClearCache forces the non-session route → the direct generate() path.
	t.Run("DirectGenerate_ClearCache", func(t *testing.T) {
		n := 0
		for range model.Generate(ctx, open, GenerateConfig{MaxTokens: 16, ClearCache: true}) {
			n++
		}
		if err := model.Err(); err != nil {
			t.Fatalf("direct Generate: %v", err)
		}
		if n == 0 {
			t.Fatal("direct generate produced no tokens")
		}
	})
}

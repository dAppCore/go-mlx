// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"context"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/mlx/internal/metaltest"
)

// TestBatch_GenerateAndClassify_Eval drives the batched lanes on a real model —
// the synthetic suite only ever runs single-stream, so batch.go's prefill/mask/
// per-sequence decode is otherwise dark. Run:
//
//	GO_MLX_BENCH_MODEL=google/gemma-4-e2b-it go test \
//	  -tags 'metal_runtime model_eval' -run TestBatch_GenerateAndClassify_Eval ./pkg/metal/
//
// Prompts are deliberately different lengths so the padding mask
// (buildOptionalBatchMask / buildBatchMask) is exercised, not skipped.
func TestBatch_GenerateAndClassify_Eval(t *testing.T) {
	if !metaltest.RunModelEvalTests {
		t.Skip("model-eval test; build with -tags model_eval")
	}
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

	// Varied lengths → the batch needs an explicit padding mask.
	prompts := []string{
		"The capital of France is",
		"2+2=",
		"Once upon a time, in a land far away,",
	}

	t.Run("BatchGenerate", func(t *testing.T) {
		results, err := model.BatchGenerate(context.Background(), prompts, GenerateConfig{MaxTokens: 12})
		if err != nil {
			t.Fatalf("BatchGenerate: %v", err)
		}
		if len(results) != len(prompts) {
			t.Fatalf("results = %d, want %d (one per prompt, original order)", len(results), len(prompts))
		}
		total := 0
		for i, r := range results {
			if r.Err != nil {
				t.Fatalf("sequence %d error: %v", i, r.Err)
			}
			total += len(r.Tokens)
		}
		if total == 0 {
			t.Fatal("batch produced no tokens across any sequence")
		}
	})

	t.Run("ClassifyWithLogits", func(t *testing.T) {
		results, err := model.Classify(context.Background(), prompts, GenerateConfig{}, true)
		if err != nil {
			t.Fatalf("Classify: %v", err)
		}
		if len(results) != len(prompts) {
			t.Fatalf("results = %d, want %d", len(results), len(prompts))
		}
		for i, r := range results {
			if r.Token.ID < 0 {
				t.Fatalf("sequence %d sampled a negative token id %d", i, r.Token.ID)
			}
			if len(r.Logits) == 0 {
				t.Fatalf("sequence %d returned no logits with returnLogits=true", i)
			}
		}
	})

	t.Run("ClassifyWithoutLogits", func(t *testing.T) {
		results, err := model.Classify(context.Background(), prompts, GenerateConfig{}, false)
		if err != nil {
			t.Fatalf("Classify: %v", err)
		}
		for i, r := range results {
			if r.Logits != nil {
				t.Fatalf("sequence %d returned logits with returnLogits=false", i)
			}
		}
	})

	// A batch larger than the size limit walks the chunking planner.
	t.Run("ChunkedBatch", func(t *testing.T) {
		old := model.batchSizeLimit
		model.batchSizeLimit = 2
		defer func() { model.batchSizeLimit = old }()
		results, err := model.BatchGenerate(context.Background(), prompts, GenerateConfig{MaxTokens: 8})
		if err != nil {
			t.Fatalf("chunked BatchGenerate: %v", err)
		}
		if len(results) != len(prompts) {
			t.Fatalf("chunked results = %d, want %d", len(results), len(prompts))
		}
	})

	// A single-prompt batch is uniform length, so the explicit padding mask is
	// skipped (batchNeedsExplicitMask == false) — the other branch.
	t.Run("SingleStreamNoMask", func(t *testing.T) {
		results, err := model.BatchGenerate(context.Background(), []string{"The capital of France is"}, GenerateConfig{MaxTokens: 8})
		if err != nil {
			t.Fatalf("single-stream BatchGenerate: %v", err)
		}
		if len(results) != 1 {
			t.Fatalf("results = %d, want 1", len(results))
		}
	})
}

// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"testing"

	"dappco.re/go/inference"
	"dappco.re/go/mlx/internal/metaltest"
)

func TestSubstrateParity_PromptCacheReplay_Good(t *testing.T) {
	if !metaltest.RunMetalTests {
		t.Skip("build with -tags metal_runtime to run the local substrate parity smoke")
	}
	modelPath := metaltest.HFModelPath(t, "mlx-community/gemma-4-e2b-it-6bit")

	model, err := LoadModel(
		modelPath,
		WithContextLength(4096),
		WithBatchSize(512),
		WithPrefillChunkSize(512),
		WithPromptCache(true),
		WithPromptCacheMinTokens(1),
	)
	if err != nil {
		t.Fatalf("LoadModel() error = %v", err)
	}
	defer func() {
		if err := model.Close(); err != nil {
			t.Fatalf("Close() error = %v", err)
		}
	}()

	messages := []inference.Message{{
		Role:    "user",
		Content: "Write exactly one short sentence about retained model state.",
	}}
	opts := []GenerateOption{
		WithMaxTokens(64),
		WithTemperature(1.0),
		WithTopP(0.95),
		WithTopK(64),
		WithSeed(42),
		WithShowThinking(),
	}

	miss, err := model.Chat(messages, opts...)
	if err != nil {
		t.Fatalf("Chat(cache miss) error = %v", err)
	}
	hit, err := model.Chat(messages, opts...)
	if err != nil {
		t.Fatalf("Chat(cache hit) error = %v", err)
	}
	if err := model.ClearPromptCache(); err != nil {
		t.Fatalf("ClearPromptCache() error = %v", err)
	}
	replay, err := model.Chat(messages, opts...)
	if err != nil {
		t.Fatalf("Chat(replay) error = %v", err)
	}

	if hit == "" {
		t.Fatal("prompt-cache hit output is empty")
	}
	if miss != hit {
		t.Fatalf("cache miss output != cache hit output\nmiss: %q\n hit: %q", miss, hit)
	}
	if hit != replay {
		t.Fatalf("cache hit output != replay output\n hit: %q\nreplay: %q", hit, replay)
	}
}

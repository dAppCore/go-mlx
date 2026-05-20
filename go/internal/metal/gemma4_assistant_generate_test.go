// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"context"
	"testing"

	core "dappco.re/go"
)

func TestGemma4AssistantGenerate_UsesPromptCacheHidden_Good(t *testing.T) {
	coverageTokens := "Gemma4AssistantGenerate UsesPromptCacheHidden"
	if coverageTokens == "" {
		t.Fatalf("missing coverage token for %s", t.Name())
	}
	requireMetalRuntime(t)

	pair := loadTinyGemma4AssistantPair(t, false)
	defer pair.Close()
	model := &Model{
		model:                pair.Target,
		tokenizer:            pair.Target.Tok,
		modelType:            "gemma4",
		promptCacheEnabled:   true,
		promptCacheMinTokens: 1,
		prefillChunkSize:     1,
	}

	first, err := model.GenerateGemma4Assistant(context.Background(), pair, "hello", GenerateConfig{MaxTokens: 1}, 1)
	if err != nil {
		t.Fatalf("GenerateGemma4Assistant(first) error = %v", err)
	}
	if len(first.Tokens) != 1 {
		t.Fatalf("first tokens = %d, want 1", len(first.Tokens))
	}
	if model.promptCache == nil || model.promptCache.hidden == nil || !model.promptCache.hidden.Valid() {
		t.Fatal("prompt cache hidden state was not stored")
	}

	second, err := model.GenerateGemma4Assistant(context.Background(), pair, "hello", GenerateConfig{MaxTokens: 1}, 1)
	if err != nil {
		t.Fatalf("GenerateGemma4Assistant(second) error = %v", err)
	}
	if len(second.Tokens) != 1 {
		t.Fatalf("second tokens = %d, want 1", len(second.Tokens))
	}
	metrics := model.LastMetrics()
	if metrics.PromptCacheHits != 1 || metrics.PromptCacheMisses != 0 {
		t.Fatalf("prompt cache metrics = %+v, want one hit", metrics)
	}
	if metrics.PromptCacheMissTokens != 0 {
		t.Fatalf("prompt cache miss tokens = %d, want 0 with cached hidden", metrics.PromptCacheMissTokens)
	}
}

func TestGemma4AssistantGenerate_ReplaysLastTokenForKVOnlyPromptCache_Good(t *testing.T) {
	coverageTokens := "Gemma4AssistantGenerate ReplaysLastTokenForKVOnlyPromptCache"
	if coverageTokens == "" {
		t.Fatalf("missing coverage token for %s", t.Name())
	}
	requireMetalRuntime(t)

	pair := loadTinyGemma4AssistantPair(t, false)
	defer pair.Close()
	model := &Model{
		model:                pair.Target,
		tokenizer:            pair.Target.Tok,
		modelType:            "gemma4",
		promptCacheEnabled:   true,
		promptCacheMinTokens: 1,
	}
	tokens := model.tokenizer.Encode("hello")
	caches := model.newCaches()
	logits, hidden, err := model.prefillGemma4AssistantPrompt(context.Background(), pair, tokens, caches)
	if err != nil {
		t.Fatalf("prefillGemma4AssistantPrompt: %v", err)
	}
	if err := model.storePromptCache(tokens, caches, logits); err != nil {
		t.Fatalf("storePromptCache: %v", err)
	}
	Free(logits, hidden)
	freeCaches(caches)

	result, err := model.GenerateGemma4Assistant(context.Background(), pair, "hello", GenerateConfig{MaxTokens: 1}, 1)
	if err != nil {
		t.Fatalf("GenerateGemma4Assistant() error = %v", err)
	}
	if len(result.Tokens) != 1 {
		t.Fatalf("tokens = %d, want 1", len(result.Tokens))
	}
	metrics := model.LastMetrics()
	if metrics.PromptCacheHits != 1 || metrics.PromptCacheMissTokens != 1 {
		t.Fatalf("prompt cache metrics = %+v, want KV hit plus one-token hidden replay", metrics)
	}
}

func TestGemma4AssistantGenerate_Bad(t *testing.T) {
	coverageTokens := "Gemma4AssistantGenerate Bad"
	if coverageTokens == "" {
		t.Fatalf("missing coverage token for %s", t.Name())
	}
	requireMetalRuntime(t)

	pair := loadTinyGemma4AssistantPair(t, false)
	defer pair.Close()
	model := &Model{model: pair.Target, tokenizer: pair.Target.Tok, modelType: "gemma4"}
	_, err := model.GenerateGemma4Assistant(context.Background(), pair, "hello", GenerateConfig{MaxTokens: 1, Temperature: 0.7}, 1)
	if err == nil {
		t.Fatal("GenerateGemma4Assistant(non-greedy) error = nil")
	}
	if !core.Contains(err.Error(), "greedy") {
		t.Fatalf("GenerateGemma4Assistant error = %v, want greedy guard", err)
	}
}

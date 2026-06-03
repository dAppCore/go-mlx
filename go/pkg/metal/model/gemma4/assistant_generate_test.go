// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma4

import (
	"context"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/mlx/pkg/metal"
)

func TestGemma4AssistantGenerate_UsesPromptCacheHidden_Good(t *testing.T) {
	coverageTokens := "Gemma4AssistantGenerate UsesPromptCacheHidden"
	if coverageTokens == "" {
		t.Fatalf("missing coverage token for %s", t.Name())
	}
	requireMetalRuntime(t)

	pair := loadTinyGemma4AssistantPair(t, false)
	defer pair.Close()
	model := &metal.Model{
		model:                pair.Target,
		tokenizer:            pair.Target.Tok,
		modelType:            "gemma4",
		promptCacheEnabled:   true,
		promptCacheMinTokens: 1,
		prefillChunkSize:     1,
	}

	first, err := model.GenerateGemma4Assistant(context.Background(), pair, "hello", metal.GenerateConfig{MaxTokens: 1}, 1)
	if err != nil {
		t.Fatalf("GenerateGemma4Assistant(first) error = %v", err)
	}
	if len(first.Tokens) != 1 {
		t.Fatalf("first tokens = %d, want 1", len(first.Tokens))
	}
	if first.FirstTokenDuration <= 0 {
		t.Fatalf("first token duration = %s, want positive", first.FirstTokenDuration)
	}
	if model.promptCache == nil || model.promptCache.hidden == nil || !model.promptCache.hidden.Valid() {
		t.Fatal("prompt cache hidden state was not stored")
	}

	second, err := model.GenerateGemma4Assistant(context.Background(), pair, "hello", metal.GenerateConfig{MaxTokens: 1}, 1)
	if err != nil {
		t.Fatalf("GenerateGemma4Assistant(second) error = %v", err)
	}
	if len(second.Tokens) != 1 {
		t.Fatalf("second tokens = %d, want 1", len(second.Tokens))
	}
	if second.FirstTokenDuration <= 0 {
		t.Fatalf("second first token duration = %s, want positive", second.FirstTokenDuration)
	}
	metrics := model.LastMetrics()
	if metrics.PromptCacheHits != 1 || metrics.PromptCacheMisses != 0 {
		t.Fatalf("prompt cache metrics = %+v, want one hit", metrics)
	}
	if metrics.FirstTokenDuration <= 0 {
		t.Fatalf("metrics first token duration = %s, want positive", metrics.FirstTokenDuration)
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
	model := &metal.Model{
		model:                pair.Target,
		tokenizer:            pair.Target.Tok,
		modelType:            "gemma4",
		promptCacheEnabled:   true,
		promptCacheMinTokens: 1,
	}
	prompt := "<bos><eos>"
	tokens := model.tokenizer.Encode(prompt)
	if len(tokens) < 2 {
		t.Fatalf("test prompt encoded to %v, want at least two tokens for final-token replay", tokens)
	}
	caches := model.newCaches()
	logits, hidden, err := model.prefillGemma4AssistantPrompt(context.Background(), pair, tokens, caches)
	if err != nil {
		t.Fatalf("prefillGemma4AssistantPrompt: %v", err)
	}
	if err := model.storePromptCache(tokens, caches, logits); err != nil {
		t.Fatalf("storePromptCache: %v", err)
	}
	metal.Free(logits, hidden)
	metal.FreeCaches(caches)

	result, err := model.GenerateGemma4Assistant(context.Background(), pair, prompt, metal.GenerateConfig{MaxTokens: 1}, 1)
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

func TestGemma4AssistantGenerate_LoadLocalAssistantPair_Good(t *testing.T) {
	coverageTokens := "Gemma4AssistantGenerate LoadLocalAssistantPair"
	if coverageTokens == "" {
		t.Fatalf("missing coverage token for %s", t.Name())
	}
	targetPath := core.Trim(core.Env("GO_MLX_GEMMA4_TARGET_MODEL"))
	assistantPath := core.Trim(core.Env("GO_MLX_GEMMA4_ASSISTANT_MODEL"))
	if targetPath == "" || assistantPath == "" {
		t.Skip("set GO_MLX_GEMMA4_TARGET_MODEL and GO_MLX_GEMMA4_ASSISTANT_MODEL to run the local assistant generation smoke")
	}

	pair, err := LoadGemma4AssistantPair(targetPath, assistantPath)
	if err != nil {
		t.Fatalf("LoadGemma4AssistantPair(%s, %s): %v", targetPath, assistantPath, err)
	}
	defer pair.Close()

	model := &metal.Model{
		model:      pair.Target,
		tokenizer:  pair.Target.Tok,
		modelType:  "gemma4",
		contextLen: 64,
		cacheMode:  string(KVCacheModePaged),
	}
	result, err := model.GenerateGemma4Assistant(context.Background(), pair, "Hello", metal.GenerateConfig{MaxTokens: 2}, 1)
	if err != nil {
		t.Fatalf("GenerateGemma4Assistant(local) error = %v", err)
	}
	if result.PromptTokens <= 0 || len(result.Tokens) == 0 || len(result.Tokens) > 2 {
		t.Fatalf("generation counts = prompt:%d generated:%d, want non-empty prompt and 1-2 generated tokens", result.PromptTokens, len(result.Tokens))
	}
	if result.FirstTokenDuration <= 0 {
		t.Fatalf("first token duration = %s, want positive", result.FirstTokenDuration)
	}
	if result.DraftCalls == 0 || result.DraftTokens == 0 || result.TargetVerifyCalls == 0 || result.TargetCalls == 0 {
		t.Fatalf("MTP counters = draft_calls:%d draft_tokens:%d verify_calls:%d target_calls:%d, want exercised assistant and target verify loop", result.DraftCalls, result.DraftTokens, result.TargetVerifyCalls, result.TargetCalls)
	}
	if result.AcceptedTokens+result.RejectedTokens != result.DraftTokens {
		t.Fatalf("acceptance accounting = accepted:%d rejected:%d draft:%d, want accepted+rejected == draft", result.AcceptedTokens, result.RejectedTokens, result.DraftTokens)
	}
	metrics := model.LastMetrics()
	if metrics.GeneratedTokens != len(result.Tokens) || metrics.DecodeTokensPerSec <= 0 {
		t.Fatalf("metrics = %+v, want generated count and positive decode rate", metrics)
	}
	if metrics.FirstTokenDuration <= 0 {
		t.Fatalf("metrics first token duration = %s, want positive", metrics.FirstTokenDuration)
	}
	if metrics.MTP == nil || metrics.MTP.ProposedTokens != result.DraftTokens || metrics.MTP.DraftCalls != result.DraftCalls || metrics.MTP.TargetVerifyCalls != result.TargetVerifyCalls {
		t.Fatalf("MTP metrics = %+v, want result counters draft_tokens:%d draft_calls:%d verify_calls:%d", metrics.MTP, result.DraftTokens, result.DraftCalls, result.TargetVerifyCalls)
	}
}

func TestGemma4AssistantGenerate_DefaultDraftTokensPolicy_Good(t *testing.T) {
	coverageTokens := "Gemma4AssistantGenerate DefaultDraftTokensPolicy"
	if coverageTokens == "" {
		t.Fatalf("missing coverage token for %s", t.Name())
	}

	for _, input := range []int{0, -4} {
		if got := gemma4AssistantResolveDraftTokens(input); got != 2 {
			t.Fatalf("gemma4AssistantResolveDraftTokens(%d) = %d, want 2", input, got)
		}
	}
	if got := gemma4AssistantResolveDraftTokens(4); got != 4 {
		t.Fatalf("gemma4AssistantResolveDraftTokens(4) = %d, want explicit value preserved", got)
	}
}

func TestGemma4AssistantGenerate_StopTokenWithheld_Good(t *testing.T) {
	coverageTokens := "Gemma4AssistantGenerate StopTokenWithheld"
	if coverageTokens == "" {
		t.Fatalf("missing coverage token for %s", t.Name())
	}

	model := &metal.Model{tokenizer: &metal.Tokenizer{invVocab: map[int32]string{7: "<turn|>", 8: "x"}}}
	result := &Gemma4AssistantGenerateResult{}

	if stopped := model.appendGemma4AssistantToken(result, 7, metal.GenerateConfig{StopTokens: []int32{7}}); !stopped {
		t.Fatal("appendGemma4AssistantToken(stop) = false, want true")
	}
	if len(result.Tokens) != 0 || result.Text != "" {
		t.Fatalf("result after stop token = tokens:%+v text:%q, want withheld visible output", result.Tokens, result.Text)
	}

	if stopped := model.appendGemma4AssistantToken(result, 8, metal.GenerateConfig{StopTokens: []int32{7}}); stopped {
		t.Fatal("appendGemma4AssistantToken(non-stop) = true, want false")
	}
	if len(result.Tokens) != 1 || result.Tokens[0].ID != 8 || result.Tokens[0].Text != "x" || result.Text != "x" {
		t.Fatalf("result after non-stop token = tokens:%+v text:%q, want visible token x", result.Tokens, result.Text)
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
	model := &metal.Model{model: pair.Target, tokenizer: pair.Target.Tok, modelType: "gemma4"}
	_, err := model.GenerateGemma4Assistant(context.Background(), pair, "hello", metal.GenerateConfig{MaxTokens: 1, metal.Temperature: 0.7}, 1)
	if err == nil {
		t.Fatal("GenerateGemma4Assistant(non-metal.Greedy) error = nil")
	}
	if !core.Contains(err.Error(), "metal.Greedy") {
		t.Fatalf("GenerateGemma4Assistant error = %v, want metal.Greedy guard", err)
	}
}

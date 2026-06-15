// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma4

import (
	"context"
	"testing"

	"dappco.re/go/mlx/pkg/metal"
)

// loadTinyGemma4AssistantRuntimeCached mirrors loadTinyGemma4AssistantRuntime
// but ENABLES the prompt cache with a 1-token minimum, so a re-run of the same
// prompt takes the prompt-cache restore path instead of a fresh prefill. The
// shared helper disables the cache (for fresh-prefill determinism in the
// equivalence tests); the cache-hit path needs its own runtime.
func loadTinyGemma4AssistantRuntimeCached(t *testing.T) (*metal.Model, *Gemma4AssistantPair) {
	t.Helper()
	requireMetalRuntime(t)

	targetDir := t.TempDir()
	writeGemma4AssistantTargetConfig(t, targetDir)
	writeMinimalTokenizer(t, targetDir)
	if err := metal.SaveSafetensors(targetDir+"/model.safetensors", gemma4VerifyPeakedTargetWeights()); err != nil {
		t.Fatalf("SaveSafetensors target: %v", err)
	}
	assistantDir := t.TempDir()
	writeGemma4AssistantConfig(t, assistantDir, false)
	writeMinimalTokenizer(t, assistantDir)
	if err := metal.SaveSafetensors(assistantDir+"/model.safetensors", gemma4AssistantTinyWeights(false)); err != nil {
		t.Fatalf("SaveSafetensors assistant: %v", err)
	}
	m, err := metal.LoadAndInit(targetDir, metal.LoadConfig{PromptCacheMinTokens: 1})
	if err != nil {
		t.Fatalf("metal.LoadAndInit(tiny target, cache on): %v", err)
	}
	pair, err := AttachGemma4Assistant(m, assistantDir)
	if err != nil {
		m.Close()
		t.Fatalf("AttachGemma4Assistant: %v", err)
	}
	t.Cleanup(func() {
		pair.Close()
		m.Close()
	})
	return m, pair
}

// TestAssistantGenerate_PromptCacheHit covers prefillGemma4AssistantFromPromptCache
// via its full-hit copy branch: a first bounded generation stores the prompt's
// caches + last-token state, and a second generation on the SAME prompt restores
// them instead of re-prefilling. The hit is asserted OBSERVABLY through the
// last-metrics PromptCacheHits counter — that counter only increments when the
// restore path returned without error, so a silently no-op'd store (which would
// leave the function uncovered yet the run green) is caught. No MTP==plain
// equivalence is asserted: a restored prefix differs from a fresh prefill in
// low-bit float order by design.
func TestAssistantGenerate_PromptCacheHit(t *testing.T) {
	m, pair := loadTinyGemma4AssistantRuntimeCached(t)
	ctx := context.Background()
	cfg := metal.GenerateConfig{MaxTokens: 3}

	// First run populates the prompt cache (miss).
	first, err := pair.GenerateWithSink(ctx, m, "hello", cfg, 2, nil)
	if err != nil {
		t.Fatalf("first GenerateWithSink: %v", err)
	}
	if first.PromptTokens == 0 {
		t.Fatal("first run tokenised to nothing")
	}
	if hits := m.LastMetrics().PromptCacheHits; hits != 0 {
		t.Fatalf("first run PromptCacheHits = %d, want 0 (cold cache)", hits)
	}

	// Second run on the identical prompt must restore from the cache.
	if _, err := pair.GenerateWithSink(ctx, m, "hello", cfg, 2, nil); err != nil {
		t.Fatalf("second GenerateWithSink: %v", err)
	}
	if hits := m.LastMetrics().PromptCacheHits; hits != 1 {
		t.Fatalf("second run PromptCacheHits = %d, want 1 (warm cache restore)", hits)
	}
}

// TestAssistantGenerate_MTPStepDiag covers the temp MTP step diagnostic, which
// the greedy speculative loop fires for its first few draft calls only when the
// GO_MLX_MTP_DIAG env var is set. Setting it for a tiny greedy bounded run drives
// the diagnostic's draft-step + rank computation. The diagnostic only logs; the
// assertion is that generation still completes with the probe wired in.
func TestAssistantGenerate_MTPStepDiag(t *testing.T) {
	t.Setenv("GO_MLX_MTP_DIAG", "1")
	m, pair := loadTinyGemma4AssistantRuntime(t)

	result, err := pair.Generate(context.Background(), m, "hello world", metal.GenerateConfig{MaxTokens: 3}, 2)
	if err != nil {
		t.Fatalf("Generate(MTP diag): %v", err)
	}
	if len(result.Tokens) == 0 {
		t.Fatal("Generate(MTP diag) emitted no tokens")
	}
}

// TestAssistantGenerate_forwardAcceptedToken_Good covers the single-token target
// forward helper directly: it advances the target one accepted token, returning
// the next-token logits and the hidden state, and detaching the caches. Driving
// it directly (rather than through the tokenizer-gated prompt-cache suffix loop,
// which the synthetic tokenizer cannot reach — every prompt collapses to BOS)
// pins the helper's success path on the pair's own target caches.
func TestAssistantGenerate_forwardAcceptedToken_Good(t *testing.T) {
	_, pair := loadTinyGemma4AssistantRuntime(t)

	caches := pair.Target.NewCache()
	defer metal.FreeCaches(caches)

	// Token 3 is in-vocab (the target declares vocab_size 10).
	logits, hidden, err := pair.forwardGemma4AssistantAcceptedToken(int32(3), caches)
	if err != nil {
		t.Fatalf("forwardGemma4AssistantAcceptedToken: %v", err)
	}
	defer metal.Free(logits, hidden)
	if logits == nil || !logits.Valid() || hidden == nil || !hidden.Valid() {
		t.Fatalf("forward returned invalid state: logits=%v hidden=%v", logits, hidden)
	}
	if shape := logits.Shape(); len(shape) == 0 || shape[len(shape)-1] != pair.Target.Cfg.VocabSize {
		t.Fatalf("logits last dim = %v, want vocab %d", shape, pair.Target.Cfg.VocabSize)
	}
}

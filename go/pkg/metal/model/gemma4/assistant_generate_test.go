// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma4

import (
	"context"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/mlx/pkg/metal"
)

// TestAssistantGenerate_validateConfig_Good covers the speculative-generation
// config gate directly: the supported sampling knobs pass, while repetition
// penalty and a probe sink — both of which fall back to plain decode at the
// serve gate — are rejected with a named error.
func TestAssistantGenerate_validateConfig_Good(t *testing.T) {
	if err := validateGemma4AssistantGenerateConfig(metal.GenerateConfig{
		MaxTokens:   8,
		Temperature: 0.7,
		TopK:        20,
		TopP:        0.9,
		MinP:        0.05,
	}); err != nil {
		t.Fatalf("validateGemma4AssistantGenerateConfig(supported) = %v, want nil", err)
	}

	err := validateGemma4AssistantGenerateConfig(metal.GenerateConfig{RepeatPenalty: 1.1})
	if err == nil || !core.Contains(err.Error(), "repetition penalty") {
		t.Fatalf("validateGemma4AssistantGenerateConfig(repeat) = %v, want repetition penalty rejection", err)
	}

	err = validateGemma4AssistantGenerateConfig(metal.GenerateConfig{
		ProbeSink: metal.ProbeSinkFunc(func(metal.ProbeEvent) {}),
	})
	if err == nil || !core.Contains(err.Error(), "probe sink") {
		t.Fatalf("validateGemma4AssistantGenerateConfig(probe) = %v, want probe sink rejection", err)
	}
}

// TestAssistantGenerate_resolveDraftTokens_Good covers the draft-token default
// resolver: a non-positive request resolves to the package default block, a
// positive request is taken as-is.
func TestAssistantGenerate_resolveDraftTokens_Good(t *testing.T) {
	if got := gemma4AssistantResolveDraftTokens(0); got != gemma4AssistantDefaultDraftTokens {
		t.Fatalf("resolve(0) = %d, want default %d", got, gemma4AssistantDefaultDraftTokens)
	}
	if got := gemma4AssistantResolveDraftTokens(-3); got != gemma4AssistantDefaultDraftTokens {
		t.Fatalf("resolve(-3) = %d, want default %d", got, gemma4AssistantDefaultDraftTokens)
	}
	if got := gemma4AssistantResolveDraftTokens(2); got != 2 {
		t.Fatalf("resolve(2) = %d, want 2", got)
	}
}

// TestAssistantGenerate_Generate_Good drives the public Generate wrapper (the
// nil-sink entry point) over the tiny synthetic target+assistant runtime. It
// asserts the bounded run completes and emits at least one token, exercising
// the wrapper plus the speculative loop body. No MTP==plain equivalence is
// asserted here — that invariant has its own dedicated tests; this one pins the
// Generate-vs-GenerateWithSink delegation and a non-empty result.
func TestAssistantGenerate_Generate_Good(t *testing.T) {
	m, pair := loadTinyGemma4AssistantRuntime(t)

	result, err := pair.Generate(context.Background(), m, "hello world", metal.GenerateConfig{MaxTokens: 4}, 2)
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if len(result.Tokens) == 0 {
		t.Fatal("Generate emitted no tokens")
	}
	if result.PromptTokens == 0 {
		t.Fatalf("Generate PromptTokens = 0, want > 0")
	}
}

// TestAssistantGenerate_GenerateWithSink_DefaultMaxTokens covers the
// MaxTokens<=0 branch (cap defaults to the model context length) and the nil
// ctx branch (defaults to context.Background) together — both run before the
// generation loop, so they are covered the instant the call executes. The sink
// caps the run at a few tokens: the synthetic model has no reliable EOS, so a
// run-to-context-length would generate for minutes. The sink-count invariant
// (every appended token calls the sink exactly once) still holds.
func TestAssistantGenerate_GenerateWithSink_DefaultMaxTokens(t *testing.T) {
	m, pair := loadTinyGemma4AssistantRuntime(t)

	streamed := 0
	sink := func(metal.Token) bool { streamed++; return streamed < 3 } // stop after 3
	//nolint:staticcheck // SA1012: nil ctx is the branch under test (defaults to Background).
	result, err := pair.GenerateWithSink(nil, m, "hello", metal.GenerateConfig{}, 2, sink)
	if err != nil {
		t.Fatalf("GenerateWithSink(nil ctx, MaxTokens=0): %v", err)
	}
	if streamed != len(result.Tokens) {
		t.Fatalf("sink saw %d tokens, result has %d", streamed, len(result.Tokens))
	}
}

// TestAssistantGenerate_GenerateWithSink_SinkStop covers the sink-requested
// stop: a sink that returns false on the first token halts generation, and the
// loop returns what it has with no error.
func TestAssistantGenerate_GenerateWithSink_SinkStop(t *testing.T) {
	m, pair := loadTinyGemma4AssistantRuntime(t)

	calls := 0
	sink := func(metal.Token) bool { calls++; return false } // stop immediately
	result, err := pair.GenerateWithSink(context.Background(), m, "hello world", metal.GenerateConfig{MaxTokens: 16}, 2, sink)
	if err != nil {
		t.Fatalf("GenerateWithSink(stop sink): %v", err)
	}
	if calls == 0 {
		t.Fatal("sink never called")
	}
	if len(result.Tokens) > calls {
		t.Fatalf("result has %d tokens but sink only saw %d before stop", len(result.Tokens), calls)
	}
}

// TestAssistantGenerate_GenerateWithSink_RejectsRepeatPenalty covers the
// validate-config reject surfacing through the public entry: a repetition
// penalty request returns the named error before any forward runs.
func TestAssistantGenerate_GenerateWithSink_RejectsRepeatPenalty(t *testing.T) {
	m, pair := loadTinyGemma4AssistantRuntime(t)

	_, err := pair.GenerateWithSink(context.Background(), m, "hello", metal.GenerateConfig{MaxTokens: 4, RepeatPenalty: 1.5}, 2, nil)
	if err == nil || !core.Contains(err.Error(), "repetition penalty") {
		t.Fatalf("GenerateWithSink(repeat) = %v, want repetition penalty rejection", err)
	}
}

// TestAssistantGenerate_GenerateWithSink_NilPair covers the nil-pair guard: a
// nil pair (or one with nil target/assistant) cannot generate.
func TestAssistantGenerate_GenerateWithSink_NilPair(t *testing.T) {
	m, _ := loadTinyGemma4AssistantRuntime(t)

	var nilPair *Gemma4AssistantPair
	_, err := nilPair.GenerateWithSink(context.Background(), m, "hello", metal.GenerateConfig{MaxTokens: 4}, 2, nil)
	if err == nil || !core.Contains(err.Error(), "attached pair") {
		t.Fatalf("GenerateWithSink(nil pair) = %v, want attached-pair error", err)
	}
}

// TestAssistantGenerate_GenerateWithSink_TargetMismatch covers the guard that
// the runtime model handed to Generate must be the pair's own target: driving
// pair A's generation with runtime B (a different loaded target) is rejected.
func TestAssistantGenerate_GenerateWithSink_TargetMismatch(t *testing.T) {
	_, pairA := loadTinyGemma4AssistantRuntime(t)
	mB, _ := loadTinyGemma4AssistantRuntime(t)

	_, err := pairA.GenerateWithSink(context.Background(), mB, "hello", metal.GenerateConfig{MaxTokens: 4}, 2, nil)
	if err == nil || !core.Contains(err.Error(), "does not match target runtime") {
		t.Fatalf("GenerateWithSink(mismatched runtime) = %v, want target-runtime mismatch", err)
	}
}

// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma4

import (
	"context"
	"testing"

	"dappco.re/go/mlx/internal/metaltest"
	"dappco.re/go/mlx/pkg/metal"
)

// gemma4Turn wraps a user instruction in the Gemma 4 dialogue control tokens so
// the instruction-tuned model answers instead of immediately emitting EOS.
//
// The pair's Generate/GenerateWithSink encode the RAW prompt string
// (m.RuntimeTokenizer().Encode in generateGemma4Assistant) — chat templating is
// the serve layer's job, which these tests deliberately stay below. A bare
// un-wrapped string is out of distribution for the -it checkpoint and the very
// first sampled token is EOS, so appendGemma4AssistantToken stops with zero
// emitted tokens. The documented turn format (reference/gemma/
// core-prompt-formatting-gemma4.md): <|turn>user\n…<turn|>\n<|turn>model\n.
func gemma4Turn(instruction string) string {
	return "<|turn>user\n" + instruction + "<turn|>\n<|turn>model\n"
}

// liveGemma4E2BPair loads the real cached e2b target + bf16 assistant drafter
// ONCE and returns the metal.Model runtime plus the attached pair. The whole
// suite of live-generate coverage subtests shares this single load (the model
// is multi-GB; one load, many drivers) and the cleanup tears it down.
//
// The 6bit target / bf16 drafter combination is the one the existing live
// draft-step smoke proves loads — it avoids the gs=4 GatherQMM kernel that is
// genuinely absent from the shipped metallib. The prompt cache is enabled with
// a tiny chunk size so the real prompt-cache suffix restore AND the chunked
// prefill loop are both reachable from real (non-BOS-collapsing) tokenisation.
func liveGemma4E2BPair(t *testing.T, loadCfg metal.LoadConfig) (*metal.Model, *Gemma4AssistantPair) {
	t.Helper()
	if !metaltest.RunMetalTests {
		t.Skip("build with -tags metal_runtime to run the live e2b generate coverage")
	}
	if !metal.MetalAvailable() {
		t.Skip("Metal runtime unavailable")
	}
	targetPath := metaltest.HFModelPath(t, "mlx-community/gemma-4-e2b-it-6bit")
	assistantPath := metaltest.HFModelPath(t, "mlx-community/gemma-4-E2B-it-assistant-bf16")

	m, err := metal.LoadAndInit(targetPath, loadCfg)
	if err != nil {
		t.Fatalf("metal.LoadAndInit(real e2b target): %v", err)
	}
	pair, err := AttachGemma4Assistant(m, assistantPath)
	if err != nil {
		m.Close()
		t.Fatalf("AttachGemma4Assistant(real bf16 drafter): %v", err)
	}
	t.Cleanup(func() {
		pair.Close()
		m.Close()
	})
	return m, pair
}

// TestLiveGemma4E2B_Generate drives the real e2b pair through the full MTP
// generate path in several configurations from a single model load. Each
// subtest is a different decode regime so the speculative draft/verify loop
// (with genuine accepts AND rejects — tiny synthetic weights tend to all-accept
// or all-reject), the real tokenizer prefill, chunked prefill, real-quant
// attention/forward kernels, and FinalLogitSoftcapping are all driven by
// genuine weights.
//
// Assertions stay loose-but-meaningful: a real forward must emit tokens, never
// panic, suppressed tokens must stay out, and a fixed seed must reproduce — but
// no exact token identity is asserted across un-seeded runs (low-bit float
// order does not guarantee it). Enough to catch a gross regression or a real
// forward bug (the bug-fix mandate is live here) without flaking.
func TestLiveGemma4E2B_Generate(t *testing.T) {
	// Small prefill chunk so a short real prompt still exercises the chunked
	// prefill branch (prefillGemma4AssistantPrompt's loop).
	m, pair := liveGemma4E2BPair(t, metal.LoadConfig{PrefillChunkSize: 8})
	ctx := context.Background()
	off := false

	// Prompts chosen (from a live probe) to emit MULTI-token answers under the
	// turn template so the speculative loop runs several draft rounds with both
	// accepted and rejected tokens. The list prompt tokenises to clearly more
	// than the 8-token chunk size, taking the chunked prefill path.
	listPrompt := gemma4Turn("List the first five prime numbers separated by commas.")
	countPrompt := gemma4Turn("Count from one to ten.")
	oceanPrompt := gemma4Turn("Write one short sentence about the ocean.")

	t.Run("greedy", func(t *testing.T) {
		res, err := pair.Generate(ctx, m, listPrompt, metal.GenerateConfig{MaxTokens: 48, EnableThinking: &off}, 4)
		if err != nil {
			t.Fatalf("greedy Generate: %v", err)
		}
		if len(res.Tokens) == 0 {
			t.Fatal("greedy Generate emitted no tokens")
		}
		if res.PromptTokens <= 8 {
			t.Fatalf("prompt tokenised to %d tokens, want >8 to exercise chunked prefill", res.PromptTokens)
		}
	})

	t.Run("greedy_with_sink", func(t *testing.T) {
		var streamed int
		sink := func(metal.Token) bool {
			streamed++
			return true // keep going
		}
		res, err := pair.GenerateWithSink(ctx, m, countPrompt, metal.GenerateConfig{MaxTokens: 48, EnableThinking: &off}, 4, sink)
		if err != nil {
			t.Fatalf("greedy GenerateWithSink: %v", err)
		}
		if streamed == 0 {
			t.Fatal("sink received no tokens")
		}
		if len(res.Tokens) != streamed {
			t.Fatalf("sink saw %d tokens, result has %d", streamed, len(res.Tokens))
		}
	})

	t.Run("sink_early_stop", func(t *testing.T) {
		// Returning false from the sink requests an early stop on the first
		// emitted token — drives the sink-stop branch in the emit loop.
		var seen int
		sink := func(metal.Token) bool {
			seen++
			return false // stop after the first emitted token
		}
		if _, err := pair.GenerateWithSink(ctx, m, countPrompt, metal.GenerateConfig{MaxTokens: 48, EnableThinking: &off}, 4, sink); err != nil {
			t.Fatalf("sink_early_stop GenerateWithSink: %v", err)
		}
		if seen == 0 {
			t.Fatal("sink_early_stop saw no tokens")
		}
	})

	t.Run("temperature_seeded", func(t *testing.T) {
		// Temperature>0 with a fixed seed builds the keyed target sampler and the
		// private drafter uniform stream (draftStepSampled / draftBlockSampled /
		// verifyDraftBlockSampledClone). Two runs with the SAME seed must commit
		// the identical sequence (seed parity with plain decode).
		cfg := metal.GenerateConfig{MaxTokens: 32, Temperature: 0.7, TopK: 40, TopP: 0.95, Seed: 12345, SeedSet: true, EnableThinking: &off}
		a, err := pair.Generate(ctx, m, oceanPrompt, cfg, 4)
		if err != nil {
			t.Fatalf("temperature_seeded Generate A: %v", err)
		}
		b, err := pair.Generate(ctx, m, oceanPrompt, cfg, 4)
		if err != nil {
			t.Fatalf("temperature_seeded Generate B: %v", err)
		}
		if len(a.Tokens) == 0 {
			t.Fatal("temperature_seeded emitted no tokens")
		}
		if !equalTokenSlices(a.Tokens, b.Tokens) {
			t.Fatalf("seeded temperature runs diverged:\n A=%v\n B=%v", tokenIDs(a.Tokens), tokenIDs(b.Tokens))
		}
	})

	t.Run("suppress_tokens", func(t *testing.T) {
		// Suppressed tokens must never appear in the output — drives the
		// suppression branch through the speculative verify path. Suppress the
		// comma + space ids the prime-list answer uses so suppression is on the
		// model's preferred tokens (a no-op suppression would not exercise the
		// reject-on-suppressed branch).
		base, err := pair.Generate(ctx, m, listPrompt, metal.GenerateConfig{MaxTokens: 48, EnableThinking: &off}, 4)
		if err != nil {
			t.Fatalf("suppress baseline Generate: %v", err)
		}
		if len(base.Tokens) == 0 {
			t.Fatal("suppress baseline emitted no tokens")
		}
		suppressed := base.Tokens[0].ID
		cfg := metal.GenerateConfig{MaxTokens: 48, SuppressTokens: []int32{suppressed}, EnableThinking: &off}
		res, err := pair.Generate(ctx, m, listPrompt, cfg, 4)
		if err != nil {
			t.Fatalf("suppress_tokens Generate: %v", err)
		}
		for _, tok := range res.Tokens {
			if tok.ID == suppressed {
				t.Fatalf("suppressed token %d appeared in output %v", suppressed, tokenIDs(res.Tokens))
			}
		}
	})

	t.Run("draft_tokens_one", func(t *testing.T) {
		// draftTokens=1 is the minimal speculative block — a distinct schedule
		// from the 4-wide blocks above.
		res, err := pair.Generate(ctx, m, countPrompt, metal.GenerateConfig{MaxTokens: 32, EnableThinking: &off}, 1)
		if err != nil {
			t.Fatalf("draft_tokens_one Generate: %v", err)
		}
		if len(res.Tokens) == 0 {
			t.Fatal("draft_tokens_one emitted no tokens")
		}
	})
}

// TestLiveGemma4E2B_PromptCacheSuffix drives the real prompt-cache restore path
// (prefillGemma4AssistantFromPromptCache + the real suffix prefill) that the
// synthetic BOS-collapsing tokenizer cannot reach: a first generation primes the
// cache from genuine multi-token tokenisation, and a second on the SAME prompt
// restores it. It uses its OWN fresh load (isolated prompt-cache state) and
// asserts the PromptCacheHits counter strictly increments across exactly the two
// runs, so a silent no-op store is caught rather than passing on stale state.
func TestLiveGemma4E2B_PromptCacheSuffix(t *testing.T) {
	m, pair := liveGemma4E2BPair(t, metal.LoadConfig{PromptCacheMinTokens: 1})
	ctx := context.Background()
	off := false
	prompt := gemma4Turn("Name the four seasons of the year.")
	cfg := metal.GenerateConfig{MaxTokens: 24, EnableThinking: &off}

	if _, err := pair.Generate(ctx, m, prompt, cfg, 4); err != nil {
		t.Fatalf("prompt-cache prime Generate: %v", err)
	}
	primed := m.LastMetrics().PromptCacheHits
	if _, err := pair.Generate(ctx, m, prompt, cfg, 4); err != nil {
		t.Fatalf("prompt-cache restore Generate: %v", err)
	}
	restored := m.LastMetrics().PromptCacheHits
	if restored <= primed {
		t.Fatalf("prompt cache not hit on repeat prompt: hits primed=%d restored=%d", primed, restored)
	}
}

// TestLiveGemma4E2B_PromptCacheSuffixForward drives the suffix-forward arm of
// prefillGemma4AssistantFromPromptCache (the bulk the identical-prompt fast path
// skips): a prompt whose tokens strictly EXTEND a cached prompt restores the
// cached prefix and forwards only the new suffix tokens (prefixLen <
// len(tokens)). It uses its OWN fresh load so the cache holds only the two
// prompts under test — the small prompt-cache LRU otherwise lets an unrelated
// earlier entry win the prefix match and shrink the hit to the bare BOS.
//
// Raw (un-templated) prompts keep the prefix token-stable: "The quick brown" is
// a strict token prefix of "The quick brown fox jumps over". The suffix forward
// runs during prefill regardless of how many tokens generation then emits.
func TestLiveGemma4E2B_PromptCacheSuffixForward(t *testing.T) {
	m, pair := liveGemma4E2BPair(t, metal.LoadConfig{PromptCacheMinTokens: 1})
	ctx := context.Background()
	off := false
	cfg := metal.GenerateConfig{MaxTokens: 4, EnableThinking: &off}

	shortP := "The quick brown"
	longP := "The quick brown fox jumps over"
	short := m.RuntimeTokenizer().Encode(shortP)
	long := m.RuntimeTokenizer().Encode(longP)
	if len(short) >= len(long) {
		t.Fatalf("suffix setup: short (%d toks) not shorter than long (%d toks)", len(short), len(long))
	}
	for i := range short {
		if short[i] != long[i] {
			t.Skipf("tokenizer did not keep %q a strict token prefix of %q (diverged at %d)", shortP, longP, i)
		}
	}

	if _, err := pair.Generate(ctx, m, shortP, cfg, 4); err != nil {
		t.Fatalf("suffix prime Generate: %v", err)
	}
	if _, err := pair.Generate(ctx, m, longP, cfg, 4); err != nil {
		t.Fatalf("suffix-forward Generate: %v", err)
	}
	mm := m.LastMetrics()
	if mm.PromptCacheHits == 0 {
		t.Fatal("extended prompt did not hit the cached prefix (PromptCacheHits=0)")
	}
	if mm.PromptCacheHitTokens != len(short) {
		t.Fatalf("cache hit tokens = %d, want the %d-token shared prefix", mm.PromptCacheHitTokens, len(short))
	}
	if mm.PromptCacheHitTokens >= mm.PromptTokens {
		t.Fatalf("no suffix to forward: hitTokens=%d promptTokens=%d (suffix loop not taken)", mm.PromptCacheHitTokens, mm.PromptTokens)
	}
}

func equalTokenSlices(a, b []metal.Token) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i].ID != b[i].ID {
			return false
		}
	}
	return true
}

func tokenIDs(toks []metal.Token) []int32 {
	out := make([]int32, len(toks))
	for i, tok := range toks {
		out[i] = tok.ID
	}
	return out
}

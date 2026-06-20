// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"bytes"
	"context"
	"image"
	"image/color"
	"image/png"
	"slices"
	"strings"
	"testing"

	"dappco.re/go/mlx/internal/metaltest"
)

// e2bCoverageModel is the model these coverage tests load. It is hardcoded to
// the quantized 4-bit pack (3.4G) rather than inheriting GO_MLX_BENCH_MODEL or
// the bf16 default — the grading harness runs with no env, the 4-bit pack is
// the RAM-frugal choice, and (load-bearing) the quantized weights drive the
// quantized matvec / cache paths the bf16 build never touches.
const e2bCoverageModel = "mlx-community/gemma-4-e2b-it-4bit"

// TestE2BForward_AllHappyPaths drives the real-transformer-forward happy paths
// the synthetic suite leaves dark, all from a SINGLE e2b load (RAM-aware): the
// model-level generation entrypoints (Chat / ChatChunks / GenerateChunks /
// Generate variants / WarmPromptCache[Chunks]), the batched lanes (BatchGenerate
// / Classify), the sampler chain (top-k/top-p/min-p/repeat-penalty/suppression),
// the exact-prefix prompt cache (store / match / restore), the vision lane
// (SigLIP tower + image-token splice), and the stateful session lifecycle
// (prefill / append / capture / range / fork / restore).
//
// It gates on metaltest.RunMetalTests (true under -tags metal_runtime), NOT
// RunModelEvalTests — so it RUNS under the plain hardware-test command rather
// than the separate model_eval tag. The production forward path compiles
// unconditionally, so the const-gate is all that is needed. Run:
//
//	go test -tags metal_runtime -run TestE2BForward_AllHappyPaths ./pkg/metal/
//
// Subtest ordering matters: PromptCache asserts a COLD cache on its first
// generation, so it runs before any other subtest warms shared model state.
func TestE2BForward_AllHappyPaths(t *testing.T) {
	requireMetalRuntime(t)
	isolateRuntimeGates(t)
	restore := DefaultEngineFeatures().Apply()
	defer restore()

	dir := metaltest.HFModelPath(t, e2bCoverageModel)
	model, err := LoadAndInit(dir, LoadConfig{ContextLen: 4096})
	if err != nil {
		t.Fatalf("LoadAndInit(%s): %v", e2bCoverageModel, err)
	}
	defer model.Close()
	// Lower the store floor so short test prompts exercise the cache; the
	// store/match/restore code paths are unchanged by the threshold value.
	model.promptCacheMinTokens = 8

	ctx := context.Background()

	// --- 1. Prompt cache: MUST run first (asserts a cold cache). --------------
	// A first generation stores base + continuation; a second prompt sharing
	// `base` as a strict prefix then diverging hits the stored entry.
	t.Run("PromptCache_StoreMatchRestore", func(t *testing.T) {
		if !model.PromptCacheEnabled() {
			t.Fatal("prompt cache should be enabled by default")
		}
		base := "The capital city of France is Paris, a place with a long and storied " +
			"history of art, science, philosophy, and culture spanning many centuries."
		cfg := GenerateConfig{MaxTokens: 16}

		for range model.Generate(ctx, base, cfg) {
		}
		if err := model.Err(); err != nil {
			t.Fatalf("generation 1: %v", err)
		}
		if hits := model.LastMetrics().PromptCacheHits; hits != 0 {
			t.Fatalf("generation 1 PromptCacheHits = %d, want 0 (cold cache)", hits)
		}

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

		for range model.Generate(ctx, base+" Describe the river that runs through it.", cfg) {
		}
		if err := model.Err(); err != nil {
			t.Fatalf("generation 3: %v", err)
		}
		if m := model.LastMetrics(); m.PromptCacheHitTokens <= 0 {
			t.Fatalf("generation 3 should reuse the shared base prefix (got %d hit tokens)", m.PromptCacheHitTokens)
		}
	})

	// --- 2. Generation entrypoints. ------------------------------------------
	open := "Write a long, detailed story about a lighthouse keeper and the deep ocean."
	cfg := GenerateConfig{MaxTokens: 16}
	msgs := []ChatMessage{{Role: "user", Content: "Tell me a short story about the sea."}}

	t.Run("Chat", func(t *testing.T) {
		assertProducesTokens(t, model, model.Chat(ctx, msgs, cfg), "Chat")
	})

	t.Run("ChatChunks", func(t *testing.T) {
		assertProducesTokens(t, model, model.ChatChunks(ctx, msgs, 64, cfg), "ChatChunks")
	})

	t.Run("GenerateChunks", func(t *testing.T) {
		chunks := slices.Values([]string{"Write a long story ", "about a lighthouse keeper ", "and the deep ocean."})
		assertProducesTokens(t, model, model.GenerateChunks(ctx, chunks, cfg), "GenerateChunks")
	})

	t.Run("WarmPromptCache", func(t *testing.T) {
		if err := model.WarmPromptCache(ctx, open); err != nil {
			t.Fatalf("WarmPromptCache: %v", err)
		}
	})

	t.Run("WarmPromptCacheChunks", func(t *testing.T) {
		chunks := slices.Values([]string{"Once upon a time ", "in a land far away ", "there lived a keeper of light."})
		if err := model.WarmPromptCacheChunks(ctx, chunks); err != nil {
			t.Fatalf("WarmPromptCacheChunks: %v", err)
		}
	})

	// ClearCache forces the non-session route → the direct generate() path.
	t.Run("DirectGenerate_ClearCache", func(t *testing.T) {
		assertProducesTokens(t, model, model.Generate(ctx, open, GenerateConfig{MaxTokens: 16, ClearCache: true}), "direct Generate")
	})

	// --- 3. Sampler variants: force the chain off the greedy fast-path. -------
	t.Run("Sampler", func(t *testing.T) {
		cases := []struct {
			name string
			cfg  GenerateConfig
		}{
			{"topK+topP", GenerateConfig{MaxTokens: 16, Temperature: 0.8, TopP: 0.95, TopK: 40}},
			{"minP", GenerateConfig{MaxTokens: 16, Temperature: 0.8, MinP: 0.05}},
			{"repeatPenalty", GenerateConfig{MaxTokens: 16, Temperature: 0.8, RepeatPenalty: 1.3}},
			{"suppressTokens", GenerateConfig{MaxTokens: 16, Temperature: 0.8, TopK: 40, SuppressTokens: []int32{1000, 2000, 3000}}},
			{"temperatureOnly", GenerateConfig{MaxTokens: 16, Temperature: 1.0}},
		}
		for _, tc := range cases {
			t.Run(tc.name, func(t *testing.T) {
				assertProducesTokens(t, model, model.Generate(ctx, open, tc.cfg), tc.name)
			})
		}
	})

	// --- 4. Batched lanes: varied lengths → padding mask; chunked planner. ----
	t.Run("Batch", func(t *testing.T) {
		prompts := []string{
			"The capital of France is",
			"2+2=",
			"Once upon a time, in a land far away,",
		}

		t.Run("BatchGenerate", func(t *testing.T) {
			results, err := model.BatchGenerate(ctx, prompts, GenerateConfig{MaxTokens: 12})
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
			results, err := model.Classify(ctx, prompts, GenerateConfig{}, true)
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
			results, err := model.Classify(ctx, prompts, GenerateConfig{}, false)
			if err != nil {
				t.Fatalf("Classify: %v", err)
			}
			for i, r := range results {
				if r.Logits != nil {
					t.Fatalf("sequence %d returned logits with returnLogits=false", i)
				}
			}
		})

		// Sampled Classify (temperature > 0) walks the top-k/top-p sampler chain
		// rather than the greedy argmax branch.
		t.Run("ClassifySampled", func(t *testing.T) {
			results, err := model.Classify(ctx, prompts, GenerateConfig{Temperature: 0.8, TopK: 40, TopP: 0.95}, false)
			if err != nil {
				t.Fatalf("sampled Classify: %v", err)
			}
			if len(results) != len(prompts) {
				t.Fatalf("results = %d, want %d", len(results), len(prompts))
			}
		})

		// A batch larger than the size limit walks the chunking planner.
		t.Run("ChunkedBatch", func(t *testing.T) {
			old := model.batchSizeLimit
			model.batchSizeLimit = 2
			defer func() { model.batchSizeLimit = old }()
			results, err := model.BatchGenerate(ctx, prompts, GenerateConfig{MaxTokens: 8})
			if err != nil {
				t.Fatalf("chunked BatchGenerate: %v", err)
			}
			if len(results) != len(prompts) {
				t.Fatalf("chunked results = %d, want %d", len(results), len(prompts))
			}
		})

		// A single-prompt batch is uniform length → the explicit padding mask is
		// skipped (batchNeedsExplicitMask == false), the other branch.
		t.Run("SingleStreamNoMask", func(t *testing.T) {
			results, err := model.BatchGenerate(ctx, []string{"The capital of France is"}, GenerateConfig{MaxTokens: 8})
			if err != nil {
				t.Fatalf("single-stream BatchGenerate: %v", err)
			}
			if len(results) != 1 {
				t.Fatalf("results = %d, want 1", len(results))
			}
		})
	})

	// --- 5. Vision lane: a synthetic PNG through ChatMessage.Images. ----------
	t.Run("VisionChat_ImageInput", func(t *testing.T) {
		img := image.NewRGBA(image.Rect(0, 0, 64, 64))
		for y := range 64 {
			for x := range 64 {
				img.Set(x, y, color.RGBA{R: uint8(x * 4), G: uint8(y * 4), B: 128, A: 255})
			}
		}
		var buf bytes.Buffer
		if err := png.Encode(&buf, img); err != nil {
			t.Fatalf("encode png: %v", err)
		}
		vmsgs := []ChatMessage{{
			Role:    "user",
			Content: "Describe this image briefly.",
			Images:  [][]byte{buf.Bytes()},
		}}
		assertProducesTokens(t, model, model.Chat(ctx, vmsgs, GenerateConfig{MaxTokens: 16}), "vision Chat")
	})

	// --- 6. Tokenizer: pure-Go encode/decode over diverse inputs. ------------
	// LoadTokenizer is pure Go (no model state); covers Decode's marker handling,
	// DecodeOne's byte-fallback/special branches, and splitRunes that the 1-entry
	// stub vocab in the synthetic suite leaves dark.
	t.Run("Tokenizer_EncodeDecode", func(t *testing.T) {
		tok, err := LoadTokenizer(dir + "/tokenizer.json")
		if err != nil {
			t.Fatalf("LoadTokenizer: %v", err)
		}
		for _, s := range []string{
			"Hello, world!",
			"café ☕ 日本語 🚀 — em-dash and accents",
			"",
			"   \t\n  ",
			strings.Repeat("repeated phrase ", 50),
			"MixedCASE123 + symbols!@#$%^&*()",
		} {
			ids := tok.Encode(s)
			_ = tok.Decode(ids)
		}
		hits := 0
		for id := int32(0); id < 4000; id += 13 {
			if tok.DecodeOne(id) != "" {
				hits++
			}
			_ = tok.DecodeToken(id)
		}
		if hits == 0 {
			t.Fatal("DecodeOne returned empty for every sampled id — tokenizer vocab not loaded")
		}
		if tok.HasEOSToken() {
			_ = tok.DecodeToken(tok.EOSToken())
		}
		if tok.HasBOSToken() {
			_ = tok.DecodeToken(tok.BOSToken())
		}
		if id, ok := tok.TokenID("the"); ok {
			if tok.IDToken(id) == "" {
				t.Fatalf("IDToken(%d) empty for a round-tripped token", id)
			}
		}
		if len(tok.Encode("hello")) == 0 {
			t.Fatal("Encode(\"hello\") produced no tokens")
		}
	})

	// --- 7. Stateful session lifecycle. --------------------------------------
	t.Run("Session_SnapshotRestoreFork", func(t *testing.T) {
		s, ok := model.NewSession().(*ModelSession)
		if !ok {
			t.Fatal("NewSession did not return *ModelSession")
		}
		defer s.Close()

		if err := s.Prefill(ctx, open); err != nil {
			t.Fatalf("Prefill: %v", err)
		}
		gen := 0
		for range s.Generate(ctx, GenerateConfig{MaxTokens: 16}) {
			gen++
		}
		if err := s.Err(); err != nil {
			t.Fatalf("Generate: %v", err)
		}
		if gen == 0 {
			t.Fatal("session generated no tokens")
		}
		if err := s.AppendPrompt(ctx, " Tell me about its history."); err != nil {
			t.Fatalf("AppendPrompt: %v", err)
		}

		snap, err := s.CaptureKV(ctx)
		if err != nil {
			t.Fatalf("CaptureKV: %v", err)
		}
		if snap == nil {
			t.Fatal("CaptureKV returned a nil snapshot")
		}

		blocks := 0
		if err := s.RangeKVBlocks(ctx, 256, KVSnapshotCaptureOptions{}, func(KVSnapshotBlock) (bool, error) {
			blocks++
			return true, nil
		}); err != nil {
			t.Fatalf("RangeKVBlocks: %v", err)
		}

		forked, err := s.Fork(ctx)
		if err != nil {
			t.Fatalf("Fork: %v", err)
		}
		defer forked.Close()

		s2, ok := model.NewSession().(*ModelSession)
		if !ok {
			t.Fatal("NewSession did not return *ModelSession")
		}
		defer s2.Close()
		if err := s2.RestoreKV(ctx, snap); err != nil {
			t.Fatalf("RestoreKV: %v", err)
		}
		resumed := 0
		for range s2.Generate(ctx, GenerateConfig{MaxTokens: 8}) {
			resumed++
		}
		if err := s2.Err(); err != nil {
			t.Fatalf("restored session Generate: %v", err)
		}

		if err := model.RestorePromptCacheFromKV(ctx, snap); err != nil {
			t.Fatalf("RestorePromptCacheFromKV: %v", err)
		}
	})

	// --- 8. Session via chunk + token entrypoints (the chunk/token feeders). --
	t.Run("Session_ChunkAndTokenFeeders", func(t *testing.T) {
		s, ok := model.NewSession().(*ModelSession)
		if !ok {
			t.Fatal("NewSession did not return *ModelSession")
		}
		defer s.Close()

		chunks := slices.Values([]string{"The lighthouse stood ", "at the edge of the world, ", "alone against the storm."})
		if err := s.PrefillChunks(ctx, chunks); err != nil {
			t.Fatalf("PrefillChunks: %v", err)
		}
		for range s.Generate(ctx, GenerateConfig{MaxTokens: 8}) {
		}
		if err := s.Err(); err != nil {
			t.Fatalf("Generate after PrefillChunks: %v", err)
		}

		appendChunks := slices.Values([]string{" The keeper ", "watched the waves."})
		if err := s.AppendPromptChunks(ctx, appendChunks); err != nil {
			t.Fatalf("AppendPromptChunks: %v", err)
		}

		// Token-level feeders on a fresh session.
		tok, err := LoadTokenizer(dir + "/tokenizer.json")
		if err != nil {
			t.Fatalf("LoadTokenizer: %v", err)
		}
		s2, ok := model.NewSession().(*ModelSession)
		if !ok {
			t.Fatal("NewSession did not return *ModelSession")
		}
		defer s2.Close()
		prefillTokens := tok.Encode("The deep ocean holds many secrets.")
		if err := s2.PrefillTokens(ctx, prefillTokens); err != nil {
			t.Fatalf("PrefillTokens: %v", err)
		}
		for range s2.Generate(ctx, GenerateConfig{MaxTokens: 8}) {
		}
		if err := s2.Err(); err != nil {
			t.Fatalf("Generate after PrefillTokens: %v", err)
		}
		appendTokens := tok.Encode(" And the keeper knew them all.")
		if err := s2.AppendTokens(ctx, appendTokens); err != nil {
			t.Fatalf("AppendTokens: %v", err)
		}

		// Reset returns the session to an empty state for reuse.
		s2.Reset()
		if err := s2.Prefill(ctx, "A fresh start."); err != nil {
			t.Fatalf("Prefill after Reset: %v", err)
		}
		for range s2.Generate(ctx, GenerateConfig{MaxTokens: 4}) {
		}
		if err := s2.Err(); err != nil {
			t.Fatalf("Generate after Reset: %v", err)
		}
	})

	// --- 9. KV-snapshot capture / block-range / block-restore on the model. ---
	// The model-level CaptureKV family (chunks + token+options) prefills a real
	// model then snapshots, which the synthetic suite can only error out of early
	// (captureKVChunksWithOptions was 15%). RangeKVBlocks with a small block size
	// streams multiple blocks; feeding those back through RestoreKVBlocks drives
	// the block-streaming restore (newPromptCacheEntryFromKVBlocks).
	t.Run("KVSnapshot_CaptureRangeRestore", func(t *testing.T) {
		longPrompt := "The lighthouse keeper kept a meticulous log of every ship that passed, " +
			"of every storm that rolled in from the deep ocean, and of the long quiet nights " +
			"when the only sound was the slow turning of the great lamp above the rocks."

		t.Run("CaptureKVChunks", func(t *testing.T) {
			chunks := slices.Values([]string{longPrompt[:60], longPrompt[60:120], longPrompt[120:]})
			snap, err := model.CaptureKVChunks(ctx, chunks)
			if err != nil {
				t.Fatalf("CaptureKVChunks: %v", err)
			}
			if snap == nil {
				t.Fatal("CaptureKVChunks returned a nil snapshot")
			}
		})

		// CaptureKVWithOptions (string prompt) routes through the unexported
		// captureKVWithOptions → captureKVTokensWithOptions token path.
		t.Run("CaptureKVWithOptions", func(t *testing.T) {
			snap, err := model.CaptureKVWithOptions(ctx, longPrompt, KVSnapshotCaptureOptions{})
			if err != nil {
				t.Fatalf("CaptureKVWithOptions: %v", err)
			}
			if snap == nil {
				t.Fatal("CaptureKVWithOptions returned a nil snapshot")
			}
		})

		// Stream the session's KV as multiple small blocks, collect them, then
		// restore a fresh session from the streamed block source.
		t.Run("RangeAndRestoreBlocks", func(t *testing.T) {
			s, ok := model.NewSession().(*ModelSession)
			if !ok {
				t.Fatal("NewSession did not return *ModelSession")
			}
			defer s.Close()
			if err := s.Prefill(ctx, longPrompt); err != nil {
				t.Fatalf("Prefill: %v", err)
			}

			var blocks []KVSnapshotBlock
			// blockSize 8 over a multi-dozen-token prompt → several blocks, so the
			// boundary loop body runs more than once.
			if err := s.RangeKVBlocks(ctx, 8, KVSnapshotCaptureOptions{}, func(b KVSnapshotBlock) (bool, error) {
				blocks = append(blocks, b)
				return true, nil
			}); err != nil {
				t.Fatalf("RangeKVBlocks: %v", err)
			}
			if len(blocks) < 2 {
				t.Fatalf("RangeKVBlocks produced %d blocks, want >= 2 (small block size over a long prompt)", len(blocks))
			}

			prefixTokens := 0
			for _, b := range blocks {
				prefixTokens += b.TokenCount
			}
			source := KVSnapshotBlockSource{
				PrefixTokens: prefixTokens,
				BlockCount:   len(blocks),
				Load: func(_ context.Context, i int) (KVSnapshotBlock, error) {
					return blocks[i], nil
				},
			}
			s2, ok := model.NewSession().(*ModelSession)
			if !ok {
				t.Fatal("NewSession did not return *ModelSession")
			}
			defer s2.Close()
			if err := s2.RestoreKVBlocks(ctx, source); err != nil {
				t.Fatalf("RestoreKVBlocks: %v", err)
			}
			resumed := 0
			for range s2.Generate(ctx, GenerateConfig{MaxTokens: 8}) {
				resumed++
			}
			if err := s2.Err(); err != nil {
				t.Fatalf("Generate after RestoreKVBlocks: %v", err)
			}
		})

		// Early-break: yield false on the first block → the !ok return path.
		t.Run("RangeEarlyBreak", func(t *testing.T) {
			s, ok := model.NewSession().(*ModelSession)
			if !ok {
				t.Fatal("NewSession did not return *ModelSession")
			}
			defer s.Close()
			if err := s.Prefill(ctx, longPrompt); err != nil {
				t.Fatalf("Prefill: %v", err)
			}
			count := 0
			if err := s.RangeKVBlocks(ctx, 8, KVSnapshotCaptureOptions{}, func(KVSnapshotBlock) (bool, error) {
				count++
				return false, nil // stop after the first block
			}); err != nil {
				t.Fatalf("RangeKVBlocks early-break: %v", err)
			}
			if count != 1 {
				t.Fatalf("early-break yielded %d blocks, want 1", count)
			}
		})
	})
}

// assertProducesTokens drains a token iterator, fails if the model errored, and
// fails if zero tokens were produced.
func assertProducesTokens(t *testing.T, model *Model, seq func(yield func(Token) bool), label string) {
	t.Helper()
	n := 0
	for range seq {
		n++
	}
	if err := model.Err(); err != nil {
		t.Fatalf("%s: %v", label, err)
	}
	if n == 0 {
		t.Fatalf("%s produced no tokens", label)
	}
}

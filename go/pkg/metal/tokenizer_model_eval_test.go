// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"strings"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/mlx/internal/metaltest"
)

// TestTokenizer_EncodeDecodeVariants_Eval exercises the real SentencePiece
// tokenizer loaded from the e2b snapshot across diverse inputs and a spread of
// token ids — the synthetic suite uses a 1-entry stub vocab, so Decode's marker
// handling, DecodeOne's byte-fallback/special branches, and splitRunes are dark.
// No model load (LoadTokenizer is pure Go), so no gate isolation needed. Run:
//
//	GO_MLX_BENCH_MODEL=google/gemma-4-e2b-it go test \
//	  -tags 'metal_runtime model_eval' -run TestTokenizer_EncodeDecodeVariants_Eval ./pkg/metal/
//
// (encodeGPT2* stays uncovered here — gemma4 is SentencePiece, not GPT2 BPE.)
func TestTokenizer_EncodeDecodeVariants_Eval(t *testing.T) {
	if !metaltest.RunModelEvalTests {
		t.Skip("model-eval test; build with -tags model_eval")
	}
	repo := core.Getenv("GO_MLX_BENCH_MODEL")
	if repo == "" {
		repo = "google/gemma-4-e2b-it"
	}
	dir := metaltest.HFModelPath(t, repo)
	tok, err := LoadTokenizer(dir + "/tokenizer.json")
	if err != nil {
		t.Fatalf("LoadTokenizer: %v", err)
	}

	// Encode/decode round-trip over diverse inputs (ascii, unicode, emoji,
	// empty, whitespace, repetition, mixed case + symbols).
	for _, s := range []string{
		"Hello, world!",
		"café ☕ 日本語 🚀 — em-dash and accents",
		"",
		"   \t\n  ",
		strings.Repeat("repeated phrase ", 50),
		"MixedCASE123 + symbols!@#$%^&*()",
	} {
		ids := tok.Encode(s)
		_ = tok.Decode(ids) // Decode + splitRunes + SentencePiece marker handling
	}

	// A spread of token ids exercises DecodeOne's byte-fallback / special-token /
	// leading-marker branches and DecodeToken.
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

	// Special tokens + TokenID/IDToken round-trip.
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
}

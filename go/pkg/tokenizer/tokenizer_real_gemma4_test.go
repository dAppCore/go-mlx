// SPDX-Licence-Identifier: EUPL-1.2

package tokenizer

import (
	"testing"

	core "dappco.re/go"

	"dappco.re/go/mlx/internal/metaltest"
)

// realGemma4TokenizerPath resolves the tokenizer.json shipped inside the real
// google/gemma-4-e2b-it Hugging Face snapshot — a 262,144-entry SentencePiece-
// style BPE vocab with 514,906 merges, not a hand-typed miniature fixture.
// Skips (via metaltest.HFModelPath) when the model is not in the local HF
// cache, so a checkout without the weights stays green.
func realGemma4TokenizerPath(t *testing.T) string {
	t.Helper()
	dir := metaltest.HFModelPath(t, "google/gemma-4-e2b-it")
	return core.PathJoin(dir, "tokenizer.json")
}

// TestTokenizer_RealGemma4_LoadAndSpecialTokens_Good loads the actual
// tokenizer.json shipped with google/gemma-4-e2b-it and pins the real special
// token ids baked into that file — not a hand-typed miniature vocab built to
// already agree with the code. <bos>=2 and <eos>=1 are added_tokens 1/2;
// <turn|>=106 is the Gemma-4 turn-end token. The real added_tokens list has no
// <end_of_turn>/<|im_start|>/<|im_end|>/<|eot_id|>/<|begin_of_text|> (those
// are Gemma-3/Qwen3/Llama-3 markers absent from Gemma 4), so the only override
// after <eos> is <turn|> — if LoadTokenizer's precedence chain or the shipped
// ids ever drifted, this fails against the real artefact.
func TestTokenizer_RealGemma4_LoadAndSpecialTokens_Good(t *testing.T) {
	tok, err := LoadTokenizer(realGemma4TokenizerPath(t))
	if err != nil {
		t.Fatalf("LoadTokenizer(real gemma-4 tokenizer.json): %v", err)
	}
	if tok.BOSToken() != 2 {
		t.Errorf("BOSToken() = %d, want 2 (<bos>, real tokenizer.json)", tok.BOSToken())
	}
	if !tok.HasEOSToken() || tok.EOSToken() != 106 {
		t.Errorf("EOSToken() = %d (has=%t), want 106 (<turn|> overrides <eos>=1)", tok.EOSToken(), tok.HasEOSToken())
	}
	// Channel markers resolve to their real vocab ids.
	if id, ok := tok.TokenID("<|channel>"); !ok || id != 100 {
		t.Errorf("TokenID(<|channel>) = (%d,%t), want (100,true)", id, ok)
	}
	if id, ok := tok.TokenID("<channel|>"); !ok || id != 101 {
		t.Errorf("TokenID(<channel|>) = (%d,%t), want (101,true)", id, ok)
	}
	if id, ok := tok.TokenID("<|turn>"); !ok || id != 105 {
		t.Errorf("TokenID(<|turn>) = (%d,%t), want (105,true)", id, ok)
	}
}

// TestTokenizer_RealGemma4_EncodePinsIndependentOracle_Good encodes a plain
// sentence with the real gemma-4 tokenizer.json and pins the exact token ids
// produced by an INDEPENDENT oracle: HuggingFace's own `tokenizers` Python
// library loading the identical file
// (Tokenizer.from_file(tokenizer.json).encode("Hello world",
// add_special_tokens=False).ids == [9259, 1902] — "Hello"=9259, "▁world"=1902
// in the real vocab). A round-trip-only assertion (Decode(Encode(x)) == x)
// would still pass if Encode and Decode shared a matching bug — e.g. both
// silently using the wrong merge precedence — so this pins the absolute ids
// against a second, independent implementation of the same tokenizer.json
// rather than checking this package's output against itself.
func TestTokenizer_RealGemma4_EncodePinsIndependentOracle_Good(t *testing.T) {
	tok, err := LoadTokenizer(realGemma4TokenizerPath(t))
	if err != nil {
		t.Fatalf("LoadTokenizer: %v", err)
	}

	got := tok.Encode("Hello world")
	want := []int32{2, 9259, 1902} // BOS + "Hello" + "▁world", per the tokenizers-library oracle
	if len(got) != len(want) {
		t.Fatalf("Encode(\"Hello world\") = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("Encode(\"Hello world\")[%d] = %d, want %d (full: %v)", i, got[i], want[i], got)
		}
	}

	if dec := tok.Decode(got); dec != "Hello world" {
		t.Errorf("Decode(Encode(\"Hello world\")) = %q, want %q", dec, "Hello world")
	}
}

// TestTokenizer_RealGemma4_ChannelMarkersSurviveDecodeToken_Good pins the
// content-bearing reasoning-channel delimiters through DecodeToken using
// their REAL ids from the shipped tokenizer.json (100/101), not synthetic
// stand-ins. go-inference's ThinkingExtractor depends on these exact literal
// marker strings surviving DecodeToken so it can split the thinking span from
// the visible reply. <|turn> (real id 105) is pure framing, not content, and
// is intentionally swallowed to "" by DecodeToken like any other special —
// unlike the channel pair — so both branches of that contract are pinned
// against the real vocab here.
func TestTokenizer_RealGemma4_ChannelMarkersSurviveDecodeToken_Good(t *testing.T) {
	tok, err := LoadTokenizer(realGemma4TokenizerPath(t))
	if err != nil {
		t.Fatalf("LoadTokenizer: %v", err)
	}

	openID, ok := tok.TokenID("<|channel>")
	if !ok {
		t.Fatal("<|channel> not in real vocab")
	}
	closeID, ok := tok.TokenID("<channel|>")
	if !ok {
		t.Fatal("<channel|> not in real vocab")
	}
	if got := tok.DecodeToken(openID); got != "<|channel>" {
		t.Errorf("DecodeToken(<|channel> id=%d) = %q, want literal %q", openID, got, "<|channel>")
	}
	if got := tok.DecodeToken(closeID); got != "<channel|>" {
		t.Errorf("DecodeToken(<channel|> id=%d) = %q, want literal %q", closeID, got, "<channel|>")
	}

	// <|turn> is framing, not content — DecodeToken swallows it like any other
	// non-channel special (empty string), unlike the channel pair above.
	turnOpenID, ok := tok.TokenID("<|turn>")
	if !ok {
		t.Fatal("<|turn> not in real vocab")
	}
	if got := tok.DecodeToken(turnOpenID); got != "" {
		t.Errorf("DecodeToken(<|turn> id=%d) = %q, want empty (framing token, not content)", turnOpenID, got)
	}
}

// TestTokenizer_RealGemma4_ChatFramingRoundTrip_Good encodes a full
// <|turn>...<turn|> chat-framed segment through the real tokenizer and
// verifies the framing ids match the tokenizers-library oracle
// (Tokenizer.from_file(...).encode("<bos><|turn>user\nHello
// world<turn|>\n", add_special_tokens=False).ids ==
// [2, 105, 2364, 107, 9259, 1902, 106, 107]), then confirms Decode strips the
// BOS/turn control tokens while the literal "user" content and newlines
// survive — the framing shape the chat template actually emits.
func TestTokenizer_RealGemma4_ChatFramingRoundTrip_Good(t *testing.T) {
	tok, err := LoadTokenizer(realGemma4TokenizerPath(t))
	if err != nil {
		t.Fatalf("LoadTokenizer: %v", err)
	}

	got := tok.Encode("<bos><|turn>user\nHello world<turn|>\n")
	want := []int32{2, 105, 2364, 107, 9259, 1902, 106, 107}
	if len(got) != len(want) {
		t.Fatalf("Encode(chat) = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("Encode(chat)[%d] = %d, want %d (full: %v)", i, got[i], want[i], got)
		}
	}

	if dec := tok.Decode(got); dec != "user\nHello world\n" {
		t.Errorf("Decode(chat) = %q, want %q (BOS/turn control tokens stripped, content survives)", dec, "user\nHello world\n")
	}
}

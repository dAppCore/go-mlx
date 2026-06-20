// SPDX-Licence-Identifier: EUPL-1.2

package tokenizer

import (
	"testing"

	"dappco.re/go"

	coreio "dappco.re/go/io"
)

// --- indexIn empty-substring fast path ---

// TestTokenizer_IndexIn_EmptySubstr_Good drives the empty-needle branch of
// indexIn directly (subLen == 0 → return 0). The encode paths never pass an
// empty special token, so this leaf was otherwise cold. Also pins the two
// non-empty outcomes so the helper's contract is locked end to end.
func TestTokenizer_IndexIn_EmptySubstr_Good(t *testing.T) {
	if got := indexIn("hello world", ""); got != 0 {
		t.Errorf("indexIn(_, \"\") = %d, want 0 (empty needle)", got)
	}
	if got := indexIn("hello world", "world"); got != 6 {
		t.Errorf("indexIn = %d, want 6", got)
	}
	if got := indexIn("hello", "xyz"); got != -1 {
		t.Errorf("indexIn(no match) = %d, want -1", got)
	}
	// Needle longer than haystack short-circuits before the scan loop.
	if got := indexIn("hi", "hiya"); got != -1 {
		t.Errorf("indexIn(needle>haystack) = %d, want -1", got)
	}
}

// --- LoadTokenizer: vocab value not an integer ---

// TestTokenizer_LoadTokenizer_NonIntVocabValue_Bad covers the vocab
// re-unmarshal failure branch in LoadTokenizer. A vocab whose value is a
// string re-marshals fine (so the re-encode branch is NOT hit) but fails to
// unmarshal into map[string]int32, so LoadTokenizer must surface a parse-vocab
// error rather than panicking or silently loading an empty vocab.
func TestTokenizer_LoadTokenizer_NonIntVocabValue_Bad(t *testing.T) {
	const badVocabJSON = `{
	  "model": {"type": "BPE", "vocab": {"a": "notanumber"}, "merges": []},
	  "added_tokens": []
	}`
	dir := t.TempDir()
	path := core.JoinPath(dir, "tokenizer.json")
	if err := coreio.Local.Write(path, badVocabJSON); err != nil {
		t.Fatalf("write: %v", err)
	}
	tok, err := LoadTokenizer(path)
	if err == nil {
		t.Fatalf("expected parse-vocab error for non-integer vocab value, got tok=%v", tok)
	}
	if tok != nil {
		t.Errorf("tokenizer = %v, want nil on vocab parse failure", tok)
	}
}

// --- empty-segment guards on the encode helpers ---

// TestTokenizer_EncodeSentencePieceSegment_Empty_Good hits the segment == ""
// guard of encodeSentencePieceSegment directly. Encode never feeds the helper
// an empty segment (the for-loop only runs while remaining != ""), so the guard
// was cold. It must return nil with no cache touch.
func TestTokenizer_EncodeSentencePieceSegment_Empty_Good(t *testing.T) {
	tok := &Tokenizer{vocab: map[string]int32{}, mergeRanks: map[string]int{}}
	if got := tok.encodeSentencePieceSegment(""); got != nil {
		t.Errorf("encodeSentencePieceSegment(\"\") = %v, want nil", got)
	}
	if len(tok.bpeCache) != 0 {
		t.Errorf("empty segment populated cache: %d entries, want 0", len(tok.bpeCache))
	}
}

// TestTokenizer_SentencePieceCacheKey_Empty_Good hits the segment == "" guard
// of sentencePieceCacheKey, which returns two empty strings without touching
// the scratch pool. Callers gate on segment != "" so this leaf was otherwise
// unreached.
func TestTokenizer_SentencePieceCacheKey_Empty_Good(t *testing.T) {
	key, spText := sentencePieceCacheKey("")
	if key != "" || spText != "" {
		t.Errorf("sentencePieceCacheKey(\"\") = (%q, %q), want two empty strings", key, spText)
	}
}

// TestTokenizer_EncodeGPT2Segment_Empty_Good hits the segment == "" guard of
// encodeGPT2Segment directly (the for-loop in encodeGPT2 never passes ""). It
// must return nil before any byte-encoding or cache work.
func TestTokenizer_EncodeGPT2Segment_Empty_Good(t *testing.T) {
	tok := gpt2Fixture()
	if got := tok.encodeGPT2Segment(""); got != nil {
		t.Errorf("encodeGPT2Segment(\"\") = %v, want nil", got)
	}
}

// TestTokenizer_EncodeGPT2Segment_AllUnmapped_Good covers the second early
// return of encodeGPT2Segment: a NON-empty segment whose every byte is absent
// from gpt2Encoder produces an empty byte-encoded key (len == prefix), so the
// `empty` branch returns nil before probing or writing the cache. Production's
// full 256-entry encoder cannot reach this, so the test uses a deliberately
// sparse encoder (only 'a' mapped) and feeds bytes that are not 'a'.
func TestTokenizer_EncodeGPT2Segment_AllUnmapped_Good(t *testing.T) {
	tok := &Tokenizer{
		vocab:       map[string]int32{},
		mergeRanks:  map[string]int{},
		isGPT2BPE:   true,
		gpt2Encoder: map[byte]rune{'a': 'a'}, // 'b'/'c' are NOT in the map
	}
	if got := tok.encodeGPT2Segment("bc"); got != nil {
		t.Errorf("encodeGPT2Segment(all-unmapped) = %v, want nil", got)
	}
	if len(tok.bpeCache) != 0 {
		t.Errorf("all-unmapped segment populated cache: %d entries, want 0", len(tok.bpeCache))
	}
}

// --- Encode → encodeGPT2 dispatch + the GPT-2 BOS prepend ---

// gpt2WithBOSFixture is gpt2Fixture plus an explicit BOS token so the
// shouldPrependBOS branch of encodeGPT2 fires (the bare gpt2Fixture has no BOS,
// leaving the prepend cold). BOS text "<s>" is not a prefix of the test inputs,
// so the prepend always triggers.
func gpt2WithBOSFixture() *Tokenizer {
	tok := gpt2Fixture()
	tok.bosToken = 1
	tok.hasBOS = true
	tok.invVocab[1] = "<s>"
	return tok
}

// TestTokenizer_Encode_GPT2Dispatch_Good covers the isGPT2BPE branch of the
// public Encode (existing GPT-2 tests call encodeGPT2 directly and never reach
// Encode's dispatch). Encode on a GPT-2 tokenizer must equal encodeGPT2 on the
// same input.
func TestTokenizer_Encode_GPT2Dispatch_Good(t *testing.T) {
	for _, text := range []string{"hello", "hello hello", "he"} {
		direct := gpt2Fixture()
		want := append([]int32(nil), direct.encodeGPT2(text)...)
		via := gpt2Fixture()
		got := via.Encode(text)
		if len(got) != len(want) {
			t.Fatalf("Encode(%q)=%v, encodeGPT2=%v (len)", text, got, want)
		}
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("Encode(%q) mismatch at %d: %v vs %v", text, i, got, want)
			}
		}
	}
}

// TestTokenizer_EncodeGPT2_PrependsBOS_Good covers the shouldPrependBOS true
// branch of encodeGPT2: with a BOS defined and absent from the input, the first
// emitted token must be the BOS id, followed by the same content tokens the
// no-BOS fixture produces.
func TestTokenizer_EncodeGPT2_PrependsBOS_Good(t *testing.T) {
	withBOS := gpt2WithBOSFixture()
	got := withBOS.encodeGPT2("hello")
	if len(got) == 0 {
		t.Fatal("encodeGPT2 returned empty, want BOS + content")
	}
	if got[0] != withBOS.BOSToken() {
		t.Fatalf("first token = %d, want BOS (%d)", got[0], withBOS.BOSToken())
	}

	// The tail (after BOS) must match the no-BOS fixture's full encoding.
	noBOS := gpt2Fixture()
	want := noBOS.encodeGPT2("hello")
	tail := got[1:]
	if len(tail) != len(want) {
		t.Fatalf("content tokens = %v, want %v (len)", tail, want)
	}
	for i := range want {
		if tail[i] != want[i] {
			t.Fatalf("content token mismatch at %d: %v vs %v", i, tail, want)
		}
	}
}

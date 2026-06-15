// SPDX-Licence-Identifier: EUPL-1.2

package tokenizer

import (
	"testing"

	"dappco.re/go"

	coreio "dappco.re/go/io"
)

// minimalTokenizerJSON is a valid HuggingFace tokenizer.json with a tiny vocab.
const minimalTokenizerJSON = `{
  "model": {
    "type": "BPE",
    "vocab": {
      "h": 0,
      "e": 1,
      "l": 2,
      "o": 3,
      "▁": 4,
      "he": 5,
      "ll": 6,
      "▁h": 7
    },
    "merges": ["h e", "l l"],
    "byte_fallback": false
  },
  "added_tokens": [
    {"id": 100, "content": "<bos>", "special": true},
    {"id": 101, "content": "<eos>", "special": true}
  ]
}`

const tokenizerWithoutSpecialsJSON = `{
  "model": {
    "type": "BPE",
    "vocab": {
      "h": 0,
      "e": 1,
      "l": 2,
      "o": 3,
      "▁": 4,
      "he": 5,
      "ll": 6
    },
    "merges": ["h e", "l l"],
    "byte_fallback": false
  },
  "added_tokens": []
}`

func writeTestTokenizer(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	path := core.JoinPath(dir, "tokenizer.json")
	if err := coreio.Local.Write(path, minimalTokenizerJSON); err != nil {
		t.Fatalf("write test tokenizer: %v", err)
	}
	return path
}

func writeTokenizerWithoutSpecials(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	path := core.JoinPath(dir, "tokenizer.json")
	if err := coreio.Local.Write(path, tokenizerWithoutSpecialsJSON); err != nil {
		t.Fatalf("write tokenizer without specials: %v", err)
	}
	return path
}

func TestTokenizer_LoadTokenizer_Good(t *testing.T) {
	path := writeTestTokenizer(t)
	tok, err := LoadTokenizer(path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if tok == nil {
		t.Fatal("tokenizer is nil")
	}
}

func TestTokenizer_LoadTokenizer_MissingFile_Bad(t *testing.T) {
	_, err := LoadTokenizer("/nonexistent/tokenizer.json")
	if err == nil {
		t.Error("expected error for missing file")
	}
}

func TestTokenizer_LoadTokenizer_InvalidJSON_Ugly(t *testing.T) {
	dir := t.TempDir()
	path := core.JoinPath(dir, "tokenizer.json")
	_ = coreio.Local.Write(path, "not json")

	_, err := LoadTokenizer(path)
	if err == nil {
		t.Error("expected error for invalid JSON")
	}
}

func TestTokenizer_BOSEOS_Good(t *testing.T) {
	path := writeTestTokenizer(t)
	tok, _ := LoadTokenizer(path)

	if tok.BOSToken() != 100 {
		t.Errorf("BOS = %d, want 100", tok.BOSToken())
	}
	if tok.EOSToken() != 101 {
		t.Errorf("EOS = %d, want 101", tok.EOSToken())
	}
}

func TestTokenizer_Lookups_Good(t *testing.T) {
	path := writeTestTokenizer(t)
	tok, _ := LoadTokenizer(path)

	if tok.BOS() != 100 {
		t.Fatalf("BOS() = %d, want 100", tok.BOS())
	}
	if tok.EOS() != 101 {
		t.Fatalf("EOS() = %d, want 101", tok.EOS())
	}
	id, ok := tok.TokenID("he")
	if !ok || id != 5 {
		t.Fatalf("TokenID(\"he\") = (%d, %t), want (5, true)", id, ok)
	}
	if tok.IDToken(6) != "ll" {
		t.Fatalf("IDToken(6) = %q, want %q", tok.IDToken(6), "ll")
	}
}

func TestTokenizer_NoSpecialTokens_DoesNotInventBOSOrEOS_Good(t *testing.T) {
	path := writeTokenizerWithoutSpecials(t)
	tok, err := LoadTokenizer(path)
	if err != nil {
		t.Fatalf("LoadTokenizer: %v", err)
	}

	if tok.HasBOSToken() {
		t.Fatal("HasBOSToken() = true, want false")
	}
	if tok.HasEOSToken() {
		t.Fatal("HasEOSToken() = true, want false")
	}
	if tok.BOSToken() != 0 {
		t.Fatalf("BOSToken() = %d, want 0 zero value", tok.BOSToken())
	}
	if tok.EOSToken() != 0 {
		t.Fatalf("EOSToken() = %d, want 0 zero value", tok.EOSToken())
	}

	tokens := tok.Encode("hello")
	want := []int32{4, 5, 6, 3}
	if len(tokens) != len(want) {
		t.Fatalf("Encode(\"hello\") = %v, want %v", tokens, want)
	}
	for i := range want {
		if tokens[i] != want[i] {
			t.Fatalf("tokens[%d] = %d, want %d", i, tokens[i], want[i])
		}
	}
}

func TestTokenizer_Encode_Good(t *testing.T) {
	path := writeTestTokenizer(t)
	tok, _ := LoadTokenizer(path)

	tokens := tok.Encode("hello")
	if len(tokens) == 0 {
		t.Fatal("Encode returned empty tokens")
	}
	// First token should be BOS
	if tokens[0] != tok.BOSToken() {
		t.Errorf("first token = %d, want BOS (%d)", tokens[0], tok.BOSToken())
	}
	// With BPE merges ("h e" → "he", "l l" → "ll"), "hello" with ▁ prefix becomes:
	// "▁" "h" "e" "l" "l" "o" → merge "h e" → "▁" "he" "l" "l" "o"
	// → merge "l l" → "▁" "he" "ll" "o"
	// No further merges. But "▁" is not "▁h" so it stays as "▁".
	// Vocab: ▁=4, he=5, ll=6, o=3. Expected: [BOS, 4, 5, 6, 3]
	want := []int32{100, 4, 5, 6, 3}
	if len(tokens) != len(want) {
		t.Fatalf("Encode(\"hello\") = %v, want %v", tokens, want)
	}
	for i := range tokens {
		if tokens[i] != want[i] {
			t.Errorf("tokens[%d] = %d, want %d", i, tokens[i], want[i])
		}
	}
}

func TestTokenizer_EncodeExplicitBOSDoesNotDuplicate_Good(t *testing.T) {
	path := writeTestTokenizer(t)
	tok, _ := LoadTokenizer(path)

	tokens := tok.Encode("<bos>hello")
	if len(tokens) < 2 {
		t.Fatalf("Encode explicit BOS = %v, want BOS plus content", tokens)
	}
	if tokens[0] != tok.BOSToken() {
		t.Fatalf("first token = %d, want BOS (%d)", tokens[0], tok.BOSToken())
	}
	if tokens[1] == tok.BOSToken() {
		t.Fatalf("Encode duplicated explicit BOS: %v", tokens)
	}
}

func TestTokenizer_Encode_MultiWordSentencePiece_Good(t *testing.T) {
	path := writeTestTokenizer(t)
	tok, _ := LoadTokenizer(path)

	tokens := tok.Encode("hello hello")
	want := []int32{100, 4, 5, 6, 3, 4, 5, 6, 3}
	if len(tokens) != len(want) {
		t.Fatalf("Encode(\"hello hello\") = %v, want %v", tokens, want)
	}
	for i := range want {
		if tokens[i] != want[i] {
			t.Fatalf("tokens[%d] = %d, want %d", i, tokens[i], want[i])
		}
	}

	if decoded := tok.Decode(tokens); decoded != "hello hello" {
		t.Fatalf("Decode(Encode(\"hello hello\")) = %q, want %q", decoded, "hello hello")
	}
}

func TestTokenizer_BPEMerge_Good(t *testing.T) {
	tok := &Tokenizer{
		mergeRanks: map[string]int{
			"h e":  0,
			"l l":  1,
			"he l": 2,
		},
	}

	// "h" "e" "l" "l" "o" → merge "h e" (rank 0) → "he" "l" "l" "o"
	// → merge "l l" (rank 1) → "he" "ll" "o"
	// → merge "he l" does NOT match "he ll" — stops here.
	symbols := []string{"h", "e", "l", "l", "o"}
	got := tok.bpeMerge(symbols)
	want := []string{"he", "ll", "o"}
	if len(got) != len(want) {
		t.Fatalf("bpeMerge = %v, want %v", got, want)
	}
	for i := range got {
		if got[i] != want[i] {
			t.Errorf("bpeMerge[%d] = %q, want %q", i, got[i], want[i])
		}
	}
}

func TestTokenizer_BPEMerge_NoMerges_Good(t *testing.T) {
	tok := &Tokenizer{mergeRanks: map[string]int{}}
	symbols := []string{"a", "b", "c"}
	got := tok.bpeMerge(symbols)
	if len(got) != 3 {
		t.Errorf("bpeMerge with no merges = %v, want [a b c]", got)
	}
}

func TestTokenizer_BPEMerge_SingleSymbol_Good(t *testing.T) {
	tok := &Tokenizer{mergeRanks: map[string]int{"a b": 0}}
	got := tok.bpeMerge([]string{"x"})
	if len(got) != 1 || got[0] != "x" {
		t.Errorf("bpeMerge single = %v, want [x]", got)
	}
}

func TestTokenizer_EncodeCachesSentencePieceSegments_Good(t *testing.T) {
	tok := &Tokenizer{
		vocab: map[string]int32{
			"▁ab": 7,
		},
		mergeRanks: map[string]int{
			"▁ a":  0,
			"▁a b": 1,
		},
	}

	first := tok.Encode("ab")
	if len(first) != 1 || first[0] != 7 {
		t.Fatalf("Encode first = %v, want [7]", first)
	}
	if len(tok.bpeCache) != 1 {
		t.Fatalf("bpe cache entries = %d, want 1", len(tok.bpeCache))
	}

	first[0] = 99
	second := tok.Encode("ab")
	if len(second) != 1 || second[0] != 7 {
		t.Fatalf("Encode second = %v, want cached [7]", second)
	}
	if len(tok.bpeCache) != 1 {
		t.Fatalf("bpe cache entries after repeat = %d, want 1", len(tok.bpeCache))
	}
}

func TestTokenizer_Decode_SpecialTokensSkipped_Good(t *testing.T) {
	path := writeTestTokenizer(t)
	tok, _ := LoadTokenizer(path)

	// Decoding BOS/EOS should produce empty string
	text := tok.Decode([]int32{100, 101})
	if text != "" {
		t.Errorf("Decode(BOS, EOS) = %q, want empty", text)
	}
}

func TestTokenizer_Decode_RegularTokens_Good(t *testing.T) {
	path := writeTestTokenizer(t)
	tok, _ := LoadTokenizer(path)

	// Decode known vocab entries
	text := tok.Decode([]int32{5, 6, 3}) // "he" + "ll" + "o"
	if text != "hello" {
		t.Errorf("Decode = %q, want %q", text, "hello")
	}
}

func TestTokenizer_DecodeToken_Regular_Good(t *testing.T) {
	path := writeTestTokenizer(t)
	tok, _ := LoadTokenizer(path)

	// "he" = token 5
	text := tok.DecodeToken(5)
	if text != "he" {
		t.Errorf("DecodeToken(5) = %q, want %q", text, "he")
	}
}

func TestTokenizer_DecodeToken_Special_Good(t *testing.T) {
	path := writeTestTokenizer(t)
	tok, _ := LoadTokenizer(path)

	// Special tokens should return empty
	text := tok.DecodeToken(100)
	if text != "" {
		t.Errorf("DecodeToken(BOS) = %q, want empty", text)
	}
}

func TestTokenizer_DecodeToken_SentencePieceSpace_Good(t *testing.T) {
	path := writeTestTokenizer(t)
	tok, _ := LoadTokenizer(path)

	// "▁h" = token 7, should decode to " h" (space prefix)
	text := tok.DecodeToken(7)
	if text != " h" {
		t.Errorf("DecodeToken(7) = %q, want %q", text, " h")
	}
}

func TestTokenizer_DecodeToken_Unknown_Bad(t *testing.T) {
	path := writeTestTokenizer(t)
	tok, _ := LoadTokenizer(path)

	text := tok.DecodeToken(9999)
	if text != "" {
		t.Errorf("DecodeToken(unknown) = %q, want empty", text)
	}
}

// DecodeOne mirrors Decode([]int32{id}) — verify byte-exact equivalence on
// regular, SentencePiece-prefixed, special, and unknown ids. This is the
// contract IDToken depends on for its no-allocation fast path.
func TestTokenizer_DecodeOne_MatchesDecodeSingle_Good(t *testing.T) {
	path := writeTestTokenizer(t)
	tok, _ := LoadTokenizer(path)

	cases := []struct {
		name string
		id   int32
	}{
		{"regular_he", 5},
		{"regular_ll", 6},
		{"sentencepiece_h", 7},
		{"special_bos", 100},
		{"special_eos", 101},
		{"unknown_high", 9999},
	}
	for _, c := range cases {
		want := tok.Decode([]int32{c.id})
		got := tok.DecodeOne(c.id)
		if got != want {
			t.Errorf("DecodeOne(%s id=%d) = %q, want %q (Decode parity)",
				c.name, c.id, got, want)
		}
	}
}

// TestTokenizer_DecodeOne_MarkerEdges_Good locks the zero-allocation fast paths
// in DecodeOne against the Replace+strip fallback that Decode([]int32{id}) takes.
// The minimal fixture vocab cannot express interior/repeated markers, so this
// builds the invVocab directly with the tricky SentencePiece pieces: a leading
// marker ("▁h"), a solo marker ("▁"), a double leading marker ("▁▁h"), an
// interior marker ("a▁b"), and a leading+interior mix ("▁a▁b"). DecodeOne must
// stay byte-exact with Decode for every one.
func TestTokenizer_DecodeOne_MarkerEdges_Good(t *testing.T) {
	pieces := map[int32]string{
		1: "he",   // no marker
		2: "▁h",   // leading marker only
		3: "▁",    // solo marker
		4: "▁▁h",  // double leading marker
		5: "a▁b",  // interior marker
		6: "▁a▁b", // leading + interior
		7: "",     // empty
	}
	tok := &Tokenizer{
		invVocab: pieces,
		special:  map[string]int32{},
		vocab:    map[string]int32{},
	}
	for id := range pieces {
		want := tok.Decode([]int32{id})
		got := tok.DecodeOne(id)
		if got != want {
			t.Errorf("DecodeOne(id=%d %q) = %q, want %q (Decode parity)",
				id, pieces[id], got, want)
		}
	}
}

func TestTokenizer_FormatGemmaPrompt_Good(t *testing.T) {
	got := FormatGemmaPrompt("What is 2+2?")
	want := "<bos><start_of_turn>user\nWhat is 2+2?<end_of_turn>\n<start_of_turn>model\n"
	if got != want {
		t.Errorf("FormatGemmaPrompt = %q, want %q", got, want)
	}
}

// --- GPT-2 byte maps ---

func TestTokenizer_BuildGPT2ByteMaps_Good(t *testing.T) {
	decoder, encoder := buildGPT2ByteMaps()

	// All 256 bytes must be mapped
	if len(encoder) != 256 {
		t.Errorf("encoder has %d entries, want 256", len(encoder))
	}
	if len(decoder) != 256 {
		t.Errorf("decoder has %d entries, want 256", len(decoder))
	}

	// Round-trip: every byte should survive encode → decode
	for b := range 256 {
		r := encoder[byte(b)]
		got := decoder[r]
		if got != byte(b) {
			t.Errorf("byte %d: encode→decode = %d, want %d", b, got, b)
		}
	}
}

func TestTokenizer_BuildGPT2ByteMaps_PrintableASCII_Good(t *testing.T) {
	_, encoder := buildGPT2ByteMaps()

	// Printable ASCII (33-126) should self-map
	for b := 33; b <= 126; b++ {
		if encoder[byte(b)] != rune(b) {
			t.Errorf("byte %d (%c): expected self-map, got %c", b, b, encoder[byte(b)])
		}
	}
}

func TestTokenizer_BuildGPT2ByteMaps_ControlChars_Good(t *testing.T) {
	_, encoder := buildGPT2ByteMaps()

	// Space (32) and control chars (0-31) should NOT self-map
	if encoder[byte(32)] == rune(32) {
		t.Error("space (32) should not self-map in GPT-2 encoding")
	}
	if encoder[byte(0)] == rune(0) {
		t.Error("null (0) should not self-map in GPT-2 encoding")
	}
}

// TestTokenizer_Encode_EmptyString_Ugly tests encoding an empty string.
// Should return only the BOS token (no panic, no out-of-bounds).
func TestTokenizer_Encode_EmptyString_Ugly(t *testing.T) {
	path := writeTestTokenizer(t)
	tok, _ := LoadTokenizer(path)

	tokens := tok.Encode("")
	// Empty input: only BOS token expected
	if len(tokens) == 0 {
		t.Fatal("Encode(\"\") returned empty slice — expected at least BOS token")
	}
	if tokens[0] != tok.BOSToken() {
		t.Errorf("first token = %d, want BOS (%d)", tokens[0], tok.BOSToken())
	}
}

// TestTokenizer_Decode_EmptySlice_Ugly tests decoding an empty token slice.
// Should return empty string without panicking.
func TestTokenizer_Decode_EmptySlice_Ugly(t *testing.T) {
	path := writeTestTokenizer(t)
	tok, _ := LoadTokenizer(path)

	text := tok.Decode([]int32{})
	if text != "" {
		t.Errorf("Decode(empty) = %q, want empty string", text)
	}
}

// TestTokenizer_DecodeToken_UnknownID_Ugly tests decoding a token ID outside vocab range.
// Should return empty string without panicking.
func TestTokenizer_DecodeToken_UnknownID_Ugly(t *testing.T) {
	path := writeTestTokenizer(t)
	tok, _ := LoadTokenizer(path)

	// Use a large ID well outside any realistic vocab range
	text := tok.DecodeToken(1 << 30)
	if text != "" {
		t.Errorf("DecodeToken(huge id) = %q, want empty", text)
	}
}

// TestTokenizer_BPEMerge_NilSymbols_Ugly tests bpeMerge with an empty symbols slice.
// Should return empty slice without panicking.
func TestTokenizer_BPEMerge_NilSymbols_Ugly(t *testing.T) {
	tok := &Tokenizer{mergeRanks: map[string]int{"a b": 0}}
	got := tok.bpeMerge([]string{})
	if len(got) != 0 {
		t.Errorf("bpeMerge(empty) = %v, want empty", got)
	}
}

// gpt2Fixture builds a minimal GPT-2 byte-level BPE Tokenizer in-process (the
// SentencePiece fixtures above never exercise the byte-level path). The vocab
// holds the encoded single-char forms plus a couple of merges so encodeGPT2
// runs a real merge + cache cycle.
func gpt2Fixture() *Tokenizer {
	dec, enc := buildGPT2ByteMaps()
	g := func(b byte) string { return string(enc[b]) }
	vocab := map[string]int32{
		g('h'): 0, g('e'): 1, g('l'): 2, g('o'): 3, g(' '): 4,
		g('h') + g('e'): 5, g('l') + g('l'): 6, g(' ') + g('h'): 7,
	}
	invVocab := map[int32]string{}
	for s, id := range vocab {
		invVocab[id] = s
	}
	mergeRanks := map[string]int{
		g('h') + " " + g('e'): 0,
		g('l') + " " + g('l'): 1,
		g(' ') + " " + g('h'): 2,
	}
	return &Tokenizer{
		vocab: vocab, invVocab: invVocab, mergeRanks: mergeRanks,
		special: map[string]int32{}, isGPT2BPE: true,
		gpt2Encoder: enc, gpt2Decoder: dec,
	}
}

// TestTokenizer_EncodeGPT2_WarmEqualsCold_Good guards the byte-level BPE warm
// path: the pooled-scratch key built for a cache hit must yield tokens
// identical to a cold cache miss across spaces, repeats, and unmapped-empty
// input. This is the GPT-2 sibling of the SentencePiece cache-equivalence test.
func TestTokenizer_EncodeGPT2_WarmEqualsCold_Good(t *testing.T) {
	inputs := []string{"", "h", "he", "hello", "hello hello", " hello", "hh", "lll", "ohel h"}
	for _, in := range inputs {
		cold := gpt2Fixture()
		want := append([]int32(nil), cold.encodeGPT2(in)...)
		warm := gpt2Fixture()
		_ = warm.encodeGPT2(in) // populate cache
		got := warm.encodeGPT2(in)
		if len(got) != len(want) {
			t.Fatalf("encodeGPT2(%q): warm=%v cold=%v (len)", in, got, want)
		}
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("encodeGPT2(%q) byte mismatch at %d: warm=%v cold=%v", in, i, got, want)
			}
		}
	}
}

// TestTokenizer_LoadTokenizer_EmptyFile_Ugly tests loading a tokenizer from an empty file.
// Should return a parse error, not panic.
func TestTokenizer_LoadTokenizer_EmptyFile_Ugly(t *testing.T) {
	dir := t.TempDir()
	path := core.JoinPath(dir, "tokenizer.json")
	_ = coreio.Local.Write(path, "")

	_, err := LoadTokenizer(path)
	if err == nil {
		t.Error("expected error for empty tokenizer file")
	}
}

// gpt2WithSpecialFixture extends gpt2Fixture with a single special token
// ("<|endoftext|>", id 50) so the GPT-2 special-token branches in encodeGPT2
// (match + split) and in Decode/DecodeToken (skip special) are exercised. The
// special is matched in ORIGINAL form, not byte-encoded — mirroring production.
func gpt2WithSpecialFixture() *Tokenizer {
	dec, enc := buildGPT2ByteMaps()
	g := func(b byte) string { return string(enc[b]) }
	vocab := map[string]int32{
		g('h'): 0, g('e'): 1, g('l'): 2, g('o'): 3, g(' '): 4,
		g('h') + g('e'): 5, g('l') + g('l'): 6, g(' ') + g('h'): 7,
		"<|endoftext|>": 50,
	}
	invVocab := map[int32]string{}
	for s, id := range vocab {
		invVocab[id] = s
	}
	mergeRanks := map[string]int{
		g('h') + " " + g('e'): 0,
		g('l') + " " + g('l'): 1,
		g(' ') + " " + g('h'): 2,
	}
	return &Tokenizer{
		vocab: vocab, invVocab: invVocab, mergeRanks: mergeRanks,
		special: map[string]int32{"<|endoftext|>": 50}, specialOrder: []string{"<|endoftext|>"},
		isGPT2BPE: true, gpt2Encoder: enc, gpt2Decoder: dec,
	}
}

// --- GPT-2 byte-level decode path (Decode / DecodeToken / DecodeOne) ---

// TestTokenizer_Decode_GPT2RoundTrip_Good locks the byte-level decode branch of
// Decode: encodeGPT2 → Decode must reproduce the original text, including the
// inter-word space that the byte-level encoder maps to its own glyph. The
// SentencePiece fixtures never reach this branch (decodeGPT2Bytes was 0% before).
func TestTokenizer_Decode_GPT2RoundTrip_Good(t *testing.T) {
	tok := gpt2Fixture()
	for _, text := range []string{"hello", "hello hello", "he", "o"} {
		ids := tok.encodeGPT2(text)
		got := tok.Decode(ids)
		if got != text {
			t.Errorf("Decode(encodeGPT2(%q)) = %q, want %q", text, got, text)
		}
	}
}

// TestTokenizer_Decode_GPT2SkipsSpecial_Good verifies the GPT-2 Decode branch
// drops special tokens (the isSpecial continue) while still decoding the
// surrounding byte-level pieces. "<|endoftext|>" sits between two "hello"s.
func TestTokenizer_Decode_GPT2SkipsSpecial_Good(t *testing.T) {
	tok := gpt2WithSpecialFixture()
	ids := tok.encodeGPT2("hello<|endoftext|>hello")
	// Special token (id 50) must appear in the encoding...
	foundSpecial := false
	for _, id := range ids {
		if id == 50 {
			foundSpecial = true
		}
	}
	if !foundSpecial {
		t.Fatalf("encodeGPT2 did not emit special token: %v", ids)
	}
	// ...but Decode strips it, joining the two words with no separator.
	got := tok.Decode(ids)
	if got != "hellohello" {
		t.Errorf("Decode(GPT-2 with special) = %q, want %q", got, "hellohello")
	}
}

// TestTokenizer_DecodeToken_GPT2_Good exercises the byte-level branch of the
// single-token streaming decode: a regular GPT-2 piece round-trips, a special
// token returns "", and an unknown id returns "".
func TestTokenizer_DecodeToken_GPT2_Good(t *testing.T) {
	tok := gpt2WithSpecialFixture()
	if got := tok.DecodeToken(5); got != "he" { // g('h')+g('e')
		t.Errorf("DecodeToken(5) = %q, want %q", got, "he")
	}
	if got := tok.DecodeToken(50); got != "" { // special skipped
		t.Errorf("DecodeToken(special) = %q, want empty", got)
	}
	if got := tok.DecodeToken(9999); got != "" { // unknown id
		t.Errorf("DecodeToken(unknown) = %q, want empty", got)
	}
}

// TestTokenizer_DecodeOne_GPT2_Good exercises the byte-level branch of DecodeOne
// (the per-emitted-token wrapper). It must match DecodeToken for the GPT-2 path
// since neither strips a SentencePiece leading space there.
func TestTokenizer_DecodeOne_GPT2_Good(t *testing.T) {
	tok := gpt2WithSpecialFixture()
	for _, id := range []int32{5, 6, 3, 50, 9999} {
		got := tok.DecodeOne(id)
		want := tok.DecodeToken(id)
		if got != want {
			t.Errorf("DecodeOne(%d) = %q, want %q (DecodeToken parity, GPT-2)", id, got, want)
		}
	}
}

// TestTokenizer_DecodeGPT2Bytes_PassThrough_Ugly drives decodeGPT2Bytes directly
// with a rune that is NOT in the decoder map (a multi-byte CJK char), forcing the
// utf8.AppendRune pass-through branch. Bytes that ARE mapped decode back to the
// original byte; unmapped runes survive as their own UTF-8.
func TestTokenizer_DecodeGPT2Bytes_PassThrough_Ugly(t *testing.T) {
	dec, enc := buildGPT2ByteMaps()
	tok := &Tokenizer{gpt2Decoder: dec, gpt2Encoder: enc}

	// A mapped sequence: byte-encode "hi" and decode it back.
	encoded := string(enc['h']) + string(enc['i'])
	if got := tok.decodeGPT2Bytes(encoded); got != "hi" {
		t.Errorf("decodeGPT2Bytes(encoded %q) = %q, want %q", "hi", got, "hi")
	}

	// An unmapped rune (中) is not in the GPT-2 decoder map → passes through.
	if got := tok.decodeGPT2Bytes("中"); got != "中" {
		t.Errorf("decodeGPT2Bytes(unmapped) = %q, want %q", got, "中")
	}

	// Empty input → empty output, no panic.
	if got := tok.decodeGPT2Bytes(""); got != "" {
		t.Errorf("decodeGPT2Bytes(empty) = %q, want empty", got)
	}
}

// --- LoadTokenizer uncovered branches ---

// gpt2DetectArrayMergeJSON has the "Ġthe" sentinel (triggers GPT-2 byte-level
// detection), array-form merges [["a","b"]] (the second merge-parse branch),
// and the full set of model-specific special aliases so the BOS/EOS resolution
// chain past <bos>/<eos> is walked. Last alias wins for each of BOS and EOS.
const gpt2DetectArrayMergeJSON = `{
  "model": {
    "type": "BPE",
    "vocab": {"Ġthe": 10, "a": 0, "b": 1, "ab": 2},
    "merges": [["a", "b"]]
  },
  "added_tokens": [
    {"id": 200, "content": "<end_of_turn>", "special": true},
    {"id": 201, "content": "<|im_end|>", "special": true},
    {"id": 202, "content": "<|im_start|>", "special": true},
    {"id": 203, "content": "<|eot_id|>", "special": true},
    {"id": 204, "content": "<|begin_of_text|>", "special": true}
  ]
}`

// TestTokenizer_LoadTokenizer_GPT2AndArrayMerges_Good covers three otherwise-cold
// LoadTokenizer branches at once: GPT-2 detection (Ġthe → isGPT2BPE + byte maps),
// the [["a","b"]] array-merge parse path, and the model-specific special aliases.
// BOS resolves to the LAST matching alias (<|begin_of_text|> = 204) and EOS to
// the last EOS alias hit before it (<|eot_id|> = 203).
func TestTokenizer_LoadTokenizer_GPT2AndArrayMerges_Good(t *testing.T) {
	dir := t.TempDir()
	path := core.JoinPath(dir, "tokenizer.json")
	if err := coreio.Local.Write(path, gpt2DetectArrayMergeJSON); err != nil {
		t.Fatalf("write: %v", err)
	}
	tok, err := LoadTokenizer(path)
	if err != nil {
		t.Fatalf("LoadTokenizer: %v", err)
	}

	if !tok.isGPT2BPE {
		t.Error("isGPT2BPE = false, want true (Ġthe sentinel present)")
	}
	if tok.gpt2Decoder == nil || tok.gpt2Encoder == nil {
		t.Error("GPT-2 byte maps not built despite detection")
	}
	if len(tok.merges) != 1 {
		t.Fatalf("merges = %d, want 1 (array form parsed)", len(tok.merges))
	}
	if rank, ok := tok.mergeRanks["a b"]; !ok || rank != 0 {
		t.Errorf("mergeRanks[\"a b\"] = (%d, %t), want (0, true)", rank, ok)
	}
	if !tok.HasBOSToken() || tok.BOSToken() != 204 {
		t.Errorf("BOS = (%d, has=%t), want (204, true) from <|begin_of_text|>", tok.BOSToken(), tok.HasBOSToken())
	}
	if !tok.HasEOSToken() || tok.EOSToken() != 203 {
		t.Errorf("EOS = (%d, has=%t), want (203, true) from <|eot_id|>", tok.EOSToken(), tok.HasEOSToken())
	}
}

// TestTokenizer_LoadTokenizer_GemmaEndOfTurn_Good covers the <end_of_turn> EOS
// alias path in isolation (Gemma's generation stop token) without the later
// aliases overriding it. <bos> stays the BOS.
func TestTokenizer_LoadTokenizer_GemmaEndOfTurn_Good(t *testing.T) {
	const gemmaJSON = `{
	  "model": {"type": "BPE", "vocab": {"a": 0}, "merges": []},
	  "added_tokens": [
	    {"id": 2, "content": "<bos>", "special": true},
	    {"id": 1, "content": "<eos>", "special": true},
	    {"id": 106, "content": "<end_of_turn>", "special": true}
	  ]
	}`
	dir := t.TempDir()
	path := core.JoinPath(dir, "tokenizer.json")
	if err := coreio.Local.Write(path, gemmaJSON); err != nil {
		t.Fatalf("write: %v", err)
	}
	tok, err := LoadTokenizer(path)
	if err != nil {
		t.Fatalf("LoadTokenizer: %v", err)
	}
	if tok.BOSToken() != 2 {
		t.Errorf("BOS = %d, want 2 (<bos>)", tok.BOSToken())
	}
	// <end_of_turn> overrides the plain <eos> as the stop token.
	if tok.EOSToken() != 106 {
		t.Errorf("EOS = %d, want 106 (<end_of_turn> overrides <eos>)", tok.EOSToken())
	}
}

// TestTokenizer_LoadTokenizer_NoMerges_Ugly loads a tokenizer whose merges field
// is entirely absent — the merge-parse blocks must be skipped without panic and
// the tokenizer must still encode via direct vocab hits.
func TestTokenizer_LoadTokenizer_NoMerges_Ugly(t *testing.T) {
	const noMergeJSON = `{
	  "model": {"type": "BPE", "vocab": {"▁hi": 0, "▁": 1}},
	  "added_tokens": []
	}`
	dir := t.TempDir()
	path := core.JoinPath(dir, "tokenizer.json")
	if err := coreio.Local.Write(path, noMergeJSON); err != nil {
		t.Fatalf("write: %v", err)
	}
	tok, err := LoadTokenizer(path)
	if err != nil {
		t.Fatalf("LoadTokenizer (no merges): %v", err)
	}
	if len(tok.merges) != 0 {
		t.Errorf("merges = %d, want 0", len(tok.merges))
	}
	// "▁hi" is a single vocab entry; with no merges the symbols don't combine,
	// but the whole-segment lookup is not used — just assert no panic + a result.
	_ = tok.Encode("hi")
}

// --- storeBPETokens cache branches (overwrite + LRU eviction) ---

// TestTokenizer_StoreBPETokens_OverwriteAndReject_Good covers two storeBPETokens
// branches: re-storing an existing key replaces the value WITHOUT growing the
// order list, and an oversized token slice is rejected (not cached).
func TestTokenizer_StoreBPETokens_OverwriteAndReject_Good(t *testing.T) {
	tok := &Tokenizer{}

	tok.storeBPETokens("k", []int32{1, 2, 3})
	if got, ok := tok.cachedBPETokensBytes([]byte("k")); !ok || len(got) != 3 {
		t.Fatalf("after first store: cachedBPETokensBytes(k) = (%v, %t), want 3 tokens", got, ok)
	}
	orderLen := len(tok.bpeCacheOrder)

	// Overwrite existing key — value swaps, order list does NOT grow.
	tok.storeBPETokens("k", []int32{9})
	if got, _ := tok.cachedBPETokensBytes([]byte("k")); len(got) != 1 || got[0] != 9 {
		t.Errorf("after overwrite: cachedBPETokensBytes(k) = %v, want [9]", got)
	}
	if len(tok.bpeCacheOrder) != orderLen {
		t.Errorf("order list grew on overwrite: %d, want %d", len(tok.bpeCacheOrder), orderLen)
	}

	// Oversized token slice is rejected (len > tokenizerBPECacheMaxTokens).
	tok.storeBPETokens("toobig", make([]int32, tokenizerBPECacheMaxTokens+1))
	if _, ok := tok.cachedBPETokensBytes([]byte("toobig")); ok {
		t.Error("oversized token slice was cached, want rejected")
	}
}

// TestTokenizer_StoreBPETokens_LRUEviction_Ugly fills the cache to its hard limit
// then stores one more, asserting the oldest key is evicted (FIFO order list) and
// the cache size stays pinned at the limit.
func TestTokenizer_StoreBPETokens_LRUEviction_Ugly(t *testing.T) {
	tok := &Tokenizer{}
	for i := range tokenizerBPECacheLimit {
		tok.storeBPETokens(core.Sprintf("k%d", i), []int32{int32(i)})
	}
	if len(tok.bpeCache) != tokenizerBPECacheLimit {
		t.Fatalf("cache size = %d, want %d", len(tok.bpeCache), tokenizerBPECacheLimit)
	}
	if _, ok := tok.cachedBPETokensBytes([]byte("k0")); !ok {
		t.Fatal("k0 evicted before overflow")
	}

	// One past the limit evicts the oldest (k0) and pins size at the limit.
	tok.storeBPETokens("overflow", []int32{-1})
	if _, ok := tok.cachedBPETokensBytes([]byte("k0")); ok {
		t.Error("k0 not evicted after overflow")
	}
	if _, ok := tok.cachedBPETokensBytes([]byte("overflow")); !ok {
		t.Error("overflow key not stored")
	}
	if len(tok.bpeCache) != tokenizerBPECacheLimit {
		t.Errorf("cache size after overflow = %d, want %d", len(tok.bpeCache), tokenizerBPECacheLimit)
	}
}

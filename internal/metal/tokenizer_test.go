//go:build darwin && arm64

package metal

import (
	"os"
	"testing"

	"dappco.re/go/core"
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

func writeTestTokenizer(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	path := core.JoinPath(dir, "tokenizer.json")
	if err := os.WriteFile(path, []byte(minimalTokenizerJSON), 0644); err != nil {
		t.Fatalf("write test tokenizer: %v", err)
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
	os.WriteFile(path, []byte("not json"), 0644)

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

func TestTokenizer_FormatGemmaPrompt_Good(t *testing.T) {
	got := FormatGemmaPrompt("What is 2+2?")
	want := "<start_of_turn>user\nWhat is 2+2?<end_of_turn>\n<start_of_turn>model\n"
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

//go:build darwin && arm64 && !nomlx

package mlx

import (
	"os"
	"path/filepath"
	"testing"
)

const rootTokenizerJSON = `{
  "model": {
    "type": "BPE",
    "vocab": {
      "▁": 1,
      "h": 2,
      "e": 3,
      "l": 4,
      "o": 5,
      "▁h": 6,
      "▁he": 7,
      "▁hel": 8,
      "▁hell": 9,
      "▁hello": 10
    },
    "merges": ["▁ h", "▁h e", "▁he l", "▁hel l", "▁hell o"]
  },
  "added_tokens": [
    {"id": 0, "content": "<bos>", "special": true},
    {"id": 11, "content": "<eos>", "special": true}
  ]
}`

func writeRootTokenizer(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, "tokenizer.json")
	if err := os.WriteFile(path, []byte(rootTokenizerJSON), 0o644); err != nil {
		t.Fatalf("write tokenizer: %v", err)
	}
	return path
}

func TestRootTokenizerEncode_StripsImplicitBOS_Good(t *testing.T) {
	tok, err := LoadTokenizer(writeRootTokenizer(t))
	if err != nil {
		t.Fatalf("LoadTokenizer: %v", err)
	}

	got, err := tok.Encode("hello")
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}

	want := []int32{10}
	if len(got) != len(want) {
		t.Fatalf("Encode(\"hello\") len = %d, want %d (%v)", len(got), len(want), got)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("Encode(\"hello\")[%d] = %d, want %d", i, got[i], want[i])
		}
	}
}

func TestRootTokenizerEncode_PreservesExplicitSpecialTokens_Good(t *testing.T) {
	tok, err := LoadTokenizer(writeRootTokenizer(t))
	if err != nil {
		t.Fatalf("LoadTokenizer: %v", err)
	}

	got, err := tok.Encode("<bos>hello")
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}

	want := []int32{0, 10}
	if len(got) != len(want) {
		t.Fatalf("Encode(\"<bos>hello\") len = %d, want %d (%v)", len(got), len(want), got)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("Encode(\"<bos>hello\")[%d] = %d, want %d", i, got[i], want[i])
		}
	}
}

func TestRootTokenizerLookups_NormalizeSentencePieceForms_Good(t *testing.T) {
	tok, err := LoadTokenizer(writeRootTokenizer(t))
	if err != nil {
		t.Fatalf("LoadTokenizer: %v", err)
	}

	id, ok := tok.TokenID("hello")
	if !ok {
		t.Fatal("TokenID(\"hello\") returned false, want true")
	}
	if id != 10 {
		t.Fatalf("TokenID(\"hello\") = %d, want 10", id)
	}

	if got := tok.IDToken(10); got != "hello" {
		t.Fatalf("IDToken(10) = %q, want %q", got, "hello")
	}
	if got := tok.IDToken(0); got != "<bos>" {
		t.Fatalf("IDToken(0) = %q, want %q", got, "<bos>")
	}
	if tok.BOS() != 0 {
		t.Fatalf("BOS() = %d, want 0", tok.BOS())
	}
	if tok.EOS() != 11 {
		t.Fatalf("EOS() = %d, want 11", tok.EOS())
	}
}

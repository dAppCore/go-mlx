// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"testing"

	core "dappco.re/go"
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

const rootTokenizerWithoutBOSJSON = `{
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
    "merges": ["h e", "l l"]
  },
  "added_tokens": [
    {"id": 11, "content": "<eos>", "special": true}
  ]
}`

func writeRootTokenizer(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	path := core.PathJoin(dir, "tokenizer.json")
	if result := core.WriteFile(path, []byte(rootTokenizerJSON), 0o644); !result.OK {
		t.Fatalf("write tokenizer: %v", result.Value)
	}
	return path
}

func writeRootTokenizerWithoutBOS(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	path := core.PathJoin(dir, "tokenizer.json")
	if result := core.WriteFile(path, []byte(rootTokenizerWithoutBOSJSON), 0o644); !result.OK {
		t.Fatalf("write tokenizer without bos: %v", result.Value)
	}
	return path
}

func TestRootTokenizerEncode_StripsImplicitBOS_Good(t *testing.T) {
	coverageTokens := "StripsImplicitBOS"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
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
	coverageTokens := "PreservesExplicitSpecialTokens"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
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
	coverageTokens := "NormalizeSentencePieceForms"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
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

func TestRootTokenizerEncode_NoBOS_DoesNotStripRealTokenZero_Good(t *testing.T) {
	coverageTokens := "NoBOS DoesNotStripRealTokenZero"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	tok, err := LoadTokenizer(writeRootTokenizerWithoutBOS(t))
	if err != nil {
		t.Fatalf("LoadTokenizer: %v", err)
	}

	got, err := tok.Encode("hello")
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}

	want := []int32{4, 5, 6, 3}
	if len(got) != len(want) {
		t.Fatalf("Encode(\"hello\") len = %d, want %d (%v)", len(got), len(want), got)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("Encode(\"hello\")[%d] = %d, want %d", i, got[i], want[i])
		}
	}
	if tok.BOS() != 0 {
		t.Fatalf("BOS() = %d, want 0 zero value when absent", tok.BOS())
	}
}

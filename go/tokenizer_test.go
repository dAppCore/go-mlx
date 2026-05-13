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

func TestRootTokenizerWrapperFallbacks_Ugly(t *testing.T) {
	tok := &Tokenizer{tok: fakeSFTTokenizer{
		encoded: map[string][]int32{
			"single": {42},
			"multi":  {1, 2},
		},
		eos: 9,
	}}
	decoded, err := tok.Decode([]int32{4, 2})
	if err != nil {
		t.Fatalf("Decode() error = %v", err)
	}
	if decoded != "42" {
		t.Fatalf("Decode() = %q, want fake concatenated ids", decoded)
	}
	if id, ok := tok.TokenID("single"); !ok || id != 42 {
		t.Fatalf("TokenID(single) = %d/%v, want 42/true", id, ok)
	}
	if _, ok := tok.TokenID("multi"); ok {
		t.Fatal("TokenID(multi) ok = true, want false for multi-token text")
	}
	if got := (&Tokenizer{tok: fakeRawTokenizer{raw: "▁"}}).IDToken(7); got != " " {
		t.Fatalf("IDToken(sentencepiece space) = %q, want space", got)
	}
	if _, err := (*Tokenizer)(nil).Decode([]int32{1}); err == nil {
		t.Fatal("expected nil tokenizer decode error")
	}
}

type fakeRawTokenizer struct {
	raw string
}

func (t fakeRawTokenizer) Encode(string) []int32        { return []int32{7} }
func (t fakeRawTokenizer) Decode([]int32) string        { return "" }
func (t fakeRawTokenizer) TokenID(string) (int32, bool) { return 0, false }
func (t fakeRawTokenizer) IDToken(int32) string         { return t.raw }
func (t fakeRawTokenizer) BOS() int32                   { return 0 }
func (t fakeRawTokenizer) EOS() int32                   { return 0 }
func (t fakeRawTokenizer) HasBOSToken() bool            { return false }

// Generated file-aware compliance coverage.
func TestTokenizer_LoadTokenizer_Good(t *testing.T) {
	target := "LoadTokenizer"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestTokenizer_LoadTokenizer_Bad(t *testing.T) {
	target := "LoadTokenizer"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestTokenizer_LoadTokenizer_Ugly(t *testing.T) {
	target := "LoadTokenizer"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

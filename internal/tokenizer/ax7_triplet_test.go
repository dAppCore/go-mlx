// SPDX-Licence-Identifier: EUPL-1.2

package tokenizer

import . "dappco.re/go"

func ax7Tokenizer(t *T) *Tokenizer {
	t.Helper()
	tok, err := LoadTokenizer(writeTestTokenizer(t))
	RequireNoError(t, err)
	RequireTrue(t, tok != nil)
	return tok
}

func TestAX7_FormatGemmaPrompt_Good(t *T) {
	got := FormatGemmaPrompt("hello")

	AssertContains(t, got, "<start_of_turn>user")
	AssertContains(t, got, "hello<end_of_turn>")
}

func TestAX7_FormatGemmaPrompt_Bad(t *T) {
	got := FormatGemmaPrompt("")

	AssertContains(t, got, "<start_of_turn>model")
	AssertContains(t, got, "<end_of_turn>")
}

func TestAX7_FormatGemmaPrompt_Ugly(t *T) {
	got := FormatGemmaPrompt("line one\nline two")

	AssertContains(t, got, "line one\nline two")
	AssertTrue(t, HasSuffix(got, "<start_of_turn>model\n"))
}

func TestAX7_LoadTokenizer_Good(t *T) {
	tok := ax7Tokenizer(t)

	AssertTrue(t, tok.HasBOSToken())
	AssertTrue(t, tok.HasEOSToken())
}

func TestAX7_LoadTokenizer_Bad(t *T) {
	tok, err := LoadTokenizer(Path(t.TempDir(), "missing-tokenizer.json"))

	AssertError(t, err)
	AssertNil(t, tok)
}

func TestAX7_LoadTokenizer_Ugly(t *T) {
	tok, err := LoadTokenizer(t.TempDir())

	AssertError(t, err)
	AssertNil(t, tok)
}

func TestAX7_Tokenizer_BOS_Good(t *T) {
	tok := ax7Tokenizer(t)

	AssertEqual(t, int32(100), tok.BOS())
	AssertEqual(t, tok.BOSToken(), tok.BOS())
}

func TestAX7_Tokenizer_BOS_Bad(t *T) {
	tok, err := LoadTokenizer(writeTokenizerWithoutSpecials(t))
	RequireNoError(t, err)

	AssertEqual(t, int32(0), tok.BOS())
}

func TestAX7_Tokenizer_BOS_Ugly(t *T) {
	var tok *Tokenizer

	AssertFalse(t, tok.HasBOSToken())
	AssertPanics(t, func() { _ = tok.BOS() })
}

func TestAX7_Tokenizer_BOSToken_Good(t *T) {
	tok := ax7Tokenizer(t)

	AssertEqual(t, int32(100), tok.BOSToken())
	AssertTrue(t, tok.HasBOSToken())
}

func TestAX7_Tokenizer_BOSToken_Bad(t *T) {
	tok, err := LoadTokenizer(writeTokenizerWithoutSpecials(t))
	RequireNoError(t, err)

	AssertEqual(t, int32(0), tok.BOSToken())
}

func TestAX7_Tokenizer_BOSToken_Ugly(t *T) {
	tok := &Tokenizer{}

	AssertFalse(t, tok.HasBOSToken())
	AssertEqual(t, int32(0), tok.BOSToken())
}

func TestAX7_Tokenizer_Decode_Good(t *T) {
	tok := ax7Tokenizer(t)

	AssertEqual(t, "hello", tok.Decode([]int32{100, 4, 5, 6, 3}))
	AssertEqual(t, "", tok.Decode([]int32{100}))
}

func TestAX7_Tokenizer_Decode_Bad(t *T) {
	tok := ax7Tokenizer(t)

	AssertEqual(t, "", tok.Decode([]int32{999}))
	AssertEqual(t, "", tok.Decode(nil))
}

func TestAX7_Tokenizer_Decode_Ugly(t *T) {
	tok := ax7Tokenizer(t)

	AssertEqual(t, "hello hello", tok.Decode(tok.Encode("hello hello")))
	AssertEqual(t, "", tok.Decode([]int32{101}))
}

func TestAX7_Tokenizer_DecodeToken_Good(t *T) {
	tok := ax7Tokenizer(t)

	AssertEqual(t, "he", tok.DecodeToken(5))
	AssertEqual(t, "ll", tok.DecodeToken(6))
}

func TestAX7_Tokenizer_DecodeToken_Bad(t *T) {
	tok := ax7Tokenizer(t)

	AssertEqual(t, "", tok.DecodeToken(999))
	AssertEqual(t, "", tok.DecodeToken(100))
}

func TestAX7_Tokenizer_DecodeToken_Ugly(t *T) {
	tok := ax7Tokenizer(t)

	AssertEqual(t, " ", tok.DecodeToken(4))
	AssertEqual(t, " h", tok.DecodeToken(7))
}

func TestAX7_Tokenizer_EOS_Good(t *T) {
	tok := ax7Tokenizer(t)

	AssertEqual(t, int32(101), tok.EOS())
	AssertEqual(t, tok.EOSToken(), tok.EOS())
}

func TestAX7_Tokenizer_EOS_Bad(t *T) {
	tok, err := LoadTokenizer(writeTokenizerWithoutSpecials(t))
	RequireNoError(t, err)

	AssertEqual(t, int32(0), tok.EOS())
}

func TestAX7_Tokenizer_EOS_Ugly(t *T) {
	tok := &Tokenizer{}

	AssertFalse(t, tok.HasEOSToken())
	AssertEqual(t, int32(0), tok.EOS())
}

func TestAX7_Tokenizer_EOSToken_Good(t *T) {
	tok := ax7Tokenizer(t)

	AssertEqual(t, int32(101), tok.EOSToken())
	AssertTrue(t, tok.HasEOSToken())
}

func TestAX7_Tokenizer_EOSToken_Bad(t *T) {
	tok, err := LoadTokenizer(writeTokenizerWithoutSpecials(t))
	RequireNoError(t, err)

	AssertEqual(t, int32(0), tok.EOSToken())
}

func TestAX7_Tokenizer_EOSToken_Ugly(t *T) {
	tok := &Tokenizer{}

	AssertFalse(t, tok.HasEOSToken())
	AssertEqual(t, int32(0), tok.EOSToken())
}

func TestAX7_Tokenizer_Encode_Good(t *T) {
	tok := ax7Tokenizer(t)

	AssertEqual(t, []int32{100, 4, 5, 6, 3}, tok.Encode("hello"))
	AssertEqual(t, int32(100), tok.Encode("hello")[0])
}

func TestAX7_Tokenizer_Encode_Bad(t *T) {
	tok := ax7Tokenizer(t)

	AssertEqual(t, []int32{100}, tok.Encode(""))
	AssertEqual(t, []int32{100, 4}, tok.Encode("???"))
}

func TestAX7_Tokenizer_Encode_Ugly(t *T) {
	tok := ax7Tokenizer(t)

	AssertEqual(t, []int32{100, 4, 5, 6, 3, 4, 5, 6, 3}, tok.Encode("hello hello"))
	AssertEqual(t, "hello hello", tok.Decode(tok.Encode("hello hello")))
}

func TestAX7_Tokenizer_HasBOSToken_Good(t *T) {
	tok := ax7Tokenizer(t)

	AssertTrue(t, tok.HasBOSToken())
	AssertEqual(t, int32(100), tok.BOSToken())
}

func TestAX7_Tokenizer_HasBOSToken_Bad(t *T) {
	tok, err := LoadTokenizer(writeTokenizerWithoutSpecials(t))
	RequireNoError(t, err)

	AssertFalse(t, tok.HasBOSToken())
}

func TestAX7_Tokenizer_HasBOSToken_Ugly(t *T) {
	var tok *Tokenizer

	AssertFalse(t, tok.HasBOSToken())
	AssertNil(t, tok)
}

func TestAX7_Tokenizer_HasEOSToken_Good(t *T) {
	tok := ax7Tokenizer(t)

	AssertTrue(t, tok.HasEOSToken())
	AssertEqual(t, int32(101), tok.EOSToken())
}

func TestAX7_Tokenizer_HasEOSToken_Bad(t *T) {
	tok, err := LoadTokenizer(writeTokenizerWithoutSpecials(t))
	RequireNoError(t, err)

	AssertFalse(t, tok.HasEOSToken())
}

func TestAX7_Tokenizer_HasEOSToken_Ugly(t *T) {
	var tok *Tokenizer

	AssertFalse(t, tok.HasEOSToken())
	AssertNil(t, tok)
}

func TestAX7_Tokenizer_IDToken_Good(t *T) {
	tok := ax7Tokenizer(t)

	AssertEqual(t, "he", tok.IDToken(5))
	AssertEqual(t, "ll", tok.IDToken(6))
}

func TestAX7_Tokenizer_IDToken_Bad(t *T) {
	tok := ax7Tokenizer(t)

	AssertEqual(t, "", tok.IDToken(999))
	AssertEqual(t, "<bos>", tok.IDToken(100))
}

func TestAX7_Tokenizer_IDToken_Ugly(t *T) {
	tok := &Tokenizer{invVocab: map[int32]string{1: "one"}}

	AssertEqual(t, "one", tok.IDToken(1))
	AssertEqual(t, "", tok.IDToken(2))
}

func TestAX7_Tokenizer_TokenID_Good(t *T) {
	tok := ax7Tokenizer(t)

	id, ok := tok.TokenID("he")
	AssertTrue(t, ok)
	AssertEqual(t, int32(5), id)
}

func TestAX7_Tokenizer_TokenID_Bad(t *T) {
	tok := ax7Tokenizer(t)

	id, ok := tok.TokenID("missing")
	AssertFalse(t, ok)
	AssertEqual(t, int32(0), id)
}

func TestAX7_Tokenizer_TokenID_Ugly(t *T) {
	tok := &Tokenizer{vocab: map[string]int32{"": 7}}

	id, ok := tok.TokenID("")
	AssertTrue(t, ok)
	AssertEqual(t, int32(7), id)
}

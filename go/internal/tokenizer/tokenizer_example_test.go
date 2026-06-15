// SPDX-Licence-Identifier: EUPL-1.2

package tokenizer

import core "dappco.re/go"

func ExampleLoadTokenizer() {
	tok, cleanup := mustExampleTokenizer()
	defer cleanup()

	core.Println(tok != nil, tok.BOSToken(), tok.EOSToken())
	// Output: true 100 101
}

func ExampleTokenizer_Encode() {
	tok, cleanup := mustExampleTokenizer()
	defer cleanup()

	core.Println(tok.Encode("hello"))
	// Output: [100 4 5 6 3]
}

func ExampleTokenizer_Decode() {
	tok, cleanup := mustExampleTokenizer()
	defer cleanup()

	core.Println(tok.Decode([]int32{100, 4, 5, 6, 3}))
	// Output: hello
}

func ExampleTokenizer_DecodeToken() {
	tok, cleanup := mustExampleTokenizer()
	defer cleanup()

	core.Println(tok.DecodeToken(5), tok.DecodeToken(7))
	// Output: he  h
}

func ExampleTokenizer_BOSToken() {
	tok, cleanup := mustExampleTokenizer()
	defer cleanup()

	core.Println(tok.BOSToken())
	// Output: 100
}

func ExampleTokenizer_EOSToken() {
	tok, cleanup := mustExampleTokenizer()
	defer cleanup()

	core.Println(tok.EOSToken())
	// Output: 101
}

func ExampleTokenizer_HasBOSToken() {
	tok, cleanup := mustExampleTokenizer()
	defer cleanup()

	core.Println(tok.HasBOSToken())
	// Output: true
}

func ExampleTokenizer_HasEOSToken() {
	tok, cleanup := mustExampleTokenizer()
	defer cleanup()

	core.Println(tok.HasEOSToken())
	// Output: true
}

func ExampleTokenizer_BOS() {
	tok, cleanup := mustExampleTokenizer()
	defer cleanup()

	core.Println(tok.BOS())
	// Output: 100
}

func ExampleTokenizer_EOS() {
	tok, cleanup := mustExampleTokenizer()
	defer cleanup()

	core.Println(tok.EOS())
	// Output: 101
}

func ExampleTokenizer_TokenID() {
	tok, cleanup := mustExampleTokenizer()
	defer cleanup()

	id, ok := tok.TokenID("he")
	core.Println(id, ok)
	// Output: 5 true
}

func ExampleTokenizer_IDToken() {
	tok, cleanup := mustExampleTokenizer()
	defer cleanup()

	core.Println(tok.IDToken(6))
	// Output: ll
}

func ExampleFormatGemmaPrompt() {
	core.Println(FormatGemmaPrompt("What is 2+2?"))
	// Output:
	// <bos><start_of_turn>user
	// What is 2+2?<end_of_turn>
	// <start_of_turn>model
}

// ExampleTokenizer_Decode_gpt2 shows the byte-level BPE decode path (Qwen, GPT,
// Llama use it). encodeGPT2 maps each byte to a printable glyph; Decode reverses
// it, so the round-trip reproduces the original text including the inter-word
// space.
func ExampleTokenizer_Decode_gpt2() {
	tok := exampleGPT2Tokenizer()
	core.Println(tok.Decode(tok.encodeGPT2("hello hello")))
	// Output: hello hello
}

// exampleGPT2Tokenizer mirrors the test-side gpt2Fixture: a minimal byte-level
// BPE tokenizer with two merges, enough to demonstrate the round-trip without a
// multi-GB model load.
func exampleGPT2Tokenizer() *Tokenizer {
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

func mustExampleTokenizer() (*Tokenizer, func()) {
	dirResult := core.MkdirTemp("", "go-mlx-tokenizer-example-*")
	if !dirResult.OK {
		panic(dirResult.Value)
	}
	dir := dirResult.Value.(string)
	path := core.PathJoin(dir, "tokenizer.json")
	if result := core.WriteFile(path, []byte(minimalTokenizerJSON), 0o644); !result.OK {
		core.RemoveAll(dir)
		panic(result.Value)
	}
	tok, err := LoadTokenizer(path)
	if err != nil {
		core.RemoveAll(dir)
		panic(err)
	}
	return tok, func() { core.RemoveAll(dir) }
}

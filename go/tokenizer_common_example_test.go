// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import core "dappco.re/go"

func ExampleTokenizer_Encode() {
	tokenizer, cleanup := mustExampleRootTokenizer()
	defer cleanup()

	tokens, err := tokenizer.Encode("hello")

	core.Println(tokens, err == nil)
	// Output: [10] true
}

func ExampleTokenizer_Decode() {
	tokenizer, cleanup := mustExampleRootTokenizer()
	defer cleanup()

	text, err := tokenizer.Decode([]int32{10})

	core.Println(text, err == nil)
	// Output: hello true
}

func ExampleTokenizer_TokenID() {
	tokenizer, cleanup := mustExampleRootTokenizer()
	defer cleanup()

	id, ok := tokenizer.TokenID("hello")

	core.Println(id, ok)
	// Output: 10 true
}

func ExampleTokenizer_IDToken() {
	tokenizer, cleanup := mustExampleRootTokenizer()
	defer cleanup()

	core.Println(tokenizer.IDToken(10), tokenizer.IDToken(0))
	// Output: hello <bos>
}

func ExampleTokenizer_BOS() {
	tokenizer, cleanup := mustExampleRootTokenizer()
	defer cleanup()

	core.Println(tokenizer.BOS())
	// Output: 0
}

func ExampleTokenizer_EOS() {
	tokenizer, cleanup := mustExampleRootTokenizer()
	defer cleanup()

	core.Println(tokenizer.EOS())
	// Output: 11
}

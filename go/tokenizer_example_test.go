// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	core "dappco.re/go"
)

func ExampleLoadTokenizer() {
	tokenizer, cleanup := mustExampleRootTokenizer()
	defer cleanup()

	tokens, err := tokenizer.Encode("hello")
	id, ok := tokenizer.TokenID("hello")

	core.Println(err == nil, tokens, id, ok, tokenizer.IDToken(id), tokenizer.EOS())
	// Output: true [10] 10 true hello 11
}

func mustExampleRootTokenizer() (*Tokenizer, func()) {
	dirResult := core.MkdirTemp("", "go-mlx-root-tokenizer-example-*")
	if !dirResult.OK {
		panic(dirResult.Value)
	}
	dir := dirResult.Value.(string)
	path := core.PathJoin(dir, "tokenizer.json")
	if result := core.WriteFile(path, []byte(rootTokenizerJSON), 0o644); !result.OK {
		core.RemoveAll(dir)
		panic(result.Value)
	}
	tokenizer, err := LoadTokenizer(path)
	if err != nil {
		core.RemoveAll(dir)
		panic(err)
	}
	return tokenizer, func() { core.RemoveAll(dir) }
}

// --- merged from tokenizer_common_example_test.go (orphan sweep:
// tokenizer_common.go moved to spine; the examples document the root API) ---
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

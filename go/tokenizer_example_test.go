// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import core "dappco.re/go"

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

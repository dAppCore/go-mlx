//go:build !(darwin && arm64) || nomlx

package mlx

import puretokenizer "dappco.re/go/mlx/internal/tokenizer"

// LoadTokenizer loads a tokenizer.json file directly using the pure-Go tokenizer implementation.
func LoadTokenizer(path string) (*Tokenizer, error) {
	tok, err := puretokenizer.LoadTokenizer(path)
	if err != nil {
		return nil, err
	}
	return &Tokenizer{tok: tok}, nil
}

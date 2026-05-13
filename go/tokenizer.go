// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && !nomlx

package mlx

import "dappco.re/go/mlx/internal/metal"

// LoadTokenizer loads a tokenizer.json file directly.
func LoadTokenizer(path string) (*Tokenizer, error) {
	tok, err := metal.LoadTokenizer(path)
	if err != nil {
		return nil, err
	}
	return &Tokenizer{tok: tok}, nil
}

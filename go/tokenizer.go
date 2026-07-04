// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"dappco.re/go/mlx/pkg/metal"
	"dappco.re/go/mlx/spine"
)

// tokenizer.go: the root tokenizer surface. The Tokenizer wrapper itself
// lives in spine so subpackages (session, train) can decode tokens
// without importing root; root keeps the on-disk loader and the aliases.

// TokenizerImpl is the pure-Go tokenizer contract the root API wraps.
type TokenizerImpl = spine.TokenizerImpl

// Tokenizer wraps a pure-Go tokenizer implementation with a root-package API.
type Tokenizer = spine.Tokenizer

// NewTokenizer wraps a TokenizerImpl in the root Tokenizer API. It is the
// bring-your-own-tokenizer seam: callers (and test packages outside mlx) build
// a Tokenizer from any implementation without reaching the unexported field.
//
//	tok := mlx.NewTokenizer(myImpl)
func NewTokenizer(impl TokenizerImpl) *Tokenizer {
	return spine.NewTokenizer(impl)
}

// LoadTokenizer loads a tokenizer.json file directly.
func LoadTokenizer(path string) (*Tokenizer, error) {
	tok, err := metal.LoadTokenizer(path)
	if err != nil {
		return nil, err
	}
	return spine.NewTokenizer(tok), nil
}

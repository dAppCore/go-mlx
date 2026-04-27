// SPDX-Licence-Identifier: EUPL-1.2

package mlx

// Note: AX-6 — errors is structural for sentinel error declaration in tokenizer; core.E is downstream of go-mlx.
import "errors"

type tokenizerImpl interface {
	Encode(string) []int32
	Decode([]int32) string
	TokenID(string) (int32, bool)
	IDToken(int32) string
	BOS() int32
	EOS() int32
	HasBOSToken() bool
}

// Tokenizer wraps a pure-Go tokenizer implementation with a root-package API.
type Tokenizer struct {
	tok tokenizerImpl
}

func stripImplicitBOS(tok tokenizerImpl, tokens []int32) []int32 {
	if tok == nil || len(tokens) == 0 {
		return append([]int32(nil), tokens...)
	}
	if tok.HasBOSToken() && tokens[0] == tok.BOS() {
		return append([]int32(nil), tokens[1:]...)
	}
	return append([]int32(nil), tokens...)
}

// Encode converts text to token IDs without the model-internal implicit BOS token.
func (t *Tokenizer) Encode(text string) ([]int32, error) {
	if t == nil || t.tok == nil {
		return nil, errors.New("mlx: tokenizer is nil")
	}
	return stripImplicitBOS(t.tok, t.tok.Encode(text)), nil
}

// Decode converts token IDs back to text.
func (t *Tokenizer) Decode(tokens []int32) (string, error) {
	if t == nil || t.tok == nil {
		return "", errors.New("mlx: tokenizer is nil")
	}
	return t.tok.Decode(tokens), nil
}

// TokenID resolves a token string to its ID.
func (t *Tokenizer) TokenID(text string) (int32, bool) {
	if t == nil || t.tok == nil {
		return 0, false
	}
	if id, ok := t.tok.TokenID(text); ok {
		return id, true
	}
	// The public tokenizer API accepts plain-text tokens such as "hello",
	// while the internal tokenizer stores model-native forms like "▁hello".
	encoded := stripImplicitBOS(t.tok, t.tok.Encode(text))
	if len(encoded) == 1 {
		return encoded[0], true
	}
	return 0, false
}

// IDToken resolves a token ID to a decoded token string when possible.
func (t *Tokenizer) IDToken(id int32) string {
	if t == nil || t.tok == nil {
		return ""
	}
	raw := t.tok.IDToken(id)
	if raw == "" {
		return ""
	}
	if decoded := t.tok.Decode([]int32{id}); decoded != "" {
		return decoded
	}
	if raw == "▁" {
		return " "
	}
	return raw
}

// BOS returns the beginning-of-sequence token ID.
func (t *Tokenizer) BOS() int32 {
	if t == nil || t.tok == nil {
		return 0
	}
	return t.tok.BOS()
}

// EOS returns the end-of-sequence token ID.
func (t *Tokenizer) EOS() int32 {
	if t == nil || t.tok == nil {
		return 0
	}
	return t.tok.EOS()
}

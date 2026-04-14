//go:build !(darwin && arm64) || nomlx

package mlx

import "errors"

// Model is a stub on unsupported builds.
type Model struct{}

// Tokenizer is a stub on unsupported builds.
type Tokenizer struct{}

// LoadModel returns an availability error on unsupported builds.
func LoadModel(_ string, _ ...LoadOption) (*Model, error) {
	return nil, errors.New("mlx: native MLX support is unavailable in this build")
}

// LoadTokenizer returns an availability error on unsupported builds.
func LoadTokenizer(_ string) (*Tokenizer, error) {
	return nil, errors.New("mlx: native MLX support is unavailable in this build")
}

// Generate returns an availability error on unsupported builds.
func (m *Model) Generate(_ string, _ ...GenerateOption) (string, error) {
	return "", errors.New("mlx: native MLX support is unavailable in this build")
}

// GenerateStream closes immediately on unsupported builds.
func (m *Model) GenerateStream(_ string, _ ...GenerateOption) <-chan Token {
	ch := make(chan Token)
	close(ch)
	return ch
}

// Err returns the availability error on unsupported builds.
func (m *Model) Err() error {
	return errors.New("mlx: native MLX support is unavailable in this build")
}

// Info returns zero values on unsupported builds.
func (m *Model) Info() ModelInfo { return ModelInfo{} }

// Tokenizer returns nil on unsupported builds.
func (m *Model) Tokenizer() *Tokenizer { return nil }

// Close is a no-op on unsupported builds.
func (m *Model) Close() error { return nil }

// Encode returns an availability error on unsupported builds.
func (t *Tokenizer) Encode(_ string) ([]int32, error) {
	return nil, errors.New("mlx: native MLX support is unavailable in this build")
}

// Decode returns an availability error on unsupported builds.
func (t *Tokenizer) Decode(_ []int32) (string, error) {
	return "", errors.New("mlx: native MLX support is unavailable in this build")
}

// TokenID returns false on unsupported builds.
func (t *Tokenizer) TokenID(_ string) (int32, bool) { return 0, false }

// IDToken returns an empty string on unsupported builds.
func (t *Tokenizer) IDToken(_ int32) string { return "" }

// BOS returns 0 on unsupported builds.
func (t *Tokenizer) BOS() int32 { return 0 }

// EOS returns 0 on unsupported builds.
func (t *Tokenizer) EOS() int32 { return 0 }

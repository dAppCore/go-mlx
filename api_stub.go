//go:build !(darwin && arm64) || nomlx

package mlx

import "errors"

// Model is a stub on unsupported builds.
type Model struct{}

// LoadModel returns an availability error on unsupported builds.
func LoadModel(_ string, _ ...LoadOption) (*Model, error) {
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

// NewLoRA returns nil on unsupported builds.
func NewLoRA(_ *Model, _ *LoRAConfig) *LoRAAdapter { return nil }

// MergeLoRA is a no-op on unsupported builds.
func (m *Model) MergeLoRA(_ *LoRAAdapter) *Model { return m }

// SPDX-Licence-Identifier: EUPL-1.2

//go:build !(darwin && arm64) || nomlx

package mlx

import (
	"context"

	core "dappco.re/go"
)

// Model is a stub on unsupported builds.
type Model struct{}

// LoadModel returns an availability error on unsupported builds.
func LoadModel(_ string, _ ...LoadOption) (*Model, error) {
	return nil, core.NewError("mlx: native MLX support is unavailable in this build")
}

// Generate returns an availability error on unsupported builds.
func (m *Model) Generate(_ string, _ ...GenerateOption) (string, error) {
	return "", core.NewError("mlx: native MLX support is unavailable in this build")
}

// Chat returns an availability error on unsupported builds.
func (m *Model) Chat(_ []Message, _ ...GenerateOption) (string, error) {
	return "", core.NewError("mlx: native MLX support is unavailable in this build")
}

// GenerateStream closes immediately on unsupported builds.
func (m *Model) GenerateStream(_ context.Context, _ string, _ ...GenerateOption) <-chan Token {
	ch := make(chan Token)
	close(ch)
	return ch
}

// ChatStream closes immediately on unsupported builds.
func (m *Model) ChatStream(_ context.Context, _ []Message, _ ...GenerateOption) <-chan Token {
	ch := make(chan Token)
	close(ch)
	return ch
}

// Classify returns an availability error on unsupported builds.
func (m *Model) Classify(_ []string, _ ...GenerateOption) ([]ClassifyResult, error) {
	return nil, core.NewError("mlx: native MLX support is unavailable in this build")
}

// BatchGenerate returns an availability error on unsupported builds.
func (m *Model) BatchGenerate(_ []string, _ ...GenerateOption) ([]BatchResult, error) {
	return nil, core.NewError("mlx: native MLX support is unavailable in this build")
}

// Err returns the availability error on unsupported builds.
func (m *Model) Err() error {
	return core.NewError("mlx: native MLX support is unavailable in this build")
}

// Metrics returns zero values on unsupported builds.
func (m *Model) Metrics() Metrics { return Metrics{} }

// ModelType returns an empty string on unsupported builds.
func (m *Model) ModelType() string { return "" }

// Info returns zero values on unsupported builds.
func (m *Model) Info() ModelInfo { return ModelInfo{} }

// InspectAttention returns an availability error on unsupported builds.
func (m *Model) InspectAttention(_ string) (*AttentionSnapshot, error) {
	return nil, core.NewError("mlx: native MLX support is unavailable in this build")
}

// Tokenizer returns nil on unsupported builds.
func (m *Model) Tokenizer() *Tokenizer { return nil }

// Close is a no-op on unsupported builds.
func (m *Model) Close() error { return nil }

// NewLoRA returns nil on unsupported builds.
func NewLoRA(_ *Model, _ *LoRAConfig) *LoRAAdapter { return nil }

// MergeLoRA is a no-op on unsupported builds.
func (m *Model) MergeLoRA(_ *LoRAAdapter) *Model { return m }

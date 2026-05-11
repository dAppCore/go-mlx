// SPDX-Licence-Identifier: EUPL-1.2

//go:build !(darwin && arm64) || nomlx

package mlx

import (
	"context"
	"iter"

	core "dappco.re/go"
	"dappco.re/go/mlx/lora"
	memvid "dappco.re/go/inference/state"
	"dappco.re/go/mlx/kv"
)

// Model is a stub on unsupported builds.
type Model struct{}

// ModelSession is unavailable on unsupported builds.
type ModelSession struct{}

// LoadModel returns an availability error on unsupported builds.
func LoadModel(_ string, _ ...LoadOption) (*Model, error) {
	return nil, core.NewError("mlx: native MLX support is unavailable in this build")
}

// Generate returns an availability error on unsupported builds.
func (m *Model) Generate(_ string, _ ...GenerateOption) (string, error) {
	return "", core.NewError("mlx: native MLX support is unavailable in this build")
}

// GenerateChunks returns an availability error on unsupported builds.
func (m *Model) GenerateChunks(_ context.Context, _ iter.Seq[string], _ ...GenerateOption) (string, error) {
	return "", core.NewError("mlx: native MLX support is unavailable in this build")
}

// Chat returns an availability error on unsupported builds.
func (m *Model) Chat(_ []Message, _ ...GenerateOption) (string, error) {
	return "", core.NewError("mlx: native MLX support is unavailable in this build")
}

// WarmPromptCache returns an availability error on unsupported builds.
func (m *Model) WarmPromptCache(_ string) error {
	return core.NewError("mlx: native MLX support is unavailable in this build")
}

// WarmPromptCacheChunks returns an availability error on unsupported builds.
func (m *Model) WarmPromptCacheChunks(_ context.Context, _ iter.Seq[string]) error {
	return core.NewError("mlx: native MLX support is unavailable in this build")
}

// WarmPromptCacheFromKV returns an availability error on unsupported builds.
func (m *Model) WarmPromptCacheFromKV(_ *kv.Snapshot) error {
	return core.NewError("mlx: native MLX support is unavailable in this build")
}

// WarmPromptCacheFromMemvidBlocks returns an availability error on unsupported builds.
func (m *Model) WarmPromptCacheFromMemvidBlocks(_ context.Context, _ memvid.Store, _ *kv.MemvidBlockBundle, _ int) error {
	return core.NewError("mlx: native MLX support is unavailable in this build")
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

// Adapter returns no active adapter on unsupported builds.
func (m *Model) Adapter() lora.AdapterInfo { return lora.AdapterInfo{} }

// InspectAttention returns an availability error on unsupported builds.
func (m *Model) InspectAttention(_ string) (*AttentionSnapshot, error) {
	return nil, core.NewError("mlx: native MLX support is unavailable in this build")
}

// CaptureKV returns an availability error on unsupported builds.
func (m *Model) CaptureKV(_ string) (*kv.Snapshot, error) {
	return nil, core.NewError("mlx: native MLX support is unavailable in this build")
}

// CaptureKVWithOptions returns an availability error on unsupported builds.
func (m *Model) CaptureKVWithOptions(_ string, _ kv.CaptureOptions) (*kv.Snapshot, error) {
	return nil, core.NewError("mlx: native MLX support is unavailable in this build")
}

// CaptureKVChunks returns an availability error on unsupported builds.
func (m *Model) CaptureKVChunks(_ context.Context, _ iter.Seq[string]) (*kv.Snapshot, error) {
	return nil, core.NewError("mlx: native MLX support is unavailable in this build")
}

// CaptureKVChunksWithOptions returns an availability error on unsupported builds.
func (m *Model) CaptureKVChunksWithOptions(_ context.Context, _ iter.Seq[string], _ kv.CaptureOptions) (*kv.Snapshot, error) {
	return nil, core.NewError("mlx: native MLX support is unavailable in this build")
}

// NewSession returns an availability error on unsupported builds.
func (m *Model) NewSession() (*ModelSession, error) {
	return nil, core.NewError("mlx: native MLX support is unavailable in this build")
}

// NewSessionFromKV returns an availability error on unsupported builds.
func (m *Model) NewSessionFromKV(_ *kv.Snapshot) (*ModelSession, error) {
	return nil, core.NewError("mlx: native MLX support is unavailable in this build")
}

// NewSessionFromBundle returns an availability error on unsupported builds.
func (m *Model) NewSessionFromBundle(_ *StateBundle) (*ModelSession, error) {
	return nil, core.NewError("mlx: native MLX support is unavailable in this build")
}

// Tokenizer returns nil on unsupported builds.
func (m *Model) Tokenizer() *Tokenizer { return nil }

// Close is a no-op on unsupported builds.
func (m *Model) Close() error { return nil }

// NewLoRA returns nil on unsupported builds.
func NewLoRA(_ *Model, _ *LoRAConfig) *LoRAAdapter { return nil }

// LoadLoRA returns an availability error on unsupported builds.
func (m *Model) LoadLoRA(_ string) (*LoRAAdapter, error) { return nil, unsupportedBuildError() }

// UnloadLoRA returns an availability error on unsupported builds.
func (m *Model) UnloadLoRA() error { return unsupportedBuildError() }

// SwapLoRA returns an availability error on unsupported builds.
func (m *Model) SwapLoRA(_ string) (*LoRAAdapter, error) { return nil, unsupportedBuildError() }

// MergeLoRA is a no-op on unsupported builds.
func (m *Model) MergeLoRA(_ *LoRAAdapter) *Model { return m }

// Prefill returns an availability error on unsupported builds.
func (s *ModelSession) Prefill(_ string) error {
	return core.NewError("mlx: native MLX support is unavailable in this build")
}

// AppendPrompt returns an availability error on unsupported builds.
func (s *ModelSession) AppendPrompt(_ string) error {
	return core.NewError("mlx: native MLX support is unavailable in this build")
}

// Generate returns an availability error on unsupported builds.
func (s *ModelSession) Generate(_ ...GenerateOption) (string, error) {
	return "", core.NewError("mlx: native MLX support is unavailable in this build")
}

// GenerateStream closes immediately on unsupported builds.
func (s *ModelSession) GenerateStream(_ context.Context, _ ...GenerateOption) <-chan Token {
	ch := make(chan Token)
	close(ch)
	return ch
}

// CaptureKV returns an availability error on unsupported builds.
func (s *ModelSession) CaptureKV() (*kv.Snapshot, error) {
	return nil, core.NewError("mlx: native MLX support is unavailable in this build")
}

// CaptureKVWithOptions returns an availability error on unsupported builds.
func (s *ModelSession) CaptureKVWithOptions(_ kv.CaptureOptions) (*kv.Snapshot, error) {
	return nil, core.NewError("mlx: native MLX support is unavailable in this build")
}

// kv.Analyze returns an availability error on unsupported builds.
func (s *ModelSession) AnalyzeKV() (*kv.Analysis, error) {
	return nil, core.NewError("mlx: native MLX support is unavailable in this build")
}

// SaveKV returns an availability error on unsupported builds.
func (s *ModelSession) SaveKV(_ string) error {
	return core.NewError("mlx: native MLX support is unavailable in this build")
}

// RestoreKV returns an availability error on unsupported builds.
func (s *ModelSession) RestoreKV(_ *kv.Snapshot) error {
	return core.NewError("mlx: native MLX support is unavailable in this build")
}

// LoadKV returns an availability error on unsupported builds.
func (s *ModelSession) LoadKV(_ string) error {
	return core.NewError("mlx: native MLX support is unavailable in this build")
}

// SaveKVToMemvid returns an availability error on unsupported builds.
func (s *ModelSession) SaveKVToMemvid(_ context.Context, _ memvid.Writer, _ kv.MemvidOptions) (memvid.ChunkRef, error) {
	return memvid.ChunkRef{}, core.NewError("mlx: native MLX support is unavailable in this build")
}

// LoadKVFromMemvid returns an availability error on unsupported builds.
func (s *ModelSession) LoadKVFromMemvid(_ context.Context, _ memvid.Store, _ memvid.ChunkRef) error {
	return core.NewError("mlx: native MLX support is unavailable in this build")
}

// SaveKVBlocksToMemvid returns an availability error on unsupported builds.
func (s *ModelSession) SaveKVBlocksToMemvid(_ context.Context, _ memvid.Writer, _ kv.MemvidBlockOptions) (*kv.MemvidBlockBundle, error) {
	return nil, core.NewError("mlx: native MLX support is unavailable in this build")
}

// LoadKVBlocksFromMemvid returns an availability error on unsupported builds.
func (s *ModelSession) LoadKVBlocksFromMemvid(_ context.Context, _ memvid.Store, _ *kv.MemvidBlockBundle) error {
	return core.NewError("mlx: native MLX support is unavailable in this build")
}

// RestoreBundle returns an availability error on unsupported builds.
func (s *ModelSession) RestoreBundle(_ *StateBundle) error {
	return core.NewError("mlx: native MLX support is unavailable in this build")
}

// RestoreBundleFromMemvid returns an availability error on unsupported builds.
func (s *ModelSession) RestoreBundleFromMemvid(_ context.Context, _ *StateBundle, _ memvid.Store) error {
	return core.NewError("mlx: native MLX support is unavailable in this build")
}

// LoadBundle returns an availability error on unsupported builds.
func (s *ModelSession) LoadBundle(_ string) error {
	return core.NewError("mlx: native MLX support is unavailable in this build")
}

// Fork returns an availability error on unsupported builds.
func (s *ModelSession) Fork() (*ModelSession, error) {
	return nil, core.NewError("mlx: native MLX support is unavailable in this build")
}

// Reset is a no-op on unsupported builds.
func (s *ModelSession) Reset() {}

// Close is a no-op on unsupported builds.
func (s *ModelSession) Close() error { return nil }

// Err returns nil on unsupported builds.
func (s *ModelSession) Err() error { return nil }

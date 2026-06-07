// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"
	"iter"

	"dappco.re/go/mlx/pkg/metal"
)

// native_model.go: the native model contract — the interface the metal engine
// implementation satisfies, plus the optional capability interfaces (prompt-cache
// warming, KV snapshotting, chunked generation, LoRA load/unload) the root probes for.

type NativeModel interface {
	ApplyLoRA(metal.LoRAConfig) *metal.LoRAAdapter
	BatchGenerate(context.Context, []string, metal.GenerateConfig) ([]metal.BatchResult, error)
	Chat(context.Context, []metal.ChatMessage, metal.GenerateConfig) iter.Seq[metal.Token]
	Classify(context.Context, []string, metal.GenerateConfig, bool) ([]metal.ClassifyResult, error)
	Close() error
	Err() error
	Generate(context.Context, string, metal.GenerateConfig) iter.Seq[metal.Token]
	Info() metal.ModelInfo
	InspectAttention(context.Context, string) (*metal.AttentionResult, error)
	LastMetrics() metal.Metrics
	ModelType() string
	Tokenizer() *metal.Tokenizer
}

// NewModel wraps an already-constructed native engine in a root Model. It is the
// construction seam for subpackage tests and for callers that build a
// NativeModel directly; LoadModel is the usual on-disk path.
//
//	m := mlx.NewModel(engine) // engine implements mlx.NativeModel
func NewModel(native NativeModel) *Model {
	return &Model{model: native}
}

// Native returns the underlying native engine, or nil for a nil Model. It is the
// accessor subpackages build on instead of reaching the unexported field.
//
//	engine := m.Native()
func (m *Model) Native() NativeModel {
	if m == nil {
		return nil
	}
	return m.model
}

type nativePromptCacheWarmer interface {
	WarmPromptCache(context.Context, string) error
}

type nativePromptCacheChunkWarmer interface {
	WarmPromptCacheChunks(context.Context, iter.Seq[string]) error
}

type nativePromptCacheClearer interface {
	ClearPromptCache()
}

type nativePromptCacheKVRestorer interface {
	RestorePromptCacheFromKV(context.Context, *metal.KVSnapshot) error
}

type nativePromptCacheKVBlockRestorer interface {
	RestorePromptCacheFromKVBlocks(context.Context, metal.KVSnapshotBlockSource) error
}

type nativeKVSnapshotter interface {
	CaptureKV(context.Context, string) (*metal.KVSnapshot, error)
}

type nativeKVSnapshotterWithOptions interface {
	CaptureKVWithOptions(context.Context, string, metal.KVSnapshotCaptureOptions) (*metal.KVSnapshot, error)
}

type nativeKVChunkSnapshotter interface {
	CaptureKVChunks(context.Context, iter.Seq[string]) (*metal.KVSnapshot, error)
}

type nativeKVChunkSnapshotterWithOptions interface {
	CaptureKVChunksWithOptions(context.Context, iter.Seq[string], metal.KVSnapshotCaptureOptions) (*metal.KVSnapshot, error)
}

type nativeChunkGenerator interface {
	GenerateChunks(context.Context, iter.Seq[string], metal.GenerateConfig) iter.Seq[metal.Token]
}

type nativeChatChunkGenerator interface {
	ChatChunks(context.Context, []metal.ChatMessage, int, metal.GenerateConfig) iter.Seq[metal.Token]
}

type nativeLoRALoader interface {
	LoadLoRA(string) (*metal.LoRAAdapter, error)
}

type nativeLoRAUnloader interface {
	UnloadLoRA() error
}

// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"
	"iter"

	state "dappco.re/go/inference/state"
	"dappco.re/go/inference/kv"
	"dappco.re/go/mlx/kvconv"
	"dappco.re/go/mlx/spine"
)

// prompt_cache.go: Model prompt-cache warming — prefilling the token-prefix cache
// from a prompt, streamed chunks, a KV snapshot, or persisted state/memvid blocks.

// WarmPromptCache prefills the exact token-prefix cache for a stable prompt prefix.
func (m *Model) WarmPromptCache(prompt string) error {
	if m == nil || m.model == nil {
		return errMLXModelNil
	}
	warmer, ok := m.model.(nativePromptCacheWarmer)
	if !ok {
		return errMLXPromptCacheWarmUnsupp
	}
	return warmer.WarmPromptCache(context.Background(), prompt)
}

// WarmPromptCacheChunks prefills the exact token-prefix cache from streaming
// prompt chunks without building or tokenizing one giant prompt string.
func (m *Model) WarmPromptCacheChunks(ctx context.Context, chunks iter.Seq[string]) error {
	if ctx == nil {
		ctx = context.Background()
	}
	if m == nil || m.model == nil {
		return errMLXModelNil
	}
	if warmer, ok := m.model.(nativePromptCacheChunkWarmer); ok {
		return warmer.WarmPromptCacheChunks(ctx, chunks)
	}
	return m.WarmPromptCache(spine.PromptChunksToString(chunks))
}

// ClearPromptCache drops the exact token-prefix KV cache without unloading the
// model. TRAD comparison runners use this to force a fresh prefill between
// turns while keeping the same loaded weights.
func (m *Model) ClearPromptCache() error {
	if m == nil || m.model == nil {
		return errMLXModelNil
	}
	clearer, ok := m.model.(nativePromptCacheClearer)
	if !ok {
		return errMLXPromptCacheClearUnsupp
	}
	clearer.ClearPromptCache()
	return nil
}

// WarmPromptCacheFromKV installs a captured K/V prefix directly as the model prompt cache.
func (m *Model) WarmPromptCacheFromKV(snapshot *kv.Snapshot) error {
	if m == nil || m.model == nil {
		return errMLXModelNil
	}
	restorer, ok := m.model.(nativePromptCacheKVRestorer)
	if !ok {
		return errMLXKVPromptRestoreUnsupp
	}
	return restorer.RestorePromptCacheFromKV(context.Background(), kvconv.ToMetalKVSnapshot(snapshot))
}

// WarmPromptCacheFromStateBlocks loads the requested State KV prefix blocks and
// installs them directly as the model prompt cache.
func (m *Model) WarmPromptCacheFromStateBlocks(ctx context.Context, store state.Store, bundle *kv.StateBlockBundle, prefixTokens int) error {
	if ctx == nil {
		ctx = context.Background()
	}
	if m == nil || m.model == nil {
		return errMLXModelNil
	}
	if restorer, ok := m.model.(nativePromptCacheKVBlockRestorer); ok {
		source, err := kvconv.MetalKVSnapshotBlockSource(ctx, store, bundle, prefixTokens)
		if err != nil {
			return err
		}
		return restorer.RestorePromptCacheFromKVBlocks(ctx, source)
	}
	snapshot, err := kv.LoadPrefixFromStateBlocks(ctx, store, bundle, prefixTokens)
	if err != nil {
		return err
	}
	restorer, ok := m.model.(nativePromptCacheKVRestorer)
	if !ok {
		return errMLXKVPromptRestoreUnsupp
	}
	return restorer.RestorePromptCacheFromKV(ctx, kvconv.ToMetalKVSnapshot(snapshot))
}

// WarmPromptCacheFromMemvidBlocks loads the requested old memvid-named State
// KV prefix blocks and installs them directly as the model prompt cache.
//
// Deprecated: use WarmPromptCacheFromStateBlocks.
func (m *Model) WarmPromptCacheFromMemvidBlocks(ctx context.Context, store state.Store, bundle *kv.MemvidBlockBundle, prefixTokens int) error {
	return m.WarmPromptCacheFromStateBlocks(ctx, store, bundle, prefixTokens)
}

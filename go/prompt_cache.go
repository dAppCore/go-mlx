// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"
	"iter"

	state "dappco.re/go/inference/state"
	"dappco.re/go/mlx/kv"
	"dappco.re/go/mlx/kvconv"
	"dappco.re/go/mlx/pkg/metal"
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
	return m.WarmPromptCache(promptChunksToString(chunks))
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
		source, err := metalKVSnapshotBlockSource(ctx, store, bundle, prefixTokens)
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

func metalKVSnapshotBlockSource(ctx context.Context, store state.Store, bundle *kv.StateBlockBundle, prefixTokens int) (metal.KVSnapshotBlockSource, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if store == nil {
		return metal.KVSnapshotBlockSource{}, errMLXStateKVStoreNil
	}
	if err := kv.ValidateStateBlockBundle(bundle); err != nil {
		return metal.KVSnapshotBlockSource{}, err
	}
	if prefixTokens <= 0 {
		prefixTokens = bundle.TokenCount
	}
	if prefixTokens > bundle.TokenCount {
		return metal.KVSnapshotBlockSource{}, errMLXStateKVPrefixExceeds
	}
	blocks := bundle.Blocks
	blockCount, err := metalKVSnapshotBlockSourceCoverage(blocks, prefixTokens)
	if err != nil {
		return metal.KVSnapshotBlockSource{}, err
	}
	source := metal.KVSnapshotBlockSource{
		TokenCount:   bundle.TokenCount,
		PrefixTokens: prefixTokens,
		BlockCount:   blockCount,
	}
	// Hoist invariants out of the per-block closure. KVEncoding is bundle-
	// scoped — checking it once at construction lets each Load call use
	// the captured loadOpts directly without re-branching on every block.
	loadOpts := kv.LoadOptions{}
	if bundle.KVEncoding == kv.EncodingNative {
		loadOpts.RawKVOnly = true
	}
	source.Load = func(loadCtx context.Context, index int) (metal.KVSnapshotBlock, error) {
		if loadCtx == nil {
			loadCtx = ctx
		}
		if index < 0 || index >= blockCount {
			return metal.KVSnapshotBlock{}, errMLXStateKVBlockOutOfRange
		}
		ref := &blocks[index]
		block, err := kv.LoadStateBlockWithOptions(loadCtx, store, *ref, loadOpts)
		if err != nil {
			return metal.KVSnapshotBlock{}, err
		}
		if block.TokenStart != ref.TokenStart || block.TokenCount != ref.TokenCount {
			return metal.KVSnapshotBlock{}, errMLXStateKVBlockMetaMismatch
		}
		snapshot := block.Snapshot
		if snapshot == nil {
			return metal.KVSnapshotBlock{}, errMLXStateKVBlockSnapshotNil
		}
		if block.TokenStart+block.TokenCount > prefixTokens {
			trimTokens := prefixTokens - block.TokenStart
			if trimTokens <= 0 {
				return metal.KVSnapshotBlock{}, errMLXStateKVPrefixInvalidTrim
			}
			baseOffset := max(kv.EffectiveTokenOffset(snapshot)-kv.EffectiveSeqLen(snapshot), 0)
			trimmed, trimErr := snapshot.SliceBlock(0, trimTokens, baseOffset, false)
			if trimErr != nil {
				return metal.KVSnapshotBlock{}, trimErr
			}
			snapshot = trimmed
			block.TokenCount = trimTokens
		}
		if block.TokenStart+block.TokenCount < bundle.TokenCount {
			kv.ClearTerminalState(snapshot)
		}
		return metal.KVSnapshotBlock{
			Index:      index,
			TokenStart: block.TokenStart,
			TokenCount: block.TokenCount,
			Snapshot:   kvconv.ToMetalKVSnapshot(snapshot),
		}, nil
	}
	return source, nil
}

func metalKVSnapshotBlockSourceCoverage(blocks []kv.StateBlockRef, prefixTokens int) (int, error) {
	if len(blocks) == 0 {
		return 0, errMLXStateKVPrefixNoCovering
	}
	nextStart := 0
	blockCount := 0
	for i := range blocks {
		ref := &blocks[i]
		if ref.TokenStart >= prefixTokens {
			break
		}
		if ref.Index != i || ref.TokenStart != nextStart || ref.TokenCount <= 0 {
			return 0, errMLXStateKVBlockMetaMismatch
		}
		nextStart += ref.TokenCount
		blockCount++
		if nextStart >= prefixTokens {
			break
		}
	}
	if blockCount == 0 || nextStart < prefixTokens {
		return 0, errMLXStateKVPrefixNoCovering
	}
	return blockCount, nil
}

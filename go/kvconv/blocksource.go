// SPDX-Licence-Identifier: EUPL-1.2

package kvconv

import (
	"context"

	core "dappco.re/go"
	state "dappco.re/go/inference/state"
	"dappco.re/go/mlx/kv"
	"dappco.re/go/mlx/pkg/metal"
)

// blocksource.go: building a metal.KVSnapshotBlockSource from persisted State KV
// blocks — the streamed, per-block restore path. Root-type-free (state/kv/metal
// only), so it lives here alongside the kv<->metal snapshot bridge rather than in
// the root package, letting both root and the session subpackage consume it.

var (
	errStateKVStoreNil          = core.NewError("mlx: state store is nil")
	errStateKVPrefixExceeds     = core.NewError("mlx: State KV prefix exceeds bundle token count")
	errStateKVPrefixNoCovering  = core.NewError("mlx: State KV prefix has no covering blocks")
	errStateKVBlockOutOfRange   = core.NewError("mlx: State KV block index is out of range")
	errStateKVBlockMetaMismatch = core.NewError("mlx: State KV block metadata mismatch")
	errStateKVBlockSnapshotNil  = core.NewError("mlx: State KV block snapshot is nil")
	errStateKVPrefixInvalidTrim = core.NewError("mlx: State KV prefix has invalid trim range")
)

// MetalKVSnapshotBlockSource builds a streamed block source that lazily loads
// and trims the State KV blocks covering prefixTokens.
//
//	src, err := kvconv.MetalKVSnapshotBlockSource(ctx, store, bundle, prefixTokens)
func MetalKVSnapshotBlockSource(ctx context.Context, store state.Store, bundle *kv.StateBlockBundle, prefixTokens int) (metal.KVSnapshotBlockSource, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if store == nil {
		return metal.KVSnapshotBlockSource{}, errStateKVStoreNil
	}
	if err := kv.ValidateStateBlockBundle(bundle); err != nil {
		return metal.KVSnapshotBlockSource{}, err
	}
	if prefixTokens <= 0 {
		prefixTokens = bundle.TokenCount
	}
	if prefixTokens > bundle.TokenCount {
		return metal.KVSnapshotBlockSource{}, errStateKVPrefixExceeds
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
			return metal.KVSnapshotBlock{}, errStateKVBlockOutOfRange
		}
		ref := &blocks[index]
		block, err := kv.LoadStateBlockWithOptions(loadCtx, store, *ref, loadOpts)
		if err != nil {
			return metal.KVSnapshotBlock{}, err
		}
		if block.TokenStart != ref.TokenStart || block.TokenCount != ref.TokenCount {
			return metal.KVSnapshotBlock{}, errStateKVBlockMetaMismatch
		}
		snapshot := block.Snapshot
		if snapshot == nil {
			return metal.KVSnapshotBlock{}, errStateKVBlockSnapshotNil
		}
		if block.TokenStart+block.TokenCount > prefixTokens {
			trimTokens := prefixTokens - block.TokenStart
			if trimTokens <= 0 {
				return metal.KVSnapshotBlock{}, errStateKVPrefixInvalidTrim
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
			Snapshot:   ToMetalKVSnapshot(snapshot),
		}, nil
	}
	return source, nil
}

func metalKVSnapshotBlockSourceCoverage(blocks []kv.StateBlockRef, prefixTokens int) (int, error) {
	if len(blocks) == 0 {
		return 0, errStateKVPrefixNoCovering
	}
	nextStart := 0
	blockCount := 0
	for i := range blocks {
		ref := &blocks[i]
		if ref.TokenStart >= prefixTokens {
			break
		}
		if ref.Index != i || ref.TokenStart != nextStart || ref.TokenCount <= 0 {
			return 0, errStateKVBlockMetaMismatch
		}
		nextStart += ref.TokenCount
		blockCount++
		if nextStart >= prefixTokens {
			break
		}
	}
	if blockCount == 0 || nextStart < prefixTokens {
		return 0, errStateKVPrefixNoCovering
	}
	return blockCount, nil
}

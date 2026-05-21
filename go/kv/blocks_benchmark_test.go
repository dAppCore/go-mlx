// SPDX-Licence-Identifier: EUPL-1.2

package kv

import (
	"context"
	"testing"

	memvid "dappco.re/go/inference/state"
)

var (
	stateBlocksBenchmarkSnapshot *Snapshot
	stateBlocksBenchmarkTokens   []int32
)

func BenchmarkLoadPrefixFromStateBlocks_MixedWindowThreeBlocks(b *testing.B) {
	ctx := context.Background()
	store, bundle := benchmarkStateBlocksFixture(b)
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		snapshot, err := LoadPrefixFromStateBlocksWithOptions(ctx, store, bundle, bundle.TokenCount, LoadOptions{RawKVOnly: true})
		if err != nil {
			b.Fatal(err)
		}
		stateBlocksBenchmarkSnapshot = snapshot
	}
}

func BenchmarkLoadPrefixTokensFromStateBlocks_MixedWindowThreeBlocks(b *testing.B) {
	ctx := context.Background()
	store, bundle := benchmarkStateBlocksFixture(b)
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		tokens, err := LoadPrefixTokensFromStateBlocksWithOptions(ctx, store, bundle, bundle.TokenCount, LoadOptions{RawKVOnly: true})
		if err != nil {
			b.Fatal(err)
		}
		stateBlocksBenchmarkTokens = tokens
	}
}

func benchmarkStateBlocksFixture(tb testing.TB) (memvid.Store, *StateBlockBundle) {
	tb.Helper()
	store := memvid.NewInMemoryStore(nil)
	snapshot := benchmarkStateBlocksSnapshot(1536, 512)
	bundle, err := snapshot.SaveStateBlocks(context.Background(), store, StateBlockOptions{
		BlockSize:  512,
		KVEncoding: EncodingNative,
	})
	if err != nil {
		tb.Fatalf("SaveStateBlocks() error = %v", err)
	}
	if len(bundle.Blocks) != 3 {
		tb.Fatalf("blocks = %d, want 3", len(bundle.Blocks))
	}
	return store, bundle
}

func benchmarkStateBlocksSnapshot(tokenCount, localWindow int) *Snapshot {
	tokens := make([]int32, tokenCount)
	fullKey := make([]float32, tokenCount)
	fullValue := make([]float32, tokenCount)
	localKey := make([]float32, localWindow)
	localValue := make([]float32, localWindow)
	for i := range tokenCount {
		tokens[i] = int32(i + 1)
		fullKey[i] = float32(i)
		fullValue[i] = float32(i + 1000)
	}
	for i := range localWindow {
		localKey[i] = float32(i + 2000)
		localValue[i] = float32(i + 3000)
	}
	return &Snapshot{
		Version:       SnapshotVersion,
		Architecture:  "gemma4_text",
		Tokens:        tokens,
		TokenOffset:   tokenCount,
		NumLayers:     2,
		NumHeads:      1,
		SeqLen:        tokenCount,
		HeadDim:       1,
		NumQueryHeads: 1,
		Layers: []LayerSnapshot{
			{
				Layer:      0,
				CacheIndex: 0,
				Heads: []HeadSnapshot{{
					Key:   fullKey,
					Value: fullValue,
				}},
			},
			{
				Layer:      1,
				CacheIndex: 1,
				Heads: []HeadSnapshot{{
					Key:   localKey,
					Value: localValue,
				}},
			},
		},
	}
}

// SPDX-Licence-Identifier: EUPL-1.2

// Lifecycle benches — surfaces that aren't the encoder/block hot paths
// but get hit on the wider session-resume / cache-mode comparison
// trail. Pegs CompareModes (currently un-benched), the full SaveState
// + LoadFromState envelope round-trip (the JSON+base64 cold-store path
// distinct from SaveStateBlocks raw-binary), and concurrent-shape
// patterns: back-to-back writes and mixed read/write sequences on a
// shared in-memory store, single-goroutine for now.
//
// Coverage map (W7-F deepening pass):
//
//   - CompareModes default config (un-benched currently)
//   - CompareModes long-context config (the LARQL / 128k path)
//   - SaveState + LoadFromState envelope round-trip @ 512 / 2048 tokens
//     — the JSON+base64 cold-store path used by the State video codec
//   - 5x back-to-back SaveStateBlocks on a shared store — measures the
//     repeated-checkpoint pattern Virgil writes during a long turn.
//   - Mixed sequence — SaveStateBlocks → LoadPrefixTokens → SliceBlock
//     → SaveStateBlocks (the prompt-cache reuse cycle in miniature).
//
// Run: go test -bench='BenchmarkLifecycle' -benchmem -run='^$' ./go/kv

package kv

import (
	"context"
	"testing"

	state "dappco.re/go/inference/state"
	"dappco.re/go/mlx/memory"
)

// --- CompareModes — un-benched mode-comparison surface ---

func BenchmarkLifecycle_CompareModes_Default(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkReport = CompareModes(BenchConfig{})
	}
}

func BenchmarkLifecycle_CompareModes_LongContext(b *testing.B) {
	cfg := BenchConfig{
		ContextLength: 131072,
		NumLayers:     32,
		HiddenSize:    3072,
		Modes: []memory.KVCacheMode{
			memory.KVCacheModeFP16,
			memory.KVCacheModeQ8,
			memory.KVCacheModeKQ8VQ4,
			memory.KVCacheModePaged,
		},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkReport = CompareModes(cfg)
	}
}

func BenchmarkLifecycle_CompareModes_ByMode(b *testing.B) {
	report := CompareModes(BenchConfig{
		ContextLength: 32768,
		NumLayers:     32,
		HiddenSize:    3072,
	})
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkModeBench = report.ByMode(memory.KVCacheModeQ8)
	}
}

// --- SaveState + LoadFromState envelope round-trip (JSON+base64 cold
// store path, distinct from SaveStateBlocks raw-binary). ---

func BenchmarkLifecycle_SaveStateLoadFromState_512Tokens(b *testing.B) {
	benchSaveStateLoadFromState(b, 512)
}

func BenchmarkLifecycle_SaveStateLoadFromState_2048Tokens(b *testing.B) {
	benchSaveStateLoadFromState(b, 2048)
}

func benchSaveStateLoadFromState(b *testing.B, tokens int) {
	b.Helper()
	snap := benchSnapshot(tokens)
	opts := StateOptions{KVEncoding: EncodingNative, URI: "state://benchsite/snapshot"}
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		store := state.NewInMemoryStore(nil)
		ref, err := snap.SaveState(ctx, store, opts)
		if err != nil {
			b.Fatal(err)
		}
		out, err := LoadFromState(ctx, store, ref)
		if err != nil {
			b.Fatal(err)
		}
		benchSinkSnapshot = out
		benchSinkRef = ref
	}
}

// --- 5x back-to-back SaveStateBlocks on a shared store. Measures the
// repeated-checkpoint pattern Virgil writes during a long turn — each
// SaveStateBlocks call appends to the InMemoryStore. Single-goroutine.
// ---

func BenchmarkLifecycle_BackToBack_SaveStateBlocks_x5(b *testing.B) {
	snap := benchSnapshot(1536)
	opts := StateBlockOptions{BlockSize: 512, KVEncoding: EncodingNative}
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		store := state.NewInMemoryStore(nil)
		for range 5 {
			bundle, err := snap.SaveStateBlocks(ctx, store, opts)
			if err != nil {
				b.Fatal(err)
			}
			if bundle != nil && len(bundle.Blocks) > 0 {
				benchSinkRef = bundle.Blocks[0].State
			}
		}
	}
}

// --- Mixed sequence: save → token-prefix-load → slice → save again.
// The prompt-cache reuse cycle in miniature. ---

func BenchmarkLifecycle_MixedSeq_SaveLoadSliceSave(b *testing.B) {
	snap := benchSnapshot(1536)
	opts := StateBlockOptions{BlockSize: 512, KVEncoding: EncodingNative}
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		store := state.NewInMemoryStore(nil)
		// Step 1: save initial bundle
		bundle, err := snap.SaveStateBlocks(ctx, store, opts)
		if err != nil {
			b.Fatal(err)
		}
		// Step 2: warm path — token-only prefix wake
		toks, err := LoadPrefixTokensFromStateBlocks(ctx, store, bundle, 1024)
		if err != nil {
			b.Fatal(err)
		}
		stateBlocksBenchmarkTokens = toks
		// Step 3: full prefix carve-out
		prefix, err := LoadPrefixFromStateBlocksWithOptions(ctx, store, bundle, 1024, LoadOptions{RawKVOnly: true})
		if err != nil {
			b.Fatal(err)
		}
		// Step 4: re-save the carved prefix as a new bundle — the
		// prompt-cache reuse path.
		newBundle, err := prefix.SaveStateBlocks(ctx, store, opts)
		if err != nil {
			b.Fatal(err)
		}
		if newBundle != nil && len(newBundle.Blocks) > 0 {
			benchSinkRef = newBundle.Blocks[0].State
		}
	}
}

// --- ReusePrefix path: a follow-up SaveStateBlocks pointed at the
// first bundle as ReusePrefix avoids re-encoding the blocks already on
// the store. The hash-match-then-skip primitive Virgil uses to compact
// rolling sessions. ---

func BenchmarkLifecycle_SaveStateBlocks_ReusePrefix(b *testing.B) {
	snap := benchSnapshot(1536)
	opts := StateBlockOptions{BlockSize: 512, KVEncoding: EncodingNative}
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		store := state.NewInMemoryStore(nil)
		first, err := snap.SaveStateBlocks(ctx, store, opts)
		if err != nil {
			b.Fatal(err)
		}
		// Second save with first bundle pinned as ReusePrefix at the
		// full token count. All three blocks should hit the
		// reusableKVSnapshotStateBlockRef hash-match branch.
		reuseOpts := opts
		reuseOpts.ReusePrefix = first
		reuseOpts.ReusePrefixTokens = first.TokenCount
		second, err := snap.SaveStateBlocks(ctx, store, reuseOpts)
		if err != nil {
			b.Fatal(err)
		}
		if second.ReusedBlocks != 3 {
			b.Fatalf("ReusedBlocks = %d, want 3", second.ReusedBlocks)
		}
	}
}

// Sinks specific to this file.
var (
	benchSinkReport    BenchReport
	benchSinkModeBench ModeBench
)

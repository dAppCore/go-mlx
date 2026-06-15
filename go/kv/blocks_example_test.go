// SPDX-Licence-Identifier: EUPL-1.2

package kv

import (
	"context"

	core "dappco.re/go"
	state "dappco.re/go/inference/state"
)

// exampleNativeLayerSnapshot builds a 4-token, 2-head snapshot whose K/V live
// as layer-level native float16 slabs ([B,H,L,D] = [1,2,4,1]) rather than
// per-head float32 vectors — the raw-tensor capture shape an MLX layer-cache
// export produces.
func exampleNativeLayerSnapshot() *Snapshot {
	keyBytes := []byte{
		1, 0, 2, 0, 3, 0, 4, 0,
		5, 0, 6, 0, 7, 0, 8, 0,
	}
	valueBytes := []byte{
		11, 0, 12, 0, 13, 0, 14, 0,
		15, 0, 16, 0, 17, 0, 18, 0,
	}
	return &Snapshot{
		Version:       SnapshotVersion,
		Architecture:  "gemma4_text",
		Tokens:        []int32{1, 2, 3, 4},
		TokenOffset:   4,
		NumLayers:     1,
		NumHeads:      2,
		SeqLen:        4,
		HeadDim:       1,
		NumQueryHeads: 2,
		Layers: []LayerSnapshot{{
			Layer:      0,
			CacheIndex: 0,
			KeyDType:   "float16",
			KeyBytes:   keyBytes,
			KeyShape:   []int32{1, 2, 4, 1},
			ValueDType: "float16",
			ValueBytes: valueBytes,
			ValueShape: []int32{1, 2, 4, 1},
			Heads:      make([]HeadSnapshot, 2),
		}},
	}
}

// ExampleAssembleBlocks splits a native-dtype snapshot into fixed-size blocks
// and reassembles it — the in-memory prefill-block round-trip. AssembleBlocks
// stitches the per-block native slabs back into the full-length layer tensors,
// recovering the original token count and raw byte payload exactly.
func ExampleAssembleBlocks() {
	source := exampleNativeLayerSnapshot()

	blocks, err := source.SplitBlocks(2)
	if err != nil {
		core.Println("split error:", err)
		return
	}

	assembled, err := AssembleBlocks(blocks)
	if err != nil {
		core.Println("assemble error:", err)
		return
	}
	core.Println("blocks:", len(blocks))
	core.Println("tokens:", len(assembled.Tokens))
	core.Println("key bytes recovered:", equalBytes(assembled.Layers[0].KeyBytes, source.Layers[0].KeyBytes))
	// Output:
	// blocks: 2
	// tokens: 4
	// key bytes recovered: true
}

// ExampleSnapshot_SaveStateBlocks_native saves a native-dtype snapshot to a
// State store with EncodingNative, then reloads it raw-only. The durable
// save→load→assemble path reconstructs the layer-level slabs from the stored
// blocks without re-expanding them into per-head vectors.
func ExampleSnapshot_SaveStateBlocks_native() {
	ctx := context.Background()
	store := state.NewInMemoryStore(nil)
	source := exampleNativeLayerSnapshot()

	bundle, err := source.SaveStateBlocks(ctx, store, StateBlockOptions{
		BlockSize:  2,
		KVEncoding: EncodingNative,
		URI:        "mlx://session/native",
	})
	if err != nil {
		core.Println("save error:", err)
		return
	}

	loaded, err := LoadFromStateBlocksWithOptions(ctx, store, bundle, LoadOptions{RawKVOnly: true})
	if err != nil {
		core.Println("load error:", err)
		return
	}
	layer := loaded.Layers[0]
	core.Println("blocks:", len(bundle.Blocks))
	core.Println("value bytes recovered:", equalBytes(layer.ValueBytes, source.Layers[0].ValueBytes))
	core.Println("per-head bytes empty:", len(layer.Heads[0].KeyBytes) == 0)
	// Output:
	// blocks: 2
	// value bytes recovered: true
	// per-head bytes empty: true
}

// ExampleSnapshot_SaveStateBlocks splits a snapshot into fixed-size KV blocks
// and writes each one to a State store, returning a manifest.
func ExampleSnapshot_SaveStateBlocks() {
	store := state.NewInMemoryStore(nil)
	snapshot := kvSnapshotBlocksTestSnapshot()

	bundle, err := snapshot.SaveStateBlocks(context.Background(), store, StateBlockOptions{
		BlockSize:  2,
		KVEncoding: EncodingQ8,
		URI:        "mlx://session/blocks",
	})
	if err != nil {
		core.Println("error:", err)
		return
	}
	core.Println("blocks:", len(bundle.Blocks))
	// Output: blocks: 2
}

// ExampleLoadFromStateBlocks restores a full snapshot from a block manifest.
func ExampleLoadFromStateBlocks() {
	ctx := context.Background()
	store := state.NewInMemoryStore(nil)
	bundle, err := kvSnapshotBlocksTestSnapshot().SaveStateBlocks(ctx, store, StateBlockOptions{BlockSize: 2})
	if err != nil {
		core.Println("error:", err)
		return
	}

	loaded, err := LoadFromStateBlocks(ctx, store, bundle)
	if err != nil {
		core.Println("error:", err)
		return
	}
	core.Println("tokens:", len(loaded.Tokens))
	// Output: tokens: 4
}

// ExampleLoadPrefixFromStateBlocks restores only the blocks needed to cover a
// token prefix — the prompt-cache warmup path.
func ExampleLoadPrefixFromStateBlocks() {
	ctx := context.Background()
	store := state.NewInMemoryStore(nil)
	bundle, err := kvSnapshotBlocksTestSnapshot().SaveStateBlocks(ctx, store, StateBlockOptions{BlockSize: 2})
	if err != nil {
		core.Println("error:", err)
		return
	}

	prefix, err := LoadPrefixFromStateBlocks(ctx, store, bundle, 2)
	if err != nil {
		core.Println("error:", err)
		return
	}
	core.Println("prefix tokens:", len(prefix.Tokens))
	// Output: prefix tokens: 2
}

// ExampleEffectiveSeqLen reports the effective sequence length, preferring the
// recorded SeqLen and falling back to the token count.
func ExampleEffectiveSeqLen() {
	core.Println(EffectiveSeqLen(&Snapshot{SeqLen: 7}))
	// Output: 7
}

// Example_loadStateBlock shows the canonical State block load path: save a
// snapshot as blocks, then read one block back into a Snapshot.
func Example_loadStateBlock() {
	ctx := context.Background()
	store := state.NewInMemoryStore(nil)
	snapshot := &Snapshot{
		Version: SnapshotVersion, Architecture: "gemma4_text",
		Tokens: []int32{1, 2}, TokenOffset: 2,
		NumLayers: 1, NumHeads: 1, SeqLen: 2, HeadDim: 2, NumQueryHeads: 1,
		Layers: []LayerSnapshot{{Heads: []HeadSnapshot{{
			Key: []float32{1, 0, 0, 1}, Value: []float32{0, 1, 1, 0},
		}}}},
	}
	bundle, err := snapshot.SaveStateBlocks(ctx, store, StateBlockOptions{BlockSize: 2, URI: "mlx://ex"})
	if err != nil {
		core.Println("error:", err)
		return
	}
	block, err := LoadStateBlockWithOptions(ctx, store, bundle.Blocks[0], LoadOptions{})
	if err != nil {
		core.Println("error:", err)
		return
	}
	core.Println("tokens:", block.TokenCount)
	// Output: tokens: 2
}

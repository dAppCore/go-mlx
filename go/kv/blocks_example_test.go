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

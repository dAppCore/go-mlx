// SPDX-Licence-Identifier: EUPL-1.2

package kv

import (
	"context"

	core "dappco.re/go"
	state "dappco.re/go/inference/state"
)

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

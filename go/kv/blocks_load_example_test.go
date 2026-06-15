// SPDX-Licence-Identifier: EUPL-1.2

package kv

import (
	"context"

	core "dappco.re/go"
	state "dappco.re/go/inference/state"
)

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

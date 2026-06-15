// SPDX-Licence-Identifier: EUPL-1.2

package kv

import (
	"context"

	core "dappco.re/go"
	state "dappco.re/go/inference/state"
)

// ExampleSnapshot_SaveState writes a KV snapshot to a State cold store as a
// base64-wrapped envelope and reports the chunk it produced.
func ExampleSnapshot_SaveState() {
	store := state.NewInMemoryStore(nil)
	snapshot := testSnapshot()

	ref, err := snapshot.SaveState(context.Background(), store, StateOptions{
		KVEncoding: EncodingQ8,
		URI:        "mlx://session/example",
	})
	if err != nil {
		core.Println("error:", err)
		return
	}
	core.Println("written:", ref.ChunkID > 0)
	// Output: written: true
}

// ExampleLoadFromState resolves and decodes a KV snapshot from a State chunk
// ref written by SaveState.
func ExampleLoadFromState() {
	ctx := context.Background()
	store := state.NewInMemoryStore(nil)
	ref, err := testSnapshot().SaveState(ctx, store, StateOptions{KVEncoding: EncodingQ8})
	if err != nil {
		core.Println("error:", err)
		return
	}

	loaded, err := LoadFromState(ctx, store, ref)
	if err != nil {
		core.Println("error:", err)
		return
	}
	core.Println("architecture:", loaded.Architecture)
	// Output: architecture: gemma4_text
}

// ExampleHashSnapshot computes a stable content-addressed identifier for a
// snapshot; the same snapshot always hashes to the same length-64 hex digest.
func ExampleHashSnapshot() {
	hash, err := HashSnapshot(testSnapshot())
	if err != nil {
		core.Println("error:", err)
		return
	}
	core.Println("hash length:", len(hash))
	// Output: hash length: 64
}

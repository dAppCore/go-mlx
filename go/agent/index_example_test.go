// SPDX-Licence-Identifier: EUPL-1.2

package agent

import (
	"context"
	"fmt"

	state "dappco.re/go/inference/state"
	pkgbundle "dappco.re/go/mlx/bundle"
	"dappco.re/go/mlx/kv"
	"dappco.re/go/mlx/memory"
)

// exampleIndexBundle builds a small two-block durable bundle used by the
// index examples below. Synthetic, no model load: four tokens split into
// two equal blocks, enough to carve named chapter spans over.
func exampleIndexBundle() *kv.StateBlockBundle {
	return &kv.StateBlockBundle{
		Version:      kv.MemvidBlockVersion,
		Kind:         kv.MemvidBlockBundleKind,
		SnapshotHash: "snapshot",
		KVEncoding:   kv.EncodingNative,
		Architecture: "gemma4_text",
		TokenCount:   4,
		BlockSize:    2,
		NumLayers:    1,
		NumHeads:     1,
		SeqLen:       4,
		HeadDim:      2,
		Blocks: []kv.StateBlockRef{
			{Index: 0, TokenStart: 0, TokenCount: 2},
			{Index: 1, TokenStart: 2, TokenCount: 2},
		},
	}
}

// ExampleNewStateIndex builds an index over a durable bundle with two named
// chapter spans and reports the prefix length needed to wake the longest.
func ExampleNewStateIndex() {
	index, err := NewStateIndex(exampleIndexBundle(), StateIndexOptions{
		BundleURI: "mlx://book/bundle",
		Title:     "full book",
		Entries: []StateIndexEntry{
			{URI: "mlx://book/chapter-1", Title: "Chapter 1", TokenStart: 0, TokenCount: 2},
			{URI: "mlx://book/chapter-2", Title: "Chapter 2", TokenStart: 2, TokenCount: 2},
		},
	})
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Println(index.Kind)
	fmt.Println(len(index.Entries))
	fmt.Println(index.RequiredContextLength())
	// Output:
	// go-mlx/kv-snapshot-bundle-index
	// 2
	// 4
}

// ExampleNewStateIndex_defaultEntry shows that an index built with no
// explicit entries gets a single full-bundle entry covering every token.
func ExampleNewStateIndex_defaultEntry() {
	index, err := NewStateIndex(exampleIndexBundle(), StateIndexOptions{BundleURI: "mlx://book/bundle"})
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Println(len(index.Entries))
	fmt.Println(index.Entries[0].URI)
	fmt.Println(index.Entries[0].TokenCount)
	// Output:
	// 1
	// mlx://book/bundle
	// 4
}

// ExampleStateIndex_Entry shows that Entry returns a defensive copy keyed by
// URI: mutating the returned entry leaves the index untouched.
func ExampleStateIndex_Entry() {
	index, _ := NewStateIndex(exampleIndexBundle(), StateIndexOptions{
		BundleURI: "mlx://book/bundle",
		Entries: []StateIndexEntry{
			{URI: "mlx://book/chapter-1", Title: "Chapter 1", TokenStart: 0, TokenCount: 2, Labels: []string{"chapter"}},
		},
	})
	entry, ok := index.Entry("mlx://book/chapter-1")
	fmt.Println(ok)
	entry.Labels[0] = "mutated"
	again, _ := index.Entry("mlx://book/chapter-1")
	fmt.Println(again.Labels[0])
	_, missing := index.Entry("mlx://book/nope")
	fmt.Println(missing)
	// Output:
	// true
	// chapter
	// false
}

// ExampleStateIndexEntry_PrefixTokens shows the prefix length an entry needs
// restored: every token up to and including the entry's own span.
func ExampleStateIndexEntry_PrefixTokens() {
	entry := StateIndexEntry{TokenStart: 2, TokenCount: 2}
	fmt.Println(entry.PrefixTokens())
	// Output:
	// 4
}

// ExampleCheckStateIndexCompatibility verifies an index against the model and
// tokenizer identity it was built for; matching identity returns no error.
func ExampleCheckStateIndexCompatibility() {
	info := memory.ModelInfo{Architecture: "gemma4_text", NumLayers: 1, QuantBits: 4, ContextLength: 8}
	tok := pkgbundle.Tokenizer{Hash: "tok-a", ChatTemplateHash: "chat-a"}
	index, _ := NewStateIndex(exampleIndexBundle(), StateIndexOptions{
		BundleURI: "mlx://book/bundle",
		ModelInfo: info,
		Tokenizer: tok,
		Entries:   []StateIndexEntry{{URI: "mlx://book/chapter-1", TokenStart: 0, TokenCount: 2}},
	})
	err := CheckStateIndexCompatibility(info, tok, index)
	fmt.Println(err)
	// Output:
	// <nil>
}

// ExampleSaveStateIndex_roundTrip stores an index then reloads it by URI from
// the same in-memory State store, recovering the same span.
func ExampleSaveStateIndex_roundTrip() {
	ctx := context.Background()
	store := state.NewInMemoryStore(nil)
	index, _ := NewStateIndex(exampleIndexBundle(), StateIndexOptions{
		BundleURI: "mlx://book/bundle",
		Entries:   []StateIndexEntry{{URI: "mlx://book/chapter-1", Title: "Chapter 1", TokenStart: 0, TokenCount: 2}},
	})
	if _, err := SaveStateIndex(ctx, store, index, "mlx://book/index"); err != nil {
		fmt.Println("save:", err)
		return
	}
	loaded, err := LoadStateIndex(ctx, store, "mlx://book/index")
	if err != nil {
		fmt.Println("load:", err)
		return
	}
	fmt.Println(loaded.Entries[0].URI)
	fmt.Println(loaded.Hash == index.Hash)
	// Output:
	// mlx://book/chapter-1
	// true
}

// ExampleLoadPrefixFromStateIndex resolves a named chapter through a saved
// index and restores only the KV prefix that chapter needs — here the first
// two tokens of a four-token bundle.
func ExampleLoadPrefixFromStateIndex() {
	ctx := context.Background()
	store := state.NewInMemoryStore(nil)
	snapshot := kvSnapshotBlocksTestSnapshot()
	blk, err := snapshot.SaveStateBlocks(ctx, store, kv.StateBlockOptions{BlockSize: 2, KVEncoding: kv.EncodingNative})
	if err != nil {
		fmt.Println("blocks:", err)
		return
	}
	if _, err := kv.SaveStateBlockBundle(ctx, store, blk, "mlx://book/bundle"); err != nil {
		fmt.Println("bundle:", err)
		return
	}
	index, err := NewStateIndex(blk, StateIndexOptions{
		BundleURI: "mlx://book/bundle",
		Entries:   []StateIndexEntry{{URI: "mlx://book/chapter-1", TokenStart: 0, TokenCount: 2}},
	})
	if err != nil {
		fmt.Println("index:", err)
		return
	}
	prefix, entry, err := LoadPrefixFromStateIndex(ctx, store, index, "mlx://book/chapter-1", kv.LoadOptions{RawKVOnly: true})
	if err != nil {
		fmt.Println("prefix:", err)
		return
	}
	fmt.Println(entry.URI)
	fmt.Println(len(prefix.Tokens))
	// Output:
	// mlx://book/chapter-1
	// 2
}

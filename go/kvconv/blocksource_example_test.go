// SPDX-Licence-Identifier: EUPL-1.2

package kvconv_test

import (
	"context"
	"fmt"

	state "dappco.re/go/inference/state"
	"dappco.re/go/inference/kv"
	"dappco.re/go/mlx/kvconv"
)

// ExampleMetalKVSnapshotBlockSource builds a streamed block source over a
// three-block bundle and reports how many blocks cover a partial prefix. A
// prefix of three tokens is covered by the first two blocks.
func ExampleMetalKVSnapshotBlockSource() {
	bundle := &kv.StateBlockBundle{
		Version:    kv.StateBlockVersion,
		Kind:       kv.StateBlockBundleKind,
		TokenCount: 6,
		Blocks: []kv.StateBlockRef{
			{Index: 0, TokenStart: 0, TokenCount: 2},
			{Index: 1, TokenStart: 2, TokenCount: 2},
			{Index: 2, TokenStart: 4, TokenCount: 2},
		},
	}
	source, err := kvconv.MetalKVSnapshotBlockSource(context.Background(), state.NewInMemoryStore(nil), bundle, 3)
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Println(source.BlockCount, source.PrefixTokens, source.TokenCount)
	// Output: 2 3 6
}

// ExampleMetalKVSnapshotBlockSource_fullPrefix shows the prefix default: a
// prefixTokens of zero (or any non-positive value) is treated as "the whole
// bundle", so every block covering the bundle's token count is included.
func ExampleMetalKVSnapshotBlockSource_fullPrefix() {
	bundle := &kv.StateBlockBundle{
		Version:    kv.StateBlockVersion,
		Kind:       kv.StateBlockBundleKind,
		TokenCount: 6,
		Blocks: []kv.StateBlockRef{
			{Index: 0, TokenStart: 0, TokenCount: 2},
			{Index: 1, TokenStart: 2, TokenCount: 2},
			{Index: 2, TokenStart: 4, TokenCount: 2},
		},
	}
	// prefixTokens = 0 -> defaulted to the bundle's full TokenCount (6).
	source, err := kvconv.MetalKVSnapshotBlockSource(context.Background(), state.NewInMemoryStore(nil), bundle, 0)
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Println(source.BlockCount, source.PrefixTokens, source.TokenCount)
	// Output: 3 6 6
}

// ExampleMetalKVSnapshotBlockSource_nilContext shows that a nil context is
// accepted at construction — it is replaced with context.Background() rather
// than panicking, so a caller without a context to hand can still build a
// source.
func ExampleMetalKVSnapshotBlockSource_nilContext() {
	bundle := &kv.StateBlockBundle{
		Version:    kv.StateBlockVersion,
		Kind:       kv.StateBlockBundleKind,
		TokenCount: 2,
		Blocks:     []kv.StateBlockRef{{Index: 0, TokenStart: 0, TokenCount: 2}},
	}
	//nolint:staticcheck // SA1012: passing nil context is the behaviour under test
	source, err := kvconv.MetalKVSnapshotBlockSource(nil, state.NewInMemoryStore(nil), bundle, 2)
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Println(source.BlockCount)
	// Output: 1
}

// SPDX-Licence-Identifier: EUPL-1.2

package kvconv_test

import (
	"context"
	"fmt"

	state "dappco.re/go/inference/state"
	"dappco.re/go/mlx/kv"
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

// SPDX-Licence-Identifier: EUPL-1.2

package kvconv

import (
	"context"
	"testing"

	state "dappco.re/go/inference/state"
	"dappco.re/go/mlx/kv"
)

func TestMetalKVSnapshotBlockSourcePartialPrefix_Good(t *testing.T) {
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

	source, err := MetalKVSnapshotBlockSource(context.Background(), state.NewInMemoryStore(nil), bundle, 3)
	if err != nil {
		t.Fatalf("MetalKVSnapshotBlockSource() error = %v", err)
	}
	if source.BlockCount != 2 || source.PrefixTokens != 3 || source.TokenCount != 6 {
		t.Fatalf("source = %+v, want two covering blocks for three-token prefix", source)
	}
}

func TestMetalKVSnapshotBlockSourceRejectsNonContiguousBundle_Bad(t *testing.T) {
	bundle := &kv.StateBlockBundle{
		Version:    kv.StateBlockVersion,
		Kind:       kv.StateBlockBundleKind,
		TokenCount: 4,
		Blocks: []kv.StateBlockRef{
			{Index: 0, TokenStart: 0, TokenCount: 2},
			{Index: 1, TokenStart: 3, TokenCount: 1},
		},
	}

	if _, err := MetalKVSnapshotBlockSource(context.Background(), state.NewInMemoryStore(nil), bundle, 4); err != errStateKVBlockMetaMismatch {
		t.Fatalf("MetalKVSnapshotBlockSource() error = %v, want metadata mismatch", err)
	}
}

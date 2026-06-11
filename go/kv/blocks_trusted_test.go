// SPDX-Licence-Identifier: EUPL-1.2

package kv

import (
	"context"
	"testing"

	state "dappco.re/go/inference/state"
)

// The trusted-prefix sleep lane: parent blocks below the boundary graft by
// reference with no capture and no hash. The stream asserts the capture side
// was never asked for the grafted range (BlockStartToken semantics).
func TestKVSnapshotStateBlocks_Good_TrustedPrefixGraftsWithoutCapture(t *testing.T) {
	ctx := context.Background()
	store := state.NewInMemoryStore(nil)
	parent := kvSnapshotBlocksTestSnapshot()
	parentBundle, err := parent.SaveStateBlocks(ctx, store, StateBlockOptions{
		BlockSize:  2,
		KVEncoding: EncodingNative,
		URI:        "mlx://trusted/parent",
	})
	if err != nil {
		t.Fatalf("SaveStateBlocks(parent) error = %v", err)
	}

	opts := StateBlockOptions{
		BlockSize:          2,
		KVEncoding:         EncodingNative,
		URI:                "mlx://trusted/child",
		ReusePrefix:        parentBundle,
		ReusePrefixTokens:  2,
		ReusePrefixTrusted: true,
	}
	if boundary := TrustedReuseBoundary(opts, 2); boundary != 2 {
		t.Fatalf("TrustedReuseBoundary = %d, want 2", boundary)
	}

	child := kvSnapshotBlocksTestSnapshot()
	captured := []int{}
	childBundle, err := SaveStateBlocksFromStream(ctx, store, opts, func(yield func(Block) (bool, error)) error {
		// Mirror the capture side: BlockStartToken skips blocks ending at or
		// before the trusted boundary.
		return child.walkBlocks(2, false, func(block Block) (bool, error) {
			if block.TokenStart+block.TokenCount <= 2 {
				return true, nil
			}
			captured = append(captured, block.TokenStart)
			return yield(block)
		})
	})
	if err != nil {
		t.Fatalf("SaveStateBlocksFromStream(trusted) error = %v", err)
	}
	if len(captured) != 1 || captured[0] != 2 {
		t.Fatalf("captured starts = %v, want only the post-boundary block [2]", captured)
	}
	if childBundle.ReusedBlocks != 1 || len(childBundle.Blocks) != 2 {
		t.Fatalf("bundle reused=%d blocks=%d, want 1 grafted + 1 streamed", childBundle.ReusedBlocks, len(childBundle.Blocks))
	}
	if childBundle.Blocks[0].State.ChunkID != parentBundle.Blocks[0].State.ChunkID {
		t.Fatalf("grafted ref = %+v, want parent ref %+v", childBundle.Blocks[0], parentBundle.Blocks[0])
	}
	if childBundle.Blocks[0].KVHash != parentBundle.Blocks[0].KVHash {
		t.Fatalf("grafted hash = %q, want parent hash %q carried", childBundle.Blocks[0].KVHash, parentBundle.Blocks[0].KVHash)
	}
	loaded, err := LoadFromStateBlocksWithOptions(ctx, store, childBundle, LoadOptions{RawKVOnly: true})
	if err != nil {
		t.Fatalf("LoadFromStateBlocksWithOptions(trusted bundle) error = %v", err)
	}
	if len(loaded.Tokens) != 4 {
		t.Fatalf("loaded tokens = %v, want full 4-token prefix", loaded.Tokens)
	}
}

func TestKVSnapshotStateBlocks_Good_TrustedBoundaryMatrix(t *testing.T) {
	parent := &StateBlockBundle{
		BlockSize:  2,
		TokenCount: 5,
		Blocks: []StateBlockRef{
			{Index: 0, TokenStart: 0, TokenCount: 2},
			{Index: 1, TokenStart: 2, TokenCount: 2},
			{Index: 2, TokenStart: 4, TokenCount: 1}, // partial tail — never grafted
		},
	}
	cases := []struct {
		name string
		opts StateBlockOptions
		size int
		want int
	}{
		{"untrusted", StateBlockOptions{ReusePrefix: parent}, 2, 0},
		{"trusted full", StateBlockOptions{ReusePrefix: parent, ReusePrefixTrusted: true}, 2, 4},
		{"trusted capped", StateBlockOptions{ReusePrefix: parent, ReusePrefixTrusted: true, ReusePrefixTokens: 3}, 2, 2},
		{"block size mismatch", StateBlockOptions{ReusePrefix: parent, ReusePrefixTrusted: true}, 4, 0},
		{"no parent", StateBlockOptions{ReusePrefixTrusted: true}, 2, 0},
	}
	for _, tc := range cases {
		if got := TrustedReuseBoundary(tc.opts, tc.size); got != tc.want {
			t.Errorf("%s: boundary = %d, want %d", tc.name, got, tc.want)
		}
	}
}

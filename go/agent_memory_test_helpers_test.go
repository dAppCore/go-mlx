// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	memvid "dappco.re/go/inference/state"
	"dappco.re/go/mlx/kv"
)

// kvSnapshotIndexTestBundle returns a small KV memvid block bundle for
// mlx-root tests (session_agent_darwin_test.go) that need fixture data.
// Duplicated from agent/index_test.go because Go test packages cannot
// import each other's internal _test.go symbols.
func kvSnapshotIndexTestBundle() *kv.MemvidBlockBundle {
	return &kv.MemvidBlockBundle{
		Version:      kv.MemvidBlockVersion,
		Kind:         kv.MemvidBlockBundleKind,
		SnapshotHash: "snapshot",
		KVEncoding:   kv.EncodingNative,
		Architecture: "gemma4_text",
		TokenCount:   4,
		TokenOffset:  4,
		BlockSize:    2,
		NumLayers:    1,
		NumHeads:     1,
		SeqLen:       4,
		HeadDim:      2,
		Blocks: []kv.MemvidBlockRef{{
			Index:      0,
			TokenStart: 0,
			TokenCount: 2,
			Memvid:     memvid.ChunkRef{ChunkID: 1},
		}},
	}
}

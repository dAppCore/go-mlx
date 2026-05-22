// SPDX-Licence-Identifier: EUPL-1.2

package agent

import (
	"context"
	"testing"

	core "dappco.re/go"
	memvid "dappco.re/go/inference/state"
	pkgbundle "dappco.re/go/mlx/bundle"
	"dappco.re/go/mlx/kv"
	"dappco.re/go/mlx/memory"
)

func TestKVSnapshotStateIndex_Good_PartialPrefixFromFullBundle(t *testing.T) {
	ctx := context.Background()
	store := memvid.NewInMemoryStore(nil)
	snapshot := kvSnapshotBlocksTestSnapshot()
	blk, err := snapshot.SaveStateBlocks(ctx, store, kv.StateBlockOptions{
		BlockSize:  2,
		KVEncoding: kv.EncodingNative,
	})
	if err != nil {
		t.Fatalf("SaveStateBlocks() error = %v", err)
	}
	if _, err := kv.SaveStateBlockBundle(ctx, store, blk, "mlx://book/full/bundle"); err != nil {
		t.Fatalf("kv.SaveStateBlockBundle() error = %v", err)
	}
	index, err := NewStateIndex(blk, StateIndexOptions{
		BundleURI: "mlx://book/full/bundle",
		Title:     "full book",
		Model:     "demo",
		ModelInfo: memory.ModelInfo{
			Architecture:  "gemma4_text",
			NumLayers:     1,
			QuantBits:     4,
			ContextLength: 8,
		},
		Tokenizer: pkgbundle.Tokenizer{Hash: "tok-a", ChatTemplateHash: "chat-a"},
		Entries: []StateIndexEntry{
			{
				URI:        "mlx://book/chapter-1",
				Title:      "Chapter 1",
				TokenStart: 0,
				TokenCount: 2,
				ByteStart:  0,
				ByteCount:  128,
				Labels:     []string{"chapter"},
				Meta:       map[string]string{"ordinal": "1"},
			},
			{
				URI:        "mlx://book/chapter-2",
				Title:      "Chapter 2",
				TokenStart: 2,
				TokenCount: 2,
				ByteStart:  128,
				ByteCount:  128,
				Labels:     []string{"chapter"},
				Meta:       map[string]string{"ordinal": "2"},
			},
		},
	})
	if err != nil {
		t.Fatalf("NewStateIndex() error = %v", err)
	}
	if index.Hash == "" || index.RequiredContextLength() != 4 {
		t.Fatalf("index hash/required = %q/%d, want hash and full required context", index.Hash, index.RequiredContextLength())
	}
	if err := CheckStateIndexCompatibility(memory.ModelInfo{Architecture: "gemma4_text", NumLayers: 1, QuantBits: 4, ContextLength: 8}, pkgbundle.Tokenizer{Hash: "tok-a", ChatTemplateHash: "chat-a"}, index); err != nil {
		t.Fatalf("CheckStateIndexCompatibility() error = %v", err)
	}
	if _, err := SaveStateIndex(ctx, store, index, "mlx://book/index"); err != nil {
		t.Fatalf("SaveStateIndex() error = %v", err)
	}
	loadedIndex, err := LoadStateIndex(ctx, store, "mlx://book/index")
	if err != nil {
		t.Fatalf("LoadStateIndex() error = %v", err)
	}
	loadedIndex.Entries[0].Labels[0] = "mutated"
	entry, ok := index.Entry("mlx://book/chapter-1")
	if !ok {
		t.Fatal("Entry(chapter-1) ok = false")
	}
	if entry.Labels[0] != "chapter" || entry.ByteStart != 0 || entry.ByteCount != 128 {
		t.Fatalf("entry clone = %+v, want original labels and byte span", entry)
	}

	recording := &indexRecordingMemvidStore{store: store}
	prefix, loadedEntry, err := LoadPrefixFromStateIndex(ctx, recording, index, "mlx://book/chapter-1", kv.LoadOptions{RawKVOnly: true})
	if err != nil {
		t.Fatalf("LoadPrefixFromStateIndex() error = %v", err)
	}
	if loadedEntry.URI != "mlx://book/chapter-1" || loadedEntry.PrefixTokens() != 2 {
		t.Fatalf("loaded entry = %+v, want chapter-1 two-token prefix", loadedEntry)
	}
	if len(prefix.Tokens) != 2 || prefix.Tokens[0] != 1 || prefix.Tokens[1] != 2 {
		t.Fatalf("prefix tokens = %v, want first two tokens", prefix.Tokens)
	}
	if len(prefix.Logits) != 0 {
		t.Fatalf("prefix logits = %v, want terminal state cleared for partial prefix", prefix.Logits)
	}
	if len(recording.resolvedURIs) != 1 || recording.resolvedURIs[0] != "mlx://book/full/bundle" {
		t.Fatalf("resolved URIs = %v, want bundle manifest URI", recording.resolvedURIs)
	}
	if len(recording.resolved) != 1 {
		t.Fatalf("resolved chunks = %v, want one covering block", recording.resolved)
	}
}

func TestKVSnapshotMemvidBundleIndex_Good_DefaultFullEntry(t *testing.T) {
	blk := kvSnapshotIndexTestBundle()

	index, err := NewMemvidIndex(blk, MemvidIndexOptions{BundleURI: "mlx://bundle"})

	if err != nil {
		t.Fatalf("NewMemvidIndex(default) error = %v", err)
	}
	if len(index.Entries) != 1 || index.Entries[0].TokenCount != blk.TokenCount || index.Entries[0].BundleURI != "mlx://bundle" {
		t.Fatalf("default entries = %+v, want full bundle entry", index.Entries)
	}
}

func TestKVSnapshotMemvidBundleIndex_Good_DerivesEntryByteSpan(t *testing.T) {
	blk := kvSnapshotIndexTestBundle()
	blk.Blocks = []kv.MemvidBlockRef{
		{
			Index:            0,
			TokenStart:       0,
			TokenCount:       2,
			PayloadByteCount: 100,
			Memvid:           memvid.ChunkRef{ChunkID: 1, FrameOffset: 64, HasFrameOffset: true},
		},
		{
			Index:            1,
			TokenStart:       2,
			TokenCount:       2,
			PayloadByteCount: 300,
			Memvid:           memvid.ChunkRef{ChunkID: 2, FrameOffset: 256, HasFrameOffset: true},
		},
	}

	index, err := NewMemvidIndex(blk, MemvidIndexOptions{
		BundleURI: "mlx://book/full/bundle",
		Entries: []MemvidIndexEntry{
			{URI: "mlx://book/chapter-1", TokenStart: 0, TokenCount: 2},
			{URI: "mlx://book/chapter-2", TokenStart: 2, TokenCount: 2},
			{URI: "mlx://book/cross-block", TokenStart: 1, TokenCount: 2},
		},
	})

	if err != nil {
		t.Fatalf("NewMemvidIndex(byte span) error = %v", err)
	}
	chapter1, _ := index.Entry("mlx://book/chapter-1")
	if chapter1.ByteStart != 64 || chapter1.ByteCount != 100 {
		t.Fatalf("chapter-1 byte span = %d/%d, want 64/100", chapter1.ByteStart, chapter1.ByteCount)
	}
	chapter2, _ := index.Entry("mlx://book/chapter-2")
	if chapter2.ByteStart != 256 || chapter2.ByteCount != 300 {
		t.Fatalf("chapter-2 byte span = %d/%d, want 256/300", chapter2.ByteStart, chapter2.ByteCount)
	}
	cross, _ := index.Entry("mlx://book/cross-block")
	if cross.ByteStart != 64 || cross.ByteCount != 400 {
		t.Fatalf("cross-block byte span = %d/%d, want first frame offset and summed payload bytes 64/400", cross.ByteStart, cross.ByteCount)
	}
}

func TestKVSnapshotMemvidBundleIndex_Bad_ValidationAndCompatibility(t *testing.T) {
	blk := kvSnapshotIndexTestBundle()
	index, err := NewMemvidIndex(blk, MemvidIndexOptions{
		BundleURI: "mlx://bundle",
		ModelInfo: memory.ModelInfo{Architecture: "gemma4_text", NumLayers: 2, QuantBits: 4, ContextLength: 4},
		Tokenizer: pkgbundle.Tokenizer{Hash: "tok-a"},
		Entries: []MemvidIndexEntry{{
			URI:        "mlx://chapter",
			TokenStart: 0,
			TokenCount: 1,
		}},
	})
	if err != nil {
		t.Fatalf("NewMemvidIndex() error = %v", err)
	}
	for _, tc := range []struct {
		name  string
		index MemvidIndex
	}{
		{name: "bad kind", index: func() MemvidIndex {
			bad := *index
			bad.Kind = "bad"
			return bad
		}()},
		{name: "bad hash", index: func() MemvidIndex {
			bad := *index
			bad.Hash = "bad"
			return bad
		}()},
		{name: "duplicate uri", index: func() MemvidIndex {
			bad := *index
			bad.Entries = append(cloneIndexEntries(index.Entries), index.Entries[0])
			bad.Hash = indexHash(&bad)
			return bad
		}()},
		{name: "entry exceeds bundle", index: func() MemvidIndex {
			bad := *index
			bad.Entries = cloneIndexEntries(index.Entries)
			bad.Entries[0].TokenCount = 99
			bad.Entries[0].Hash = indexEntryHash(&bad.Entries[0])
			bad.Hash = indexHash(&bad)
			return bad
		}()},
		{name: "entry hash", index: func() MemvidIndex {
			bad := *index
			bad.Entries = cloneIndexEntries(index.Entries)
			bad.Entries[0].Hash = "bad"
			bad.Hash = ""
			return bad
		}()},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if err := tc.index.Validate(); err == nil {
				t.Fatal("Validate() error = nil")
			}
		})
	}

	if err := CheckMemvidIndexCompatibility(memory.ModelInfo{Architecture: "qwen3", NumLayers: 2, QuantBits: 4, ContextLength: 4}, pkgbundle.Tokenizer{Hash: "tok-a"}, index); err == nil {
		t.Fatal("expected architecture mismatch")
	}
	if err := CheckMemvidIndexCompatibility(memory.ModelInfo{Architecture: "gemma4_text", NumLayers: 1, QuantBits: 4, ContextLength: 4}, pkgbundle.Tokenizer{Hash: "tok-a"}, index); err == nil {
		t.Fatal("expected layer mismatch")
	}
	if err := CheckMemvidIndexCompatibility(memory.ModelInfo{Architecture: "gemma4_text", NumLayers: 2, QuantBits: 8, ContextLength: 4}, pkgbundle.Tokenizer{Hash: "tok-a"}, index); err == nil {
		t.Fatal("expected quantization mismatch")
	}
	hashIndex, err := NewMemvidIndex(blk, MemvidIndexOptions{
		BundleURI: "mlx://bundle",
		ModelInfo: memory.ModelInfo{Architecture: "gemma4_text", NumLayers: 2, QuantBits: 4, ContextLength: 4},
		Entries: []MemvidIndexEntry{{
			URI:        "mlx://chapter",
			TokenStart: 0,
			TokenCount: 1,
		}},
	})
	if err != nil {
		t.Fatalf("NewMemvidIndex(hash) error = %v", err)
	}
	hashIndex.Model.Hash = "different-model-hash"
	hashIndex.Hash = indexHash(hashIndex)
	if err := CheckMemvidIndexCompatibility(memory.ModelInfo{Architecture: "gemma4_text", NumLayers: 2, QuantBits: 4, ContextLength: 4}, pkgbundle.Tokenizer{}, hashIndex); err == nil {
		t.Fatal("expected model hash mismatch")
	}
	if err := CheckMemvidIndexCompatibility(memory.ModelInfo{Architecture: "gemma4_text", NumLayers: 2, QuantBits: 4, ContextLength: 0}, pkgbundle.Tokenizer{Hash: "tok-b"}, index); err == nil {
		t.Fatal("expected tokenizer mismatch")
	}
	if err := CheckMemvidIndexCompatibility(memory.ModelInfo{Architecture: "gemma4_text", NumLayers: 2, QuantBits: 4, ContextLength: 0}, pkgbundle.Tokenizer{Hash: "tok-a"}, index); err != nil {
		t.Fatalf("zero context should skip context compatibility, got %v", err)
	}
}

func TestKVSnapshotMemvidBundleIndex_Bad_LoadAndStoreErrors(t *testing.T) {
	ctx := context.Background()
	store := memvid.NewInMemoryStore(nil)
	blk := kvSnapshotIndexTestBundle()
	index, err := NewMemvidIndex(blk, MemvidIndexOptions{
		BundleURI: "mlx://bundle",
		Entries: []MemvidIndexEntry{{
			URI:        "mlx://chapter",
			TokenStart: 0,
			TokenCount: 1,
		}},
	})
	if err != nil {
		t.Fatalf("NewMemvidIndex() error = %v", err)
	}
	if _, err := SaveMemvidIndex(ctx, nil, index, "mlx://index"); err == nil {
		t.Fatal("SaveMemvidIndex(nil store) error = nil")
	}
	if _, err := SaveMemvidIndex(ctx, store, index, ""); err == nil {
		t.Fatal("SaveMemvidIndex(empty URI) error = nil")
	}
	if _, err := LoadMemvidIndex(ctx, nil, "mlx://index"); err == nil {
		t.Fatal("LoadMemvidIndex(nil store) error = nil")
	}
	if _, err := LoadMemvidIndex(ctx, store, ""); err == nil {
		t.Fatal("LoadMemvidIndex(empty URI) error = nil")
	}
	if _, _, err := LoadPrefixFromMemvidIndex(ctx, nil, index, "mlx://chapter", kv.LoadOptions{}); err == nil {
		t.Fatal("LoadPrefixFromMemvidIndex(nil store) error = nil")
	}
	if _, _, err := LoadPrefixFromMemvidIndex(ctx, store, index, "mlx://missing", kv.LoadOptions{}); err == nil {
		t.Fatal("LoadPrefixFromMemvidIndex(missing entry) error = nil")
	}
	if _, _, err := LoadPrefixFromMemvidIndex(ctx, store, index, "mlx://chapter", kv.LoadOptions{}); err == nil {
		t.Fatal("LoadPrefixFromMemvidIndex(missing bundle) error = nil")
	}
	corrupt := core.JSONMarshalString(map[string]any{"version": 1, "kind": MemvidIndexKind})
	if _, err := store.Put(ctx, corrupt, memvid.PutOptions{URI: "mlx://bad-index"}); err != nil {
		t.Fatalf("write corrupt index: %v", err)
	}
	if _, err := LoadMemvidIndex(ctx, store, "mlx://bad-index"); err == nil {
		t.Fatal("LoadMemvidIndex(corrupt) error = nil")
	}
}

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

type indexRecordingMemvidStore struct {
	store        memvid.Store
	resolved     []int
	resolvedURIs []string
}

func (s *indexRecordingMemvidStore) Get(ctx context.Context, chunkID int) (string, error) {
	s.resolved = append(s.resolved, chunkID)
	return s.store.Get(ctx, chunkID)
}

func (s *indexRecordingMemvidStore) Resolve(ctx context.Context, chunkID int) (memvid.Chunk, error) {
	s.resolved = append(s.resolved, chunkID)
	return memvid.Resolve(ctx, s.store, chunkID)
}

func (s *indexRecordingMemvidStore) ResolveBytes(ctx context.Context, chunkID int) (memvid.Chunk, error) {
	s.resolved = append(s.resolved, chunkID)
	return memvid.ResolveBytes(ctx, s.store, chunkID)
}

func (s *indexRecordingMemvidStore) ResolveURI(ctx context.Context, uri string) (memvid.Chunk, error) {
	s.resolvedURIs = append(s.resolvedURIs, uri)
	return memvid.ResolveURI(ctx, s.store, uri)
}

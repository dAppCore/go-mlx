// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"
	"testing"

	core "dappco.re/go"
	memvid "dappco.re/go/inference/state"
)

func TestKVSnapshotMemvidBundleIndex_Good_PartialPrefixFromFullBundle(t *testing.T) {
	ctx := context.Background()
	store := memvid.NewInMemoryStore(nil)
	snapshot := kvSnapshotBlocksTestSnapshot()
	bundle, err := snapshot.SaveMemvidBlocks(ctx, store, KVSnapshotMemvidBlockOptions{
		BlockSize:  2,
		KVEncoding: KVSnapshotEncodingNative,
	})
	if err != nil {
		t.Fatalf("SaveMemvidBlocks() error = %v", err)
	}
	if _, err := SaveKVSnapshotMemvidBlockBundle(ctx, store, bundle, "mlx://book/full/bundle"); err != nil {
		t.Fatalf("SaveKVSnapshotMemvidBlockBundle() error = %v", err)
	}
	index, err := NewKVSnapshotMemvidBundleIndex(bundle, KVSnapshotMemvidBundleIndexOptions{
		BundleURI: "mlx://book/full/bundle",
		Title:     "full book",
		Model:     "demo",
		ModelInfo: ModelInfo{
			Architecture:  "gemma4_text",
			NumLayers:     1,
			QuantBits:     4,
			ContextLength: 8,
		},
		Tokenizer: StateBundleTokenizer{Hash: "tok-a", ChatTemplateHash: "chat-a"},
		Entries: []KVSnapshotMemvidBundleIndexEntry{
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
		t.Fatalf("NewKVSnapshotMemvidBundleIndex() error = %v", err)
	}
	if index.Hash == "" || index.RequiredContextLength() != 4 {
		t.Fatalf("index hash/required = %q/%d, want hash and full required context", index.Hash, index.RequiredContextLength())
	}
	if err := CheckKVSnapshotMemvidBundleIndexCompatibility(ModelInfo{Architecture: "gemma4_text", NumLayers: 1, QuantBits: 4, ContextLength: 8}, StateBundleTokenizer{Hash: "tok-a", ChatTemplateHash: "chat-a"}, index); err != nil {
		t.Fatalf("CheckKVSnapshotMemvidBundleIndexCompatibility() error = %v", err)
	}
	if _, err := SaveKVSnapshotMemvidBundleIndex(ctx, store, index, "mlx://book/index"); err != nil {
		t.Fatalf("SaveKVSnapshotMemvidBundleIndex() error = %v", err)
	}
	loadedIndex, err := LoadKVSnapshotMemvidBundleIndex(ctx, store, "mlx://book/index")
	if err != nil {
		t.Fatalf("LoadKVSnapshotMemvidBundleIndex() error = %v", err)
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
	prefix, loadedEntry, err := LoadKVSnapshotPrefixFromMemvidBundleIndex(ctx, recording, index, "mlx://book/chapter-1", KVSnapshotLoadOptions{RawKVOnly: true})
	if err != nil {
		t.Fatalf("LoadKVSnapshotPrefixFromMemvidBundleIndex() error = %v", err)
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
	bundle := kvSnapshotIndexTestBundle()

	index, err := NewKVSnapshotMemvidBundleIndex(bundle, KVSnapshotMemvidBundleIndexOptions{BundleURI: "mlx://bundle"})

	if err != nil {
		t.Fatalf("NewKVSnapshotMemvidBundleIndex(default) error = %v", err)
	}
	if len(index.Entries) != 1 || index.Entries[0].TokenCount != bundle.TokenCount || index.Entries[0].BundleURI != "mlx://bundle" {
		t.Fatalf("default entries = %+v, want full bundle entry", index.Entries)
	}
}

func TestKVSnapshotMemvidBundleIndex_Good_DerivesEntryByteSpan(t *testing.T) {
	bundle := kvSnapshotIndexTestBundle()
	bundle.Blocks = []KVSnapshotMemvidBlockRef{
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

	index, err := NewKVSnapshotMemvidBundleIndex(bundle, KVSnapshotMemvidBundleIndexOptions{
		BundleURI: "mlx://book/full/bundle",
		Entries: []KVSnapshotMemvidBundleIndexEntry{
			{URI: "mlx://book/chapter-1", TokenStart: 0, TokenCount: 2},
			{URI: "mlx://book/chapter-2", TokenStart: 2, TokenCount: 2},
			{URI: "mlx://book/cross-block", TokenStart: 1, TokenCount: 2},
		},
	})

	if err != nil {
		t.Fatalf("NewKVSnapshotMemvidBundleIndex(byte span) error = %v", err)
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
	bundle := kvSnapshotIndexTestBundle()
	index, err := NewKVSnapshotMemvidBundleIndex(bundle, KVSnapshotMemvidBundleIndexOptions{
		BundleURI: "mlx://bundle",
		ModelInfo: ModelInfo{Architecture: "gemma4_text", NumLayers: 2, QuantBits: 4, ContextLength: 4},
		Tokenizer: StateBundleTokenizer{Hash: "tok-a"},
		Entries: []KVSnapshotMemvidBundleIndexEntry{{
			URI:        "mlx://chapter",
			TokenStart: 0,
			TokenCount: 1,
		}},
	})
	if err != nil {
		t.Fatalf("NewKVSnapshotMemvidBundleIndex() error = %v", err)
	}
	for _, tc := range []struct {
		name  string
		index KVSnapshotMemvidBundleIndex
	}{
		{name: "bad kind", index: func() KVSnapshotMemvidBundleIndex {
			bad := *index
			bad.Kind = "bad"
			return bad
		}()},
		{name: "bad hash", index: func() KVSnapshotMemvidBundleIndex {
			bad := *index
			bad.Hash = "bad"
			return bad
		}()},
		{name: "duplicate uri", index: func() KVSnapshotMemvidBundleIndex {
			bad := *index
			bad.Entries = append(cloneKVSnapshotMemvidBundleIndexEntries(index.Entries), index.Entries[0])
			bad.Hash = kvSnapshotMemvidBundleIndexHash(&bad)
			return bad
		}()},
		{name: "entry exceeds bundle", index: func() KVSnapshotMemvidBundleIndex {
			bad := *index
			bad.Entries = cloneKVSnapshotMemvidBundleIndexEntries(index.Entries)
			bad.Entries[0].TokenCount = 99
			bad.Entries[0].Hash = kvSnapshotMemvidBundleIndexEntryHash(bad.Entries[0])
			bad.Hash = kvSnapshotMemvidBundleIndexHash(&bad)
			return bad
		}()},
		{name: "entry hash", index: func() KVSnapshotMemvidBundleIndex {
			bad := *index
			bad.Entries = cloneKVSnapshotMemvidBundleIndexEntries(index.Entries)
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

	if err := CheckKVSnapshotMemvidBundleIndexCompatibility(ModelInfo{Architecture: "qwen3", NumLayers: 2, QuantBits: 4, ContextLength: 4}, StateBundleTokenizer{Hash: "tok-a"}, index); err == nil {
		t.Fatal("expected architecture mismatch")
	}
	if err := CheckKVSnapshotMemvidBundleIndexCompatibility(ModelInfo{Architecture: "gemma4_text", NumLayers: 1, QuantBits: 4, ContextLength: 4}, StateBundleTokenizer{Hash: "tok-a"}, index); err == nil {
		t.Fatal("expected layer mismatch")
	}
	if err := CheckKVSnapshotMemvidBundleIndexCompatibility(ModelInfo{Architecture: "gemma4_text", NumLayers: 2, QuantBits: 8, ContextLength: 4}, StateBundleTokenizer{Hash: "tok-a"}, index); err == nil {
		t.Fatal("expected quantization mismatch")
	}
	hashIndex, err := NewKVSnapshotMemvidBundleIndex(bundle, KVSnapshotMemvidBundleIndexOptions{
		BundleURI: "mlx://bundle",
		ModelInfo: ModelInfo{Architecture: "gemma4_text", NumLayers: 2, QuantBits: 4, ContextLength: 4},
		Entries: []KVSnapshotMemvidBundleIndexEntry{{
			URI:        "mlx://chapter",
			TokenStart: 0,
			TokenCount: 1,
		}},
	})
	if err != nil {
		t.Fatalf("NewKVSnapshotMemvidBundleIndex(hash) error = %v", err)
	}
	hashIndex.Model.Hash = "different-model-hash"
	hashIndex.Hash = kvSnapshotMemvidBundleIndexHash(hashIndex)
	if err := CheckKVSnapshotMemvidBundleIndexCompatibility(ModelInfo{Architecture: "gemma4_text", NumLayers: 2, QuantBits: 4, ContextLength: 4}, StateBundleTokenizer{}, hashIndex); err == nil {
		t.Fatal("expected model hash mismatch")
	}
	if err := CheckKVSnapshotMemvidBundleIndexCompatibility(ModelInfo{Architecture: "gemma4_text", NumLayers: 2, QuantBits: 4, ContextLength: 0}, StateBundleTokenizer{Hash: "tok-b"}, index); err == nil {
		t.Fatal("expected tokenizer mismatch")
	}
	if err := CheckKVSnapshotMemvidBundleIndexCompatibility(ModelInfo{Architecture: "gemma4_text", NumLayers: 2, QuantBits: 4, ContextLength: 0}, StateBundleTokenizer{Hash: "tok-a"}, index); err != nil {
		t.Fatalf("zero context should skip context compatibility, got %v", err)
	}
}

func TestKVSnapshotMemvidBundleIndex_Bad_LoadAndStoreErrors(t *testing.T) {
	ctx := context.Background()
	store := memvid.NewInMemoryStore(nil)
	bundle := kvSnapshotIndexTestBundle()
	index, err := NewKVSnapshotMemvidBundleIndex(bundle, KVSnapshotMemvidBundleIndexOptions{
		BundleURI: "mlx://bundle",
		Entries: []KVSnapshotMemvidBundleIndexEntry{{
			URI:        "mlx://chapter",
			TokenStart: 0,
			TokenCount: 1,
		}},
	})
	if err != nil {
		t.Fatalf("NewKVSnapshotMemvidBundleIndex() error = %v", err)
	}
	if _, err := SaveKVSnapshotMemvidBundleIndex(ctx, nil, index, "mlx://index"); err == nil {
		t.Fatal("SaveKVSnapshotMemvidBundleIndex(nil store) error = nil")
	}
	if _, err := SaveKVSnapshotMemvidBundleIndex(ctx, store, index, ""); err == nil {
		t.Fatal("SaveKVSnapshotMemvidBundleIndex(empty URI) error = nil")
	}
	if _, err := LoadKVSnapshotMemvidBundleIndex(ctx, nil, "mlx://index"); err == nil {
		t.Fatal("LoadKVSnapshotMemvidBundleIndex(nil store) error = nil")
	}
	if _, err := LoadKVSnapshotMemvidBundleIndex(ctx, store, ""); err == nil {
		t.Fatal("LoadKVSnapshotMemvidBundleIndex(empty URI) error = nil")
	}
	if _, _, err := LoadKVSnapshotPrefixFromMemvidBundleIndex(ctx, nil, index, "mlx://chapter", KVSnapshotLoadOptions{}); err == nil {
		t.Fatal("LoadKVSnapshotPrefixFromMemvidBundleIndex(nil store) error = nil")
	}
	if _, _, err := LoadKVSnapshotPrefixFromMemvidBundleIndex(ctx, store, index, "mlx://missing", KVSnapshotLoadOptions{}); err == nil {
		t.Fatal("LoadKVSnapshotPrefixFromMemvidBundleIndex(missing entry) error = nil")
	}
	if _, _, err := LoadKVSnapshotPrefixFromMemvidBundleIndex(ctx, store, index, "mlx://chapter", KVSnapshotLoadOptions{}); err == nil {
		t.Fatal("LoadKVSnapshotPrefixFromMemvidBundleIndex(missing bundle) error = nil")
	}
	corrupt := core.JSONMarshalString(map[string]any{"version": 1, "kind": KVSnapshotMemvidBundleIndexKind})
	if _, err := store.Put(ctx, corrupt, memvid.PutOptions{URI: "mlx://bad-index"}); err != nil {
		t.Fatalf("write corrupt index: %v", err)
	}
	if _, err := LoadKVSnapshotMemvidBundleIndex(ctx, store, "mlx://bad-index"); err == nil {
		t.Fatal("LoadKVSnapshotMemvidBundleIndex(corrupt) error = nil")
	}
}

func kvSnapshotIndexTestBundle() *KVSnapshotMemvidBlockBundle {
	return &KVSnapshotMemvidBlockBundle{
		Version:      KVSnapshotMemvidBlockVersion,
		Kind:         KVSnapshotMemvidBlockBundleKind,
		SnapshotHash: "snapshot",
		KVEncoding:   KVSnapshotEncodingNative,
		Architecture: "gemma4_text",
		TokenCount:   4,
		TokenOffset:  4,
		BlockSize:    2,
		NumLayers:    1,
		NumHeads:     1,
		SeqLen:       4,
		HeadDim:      2,
		Blocks: []KVSnapshotMemvidBlockRef{{
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

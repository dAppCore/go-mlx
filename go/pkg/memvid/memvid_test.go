// SPDX-Licence-Identifier: EUPL-1.2

package memvid

import (
	"context"
	"testing"

	core "dappco.re/go"
)

func TestMemvid_InMemoryStore_Good(t *testing.T) {
	store := NewInMemoryStore(map[int]string{7: "chunk seven"})

	text, err := store.Get(context.Background(), 7)
	if err != nil {
		t.Fatalf("Get() error = %v", err)
	}
	if text != "chunk seven" {
		t.Fatalf("Get() = %q, want chunk seven", text)
	}
	chunk, err := Resolve(context.Background(), store, 7)
	if err != nil {
		t.Fatalf("Resolve() error = %v", err)
	}
	if chunk.Ref.ChunkID != 7 || !chunk.Ref.HasFrameOffset || chunk.Ref.FrameOffset != 7 || chunk.Ref.Codec != CodecMemory {
		t.Fatalf("chunk ref = %#v", chunk.Ref)
	}
}

func TestMemvid_InMemoryStore_Bad(t *testing.T) {
	store := NewInMemoryStore(nil)

	_, err := store.Get(context.Background(), 42)

	if !core.Is(err, ErrChunkNotFound) {
		t.Fatalf("missing chunk error = %v, want ErrChunkNotFound", err)
	}
}

func TestMemvid_InMemoryStore_Ugly(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	store := NewInMemoryStore(map[int]string{1: "present"})

	_, err := store.Get(ctx, 1)

	if !core.Is(err, context.Canceled) {
		t.Fatalf("canceled context error = %v, want context.Canceled", err)
	}
}

func TestMemvid_WriterManifest_Good(t *testing.T) {
	store := NewInMemoryStoreWithManifest(
		map[int]string{3: "encoded chunk"},
		map[int]ChunkRef{3: {FrameOffset: 99, HasFrameOffset: true, Codec: CodecQRVideo, Segment: "book-a.mp4"}},
	)

	chunk, err := store.Resolve(context.Background(), 3)
	if err != nil {
		t.Fatalf("Resolve() error = %v", err)
	}
	if chunk.Ref.ChunkID != 3 || chunk.Ref.FrameOffset != 99 || chunk.Ref.Codec != CodecQRVideo || chunk.Ref.Segment != "book-a.mp4" {
		t.Fatalf("manifest ref = %#v", chunk.Ref)
	}
	ref, err := store.Put(context.Background(), "new artifact", PutOptions{Title: "artifact"})
	if err != nil {
		t.Fatalf("Put() error = %v", err)
	}
	if ref.ChunkID != 4 || ref.Codec != CodecMemory {
		t.Fatalf("put ref = %#v, want next memory chunk", ref)
	}
	merged := MergeRef(ChunkRef{ChunkID: 3, Codec: CodecMemory}, ChunkRef{ChunkID: 3, FrameOffset: 12, HasFrameOffset: true})
	if !merged.HasFrameOffset || merged.FrameOffset != 12 || merged.Codec != CodecMemory {
		t.Fatalf("merged ref = %#v", merged)
	}
}

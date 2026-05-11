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

func TestMemvid_ResolveErrors_Bad(t *testing.T) {
	if _, err := Resolve(context.Background(), nil, 7); !core.Is(err, ErrChunkNotFound) {
		t.Fatalf("Resolve(nil) error = %v, want ErrChunkNotFound", err)
	}
	if _, err := ResolveBytes(context.Background(), nil, 7); !core.Is(err, ErrChunkNotFound) {
		t.Fatalf("ResolveBytes(nil) error = %v, want ErrChunkNotFound", err)
	}
	if _, err := ResolveURI(context.Background(), nil, "mlx://missing"); !core.Is(err, ErrChunkNotFound) {
		t.Fatalf("ResolveURI(nil) error = %v, want ErrChunkNotFound", err)
	}
	if got := (&ChunkNotFoundError{ID: 3}).Error(); got != "memvid chunk 3 not found" {
		t.Fatalf("ChunkNotFoundError.Error() = %q", got)
	}
	if got := (&URIChunkNotFoundError{}).Error(); got != "memvid chunk URI not found" {
		t.Fatalf("URIChunkNotFoundError(empty).Error() = %q", got)
	}
	if got := (&URIChunkNotFoundError{URI: "mlx://missing"}).Error(); got != `memvid chunk URI "mlx://missing" not found` {
		t.Fatalf("URIChunkNotFoundError(uri).Error() = %q", got)
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

func TestMemvid_InMemoryStoreCancellation_Ugly(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	store := NewInMemoryStore(map[int]string{1: "present"})

	if _, err := store.ResolveBytes(ctx, 1); !core.Is(err, context.Canceled) {
		t.Fatalf("ResolveBytes(cancelled) error = %v, want context.Canceled", err)
	}
	if _, err := store.ResolveURI(ctx, "mlx://missing"); !core.Is(err, context.Canceled) {
		t.Fatalf("ResolveURI(cancelled) error = %v, want context.Canceled", err)
	}
	if _, err := store.Put(ctx, "text", PutOptions{}); !core.Is(err, context.Canceled) {
		t.Fatalf("Put(cancelled) error = %v, want context.Canceled", err)
	}
	if _, err := store.PutBytes(ctx, []byte("bytes"), PutOptions{}); !core.Is(err, context.Canceled) {
		t.Fatalf("PutBytes(cancelled) error = %v, want context.Canceled", err)
	}
}

func TestMemvid_ResolveBytesFallback_Good(t *testing.T) {
	store := &textOnlyStore{store: NewInMemoryStore(map[int]string{2: "plain"})}

	chunk, err := ResolveBytes(context.Background(), store, 2)
	if err != nil {
		t.Fatalf("ResolveBytes(text fallback) error = %v", err)
	}
	if chunk.Text != "plain" || string(chunk.Data) != "plain" {
		t.Fatalf("ResolveBytes(text fallback) chunk = %+v, want text and byte payload", chunk)
	}
}

func TestMemvid_ResolveRefBytesFallback_Good(t *testing.T) {
	store := &textOnlyStore{store: NewInMemoryStore(map[int]string{2: "plain"})}

	chunk, err := ResolveRefBytes(context.Background(), store, ChunkRef{ChunkID: 2, FrameOffset: 99, HasFrameOffset: true})

	if err != nil {
		t.Fatalf("ResolveRefBytes(fallback) error = %v", err)
	}
	if chunk.Ref.ChunkID != 2 || chunk.Text != "plain" || string(chunk.Data) != "plain" {
		t.Fatalf("ResolveRefBytes(fallback) chunk = %+v, want chunk 2 bytes", chunk)
	}
	if _, err := ResolveRefBytes(context.Background(), nil, ChunkRef{ChunkID: 9}); !core.Is(err, ErrChunkNotFound) {
		t.Fatalf("ResolveRefBytes(nil) error = %v, want ErrChunkNotFound", err)
	}
	if _, err := ResolveRefBytes(context.Background(), store, ChunkRef{}); !core.Is(err, ErrChunkNotFound) {
		t.Fatalf("ResolveRefBytes(empty ref) error = %v, want ErrChunkNotFound", err)
	}
}

func TestMemvid_ResolveGetOnlyFallback_Good(t *testing.T) {
	store := getOnlyStore{chunks: map[int]string{5: "from get"}}

	chunk, err := Resolve(context.Background(), store, 5)
	if err != nil {
		t.Fatalf("Resolve(get only) error = %v", err)
	}
	if chunk.Ref.ChunkID != 5 || chunk.Text != "from get" {
		t.Fatalf("Resolve(get only) chunk = %+v", chunk)
	}
	bytesChunk, err := ResolveBytes(context.Background(), store, 5)
	if err != nil {
		t.Fatalf("ResolveBytes(get only) error = %v", err)
	}
	if bytesChunk.Text != "from get" || string(bytesChunk.Data) != "from get" {
		t.Fatalf("ResolveBytes(get only) chunk = %+v", bytesChunk)
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
	overlay := MergeRef(ChunkRef{ChunkID: 1}, ChunkRef{ChunkID: 2, Codec: CodecQRVideo, Segment: "book.mp4"})
	if overlay.ChunkID != 2 || overlay.Codec != CodecQRVideo || overlay.Segment != "book.mp4" {
		t.Fatalf("overlay ref = %#v, want overlay id/codec/segment", overlay)
	}
	kept := MergeRef(ChunkRef{ChunkID: 9, Codec: CodecMemory}, ChunkRef{})
	if kept.ChunkID != 9 || kept.Codec != CodecMemory {
		t.Fatalf("empty overlay ref = %#v, want base kept", kept)
	}
}

func TestMemvid_BinaryStore_Good(t *testing.T) {
	store := NewInMemoryStore(nil)
	payload := []byte{0, 1, 2, 255}

	ref, err := store.PutBytes(context.Background(), payload, PutOptions{URI: "mlx://binary/1"})
	if err != nil {
		t.Fatalf("PutBytes() error = %v", err)
	}
	payload[1] = 99

	chunk, err := ResolveBytes(context.Background(), store, ref.ChunkID)
	if err != nil {
		t.Fatalf("ResolveBytes() error = %v", err)
	}
	if chunk.Ref.ChunkID != ref.ChunkID || len(chunk.Data) != 4 || chunk.Data[1] != 1 || chunk.Data[3] != 255 {
		t.Fatalf("ResolveBytes() chunk = %+v, want copied binary payload", chunk)
	}
	chunk.Data[2] = 88
	again, err := ResolveBytes(context.Background(), store, ref.ChunkID)
	if err != nil {
		t.Fatalf("ResolveBytes(second) error = %v", err)
	}
	if again.Data[2] != 2 {
		t.Fatalf("ResolveBytes() returned aliased data = %v", again.Data)
	}
	if text, err := store.Get(context.Background(), ref.ChunkID); err != nil || text != string([]byte{0, 1, 2, 255}) {
		t.Fatalf("Get(binary) = %q, %v; want text fallback", text, err)
	}
	byURI, err := ResolveURI(context.Background(), store, "mlx://binary/1")
	if err != nil {
		t.Fatalf("ResolveURI(binary) error = %v", err)
	}
	if len(byURI.Data) != 4 || byURI.Data[0] != 0 {
		t.Fatalf("ResolveURI(binary) chunk = %+v, want binary data", byURI)
	}
}

func TestMemvid_BinaryStoreErrors_Bad(t *testing.T) {
	var store *InMemoryStore
	if _, err := store.Put(context.Background(), "text", PutOptions{}); !core.Is(err, ErrChunkNotFound) {
		t.Fatalf("Put(nil store) error = %v, want ErrChunkNotFound", err)
	}
	if _, err := store.PutBytes(context.Background(), []byte("bytes"), PutOptions{}); !core.Is(err, ErrChunkNotFound) {
		t.Fatalf("PutBytes(nil store) error = %v, want ErrChunkNotFound", err)
	}
	if _, err := store.Resolve(context.Background(), 1); !core.Is(err, ErrChunkNotFound) {
		t.Fatalf("Resolve(nil store) error = %v, want ErrChunkNotFound", err)
	}
	if _, err := store.ResolveBytes(context.Background(), 1); !core.Is(err, ErrChunkNotFound) {
		t.Fatalf("ResolveBytes(nil store) error = %v, want ErrChunkNotFound", err)
	}
	if _, err := store.ResolveURI(context.Background(), "mlx://missing"); !core.Is(err, ErrChunkNotFound) {
		t.Fatalf("ResolveURI(nil store) error = %v, want ErrChunkNotFound", err)
	}
}

type textOnlyStore struct {
	store *InMemoryStore
}

func (s *textOnlyStore) Get(ctx context.Context, chunkID int) (string, error) {
	return s.store.Get(ctx, chunkID)
}

func (s *textOnlyStore) Resolve(ctx context.Context, chunkID int) (Chunk, error) {
	return s.store.Resolve(ctx, chunkID)
}

type getOnlyStore struct {
	chunks map[int]string
}

func (s getOnlyStore) Get(_ context.Context, chunkID int) (string, error) {
	text, ok := s.chunks[chunkID]
	if !ok {
		return "", &ChunkNotFoundError{ID: chunkID}
	}
	return text, nil
}

func TestMemvid_ResolveURI_Good(t *testing.T) {
	store := NewInMemoryStore(nil)
	ref, err := store.Put(context.Background(), "manifest", PutOptions{URI: "mlx://bundle/1"})
	if err != nil {
		t.Fatalf("Put() error = %v", err)
	}

	chunk, err := ResolveURI(context.Background(), store, "mlx://bundle/1")
	if err != nil {
		t.Fatalf("ResolveURI() error = %v", err)
	}
	if chunk.Text != "manifest" || chunk.Ref.ChunkID != ref.ChunkID {
		t.Fatalf("ResolveURI() chunk = %+v, want manifest ref %d", chunk, ref.ChunkID)
	}
	_, err = ResolveURI(context.Background(), store, "mlx://missing")
	if !core.Is(err, ErrChunkNotFound) {
		t.Fatalf("ResolveURI(missing) error = %v, want ErrChunkNotFound", err)
	}
}

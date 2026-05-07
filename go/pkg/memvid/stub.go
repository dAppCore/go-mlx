// SPDX-Licence-Identifier: EUPL-1.2

package memvid

import "context"

type InMemoryStore struct {
	chunks map[int]string
	refs   map[int]ChunkRef
	nextID int
}

func NewInMemoryStore(chunks map[int]string) *InMemoryStore {
	return NewInMemoryStoreWithManifest(chunks, nil)
}

func NewInMemoryStoreWithManifest(chunks map[int]string, refs map[int]ChunkRef) *InMemoryStore {
	copyMap := make(map[int]string, len(chunks))
	nextID := 1
	for id, text := range chunks {
		copyMap[id] = text
		if id >= nextID {
			nextID = id + 1
		}
	}
	refMap := make(map[int]ChunkRef, len(copyMap))
	for id := range copyMap {
		refMap[id] = ChunkRef{
			ChunkID:        id,
			FrameOffset:    uint64(id),
			HasFrameOffset: true,
			Codec:          CodecMemory,
		}
	}
	for id, ref := range refs {
		ref.ChunkID = id
		refMap[id] = ref
		if id >= nextID {
			nextID = id + 1
		}
	}
	return &InMemoryStore{
		chunks: copyMap,
		refs:   refMap,
		nextID: nextID,
	}
}

func (s *InMemoryStore) Get(ctx context.Context, chunkID int) (string, error) {
	chunk, err := s.Resolve(ctx, chunkID)
	if err != nil {
		return "", err
	}
	return chunk.Text, nil
}

func (s *InMemoryStore) Resolve(ctx context.Context, chunkID int) (Chunk, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	select {
	case <-ctx.Done():
		return Chunk{}, ctx.Err()
	default:
	}
	if s == nil {
		return Chunk{}, &ChunkNotFoundError{ID: chunkID}
	}
	text, ok := s.chunks[chunkID]
	if !ok {
		return Chunk{}, &ChunkNotFoundError{ID: chunkID}
	}
	ref := s.refs[chunkID]
	if ref.ChunkID != chunkID {
		ref.ChunkID = chunkID
	}
	return Chunk{Ref: ref, Text: text}, nil
}

func (s *InMemoryStore) Put(ctx context.Context, text string, _ PutOptions) (ChunkRef, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	select {
	case <-ctx.Done():
		return ChunkRef{}, ctx.Err()
	default:
	}
	if s == nil {
		return ChunkRef{}, &ChunkNotFoundError{}
	}
	if s.chunks == nil {
		s.chunks = make(map[int]string)
	}
	if s.refs == nil {
		s.refs = make(map[int]ChunkRef)
	}
	if s.nextID <= 0 {
		s.nextID = 1
	}
	id := s.nextID
	s.nextID++
	ref := ChunkRef{
		ChunkID:        id,
		FrameOffset:    uint64(id),
		HasFrameOffset: true,
		Codec:          CodecMemory,
	}
	s.chunks[id] = text
	s.refs[id] = ref
	return ref, nil
}

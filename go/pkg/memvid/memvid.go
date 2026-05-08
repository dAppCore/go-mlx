// SPDX-Licence-Identifier: EUPL-1.2

// Package memvid defines the cold-store contract used by go-mlx artifacts.
package memvid

import (
	"context"

	core "dappco.re/go"
)

var ErrChunkNotFound = core.NewError("memvid chunk not found")

const (
	CodecMemory  = "memory/plaintext"
	CodecQRVideo = "memvid/qr-video"
)

type Store interface {
	Get(ctx context.Context, chunkID int) (string, error)
}

type Resolver interface {
	Resolve(ctx context.Context, chunkID int) (Chunk, error)
}

type Writer interface {
	Put(ctx context.Context, text string, opts PutOptions) (ChunkRef, error)
}

type PutOptions struct {
	URI    string            `json:"uri,omitempty"`
	Title  string            `json:"title,omitempty"`
	Kind   string            `json:"kind,omitempty"`
	Track  string            `json:"track,omitempty"`
	Tags   map[string]string `json:"tags,omitempty"`
	Labels []string          `json:"labels,omitempty"`
}

type Chunk struct {
	Ref  ChunkRef `json:"ref"`
	Text string   `json:"text"`
}

type ChunkRef struct {
	ChunkID        int    `json:"chunk_id"`
	FrameOffset    uint64 `json:"frame_offset,omitempty"`
	HasFrameOffset bool   `json:"has_frame_offset,omitempty"`
	Codec          string `json:"codec,omitempty"`
	Segment        string `json:"segment,omitempty"`
}

type ChunkNotFoundError struct {
	ID int
}

func (e *ChunkNotFoundError) Error() string {
	return core.Sprintf("memvid chunk %d not found", e.ID)
}

func (e *ChunkNotFoundError) Unwrap() error {
	return ErrChunkNotFound
}

func Resolve(ctx context.Context, store Store, chunkID int) (Chunk, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if store == nil {
		return Chunk{}, &ChunkNotFoundError{ID: chunkID}
	}
	if resolver, ok := store.(Resolver); ok {
		return resolver.Resolve(ctx, chunkID)
	}
	text, err := store.Get(ctx, chunkID)
	if err != nil {
		return Chunk{}, err
	}
	return Chunk{
		Ref:  ChunkRef{ChunkID: chunkID},
		Text: text,
	}, nil
}

func MergeRef(base, overlay ChunkRef) ChunkRef {
	out := base
	if overlay.ChunkID != 0 || base.ChunkID == 0 {
		out.ChunkID = overlay.ChunkID
	}
	if overlay.HasFrameOffset {
		out.FrameOffset = overlay.FrameOffset
		out.HasFrameOffset = true
	}
	if overlay.Codec != "" {
		out.Codec = overlay.Codec
	}
	if overlay.Segment != "" {
		out.Segment = overlay.Segment
	}
	return out
}

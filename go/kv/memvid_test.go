// SPDX-Licence-Identifier: EUPL-1.2

package kv

import (
	"context"
	"testing"

	core "dappco.re/go"
	memvid "dappco.re/go/inference/state"
)

func TestKVSnapshotMemvid_Good_SaveLoadRoundTrip(t *testing.T) {
	store := memvid.NewInMemoryStore(nil)
	snapshot := testSnapshot()

	ref, err := snapshot.SaveMemvid(context.Background(), store, MemvidOptions{
		KVEncoding: EncodingQ8,
		URI:        "mlx://session/test",
		Title:      "test session",
		Labels:     []string{"session-kv"},
	})
	if err != nil {
		t.Fatalf("SaveMemvid() error = %v", err)
	}
	if ref.ChunkID == 0 || ref.Codec != memvid.CodecMemory {
		t.Fatalf("memvid ref = %+v, want in-memory chunk ref", ref)
	}
	chunk, err := memvid.Resolve(context.Background(), store, ref.ChunkID)
	if err != nil {
		t.Fatalf("Resolve() error = %v", err)
	}
	if !core.Contains(chunk.Text, `"kind":"`+KVSnapshotMemvidKind+`"`) || !core.Contains(chunk.Text, `"binary_encoding":"base64"`) {
		t.Fatalf("memvid payload = %s, want KV envelope", chunk.Text)
	}

	loaded, err := LoadFromMemvid(context.Background(), store, ref)
	if err != nil {
		t.Fatalf("LoadFromMemvid() error = %v", err)
	}
	if loaded.Architecture != snapshot.Architecture || loaded.TokenOffset != snapshot.TokenOffset || loaded.NumLayers != snapshot.NumLayers {
		t.Fatalf("loaded metadata = %+v, want %+v", loaded, snapshot)
	}
	head, ok := loaded.Head(0, 0)
	if !ok {
		t.Fatal("loaded Head(0, 0) ok = false, want true")
	}
	if len(head.Key) != len(snapshot.Layers[0].Heads[0].Key) || len(head.Value) != len(snapshot.Layers[0].Heads[0].Value) {
		t.Fatalf("loaded head = %+v, want same tensor sizes", head)
	}
}

func TestKVSnapshotMemvid_Bad_LoadRejectsHashMismatch(t *testing.T) {
	store := memvid.NewInMemoryStore(map[int]string{
		1: `{"version":1,"kind":"` + KVSnapshotMemvidKind + `","binary_encoding":"base64","kv_hash":"sha256:not-it","data":"` + core.Base64Encode([]byte(kvSnapshotMagic)) + `"}`,
	})

	_, err := LoadFromMemvid(context.Background(), store, memvid.ChunkRef{ChunkID: 1})

	if err == nil {
		t.Fatal("LoadFromMemvid() error = nil, want hash mismatch")
	}
}

func TestKVSnapshotMemvid_Bad_SaveErrors(t *testing.T) {
	var snapshot *Snapshot
	if _, err := snapshot.SaveMemvid(context.Background(), memvid.NewInMemoryStore(nil), MemvidOptions{}); err == nil {
		t.Fatal("SaveMemvid(nil snapshot) error = nil")
	}
	if _, err := testSnapshot().SaveMemvid(context.Background(), nil, MemvidOptions{}); err == nil {
		t.Fatal("SaveMemvid(nil store) error = nil")
	}
	if _, err := testSnapshot().SaveMemvid(context.Background(), memvid.NewInMemoryStore(nil), MemvidOptions{KVEncoding: "q2"}); err == nil {
		t.Fatal("SaveMemvid(bad encoding) error = nil")
	}
	if _, err := testSnapshot().SaveMemvid(nil, failingMemvidWriter{}, MemvidOptions{}); err == nil {
		t.Fatal("SaveMemvid(write failure) error = nil")
	}
}

func TestKVSnapshotMemvid_Bad_LoadEnvelopeErrors(t *testing.T) {
	if _, err := LoadFromMemvid(context.Background(), nil, memvid.ChunkRef{ChunkID: 1}); err == nil {
		t.Fatal("LoadFromMemvid(nil store) error = nil")
	}
	store := memvid.NewInMemoryStore(map[int]string{1: "{"})
	if _, err := LoadFromMemvid(nil, store, memvid.ChunkRef{ChunkID: 1}); err == nil {
		t.Fatal("LoadFromMemvid(corrupt JSON) error = nil")
	}

	for _, envelope := range []kvSnapshotMemvidEnvelope{
		{Version: KVSnapshotMemvidVersion + 1, Kind: KVSnapshotMemvidKind, BinaryEncoding: "base64"},
		{Version: KVSnapshotMemvidVersion, Kind: "wrong", BinaryEncoding: "base64"},
		{Version: KVSnapshotMemvidVersion, Kind: KVSnapshotMemvidKind, BinaryEncoding: "hex"},
		{Version: KVSnapshotMemvidVersion, Kind: KVSnapshotMemvidKind, BinaryEncoding: "base64", Data: "not base64"},
		{Version: KVSnapshotMemvidVersion, Kind: KVSnapshotMemvidKind, BinaryEncoding: "base64", Data: core.Base64Encode([]byte("x")), PayloadByteCount: 2},
	} {
		if _, err := decodeKVSnapshotMemvidEnvelope(envelope); err == nil {
			t.Fatalf("decodeKVSnapshotMemvidEnvelope(%+v) error = nil", envelope)
		}
	}
	if data, err := decodeKVSnapshotMemvidEnvelope(kvSnapshotMemvidEnvelope{
		Version:        KVSnapshotMemvidVersion,
		Kind:           KVSnapshotMemvidKind,
		BinaryEncoding: "base64",
		Data:           core.Base64Encode([]byte("x")),
	}); err != nil || string(data) != "x" {
		t.Fatalf("decodeKVSnapshotMemvidEnvelope(valid) = %q/%v, want x/nil", string(data), err)
	}
}

func TestKVSnapshotMemvidHelpers_Good(t *testing.T) {
	snapshot := testSnapshot()
	snapshot.Version = 0
	opts := kvSnapshotMemvidPutOptions(snapshot, MemvidOptions{
		Kind:   "custom-kind",
		Track:  "custom-track",
		URI:    "mlx://custom",
		Title:  "custom title",
		Tags:   map[string]string{"caller": "yes"},
		Labels: []string{"caller-label"},
	}, kvSnapshotMemvidEnvelope{
		KVHash:           "hash",
		KVEncoding:       string(EncodingNative),
		Architecture:     "gemma4_text",
		TokenCount:       2,
		PayloadByteCount: 32,
	})
	if opts.Kind != "custom-kind" || opts.Track != "custom-track" || opts.URI != "mlx://custom" || opts.Title != "custom title" {
		t.Fatalf("put options = %+v, want caller metadata", opts)
	}
	if opts.Tags["caller"] != "yes" || opts.Tags["kv_hash"] != "hash" || opts.Tags["payload_bytes"] != "32" {
		t.Fatalf("put option tags = %+v, want caller and KV tags", opts.Tags)
	}
	if got := effectiveVersion(snapshot, EncodingQ8); got != 3 {
		t.Fatalf("effectiveVersion(q8) = %d, want 3", got)
	}
	if got := EffectiveTokenOffset(&Snapshot{Tokens: []int32{1, 2, 3}}); got != 3 {
		t.Fatalf("EffectiveTokenOffset(default) = %d, want token length", got)
	}
	if got := EffectiveTokenOffset(nil); got != 0 {
		t.Fatalf("EffectiveTokenOffset(nil) = %d, want 0", got)
	}
	sourceTags := map[string]string{"a": "b"}
	tags := cloneKVSnapshotMemvidTags(sourceTags)
	tags["a"] = "changed"
	if sourceTags["a"] != "b" {
		t.Fatalf("source tags were mutated: %+v", sourceTags)
	}
}

type failingMemvidWriter struct{}

func (failingMemvidWriter) Put(context.Context, string, memvid.PutOptions) (memvid.ChunkRef, error) {
	return memvid.ChunkRef{}, core.NewError("put failed")
}

// SPDX-Licence-Identifier: EUPL-1.2

package kv

import (
	"context"
	"testing"

	core "dappco.re/go"
	state "dappco.re/go/inference/state"
)

func TestKVSnapshotState_Good_SaveLoadRoundTrip(t *testing.T) {
	store := state.NewInMemoryStore(nil)
	snapshot := testSnapshot()

	ref, err := snapshot.SaveState(context.Background(), store, StateOptions{
		KVEncoding: EncodingQ8,
		URI:        "mlx://session/test",
		Title:      "test session",
		Labels:     []string{"session-kv"},
	})
	if err != nil {
		t.Fatalf("SaveState() error = %v", err)
	}
	if ref.ChunkID == 0 || ref.Codec != state.CodecMemory {
		t.Fatalf("State ref = %+v, want in-memory chunk ref", ref)
	}
	chunk, err := state.Resolve(context.Background(), store, ref.ChunkID)
	if err != nil {
		t.Fatalf("Resolve() error = %v", err)
	}
	if !core.Contains(chunk.Text, `"kind":"`+KVSnapshotStateKind+`"`) || !core.Contains(chunk.Text, `"binary_encoding":"base64"`) {
		t.Fatalf("State payload = %s, want KV envelope", chunk.Text)
	}

	loaded, err := LoadFromState(context.Background(), store, ref)
	if err != nil {
		t.Fatalf("LoadFromState() error = %v", err)
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

func TestKVSnapshotState_Bad_LoadRejectsHashMismatch(t *testing.T) {
	store := state.NewInMemoryStore(map[int]string{
		1: `{"version":1,"kind":"` + KVSnapshotStateKind + `","binary_encoding":"base64","kv_hash":"sha256:not-it","data":"` + core.Base64Encode([]byte(kvSnapshotMagic)) + `"}`,
	})

	_, err := LoadFromState(context.Background(), store, state.ChunkRef{ChunkID: 1})

	if err == nil {
		t.Fatal("LoadFromState() error = nil, want hash mismatch")
	}
}

func TestKVSnapshotState_Bad_SaveErrors(t *testing.T) {
	var snapshot *Snapshot
	if _, err := snapshot.SaveState(context.Background(), state.NewInMemoryStore(nil), StateOptions{}); err == nil {
		t.Fatal("SaveState(nil snapshot) error = nil")
	}
	if _, err := testSnapshot().SaveState(context.Background(), nil, StateOptions{}); err == nil {
		t.Fatal("SaveState(nil store) error = nil")
	}
	if _, err := testSnapshot().SaveState(context.Background(), state.NewInMemoryStore(nil), StateOptions{KVEncoding: "q2"}); err == nil {
		t.Fatal("SaveState(bad encoding) error = nil")
	}
	if _, err := testSnapshot().SaveState(nil, failingStateWriter{}, StateOptions{}); err == nil {
		t.Fatal("SaveState(write failure) error = nil")
	}
}

func TestKVSnapshotState_Bad_LoadEnvelopeErrors(t *testing.T) {
	if _, err := LoadFromState(context.Background(), nil, state.ChunkRef{ChunkID: 1}); err == nil {
		t.Fatal("LoadFromState(nil store) error = nil")
	}
	store := state.NewInMemoryStore(map[int]string{1: "{"})
	if _, err := LoadFromState(nil, store, state.ChunkRef{ChunkID: 1}); err == nil {
		t.Fatal("LoadFromState(corrupt JSON) error = nil")
	}

	for _, envelope := range []kvSnapshotStateEnvelope{
		{Version: KVSnapshotStateVersion + 1, Kind: KVSnapshotStateKind, BinaryEncoding: "base64"},
		{Version: KVSnapshotStateVersion, Kind: "wrong", BinaryEncoding: "base64"},
		{Version: KVSnapshotStateVersion, Kind: KVSnapshotStateKind, BinaryEncoding: "hex"},
		{Version: KVSnapshotStateVersion, Kind: KVSnapshotStateKind, BinaryEncoding: "base64", Data: "not base64"},
		{Version: KVSnapshotStateVersion, Kind: KVSnapshotStateKind, BinaryEncoding: "base64", Data: core.Base64Encode([]byte("x")), PayloadByteCount: 2},
	} {
		if _, err := decodeKVSnapshotStateEnvelope(envelope); err == nil {
			t.Fatalf("decodeKVSnapshotStateEnvelope(%+v) error = nil", envelope)
		}
	}
	if data, err := decodeKVSnapshotStateEnvelope(kvSnapshotStateEnvelope{
		Version:        KVSnapshotStateVersion,
		Kind:           KVSnapshotStateKind,
		BinaryEncoding: "base64",
		Data:           core.Base64Encode([]byte("x")),
	}); err != nil || string(data) != "x" {
		t.Fatalf("decodeKVSnapshotStateEnvelope(valid) = %q/%v, want x/nil", string(data), err)
	}
}

func TestKVSnapshotStateHelpers_Good(t *testing.T) {
	snapshot := testSnapshot()
	snapshot.Version = 0
	opts := kvSnapshotStatePutOptions(snapshot, StateOptions{
		Kind:   "custom-kind",
		Track:  "custom-track",
		URI:    "mlx://custom",
		Title:  "custom title",
		Tags:   map[string]string{"caller": "yes"},
		Labels: []string{"caller-label"},
	}, kvSnapshotStateEnvelope{
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
	if got := effectiveVersion(snapshot, EncodingQ8); got != SnapshotVersion {
		t.Fatalf("effectiveVersion(q8) = %d, want %d", got, SnapshotVersion)
	}
	if got := EffectiveTokenOffset(&Snapshot{Tokens: []int32{1, 2, 3}}); got != 3 {
		t.Fatalf("EffectiveTokenOffset(default) = %d, want token length", got)
	}
	if got := EffectiveTokenOffset(nil); got != 0 {
		t.Fatalf("EffectiveTokenOffset(nil) = %d, want 0", got)
	}
	sourceTags := map[string]string{"a": "b"}
	tags := cloneKVSnapshotStateTags(sourceTags)
	tags["a"] = "changed"
	if sourceTags["a"] != "b" {
		t.Fatalf("source tags were mutated: %+v", sourceTags)
	}
}

type failingStateWriter struct{}

func (failingStateWriter) Put(context.Context, string, state.PutOptions) (state.ChunkRef, error) {
	return state.ChunkRef{}, core.NewError("put failed")
}

// TestStateStore_SaveMemvid_Good asserts the deprecated SaveMemvid alias writes
// a chunk that the canonical LoadFromState path decodes back to the same KV
// state — the alias must be a transparent forward to SaveState.
func TestStateStore_SaveMemvid_Good(t *testing.T) {
	store := state.NewInMemoryStore(nil)
	snapshot := testSnapshot()

	ref, err := snapshot.SaveMemvid(context.Background(), store, MemvidOptions{
		KVEncoding: EncodingQ8,
		URI:        "mlx://session/memvid",
		Title:      "memvid session",
	})
	if err != nil {
		t.Fatalf("SaveMemvid() error = %v", err)
	}
	if ref.ChunkID == 0 {
		t.Fatalf("SaveMemvid() ref = %+v, want a written chunk", ref)
	}

	loaded, err := LoadFromState(context.Background(), store, ref)
	if err != nil {
		t.Fatalf("LoadFromState() error = %v", err)
	}
	if loaded.Architecture != snapshot.Architecture || loaded.NumLayers != snapshot.NumLayers {
		t.Fatalf("loaded metadata = %+v, want %+v", loaded, snapshot)
	}
}

// TestStateStore_LoadFromMemvid_Good asserts the deprecated LoadFromMemvid alias
// decodes a chunk written by the canonical SaveState path.
func TestStateStore_LoadFromMemvid_Good(t *testing.T) {
	store := state.NewInMemoryStore(nil)
	snapshot := testSnapshot()
	ref, err := snapshot.SaveState(context.Background(), store, StateOptions{KVEncoding: EncodingQ8})
	if err != nil {
		t.Fatalf("SaveState() error = %v", err)
	}

	loaded, err := LoadFromMemvid(context.Background(), store, ref)
	if err != nil {
		t.Fatalf("LoadFromMemvid() error = %v", err)
	}
	if loaded.TokenOffset != snapshot.TokenOffset || loaded.NumHeads != snapshot.NumHeads {
		t.Fatalf("loaded metadata = %+v, want %+v", loaded, snapshot)
	}
}

// TestStateStore_LoadFromMemvidWithOptions_Good asserts the deprecated
// LoadFromMemvidWithOptions alias forwards decode options to
// LoadFromStateWithOptions: RawKVOnly skips float32 reconstruction so the loaded
// head exposes raw bytes rather than decoded values.
func TestStateStore_LoadFromMemvidWithOptions_Good(t *testing.T) {
	store := state.NewInMemoryStore(nil)
	snapshot := testSnapshot()
	ref, err := snapshot.SaveState(context.Background(), store, StateOptions{KVEncoding: EncodingNative})
	if err != nil {
		t.Fatalf("SaveState() error = %v", err)
	}

	loaded, err := LoadFromMemvidWithOptions(context.Background(), store, ref, LoadOptions{RawKVOnly: true})
	if err != nil {
		t.Fatalf("LoadFromMemvidWithOptions() error = %v", err)
	}
	head, ok := loaded.Head(0, 0)
	if !ok {
		t.Fatal("loaded Head(0, 0) ok = false, want true")
	}
	if len(head.KeyBytes) == 0 {
		t.Fatalf("loaded head = %+v, want raw key bytes retained under RawKVOnly", head)
	}
}

// TestStateStore_LoadFromMemvid_Bad asserts the deprecated load aliases surface
// the same guard errors as the canonical path (nil store, missing chunk).
func TestStateStore_LoadFromMemvid_Bad(t *testing.T) {
	if _, err := LoadFromMemvid(context.Background(), nil, state.ChunkRef{ChunkID: 1}); err == nil {
		t.Fatal("LoadFromMemvid(nil store) error = nil, want store error")
	}
	if _, err := LoadFromMemvidWithOptions(context.Background(), nil, state.ChunkRef{ChunkID: 1}, LoadOptions{}); err == nil {
		t.Fatal("LoadFromMemvidWithOptions(nil store) error = nil, want store error")
	}
	store := state.NewInMemoryStore(nil)
	if _, err := LoadFromMemvid(context.Background(), store, state.ChunkRef{ChunkID: 999}); err == nil {
		t.Fatal("LoadFromMemvid(missing chunk) error = nil, want resolve error")
	}
}

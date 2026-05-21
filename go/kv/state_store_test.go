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

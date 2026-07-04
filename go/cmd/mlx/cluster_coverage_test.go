// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"testing"

	core "dappco.re/go"
)

// stateKVContainerEncode fails when the output directory can't be created
// (parent is a file), and when the payload file is missing.
func TestStateKVContainerEncode_Errors_Bad(t *testing.T) {
	dir := t.TempDir()
	// A real payload to copy from.
	payload := core.PathJoin(dir, "payload.bin")
	if r := core.WriteFile(payload, []byte("data"), 0o644); !r.OK {
		t.Fatalf("write payload: %v", r.Value)
	}
	// Output parent is a FILE → MkdirAll(parent) fails.
	blocker := core.PathJoin(dir, "blocker")
	if r := core.WriteFile(blocker, []byte("x"), 0o644); !r.OK {
		t.Fatalf("write blocker: %v", r.Value)
	}
	if _, err := stateKVContainerEncode(core.PathJoin(blocker, "sub", "out.kv"), map[string]any{}, payload); err == nil {
		t.Fatal("encode under a file error = nil, want the mkdir failure")
	}
	// Missing payload file → open error.
	if _, err := stateKVContainerEncode(core.PathJoin(dir, "out.kv"), map[string]any{}, core.PathJoin(dir, "absent-payload")); err == nil {
		t.Fatal("encode with a missing payload error = nil, want the open failure")
	}
}

// stateKVContainerFileHasMagic reports false (no error) for a short/empty file
// and surfaces a read error for a missing file.
func TestStateKVContainerFileHasMagic_Edges(t *testing.T) {
	dir := t.TempDir()
	// Empty file → not a container, no error.
	empty := core.PathJoin(dir, "empty.bin")
	if r := core.WriteFile(empty, nil, 0o644); !r.OK {
		t.Fatalf("write empty: %v", r.Value)
	}
	has, err := stateKVContainerFileHasMagic(empty)
	if err != nil || has {
		t.Fatalf("empty file = (%v, %v), want (false, nil)", has, err)
	}
	// A short file (< 4 bytes) → not a container, no error.
	short := core.PathJoin(dir, "short.bin")
	if r := core.WriteFile(short, []byte("KV"), 0o644); !r.OK {
		t.Fatalf("write short: %v", r.Value)
	}
	if has, err := stateKVContainerFileHasMagic(short); err != nil || has {
		t.Fatalf("short file = (%v, %v), want (false, nil)", has, err)
	}
	// Missing file → read error.
	if _, err := stateKVContainerFileHasMagic(core.PathJoin(dir, "absent")); err == nil {
		t.Fatal("missing file error = nil, want the open error")
	}
}

// stateWakeProfileMarkerSourceFromFile surfaces the read error for a missing
// file, and the JSON error for a non-container file that isn't valid marker
// JSON.
func TestStateWakeProfileMarkerSourceFromFile_Errors_Bad(t *testing.T) {
	dir := t.TempDir()
	if _, err := stateWakeProfileMarkerSourceFromFile(core.PathJoin(dir, "absent")); err == nil {
		t.Fatal("missing file error = nil, want a read error")
	}
	// A non-container file with invalid JSON.
	badJSON := core.PathJoin(dir, "bad.json")
	if r := core.WriteFile(badJSON, []byte("{not json"), 0o644); !r.OK {
		t.Fatalf("write: %v", r.Value)
	}
	if _, err := stateWakeProfileMarkerSourceFromFile(badJSON); err == nil {
		t.Fatal("invalid-JSON marker error = nil, want a decode error")
	}
}

// stateKVContainerMarkerSourceFromFile errors on a file that is not a valid
// Trix container (bad header).
func TestStateKVContainerMarkerSourceFromFile_BadContainer_Bad(t *testing.T) {
	dir := t.TempDir()
	// Starts with the magic so the caller treats it as a container, but the
	// rest is junk → ReadHeaderInfo fails.
	fake := core.PathJoin(dir, "fake.kv")
	if r := core.WriteFile(fake, []byte("KVST garbage-not-a-real-trix-header"), 0o644); !r.OK {
		t.Fatalf("write: %v", r.Value)
	}
	if _, err := stateKVContainerMarkerSourceFromFile(fake); err == nil {
		t.Fatal("bad container error = nil, want a header-read error")
	}
	// Missing file → open error.
	if _, err := stateKVContainerMarkerSourceFromFile(core.PathJoin(dir, "absent")); err == nil {
		t.Fatal("missing container error = nil, want an open error")
	}
}

// stateKVContainerMarkerFromHeader rejects a header with the wrong content type
// (the kind matches but the content_type does not).
func TestStateKVContainerMarkerFromHeader_WrongContentType_Bad(t *testing.T) {
	header := map[string]any{
		"kind":         stateKVContainerKind,
		"content_type": "application/x-wrong",
	}
	if _, err := stateKVContainerMarkerFromHeader(header, 0); err == nil {
		t.Fatal("wrong content_type error = nil, want a content-type mismatch error")
	}
}

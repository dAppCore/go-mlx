// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"context"
	"testing"

	core "dappco.re/go"
	trix "forge.lthn.ai/Snider/Enchantrix/pkg/trix"
)

func TestRunCommand_StatePack_Good(t *testing.T) {
	dir := t.TempDir()
	statePath := core.PathJoin(dir, "session.mvlog")
	markerPath := core.PathJoin(dir, "ramp-report.json")
	outputPath := core.PathJoin(dir, "session.kv")
	payload := []byte("go-mlx-state-log\nbinary\x00tail")
	if result := core.WriteFile(statePath, payload, 0o600); !result.OK {
		t.Fatalf("write state: %v", result.Value)
	}
	writeCLIPackFile(t, markerPath, `{
  "fold": {
    "compact_marker": {
      "store_path": "`+statePath+`",
      "index_uri": "mlx://state-ramp/fold/1/folded/index",
      "entry_uri": "mlx://state-ramp/fold/1/folded",
      "bundle_uri": "mlx://state-ramp/fold/1/folded/bundle",
      "token_count": 206
    }
  }
}`)
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{
		"state-pack",
		"-json",
		"-marker-file", markerPath,
		"-output", outputPath,
	}, stdout, stderr)

	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr=%q stdout=%q", code, stderr.String(), stdout.String())
	}
	if !core.Contains(stdout.String(), `"magic": "KVST"`) || !core.Contains(stdout.String(), core.Sprintf(`"payload_bytes": %d`, len(payload))) {
		t.Fatalf("stdout = %q, want pack report", stdout.String())
	}
	read := core.ReadFile(outputPath)
	if !read.OK {
		t.Fatalf("read output: %v", read.Value)
	}
	decoded, err := trix.Decode(read.Value.([]byte), stateKVContainerMagic, nil)
	if err != nil {
		t.Fatalf("decode trix: %v", err)
	}
	if string(decoded.Payload) != string(payload) {
		t.Fatalf("payload = %q, want original payload", string(decoded.Payload))
	}
	if decoded.Header["kind"] != stateKVContainerKind || decoded.Header["content_type"] != stateKVContainerContentType {
		t.Fatalf("header = %#v, want State KV metadata", decoded.Header)
	}
	if decoded.Header["index_uri"] != "mlx://state-ramp/fold/1/folded/index" {
		t.Fatalf("index_uri = %#v, want folded index", decoded.Header["index_uri"])
	}
}

func TestRunCommand_StatePackValidation_Bad(t *testing.T) {
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{"state-pack", "-output", "state.kv"}, stdout, stderr)

	if code != 2 {
		t.Fatalf("exit code = %d, want 2", code)
	}
	if !core.Contains(stderr.String(), "marker file is required") {
		t.Fatalf("stderr = %q, want marker validation", stderr.String())
	}
}

// ---- pure header accessors -------------------------------------------

// TestStatePack_StateKVHeaderString_Good — a string-typed key reads back
// verbatim.
func TestStatePack_StateKVHeaderString_Good(t *testing.T) {
	header := map[string]any{"kind": stateKVContainerKind}
	if got := stateKVHeaderString(header, "kind"); got != stateKVContainerKind {
		t.Fatalf("stateKVHeaderString(kind) = %q, want %q", got, stateKVContainerKind)
	}
}

// TestStatePack_StateKVHeaderString_Bad — a missing key yields "".
func TestStatePack_StateKVHeaderString_Bad(t *testing.T) {
	if got := stateKVHeaderString(map[string]any{}, "absent"); got != "" {
		t.Fatalf("stateKVHeaderString(absent) = %q, want empty", got)
	}
}

// TestStatePack_StateKVHeaderString_Ugly — a key present but non-string
// (the JSON decoded it as a number) yields "" rather than panicking on
// the type assertion.
func TestStatePack_StateKVHeaderString_Ugly(t *testing.T) {
	if got := stateKVHeaderString(map[string]any{"token_count": float64(7)}, "token_count"); got != "" {
		t.Fatalf("stateKVHeaderString(number) = %q, want empty", got)
	}
}

// TestStatePack_StateKVHeaderInt64_Good — int, int64 and float64 (the
// shape encoding/json produces for numbers) all coerce to int64.
func TestStatePack_StateKVHeaderInt64_Good(t *testing.T) {
	for name, v := range map[string]any{"int": 5, "int64": int64(5), "float64": float64(5)} {
		header := map[string]any{"n": v}
		if got := stateKVHeaderInt64(header, "n"); got != 5 {
			t.Fatalf("stateKVHeaderInt64(%s) = %d, want 5", name, got)
		}
	}
}

// TestStatePack_StateKVHeaderInt64_Bad — a missing key yields 0.
func TestStatePack_StateKVHeaderInt64_Bad(t *testing.T) {
	if got := stateKVHeaderInt64(map[string]any{}, "absent"); got != 0 {
		t.Fatalf("stateKVHeaderInt64(absent) = %d, want 0", got)
	}
}

// TestStatePack_StateKVHeaderInt64_Ugly — a string-typed value falls
// through the type switch to the 0 default rather than parsing it.
func TestStatePack_StateKVHeaderInt64_Ugly(t *testing.T) {
	if got := stateKVHeaderInt64(map[string]any{"n": "42"}, "n"); got != 0 {
		t.Fatalf("stateKVHeaderInt64(string) = %d, want 0", got)
	}
}

// ---- header → marker validation --------------------------------------

func validStateKVHeader() map[string]any {
	return map[string]any{
		"kind":             stateKVContainerKind,
		"content_type":     stateKVContainerContentType,
		"payload_bytes":    int64(128),
		"state_store_path": "/runs/session.mvlog",
		"index_uri":        "mlx://state-ramp/fold/1/folded/index",
		"entry_uri":        "mlx://state-ramp/fold/1/folded",
		"bundle_uri":       "mlx://state-ramp/fold/1/folded/bundle",
		"token_count":      int64(206),
	}
}

// TestStatePack_MarkerFromHeader_Good — a well-formed header whose
// payload_bytes matches the on-disk size yields the full marker.
func TestStatePack_MarkerFromHeader_Good(t *testing.T) {
	marker, err := stateKVContainerMarkerFromHeader(validStateKVHeader(), 128)
	if err != nil {
		t.Fatalf("stateKVContainerMarkerFromHeader: %v", err)
	}
	if marker.IndexURI != "mlx://state-ramp/fold/1/folded/index" || marker.TokenCount != 206 {
		t.Fatalf("marker = %#v, want folded index + token count", marker)
	}
}

// TestStatePack_MarkerFromHeader_Bad — a header with the wrong kind is
// rejected (guards against feeding a foreign trix container to wake).
func TestStatePack_MarkerFromHeader_Bad(t *testing.T) {
	header := validStateKVHeader()
	header["kind"] = "some/other-kind"
	if _, err := stateKVContainerMarkerFromHeader(header, 128); err == nil {
		t.Fatal("expected error for wrong kind, got nil")
	}
}

// TestStatePack_MarkerFromHeader_Ugly — a recorded payload_bytes that
// disagrees with the actual on-disk size is a truncation/corruption
// signal and must error.
func TestStatePack_MarkerFromHeader_Ugly(t *testing.T) {
	if _, err := stateKVContainerMarkerFromHeader(validStateKVHeader(), 999); err == nil {
		t.Fatal("expected payload-bytes mismatch error, got nil")
	}
	// Also: a header that passes the type/size gates but omits index_uri
	// (the one field wake genuinely needs) must still be rejected.
	header := validStateKVHeader()
	delete(header, "index_uri")
	if _, err := stateKVContainerMarkerFromHeader(header, 128); err == nil {
		t.Fatal("expected missing index_uri error, got nil")
	}
}

// ---- magic detection + round-trip ------------------------------------

// TestStatePack_FileHasMagic_Good — a freshly packed .kv container is
// recognised by its KVST magic.
func TestStatePack_FileHasMagic_Good(t *testing.T) {
	dir := t.TempDir()
	payload := core.PathJoin(dir, "session.mvlog")
	if r := core.WriteFile(payload, []byte("state-binary-payload"), 0o600); !r.OK {
		t.Fatalf("write payload: %v", r.Value)
	}
	container := core.PathJoin(dir, "session.kv")
	header := map[string]any{"kind": stateKVContainerKind}
	if _, err := stateKVContainerEncode(container, header, payload); err != nil {
		t.Fatalf("encode: %v", err)
	}
	has, err := stateKVContainerFileHasMagic(container)
	if err != nil {
		t.Fatalf("hasMagic: %v", err)
	}
	if !has {
		t.Fatal("packed container not recognised by magic")
	}
}

// TestStatePack_FileHasMagic_Bad — a plain JSON marker (not a container)
// reports no magic so the source resolver falls through to the JSON path.
func TestStatePack_FileHasMagic_Bad(t *testing.T) {
	dir := t.TempDir()
	jsonMarker := core.PathJoin(dir, "marker.json")
	if r := core.WriteFile(jsonMarker, []byte(`{"index_uri":"mlx://x"}`), 0o600); !r.OK {
		t.Fatalf("write marker: %v", r.Value)
	}
	has, err := stateKVContainerFileHasMagic(jsonMarker)
	if err != nil {
		t.Fatalf("hasMagic: %v", err)
	}
	if has {
		t.Fatal("plain JSON marker wrongly reported as a KV container")
	}
}

// TestStatePack_FileHasMagic_Ugly — a file shorter than the 4-byte magic
// is not a container and must report false without an error (empty/short
// files are a legitimate not-a-container case, not a read failure).
func TestStatePack_FileHasMagic_Ugly(t *testing.T) {
	dir := t.TempDir()
	short := core.PathJoin(dir, "short")
	if r := core.WriteFile(short, []byte("KV"), 0o600); !r.OK {
		t.Fatalf("write short: %v", r.Value)
	}
	has, err := stateKVContainerFileHasMagic(short)
	if err != nil {
		t.Fatalf("hasMagic(short) error = %v, want nil", err)
	}
	if has {
		t.Fatal("2-byte file reported as a container")
	}
}

// TestStatePack_MarkerSourceFromFile_Good — the full resolver path: pack
// a marker+payload into a .kv, then read it back through the magic-detect
// → container-header branch. The returned marker's StorePath is rewritten
// to the container itself and the original store path is preserved as the
// segment alias.
func TestStatePack_MarkerSourceFromFile_Good(t *testing.T) {
	dir := t.TempDir()
	payload := []byte("go-mlx-state-log\x00binary")
	statePath := core.PathJoin(dir, "session.mvlog")
	if r := core.WriteFile(statePath, payload, 0o600); !r.OK {
		t.Fatalf("write state: %v", r.Value)
	}
	container := core.PathJoin(dir, "session.kv")
	header := stateKVContainerHeader(statePackOptions{
		MarkerFile:     core.PathJoin(dir, "marker.json"),
		StateStorePath: statePath,
		OutputPath:     container,
	}, stateRampFoldMarker{
		StorePath:  statePath,
		IndexURI:   "mlx://state-ramp/fold/1/folded/index",
		EntryURI:   "mlx://state-ramp/fold/1/folded",
		BundleURI:  "mlx://state-ramp/fold/1/folded/bundle",
		TokenCount: 206,
	}, int64(len(payload)))
	if _, err := stateKVContainerEncode(container, header, statePath); err != nil {
		t.Fatalf("encode: %v", err)
	}

	src, err := stateWakeProfileMarkerSourceFromFile(container)
	if err != nil {
		t.Fatalf("MarkerSourceFromFile: %v", err)
	}
	if src.Marker.IndexURI != "mlx://state-ramp/fold/1/folded/index" {
		t.Fatalf("index_uri = %q, want folded index", src.Marker.IndexURI)
	}
	if src.Marker.StorePath != container {
		t.Fatalf("StorePath = %q, want container path %q", src.Marker.StorePath, container)
	}
	if src.SegmentAlias != statePath {
		t.Fatalf("SegmentAlias = %q, want original store path %q", src.SegmentAlias, statePath)
	}
	if src.PayloadBytes != int64(len(payload)) {
		t.Fatalf("PayloadBytes = %d, want %d", src.PayloadBytes, len(payload))
	}
}

// TestStatePack_MarkerSourceFromFile_Bad — a plain JSON marker (no magic)
// resolves through the JSON branch to a marker with the recorded index.
func TestStatePack_MarkerSourceFromFile_Bad(t *testing.T) {
	dir := t.TempDir()
	jsonMarker := core.PathJoin(dir, "marker.json")
	if r := core.WriteFile(jsonMarker, []byte(`{"store_path":"/runs/s.mvlog","index_uri":"mlx://flat/index"}`), 0o600); !r.OK {
		t.Fatalf("write marker: %v", r.Value)
	}
	src, err := stateWakeProfileMarkerSourceFromFile(jsonMarker)
	if err != nil {
		t.Fatalf("MarkerSourceFromFile(json): %v", err)
	}
	if src.Marker.IndexURI != "mlx://flat/index" {
		t.Fatalf("index_uri = %q, want flat marker index", src.Marker.IndexURI)
	}
}

// TestStatePack_MarkerSourceFromFile_Ugly — a JSON marker that yields no
// index (neither flat nor fold) is rejected with a clear error rather
// than handing back an unusable empty marker.
func TestStatePack_MarkerSourceFromFile_Ugly(t *testing.T) {
	dir := t.TempDir()
	jsonMarker := core.PathJoin(dir, "empty-marker.json")
	if r := core.WriteFile(jsonMarker, []byte(`{"store_path":"/runs/s.mvlog"}`), 0o600); !r.OK {
		t.Fatalf("write marker: %v", r.Value)
	}
	if _, err := stateWakeProfileMarkerSourceFromFile(jsonMarker); err == nil {
		t.Fatal("expected missing-index error for marker with no index_uri, got nil")
	}
}

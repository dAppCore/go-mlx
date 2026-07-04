// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/state/agent"
)

// writeAdminToken with a parent that is a regular FILE trips MkdirAll (the
// fail-closed checkpoint per Cerberus §5.1 — serve must refuse to boot rather
// than bind an unprotected admin surface).
func TestWriteAdminToken_MkdirParentFails_Bad(t *testing.T) {
	tmp := t.TempDir()
	fileAsDir := core.PathJoin(tmp, "afile")
	if r := core.WriteFile(fileAsDir, []byte("x"), 0o644); !r.OK {
		t.Fatalf("seed file: %v", r.Value)
	}
	err := writeAdminToken(core.PathJoin(fileAsDir, "sub", "admin.token"), "lthn-mlx_x")
	if err == nil {
		t.Fatal("want an error when the token parent path is a regular file")
	}
}

// writeAdminToken with the target path being an existing DIRECTORY trips the
// WriteFile branch.
func TestWriteAdminToken_WriteFails_Bad(t *testing.T) {
	tmp := t.TempDir()
	dirAsFile := core.PathJoin(tmp, "adir")
	if r := core.MkdirAll(dirAsFile, 0o755); !r.OK {
		t.Fatalf("seed dir: %v", r.Value)
	}
	if err := writeAdminToken(dirAsFile, "lthn-mlx_x"); err == nil {
		t.Fatal("want an error when the token target path is a directory")
	}
}

// ensureAdminToken propagates the write failure: a fresh path whose parent is a
// file generates a token then fails to persist it, so the whole call errors
// (serve must abort).
func TestEnsureAdminToken_WriteFails_Bad(t *testing.T) {
	tmp := t.TempDir()
	fileAsDir := core.PathJoin(tmp, "afile")
	if r := core.WriteFile(fileAsDir, []byte("x"), 0o644); !r.OK {
		t.Fatalf("seed file: %v", r.Value)
	}
	_, generated, err := ensureAdminToken(core.PathJoin(fileAsDir, "sub", "admin.token"))
	if err == nil {
		t.Fatal("want an error when the token cannot be written")
	}
	if generated {
		t.Fatal("generated must be false on the write-failure path")
	}
}

// loadAdminToken treats a directory (read returns non-byte / fails) as
// "no token yet" rather than fatal.
func TestLoadAdminToken_DirectoryIsAbsent(t *testing.T) {
	tmp := t.TempDir()
	dir := core.PathJoin(tmp, "adir")
	if r := core.MkdirAll(dir, 0o755); !r.OK {
		t.Fatalf("seed dir: %v", r.Value)
	}
	tok, exists, err := loadAdminToken(dir)
	if err != nil {
		t.Fatalf("loadAdminToken(dir) erred: %v (must treat as absent)", err)
	}
	if exists || tok != "" {
		t.Fatalf("loadAdminToken(dir) = (%q,%v), want absent", tok, exists)
	}
}

// loadAdminToken on a whitespace-only file reports absent (the trimmed-empty
// branch).
func TestLoadAdminToken_WhitespaceFileIsAbsent(t *testing.T) {
	tmp := t.TempDir()
	path := core.PathJoin(tmp, "empty.token")
	if r := core.WriteFile(path, []byte("   \n\t"), 0o644); !r.OK {
		t.Fatalf("seed file: %v", r.Value)
	}
	tok, exists, _ := loadAdminToken(path)
	if exists || tok != "" {
		t.Fatalf("loadAdminToken(whitespace) = (%q,%v), want absent", tok, exists)
	}
}

// stateWakeProfileCompactMarkerFromFile on a missing path returns the read
// error (the read-fail branch).
func TestStateWakeMarkerFromFile_MissingPath_Bad(t *testing.T) {
	_, err := stateWakeProfileCompactMarkerFromFile(core.PathJoin(t.TempDir(), "nope.json"))
	if err == nil {
		t.Fatal("want a read error for a missing marker file")
	}
}

// stateWakeProfileCompactMarkerFromFile on malformed JSON returns the
// unmarshal error.
func TestStateWakeMarkerFromFile_BadJSON_Bad(t *testing.T) {
	path := core.PathJoin(t.TempDir(), "marker.json")
	if r := core.WriteFile(path, []byte("{not json"), 0o644); !r.OK {
		t.Fatalf("seed: %v", r.Value)
	}
	if _, err := stateWakeProfileCompactMarkerFromFile(path); err == nil {
		t.Fatal("want a JSON unmarshal error for malformed marker JSON")
	}
}

// stateWakeProfileCompactMarkerFromFile with valid JSON that yields no index
// returns the explicit "missing store_path or index_uri" error.
func TestStateWakeMarkerFromFile_NoIndex_Bad(t *testing.T) {
	path := core.PathJoin(t.TempDir(), "marker.json")
	if r := core.WriteFile(path, []byte(`{"store_path":"/tmp/s.kv"}`), 0o644); !r.OK {
		t.Fatalf("seed: %v", r.Value)
	}
	if _, err := stateWakeProfileCompactMarkerFromFile(path); err == nil {
		t.Fatal("want the missing-index error when no index_uri resolves")
	}
}

// stateWakeProfileCompactMarkerFromFile resolves a flat marker file to a
// compact marker (the success path through the file reader).
func TestStateWakeMarkerFromFile_FlatMarker_Good(t *testing.T) {
	path := core.PathJoin(t.TempDir(), "marker.json")
	if r := core.WriteFile(path, []byte(
		`{"store_path":"/tmp/s.kv","index_uri":"mlx://agent/x/index","entry_uri":"mlx://agent/x"}`), 0o644); !r.OK {
		t.Fatalf("seed: %v", r.Value)
	}
	marker, err := stateWakeProfileCompactMarkerFromFile(path)
	if err != nil {
		t.Fatalf("resolve flat marker: %v", err)
	}
	if marker.IndexURI != "mlx://agent/x/index" || marker.StorePath != "/tmp/s.kv" {
		t.Fatalf("marker = %+v, want the flat fields preserved", marker)
	}
}

// stateWakeProfileCompactMarkerFromPayload: the nested fold with an explicit
// CompactMarker wins over the folded report.
func TestStateWakeMarkerFromPayload_FoldCompactMarker(t *testing.T) {
	payload := stateWakeProfileMarkerFile{
		Fold: &stateWakeProfileMarkerFold{
			CompactMarker: &stateRampFoldMarker{StorePath: "/s", IndexURI: "mlx://agent/f/index"},
		},
	}
	marker := stateWakeProfileCompactMarkerFromPayload(payload)
	if marker.IndexURI != "mlx://agent/f/index" {
		t.Fatalf("marker = %+v, want the explicit fold CompactMarker", marker)
	}
}

// stateWakeProfileCompactMarkerFromPayload: a fold carrying only a folded
// SleepReport derives the compact marker from it (TokenCount carried through).
func TestStateWakeMarkerFromPayload_FoldedReport(t *testing.T) {
	payload := stateWakeProfileMarkerFile{
		Fold: &stateWakeProfileMarkerFold{
			StorePath: "/s",
			Folded: &agent.SleepReport{
				IndexURI:   "mlx://agent/g/index",
				EntryURI:   "mlx://agent/g",
				TokenCount: 7,
			},
		},
	}
	marker := stateWakeProfileCompactMarkerFromPayload(payload)
	if marker.IndexURI != "mlx://agent/g/index" || marker.TokenCount != 7 || marker.StorePath != "/s" {
		t.Fatalf("marker = %+v, want fields derived from the folded report", marker)
	}
}

// stateWakeProfileCompactMarkerFromPayload: a fold with neither a compact
// marker nor a usable folded report yields the zero marker.
func TestStateWakeMarkerFromPayload_EmptyFold(t *testing.T) {
	payload := stateWakeProfileMarkerFile{Fold: &stateWakeProfileMarkerFold{StorePath: "/s"}}
	marker := stateWakeProfileCompactMarkerFromPayload(payload)
	if marker.IndexURI != "" {
		t.Fatalf("marker = %+v, want the zero marker for an empty fold", marker)
	}
}

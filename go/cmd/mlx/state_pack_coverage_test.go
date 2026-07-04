// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"context"
	"strings"
	"testing"

	core "dappco.re/go"
)

// writeStateMarkerAndPayload writes a compact-marker JSON + the binary .mvlog
// payload it points at, returning (markerPath, payloadBytes). Shared by the
// state-pack human-summary + round-trip coverage cases.
func writeStateMarkerAndPayload(t *testing.T) (string, []byte) {
	t.Helper()
	dir := t.TempDir()
	statePath := core.PathJoin(dir, "session.mvlog")
	markerPath := core.PathJoin(dir, "ramp-report.json")
	payload := []byte("go-mlx-state-log\nbinary\x00tail-bytes")
	if r := core.WriteFile(statePath, payload, 0o600); !r.OK {
		t.Fatalf("write state: %v", r.Value)
	}
	if r := core.WriteFile(markerPath, []byte(`{
  "fold": {
    "compact_marker": {
      "store_path": "`+statePath+`",
      "index_uri": "mlx://state-ramp/fold/1/folded/index",
      "entry_uri": "mlx://state-ramp/fold/1/folded",
      "bundle_uri": "mlx://state-ramp/fold/1/folded/bundle",
      "token_count": 206
    }
  }
}`), 0o644); !r.OK {
		t.Fatalf("write marker: %v", r.Value)
	}
	return markerPath, payload
}

// state-pack without -json prints the human-readable "packed ... into ..."
// summary line (the non-JSON output lane that the JSON test skips).
func TestRunStatePack_HumanSummary_Good(t *testing.T) {
	markerPath, payload := writeStateMarkerAndPayload(t)
	output := core.PathJoin(t.TempDir(), "out", "session.kv")
	stdout, stderr := core.NewBuffer(), core.NewBuffer()
	code := runCommand(context.Background(), []string{
		"state-pack", "-marker-file", markerPath, "-output", output,
	}, stdout, stderr)
	if code != 0 {
		t.Fatalf("exit = %d, want 0; stderr=%q", code, stderr.String())
	}
	out := stdout.String()
	if !strings.Contains(out, "packed") || !strings.Contains(out, "into") {
		t.Fatalf("stdout = %q, want the human packed-into summary", out)
	}
	if !strings.Contains(out, core.Sprintf("%d payload bytes", len(payload))) {
		t.Fatalf("stdout = %q, want the payload byte count", out)
	}
}

// state-pack with a missing -output is a validation error (exit 2).
func TestRunStatePack_MissingOutput_Bad(t *testing.T) {
	markerPath, _ := writeStateMarkerAndPayload(t)
	stdout, stderr := core.NewBuffer(), core.NewBuffer()
	code := runCommand(context.Background(), []string{"state-pack", "-marker-file", markerPath}, stdout, stderr)
	if code != 2 {
		t.Fatalf("exit = %d, want 2 (missing output)", code)
	}
	if !strings.Contains(stderr.String(), "output path is required") {
		t.Fatalf("stderr = %q, want output validation", stderr.String())
	}
}

// state-pack with an unexpected positional argument is a usage error (exit 2).
func TestRunStatePack_UnexpectedPositional_Bad(t *testing.T) {
	stdout, stderr := core.NewBuffer(), core.NewBuffer()
	code := runCommand(context.Background(), []string{"state-pack", "stray"}, stdout, stderr)
	if code != 2 {
		t.Fatalf("exit = %d, want 2 (unexpected positional)", code)
	}
	if !strings.Contains(stderr.String(), "expected no positional arguments") {
		t.Fatalf("stderr = %q, want the positional-args notice", stderr.String())
	}
}

// state-pack with a bad flag fails to parse (exit 2).
func TestRunStatePack_BadFlag_Bad(t *testing.T) {
	stdout, stderr := core.NewBuffer(), core.NewBuffer()
	code := runCommand(context.Background(), []string{"state-pack", "-not-a-flag"}, stdout, stderr)
	if code != 2 {
		t.Fatalf("exit = %d, want 2 (bad flag)", code)
	}
}

// state-pack surfaces a runStatePack error as exit 1 (mocked seam).
func TestRunStatePack_PackError_Bad(t *testing.T) {
	orig := runStatePack
	t.Cleanup(func() { runStatePack = orig })
	runStatePack = func(_ context.Context, _ statePackOptions) (*statePackReport, error) {
		return nil, core.NewError("pack boom")
	}
	stdout, stderr := core.NewBuffer(), core.NewBuffer()
	code := runCommand(context.Background(), []string{
		"state-pack", "-marker-file", "/m.json", "-output", "/o.kv",
	}, stdout, stderr)
	if code != 1 {
		t.Fatalf("exit = %d, want 1 (pack error)", code)
	}
	if !strings.Contains(stderr.String(), "pack boom") {
		t.Fatalf("stderr = %q, want the surfaced pack error", stderr.String())
	}
}

// defaultRunStatePack fails when the marker declares no store_path and none is
// supplied explicitly (the "State store path is required" guard).
func TestDefaultRunStatePack_NoStorePath_Bad(t *testing.T) {
	dir := t.TempDir()
	markerPath := core.PathJoin(dir, "marker.json")
	// Marker with index_uri but no store_path.
	if r := core.WriteFile(markerPath, []byte(`{
  "fold": {"compact_marker": {"index_uri": "mlx://x/index", "entry_uri": "mlx://x"}}
}`), 0o644); !r.OK {
		t.Fatalf("write marker: %v", r.Value)
	}
	_, err := defaultRunStatePack(context.Background(), statePackOptions{
		MarkerFile: markerPath,
		OutputPath: core.PathJoin(dir, "o.kv"),
	})
	if err == nil {
		t.Fatal("defaultRunStatePack error = nil, want store-path-required")
	}
}

// defaultRunStatePack fails when the resolved state store path does not exist
// on disk (the stat-failure guard).
func TestDefaultRunStatePack_MissingStore_Bad(t *testing.T) {
	dir := t.TempDir()
	markerPath := core.PathJoin(dir, "marker.json")
	if r := core.WriteFile(markerPath, []byte(`{
  "fold": {"compact_marker": {"store_path": "`+core.PathJoin(dir, "absent.mvlog")+`", "index_uri": "mlx://x/index"}}
}`), 0o644); !r.OK {
		t.Fatalf("write marker: %v", r.Value)
	}
	_, err := defaultRunStatePack(context.Background(), statePackOptions{
		MarkerFile: markerPath,
		OutputPath: core.PathJoin(dir, "o.kv"),
	})
	if err == nil {
		t.Fatal("defaultRunStatePack error = nil, want a missing-store stat error")
	}
}

// stateWakeProfileMarkerSourceFromFile round-trips a packed .kv container back
// to its marker (the magic-detected container branch), and reads a plain
// marker JSON through the non-container branch.
func TestStateWakeProfileMarkerSourceFromFile_Coverage_Good(t *testing.T) {
	markerPath, _ := writeStateMarkerAndPayload(t)
	containerPath := core.PathJoin(t.TempDir(), "packed.kv")
	report, err := defaultRunStatePack(context.Background(), statePackOptions{
		MarkerFile: markerPath,
		OutputPath: containerPath,
	})
	if err != nil {
		t.Fatalf("pack: %v", err)
	}
	if report == nil {
		t.Fatal("pack returned a nil report")
	}
	// The packed container is magic-detected and its header decoded back.
	src, err := stateWakeProfileMarkerSourceFromFile(containerPath)
	if err != nil {
		t.Fatalf("marker source from container: %v", err)
	}
	if src.Marker.IndexURI != "mlx://state-ramp/fold/1/folded/index" {
		t.Fatalf("container marker index_uri = %q, want the folded index", src.Marker.IndexURI)
	}
	if src.PayloadBytes <= 0 {
		t.Fatalf("container PayloadBytes = %d, want > 0", src.PayloadBytes)
	}

	// A plain (non-container) marker JSON reads through the JSON branch.
	plain, err := stateWakeProfileMarkerSourceFromFile(markerPath)
	if err != nil {
		t.Fatalf("marker source from plain JSON: %v", err)
	}
	if plain.Marker.IndexURI == "" {
		t.Fatalf("plain marker index_uri empty, want the folded index")
	}
}

// stateWakeProfileMarkerSourceFromFile errors on a plain JSON marker that has
// neither a store_path nor an index_uri.
func TestStateWakeProfileMarkerSourceFromFile_NoIndex_Bad(t *testing.T) {
	path := core.PathJoin(t.TempDir(), "empty-marker.json")
	if r := core.WriteFile(path, []byte(`{"fold":{"compact_marker":{}}}`), 0o644); !r.OK {
		t.Fatalf("write: %v", r.Value)
	}
	if _, err := stateWakeProfileMarkerSourceFromFile(path); err == nil {
		t.Fatal("marker source error = nil, want missing store_path/index_uri")
	}
}

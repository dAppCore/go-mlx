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

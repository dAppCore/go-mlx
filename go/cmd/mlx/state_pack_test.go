// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"context"
	"testing"
	"time"

	core "dappco.re/go"
	mlx "dappco.re/go/mlx"
	"dappco.re/go/mlx/agent"
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

func TestRunCommand_StateWakeProfileMarkerFileKV_Good(t *testing.T) {
	originalRun := runStateWakeProfile
	t.Cleanup(func() { runStateWakeProfile = originalRun })
	var gotCfg stateWakeProfileOptions
	var embeddedPayload string
	runStateWakeProfile = func(_ context.Context, modelPath string, _ []mlx.LoadOption, cfg stateWakeProfileOptions) (*stateWakeProfileReport, error) {
		gotCfg = cfg
		read := core.ReadFile(cfg.StateStorePath)
		if !read.OK {
			t.Fatalf("read state container: %v", read.Value)
		}
		container := read.Value.([]byte)
		start := cfg.StateStorePayloadOffset
		end := start + cfg.StateStorePayloadBytes
		if start < 0 || end < start || end > int64(len(container)) {
			t.Fatalf("state payload window = [%d:%d], container bytes=%d", start, end, len(container))
		}
		embeddedPayload = string(container[int(start):int(end)])
		return &stateWakeProfileReport{
			Version:                 1,
			ModelPath:               modelPath,
			StateStorePath:          cfg.StateStorePath,
			StateStoreAlias:         cfg.StateStoreSegmentAlias,
			StateStorePayloadOffset: cfg.StateStorePayloadOffset,
			StateStorePayloadBytes:  cfg.StateStorePayloadBytes,
			IndexURI:                cfg.IndexURI,
			MaxTokens:               cfg.MaxTokens,
			Wake: &agent.WakeReport{
				IndexURI:        cfg.IndexURI,
				PrefixTokens:    206,
				RestoreStrategy: "folded-prefill",
			},
			Turn: &stateRampProfileTurn{
				VisibleTokens: 4,
				Metrics: mlx.Metrics{
					GeneratedTokens:    4,
					DecodeDuration:     time.Second,
					DecodeTokensPerSec: 4,
				},
			},
		}, nil
	}
	dir := t.TempDir()
	statePath := core.PathJoin(dir, "session.mvlog")
	markerPath := core.PathJoin(dir, "ramp-report.json")
	outputPath := core.PathJoin(dir, "session.kv")
	payload := []byte("state-log payload for direct kv wake")
	if result := core.WriteFile(statePath, payload, 0o600); !result.OK {
		t.Fatalf("write state: %v", result.Value)
	}
	writeCLIPackFile(t, markerPath, `{
  "fold": {
    "compact_marker": {
      "store_path": "`+statePath+`",
      "index_uri": "mlx://state-ramp/fold/kv/folded/index",
      "entry_uri": "mlx://state-ramp/fold/kv/folded",
      "bundle_uri": "mlx://state-ramp/fold/kv/folded/bundle",
      "token_count": 206
    }
  }
}`)
	if _, err := defaultRunStatePack(context.Background(), statePackOptions{
		MarkerFile: markerPath,
		OutputPath: outputPath,
	}); err != nil {
		t.Fatalf("pack state kv: %v", err)
	}
	if result := core.Remove(statePath); !result.OK {
		t.Fatalf("remove original state: %v", result.Value)
	}
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{
		"state-wake-profile",
		"-json",
		"-marker-file", outputPath,
		"-max-tokens", "64",
		"/models/demo",
	}, stdout, stderr)

	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr=%q stdout=%q", code, stderr.String(), stdout.String())
	}
	if gotCfg.IndexURI != "mlx://state-ramp/fold/kv/folded/index" {
		t.Fatalf("index URI = %q, want KV header marker", gotCfg.IndexURI)
	}
	if gotCfg.StateStorePath != outputPath {
		t.Fatalf("state store path = %q, want KV container path %q", gotCfg.StateStorePath, outputPath)
	}
	if gotCfg.StateStoreSegmentAlias != statePath {
		t.Fatalf("segment alias = %q, want original segment path %q", gotCfg.StateStoreSegmentAlias, statePath)
	}
	if gotCfg.StateStorePayloadOffset <= 0 {
		t.Fatalf("state payload offset = %d, want container payload offset", gotCfg.StateStorePayloadOffset)
	}
	if gotCfg.StateStorePayloadBytes != int64(len(payload)) {
		t.Fatalf("state payload bytes = %d, want %d", gotCfg.StateStorePayloadBytes, len(payload))
	}
	if embeddedPayload != string(payload) {
		t.Fatalf("embedded payload = %q, want original payload", embeddedPayload)
	}
	if stat := core.Stat(statePath); stat.OK {
		t.Fatalf("original state path was recreated instead of using alias: %q", statePath)
	}
	if !core.Contains(stdout.String(), `"index_uri": "mlx://state-ramp/fold/kv/folded/index"`) {
		t.Fatalf("stdout = %q, want folded index", stdout.String())
	}
	if !core.Contains(stdout.String(), `"state_store_payload_bytes": `) {
		t.Fatalf("stdout = %q, want payload window fields", stdout.String())
	}
}

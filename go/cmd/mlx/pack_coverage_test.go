// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"context"
	"strings"
	"testing"

	core "dappco.re/go"
)

// writeValidCLIPack writes a minimal valid qwen3 model pack (config + tokenizer
// + safetensors stub) into a fresh temp dir and returns it.
func writeValidCLIPack(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	writeCLIPackFile(t, core.PathJoin(dir, "config.json"), `{
		"model_type": "qwen3",
		"max_position_embeddings": 32768,
		"quantization_config": {"bits": 4, "group_size": 64}
	}`)
	writeCLIPackFile(t, core.PathJoin(dir, "tokenizer.json"), cliTokenizerJSON)
	writeCLIPackFile(t, core.PathJoin(dir, "model.safetensors"), "stub")
	return dir
}

// pack (human-readable, valid) prints the "valid model pack: ..." summary line.
func TestRunPack_HumanValid_Good(t *testing.T) {
	dir := writeValidCLIPack(t)
	stdout, stderr := core.NewBuffer(), core.NewBuffer()
	code := runCommand(context.Background(), []string{"pack", dir}, stdout, stderr)
	if code != 0 {
		t.Fatalf("exit = %d, want 0; stderr=%q", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), "valid model pack:") || !strings.Contains(stdout.String(), "qwen3") {
		t.Fatalf("stdout = %q, want the valid-pack summary", stdout.String())
	}
}

// pack -json over an INVALID pack prints the JSON report and exits 1 (the
// JSON-invalid lane the existing JSON-good test skips).
func TestRunPack_JSONInvalid_Bad(t *testing.T) {
	dir := t.TempDir()
	writeCLIPackFile(t, core.PathJoin(dir, "config.json"), `{"model_type":"unknown"}`)
	writeCLIPackFile(t, core.PathJoin(dir, "model.safetensors"), "stub")
	stdout, stderr := core.NewBuffer(), core.NewBuffer()
	code := runCommand(context.Background(), []string{"pack", "-json", dir}, stdout, stderr)
	if code != 1 {
		t.Fatalf("exit = %d, want 1 (invalid pack)", code)
	}
	if !strings.Contains(stdout.String(), `"valid":false`) {
		t.Fatalf("stdout = %q, want the JSON report with valid:false", stdout.String())
	}
}

// pack -quantization with a mismatched required bit count makes the pack invalid
// (the quantization option is applied, then the human issues print).
func TestRunPack_QuantizationMismatch_Bad(t *testing.T) {
	dir := writeValidCLIPack(t) // q4 pack
	stdout, stderr := core.NewBuffer(), core.NewBuffer()
	code := runCommand(context.Background(), []string{"pack", "-quantization", "8", "-max-context", "65536", dir}, stdout, stderr)
	if code != 1 {
		t.Fatalf("exit = %d, want 1 (quant mismatch)", code)
	}
	if !strings.Contains(stderr.String(), "invalid model pack") {
		t.Fatalf("stderr = %q, want the invalid-pack issues", stderr.String())
	}
}

// pack with a max-context below the model's context length fails the cap (the
// max-context option is applied).
func TestRunPack_MaxContextExceeded_Bad(t *testing.T) {
	dir := writeValidCLIPack(t) // context 32768
	stdout, stderr := core.NewBuffer(), core.NewBuffer()
	code := runCommand(context.Background(), []string{"pack", "-max-context", "1024", dir}, stdout, stderr)
	if code != 1 {
		t.Fatalf("exit = %d, want 1 (max-context exceeded)", code)
	}
}

// pack against a path with no config.json reaches the model.Inspect error path
// (exit 1).
func TestRunPack_InspectError_Bad(t *testing.T) {
	stdout, stderr := core.NewBuffer(), core.NewBuffer()
	code := runCommand(context.Background(), []string{"pack", core.JoinPath(t.TempDir(), "absent")}, stdout, stderr)
	if code != 1 {
		t.Fatalf("exit = %d, want 1 (inspect error); stderr=%q", code, stderr.String())
	}
	if !strings.Contains(stderr.String(), "pack:") {
		t.Fatalf("stderr = %q, want the pack error", stderr.String())
	}
}

// pack with no model path is a usage error (exit 2).
func TestRunPack_NoModelArg_Bad(t *testing.T) {
	stdout, stderr := core.NewBuffer(), core.NewBuffer()
	code := runCommand(context.Background(), []string{"pack"}, stdout, stderr)
	if code != 2 {
		t.Fatalf("exit = %d, want 2 (no model path)", code)
	}
	if !strings.Contains(stderr.String(), "exactly one model path") {
		t.Fatalf("stderr = %q, want the one-path notice", stderr.String())
	}
}

// pack with a bad flag fails to parse (exit 2).
func TestRunPack_BadFlag_Bad(t *testing.T) {
	stdout, stderr := core.NewBuffer(), core.NewBuffer()
	code := runCommand(context.Background(), []string{"pack", "-not-a-flag", "x"}, stdout, stderr)
	if code != 2 {
		t.Fatalf("exit = %d, want 2 (bad flag)", code)
	}
}

// pack -h prints usage and exits 0.
func TestRunPack_Help_Good(t *testing.T) {
	stdout, stderr := core.NewBuffer(), core.NewBuffer()
	code := runCommand(context.Background(), []string{"pack", "-h"}, stdout, stderr)
	if code != 0 {
		t.Fatalf("exit = %d, want 0 for -h", code)
	}
	if !strings.Contains(stderr.String(), "Validate a model pack") {
		t.Fatalf("stderr = %q, want pack help body", stderr.String())
	}
}

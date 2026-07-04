// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"context"
	"strings"
	"testing"

	core "dappco.re/go"
)

// sft with a bad flag fails to parse (exit 2).
func TestRunSFT_BadFlag_Bad(t *testing.T) {
	stdout, stderr := core.NewBuffer(), core.NewBuffer()
	code := runCommand(context.Background(), []string{"sft", "-not-a-flag"}, stdout, stderr)
	if code != 2 {
		t.Fatalf("exit = %d, want 2 (bad flag)", code)
	}
}

// sft with malformed training JSONL reaches the dataset-parse error (exit 1).
func TestRunSFT_DataParseError_Bad(t *testing.T) {
	dir := t.TempDir()
	data := core.JoinPath(dir, "train.jsonl")
	if r := core.WriteFile(data, []byte("{not valid json\n"), 0o644); !r.OK {
		t.Fatal(r.Value)
	}
	stdout, stderr := core.NewBuffer(), core.NewBuffer()
	code := runCommand(context.Background(), []string{"sft", "--model", dir, "--data", data}, stdout, stderr)
	if code != 1 {
		t.Fatalf("exit = %d, want 1 (data parse error); stderr=%q", code, stderr.String())
	}
	if !strings.Contains(stderr.String(), "training data parse") {
		t.Fatalf("stderr = %q, want the data-parse notice", stderr.String())
	}
}

// sft with a --valid path that doesn't exist reaches the validation-unreadable
// error (exit 1) — after the training data parses but before the model loads.
func TestRunSFT_ValidUnreadable_Bad(t *testing.T) {
	dir := t.TempDir()
	data := core.JoinPath(dir, "train.jsonl")
	if r := core.WriteFile(data, []byte(`{"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"yo"}]}`+"\n"), 0o644); !r.OK {
		t.Fatal(r.Value)
	}
	stdout, stderr := core.NewBuffer(), core.NewBuffer()
	code := runCommand(context.Background(), []string{
		"sft", "--model", dir, "--data", data, "--valid", core.JoinPath(dir, "absent-valid.jsonl"),
		"--eval-prompts", writeOneLineFile(t, dir, "probes.txt", "what is 2+2?"),
	}, stdout, stderr)
	if code != 1 {
		t.Fatalf("exit = %d, want 1 (valid unreadable); stderr=%q", code, stderr.String())
	}
	if !strings.Contains(stderr.String(), "validation data unreadable") {
		t.Fatalf("stderr = %q, want the validation-unreadable notice", stderr.String())
	}
}

// sft with a malformed --valid JSONL reaches the validation-parse error
// (exit 1). --eval-prompts supplies probes so the --valid derivation is skipped
// and the failure is specifically the val-loss dataset parse.
func TestRunSFT_ValidParseError_Bad(t *testing.T) {
	dir := t.TempDir()
	data := core.JoinPath(dir, "train.jsonl")
	if r := core.WriteFile(data, []byte(`{"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"yo"}]}`+"\n"), 0o644); !r.OK {
		t.Fatal(r.Value)
	}
	validBad := core.JoinPath(dir, "valid.jsonl")
	if r := core.WriteFile(validBad, []byte("{not json\n"), 0o644); !r.OK {
		t.Fatal(r.Value)
	}
	stdout, stderr := core.NewBuffer(), core.NewBuffer()
	code := runCommand(context.Background(), []string{
		"sft", "--model", dir, "--data", data, "--valid", validBad,
		"--eval-prompts", writeOneLineFile(t, dir, "probes.txt", "probe one"),
	}, stdout, stderr)
	if code != 1 {
		t.Fatalf("exit = %d, want 1 (valid parse error); stderr=%q", code, stderr.String())
	}
	if !strings.Contains(stderr.String(), "validation data parse") {
		t.Fatalf("stderr = %q, want the validation-parse notice", stderr.String())
	}
}

// sft deriving probes from a malformed --valid (no --eval-prompts) reaches the
// probe-derivation error (exit 1).
func TestRunSFT_ProbeDerivationError_Bad(t *testing.T) {
	dir := t.TempDir()
	data := core.JoinPath(dir, "train.jsonl")
	if r := core.WriteFile(data, []byte(`{"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"yo"}]}`+"\n"), 0o644); !r.OK {
		t.Fatal(r.Value)
	}
	stdout, stderr := core.NewBuffer(), core.NewBuffer()
	code := runCommand(context.Background(), []string{
		"sft", "--model", dir, "--data", data, "--valid", core.JoinPath(dir, "absent-valid.jsonl"),
	}, stdout, stderr)
	if code != 1 {
		t.Fatalf("exit = %d, want 1 (probe derivation error); stderr=%q", code, stderr.String())
	}
	if !strings.Contains(stderr.String(), "deriving probes from --valid failed") {
		t.Fatalf("stderr = %q, want the probe-derivation notice", stderr.String())
	}
}

func writeOneLineFile(t *testing.T, dir, name, line string) string {
	t.Helper()
	p := core.JoinPath(dir, name)
	if r := core.WriteFile(p, []byte(line+"\n"), 0o644); !r.OK {
		t.Fatalf("write %s: %v", name, r.Value)
	}
	return p
}

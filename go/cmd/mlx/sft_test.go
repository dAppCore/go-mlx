// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"context"
	"testing"

	core "dappco.re/go"
	coreio "dappco.re/go/io"
)

// Probe derivation from a validation set shaped exactly like the LEM
// gold-full rows: the first user turn of each line, fixed count, blanks
// and malformed lines skipped.
func TestSFT_ProbesFromValid_Good(t *testing.T) {
	path := core.JoinPath(t.TempDir(), "valid.jsonl")
	rows := `{"messages": [{"role": "user", "content": "design an offline-first social graph"}, {"role": "assistant", "content": "..."}]}

not json
{"messages": [{"role": "system", "content": "be kind"}, {"role": "user", "content": "what survives device seizure?"}, {"role": "assistant", "content": "..."}]}
{"messages": [{"role": "user", "content": "third probe"}, {"role": "assistant", "content": "..."}]}
`
	if err := coreio.Local.Write(path, rows); err != nil {
		t.Fatalf("write fixture: %v", err)
	}
	prompts, err := sftProbesFromValid(path, 2)
	if err != nil {
		t.Fatalf("sftProbesFromValid: %v", err)
	}
	if len(prompts) != 2 {
		t.Fatalf("prompts = %d, want 2 (fixed probe count)", len(prompts))
	}
	if prompts[0] != "design an offline-first social graph" {
		t.Fatalf("prompt[0] = %q, want the first user turn", prompts[0])
	}
	if prompts[1] != "what survives device seizure?" {
		t.Fatalf("prompt[1] = %q, want the user turn past the system message", prompts[1])
	}
}

func TestSFT_ProbesFromValid_Bad(t *testing.T) {
	path := core.JoinPath(t.TempDir(), "valid.jsonl")
	if err := coreio.Local.Write(path, `{"messages": [{"role": "assistant", "content": "no user turns here"}]}`); err != nil {
		t.Fatalf("write fixture: %v", err)
	}
	if _, err := sftProbesFromValid(path, 4); err == nil {
		t.Fatal("a set with no user turns must refuse probe derivation")
	}
	if _, err := sftProbesFromValid(core.JoinPath(t.TempDir(), "missing.jsonl"), 4); err == nil {
		t.Fatal("a missing file must error")
	}
}

// sft with no --model/--data is a usage error (exit 2) before any load.
func TestRunSFT_MissingRequiredFlags_Bad(t *testing.T) {
	stdout, stderr := core.NewBuffer(), core.NewBuffer()
	code := runCommand(context.Background(), []string{"sft"}, stdout, stderr)
	if code != 2 {
		t.Fatalf("exit = %d, want 2 (missing --model/--data)", code)
	}
	if !core.Contains(stderr.String(), "Usage:") {
		t.Fatalf("stderr = %q, want usage", stderr.String())
	}
}

// sft with an unreadable --eval-prompts file errors (exit 1) before the model
// loads — the explicit-eval-prompts branch.
func TestRunSFT_UnreadableEvalPrompts_Bad(t *testing.T) {
	dir := t.TempDir()
	data := core.JoinPath(dir, "train.jsonl")
	if r := core.WriteFile(data, []byte(`{"messages":[{"role":"user","content":"hi"}]}`+"\n"), 0o644); !r.OK {
		t.Fatal(r.Value)
	}
	stdout, stderr := core.NewBuffer(), core.NewBuffer()
	code := runCommand(context.Background(), []string{
		"sft", "--model", dir, "--data", data, "--eval-prompts", core.JoinPath(dir, "absent-prompts.txt"),
	}, stdout, stderr)
	if code != 1 {
		t.Fatalf("exit = %d, want 1 (unreadable eval prompts)", code)
	}
	if !core.Contains(stderr.String(), "eval prompts unreadable") {
		t.Fatalf("stderr = %q, want the eval-prompts-unreadable notice", stderr.String())
	}
}

// sft with an unreadable --data path errors (exit 1) before the model loads.
func TestRunSFT_UnreadableData_Bad(t *testing.T) {
	stdout, stderr := core.NewBuffer(), core.NewBuffer()
	code := runCommand(context.Background(), []string{
		"sft", "--model", t.TempDir(), "--data", core.JoinPath(t.TempDir(), "missing.jsonl"),
	}, stdout, stderr)
	if code != 1 {
		t.Fatalf("exit = %d, want 1 (unreadable data)", code)
	}
	if !core.Contains(stderr.String(), "training data unreadable") {
		t.Fatalf("stderr = %q, want the unreadable-data notice", stderr.String())
	}
}

// sft with valid data but a bad model path reaches the model-load error path
// (exit 1) — confirms the runner advances past dataset parse into the real
// load.
func TestRunSFT_BadModelPath_Bad(t *testing.T) {
	dir := t.TempDir()
	data := core.JoinPath(dir, "train.jsonl")
	if r := core.WriteFile(data, []byte(`{"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"yo"}]}`+"\n"), 0o644); !r.OK {
		t.Fatal(r.Value)
	}
	stdout, stderr := core.NewBuffer(), core.NewBuffer()
	code := runCommand(context.Background(), []string{
		"sft", "--model", core.JoinPath(dir, "nope"), "--data", data, "--metrics-lp", "off",
	}, stdout, stderr)
	if code != 1 {
		t.Fatalf("exit = %d, want 1 (model load failure)", code)
	}
	if !core.Contains(stderr.String(), "model load") {
		t.Fatalf("stderr = %q, want a model-load error", stderr.String())
	}
}

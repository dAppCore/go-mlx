// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package main

import (
	"context"
	"testing"

	core "dappco.re/go"
)

// ssd with no --model/--data is a usage error (exit 2) before any load.
func TestRunSSD_MissingRequiredFlags_Bad(t *testing.T) {
	stdout, stderr := core.NewBuffer(), core.NewBuffer()
	code := runCommand(context.Background(), []string{"ssd"}, stdout, stderr)
	if code != 2 {
		t.Fatalf("exit = %d, want 2 (missing --model/--data)", code)
	}
	if !core.Contains(stderr.String(), "Usage:") {
		t.Fatalf("stderr = %q, want usage", stderr.String())
	}
}

// ssd with an unreadable --data path errors (exit 1) before the model loads.
func TestRunSSD_UnreadableData_Bad(t *testing.T) {
	stdout, stderr := core.NewBuffer(), core.NewBuffer()
	code := runCommand(context.Background(), []string{
		"ssd", "--model", t.TempDir(), "--data", core.JoinPath(t.TempDir(), "missing.jsonl"),
	}, stdout, stderr)
	if code != 1 {
		t.Fatalf("exit = %d, want 1 (unreadable data)", code)
	}
	if !core.Contains(stderr.String(), "prompt data unreadable") {
		t.Fatalf("stderr = %q, want the unreadable-data notice", stderr.String())
	}
}

// ssd with a readable prompt set but an unreadable --kernel path errors before
// load — the kernel-read branch.
func TestRunSSD_UnreadableKernel_Bad(t *testing.T) {
	dir := t.TempDir()
	data := core.JoinPath(dir, "prompts.jsonl")
	if r := core.WriteFile(data, []byte(`{"prompt":"hello"}`+"\n"), 0o644); !r.OK {
		t.Fatal(r.Value)
	}
	stdout, stderr := core.NewBuffer(), core.NewBuffer()
	code := runCommand(context.Background(), []string{
		"ssd", "--model", dir, "--data", data, "--kernel", core.JoinPath(dir, "absent-kernel.txt"),
	}, stdout, stderr)
	if code != 1 {
		t.Fatalf("exit = %d, want 1 (unreadable kernel)", code)
	}
	if !core.Contains(stderr.String(), "kernel unreadable") {
		t.Fatalf("stderr = %q, want the kernel-unreadable notice", stderr.String())
	}
}

// ssd with valid data + kernel but a bad model path reaches the model-load
// error path (exit 1) — confirms the runner advances past data/kernel parsing
// into the real load.
func TestRunSSD_BadModelPath_Bad(t *testing.T) {
	dir := t.TempDir()
	data := core.JoinPath(dir, "prompts.jsonl")
	if r := core.WriteFile(data, []byte(`{"prompt":"hello"}`+"\n"), 0o644); !r.OK {
		t.Fatal(r.Value)
	}
	stdout, stderr := core.NewBuffer(), core.NewBuffer()
	code := runCommand(context.Background(), []string{
		"ssd", "--model", core.JoinPath(dir, "nope"), "--data", data,
	}, stdout, stderr)
	if code != 1 {
		t.Fatalf("exit = %d, want 1 (model load failure)", code)
	}
	if !core.Contains(stderr.String(), "model load") {
		t.Fatalf("stderr = %q, want a model-load error", stderr.String())
	}
}

// ssd over a TINY synthetic gemma3 runs the real frozen-base self-distillation
// sampling lane end-to-end and prints the self-sample summary. Non-unit temp +
// a couple of short prompts; capped at a handful of tokens.
func TestRunSSD_SyntheticSampling_Good(t *testing.T) {
	requireSyntheticRuntime(t)
	model := writeSyntheticGemma3Model(t, t.TempDir())
	dir := t.TempDir()
	data := core.JoinPath(dir, "prompts.jsonl")
	if r := core.WriteFile(data, []byte(
		`{"messages":[{"role":"user","content":"hello"}]}`+"\n"+
			`{"messages":[{"role":"user","content":"world"}]}`+"\n"), 0o644); !r.OK {
		t.Fatal(r.Value)
	}
	stdout, stderr := core.NewBuffer(), core.NewBuffer()
	code := runCommand(context.Background(), []string{
		"ssd", "--model", model, "--data", data,
		"--sample-max-tokens", "6", "--sample-temp", "0.7",
		"--checkpoint-dir", core.JoinPath(dir, "out"), "--metrics-lp", "off",
	}, stdout, stderr)
	if code != 0 {
		t.Fatalf("exit = %d, want 0; stderr=%q", code, stderr.String())
	}
	if !core.Contains(stdout.String(), "self-samples") {
		t.Fatalf("stdout = %q, want the self-samples summary", stdout.String())
	}
}

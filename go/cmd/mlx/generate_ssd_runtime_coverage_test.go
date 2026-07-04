// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package main

import (
	"context"
	"strings"
	"testing"

	core "dappco.re/go"
)

// generate -state WITHOUT -state-store falls back to the ~/Lethean/data/state
// default store path (HOME-derived) — the default-store-resolution branch of
// runGenerateState. HOME is redirected so the real store is never touched.
func TestRunGenerateState_DefaultStorePath_Good(t *testing.T) {
	requireSyntheticRuntime(t)
	home := t.TempDir()
	t.Setenv("HOME", home)
	model := writeSyntheticGemma3Model(t, t.TempDir())
	stdout, stderr := core.NewBuffer(), core.NewBuffer()
	code := runCommand(context.Background(), []string{
		"generate", "-state", "chatX", "-max-tokens", "5", "-temp", "0", "-prompt", "hello", model,
	}, stdout, stderr)
	if code != 0 {
		t.Fatalf("exit = %d, want 0; stderr=%q", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), "state: chatX") {
		t.Fatalf("stdout = %q, want the state turn report", stdout.String())
	}
	// The default store landed under ~/Lethean/data/state/agent.kv.
	if !core.Stat(core.PathJoin(home, "Lethean", "data", "state", "agent.kv")).OK {
		t.Fatalf("default state store was not created under HOME")
	}
}

// ssd over a TINY synthetic WITH a --kernel file (read verbatim) and a default
// metrics-lp path (under --checkpoint-dir) drives the kernel-read branch + the
// metrics-sink construction branch that the -metrics-lp=off test skips.
func TestRunSSD_KernelAndMetrics_Good(t *testing.T) {
	requireSyntheticRuntime(t)
	model := writeSyntheticGemma3Model(t, t.TempDir())
	dir := t.TempDir()
	data := core.JoinPath(dir, "prompts.jsonl")
	if r := core.WriteFile(data, []byte(`{"messages":[{"role":"user","content":"hello"}]}`+"\n"), 0o644); !r.OK {
		t.Fatalf("write data: %v", r.Value)
	}
	kernel := core.JoinPath(dir, "kernel.txt")
	if r := core.WriteFile(kernel, []byte("chadi frequency kernel prefix"), 0o644); !r.OK {
		t.Fatalf("write kernel: %v", r.Value)
	}
	checkpointDir := core.JoinPath(dir, "out")
	stdout, stderr := core.NewBuffer(), core.NewBuffer()
	code := runCommand(context.Background(), []string{
		"ssd", "--model", model, "--data", data,
		"--kernel", kernel, "--sample-max-tokens", "6", "--sample-temp", "0.7",
		"--checkpoint-dir", checkpointDir, // default metrics-lp lands here (not "off")
	}, stdout, stderr)
	if code != 0 {
		t.Fatalf("exit = %d, want 0; stderr=%q", code, stderr.String())
	}
	out := stdout.String()
	if !strings.Contains(out, "self-samples") {
		t.Fatalf("stdout = %q, want the self-samples summary", out)
	}
	// The metrics line is printed when a sink is active (lp path under the dir).
	if !strings.Contains(out, "metrics") {
		t.Fatalf("stdout = %q, want the metrics summary line", out)
	}
}

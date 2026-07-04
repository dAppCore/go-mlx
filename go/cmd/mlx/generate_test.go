// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package main

import (
	"context"
	"testing"

	core "dappco.re/go"
	mlx "dappco.re/go/mlx"
)

// resolvedDraftBlock: a positive flag wins; 0 resolves to the engine default.
func TestResolvedDraftBlock_Good(t *testing.T) {
	if got := resolvedDraftBlock(7); got != 7 {
		t.Fatalf("resolvedDraftBlock(7) = %d, want 7 (explicit flag wins)", got)
	}
	if got := resolvedDraftBlock(0); got != mlx.MTPDefaultDraftBlock {
		t.Fatalf("resolvedDraftBlock(0) = %d, want engine default %d", got, mlx.MTPDefaultDraftBlock)
	}
	if got := resolvedDraftBlock(-3); got != mlx.MTPDefaultDraftBlock {
		t.Fatalf("resolvedDraftBlock(-3) = %d, want engine default (non-positive → default)", got)
	}
}

// printGenerateMTPMetrics over a model that exposes no MTP metrics writes
// nothing — the autoregressive (non-speculative) path stays silent.
func TestPrintGenerateMTPMetrics_NoMTPIsSilent_Good(t *testing.T) {
	stdout := core.NewBuffer()
	printGenerateMTPMetrics(stdout, nil)
	if stdout.String() != "" {
		t.Fatalf("printGenerateMTPMetrics(nil) wrote %q, want empty (no MTP provider → no line)", stdout.String())
	}
}

// generate with no model-path argument is a usage error (exit 2) before any
// load is attempted.
func TestRunGenerate_NoModelArg_Bad(t *testing.T) {
	stdout, stderr := core.NewBuffer(), core.NewBuffer()
	code := runCommand(context.Background(), []string{"generate"}, stdout, stderr)
	if code != 2 {
		t.Fatalf("exit = %d, want 2 (missing model path)", code)
	}
	if !core.Contains(stderr.String(), "exactly one model path") {
		t.Fatalf("stderr = %q, want the one-model-path usage notice", stderr.String())
	}
}

// generate with two positional args is also a usage error.
func TestRunGenerate_TwoModelArgs_Bad(t *testing.T) {
	stdout, stderr := core.NewBuffer(), core.NewBuffer()
	code := runCommand(context.Background(), []string{"generate", "/a", "/b"}, stdout, stderr)
	if code != 2 {
		t.Fatalf("exit = %d, want 2 (two model paths)", code)
	}
}

// generate -h prints usage and exits 0 (flag.ErrHelp path).
func TestRunGenerate_Help_Good(t *testing.T) {
	stdout, stderr := core.NewBuffer(), core.NewBuffer()
	code := runCommand(context.Background(), []string{"generate", "-h"}, stdout, stderr)
	if code != 0 {
		t.Fatalf("exit = %d, want 0 for -h", code)
	}
	if !core.Contains(stderr.String(), "Usage:") || !core.Contains(stderr.String(), "generate") {
		t.Fatalf("stderr = %q, want generate usage", stderr.String())
	}
}

// generate against a nonexistent model path fails to load (exit 1) — the load
// error path, no synthetic needed.
func TestRunGenerate_BadModelPath_Bad(t *testing.T) {
	stdout, stderr := core.NewBuffer(), core.NewBuffer()
	code := runCommand(context.Background(), []string{"generate", "-max-tokens", "4", core.JoinPath(t.TempDir(), "nope")}, stdout, stderr)
	if code != 1 {
		t.Fatalf("exit = %d, want 1 (load failure)", code)
	}
	if !core.Contains(stderr.String(), "load:") {
		t.Fatalf("stderr = %q, want a load error", stderr.String())
	}
}

// generate over a TINY synthetic gemma3 runs the real load + warm + decode
// path end-to-end and prints the decode-rate line. -draft ” keeps it
// autoregressive (no drafter beside the synthetic).
func TestRunGenerate_SyntheticModel_Good(t *testing.T) {
	requireSyntheticRuntime(t)
	model := writeSyntheticGemma3Model(t, t.TempDir())
	stdout, stderr := core.NewBuffer(), core.NewBuffer()
	code := runCommand(context.Background(), []string{
		"generate", "-max-tokens", "6", "-temp", "0", "-draft", "", "-prompt", "hello", model,
	}, stdout, stderr)
	if code != 0 {
		t.Fatalf("exit = %d, want 0; stderr=%q", code, stderr.String())
	}
	if !core.Contains(stdout.String(), "tok/s") {
		t.Fatalf("stdout = %q, want the decode-rate report", stdout.String())
	}
}

// runGenerateTrace path (-trace) loads once and prints the per-token phase
// budget for the greedy + sampled lanes.
func TestRunGenerate_SyntheticTrace_Good(t *testing.T) {
	requireSyntheticRuntime(t)
	model := writeSyntheticGemma3Model(t, t.TempDir())
	stdout, stderr := core.NewBuffer(), core.NewBuffer()
	code := runCommand(context.Background(), []string{
		"generate", "-trace", "-max-tokens", "5", "-draft", "", "-prompt", "hello", model,
	}, stdout, stderr)
	if code != 0 {
		t.Fatalf("exit = %d, want 0; stderr=%q", code, stderr.String())
	}
	// Either a phase budget or the "no warm token phases" notice — both are
	// the trace lane having run; what matters is the runner reached it.
	out := stdout.String()
	if !core.Contains(out, "greedy") && !core.Contains(out, "token") {
		t.Fatalf("trace stdout = %q, want greedy/sampled phase output", out)
	}
}

// runGenerateState path (-state): a fresh state name prefills, generates, and
// sleeps to the store, printing the turn + state lines. The store dir is
// created on demand. (The wake-from-store half is exercised by the state
// system's own package tests — a degenerate 1-layer synthetic can't round-trip
// a KV snapshot through wake, which is an engine constraint, not a runner one.)
func TestRunGenerate_SyntheticState_Good(t *testing.T) {
	requireSyntheticRuntime(t)
	model := writeSyntheticGemma3Model(t, t.TempDir())
	store := core.JoinPath(t.TempDir(), "made", "on", "demand", "agent.kv")
	stdout, stderr := core.NewBuffer(), core.NewBuffer()
	code := runCommand(context.Background(), []string{
		"generate", "-state", "chat1", "-state-store", store, "-max-tokens", "5", "-temp", "0", "-prompt", "hello", model,
	}, stdout, stderr)
	if code != 0 {
		t.Fatalf("fresh-state turn exit = %d, want 0; stderr=%q", code, stderr.String())
	}
	out := stdout.String()
	if !core.Contains(out, "state:") || !core.Contains(out, "fresh state") {
		t.Fatalf("stdout = %q, want the fresh-state turn + state report lines", out)
	}
	if !core.Contains(out, "slept") {
		t.Fatalf("stdout = %q, want the slept-to-store line", out)
	}
	if !core.Stat(store).OK {
		t.Fatalf("state store %s was not created on demand", store)
	}
}

// runGenerateState path (-state -raw): the low-level token-loop instrument —
// today's original completion-style turn, unchanged, over a real (synthetic)
// load. Same structural assertions as the chat-framed default above; the
// byte-exact "-raw skips the template" proof is the hermetic
// TestRunGenerateStateSession_Raw_BypassesChatTemplate_Good.
func TestRunGenerate_SyntheticStateRaw_Good(t *testing.T) {
	requireSyntheticRuntime(t)
	model := writeSyntheticGemma3Model(t, t.TempDir())
	store := core.JoinPath(t.TempDir(), "made", "on", "demand", "agent.kv")
	stdout, stderr := core.NewBuffer(), core.NewBuffer()
	code := runCommand(context.Background(), []string{
		"generate", "-state", "chat1", "-raw", "-state-store", store, "-max-tokens", "5", "-temp", "0", "-prompt", "hello", model,
	}, stdout, stderr)
	if code != 0 {
		t.Fatalf("fresh-state raw turn exit = %d, want 0; stderr=%q", code, stderr.String())
	}
	out := stdout.String()
	if !core.Contains(out, "state:") || !core.Contains(out, "fresh state") {
		t.Fatalf("stdout = %q, want the fresh-state turn + state report lines", out)
	}
	if !core.Contains(out, "slept") {
		t.Fatalf("stdout = %q, want the slept-to-store line", out)
	}
}

// openOrCreateStateStore creates the store (and parent dir) on first use, then
// opens the existing file on the second call.
func TestOpenOrCreateStateStore_CreatesThenOpens_Good(t *testing.T) {
	path := core.JoinPath(t.TempDir(), "nested", "dir", "state.kv")
	store, err := openOrCreateStateStore(context.Background(), path)
	if err != nil {
		t.Fatalf("create: %v", err)
	}
	if err := store.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}
	if !core.Stat(path).OK {
		t.Fatalf("store file %s was not created", path)
	}
	// Second call opens the existing file.
	store2, err := openOrCreateStateStore(context.Background(), path)
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	if err := store2.Close(); err != nil {
		t.Fatalf("reopen close: %v", err)
	}
}

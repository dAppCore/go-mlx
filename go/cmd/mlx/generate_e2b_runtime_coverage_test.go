// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package main

import (
	"context"
	"strings"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/mlx/internal/metaltest"
)

// e2bModelPath resolves the cached 4-bit Gemma-4 E2B snapshot — the smallest
// real instruction-tuned model on disk (~3.6 GB). The runtime command paths
// (generate decode-success tail, the wake/sleep state round-trip) need a model
// that produces a coherent multi-token KV cache, which the degenerate 1-layer
// synthetic cannot: its 5-token KV snapshot fails the wake-side dimension check.
// A weightless checkout skips these tests green via HFModelPath's t.Skip.
//
// AX-11 is the BENCHMARKING standard (synthetic micro-benches, no model loads);
// it explicitly does NOT bar loading a real model to verify a command path
// FUNCTIONS — that's the lem.sh smoke pattern, Snider's call, capped at one
// resident model. These tests load e2b-4bit at most once at a time (defer Close
// between invocations; never two concurrent) and decode only a handful of
// tokens, so peak RAM stays at one model.
func e2bModelPath(t *testing.T) string {
	t.Helper()
	requireSyntheticRuntime(t) // Metal runtime + device guard.
	return metaltest.HFModelPath(t, "mlx-community/gemma-4-e2b-it-4bit")
}

// generate over the real e2b-4bit model runs the full default success tail:
// LoadModelAsTextModel (no MTP drafter is cached beside e2b-4bit, so -draft ""
// keeps the plain text-model load), the warm pass, the timed decode, and the
// n>=2 decode-rate report. This is the coherent-multi-token path the synthetic
// (5-token degenerate decode) never reaches all the way through.
func TestRunGenerate_E2BDecodeSuccessTail_Good(t *testing.T) {
	model := e2bModelPath(t)
	stdout, stderr := core.NewBuffer(), core.NewBuffer()
	code := runCommand(context.Background(), []string{
		"generate", "-max-tokens", "12", "-temp", "0", "-draft", "",
		"-prompt", "Reply with one short sentence.", model,
	}, stdout, stderr)
	if code != 0 {
		t.Fatalf("exit = %d, want 0; stderr=%q", code, stderr.String())
	}
	// The decode-rate report only prints when n>=2 tokens decoded — proof the
	// real model produced a coherent multi-token continuation, not a degenerate
	// single-token stop.
	if !strings.Contains(stdout.String(), "tok/s") {
		t.Fatalf("stdout = %q, want the decode-rate report (n>=2 tokens)", stdout.String())
	}
}

// generate -state over the real e2b-4bit model drives the durable-state turn
// loop end to end across two invocations:
//
//	turn 1 — fresh state: prefill, decode, sleep the KV blocks to the store;
//	turn 2 — wake: restore the KV blocks (no replay), append the new turn,
//	         decode, sleep again.
//
// Turn 2 exercises runGenerateState's woke=true branch (WakeAgentMemory +
// the "woke N prefix tokens (no replay)" report) — the largest uncovered span
// in generate.go, unreachable on the synthetic because the synthetic's tiny KV
// snapshot trips the wake-side dimension check. Both invocations load the same
// e2b model and Close it before returning, so only one model is ever resident.
func TestRunGenerate_E2BStateWakeRoundTrip_Good(t *testing.T) {
	model := e2bModelPath(t)
	store := core.JoinPath(t.TempDir(), "state.kv")

	// Turn 1 — fresh state. Sleeps the KV blocks back to the store.
	out1, err1 := core.NewBuffer(), core.NewBuffer()
	code1 := runCommand(context.Background(), []string{
		"generate", "-state", "chat1", "-state-store", store,
		"-max-tokens", "8", "-temp", "0", "-prompt", "My name is Ada.", model,
	}, out1, err1)
	if code1 != 0 {
		t.Fatalf("turn 1 exit = %d, want 0; stderr=%q", code1, err1.String())
	}
	if !strings.Contains(out1.String(), "fresh state") {
		t.Fatalf("turn 1 stdout = %q, want the fresh-state report", out1.String())
	}
	if !strings.Contains(out1.String(), "slept") {
		t.Fatalf("turn 1 stdout = %q, want the sleep report", out1.String())
	}

	// Turn 2 — wake. Restores the KV blocks (no replay) and appends the new turn.
	out2, err2 := core.NewBuffer(), core.NewBuffer()
	code2 := runCommand(context.Background(), []string{
		"generate", "-state", "chat1", "-state-store", store,
		"-max-tokens", "8", "-temp", "0", "-prompt", "What is my name?", model,
	}, out2, err2)
	if code2 != 0 {
		t.Fatalf("turn 2 exit = %d, want 0 (wake round-trip); stderr=%q", code2, err2.String())
	}
	if !strings.Contains(out2.String(), "woke") {
		t.Fatalf("turn 2 stdout = %q, want the woke (no-replay) report", out2.String())
	}
	if !strings.Contains(out2.String(), "no replay") {
		t.Fatalf("turn 2 stdout = %q, want the no-replay notice", out2.String())
	}
}

// generate -state over the real e2b-4bit model proves the chat-framed turn
// loop's whole point: recall through a genuine no-replay wake. Turn 1 plants
// a name; turn 2 asks for it back after the session wakes from the store —
// only the new turn's tokens prefill (the "no replay" report), yet the
// woken turn is intelligible to the model as a question because it is
// chat-framed (FormatChatContinuation), not a bare raw-text continuation.
// Before this change the same two turns (as -raw still reproduces) EOS on
// an empty answer; this test is the default path's proof that regressed.
//
// -state has no -think wiring (a pre-existing gap, unrelated to this turn
// loop) and Gemma 4's chat template defaults thinking ON, so both turns get
// a generous token budget — enough for the thought channel to run its
// course and still leave room for the visible answer the assertions check.
func TestRunGenerate_E2BStateChatFramedRecall_Good(t *testing.T) {
	model := e2bModelPath(t)
	store := core.JoinPath(t.TempDir(), "state.kv")

	// Turn 1 — fresh state, chat-framed by default (no -raw).
	out1, err1 := core.NewBuffer(), core.NewBuffer()
	code1 := runCommand(context.Background(), []string{
		"generate", "-state", "chat-marker", "-state-store", store,
		"-max-tokens", "300", "-temp", "0", "-prompt", "My name is Marker.", model,
	}, out1, err1)
	if code1 != 0 {
		t.Fatalf("turn 1 exit = %d, want 0; stderr=%q", code1, err1.String())
	}

	// Turn 2 — wake, ask for the planted fact back.
	out2, err2 := core.NewBuffer(), core.NewBuffer()
	code2 := runCommand(context.Background(), []string{
		"generate", "-state", "chat-marker", "-state-store", store,
		"-max-tokens", "300", "-temp", "0", "-prompt", "What is my name?", model,
	}, out2, err2)
	if code2 != 0 {
		t.Fatalf("turn 2 exit = %d, want 0 (wake round-trip); stderr=%q", code2, err2.String())
	}
	if !strings.Contains(out2.String(), "no replay") {
		t.Fatalf("turn 2 stdout = %q, want the no-replay wake report", out2.String())
	}
	if !strings.Contains(out2.String(), "Marker") {
		t.Fatalf("turn 2 stdout = %q, want the answer to recall the planted name", out2.String())
	}
}

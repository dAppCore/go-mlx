// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

// Hermetic tests for the -state turn loop's chat-framed construction
// (runGenerateStateSession in generate.go): each turn appends a properly
// templated user turn + assistant opener onto the woken state — no prompt
// replay — and -raw bypasses the template entirely, byte for byte, exactly
// as the loop always behaved. These drive runGenerateStateSession directly
// against a fake session handle (dappco.re/go/mlx/internal/sessionfake) and
// an in-memory state store, so no model ever loads; the model-gated live
// recall proof lives in generate_e2b_runtime_coverage_test.go.

package main

import (
	"context"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
	state "dappco.re/go/inference/state"
	"dappco.re/go/mlx/chat"
	"dappco.re/go/mlx/internal/sessionfake"
	"dappco.re/go/mlx/pkg/metal"
	"dappco.re/go/mlx/session"
	"dappco.re/go/mlx/spine"
)

// chatArchFormatter drives the real chat.Format pipeline for a fixed
// architecture — the exact rendering mlx.Model.FormatChatPrompt/
// FormatChatContinuation and the native lane's equivalents use (chat_config.go,
// register_native.go) — so these construction tests prove genuine template
// output (BOS once, assistant opener, continuation skip) without loading a
// model.
type chatArchFormatter struct {
	architecture string
}

func (f chatArchFormatter) FormatChatPrompt(messages []inference.Message) string {
	return chat.Format(messages, chat.Config{Architecture: f.architecture})
}

func (f chatArchFormatter) FormatChatContinuation(messages []inference.Message) string {
	return chat.Format(messages, chat.Config{Architecture: f.architecture, Continuation: true})
}

// runGenerateStateSession's fresh-turn branch (no matching state in the
// store) prefills the model's full chat template from empty history — BOS,
// the user turn, and the assistant opener — applied exactly once.
func TestRunGenerateStateSession_FreshTurn_ChatFramed_Good(t *testing.T) {
	ctx := context.Background()
	store := state.NewInMemoryStore(nil)
	handle := &sessionfake.Handle{
		KV:     sessionfake.TestKVSnapshot(),
		Tokens: []metal.Token{{ID: 9, Text: "hi"}},
	}
	sess := session.New(handle, spine.ModelInfo{Architecture: "gemma4_text"}, nil)
	formatter := chatArchFormatter{architecture: "gemma4_text"}
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runGenerateStateSession(ctx, "Hello there", "chat-fresh", "mem", 4, 0, store, sess, formatter, stdout, stderr)
	if code != 0 {
		t.Fatalf("exit = %d, want 0; stderr=%q", code, stderr.String())
	}

	want := chat.Format([]inference.Message{{Role: "user", Content: "Hello there"}}, chat.Config{Architecture: "gemma4_text"})
	if handle.PrefillPrompt != want {
		t.Fatalf("fresh prefill = %q, want the chat-framed turn %q", handle.PrefillPrompt, want)
	}
	if core.Count(handle.PrefillPrompt, "<bos>") != 1 {
		t.Fatalf("fresh prefill = %q, want exactly one <bos> (template applied once)", handle.PrefillPrompt)
	}
	if !core.Contains(handle.PrefillPrompt, "<|turn>model") {
		t.Fatalf("fresh prefill = %q, want the assistant opener", handle.PrefillPrompt)
	}
	if !core.Contains(stdout.String(), "fresh state") {
		t.Fatalf("stdout = %q, want the fresh-state report", stdout.String())
	}
}

// runGenerateStateSession's woken branch (a matching state exists) appends
// only the new turn onto the retained state: the continuation form closes
// the prior open model turn, renders the new user turn once, and reopens
// the assistant header — no BOS (the woken KV state already holds it) and
// no replay of the earlier turn.
func TestRunGenerateStateSession_WokenTurn_ChatFramed_Good(t *testing.T) {
	ctx := context.Background()
	store := state.NewInMemoryStore(nil)
	formatter := chatArchFormatter{architecture: "gemma4_text"}
	info := spine.ModelInfo{Architecture: "gemma4_text"}

	// Turn 1 — fresh. Seeds the store so turn 2 wakes.
	handle1 := &sessionfake.Handle{KV: sessionfake.TestKVSnapshot(), Tokens: []metal.Token{{ID: 1, Text: "hi"}}}
	sess1 := session.New(handle1, info, nil)
	out1, err1 := core.NewBuffer(), core.NewBuffer()
	if code := runGenerateStateSession(ctx, "My name is Marker.", "chat-woken", "mem", 4, 0, store, sess1, formatter, out1, err1); code != 0 {
		t.Fatalf("turn 1 exit = %d, want 0; stderr=%q", code, err1.String())
	}

	// Turn 2 — wake. A fresh handle simulates a new process picking up the
	// same named state (the -state contract: "lthn-mlx generate -state
	// chat1 ..." run twice).
	handle2 := &sessionfake.Handle{KV: sessionfake.TestKVSnapshot(), Tokens: []metal.Token{{ID: 2, Text: "Marker"}}}
	sess2 := session.New(handle2, info, nil)
	out2, err2 := core.NewBuffer(), core.NewBuffer()
	code := runGenerateStateSession(ctx, "What is my name?", "chat-woken", "mem", 4, 0, store, sess2, formatter, out2, err2)
	if code != 0 {
		t.Fatalf("turn 2 exit = %d, want 0; stderr=%q", code, err2.String())
	}

	want := chat.Format([]inference.Message{{Role: "user", Content: "What is my name?"}}, chat.Config{Architecture: "gemma4_text", Continuation: true})
	if handle2.AppendPromptSeen != want {
		t.Fatalf("woken append = %q, want the chat-framed continuation %q", handle2.AppendPromptSeen, want)
	}
	if core.Contains(handle2.AppendPromptSeen, "<bos>") {
		t.Fatalf("woken append = %q, must not re-emit BOS (no-replay contract)", handle2.AppendPromptSeen)
	}
	if core.Count(handle2.AppendPromptSeen, "<|turn>user") != 1 {
		t.Fatalf("woken append = %q, want exactly one new user turn (no prior-turn replay)", handle2.AppendPromptSeen)
	}
	if !core.Contains(handle2.AppendPromptSeen, "<|turn>model") {
		t.Fatalf("woken append = %q, want the assistant opener", handle2.AppendPromptSeen)
	}
	if !core.Contains(out2.String(), "no replay") {
		t.Fatalf("turn 2 stdout = %q, want the no-replay wake report", out2.String())
	}
}

// A nil formatter is the -raw contract: runGenerateStateSession must prefill
// and append the prompt byte-for-byte with no template, exactly the loop's
// original completion-style behaviour, on both the fresh and woken branches.
func TestRunGenerateStateSession_Raw_BypassesChatTemplate_Good(t *testing.T) {
	ctx := context.Background()
	store := state.NewInMemoryStore(nil)
	info := spine.ModelInfo{Architecture: "gemma4_text"}

	// Turn 1 — fresh, raw.
	handle1 := &sessionfake.Handle{KV: sessionfake.TestKVSnapshot(), Tokens: []metal.Token{{ID: 1, Text: "hi"}}}
	sess1 := session.New(handle1, info, nil)
	out1, err1 := core.NewBuffer(), core.NewBuffer()
	if code := runGenerateStateSession(ctx, "My name is Marker.", "chat-raw", "mem", 4, 0, store, sess1, nil, out1, err1); code != 0 {
		t.Fatalf("turn 1 exit = %d, want 0; stderr=%q", code, err1.String())
	}
	if handle1.PrefillPrompt != "My name is Marker." {
		t.Fatalf("raw fresh prefill = %q, want the untouched prompt (no template)", handle1.PrefillPrompt)
	}

	// Turn 2 — woken, raw: the historic "\n"+prompt append, byte for byte.
	handle2 := &sessionfake.Handle{KV: sessionfake.TestKVSnapshot(), Tokens: []metal.Token{{ID: 2, Text: "?"}}}
	sess2 := session.New(handle2, info, nil)
	out2, err2 := core.NewBuffer(), core.NewBuffer()
	if code := runGenerateStateSession(ctx, "What is my name?", "chat-raw", "mem", 4, 0, store, sess2, nil, out2, err2); code != 0 {
		t.Fatalf("turn 2 exit = %d, want 0; stderr=%q", code, err2.String())
	}
	if handle2.AppendPromptSeen != "\nWhat is my name?" {
		t.Fatalf("raw woken append = %q, want the untemplated \"\\n\"+prompt", handle2.AppendPromptSeen)
	}
}

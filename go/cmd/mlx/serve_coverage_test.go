// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"context"
	"strings"
	"testing"
	"time"

	core "dappco.re/go"
)

// serve with an unrecognised flag fails to parse and prints the full usage
// block (the fs.Usage closure: examples + route list + admin token notes),
// exiting 2.
func TestRunServe_UnknownFlag_PrintsUsage_Bad(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	stdout, stderr := core.NewBuffer(), core.NewBuffer()
	code := runCommand(context.Background(), []string{"serve", "-no-such-flag"}, stdout, stderr)
	if code != 2 {
		t.Fatalf("exit = %d, want 2 (unknown flag)", code)
	}
	out := stderr.String()
	for _, want := range []string{"Usage:", "Inference routes", "Admin routes", "/v1/chat/completions"} {
		if !strings.Contains(out, want) {
			t.Fatalf("stderr missing %q — the full serve usage block must print; got %q", want, out[:min(len(out), 200)])
		}
	}
}

// serve -h prints usage and exits 0 (flag.ErrHelp short-circuit).
func TestRunServe_Help_Good(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	stdout, stderr := core.NewBuffer(), core.NewBuffer()
	code := runCommand(context.Background(), []string{"serve", "-h"}, stdout, stderr)
	if code != 0 {
		t.Fatalf("exit = %d, want 0 for -h", code)
	}
	if !strings.Contains(stderr.String(), "Host an OpenAI") {
		t.Fatalf("stderr = %q, want serve help body", stderr.String())
	}
}

// runModellessServe boots serve model-less with the given extra flags on a
// loopback :0 listener, waits briefly, cancels the context, and asserts the
// graceful-shutdown exit 0. Shared driver for the boot-path coverage cases.
func runModellessServe(t *testing.T, extra ...string) string {
	t.Helper()
	t.Setenv("HOME", t.TempDir())
	ctx, cancel := context.WithCancel(context.Background())
	stdout, stderr := core.NewBuffer(), core.NewBuffer()
	args := append([]string{"serve", "-addr", "127.0.0.1:0", "-shutdown-timeout", "2s"}, extra...)
	done := make(chan int, 1)
	go func() { done <- runCommand(ctx, args, stdout, stderr) }()
	time.Sleep(200 * time.Millisecond)
	cancel()
	select {
	case code := <-done:
		if code != 0 {
			t.Fatalf("model-less serve exit = %d, want 0 on graceful shutdown; stderr=%q", code, stderr.String())
		}
	case <-time.After(10 * time.Second):
		t.Fatal("serve did not shut down within 10s of context cancel")
	}
	return stderr.String()
}

// serve model-less with -context + -kv-cache paged applies both load options
// and snapshots them into the boot status, then shuts down cleanly. Exercises
// the context-override + cache-mode branches of runServeCommand.
func TestRunServe_ContextAndCacheMode_Good(t *testing.T) {
	out := runModellessServe(t, "-context", "8192", "-kv-cache", "paged", "-state-conversations=false")
	if !strings.Contains(out, "listening on") {
		t.Fatalf("stderr = %q, want the listening notice", out)
	}
}

// serve model-less with -native boots the no-cgo contract path (the loader is
// swapped) and shuts down cleanly.
func TestRunServe_NativeBackend_Good(t *testing.T) {
	out := runModellessServe(t, "-native", "-state-conversations=false")
	if !strings.Contains(out, "no-cgo native token-loop contract") {
		t.Fatalf("stderr = %q, want the native-backend notice", out)
	}
}

func TestRunServe_NativeBackendDoesNotDisableContinuity_Good(t *testing.T) {
	out := runModellessServe(t, "-native")
	if strings.Contains(out, "continuity off") {
		t.Fatalf("stderr = %q, native serve must not hard-disable continuity", out)
	}
}

// serve model-less with conversation continuity ON (the default) opens the
// state store under HOME, wiring the onLoad hook, then shuts down cleanly —
// the state-store boot branch.
func TestRunServe_StateConversationsStore_Good(t *testing.T) {
	out := runModellessServe(t) // state-conversations defaults ON
	// Either it wired the store (silent) or printed a degraded notice — both
	// mean the continuity branch executed. The listening notice confirms boot.
	if !strings.Contains(out, "listening on") {
		t.Fatalf("stderr = %q, want the listening notice", out)
	}
}

// serve with an explicit -state-store under a fresh dir creates the store on
// demand during boot.
func TestRunServe_ExplicitStateStore_Good(t *testing.T) {
	home := t.TempDir()
	store := core.JoinPath(home, "made", "on", "demand", "conv.kv")
	ctx, cancel := context.WithCancel(context.Background())
	stdout, stderr := core.NewBuffer(), core.NewBuffer()
	done := make(chan int, 1)
	go func() {
		done <- runCommand(ctx, []string{
			"serve", "-addr", "127.0.0.1:0", "-shutdown-timeout", "2s", "-state-store", store,
		}, stdout, stderr)
	}()
	time.Sleep(200 * time.Millisecond)
	cancel()
	select {
	case code := <-done:
		if code != 0 {
			t.Fatalf("serve exit = %d, want 0; stderr=%q", code, stderr.String())
		}
	case <-time.After(10 * time.Second):
		t.Fatal("serve did not shut down in time")
	}
	if !core.Stat(store).OK {
		t.Fatalf("explicit state store %s was not created on boot", store)
	}
}

// serve with an unbindable listen address (a port already in use is racy, so
// use a syntactically invalid address) reaches the listen-failure return path
// (exit 1).
func TestRunServe_ListenFailure_Bad(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	stdout, stderr := core.NewBuffer(), core.NewBuffer()
	code := runCommand(context.Background(), []string{
		"serve", "-addr", "256.256.256.256:99999", "-state-conversations=false",
	}, stdout, stderr)
	if code != 1 {
		t.Fatalf("exit = %d, want 1 (listen failure); stderr=%q", code, stderr.String())
	}
	if !strings.Contains(stderr.String(), "listen failed") {
		t.Fatalf("stderr = %q, want the listen-failed notice", stderr.String())
	}
}

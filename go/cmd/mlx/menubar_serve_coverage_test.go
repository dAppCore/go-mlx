// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package main

import (
	"testing"
	"time"
)

// isInsideAppBundle returns false in a plain test binary (no Info.plist bundle
// identifier). The branch matters for the no-arg CLI dispatch (help vs menubar).
func TestMenubar_IsInsideAppBundle_Good(t *testing.T) {
	if isInsideAppBundle() {
		t.Fatal("isInsideAppBundle() = true under `go test`, want false (no .app bundle)")
	}
}

// startMenubarServe binds the OpenAI HTTP mux on the configured address in a
// background goroutine; stopMenubarServe shuts it down cleanly. Driven on a
// loopback :0 address so the kernel assigns a free port (no collision with a
// real serve on :36911), with a model-less resolver (no weights loaded).
func TestMenubar_StartStopServe_Good(t *testing.T) {
	state := &menubarState{addr: "127.0.0.1:0"}
	refreshed := make(chan struct{}, 4)
	refresh := func() {
		select {
		case refreshed <- struct{}{}:
		default:
		}
	}

	startMenubarServe(state, refresh)
	if !state.serving.Load() {
		t.Fatal("after startMenubarServe, serving = false; want true")
	}
	if state.server == nil {
		t.Fatal("startMenubarServe did not record the *http.Server")
	}

	// Give the listener a beat to bind, then stop it.
	time.Sleep(100 * time.Millisecond)
	stopMenubarServe(state)
	if state.serving.Load() {
		t.Fatal("after stopMenubarServe, serving = true; want false")
	}
	if state.server != nil {
		t.Fatal("stopMenubarServe did not clear the *http.Server")
	}

	// The serve goroutine's deferred refresh fires once ListenAndServe returns
	// (ErrServerClosed after Shutdown) — wait for it so the goroutine is done.
	select {
	case <-refreshed:
	case <-time.After(2 * time.Second):
		t.Fatal("serve goroutine never fired refresh after shutdown")
	}
}

// stopMenubarServe is a safe no-op when no server is running (the guard for a
// stop click while idle).
func TestMenubar_StopServe_NoServer_Good(t *testing.T) {
	state := &menubarState{addr: "127.0.0.1:0"}
	stopMenubarServe(state) // must not panic with a nil server
	if state.serving.Load() {
		t.Fatal("idle stopMenubarServe set serving = true")
	}
}

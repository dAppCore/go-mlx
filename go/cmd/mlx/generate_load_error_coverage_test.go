// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package main

import (
	"context"
	"strings"
	"testing"

	core "dappco.re/go"
)

// generate -state with a WRITABLE store but a nonexistent model path reaches
// runGenerateState's model-load error (generate.go ~228): the store opens
// successfully, then LoadModel fails. (The store-error sibling test uses a bad
// store path, which fails earlier — this one flips that to hit the load error.)
func TestRunGenerateState_ModelLoadFails_Bad(t *testing.T) {
	dir := t.TempDir()
	store := core.JoinPath(dir, "state.kv") // writable — the store opens fine
	stdout, stderr := core.NewBuffer(), core.NewBuffer()
	code := runCommand(context.Background(), []string{
		"generate", "-state", "chat", "-state-store", store,
		"-max-tokens", "2", core.JoinPath(dir, "no-such-model"),
	}, stdout, stderr)
	if code != 1 {
		t.Fatalf("exit = %d, want 1 (model load failure after store opens); stderr=%q", code, stderr.String())
	}
	if !strings.Contains(stderr.String(), "load") {
		t.Fatalf("stderr = %q, want the model-load error", stderr.String())
	}
}

// generate -trace against a nonexistent model reaches runGenerateTrace's
// load-error branch (generate.go ~339).
func TestRunGenerateTrace_ModelLoadFails_Bad(t *testing.T) {
	stdout, stderr := core.NewBuffer(), core.NewBuffer()
	code := runCommand(context.Background(), []string{
		"generate", "-trace", "-max-tokens", "2", core.JoinPath(t.TempDir(), "no-such-model"),
	}, stdout, stderr)
	if code != 1 {
		t.Fatalf("exit = %d, want 1 (trace model load failure); stderr=%q", code, stderr.String())
	}
	if !strings.Contains(stderr.String(), "load") {
		t.Fatalf("stderr = %q, want the model-load error", stderr.String())
	}
}

// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package main

import (
	"context"
	"strings"
	"testing"

	core "dappco.re/go"
)

// runGenerateState fails at the state-store open BEFORE any model load when the
// store path is unreachable (its parent is a regular file). This exercises the
// store-error branch of runGenerateState (generate.go ~221) without loading a
// model — the store opens before LoadModel, so the missing model path is never
// reached.
func TestRunGenerateState_BadStorePath_Bad(t *testing.T) {
	tmp := t.TempDir()
	fileAsDir := core.PathJoin(tmp, "afile")
	if r := core.WriteFile(fileAsDir, []byte("x"), 0o644); !r.OK {
		t.Fatalf("seed file: %v", r.Value)
	}
	badStore := core.PathJoin(fileAsDir, "sub", "agent.kv") // parent is a file
	stdout, stderr := core.NewBuffer(), core.NewBuffer()
	code := runCommand(context.Background(), []string{
		"generate", "-state", "chat", "-state-store", badStore,
		"-max-tokens", "2", core.PathJoin(tmp, "no-model"),
	}, stdout, stderr)
	if code != 1 {
		t.Fatalf("exit = %d, want 1 (state store unreachable); stderr=%q", code, stderr.String())
	}
	if !strings.Contains(stderr.String(), "state store") {
		t.Fatalf("stderr = %q, want the state-store open error", stderr.String())
	}
}

// openOrCreateStateStore returns the mkdir error when the store's parent
// directory cannot be created (the parent path is a regular file).
func TestOpenOrCreateStateStore_MkdirFails_Bad(t *testing.T) {
	tmp := t.TempDir()
	fileAsDir := core.PathJoin(tmp, "afile")
	if r := core.WriteFile(fileAsDir, []byte("x"), 0o644); !r.OK {
		t.Fatalf("seed file: %v", r.Value)
	}
	_, err := openOrCreateStateStore(context.Background(), core.PathJoin(fileAsDir, "sub", "agent.kv"))
	if err == nil {
		t.Fatal("want a mkdir error when the store parent path is a regular file")
	}
}

// openOrCreateStateStore creates a fresh store (and its parent directory) when
// the path does not yet exist — the create-from-absent path.
func TestOpenOrCreateStateStore_CreatesFresh_Good(t *testing.T) {
	path := core.PathJoin(t.TempDir(), "nested", "agent.kv")
	store, err := openOrCreateStateStore(context.Background(), path)
	if err != nil {
		t.Fatalf("create fresh store: %v", err)
	}
	defer store.Close()
	if !core.Stat(path).OK {
		t.Fatalf("store file %q was not created", path)
	}
}

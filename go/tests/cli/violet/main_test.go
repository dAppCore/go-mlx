// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"context"
	"net"
	"syscall"
	"testing"
	"time"

	core "dappco.re/go"
)

func TestMain_RepoRootAndWaitForSocket_Good(t *testing.T) {
	root, err := repoRoot()
	if err != nil {
		t.Fatalf("repoRoot() error = %v", err)
	}
	if !core.Stat(core.PathJoin(root, "go.mod")).OK {
		t.Fatalf("repoRoot() = %q, want directory containing go.mod", root)
	}

	socketPath := core.PathJoin(shortTempDir(t), "violet.sock")
	listener, err := net.Listen("unix", socketPath)
	if err != nil {
		t.Fatalf("listen unix: %v", err)
	}
	defer listener.Close()
	if err := waitForSocket(socketPath); err != nil {
		t.Fatalf("waitForSocket() error = %v", err)
	}
}

func TestMain_WaitForSocket_Bad(t *testing.T) {
	start := time.Now()
	err := waitForSocket(core.PathJoin(shortTempDir(t), "missing.sock"))
	if err == nil {
		t.Fatal("expected missing socket error")
	}
	if !core.Contains(err.Error(), "was not created") {
		t.Fatalf("error = %v", err)
	}
	if time.Since(start) < 4*time.Second {
		t.Fatalf("waitForSocket returned too quickly; missing socket polling changed")
	}
}

func TestMain_ProcessRunAndWait_Good(t *testing.T) {
	output, err := runCommand(t.TempDir(), "/bin/echo", "violet-ok")
	if err != nil {
		t.Fatalf("runCommand() error = %v output=%q", err, output)
	}
	if !core.Contains(output, "violet-ok") {
		t.Fatalf("output = %q", output)
	}
}

func TestMain_ProcessWaitAndKill_Bad(t *testing.T) {
	proc, stdout, stderr, err := startCommand(context.Background(), t.TempDir(), "/bin/sh", "-c", "exit 7")
	if err != nil {
		t.Fatalf("startCommand() error = %v", err)
	}
	if err := proc.Wait(); err == nil {
		t.Fatalf("Wait() error = nil, stdout=%q stderr=%q", stdout.String(), stderr.String())
	}
	if err := (*cliprocess)(nil).Wait(); err != nil {
		t.Fatalf("nil Wait() error = %v", err)
	}
	if err := (*cliprocess)(nil).Kill(); err != nil {
		t.Fatalf("nil Kill() error = %v", err)
	}
}

func TestMain_LookPathExecutableAndCloseFDs_Ugly(t *testing.T) {
	dir := t.TempDir()
	exe := core.PathJoin(dir, "tool.sh")
	if result := core.WriteFile(exe, []byte("#!/bin/sh\nexit 0\n"), 0o755); !result.OK {
		t.Fatalf("write executable: %v", result.Value)
	}
	if got, err := lookPath(exe); err != nil || got != exe {
		t.Fatalf("lookPath(direct) = %q, %v", got, err)
	}
	if !executable(exe) {
		t.Fatalf("executable(%q) = false", exe)
	}
	notExecutable := core.PathJoin(dir, "data.txt")
	if result := core.WriteFile(notExecutable, []byte("x"), 0o644); !result.OK {
		t.Fatalf("write file: %v", result.Value)
	}
	if executable(notExecutable) {
		t.Fatalf("executable(%q) = true", notExecutable)
	}
	if _, err := lookPath(notExecutable); err == nil {
		t.Fatal("expected lookPath error for non-executable direct path")
	}
	if !contains([]string{"info", "generate"}, "info") || contains([]string{"info"}, "missing") {
		t.Fatal("contains() returned unexpected result")
	}

	fds := []int{0, 0}
	if err := syscall.Pipe(fds); err != nil {
		t.Fatalf("pipe: %v", err)
	}
	if err := closeFDs(0, fds[0], fds[1]); err != nil {
		t.Fatalf("closeFDs() error = %v", err)
	}
}

func shortTempDir(t *testing.T) string {
	t.Helper()
	result := core.MkdirTemp("/tmp", "vlt-*")
	if !result.OK {
		t.Fatalf("mkdir temp: %v", result.Value)
	}
	dir := result.Value.(string)
	t.Cleanup(func() { core.RemoveAll(dir) })
	return dir
}

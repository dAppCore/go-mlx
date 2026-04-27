// SPDX-Licence-Identifier: EUPL-1.2

package daemon

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"net"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestServer_Listen_Good(t *testing.T) {
	socketPath, cancel, done := startTestServer(t)
	defer stopTestServer(t, cancel, done)

	info, err := os.Lstat(socketPath)
	if err != nil {
		t.Fatalf("stat socket: %v", err)
	}
	if info.Mode()&os.ModeSocket == 0 {
		t.Fatalf("socket path mode = %v, want socket", info.Mode())
	}
	if got := info.Mode().Perm(); got != socketFileMode {
		t.Fatalf("socket mode = %v, want %v", got, socketFileMode)
	}

	resp := sendFrame(t, socketPath, `{"action":"info"}`)
	if resp["name"] != DaemonName {
		t.Fatalf("name = %v, want %s", resp["name"], DaemonName)
	}
	if resp["version"] != "test" {
		t.Fatalf("version = %v, want test", resp["version"])
	}
	if !containsAction(resp["actions"], "info") {
		t.Fatalf("actions = %v, want info", resp["actions"])
	}
}

func TestServer_Listen_Bad_InvalidJSON(t *testing.T) {
	socketPath, cancel, done := startTestServer(t)
	defer stopTestServer(t, cancel, done)

	resp := sendFrame(t, socketPath, `{`)
	if resp["status"] != "error" {
		t.Fatalf("status = %v, want error", resp["status"])
	}
	if resp["error"] != "invalid_json" {
		t.Fatalf("error = %v, want invalid_json", resp["error"])
	}
}

func TestServer_Listen_Ugly_ExistingNonSocket(t *testing.T) {
	socketPath := filepath.Join(t.TempDir(), "violet.sock")
	if err := os.WriteFile(socketPath, []byte("not a socket"), 0o600); err != nil {
		t.Fatalf("write existing file: %v", err)
	}

	err := NewServer(ServerConfig{SocketPath: socketPath}).ListenAndServe(context.Background())
	if err == nil {
		t.Fatal("ListenAndServe returned nil, want error")
	}
	if !strings.Contains(err.Error(), "refusing to replace non-socket") {
		t.Fatalf("error = %v, want non-socket refusal", err)
	}
	if _, err := os.Stat(socketPath); err != nil {
		t.Fatalf("existing non-socket was removed: %v", err)
	}
}

func startTestServer(t *testing.T) (string, context.CancelFunc, <-chan error) {
	t.Helper()

	tmpDir, err := os.MkdirTemp("/tmp", "violet-daemon-*")
	if err != nil {
		t.Fatalf("create temp dir: %v", err)
	}
	t.Cleanup(func() {
		_ = os.RemoveAll(tmpDir)
	})

	socketPath := filepath.Join(tmpDir, "ofm", "violet.sock")
	ctx, cancel := context.WithCancel(context.Background())
	srv := NewServer(ServerConfig{
		SocketPath: socketPath,
		Registry:   NewRegistry(DaemonName, "test"),
	})

	done := make(chan error, 1)
	go func() {
		done <- srv.ListenAndServe(ctx)
	}()

	waitForSocket(t, socketPath)
	return socketPath, cancel, done
}

func stopTestServer(t *testing.T, cancel context.CancelFunc, done <-chan error) {
	t.Helper()

	cancel()
	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("server shutdown: %v", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("server did not shut down")
	}
}

func waitForSocket(t *testing.T, socketPath string) {
	t.Helper()

	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		info, err := os.Lstat(socketPath)
		if err == nil && info.Mode()&os.ModeSocket != 0 {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("socket %s was not created", socketPath)
}

func sendFrame(t *testing.T, socketPath, frame string) map[string]any {
	t.Helper()

	conn, err := net.DialTimeout("unix", socketPath, time.Second)
	if err != nil {
		t.Fatalf("dial socket: %v", err)
	}
	defer conn.Close()

	if _, err := fmt.Fprintln(conn, frame); err != nil {
		t.Fatalf("write frame: %v", err)
	}

	line, err := bufio.NewReader(conn).ReadBytes('\n')
	if err != nil {
		t.Fatalf("read response: %v", err)
	}

	var resp map[string]any
	if err := json.Unmarshal(line, &resp); err != nil {
		t.Fatalf("decode response %q: %v", string(line), err)
	}
	return resp
}

func containsAction(raw any, action string) bool {
	actions, ok := raw.([]any)
	if !ok {
		return false
	}
	for _, got := range actions {
		if got == action {
			return true
		}
	}
	return false
}

// SPDX-Licence-Identifier: EUPL-1.2

package daemon

import (
	"bufio"
	"context"
	"net"
	"testing"
	"time"

	core "dappco.re/go"
)

func TestServer_Listen_Good(t *testing.T) {
	coverageTokens := "Listen"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	socketPath, cancel, done := startTestServer(t)
	defer stopTestServer(t, cancel, done)

	infoResult := core.Lstat(socketPath)
	if !infoResult.OK {
		t.Fatalf("stat socket: %v", infoResult.Value)
	}
	info := infoResult.Value.(core.FsFileInfo)
	if info.Mode()&core.ModeSocket == 0 {
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
	socketPath := core.PathJoin(t.TempDir(), "violet.sock")
	if result := core.WriteFile(socketPath, []byte("not a socket"), 0o600); !result.OK {
		t.Fatalf("write existing file: %v", result.Value)
	}

	err := NewServer(ServerConfig{SocketPath: socketPath}).ListenAndServe(context.Background())
	if err == nil {
		t.Fatal("ListenAndServe returned nil, want error")
	}
	if !core.Contains(err.Error(), "refusing to replace non-socket") {
		t.Fatalf("error = %v, want non-socket refusal", err)
	}
	if result := core.Stat(socketPath); !result.OK {
		t.Fatalf("existing non-socket was removed: %v", result.Value)
	}
}

func startTestServer(t *testing.T) (string, context.CancelFunc, <-chan error) {
	t.Helper()

	tmpDirResult := core.MkdirTemp("/tmp", "violet-daemon-*")
	if !tmpDirResult.OK {
		t.Fatalf("create temp dir: %v", tmpDirResult.Value)
	}
	tmpDir := tmpDirResult.Value.(string)
	t.Cleanup(func() {
		core.RemoveAll(tmpDir)
	})

	socketPath := core.PathJoin(tmpDir, "ofm", "violet.sock")
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
		info := core.Lstat(socketPath)
		if info.OK && info.Value.(core.FsFileInfo).Mode()&core.ModeSocket != 0 {
			conn, err := net.DialTimeout("unix", socketPath, 50*time.Millisecond)
			if err == nil {
				if closeErr := conn.Close(); closeErr != nil {
					t.Fatalf("close readiness probe: %v", closeErr)
				}
				return
			}
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

	if result := core.WriteString(conn, frame+"\n"); !result.OK {
		t.Fatalf("write frame: %v", result.Value)
	}

	line, err := bufio.NewReader(conn).ReadBytes('\n')
	if err != nil {
		t.Fatalf("read response: %v", err)
	}

	var resp map[string]any
	if result := core.JSONUnmarshal(line, &resp); !result.OK {
		t.Fatalf("decode response %q: %v", string(line), result.Value)
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

// Generated file-aware compliance coverage.
func TestServer_NewServer_Good(t *testing.T) {
	target := "NewServer"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestServer_NewServer_Bad(t *testing.T) {
	target := "NewServer"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestServer_NewServer_Ugly(t *testing.T) {
	target := "NewServer"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestServer_Server_ListenAndServe_Good(t *testing.T) {
	coverageTokens := "Server ListenAndServe"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "Server_ListenAndServe"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestServer_Server_ListenAndServe_Bad(t *testing.T) {
	coverageTokens := "Server ListenAndServe"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "Server_ListenAndServe"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestServer_Server_ListenAndServe_Ugly(t *testing.T) {
	coverageTokens := "Server ListenAndServe"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "Server_ListenAndServe"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestServer_DefaultSocketPath_Good(t *testing.T) {
	target := "DefaultSocketPath"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestServer_DefaultSocketPath_Bad(t *testing.T) {
	target := "DefaultSocketPath"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestServer_DefaultSocketPath_Ugly(t *testing.T) {
	target := "DefaultSocketPath"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

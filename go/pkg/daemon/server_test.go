// SPDX-Licence-Identifier: EUPL-1.2

package daemon

import (
	"bufio"
	"context"
	"net"
	"runtime"
	"testing"
	"time"

	core "dappco.re/go"
)

func TestServer_Listen_Good_LiveSocket(t *testing.T) {
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

func TestServer_Listen_Good_GenerateBackend(t *testing.T) {
	backend := &fakeGenerateBackend{result: GenerateResult{Text: "native ok", Model: "default"}}
	socketPath, cancel, done := startTestServerWithConfig(t, ServerConfig{GenerateBackend: backend})
	defer stopTestServer(t, cancel, done)

	resp := sendFrame(t, socketPath, `{"action":"generate","prompt":"hello","model":"default","max_tokens":8}`)

	if resp["status"] != "ok" {
		t.Fatalf("status = %v, want ok", resp["status"])
	}
	if resp["text"] != "native ok" {
		t.Fatalf("text = %v, want native ok", resp["text"])
	}
	if backend.request.Prompt != "hello" {
		t.Fatalf("backend prompt = %q, want hello", backend.request.Prompt)
	}
	if backend.request.MaxTokens != 8 {
		t.Fatalf("backend max tokens = %d, want 8", backend.request.MaxTokens)
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

func TestNewServer_ConfigCloneAndDefaults_Good_ClonesAndDefaults(t *testing.T) {
	modelPaths := map[string]string{"default": "/models/qwen"}
	server := NewServer(ServerConfig{ModelPaths: modelPaths})
	modelPaths["default"] = "/mutated"

	if server.Registry == nil {
		t.Fatal("Registry = nil, want default registry")
	}
	if server.ModelPaths["default"] != "/models/qwen" {
		t.Fatalf("ModelPaths was not cloned: %+v", server.ModelPaths)
	}
	if server.GenerateBackend == nil {
		t.Fatal("GenerateBackend = nil, want native backend for configured model paths")
	}

	explicit := NewServer(ServerConfig{Registry: NewRegistry("violet-test", "1"), GenerateBackend: &fakeGenerateBackend{}})
	if explicit.GenerateBackend == nil {
		t.Fatal("explicit GenerateBackend was not retained")
	}
}

type closeableGenerateBackend struct {
	fakeGenerateBackend
	closeErr error
	closed   bool
}

func (backend *closeableGenerateBackend) Close() error {
	backend.closed = true
	return backend.closeErr
}

func TestCloseGenerateBackend_Good(t *testing.T) {
	if err := closeGenerateBackend(nil); err != nil {
		t.Fatalf("closeGenerateBackend(nil) = %v", err)
	}
	if err := closeGenerateBackend(&fakeGenerateBackend{}); err != nil {
		t.Fatalf("closeGenerateBackend(non-closer) = %v", err)
	}

	wantErr := core.NewError("close failed")
	backend := &closeableGenerateBackend{closeErr: wantErr}
	if err := closeGenerateBackend(backend); !core.Is(err, wantErr) {
		t.Fatalf("closeGenerateBackend(closer) = %v, want %v", err, wantErr)
	}
	if !backend.closed {
		t.Fatal("closeable backend was not closed")
	}
}

func TestServer_ResolvedSocketPathAndDefaults_Good_Fallbacks(t *testing.T) {
	socketPath := core.PathJoin(t.TempDir(), "violet.sock")
	server := &Server{SocketPath: socketPath}
	got, err := server.resolvedSocketPath()
	if err != nil {
		t.Fatalf("resolvedSocketPath(explicit): %v", err)
	}
	if got != socketPath {
		t.Fatalf("resolvedSocketPath(explicit) = %q, want %q", got, socketPath)
	}

	defaultPath, err := DefaultSocketPath()
	if err != nil {
		t.Fatalf("DefaultSocketPath(): %v", err)
	}
	if defaultPath == "" || !core.Contains(defaultPath, "violet.sock") {
		t.Fatalf("DefaultSocketPath() = %q, want violet.sock path", defaultPath)
	}

	// A nil server and an empty SocketPath both fall through to the default path.
	var nilServer *Server
	nilPath, err := nilServer.resolvedSocketPath()
	if err != nil {
		t.Fatalf("resolvedSocketPath(nil server): %v", err)
	}
	if nilPath != defaultPath {
		t.Fatalf("resolvedSocketPath(nil) = %q, want default %q", nilPath, defaultPath)
	}
	emptyPath, err := (&Server{}).resolvedSocketPath()
	if err != nil {
		t.Fatalf("resolvedSocketPath(empty): %v", err)
	}
	if emptyPath != defaultPath {
		t.Fatalf("resolvedSocketPath(empty) = %q, want default %q", emptyPath, defaultPath)
	}
}

func TestPrepareSocketPath_Validation_Bad_EmptyAndStale(t *testing.T) {
	if err := prepareSocketPath(""); err == nil {
		t.Fatal("expected empty socket path error")
	}

	dirResult := core.MkdirTemp("/tmp", "violet-daemon-stale-*")
	if !dirResult.OK {
		t.Fatalf("create short temp dir: %v", dirResult.Value)
	}
	dir := dirResult.Value.(string)
	t.Cleanup(func() { core.RemoveAll(dir) })
	socketPath := core.PathJoin(dir, "stale.sock")
	ln, err := net.Listen("unix", socketPath)
	if err != nil {
		t.Fatalf("listen stale socket: %v", err)
	}
	if err := ln.Close(); err != nil {
		t.Fatalf("close stale listener: %v", err)
	}
	if err := prepareSocketPath(socketPath); err != nil {
		t.Fatalf("prepareSocketPath(stale socket): %v", err)
	}
	if stat := core.Lstat(socketPath); stat.OK {
		t.Fatal("stale socket path should have been removed")
	}
}

func TestWriteJSONLineAndRemovePath_Bad(t *testing.T) {
	buf := core.NewBuffer()
	if err := writeJSONLine(buf, map[string]string{"status": "ok"}); err != nil {
		t.Fatalf("writeJSONLine(valid): %v", err)
	}
	if got := buf.String(); got != "{\"status\":\"ok\"}\n" {
		t.Fatalf("writeJSONLine output = %q", got)
	}
	if err := writeJSONLine(core.NewBuffer(), map[string]any{"bad": make(chan int)}); err == nil {
		t.Fatal("expected JSON marshal error")
	}

	path := core.PathJoin(t.TempDir(), "delete-me")
	if result := core.WriteFile(path, []byte("x"), 0o644); !result.OK {
		t.Fatalf("write file: %v", result.Value)
	}
	if err := removePath(path); err != nil {
		t.Fatalf("removePath(existing): %v", err)
	}
	if err := removePath(path); err == nil {
		t.Fatal("expected removePath missing file error")
	}

	if err := daemonResultError(core.Result{Value: "bad", OK: false}); err == nil || !core.Contains(err.Error(), "daemon operation failed") {
		t.Fatalf("daemonResultError(non-error) = %v", err)
	}
}

// shortSocketPath returns a socket path under /tmp (not t.TempDir, whose
// /var/folders/... prefix on macOS overflows the 104-byte sun_path limit).
func shortSocketPath(t *testing.T) string {
	t.Helper()
	dirResult := core.MkdirTemp("/tmp", "violet-daemon-*")
	if !dirResult.OK {
		t.Fatalf("create short temp dir: %v", dirResult.Value)
	}
	dir := dirResult.Value.(string)
	t.Cleanup(func() { core.RemoveAll(dir) })
	return core.PathJoin(dir, "violet.sock")
}

func startTestServer(t *testing.T) (string, context.CancelFunc, <-chan error) {
	t.Helper()

	return startTestServerWithConfig(t, ServerConfig{
		Registry: NewRegistry(DaemonName, "test"),
	})
}

func startTestServerWithConfig(t *testing.T, cfg ServerConfig) (string, context.CancelFunc, <-chan error) {
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
	cfg.SocketPath = socketPath
	if cfg.Registry == nil {
		cfg.Registry = NewRegistry(DaemonName, "test")
	}
	srv := NewServer(cfg)

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

// --- v0.9.0 canonical AX-7 triplets (file prefix: Server) -------------------
// Test<Server>_<Symbol>_{Good,Bad,Ugly}. The Listen scenario tests above stay
// as the live-socket coverage; these canonical entries each name their symbol
// directly so the audit can tie the test to its subject.

func TestServer_NewServer_Good(t *testing.T) {
	modelPaths := map[string]string{"default": "/models/qwen"}
	server := NewServer(ServerConfig{ModelPaths: modelPaths})
	if server == nil {
		t.Fatal("NewServer() = nil, want server")
	}
	if server.Registry == nil {
		t.Fatal("NewServer() Registry = nil, want default registry")
	}
	if server.GenerateBackend == nil {
		t.Fatal("NewServer() GenerateBackend = nil, want native backend for configured paths")
	}
}

func TestServer_NewServer_Bad(t *testing.T) {
	// Not an error path: an empty config is degenerate input. NewServer must
	// still return a usable server with the default registry and no backend
	// (no model paths configured), never nil.
	server := NewServer(ServerConfig{})
	if server == nil {
		t.Fatal("NewServer(empty) = nil, want server")
	}
	if server.Registry == nil {
		t.Fatal("NewServer(empty) Registry = nil, want default registry")
	}
	if server.GenerateBackend != nil {
		t.Fatalf("NewServer(empty) GenerateBackend = %v, want nil with no model paths", server.GenerateBackend)
	}
}

func TestServer_NewServer_Ugly(t *testing.T) {
	// NewServer must clone the caller's ModelPaths map — mutating it after
	// construction must not change the server's recorded paths.
	modelPaths := map[string]string{"default": "/models/qwen"}
	server := NewServer(ServerConfig{ModelPaths: modelPaths})
	modelPaths["default"] = "/mutated"
	if server.ModelPaths["default"] != "/models/qwen" {
		t.Fatalf("NewServer did not clone ModelPaths: %+v", server.ModelPaths)
	}
}

func TestServer_Server_ListenAndServe_Good(t *testing.T) {
	socketPath := shortSocketPath(t)
	server := NewServer(ServerConfig{SocketPath: socketPath, Registry: NewRegistry(DaemonName, "test")})

	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() { done <- server.ListenAndServe(ctx) }()
	waitForSocket(t, socketPath)

	resp := sendFrame(t, socketPath, `{"action":"info"}`)
	if resp["name"] != DaemonName {
		t.Fatalf("name = %v, want %s", resp["name"], DaemonName)
	}

	cancel()
	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("ListenAndServe shutdown error = %v", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("ListenAndServe did not return after cancel")
	}
}

func TestServer_Server_ListenAndServe_Bad(t *testing.T) {
	// An existing non-socket file at the path must make ListenAndServe refuse
	// rather than clobber it.
	socketPath := core.PathJoin(t.TempDir(), "violet.sock")
	if result := core.WriteFile(socketPath, []byte("not a socket"), 0o600); !result.OK {
		t.Fatalf("write existing file: %v", result.Value)
	}
	err := NewServer(ServerConfig{SocketPath: socketPath}).ListenAndServe(context.Background())
	if err == nil {
		t.Fatal("ListenAndServe over a non-socket returned nil, want error")
	}
	if !core.Contains(err.Error(), "refusing to replace non-socket") {
		t.Fatalf("ListenAndServe error = %v, want non-socket refusal", err)
	}
}

func TestServer_Server_ListenAndServe_Ugly(t *testing.T) {
	// An already-cancelled context: ListenAndServe must bind, observe the dead
	// context, and shut down cleanly with a nil error rather than hang.
	socketPath := shortSocketPath(t)
	server := NewServer(ServerConfig{SocketPath: socketPath, Registry: NewRegistry(DaemonName, "test")})

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	done := make(chan error, 1)
	go func() { done <- server.ListenAndServe(ctx) }()
	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("ListenAndServe(cancelled ctx) = %v, want nil", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("ListenAndServe did not return for a pre-cancelled context")
	}
}

func TestServer_DefaultSocketPath_Good(t *testing.T) {
	path, err := DefaultSocketPath()
	if err != nil {
		t.Fatalf("DefaultSocketPath() error = %v", err)
	}
	if path == "" {
		t.Fatal("DefaultSocketPath() = empty, want a path")
	}
	if core.PathBase(path) != "violet.sock" {
		t.Fatalf("DefaultSocketPath() basename = %q, want violet.sock", core.PathBase(path))
	}
}

func TestServer_DefaultSocketPath_Bad(t *testing.T) {
	// On Linux the path derives from XDG_RUNTIME_DIR; with it unset
	// DefaultSocketPath must error. On darwin the home dir always resolves, so
	// the function cannot fail there — assert the no-error contract instead.
	if runtime.GOOS == "darwin" {
		if _, err := DefaultSocketPath(); err != nil {
			t.Fatalf("DefaultSocketPath() on darwin error = %v, want nil", err)
		}
		return
	}
	t.Setenv("XDG_RUNTIME_DIR", "")
	if _, err := DefaultSocketPath(); err == nil {
		t.Fatal("DefaultSocketPath() with XDG_RUNTIME_DIR unset returned nil error, want error")
	}
}

func TestServer_DefaultSocketPath_Ugly(t *testing.T) {
	// A weird-but-set XDG_RUNTIME_DIR (Linux) must be honoured verbatim under
	// the ofm/violet.sock suffix. On darwin XDG is ignored, so assert the
	// returned path still lands under the ofm cache dir.
	if runtime.GOOS != "darwin" {
		t.Setenv("XDG_RUNTIME_DIR", "/tmp/weird runtime")
		path, err := DefaultSocketPath()
		if err != nil {
			t.Fatalf("DefaultSocketPath() error = %v", err)
		}
		if !core.Contains(path, "/tmp/weird runtime") || core.PathBase(path) != "violet.sock" {
			t.Fatalf("DefaultSocketPath() = %q, want it under the set XDG dir", path)
		}
		return
	}
	path, err := DefaultSocketPath()
	if err != nil {
		t.Fatalf("DefaultSocketPath() error = %v", err)
	}
	if !core.Contains(path, "ofm") || core.PathBase(path) != "violet.sock" {
		t.Fatalf("DefaultSocketPath() = %q, want an ofm/violet.sock path", path)
	}
}

// SPDX-Licence-Identifier: EUPL-1.2

package daemon

import (
	"bufio"
	"context"
	"net"
	"runtime"
	"sync"
	"syscall"

	core "dappco.re/go"
)

const (
	socketFileMode core.FileMode = 0o600
	socketDirMode  core.FileMode = 0o700
	maxFrameBytes                = 16 * 1024 * 1024
)

type ServerConfig struct {
	SocketPath string
	Registry   *Registry

	// ModelPaths is populated from config/env by cmd/violet. Violet is one
	// process for multiple configured models; actual model loading is a follow-up
	// and should load once at startup, with restart as the swap mechanism.
	ModelPaths map[string]string
}

type Server struct {
	SocketPath string
	Registry   *Registry
	ModelPaths map[string]string
}

type errorResponse struct {
	Status  string `json:"status"`
	Error   string `json:"error"`
	Message string `json:"message,omitempty"`
}

func NewServer(cfg ServerConfig) *Server {
	modelPaths := make(map[string]string, len(cfg.ModelPaths))
	for name, path := range cfg.ModelPaths {
		modelPaths[name] = path
	}

	if cfg.Registry == nil {
		cfg.Registry = DefaultRegistryForDaemon()
	}

	return &Server{
		SocketPath: cfg.SocketPath,
		Registry:   cfg.Registry,
		ModelPaths: modelPaths,
	}
}

func (s *Server) ListenAndServe(ctx context.Context) error {
	if ctx == nil {
		ctx = context.Background()
	}
	if s.Registry == nil {
		s.Registry = DefaultRegistryForDaemon()
	}

	socketPath, err := s.resolvedSocketPath()
	if err != nil {
		return err
	}
	if err := prepareSocketPath(socketPath); err != nil {
		return err
	}

	ln, err := net.Listen("unix", socketPath)
	if err != nil {
		return core.Errorf("listen unix %s: %w", socketPath, err)
	}
	if err := chmod(socketPath, socketFileMode); err != nil {
		err = core.ErrorJoin(err, ln.Close(), removePath(socketPath))
		return core.Errorf("chmod socket %s: %w", socketPath, err)
	}

	s.SocketPath = socketPath
	defer func() {
		if err := ln.Close(); err != nil && !core.Is(err, net.ErrClosed) {
			core.Print(core.Stderr(), "violet daemon: close listener: %v", err)
		}
		if err := removePath(socketPath); err != nil && !core.IsNotExist(err) {
			core.Print(core.Stderr(), "violet daemon: remove socket: %v", err)
		}
	}()

	return s.serve(ctx, ln)
}

func (s *Server) serve(ctx context.Context, ln net.Listener) error {
	var wg sync.WaitGroup
	var conns sync.Map
	done := make(chan struct{})

	go func() {
		select {
		case <-ctx.Done():
			if err := ln.Close(); err != nil && !core.Is(err, net.ErrClosed) {
				core.Print(core.Stderr(), "violet daemon: close listener: %v", err)
			}
			conns.Range(func(key, _ any) bool {
				if err := key.(net.Conn).Close(); err != nil && !core.Is(err, net.ErrClosed) {
					core.Print(core.Stderr(), "violet daemon: close connection: %v", err)
				}
				return true
			})
		case <-done:
		}
	}()

	defer func() {
		close(done)
		wg.Wait()
	}()

	for {
		conn, err := ln.Accept()
		if err != nil {
			if ctx.Err() != nil || core.Is(err, net.ErrClosed) {
				return nil
			}
			return core.Errorf("accept unix connection: %w", err)
		}

		conns.Store(conn, struct{}{})
		wg.Add(1)
		go func(conn net.Conn) {
			defer wg.Done()
			defer conns.Delete(conn)
			if err := s.handleConn(ctx, conn); err != nil {
				core.Print(core.Stderr(), "violet daemon: handle connection: %v", err)
			}
		}(conn)
	}
}

func (s *Server) handleConn(ctx context.Context, conn net.Conn) error {
	defer conn.Close()

	scanner := bufio.NewScanner(conn)
	scanner.Buffer(make([]byte, 0, 64*1024), maxFrameBytes)

	for scanner.Scan() {
		if ctx.Err() != nil {
			return nil
		}

		line := core.Trim(string(scanner.Bytes()))
		if line == "" {
			continue
		}

		var req Request
		if result := core.JSONUnmarshalString(line, &req); !result.OK {
			if encodeErr := writeJSONLine(conn, errorResponse{
				Status:  "error",
				Error:   "invalid_json",
				Message: daemonResultError(result).Error(),
			}); encodeErr != nil {
				return encodeErr
			}
			continue
		}

		resp, err := s.Registry.Dispatch(ctx, req)
		if err != nil {
			if encodeErr := writeJSONLine(conn, errorResponse{
				Status:  "error",
				Error:   "dispatch_error",
				Message: err.Error(),
			}); encodeErr != nil {
				return encodeErr
			}
			continue
		}

		if err := writeJSONLine(conn, resp); err != nil {
			return err
		}
	}

	if err := scanner.Err(); err != nil && ctx.Err() == nil {
		return err
	}
	return nil
}

func (s *Server) resolvedSocketPath() (string, error) {
	if s != nil && s.SocketPath != "" {
		return s.SocketPath, nil
	}
	return DefaultSocketPath()
}

func DefaultSocketPath() (string, error) {
	if runtime.GOOS == "darwin" {
		home := core.UserHomeDir()
		if !home.OK {
			return "", core.Errorf("resolve home directory: %w", daemonResultError(home))
		}
		return core.PathJoin(home.Value.(string), "Library", "Caches", "ofm", "violet.sock"), nil
	}

	runtimeDir := core.Getenv("XDG_RUNTIME_DIR")
	if runtimeDir == "" {
		return "", core.NewError("XDG_RUNTIME_DIR is not set")
	}
	return core.PathJoin(runtimeDir, "ofm", "violet.sock"), nil
}

func prepareSocketPath(socketPath string) error {
	if socketPath == "" {
		return core.NewError("socket path is required")
	}
	if r := core.MkdirAll(core.PathDir(socketPath), socketDirMode); !r.OK {
		return core.Errorf("create socket directory: %w", daemonResultError(r))
	}

	infoResult := core.Lstat(socketPath)
	if !infoResult.OK && core.IsNotExist(daemonResultError(infoResult)) {
		return nil
	}
	if !infoResult.OK {
		return core.Errorf("stat socket path: %w", daemonResultError(infoResult))
	}
	info := infoResult.Value.(core.FsFileInfo)
	if info.Mode()&core.ModeSocket == 0 {
		return core.Errorf("refusing to replace non-socket path %s", socketPath)
	}
	if err := removePath(socketPath); err != nil {
		return core.Errorf("remove stale socket %s: %w", socketPath, err)
	}
	return nil
}

func writeJSONLine(w core.Writer, value any) error {
	encoded := core.JSONMarshal(value)
	if !encoded.OK {
		return daemonResultError(encoded)
	}
	if written := core.WriteString(w, string(encoded.Value.([]byte))+"\n"); !written.OK {
		return daemonResultError(written)
	}
	return nil
}

func removePath(path string) error {
	if result := core.Remove(path); !result.OK {
		return daemonResultError(result)
	}
	return nil
}

func chmod(path string, mode core.FileMode) error {
	return syscall.Chmod(path, uint32(mode.Perm()))
}

func daemonResultError(result core.Result) error {
	if err, ok := result.Value.(error); ok {
		return err
	}
	return core.NewError("daemon operation failed")
}

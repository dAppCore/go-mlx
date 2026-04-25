package daemon

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"os"
	"path/filepath"
	"runtime"
	"sync"
)

const (
	socketFileMode os.FileMode = 0o600
	socketDirMode  os.FileMode = 0o700
	maxFrameBytes              = 16 * 1024 * 1024
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
		return fmt.Errorf("listen unix %s: %w", socketPath, err)
	}
	if err := os.Chmod(socketPath, socketFileMode); err != nil {
		_ = ln.Close()
		_ = os.Remove(socketPath)
		return fmt.Errorf("chmod socket %s: %w", socketPath, err)
	}

	s.SocketPath = socketPath
	defer func() {
		_ = ln.Close()
		_ = os.Remove(socketPath)
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
			_ = ln.Close()
			conns.Range(func(key, _ any) bool {
				_ = key.(net.Conn).Close()
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
			if ctx.Err() != nil || errors.Is(err, net.ErrClosed) {
				return nil
			}
			return fmt.Errorf("accept unix connection: %w", err)
		}

		conns.Store(conn, struct{}{})
		wg.Add(1)
		go func(conn net.Conn) {
			defer wg.Done()
			defer conns.Delete(conn)
			_ = s.handleConn(ctx, conn)
		}(conn)
	}
}

func (s *Server) handleConn(ctx context.Context, conn net.Conn) error {
	defer conn.Close()

	scanner := bufio.NewScanner(conn)
	scanner.Buffer(make([]byte, 0, 64*1024), maxFrameBytes)
	encoder := json.NewEncoder(conn)

	for scanner.Scan() {
		if ctx.Err() != nil {
			return nil
		}

		line := bytes.TrimSpace(scanner.Bytes())
		if len(line) == 0 {
			continue
		}

		var req Request
		if err := json.Unmarshal(line, &req); err != nil {
			if encodeErr := encoder.Encode(errorResponse{
				Status:  "error",
				Error:   "invalid_json",
				Message: err.Error(),
			}); encodeErr != nil {
				return encodeErr
			}
			continue
		}

		resp, err := s.Registry.Dispatch(ctx, req)
		if err != nil {
			if encodeErr := encoder.Encode(errorResponse{
				Status:  "error",
				Error:   "dispatch_error",
				Message: err.Error(),
			}); encodeErr != nil {
				return encodeErr
			}
			continue
		}

		if err := encoder.Encode(resp); err != nil {
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
		home, err := os.UserHomeDir()
		if err != nil {
			return "", fmt.Errorf("resolve home directory: %w", err)
		}
		return filepath.Join(home, "Library", "Caches", "ofm", "violet.sock"), nil
	}

	runtimeDir := os.Getenv("XDG_RUNTIME_DIR")
	if runtimeDir == "" {
		return "", errors.New("XDG_RUNTIME_DIR is not set")
	}
	return filepath.Join(runtimeDir, "ofm", "violet.sock"), nil
}

func prepareSocketPath(socketPath string) error {
	if socketPath == "" {
		return errors.New("socket path is required")
	}
	if err := os.MkdirAll(filepath.Dir(socketPath), socketDirMode); err != nil {
		return fmt.Errorf("create socket directory: %w", err)
	}

	info, err := os.Lstat(socketPath)
	if errors.Is(err, os.ErrNotExist) {
		return nil
	}
	if err != nil {
		return fmt.Errorf("stat socket path: %w", err)
	}
	if info.Mode()&os.ModeSocket == 0 {
		return fmt.Errorf("refusing to replace non-socket path %s", socketPath)
	}
	if err := os.Remove(socketPath); err != nil {
		return fmt.Errorf("remove stale socket %s: %w", socketPath, err)
	}
	return nil
}

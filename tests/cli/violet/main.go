// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"time"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run() error {
	root, err := repoRoot()
	if err != nil {
		return err
	}

	tmpDir, err := os.MkdirTemp("/tmp", "violet-cli-*")
	if err != nil {
		return err
	}
	defer os.RemoveAll(tmpDir)

	binary := ""
	if len(os.Args) > 1 {
		binary = os.Args[1]
	} else {
		binary = filepath.Join(tmpDir, "violet")
		if err := buildBinary(root, binary); err != nil {
			return err
		}
	}

	socketPath := filepath.Join(tmpDir, "runtime", "ofm", "violet.sock")
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	cmd := exec.CommandContext(ctx, binary, "--socket", socketPath)
	cmd.Dir = root
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Start(); err != nil {
		return fmt.Errorf("start violet: %w", err)
	}
	defer func() {
		cancel()
		if err := cmd.Wait(); err != nil && ctx.Err() == nil {
			fmt.Fprintf(os.Stderr, "wait violet: %v\n", err)
		}
	}()

	if err := waitForSocket(socketPath); err != nil {
		return fmt.Errorf("%w\nstdout:\n%s\nstderr:\n%s", err, stdout.String(), stderr.String())
	}

	conn, err := net.DialTimeout("unix", socketPath, time.Second)
	if err != nil {
		return fmt.Errorf("dial violet socket: %w", err)
	}
	defer conn.Close()

	if _, err := fmt.Fprintln(conn, `{"action":"info"}`); err != nil {
		return fmt.Errorf("write info frame: %w", err)
	}

	line, err := bufio.NewReader(conn).ReadBytes('\n')
	if err != nil {
		return fmt.Errorf("read info response: %w", err)
	}

	var resp struct {
		Name    string   `json:"name"`
		Version string   `json:"version"`
		Actions []string `json:"actions"`
	}
	if err := json.Unmarshal(line, &resp); err != nil {
		return fmt.Errorf("decode info response %q: %w", string(line), err)
	}
	if resp.Name != "violet" {
		return fmt.Errorf("name = %q, want violet", resp.Name)
	}
	if resp.Version == "" {
		return fmt.Errorf("version is empty")
	}
	if !contains(resp.Actions, "info") {
		return fmt.Errorf("actions = %v, want info", resp.Actions)
	}

	return nil
}

func buildBinary(root, binary string) error {
	cmd := exec.Command("go", "build", "-o", binary, "./cmd/violet")
	cmd.Dir = root
	var output bytes.Buffer
	cmd.Stdout = &output
	cmd.Stderr = &output
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("build violet: %w\n%s", err, output.String())
	}
	return nil
}

func repoRoot() (string, error) {
	dir, err := os.Getwd()
	if err != nil {
		return "", err
	}
	for {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir, nil
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			return "", fmt.Errorf("could not find repo root from %s", dir)
		}
		dir = parent
	}
}

func waitForSocket(socketPath string) error {
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		info, err := os.Lstat(socketPath)
		if err == nil && info.Mode()&os.ModeSocket != 0 {
			return nil
		}
		time.Sleep(20 * time.Millisecond)
	}
	return fmt.Errorf("socket %s was not created", socketPath)
}

func contains(values []string, want string) bool {
	for _, value := range values {
		if value == want {
			return true
		}
	}
	return false
}

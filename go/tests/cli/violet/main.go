// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"bufio"
	"context"
	"net"
	"syscall"
	"time"

	core "dappco.re/go"
)

type textBuffer interface {
	core.Writer
	String() string
}

type cliprocess struct {
	pid      int
	done     chan error
	finished chan struct{}
}

func main() {
	if err := run(); err != nil {
		core.Print(core.Stderr(), "%v", err)
		core.Exit(1)
	}
}

func run() error {
	root, err := repoRoot()
	if err != nil {
		return err
	}

	tmpDirResult := core.MkdirTemp("/tmp", "violet-cli-*")
	if !tmpDirResult.OK {
		return tmpDirResult.Value.(error)
	}
	tmpDir := tmpDirResult.Value.(string)
	defer core.RemoveAll(tmpDir)

	binary := ""
	if len(core.Args()) > 1 {
		binary = core.Args()[1]
	} else {
		binary = core.PathJoin(tmpDir, "violet")
		if err := buildBinary(root, binary); err != nil {
			return err
		}
	}

	socketPath := core.PathJoin(tmpDir, "runtime", "ofm", "violet.sock")
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	proc, stdout, stderr, err := startCommand(ctx, root, binary, "--socket", socketPath)
	if err != nil {
		return core.Errorf("start violet: %w", err)
	}
	defer func() {
		cancel()
		if err := proc.Wait(); err != nil && ctx.Err() == nil {
			core.Print(core.Stderr(), "wait violet: %v", err)
		}
	}()

	if err := waitForSocket(socketPath); err != nil {
		return core.Errorf("%w\nstdout:\n%s\nstderr:\n%s", err, stdout.String(), stderr.String())
	}

	conn, err := net.DialTimeout("unix", socketPath, time.Second)
	if err != nil {
		return core.Errorf("dial violet socket: %w", err)
	}
	defer conn.Close()

	if result := core.WriteString(conn, `{"action":"info"}`+"\n"); !result.OK {
		return core.Errorf("write info frame: %w", result.Value.(error))
	}

	line, err := bufio.NewReader(conn).ReadBytes('\n')
	if err != nil {
		return core.Errorf("read info response: %w", err)
	}

	var resp struct {
		Name    string   `json:"name"`
		Version string   `json:"version"`
		Actions []string `json:"actions"`
	}
	if result := core.JSONUnmarshal(line, &resp); !result.OK {
		return core.Errorf("decode info response %q: %w", string(line), result.Value.(error))
	}
	if resp.Name != "violet" {
		return core.Errorf("name = %q, want violet", resp.Name)
	}
	if resp.Version == "" {
		return core.NewError("version is empty")
	}
	if !contains(resp.Actions, "info") {
		return core.Errorf("actions = %v, want info", resp.Actions)
	}

	return nil
}

func buildBinary(root, binary string) error {
	output, err := runCommand(root, "go", "build", "-o", binary, "./cmd/violet")
	if err != nil {
		return core.Errorf("build violet: %w\n%s", err, output)
	}
	return nil
}

func repoRoot() (string, error) {
	dirResult := core.Getwd()
	if !dirResult.OK {
		return "", dirResult.Value.(error)
	}
	dir := dirResult.Value.(string)
	for {
		if core.Stat(core.PathJoin(dir, "go.mod")).OK {
			return dir, nil
		}
		parent := core.PathDir(dir)
		if parent == dir {
			return "", core.Errorf("could not find repo root from %s", dir)
		}
		dir = parent
	}
}

func waitForSocket(socketPath string) error {
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		info := core.Lstat(socketPath)
		if info.OK && info.Value.(core.FsFileInfo).Mode()&core.ModeSocket != 0 {
			return nil
		}
		time.Sleep(20 * time.Millisecond)
	}
	return core.Errorf("socket %s was not created", socketPath)
}

func contains(values []string, want string) bool {
	for _, value := range values {
		if value == want {
			return true
		}
	}
	return false
}

func runCommand(dir, command string, args ...string) (string, error) {
	ctx := context.Background()
	proc, stdout, stderr, err := startCommand(ctx, dir, command, args...)
	if err != nil {
		return "", err
	}
	err = proc.Wait()
	return stdout.String() + stderr.String(), err
}

func startCommand(ctx context.Context, dir, command string, args ...string) (*cliprocess, textBuffer, textBuffer, error) {
	path, err := lookPath(command)
	if err != nil {
		return nil, nil, nil, err
	}

	stdoutPipe := []int{0, 0}
	if err := syscall.Pipe(stdoutPipe); err != nil {
		return nil, nil, nil, err
	}
	stderrPipe := []int{0, 0}
	if err := syscall.Pipe(stderrPipe); err != nil {
		closeFDs(stdoutPipe...)
		return nil, nil, nil, err
	}
	syscall.CloseOnExec(stdoutPipe[0])
	syscall.CloseOnExec(stdoutPipe[1])
	syscall.CloseOnExec(stderrPipe[0])
	syscall.CloseOnExec(stderrPipe[1])

	argv := append([]string{command}, args...)
	pid, err := syscall.ForkExec(path, argv, &syscall.ProcAttr{
		Dir:   dir,
		Env:   core.Environ(),
		Files: []uintptr{0, uintptr(stdoutPipe[1]), uintptr(stderrPipe[1])},
	})
	err = core.ErrorJoin(err, closeFDs(stdoutPipe[1], stderrPipe[1]))
	if err != nil {
		closeFDs(stdoutPipe[0], stderrPipe[0])
		return nil, nil, nil, err
	}

	stdout := core.NewBuffer()
	stderr := core.NewBuffer()
	go readFD(stdoutPipe[0], stdout)
	go readFD(stderrPipe[0], stderr)

	proc := &cliprocess{pid: pid, done: make(chan error, 1), finished: make(chan struct{})}
	go func() {
		var status syscall.WaitStatus
		_, waitErr := syscall.Wait4(pid, &status, 0, nil)
		if waitErr == nil && (!status.Exited() || status.ExitStatus() != 0) {
			waitErr = core.Errorf("exit status %d", status.ExitStatus())
		}
		proc.done <- waitErr
		close(proc.finished)
	}()
	go func() {
		select {
		case <-ctx.Done():
			proc.Kill()
		case <-proc.finished:
		}
	}()
	return proc, stdout, stderr, nil
}

func readFD(fd int, dst core.Writer) {
	defer syscall.Close(fd)
	buf := make([]byte, 32*1024)
	for {
		n, err := syscall.Read(fd, buf)
		if n > 0 {
			core.WriteString(dst, string(buf[:n]))
		}
		if err != nil || n == 0 {
			return
		}
	}
}

func (proc *cliprocess) Wait() error {
	if proc == nil {
		return nil
	}
	err, ok := <-proc.done
	if !ok {
		return nil
	}
	return err
}

func (proc *cliprocess) Kill() error {
	if proc == nil || proc.pid <= 0 {
		return nil
	}
	return syscall.Kill(proc.pid, syscall.SIGKILL)
}

func lookPath(command string) (string, error) {
	if core.Contains(command, string(core.PathSeparator)) {
		if executable(command) {
			return command, nil
		}
		return "", core.Errorf("executable not found: %s", command)
	}
	for _, dir := range core.Split(core.Getenv("PATH"), string(core.PathListSeparator)) {
		if dir == "" {
			dir = "."
		}
		path := core.PathJoin(dir, command)
		if executable(path) {
			return path, nil
		}
	}
	return "", core.Errorf("executable not found: %s", command)
}

func executable(path string) bool {
	info := core.Stat(path)
	return info.OK && !info.Value.(core.FsFileInfo).IsDir() && info.Value.(core.FsFileInfo).Mode()&0111 != 0
}

func closeFDs(fds ...int) error {
	var err error
	for _, fd := range fds {
		if fd > 0 {
			err = core.ErrorJoin(err, syscall.Close(fd))
		}
	}
	return err
}


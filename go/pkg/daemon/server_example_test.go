// SPDX-Licence-Identifier: EUPL-1.2

package daemon

import (
	"bufio"
	"context"
	"net"
	"time"

	core "dappco.re/go"
)

// Runnable examples that invoke the public server API with deterministic output.

func ExampleNewServer() {
	server := NewServer(ServerConfig{ModelPaths: map[string]string{"default": "/models/main"}})
	core.Println(server.GenerateBackend != nil)
	// Output: true
}

func ExampleDefaultSocketPath() {
	path, err := DefaultSocketPath()
	if err != nil {
		core.Println(err)
		return
	}
	// Print only the basename — the directory is host-specific.
	core.Println(core.PathBase(path))
	// Output: violet.sock
}

func ExampleServer_ListenAndServe() {
	dir := core.MkdirTemp("", "violet-example-*")
	if !dir.OK {
		return
	}
	defer core.RemoveAll(dir.Value.(string))

	socketPath := core.PathJoin(dir.Value.(string), "violet.sock")
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	server := NewServer(ServerConfig{SocketPath: socketPath, Registry: NewRegistry("violet", "example")})
	go func() { _ = server.ListenAndServe(ctx) }()

	// Wait for the socket, then ask for info and print the daemon name.
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if info := core.Lstat(socketPath); info.OK && info.Value.(core.FsFileInfo).Mode()&core.ModeSocket != 0 {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}

	conn, err := net.DialTimeout("unix", socketPath, time.Second)
	if err != nil {
		core.Println(err)
		return
	}
	defer conn.Close()
	_ = core.WriteString(conn, `{"action":"info"}`+"\n")
	line, _ := bufio.NewReader(conn).ReadBytes('\n')
	var resp map[string]any
	_ = core.JSONUnmarshal(line, &resp)
	core.Println(resp["name"])
	// Output: violet
}

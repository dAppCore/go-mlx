// SPDX-Licence-Identifier: EUPL-1.2

package daemon

import core "dappco.re/go"

func TestAX7_DefaultRegistryForDaemon_Good(t *core.T) {
	registry := DefaultRegistryForDaemon()

	core.AssertNotNil(t, registry)
	core.AssertEqual(t, []string{"embed", "score", "generate", "info"}, registry.Actions())
}

func TestAX7_DefaultRegistryForDaemon_Bad(t *core.T) {
	registry := DefaultRegistryForDaemon()
	resp, err := registry.Dispatch(core.Background(), Request{Action: "missing"})

	core.AssertError(t, err)
	core.AssertNil(t, resp)
}

func TestAX7_DefaultRegistryForDaemon_Ugly(t *core.T) {
	first := DefaultRegistryForDaemon()
	second := DefaultRegistryForDaemon()

	core.AssertNotEqual(t, first, second)
	core.AssertEqual(t, first.Actions(), second.Actions())
}

func TestAX7_DefaultSocketPath_Good(t *core.T) {
	path, err := DefaultSocketPath()

	core.AssertNoError(t, err)
	core.AssertContains(t, path, "violet.sock")
}

func TestAX7_DefaultSocketPath_Bad(t *core.T) {
	path, err := DefaultSocketPath()

	core.AssertTrue(t, err == nil || path == "")
	core.AssertTrue(t, err != nil || core.HasSuffix(path, "violet.sock"))
}

func TestAX7_DefaultSocketPath_Ugly(t *core.T) {
	path, err := DefaultSocketPath()

	core.AssertTrue(t, err != nil || core.PathBase(path) == "violet.sock")
	core.AssertTrue(t, err == nil || path == "")
}

func TestAX7_NewRegistry_Good(t *core.T) {
	registry := NewRegistry("agent", "1.0.0")

	core.AssertNotNil(t, registry)
	core.AssertEqual(t, []string{"embed", "score", "generate", "info"}, registry.Actions())
}

func TestAX7_NewRegistry_Bad(t *core.T) {
	registry := NewRegistry("", "")
	resp, err := registry.Dispatch(core.Background(), Request{Action: "embed"})

	core.AssertNoError(t, err)
	core.AssertEqual(t, "stub", resp["status"])
}

func TestAX7_NewRegistry_Ugly(t *core.T) {
	registry := NewRegistry("  ", "  ")
	resp, err := registry.Dispatch(core.Background(), Request{Action: " info "})

	core.AssertNoError(t, err)
	core.AssertEqual(t, []string{"embed", "score", "generate", "info"}, resp["actions"])
}

func TestAX7_NewServer_Good(t *core.T) {
	registry := NewRegistry("agent", "1.0.0")
	server := NewServer(ServerConfig{SocketPath: core.Path(t.TempDir(), "violet.sock"), Registry: registry})

	core.AssertEqual(t, registry, server.Registry)
	core.AssertContains(t, server.SocketPath, "violet.sock")
}

func TestAX7_NewServer_Bad(t *core.T) {
	server := NewServer(ServerConfig{})

	core.AssertNotNil(t, server.Registry)
	core.AssertEmpty(t, server.ModelPaths)
}

func TestAX7_NewServer_Ugly(t *core.T) {
	input := map[string]string{"tiny": "/models/tiny"}
	server := NewServer(ServerConfig{ModelPaths: input})
	input["tiny"] = "/mutated"

	core.AssertEqual(t, "/models/tiny", server.ModelPaths["tiny"])
	core.AssertNotEqual(t, input["tiny"], server.ModelPaths["tiny"])
}

func TestAX7_Registry_Actions_Good(t *core.T) {
	registry := NewRegistry("agent", "1.0.0")

	core.AssertEqual(t, []string{"embed", "score", "generate", "info"}, registry.Actions())
	core.AssertLen(t, registry.Actions(), 4)
}

func TestAX7_Registry_Actions_Bad(t *core.T) {
	var registry *Registry

	core.AssertNil(t, registry.Actions())
	core.AssertFalse(t, registry != nil)
}

func TestAX7_Registry_Actions_Ugly(t *core.T) {
	registry := NewRegistry("agent", "1.0.0")
	actions := registry.Actions()
	actions[0] = "mutated"

	core.AssertEqual(t, "embed", registry.Actions()[0])
}

func TestAX7_Registry_Dispatch_Good(t *core.T) {
	registry := NewRegistry("agent", "1.0.0")
	resp, err := registry.Dispatch(core.Background(), Request{Action: " EMBED "})

	core.AssertNoError(t, err)
	core.AssertEqual(t, "embed", resp["action"])
}

func TestAX7_Registry_Dispatch_Bad(t *core.T) {
	registry := NewRegistry("agent", "1.0.0")
	resp, err := registry.Dispatch(core.Background(), Request{})

	core.AssertError(t, err)
	core.AssertNil(t, resp)
}

func TestAX7_Registry_Dispatch_Ugly(t *core.T) {
	var registry *Registry
	resp, err := registry.Dispatch(core.Background(), Request{Action: "info"})

	core.AssertError(t, err)
	core.AssertNil(t, resp)
}

func TestAX7_Registry_Register_Good(t *core.T) {
	registry := NewRegistry("agent", "1.0.0")
	err := registry.Register(" custom.action ", func(core.Context, Request) (Response, error) {
		return Response{"ok": true}, nil
	})

	core.AssertNoError(t, err)
	core.AssertContains(t, registry.Actions(), "custom.action")
}

func TestAX7_Registry_Register_Bad(t *core.T) {
	registry := NewRegistry("agent", "1.0.0")
	err := registry.Register("", func(core.Context, Request) (Response, error) {
		return Response{}, nil
	})

	core.AssertError(t, err)
	core.AssertFalse(t, registry.Actions()[0] == "")
}

func TestAX7_Registry_Register_Ugly(t *core.T) {
	registry := NewRegistry("agent", "1.0.0")
	err := registry.Register("custom.action", nil)

	core.AssertError(t, err)
	core.AssertNotContains(t, registry.Actions(), "custom.action")
}

func TestAX7_Server_ListenAndServe_Good(t *core.T) {
	ctx, cancel := core.WithCancel(core.Background())
	cancel()
	server := NewServer(ServerConfig{SocketPath: "/tmp/v090-daemon-ax7-good.sock"})

	err := server.ListenAndServe(ctx)

	core.AssertNoError(t, err)
	core.AssertNotNil(t, server.Registry)
}

func TestAX7_Server_ListenAndServe_Bad(t *core.T) {
	server := NewServer(ServerConfig{SocketPath: "/tmp/v090-daemon-ax7-bad.sock"})
	ctx, cancel := core.WithCancel(core.Background())
	cancel()

	err := server.ListenAndServe(ctx)

	core.AssertNoError(t, err)
	core.AssertNotNil(t, server.Registry)
}

func TestAX7_Server_ListenAndServe_Ugly(t *core.T) {
	path := "/tmp/v090-daemon-ax7-not-a-socket"
	result := core.WriteFile(path, []byte("occupied"), 0o600)
	core.RequireTrue(t, result.OK)
	server := NewServer(ServerConfig{SocketPath: path})

	err := server.ListenAndServe(core.Background())

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "non-socket")
}

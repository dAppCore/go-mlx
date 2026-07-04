// SPDX-Licence-Identifier: EUPL-1.2

package daemon

import (
	"context"

	core "dappco.re/go"
)

// Runnable examples that invoke the public registry API with deterministic output.

func ExampleNewRegistry() {
	registry := NewRegistry("violet", "1.0")
	resp, _ := registry.Dispatch(context.Background(), Request{Action: "info"})
	core.Println(resp["version"])
	// Output: 1.0
}

func ExampleDefaultRegistryForDaemon() {
	registry := DefaultRegistryForDaemon()
	resp, _ := registry.Dispatch(context.Background(), Request{Action: "info"})
	core.Println(resp["name"])
	// Output: violet
}

func ExampleRegistry_Register() {
	registry := NewRegistry("violet", "test")
	_ = registry.Register("ping", func(context.Context, Request) (Response, error) {
		return Response{"status": "pong"}, nil
	})
	resp, _ := registry.Dispatch(context.Background(), Request{Action: "ping"})
	core.Println(resp["status"])
	// Output: pong
}

func ExampleRegistry_RegisterGenerateBackend() {
	registry := NewRegistry("violet", "test")
	_ = registry.RegisterGenerateBackend(exampleBackend{})
	resp, _ := registry.Dispatch(context.Background(), Request{Action: "generate", Prompt: "hi"})
	core.Println(resp["text"])
	// Output: hello from backend
}

func ExampleRegistry_Dispatch() {
	registry := NewRegistry("violet", "test")
	resp, _ := registry.Dispatch(context.Background(), Request{Action: "embed"})
	core.Println(resp["status"])
	// Output: stub
}

func ExampleRegistry_Actions() {
	registry := NewRegistry("violet", "test")
	core.Println(len(registry.Actions()))
	// Output: 4
}

type exampleBackend struct{}

func (exampleBackend) Generate(context.Context, GenerateRequest) (GenerateResult, error) {
	return GenerateResult{Text: "hello from backend"}, nil
}

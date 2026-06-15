// SPDX-Licence-Identifier: EUPL-1.2

package daemon

import (
	"context"
	"testing"

	core "dappco.re/go"
)

type fakeGenerateBackend struct {
	request GenerateRequest
	result  GenerateResult
	err     error
}

func (backend *fakeGenerateBackend) Generate(_ context.Context, request GenerateRequest) (GenerateResult, error) {
	backend.request = request
	return backend.result, backend.err
}

func TestRegistry_RegisterGenerateBackend_Good_DispatchesGenerate(t *testing.T) {
	backend := &fakeGenerateBackend{
		result: GenerateResult{
			Text:  "pong",
			Model: "main",
			Metrics: GenerateMetrics{
				PromptTokens:    4,
				GeneratedTokens: 1,
			},
		},
	}
	registry := NewRegistry(DaemonName, "test")
	if err := registry.RegisterGenerateBackend(backend); err != nil {
		t.Fatalf("RegisterGenerateBackend() error = %v", err)
	}

	resp, err := registry.Dispatch(context.Background(), Request{
		Action:      "generate",
		Prompt:      "ping",
		Model:       "main",
		MaxTokens:   32,
		Temperature: 0.2,
	})

	if err != nil {
		t.Fatalf("Dispatch() error = %v", err)
	}
	if resp["status"] != "ok" {
		t.Fatalf("status = %v, want ok", resp["status"])
	}
	if resp["action"] != "generate" {
		t.Fatalf("action = %v, want generate", resp["action"])
	}
	if resp["text"] != "pong" {
		t.Fatalf("text = %v, want pong", resp["text"])
	}
	if resp["model"] != "main" {
		t.Fatalf("model = %v, want main", resp["model"])
	}
	if backend.request.Prompt != "ping" {
		t.Fatalf("backend prompt = %q, want ping", backend.request.Prompt)
	}
	if backend.request.MaxTokens != 32 {
		t.Fatalf("backend max tokens = %d, want 32", backend.request.MaxTokens)
	}
	if backend.request.Temperature != 0.2 {
		t.Fatalf("backend temperature = %f, want 0.2", backend.request.Temperature)
	}
	if _, ok := resp["metrics"].(GenerateMetrics); !ok {
		t.Fatalf("metrics = %#v, want GenerateMetrics", resp["metrics"])
	}
}

func TestRegistry_RegisterGenerateBackend_Bad_NilBackend(t *testing.T) {
	registry := NewRegistry(DaemonName, "test")

	err := registry.RegisterGenerateBackend(nil)

	if err == nil {
		t.Fatal("RegisterGenerateBackend(nil) returned nil, want error")
	}
	if !core.Contains(err.Error(), "generate backend is nil") {
		t.Fatalf("error = %v, want nil backend", err)
	}
}

func TestRegistry_RegisterGenerateBackend_Ugly_TextFallback(t *testing.T) {
	backend := &fakeGenerateBackend{result: GenerateResult{Text: "ok"}}
	registry := NewRegistry(DaemonName, "test")
	if err := registry.RegisterGenerateBackend(backend); err != nil {
		t.Fatalf("RegisterGenerateBackend() error = %v", err)
	}

	_, err := registry.Dispatch(context.Background(), Request{Action: "generate", Text: "fallback"})

	if err != nil {
		t.Fatalf("Dispatch() error = %v", err)
	}
	if backend.request.Prompt != "fallback" {
		t.Fatalf("backend prompt = %q, want fallback", backend.request.Prompt)
	}
}

func TestRegistry_NewRegistry_Good_DefaultsAndBuiltinActions(t *testing.T) {
	registry := NewRegistry("", "")

	resp, err := registry.Dispatch(context.Background(), Request{Action: "info"})
	if err != nil {
		t.Fatalf("Dispatch(info) error = %v", err)
	}
	if resp["name"] != DaemonName {
		t.Fatalf("name = %v, want %s (empty name should default)", resp["name"], DaemonName)
	}
	if resp["version"] != DefaultVersion {
		t.Fatalf("version = %v, want %s (empty version should default)", resp["version"], DefaultVersion)
	}

	actions := registry.Actions()
	for _, want := range []string{"embed", "score", "generate", "info"} {
		var found bool
		for _, got := range actions {
			if got == want {
				found = true
				break
			}
		}
		if !found {
			t.Fatalf("built-in action %q missing from %v", want, actions)
		}
	}
}

func TestRegistry_Dispatch_Good_StubAction(t *testing.T) {
	registry := NewRegistry(DaemonName, "test")

	resp, err := registry.Dispatch(context.Background(), Request{Action: "  EMBED  "})
	if err != nil {
		t.Fatalf("Dispatch(embed) error = %v", err)
	}
	if resp["status"] != "stub" {
		t.Fatalf("status = %v, want stub", resp["status"])
	}
	if resp["action"] != "embed" {
		t.Fatalf("action = %v, want embed (should normalise case+space)", resp["action"])
	}
}

func TestRegistry_Dispatch_Bad_NilRegistryAndEmptyAction(t *testing.T) {
	var nilRegistry *Registry
	if _, err := nilRegistry.Dispatch(context.Background(), Request{Action: "info"}); err == nil {
		t.Fatal("Dispatch on nil registry returned nil, want error")
	} else if !core.Contains(err.Error(), "registry is nil") {
		t.Fatalf("error = %v, want registry is nil", err)
	}

	registry := NewRegistry(DaemonName, "test")
	if _, err := registry.Dispatch(context.Background(), Request{Action: "   "}); err == nil {
		t.Fatal("Dispatch with blank action returned nil, want error")
	} else if !core.Contains(err.Error(), "action is required") {
		t.Fatalf("error = %v, want action is required", err)
	}
}

func TestRegistry_Dispatch_Ugly_UnsupportedAction(t *testing.T) {
	registry := NewRegistry(DaemonName, "test")

	_, err := registry.Dispatch(context.Background(), Request{Action: "teleport"})
	if err == nil {
		t.Fatal("Dispatch(teleport) returned nil, want unsupported error")
	}
	if !core.Contains(err.Error(), "unsupported action") {
		t.Fatalf("error = %v, want unsupported action", err)
	}
}

func TestRegistry_Register_Good_AddsAndReplaces(t *testing.T) {
	registry := NewRegistry(DaemonName, "test")
	before := len(registry.Actions())

	called := 0
	if err := registry.Register("custom", func(context.Context, Request) (Response, error) {
		called++
		return Response{"status": "custom-v1"}, nil
	}); err != nil {
		t.Fatalf("Register(custom) error = %v", err)
	}
	if got := len(registry.Actions()); got != before+1 {
		t.Fatalf("action count = %d, want %d after new register", got, before+1)
	}

	// Replacement register: same action keeps the order length but swaps the handler.
	if err := registry.Register("custom", func(context.Context, Request) (Response, error) {
		return Response{"status": "custom-v2"}, nil
	}); err != nil {
		t.Fatalf("Register(custom replace) error = %v", err)
	}
	if got := len(registry.Actions()); got != before+1 {
		t.Fatalf("action count = %d, want %d after replace (no duplicate)", got, before+1)
	}

	resp, err := registry.Dispatch(context.Background(), Request{Action: "custom"})
	if err != nil {
		t.Fatalf("Dispatch(custom) error = %v", err)
	}
	if resp["status"] != "custom-v2" {
		t.Fatalf("status = %v, want custom-v2 (replacement handler should win)", resp["status"])
	}
	if called != 0 {
		t.Fatalf("v1 handler called %d times, want 0 after replacement", called)
	}
}

func TestRegistry_Register_Bad_EmptyActionAndNilHandler(t *testing.T) {
	registry := NewRegistry(DaemonName, "test")

	if err := registry.Register("   ", func(context.Context, Request) (Response, error) { return nil, nil }); err == nil {
		t.Fatal("Register(blank action) returned nil, want error")
	} else if !core.Contains(err.Error(), "action is required") {
		t.Fatalf("error = %v, want action is required", err)
	}

	if err := registry.Register("custom", nil); err == nil {
		t.Fatal("Register(nil handler) returned nil, want error")
	} else if !core.Contains(err.Error(), "is nil") {
		t.Fatalf("error = %v, want handler nil error", err)
	}
}

func TestRegistry_Actions_Good_ReturnsCopy(t *testing.T) {
	registry := NewRegistry(DaemonName, "test")

	actions := registry.Actions()
	if len(actions) == 0 {
		t.Fatal("Actions() = empty, want built-in actions")
	}
	// Mutating the returned slice must not corrupt the registry's order.
	actions[0] = "MUTATED"

	again := registry.Actions()
	if again[0] == "MUTATED" {
		t.Fatal("Actions() returned an aliased slice; caller mutation leaked into registry")
	}

	var nilRegistry *Registry
	if got := nilRegistry.Actions(); got != nil {
		t.Fatalf("nil registry Actions() = %v, want nil", got)
	}
}

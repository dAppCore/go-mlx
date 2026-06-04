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

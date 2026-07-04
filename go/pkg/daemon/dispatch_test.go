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

// --- v0.9.0 canonical AX-7 triplets (file prefix: Dispatch) -----------------
// Each Test<Dispatch>_<Symbol>_{Good,Bad,Ugly} names its symbol directly and
// exercises a distinct case. The scenario-suffixed tests above remain the
// detailed coverage; these are the canonical-shape entry points.

func TestDispatch_NewRegistry_Good(t *testing.T) {
	registry := NewRegistry("violet", "9.0")
	if registry == nil {
		t.Fatal("NewRegistry() = nil, want registry")
	}
	resp, err := registry.Dispatch(context.Background(), Request{Action: "info"})
	if err != nil {
		t.Fatalf("Dispatch(info) error = %v", err)
	}
	if resp["name"] != "violet" {
		t.Fatalf("name = %v, want violet", resp["name"])
	}
	if resp["version"] != "9.0" {
		t.Fatalf("version = %v, want 9.0", resp["version"])
	}
}

func TestDispatch_NewRegistry_Bad(t *testing.T) {
	// Blank name and version are not an error path — NewRegistry must fall
	// back to the package defaults rather than emit an empty info frame.
	registry := NewRegistry("", "")
	resp, err := registry.Dispatch(context.Background(), Request{Action: "info"})
	if err != nil {
		t.Fatalf("Dispatch(info) error = %v", err)
	}
	if resp["name"] != DaemonName {
		t.Fatalf("name = %v, want default %s for blank input", resp["name"], DaemonName)
	}
	if resp["version"] != DefaultVersion {
		t.Fatalf("version = %v, want default %s for blank input", resp["version"], DefaultVersion)
	}
}

func TestDispatch_NewRegistry_Ugly(t *testing.T) {
	// Whitespace-only name/version are ugly input that must still resolve to
	// the defaults — NewRegistry only treats the empty string as "unset", so a
	// space name survives verbatim while a genuinely empty version defaults.
	registry := NewRegistry(" ", "")
	resp, err := registry.Dispatch(context.Background(), Request{Action: "info"})
	if err != nil {
		t.Fatalf("Dispatch(info) error = %v", err)
	}
	if resp["name"] != " " {
		t.Fatalf("name = %q, want the literal space (NewRegistry only defaults empty)", resp["name"])
	}
	if resp["version"] != DefaultVersion {
		t.Fatalf("version = %v, want default %s", resp["version"], DefaultVersion)
	}
}

func TestDispatch_DefaultRegistryForDaemon_Good(t *testing.T) {
	registry := DefaultRegistryForDaemon()
	if registry == nil {
		t.Fatal("DefaultRegistryForDaemon() = nil, want registry")
	}
	resp, err := registry.Dispatch(context.Background(), Request{Action: "info"})
	if err != nil {
		t.Fatalf("Dispatch(info) error = %v", err)
	}
	if resp["name"] != DaemonName {
		t.Fatalf("name = %v, want %s", resp["name"], DaemonName)
	}
	if resp["version"] != DefaultVersion {
		t.Fatalf("version = %v, want %s", resp["version"], DefaultVersion)
	}
}

func TestDispatch_DefaultRegistryForDaemon_Bad(t *testing.T) {
	// There is no failure constructor, so the "bad" case is the absence of an
	// undeclared action: a default registry must reject anything outside its
	// four built-ins rather than silently answering.
	registry := DefaultRegistryForDaemon()
	if _, err := registry.Dispatch(context.Background(), Request{Action: "summon"}); err == nil {
		t.Fatal("DefaultRegistryForDaemon dispatch of unknown action returned nil, want error")
	}
}

func TestDispatch_DefaultRegistryForDaemon_Ugly(t *testing.T) {
	// Two independent DefaultRegistryForDaemon registries must not share
	// handler state — registering on one leaves the other at the stock four.
	first := DefaultRegistryForDaemon()
	second := DefaultRegistryForDaemon()
	if err := first.Register("extra", func(context.Context, Request) (Response, error) {
		return Response{"status": "extra"}, nil
	}); err != nil {
		t.Fatalf("Register on first registry error = %v", err)
	}
	if len(first.Actions()) == len(second.Actions()) {
		t.Fatalf("registries share state: first=%d second=%d", len(first.Actions()), len(second.Actions()))
	}
	if _, err := second.Dispatch(context.Background(), Request{Action: "extra"}); err == nil {
		t.Fatal("second DefaultRegistryForDaemon saw a handler registered on the first")
	}
}

func TestDispatch_Registry_Register_Good(t *testing.T) {
	registry := NewRegistry(DaemonName, "test")
	before := len(registry.Actions())
	if err := registry.Register("custom", func(context.Context, Request) (Response, error) {
		return Response{"status": "custom-ok"}, nil
	}); err != nil {
		t.Fatalf("Register(custom) error = %v", err)
	}
	if got := len(registry.Actions()); got != before+1 {
		t.Fatalf("action count = %d, want %d after Register", got, before+1)
	}
	resp, err := registry.Dispatch(context.Background(), Request{Action: "custom"})
	if err != nil {
		t.Fatalf("Dispatch(custom) error = %v", err)
	}
	if resp["status"] != "custom-ok" {
		t.Fatalf("status = %v, want custom-ok", resp["status"])
	}
}

func TestDispatch_Registry_Register_Bad(t *testing.T) {
	registry := NewRegistry(DaemonName, "test")
	if err := registry.Register("   ", func(context.Context, Request) (Response, error) { return nil, nil }); err == nil {
		t.Fatal("Register(blank action) returned nil, want error")
	} else if !core.Contains(err.Error(), "action is required") {
		t.Fatalf("Register(blank) error = %v, want action is required", err)
	}
	if err := registry.Register("custom", nil); err == nil {
		t.Fatal("Register(nil handler) returned nil, want error")
	} else if !core.Contains(err.Error(), "is nil") {
		t.Fatalf("Register(nil handler) error = %v, want handler nil", err)
	}
}

func TestDispatch_Registry_Register_Ugly(t *testing.T) {
	// Re-registering the same action must replace the handler in place without
	// growing the action order — the duplicate-key edge case.
	registry := NewRegistry(DaemonName, "test")
	if err := registry.Register("custom", func(context.Context, Request) (Response, error) {
		return Response{"status": "v1"}, nil
	}); err != nil {
		t.Fatalf("Register(custom v1) error = %v", err)
	}
	count := len(registry.Actions())
	if err := registry.Register("custom", func(context.Context, Request) (Response, error) {
		return Response{"status": "v2"}, nil
	}); err != nil {
		t.Fatalf("Register(custom v2) error = %v", err)
	}
	if got := len(registry.Actions()); got != count {
		t.Fatalf("action count = %d, want %d (replacement must not duplicate)", got, count)
	}
	resp, err := registry.Dispatch(context.Background(), Request{Action: "custom"})
	if err != nil {
		t.Fatalf("Dispatch(custom) error = %v", err)
	}
	if resp["status"] != "v2" {
		t.Fatalf("status = %v, want v2 (replacement handler should win)", resp["status"])
	}
}

func TestDispatch_Registry_RegisterGenerateBackend_Good(t *testing.T) {
	backend := &fakeGenerateBackend{result: GenerateResult{Text: "pong", Model: "main"}}
	registry := NewRegistry(DaemonName, "test")
	if err := registry.RegisterGenerateBackend(backend); err != nil {
		t.Fatalf("RegisterGenerateBackend() error = %v", err)
	}
	resp, err := registry.Dispatch(context.Background(), Request{Action: "generate", Prompt: "ping"})
	if err != nil {
		t.Fatalf("Dispatch(generate) error = %v", err)
	}
	if resp["text"] != "pong" {
		t.Fatalf("text = %v, want pong", resp["text"])
	}
	if backend.request.Prompt != "ping" {
		t.Fatalf("backend prompt = %q, want ping", backend.request.Prompt)
	}
}

func TestDispatch_Registry_RegisterGenerateBackend_Bad(t *testing.T) {
	registry := NewRegistry(DaemonName, "test")
	if err := registry.RegisterGenerateBackend(nil); err == nil {
		t.Fatal("RegisterGenerateBackend(nil) returned nil, want error")
	} else if !core.Contains(err.Error(), "generate backend is nil") {
		t.Fatalf("RegisterGenerateBackend(nil) error = %v, want nil backend", err)
	}
}

func TestDispatch_Registry_RegisterGenerateBackend_Ugly(t *testing.T) {
	// Backend that returns an error: RegisterGenerateBackend wires the handler,
	// and the wrapped dispatch must surface the backend error rather than an ok
	// frame — the failure-propagation edge.
	backend := &fakeGenerateBackend{err: core.NewError("backend exploded")}
	registry := NewRegistry(DaemonName, "test")
	if err := registry.RegisterGenerateBackend(backend); err != nil {
		t.Fatalf("RegisterGenerateBackend() error = %v", err)
	}
	if _, err := registry.Dispatch(context.Background(), Request{Action: "generate", Prompt: "boom"}); err == nil {
		t.Fatal("Dispatch via failing backend returned nil, want propagated error")
	} else if !core.Contains(err.Error(), "backend exploded") {
		t.Fatalf("error = %v, want backend exploded", err)
	}
}

func TestDispatch_Registry_Dispatch_Good(t *testing.T) {
	registry := NewRegistry(DaemonName, "test")
	resp, err := registry.Dispatch(context.Background(), Request{Action: "  EMBED  "})
	if err != nil {
		t.Fatalf("Dispatch(embed) error = %v", err)
	}
	if resp["status"] != "stub" {
		t.Fatalf("status = %v, want stub", resp["status"])
	}
	if resp["action"] != "embed" {
		t.Fatalf("action = %v, want embed (Dispatch normalises case+space)", resp["action"])
	}
}

func TestDispatch_Registry_Dispatch_Bad(t *testing.T) {
	registry := NewRegistry(DaemonName, "test")
	if _, err := registry.Dispatch(context.Background(), Request{Action: "   "}); err == nil {
		t.Fatal("Dispatch(blank action) returned nil, want error")
	} else if !core.Contains(err.Error(), "action is required") {
		t.Fatalf("Dispatch(blank) error = %v, want action is required", err)
	}
}

func TestDispatch_Registry_Dispatch_Ugly(t *testing.T) {
	// Nil receiver and an unsupported action are the two boundary cases for
	// Dispatch — both must error rather than panic or answer.
	var nilRegistry *Registry
	if _, err := nilRegistry.Dispatch(context.Background(), Request{Action: "info"}); err == nil {
		t.Fatal("Dispatch on nil registry returned nil, want error")
	} else if !core.Contains(err.Error(), "registry is nil") {
		t.Fatalf("nil Dispatch error = %v, want registry is nil", err)
	}
	registry := NewRegistry(DaemonName, "test")
	if _, err := registry.Dispatch(context.Background(), Request{Action: "teleport"}); err == nil {
		t.Fatal("Dispatch(unsupported) returned nil, want error")
	} else if !core.Contains(err.Error(), "unsupported action") {
		t.Fatalf("Dispatch(unsupported) error = %v, want unsupported action", err)
	}
}

func TestDispatch_Registry_Actions_Good(t *testing.T) {
	registry := NewRegistry(DaemonName, "test")
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
			t.Fatalf("Actions() = %v, missing built-in %q", actions, want)
		}
	}
}

func TestDispatch_Registry_Actions_Bad(t *testing.T) {
	// Not an error path: the boundary is the nil receiver, which Actions must
	// answer with a nil slice instead of panicking.
	var nilRegistry *Registry
	if got := nilRegistry.Actions(); got != nil {
		t.Fatalf("nil registry Actions() = %v, want nil", got)
	}
}

func TestDispatch_Registry_Actions_Ugly(t *testing.T) {
	// The returned slice must be a defensive copy — mutating it cannot corrupt
	// the registry's internal order on a later Actions() call.
	registry := NewRegistry(DaemonName, "test")
	first := registry.Actions()
	if len(first) == 0 {
		t.Fatal("Actions() = empty, want built-ins")
	}
	first[0] = "CLOBBERED"
	again := registry.Actions()
	if again[0] == "CLOBBERED" {
		t.Fatal("Actions() returned an aliased slice; caller mutation leaked")
	}
}

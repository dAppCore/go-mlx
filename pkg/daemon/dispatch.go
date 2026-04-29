// SPDX-Licence-Identifier: EUPL-1.2

package daemon

import (
	"context"

	core "dappco.re/go"
)

const (
	DaemonName     = "violet"
	DefaultVersion = "dev"
)

// Request is one JSON-line frame from a local Violet client.
type Request struct {
	Action string `json:"action"`
	Text   string `json:"text,omitempty"`
	Prompt string `json:"prompt,omitempty"`
	Model  string `json:"model,omitempty"`
}

// Response is encoded as one complete JSON-line frame. Streaming responses are
// intentionally deferred so the initial UDS contract stays simple.
type Response map[string]any

// Handler processes one action request.
type Handler func(context.Context, Request) (Response, error)

// Registry maps daemon actions to handlers. It preserves registration order so
// the info response is stable and human-readable.
type Registry struct {
	name     string
	version  string
	handlers map[string]Handler
	order    []string
}

func NewRegistry(name, version string) *Registry {
	if name == "" {
		name = DaemonName
	}
	if version == "" {
		version = DefaultVersion
	}

	r := &Registry{
		name:     name,
		version:  version,
		handlers: make(map[string]Handler),
	}

	if err := r.Register("embed", stubHandler("embed")); err != nil {
		panic(err)
	}
	if err := r.Register("score", stubHandler("score")); err != nil {
		panic(err)
	}
	if err := r.Register("generate", stubHandler("generate")); err != nil {
		panic(err)
	}
	if err := r.Register("info", func(context.Context, Request) (Response, error) {
		return Response{
			"name":    r.name,
			"version": r.version,
			"actions": r.Actions(),
		}, nil
	}); err != nil {
		panic(err)
	}

	return r
}

func DefaultRegistryForDaemon() *Registry {
	return NewRegistry(DaemonName, DefaultVersion)
}

func (r *Registry) Register(action string, handler Handler) error {
	action = normalizeAction(action)
	if action == "" {
		return core.NewError("action is required")
	}
	if handler == nil {
		return core.Errorf("handler for action %q is nil", action)
	}
	if r.handlers == nil {
		r.handlers = make(map[string]Handler)
	}
	if _, exists := r.handlers[action]; !exists {
		r.order = append(r.order, action)
	}
	r.handlers[action] = handler
	return nil
}

func (r *Registry) Dispatch(ctx context.Context, req Request) (Response, error) {
	if r == nil {
		return nil, core.NewError("registry is nil")
	}

	action := normalizeAction(req.Action)
	if action == "" {
		return nil, core.NewError("action is required")
	}

	handler, ok := r.handlers[action]
	if !ok {
		return nil, core.Errorf("unsupported action %q", action)
	}

	req.Action = action
	return handler(ctx, req)
}

func (r *Registry) Actions() []string {
	if r == nil {
		return nil
	}
	actions := make([]string, len(r.order))
	copy(actions, r.order)
	return actions
}

func normalizeAction(action string) string {
	return core.Lower(core.Trim(action))
}

func stubHandler(action string) Handler {
	return func(context.Context, Request) (Response, error) {
		return Response{
			"status": "stub",
			"action": action,
		}, nil
	}
}

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

var (
	errRegistryNil    = core.NewError("registry is nil")
	errActionRequired = core.NewError("action is required")
)

// Request is one JSON-line frame from a local Violet client.
type Request struct {
	Action      string    `json:"action"`
	Text        string    `json:"text,omitempty"`
	Prompt      string    `json:"prompt,omitempty"`
	Model       string    `json:"model,omitempty"`
	Messages    []Message `json:"messages,omitempty"`
	MaxTokens   int       `json:"max_tokens,omitempty"`
	Temperature float64   `json:"temperature,omitempty"`
}

// Response is encoded as one complete JSON-line frame. Streaming responses are
// intentionally deferred so the initial UDS contract stays simple.
type Response map[string]any

// Message is a chat message sent to the native generate backend.
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// GenerateRequest is the normalized input passed to a generate backend.
type GenerateRequest struct {
	Prompt      string
	Model       string
	Messages    []Message
	MaxTokens   int
	Temperature float64
}

// GenerateResult is returned by native generation backends.
type GenerateResult struct {
	Text    string
	Model   string
	Metrics GenerateMetrics
}

// GenerateMetrics are JSON-friendly counters from a backend generation call.
type GenerateMetrics struct {
	PromptTokens             int     `json:"prompt_tokens"`
	GeneratedTokens          int     `json:"generated_tokens"`
	PrefillSeconds           float64 `json:"prefill_seconds,omitempty"`
	DecodeSeconds            float64 `json:"decode_seconds,omitempty"`
	TotalSeconds             float64 `json:"total_seconds,omitempty"`
	PrefillTokensPerSec      float64 `json:"prefill_tokens_per_sec,omitempty"`
	DecodeTokensPerSec       float64 `json:"decode_tokens_per_sec,omitempty"`
	PeakMemoryBytes          uint64  `json:"peak_memory_bytes,omitempty"`
	ActiveMemoryBytes        uint64  `json:"active_memory_bytes,omitempty"`
	PromptCacheHits          int     `json:"prompt_cache_hits,omitempty"`
	PromptCacheMisses        int     `json:"prompt_cache_misses,omitempty"`
	PromptCacheHitTokens     int     `json:"prompt_cache_hit_tokens,omitempty"`
	PromptCacheMissTokens    int     `json:"prompt_cache_miss_tokens,omitempty"`
	PromptCacheRestoreMillis float64 `json:"prompt_cache_restore_ms,omitempty"`
}

// Handler processes one action request.
type Handler func(context.Context, Request) (Response, error)

// GenerateBackend handles native non-HTTP generation requests.
type GenerateBackend interface {
	Generate(context.Context, GenerateRequest) (GenerateResult, error)
}

// Registry maps daemon actions to handlers. It preserves registration order so
// the info response is stable and human-readable.
type Registry struct {
	name     string
	version  string
	handlers map[string]Handler
	order    []string
	// infoResponse caches the rendered info Response so the steady
	// state Dispatch("info") path allocates nothing. Built lazily on
	// first read after creation or any Register that invalidates it.
	// Like handlers/order, accessed without a mutex — Register is not
	// safe to call concurrently with Dispatch (existing convention).
	infoResponse Response
}

func NewRegistry(name, version string) *Registry {
	if name == "" {
		name = DaemonName
	}
	if version == "" {
		version = DefaultVersion
	}

	// Four handlers are registered immediately below; pre-sizing the
	// map and the order slice avoids the initial map/slice grow steps.
	r := &Registry{
		name:     name,
		version:  version,
		handlers: make(map[string]Handler, 4),
		order:    make([]string, 0, 4),
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
		// JSON-marshalling reads the cached map; built once when the
		// cache is empty, invalidated by Register. Steady state is
		// zero-alloc — the JSON marshal walks the same map every call.
		// JSON-marshalling a []string just iterates; no retention,
		// so the internal r.order can be returned as-is and skip the
		// defensive copy that Actions() does for external callers.
		if r.infoResponse == nil {
			r.infoResponse = Response{
				"name":    r.name,
				"version": r.version,
				"actions": r.order,
			}
		}
		return r.infoResponse, nil
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
		return errActionRequired
	}
	if handler == nil {
		return core.Errorf("handler for action %q is nil", action)
	}
	if r.handlers == nil {
		r.handlers = make(map[string]Handler)
	}
	if _, exists := r.handlers[action]; !exists {
		r.order = append(r.order, action)
		// New action in the order list invalidates the cached info
		// response. The next info dispatch rebuilds with the fresh
		// order slice. (Replacement-only registers — e.g. swapping
		// the generate stub for a real backend — leave order untouched
		// and don't need to invalidate.)
		r.infoResponse = nil
	}
	r.handlers[action] = handler
	return nil
}

// RegisterGenerateBackend replaces the default generate stub with a native backend.
func (r *Registry) RegisterGenerateBackend(backend GenerateBackend) error {
	if backend == nil {
		return core.NewError("generate backend is nil")
	}
	return r.Register("generate", func(ctx context.Context, req Request) (Response, error) {
		result, err := backend.Generate(ctx, generateRequestFromRequest(req))
		if err != nil {
			return nil, err
		}
		return generateResponseFromResult(result), nil
	})
}

func (r *Registry) Dispatch(ctx context.Context, req Request) (Response, error) {
	if r == nil {
		return nil, errRegistryNil
	}

	action := normalizeAction(req.Action)
	if action == "" {
		return nil, errActionRequired
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

func generateRequestFromRequest(req Request) GenerateRequest {
	prompt := req.Prompt
	if prompt == "" {
		prompt = req.Text
	}
	// req.Messages is owned by the Dispatch caller and is not retained
	// past backend.Generate's return (the native backend rebuilds into
	// inference.Message via toMLXMessages). Pass the slice through —
	// no defensive clone needed on the hot path.
	return GenerateRequest{
		Prompt:      prompt,
		Model:       req.Model,
		Messages:    req.Messages,
		MaxTokens:   req.MaxTokens,
		Temperature: req.Temperature,
	}
}

func generateResponseFromResult(result GenerateResult) Response {
	resp := Response{
		"status": "ok",
		"action": "generate",
		"text":   result.Text,
	}
	if result.Model != "" {
		resp["model"] = result.Model
	}
	if hasGenerateMetrics(result.Metrics) {
		resp["metrics"] = result.Metrics
	}
	return resp
}

func hasGenerateMetrics(metrics GenerateMetrics) bool {
	return metrics.PromptTokens != 0 ||
		metrics.GeneratedTokens != 0 ||
		metrics.PrefillSeconds != 0 ||
		metrics.DecodeSeconds != 0 ||
		metrics.TotalSeconds != 0 ||
		metrics.PrefillTokensPerSec != 0 ||
		metrics.DecodeTokensPerSec != 0 ||
		metrics.PeakMemoryBytes != 0 ||
		metrics.ActiveMemoryBytes != 0
}

func normalizeAction(action string) string {
	return core.Lower(core.Trim(action))
}

// Stub responses are pre-built once and shared across every dispatch.
// Returning the same map is safe — the dispatch path passes the value
// straight to writeJSONLine which only marshals (read-only) and no
// other consumer mutates a Response after Dispatch returns.
// (See dispatch.go's only resp[k]= writers — both build a fresh map
// in generateResponseFromResult, never touch a stub.)
var (
	stubEmbedResponse    = Response{"status": "stub", "action": "embed"}
	stubScoreResponse    = Response{"status": "stub", "action": "score"}
	stubGenerateResponse = Response{"status": "stub", "action": "generate"}

	stubEmbedHandler    Handler = func(context.Context, Request) (Response, error) { return stubEmbedResponse, nil }
	stubScoreHandler    Handler = func(context.Context, Request) (Response, error) { return stubScoreResponse, nil }
	stubGenerateHandler Handler = func(context.Context, Request) (Response, error) { return stubGenerateResponse, nil }
)

func stubHandler(action string) Handler {
	switch action {
	case "embed":
		return stubEmbedHandler
	case "score":
		return stubScoreHandler
	case "generate":
		return stubGenerateHandler
	}
	// Fallback for any future stub registration — fresh closure +
	// map so the action label is captured. The three built-in stubs
	// above cover the only call sites today.
	return func(context.Context, Request) (Response, error) {
		return Response{
			"status": "stub",
			"action": action,
		}, nil
	}
}

<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# adapter.go — buffered/string adapter for inference.TextModel

**Package**: `dappco.re/go/mlx`
**File**: `go/adapter.go`

## What this is

`InferenceAdapter` — a thin wrapper around `inference.TextModel` that exposes a **buffered, string-returning** API for callers that don't want to consume the iter.Seq[Token] surface directly. Used by:

- The `book-state-demo` binary and other quick-script callers
- Adapter-style API at the root of the mlx package (`mlx.Generate(prompt) string`)
- `mlx.NewMLXBackend(path)` — the load-and-wrap entry for the CGo-style "give me a thing I can call .Generate on" usage

## Naming

This `InferenceAdapter` is the **client-side adapter** — it consumes a `TextModel` and produces a string. The complementary `metaladapter` in `register_metal.go` is the **server-side adapter** — it implements `TextModel` over `metal.Model`. Two different jobs, both called "adapter" because both do the inference↔native shape translation in their direction.

## Types

```go
type Message = inference.Message    // alias for callers who don't want the inference import

type GenOpts struct {
    MaxTokens int
    Temp      float64               // float64 here vs float32 in inference (legacy convenience)
}

type Result struct {
    Text    string
    Metrics *inference.GenerateMetrics
}

type TokenCallback func(token string) error

type InferenceAdapter struct {
    model inference.TextModel
    name  string
}
```

## Construction

```go
adapter := mlx.NewInferenceAdapter(model, "mlx")        // wrap a loaded TextModel
adapter, err := mlx.NewMLXBackend(path, loadOpts...)    // load + wrap in one call (metal backend forced)
```

`NewMLXBackend` is the common entry — adds `inference.WithBackend("metal")` to any caller-supplied LoadOption, calls `inference.LoadModel`, type-asserts to TextModel, wraps in an adapter named `"mlx"`.

## Surface

| Method | Returns | Notes |
|--------|---------|-------|
| `Name()` | string | as-constructed name (`"mlx"` or caller-supplied) |
| `Available()` | bool | adapter present + model not Closed |
| `Model()` | `inference.TextModel` | unwrap — for callers that need the iter.Seq path |
| `Close()` | error | idempotent — once closed, subsequent Close returns nil |
| `Generate(ctx, prompt, GenOpts)` | `(Result, error)` | buffered: collect all tokens, return text + metrics |
| `GenerateStream(ctx, prompt, GenOpts, TokenCallback)` | error | streaming: callback per token, callback err cancels ctx |
| `Chat(ctx, []Message, GenOpts)` | `(Result, error)` | buffered chat |
| `ChatStream(ctx, []Message, GenOpts, TokenCallback)` | error | streaming chat |
| `Classify(ctx, []string, GenOpts)` | `([]ClassifyResult, error)` | passthrough |
| `BatchGenerate(ctx, []string, GenOpts)` | `([]BatchResult, error)` | passthrough |
| `InspectAttention(ctx, prompt, GenOpts)` | `core.Result` | type-asserts to `inference.AttentionInspector` first |
| `Capabilities()` | `inference.CapabilityReport` | type-asserts to `inference.CapabilityReporter` |
| `Metrics()` | `inference.GenerateMetrics` | model's last metrics |
| `ModelType()` | string | model's architecture string |

## Buffered vs streaming

Both shapes exist because:

- **Buffered** (`Generate`, `Chat`) — the answer is a single string. Easy to log, easy to test, easy to JSON-encode for an HTTP response. Used by the BookState demo's teacher/student calls.
- **Streaming** (`GenerateStream`, `ChatStream`) — token-by-token callback. Used by the IDE chat UI to render as tokens arrive.

Buffered internally uses `core.NewBuilder()` (no string concat allocs); streaming wires `context.WithCancel` so an error from the callback cancels the underlying iterator promptly.

## Error wrapping

`InferenceAdapter` returns errors using `core.E(scope, msg, cause)` not `fmt.Errorf` — the convention everywhere in this codebase. A nil adapter, nil model, or nil callback is a programmer error returned as `"mlx: <thing> is nil"`.

## Why this is in go-mlx not go-ml

`go-ml` has its own `InferenceAdapter` shape (defined in `ml/adapter.go`) for the scoring engine — same name, different package, different surface. The mlx-side adapter targets the simple "string in, string out" use case; the ml-side adapter targets the Backend interface with capability reports + judging. They don't conflict because they're in separate packages.

## Related

- [register_metal.md](register_metal.md) — `metaladapter` (server side)
- `../../../go-inference/docs/inference/inference.md` — `TextModel` surface this wraps
- `../../../go-ml/docs/backend/adapter.md` (planned) — the scoring-engine-side InferenceAdapter

# Backend Abstraction Design

**Date:** 2026-02-19
**Status:** Approved (Virgil-reviewed 19 Feb 2026)
**Author:** GoLand Claude (domain expert, go-mlx)

## Problem

go-mlx is a monolithic CGO package. All types, ops, models, tokenisation, sampling, and caching live in the root package or shallow sub-packages, all gated behind `//go:build darwin && arm64`. This creates three problems:

1. **Memory leaks** under sustained inference. `runtime.SetFinalizer` on every `Array` can't keep up — finalisers queue faster than GC drains them. The package is currently unusable in production.
2. **No high-level API.** Consumers must wire tokenisation, forward passes, KV caches, and sampling manually. `model.Generate()` is referenced in docs but doesn't exist.
3. **No backend extensibility.** mlx_lm (Python) support is needed for the ecosystem, but there's no abstraction point to plug it in.

## Decision

Approach B: full internal reorganisation. Move all CGO code into `internal/metal/`. Root package becomes a clean public interface. Backends register via build-tagged `init()`.

### Why B over alternatives

- **vs Model-level abstraction (A):** A leaves CGO scattered across root + sub-packages. B gives clean ownership boundaries — `internal/metal/` is the CLion Claude's domain, root is the public contract.
- **vs Separate module (C):** C requires three modules to coordinate. B keeps everything in one module with clean internal boundaries.

### Design constraints

- Native Metal is the product. mlx_lm is a compatibility shim.
- No downstream consumers currently depend on the API (memory leaks block adoption). Breaking changes are free.
- darwin/arm64 only for the metal backend. Root interfaces compile everywhere but we don't optimise for other platforms.
- Communicate API changes to go-ai and go-i18n via FINDINGS.md.

## Package Layout

```
go-mlx/
├── mlx.go                  Public API: TextModel, Token, LoadModel()
├── options.go              LoadOption, GenerateOption (functional options)
├── backend.go              Backend interface, Register/Get/Default
├── register_metal.go       //go:build darwin && arm64 — auto-registers metal
├── mlx_stub.go             //go:build !darwin || !arm64 — MetalAvailable() false
│
├── internal/
│   └── metal/              All CGO code (darwin/arm64 build tags)
│       ├── metal.go        Init, Materialize, error handler, stream
│       ├── array.go        Array type, creation, data access
│       ├── dtype.go        DType constants
│       ├── ops.go          Element-wise, reduction, shape ops
│       ├── fast.go         Fused Metal kernels (RMSNorm, RoPE, SDPA)
│       ├── nn.go           Linear, Embedding, RMSNormModule
│       ├── compile.go      CompiledFunc
│       ├── slice.go        Array slicing
│       ├── random.go       RandomCategorical, RandomUniform
│       ├── io.go           Safetensors loading
│       ├── model.go        Internal Model interface + architecture dispatch
│       ├── gemma3.go       Gemma3 decoder
│       ├── qwen3.go        Qwen3 decoder
│       ├── cache.go        KVCache + RotatingKVCache
│       ├── sample.go       Sampling chain (greedy, temp, topK, topP)
│       ├── tokenizer.go    BPE tokenizer
│       ├── grad.go         VJP
│       ├── lora.go         LoRA adapters
│       ├── optim.go        AdamW
│       ├── generate.go     NEW: autoregressive generation loop
│       └── backend.go      Implements mlx.TextModel, exports New()
│
├── mlxlm/                  Future: Python subprocess backend
│   └── backend.go          Implements mlx.Backend via core/go/pkg/process
│
├── cpp/                    CLion Claude workspace (unchanged)
│   ├── CMakeLists.txt
│   ├── CLAUDE.md
│   ├── TODO.md
│   └── FINDINGS.md
│
└── docs/plans/             This file
```

## Public Interface

### Types

```go
package mlx

type Token struct {
    ID   int32
    Text string
}

type TextModel interface {
    // Generate streams tokens for the given prompt. Respects ctx cancellation.
    Generate(ctx context.Context, prompt string, opts ...GenerateOption) iter.Seq[Token]

    // Chat formats messages using the model's native template, then generates.
    // Deferred to Phase 5 if needed — model owns its chat template.
    Chat(ctx context.Context, messages []Message, opts ...GenerateOption) iter.Seq[Token]

    // ModelType returns the architecture identifier (e.g. "gemma3", "qwen3").
    ModelType() string

    // Err returns the last generation error (OOM, C-level failure, etc.).
    // Distinguishes normal stop (EOS, max tokens) from errors.
    Err() error

    // Close releases all resources (GPU memory, caches, subprocess).
    Close() error
}

// Message represents a chat turn for Chat().
type Message struct {
    Role    string // "user", "assistant", "system"
    Content string
}
```

### Entry point

```go
func LoadModel(path string, opts ...LoadOption) (TextModel, error)
func MetalAvailable() bool

// Hardware-level memory controls (delegate to internal/metal).
// These are not model-level — they control the Metal allocator directly.
func SetCacheLimit(limit uint64) uint64
func SetMemoryLimit(limit uint64) uint64
func GetActiveMemory() uint64
func GetPeakMemory() uint64
func ClearCache()
```

### Functional options

```go
type GenerateOption func(*GenerateConfig)

func WithMaxTokens(n int) GenerateOption
func WithTemperature(t float32) GenerateOption
func WithTopK(k int) GenerateOption
func WithTopP(p float32) GenerateOption
func WithStopTokens(ids ...int32) GenerateOption

type LoadOption func(*LoadConfig)

func WithBackend(name string) LoadOption
```

### Consumer usage

```go
import "forge.lthn.ai/core/go-mlx"

m, err := mlx.LoadModel("/Volumes/Data/lem/safetensors/gemma-3/")
if err != nil { log.Fatal(err) }
defer m.Close()

for tok := range m.Generate("What is 2+2?",
    mlx.WithMaxTokens(128),
    mlx.WithTemperature(0.7),
) {
    fmt.Print(tok.Text)
}
```

### Backend registration

```go
// backend.go
type Backend interface {
    Name() string
    LoadModel(path string, opts ...LoadOption) (TextModel, error)
}

func Register(b Backend)
func Get(name string) (Backend, bool)
func Default() (Backend, error)

// register_metal.go (//go:build darwin && arm64)
func init() { Register(metal.New()) }
func MetalAvailable() bool { return true }
```

## Memory Management

### Problem

`runtime.SetFinalizer` on every Array. Under sustained inference, GC can't drain finalisers fast enough. C-side memory grows unbounded.

### Fix: Two levels

**Level 1 — TextModel.Close():**

`Close()` walks the model's weight map, KV caches, and retained arrays. Calls explicit `Free()` on each. No reliance on GC for large allocations (model weights, cache buffers).

**Level 2 — Per-step intermediate cleanup:**

Each decode step in `generate.go` creates intermediate arrays (logits, attention, MLP). These are freed explicitly after each step rather than waiting for GC:

```go
for i := 0; i < cfg.maxTokens; i++ {
    logits := m.model.Forward(input, caches)
    materialize(logits)
    next := sampler.Sample(lastLogits(logits))
    materialize(next)

    freeIntermediates(logits)  // deterministic, don't wait for GC

    tok := tokenFromArray(next)
    if !yield(tok) { return }
}
```

The exact mechanism (pool, free list, or per-step `ClearCache()`) depends on CLion Claude's research into `mlx_clear_cache` behaviour and whether `mlx_array_free` is safe on arrays still in the computation graph.

### CLion Claude research needed

Added to `cpp/TODO.md`:
- What does `mlx_clear_cache()` actually release? Can we call it per decode step?
- Is `mlx_array_free()` safe on arrays referenced by other arrays in the graph?
- Does the MLX allocator pool reuse freed memory or return it to the system?

## Error Handling

### Problem

`checkError()` logs to slog and swallows. No error propagation to callers.

### Fix

Internal ops return errors:

```go
// internal/metal/ops.go
func matmul(a, b *array) (*array, error) {
    out := newArray("MATMUL")
    C.mlx_matmul(&out.ctx, a.ctx, b.ctx, defaultStream().ctx)
    if err := lastError(); err != nil {
        return nil, fmt.Errorf("matmul: %w", err)
    }
    return out, nil
}
```

Errors propagate through the model forward pass and surface via the public API. `Generate()` stops yielding tokens on error; `Close()` returns the last error if generation was interrupted.

## Migration Path

Mechanical move in dependency order. Tests pass at each step.

### Step 1: Define public surface

Create root-level `mlx.go`, `options.go`, `backend.go` with interfaces. No code moves yet. Existing code untouched.

### Step 2: Create internal/metal/ and move code

Move files in dependency order:

```
 1. dtype.go          no dependencies
 2. array.go          depends on dtype
 3. stream.go         depends on array
 4. ops.go            depends on array, stream
 5. slice.go          depends on array, stream
 6. random.go         depends on array, stream
 7. fast.go           depends on array, stream
 8. nn.go             depends on array, ops, fast
 9. compile.go        depends on array
10. io.go             depends on array
11. grad.go           depends on array, ops
12. lora.go           depends on nn, grad
13. optim.go          depends on array, ops, lora
14. tokenizer.go      standalone
15. cache.go          depends on array, ops, slice
16. sample.go         depends on array, ops, random
17. gemma3.go         depends on everything above
18. qwen3.go          same
19. model.go          architecture dispatch
```

Each file: `package mlx` becomes `package metal`. Sub-package files (`model/`, `tokenizer/`, `sample/`, `cache/`) flatten into the metal package. Name collisions resolved by keeping the simpler name (e.g., `Cache` not `cache.Cache`).

Tests move with their code. After each batch: `go test ./internal/metal/`.

### Step 3: Wire the backend

Add `generate.go` (autoregressive loop), `backend.go` (implements TextModel), root `register_metal.go`. Now `mlx.LoadModel()` works end-to-end.

### Step 4: Memory and error fixes

Implement explicit cleanup in `Close()` and per-step `freeIntermediates()`. Convert internal ops to return errors. Based on CLion Claude's findings on `mlx_array_free` safety.

### Step 5: mlxlm/ backend (separate effort, future)

Thin subprocess wrapper using `core/go/pkg/process`. Implements `mlx.Backend`. Registered via `import _ "forge.lthn.ai/core/go-mlx/mlxlm"`.

## Testing

- All 148 existing tests move into `internal/metal/` and must pass
- New `generate_test.go` — autoregressive loop with Gemma3-1B from `/Volumes/Data/lem/safetensors/`
- New `backend_test.go` — end-to-end LoadModel + Generate
- New `memory_test.go` — 1000-token generation, assert GetPeakMemory() bounded
- Root `mlx_test.go` — integration via public mlx.LoadModel() API

## Communication

### To go-ai (via FINDINGS.md)

The old API (`Array`, `MatMul`, `model.LoadModel`, etc.) is gone. Migrate `backend_mlx.go` to:
```go
m, _ := mlx.LoadModel(path)
for tok := range m.Generate(prompt, mlx.WithMaxTokens(n)) { ... }
```

### To go-i18n (via FINDINGS.md)

The API for Gemma3-1B inference will be:
```go
m, _ := mlx.LoadModel("/path/to/gemma-3-1b/")
for tok := range m.Generate(sentence, mlx.WithMaxTokens(32)) { ... }
```
Streaming via `iter.Seq[Token]`. No tokenisation or sampling to handle.

### To CLion Claude (via cpp/TODO.md)

- Research `mlx_clear_cache()` per-step safety
- Research `mlx_array_free()` on graph-referenced arrays
- MLX allocator pool behaviour

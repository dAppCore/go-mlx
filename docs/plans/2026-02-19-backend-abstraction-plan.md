# Backend Abstraction Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Restructure go-mlx so all CGO code lives in `internal/metal/`, the root package exposes a clean `TextModel` interface with `LoadModel()` + `Generate()`, and memory leaks are fixed with deterministic cleanup.

**Architecture:** Root package defines interfaces (`TextModel`, `Backend`, `Token`) and functional options. `internal/metal/` contains all CGO, model architectures, tokenisation, sampling, caching, and a new autoregressive generation loop. Metal backend auto-registers via build-tagged `init()` in root.

**Tech Stack:** Go 1.25.5, CGO, mlx-c v0.4.1, Apple Metal, CMake

**Design doc:** `docs/plans/2026-02-19-backend-abstraction-design.md`

**Virgil review (19 Feb 2026) — folded into plan:**
1. **`context.Context` on Generate()** — REQUIRED. HTTP handlers need cancellation. Check `ctx.Done()` in decode loop.
2. **`Err() error` on TextModel** — Distinguish EOS/max-tokens from OOM/C-errors. Check after iteration.
3. **`Chat()` on TextModel** — Model owns its chat template. Can defer to Phase 5 but recommended now.
4. **Memory control functions at root** — `SetCacheLimit`, `SetMemoryLimit`, `GetActiveMemory`, `GetPeakMemory`, `ClearCache` stay public, delegate to `internal/metal`.

---

## Pre-flight

Before starting: the `dist/` directory (CMake install output) needs gitignoring and all 148 existing tests must pass.

```bash
# Verify clean starting point
cd /Users/snider/Code/go-mlx
go test ./...
# Expected: ok (148 tests across 10 files)
```

---

### Task 1: Gitignore housekeeping

**Files:**
- Modify: `.gitignore`

**Step 1: Add dist/ to gitignore**

Append to `.gitignore`:
```
# CMake install output
dist/
```

**Step 2: Commit**

```bash
git add .gitignore
git commit -m "chore: gitignore dist/ (CMake install output)"
```

---

### Task 2: Create root public interface

Define the public API types. No CGO, no build tags. This file will eventually replace `mlx.go` but for now coexists.

**Files:**
- Create: `textmodel.go`
- Create: `options.go`
- Create: `backend.go`

**Step 1: Write `textmodel.go`**

```go
package mlx

import (
	"context"
	"iter"
)

// Token represents a single generated token for streaming.
type Token struct {
	ID   int32
	Text string
}

// Message represents a chat turn for Chat().
type Message struct {
	Role    string // "user", "assistant", "system"
	Content string
}

// TextModel generates text from a loaded model.
type TextModel interface {
	// Generate streams tokens for the given prompt.
	// Respects ctx cancellation (HTTP handlers, timeouts, graceful shutdown).
	Generate(ctx context.Context, prompt string, opts ...GenerateOption) iter.Seq[Token]

	// Chat formats messages using the model's native chat template, then generates.
	// The model owns its template — callers don't need to know Gemma vs Qwen formatting.
	Chat(ctx context.Context, messages []Message, opts ...GenerateOption) iter.Seq[Token]

	// ModelType returns the architecture identifier (e.g. "gemma3", "qwen3").
	ModelType() string

	// Err returns the error from the last Generate/Chat call, if any.
	// Distinguishes normal stop (EOS, max tokens) from failures (OOM, C-level error).
	// Returns nil if generation completed normally.
	Err() error

	// Close releases all resources (GPU memory, caches, subprocess).
	Close() error
}
```

**Step 2: Write `options.go`**

```go
package mlx

// GenerateConfig holds generation parameters.
type GenerateConfig struct {
	MaxTokens   int
	Temperature float32
	TopK        int
	TopP        float32
	StopTokens  []int32
}

// DefaultGenerateConfig returns sensible defaults.
func DefaultGenerateConfig() GenerateConfig {
	return GenerateConfig{
		MaxTokens:   256,
		Temperature: 0.0,
	}
}

// GenerateOption configures text generation.
type GenerateOption func(*GenerateConfig)

// WithMaxTokens sets the maximum number of tokens to generate.
func WithMaxTokens(n int) GenerateOption {
	return func(c *GenerateConfig) { c.MaxTokens = n }
}

// WithTemperature sets the sampling temperature. 0 = greedy.
func WithTemperature(t float32) GenerateOption {
	return func(c *GenerateConfig) { c.Temperature = t }
}

// WithTopK sets top-k sampling. 0 = disabled.
func WithTopK(k int) GenerateOption {
	return func(c *GenerateConfig) { c.TopK = k }
}

// WithTopP sets nucleus sampling threshold. 0 = disabled.
func WithTopP(p float32) GenerateOption {
	return func(c *GenerateConfig) { c.TopP = p }
}

// WithStopTokens sets token IDs that stop generation.
func WithStopTokens(ids ...int32) GenerateOption {
	return func(c *GenerateConfig) { c.StopTokens = ids }
}

// LoadConfig holds model loading parameters.
type LoadConfig struct {
	Backend string // "metal" (default), "mlx_lm"
}

// LoadOption configures model loading.
type LoadOption func(*LoadConfig)

// WithBackend selects a specific inference backend by name.
func WithBackend(name string) LoadOption {
	return func(c *LoadConfig) { c.Backend = name }
}

// ApplyGenerateOpts builds a GenerateConfig from options.
func ApplyGenerateOpts(opts []GenerateOption) GenerateConfig {
	cfg := DefaultGenerateConfig()
	for _, o := range opts {
		o(&cfg)
	}
	return cfg
}

// ApplyLoadOpts builds a LoadConfig from options.
func ApplyLoadOpts(opts []LoadOption) LoadConfig {
	var cfg LoadConfig
	for _, o := range opts {
		o(&cfg)
	}
	return cfg
}
```

**Step 3: Write `backend.go`**

```go
package mlx

import (
	"fmt"
	"sync"
)

// Backend is a named inference engine that can load models.
type Backend interface {
	// Name returns the backend identifier (e.g. "metal", "mlx_lm").
	Name() string

	// LoadModel loads a model from the given path.
	LoadModel(path string, opts ...LoadOption) (TextModel, error)
}

var (
	backendsMu sync.RWMutex
	backends   = map[string]Backend{}
)

// Register adds a backend to the registry.
func Register(b Backend) {
	backendsMu.Lock()
	defer backendsMu.Unlock()
	backends[b.Name()] = b
}

// Get returns a registered backend by name.
func Get(name string) (Backend, bool) {
	backendsMu.RLock()
	defer backendsMu.RUnlock()
	b, ok := backends[name]
	return b, ok
}

// Default returns the first available backend.
// Prefers "metal" if registered.
func Default() (Backend, error) {
	backendsMu.RLock()
	defer backendsMu.RUnlock()
	if b, ok := backends["metal"]; ok {
		return b, nil
	}
	for _, b := range backends {
		return b, nil
	}
	return nil, fmt.Errorf("mlx: no backends registered")
}

// LoadModel loads a model using the specified or default backend.
func LoadModel(path string, opts ...LoadOption) (TextModel, error) {
	cfg := ApplyLoadOpts(opts)
	if cfg.Backend != "" {
		b, ok := Get(cfg.Backend)
		if !ok {
			return nil, fmt.Errorf("mlx: backend %q not registered", cfg.Backend)
		}
		return b.LoadModel(path, opts...)
	}
	b, err := Default()
	if err != nil {
		return nil, err
	}
	return b.LoadModel(path, opts...)
}
```

**Step 4: Run tests to verify no breakage**

```bash
go test ./...
# Expected: existing 148 tests still pass — new files have no tests yet but don't conflict
```

**Step 5: Commit**

```bash
git add textmodel.go options.go backend.go
git commit -m "feat(api): define public TextModel, Backend, and options interfaces"
```

---

### Task 3: Create `internal/metal/` and move foundation files

Move the dependency-free files first: `dtype.go`, then `array.go`, then `stream.go`.

**Files:**
- Create: `internal/metal/dtype.go` (from `dtype.go`)
- Create: `internal/metal/array.go` (from `array.go`)
- Create: `internal/metal/metal.go` (from `mlx.go` — the CGO init/Materialize/error handler)
- Create: `internal/metal/stream.go` (from `stream.go`)
- Delete: `dtype.go`, `array.go`, `mlx.go` (CGO version), `stream.go`

**Step 1: Create `internal/metal/` directory**

```bash
mkdir -p internal/metal
```

**Step 2: Move dtype.go**

Copy `dtype.go` to `internal/metal/dtype.go`. Change:
- `package mlx` → `package metal`
- Keep build tag `//go:build darwin && arm64`
- Keep CGO import — `DType` is defined as `C.mlx_dtype`

**Step 3: Move mlx.go → internal/metal/metal.go**

Copy core CGO bridge from `mlx.go` to `internal/metal/metal.go`. Change:
- `package mlx` → `package metal`
- Rename `Materialize` → `Materialize` (stays exported, used within internal/metal by other files)
- Rename `checkError` → stays unexported
- Remove the `go:generate` directives (they stay in root or a separate generate.go)

**Step 4: Move array.go → internal/metal/array.go**

Copy `array.go` to `internal/metal/array.go`. Change:
- `package mlx` → `package metal`
- All types (`Array`, `newArray`, `FromValue`, `FromValues`, `Zeros`, `Free`) stay exported within the internal package

**Step 5: Move stream.go → internal/metal/stream.go**

Copy `stream.go` to `internal/metal/stream.go`. Change:
- `package mlx` → `package metal`

**Step 6: Create root `generate.go` for go:generate directives**

The `go:generate` CMake directives from `mlx.go` need a home in root:

```go
//go:build ignore

package mlx

//go:generate cmake -S . -B build -DCMAKE_INSTALL_PREFIX=dist -DCMAKE_BUILD_TYPE=Release
//go:generate cmake --build build --parallel
//go:generate cmake --install build
```

Actually — `go:generate` directives work regardless of build tags. Create a minimal file:

```go
package mlx

//go:generate cmake -S . -B build -DCMAKE_INSTALL_PREFIX=dist -DCMAKE_BUILD_TYPE=Release
//go:generate cmake --build build --parallel
//go:generate cmake --install build
```

**Step 7: Delete originals**

Remove `dtype.go`, `array.go`, `stream.go` from root. Replace `mlx.go` with the slim version holding only `go:generate` + `MetalAvailable` stub import.

**Step 8: Run tests**

```bash
go test ./internal/metal/
# Expected: won't compile yet — ops.go etc. still in root referencing old package
# This is expected — we're mid-migration
```

**Step 9: Commit**

```bash
git add internal/metal/ generate.go
git rm dtype.go array.go stream.go
git commit -m "refactor(metal): move dtype, array, stream to internal/metal"
```

---

### Task 4: Move ops, slice, random, fast, compile

The second batch — depends on array, stream, dtype.

**Files:**
- Move: `ops.go` → `internal/metal/ops.go`
- Move: `slice.go` → `internal/metal/slice.go`
- Move: `random.go` → `internal/metal/random.go`
- Move: `fast.go` → `internal/metal/fast.go`
- Move: `compile.go` → `internal/metal/compile.go`

**Step 1: Copy each file to `internal/metal/`**

For each file: `package mlx` → `package metal`. All function signatures stay the same (they're all in the same package now).

**Step 2: Delete originals from root**

```bash
git rm ops.go slice.go random.go fast.go compile.go
```

**Step 3: Commit**

```bash
git add internal/metal/
git commit -m "refactor(metal): move ops, slice, random, fast, compile to internal/metal"
```

---

### Task 5: Move nn, io, grad, lora, optim

Third batch — depends on ops, array.

**Files:**
- Move: `nn.go` → `internal/metal/nn.go`
- Move: `io.go` → `internal/metal/io.go`
- Move: `grad.go` → `internal/metal/grad.go`
- Move: `lora.go` → `internal/metal/lora.go`
- Move: `optim.go` → `internal/metal/optim.go`

**Step 1: Copy each file to `internal/metal/`**

`package mlx` → `package metal`. Internal cross-references (e.g., `LoRALinear` referencing `Linear`) now resolve within the same package.

**Step 2: Delete originals**

```bash
git rm nn.go io.go grad.go lora.go optim.go
```

**Step 3: Commit**

```bash
git add internal/metal/
git commit -m "refactor(metal): move nn, io, grad, lora, optim to internal/metal"
```

---

### Task 6: Flatten sub-packages into internal/metal/

The sub-packages (`model/`, `tokenizer/`, `sample/`, `cache/`) merge into `internal/metal/`.

**Files:**
- Move: `tokenizer/tokenizer.go` → `internal/metal/tokenizer.go`
- Move: `sample/sample.go` → `internal/metal/sample.go`
- Move: `cache/cache.go` → `internal/metal/cache.go`
- Move: `model/model.go` → `internal/metal/model.go`
- Move: `model/gemma3.go` → `internal/metal/gemma3.go`
- Move: `model/qwen3.go` → `internal/metal/qwen3.go`

**Step 1: Copy files, fix package declarations**

For each file:
- `package tokenizer` / `package sample` / `package cache` / `package model` → `package metal`
- Remove all `import "forge.lthn.ai/core/go-mlx"` lines — the types are now in the same package
- Remove all `import "forge.lthn.ai/core/go-mlx/cache"` etc. — same package
- Remove `mlx.` prefixes on all type and function references (e.g., `mlx.Array` → `Array`, `mlx.MatMul` → `MatMul`, `mlx.Linear` → `Linear`)
- Remove `cache.` prefixes (`cache.Cache` → `Cache`, `cache.NewKVCache` → `NewKVCache`)
- Remove `tokenizer.` prefixes
- Remove `sample.` prefixes

**Step 2: Handle name collisions**

Check for conflicts when flattening:
- `model.Model` interface → rename to `InternalModel` (the public one is `mlx.TextModel`)
- `model.LoadModel` function → rename to `loadModel` (unexported, called by backend.go)
- `tokenizer.Load` → rename to `loadTokenizer`
- `tokenizer.Tokenizer` struct → stays `Tokenizer`
- `sample.Sampler` interface → stays `Sampler`
- `sample.newArray` → rename to `newSampler`
- `cache.Cache` interface → stays `Cache`

**Step 3: Delete old sub-package directories**

```bash
git rm -r model/ tokenizer/ sample/ cache/
```

**Step 4: Verify compilation**

```bash
go build ./internal/metal/
# Expected: compiles (all types in one package now)
```

**Step 5: Commit**

```bash
git add internal/metal/
git commit -m "refactor(metal): flatten model, tokenizer, sample, cache into internal/metal"
```

---

### Task 7: Move all tests into internal/metal/

**Files:**
- Move: `array_test.go` → `internal/metal/array_test.go`
- Move: `ops_test.go` → `internal/metal/ops_test.go`
- Move: `nn_test.go` → `internal/metal/nn_test.go`
- Move: `fast_test.go` → `internal/metal/fast_test.go`
- Move: `grad_test.go` → `internal/metal/grad_test.go`
- Move: `lora_test.go` → `internal/metal/lora_test.go`
- Move: `optim_test.go` → `internal/metal/optim_test.go`
- Move: `tokenizer/tokenizer_test.go` → `internal/metal/tokenizer_test.go`
- Move: `sample/sample_test.go` → `internal/metal/sample_test.go`
- Move: `cache/cache_test.go` → `internal/metal/cache_test.go`

**Step 1: Copy all test files**

For each:
- `package mlx` / `package tokenizer` / etc. → `package metal`
- Remove `import "forge.lthn.ai/core/go-mlx"` and sub-package imports
- Remove `mlx.` / `cache.` / `tokenizer.` / `sample.` prefixes on all references
- Adjust any renamed functions (e.g., `tokenizer.Load` → `loadTokenizer`)

**Step 2: Delete originals**

```bash
git rm array_test.go ops_test.go nn_test.go fast_test.go grad_test.go lora_test.go optim_test.go
```

**Step 3: Run all tests**

```bash
go test ./internal/metal/ -count=1
# Expected: all 148 tests pass
```

This is the critical checkpoint. If tests fail here, fix before continuing.

**Step 4: Commit**

```bash
git add internal/metal/
git commit -m "refactor(metal): move all tests to internal/metal (148 tests passing)"
```

---

### Task 8: Clean up root package

Remove all old CGO files from root. Root should now contain only: public interfaces, options, backend registry, build-tagged registration, stub.

**Files:**
- Rewrite: `mlx.go` — just `go:generate` directives
- Keep: `textmodel.go` (from Task 2)
- Keep: `options.go` (from Task 2)
- Keep: `backend.go` (from Task 2)
- Rewrite: `register_metal.go` — build-tagged init that registers metal backend
- Keep: `mlx_stub.go` — unchanged

**Step 1: Write `register_metal.go`**

```go
//go:build darwin && arm64

package mlx

import "forge.lthn.ai/core/go-mlx/internal/metal"

func init() {
	Register(metal.NewBackend())
}

// MetalAvailable reports whether native Metal inference is available.
func MetalAvailable() bool { return true }

// Hardware-level memory controls — delegate to internal/metal.
// These are not model-level; they control the Metal allocator directly.

func SetCacheLimit(limit uint64) uint64  { return metal.SetCacheLimit(limit) }
func SetMemoryLimit(limit uint64) uint64 { return metal.SetMemoryLimit(limit) }
func GetActiveMemory() uint64            { return metal.GetActiveMemory() }
func GetPeakMemory() uint64              { return metal.GetPeakMemory() }
func ClearCache()                        { metal.ClearCache() }
```

**Step 2: Write final `mlx.go`**

```go
// Package mlx provides Go bindings for Apple's MLX framework.
//
// Build mlx-c before use:
//
//	go generate ./...
package mlx

//go:generate cmake -S . -B build -DCMAKE_INSTALL_PREFIX=dist -DCMAKE_BUILD_TYPE=Release
//go:generate cmake --build build --parallel
//go:generate cmake --install build
```

**Step 3: Verify root compiles**

```bash
go build .
# Expected: compiles on darwin/arm64 (metal registered)
```

**Step 4: Verify all tests pass**

```bash
go test ./...
# Expected: all 148 tests pass (now in internal/metal/)
```

**Step 5: Commit**

```bash
git add mlx.go register_metal.go
git commit -m "refactor(api): clean root package — interfaces only, metal auto-registered"
```

---

### Task 9: Implement the generate loop

The core new functionality — autoregressive text generation with streaming.

**Files:**
- Create: `internal/metal/generate.go`

**Step 1: Write failing test**

Create `internal/metal/generate_test.go`:

```go
//go:build darwin && arm64

package metal

import (
	"testing"

	mlx "forge.lthn.ai/core/go-mlx"
)

// TestGenerate_Greedy requires Gemma3-1B on disk.
// Skip in CI, run locally.
func TestGenerate_Greedy(t *testing.T) {
	const modelPath = "/Volumes/Data/lem/safetensors/gemma-3/"
	if !fileExists(modelPath) {
		t.Skip("model not available at", modelPath)
	}

	b := NewBackend()
	m, err := b.LoadModel(modelPath)
	if err != nil {
		t.Fatalf("LoadModel: %v", err)
	}
	defer m.Close()

	ctx := context.Background()
	var tokens []mlx.Token
	for tok := range m.Generate(ctx, "What is 2+2?", mlx.WithMaxTokens(16)) {
		tokens = append(tokens, tok)
	}
	if err := m.Err(); err != nil {
		t.Fatalf("Generate error: %v", err)
	}

	if len(tokens) == 0 {
		t.Fatal("Generate produced no tokens")
	}
	t.Logf("Generated %d tokens", len(tokens))
	for _, tok := range tokens {
		t.Logf("  [%d] %q", tok.ID, tok.Text)
	}
}

func fileExists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}
```

**Step 2: Run test to verify it fails**

```bash
go test ./internal/metal/ -run TestGenerate_Greedy -v
# Expected: FAIL — NewBackend, LoadModel, Generate not implemented yet
```

**Step 3: Implement `generate.go`**

```go
//go:build darwin && arm64

package metal

import (
	"iter"

	mlx "forge.lthn.ai/core/go-mlx"
)

// metalModel wraps a loaded model and provides the TextModel interface.
type metalModel struct {
	model     InternalModel
	tokenizer *Tokenizer
	modelType string
	lastErr   error // set by Generate/Chat on failure
}

func (m *metalModel) ModelType() string { return m.modelType }
func (m *metalModel) Err() error        { return m.lastErr }

func (m *metalModel) Close() error {
	// TODO: explicit Free() on all model weight arrays and caches
	// For now, rely on GC finalisers (existing behaviour)
	return nil
}

func (m *metalModel) Chat(ctx context.Context, messages []mlx.Message, opts ...mlx.GenerateOption) iter.Seq[mlx.Token] {
	prompt := m.formatChat(messages) // model-specific template
	return m.Generate(ctx, prompt, opts...)
}

func (m *metalModel) Generate(ctx context.Context, prompt string, opts ...mlx.GenerateOption) iter.Seq[mlx.Token] {
	cfg := mlx.ApplyGenerateOpts(opts)
	m.lastErr = nil // reset per-generation

	return func(yield func(mlx.Token) bool) {
		tokens := m.tokenizer.Encode(prompt)
		caches := m.model.NewCache()
		sampler := newSampler(cfg.Temperature, cfg.TopP, 0, cfg.TopK)

		// Prefill: process entire prompt
		input := FromValues(tokens, len(tokens))
		input = Reshape(input, 1, int32(len(tokens))) // [1, seqLen]
		logits := m.model.Forward(input, caches)
		Materialize(logits)

		for i := 0; i < cfg.MaxTokens; i++ {
			// Check context cancellation (HTTP timeout, shutdown)
			select {
			case <-ctx.Done():
				m.lastErr = ctx.Err()
				return
			default:
			}

			// Sample from last position logits
			lastPos := SliceAxis(logits, 1, int32(logits.Dim(1)-1), int32(logits.Dim(1)))
			lastPos = Reshape(lastPos, 1, int32(lastPos.Dim(2))) // [1, vocab]
			next := sampler.Sample(lastPos)
			Materialize(next)

			id := int32(next.Int())

			// Check stop conditions
			if id == m.tokenizer.EOSToken() {
				return
			}
			for _, stop := range cfg.StopTokens {
				if id == stop {
					return
				}
			}

			text := m.tokenizer.DecodeToken(id)
			if !yield(mlx.Token{ID: id, Text: text}) {
				return
			}

			// Next step input
			nextInput := FromValues([]int32{id}, 1)
			nextInput = Reshape(nextInput, 1, 1) // [1, 1]
			logits = m.model.Forward(nextInput, caches)
			Materialize(logits)

			// Free previous step intermediates
			ClearCache()
		}
	}
}
```

**Step 4: Implement `backend.go` (internal/metal/)**

```go
//go:build darwin && arm64

package metal

import mlx "forge.lthn.ai/core/go-mlx"

// metalBackend implements mlx.Backend for native Metal inference.
type metalBackend struct{}

// NewBackend creates the Metal inference backend.
func NewBackend() mlx.Backend {
	return &metalBackend{}
}

func (b *metalBackend) Name() string { return "metal" }

func (b *metalBackend) LoadModel(path string, opts ...mlx.LoadOption) (mlx.TextModel, error) {
	Init()
	model, err := loadModel(path)
	if err != nil {
		return nil, err
	}
	return &metalModel{
		model:     model,
		tokenizer: model.Tokenizer(),
		modelType: model.ModelType(),
	}, nil
}
```

**Step 5: Run test**

```bash
go test ./internal/metal/ -run TestGenerate_Greedy -v -timeout 120s
# Expected: PASS (if model files available) or SKIP
```

**Step 6: Commit**

```bash
git add internal/metal/generate.go internal/metal/backend.go internal/metal/generate_test.go
git commit -m "feat(metal): implement autoregressive Generate with streaming iter.Seq[Token]"
```

---

### Task 10: Memory management — deterministic cleanup

Address the memory leak with explicit resource management.

**Files:**
- Modify: `internal/metal/generate.go` — per-step cleanup
- Modify: `internal/metal/backend.go` — Close() implementation
- Create: `internal/metal/memory_test.go`

**Step 1: Write failing test**

```go
//go:build darwin && arm64

package metal

import (
	"testing"

	mlx "forge.lthn.ai/core/go-mlx"
)

func TestMemory_GenerateDoesNotLeak(t *testing.T) {
	const modelPath = "/Volumes/Data/lem/safetensors/gemma-3/"
	if !fileExists(modelPath) {
		t.Skip("model not available at", modelPath)
	}

	b := NewBackend()
	m, err := b.LoadModel(modelPath)
	if err != nil {
		t.Fatalf("LoadModel: %v", err)
	}

	// Generate 100 tokens, measure peak memory
	ClearCache()
	beforeMem := GetActiveMemory()

	ctx := context.Background()
	for tok := range m.Generate(ctx, "Tell me a story",
		mlx.WithMaxTokens(100), mlx.WithTemperature(0.7)) {
		_ = tok
	}

	afterGenMem := GetPeakMemory()
	t.Logf("Memory: before=%dMB peak=%dMB", beforeMem/1024/1024, afterGenMem/1024/1024)

	// Close should release model weights
	m.Close()
	ClearCache()

	afterCloseMem := GetActiveMemory()
	t.Logf("Memory after Close: %dMB", afterCloseMem/1024/1024)

	// Active memory after Close should be significantly less than peak
	if afterCloseMem > afterGenMem/2 {
		t.Errorf("possible leak: active=%dMB still > 50%% of peak=%dMB",
			afterCloseMem/1024/1024, afterGenMem/1024/1024)
	}
}
```

**Step 2: Implement Close() with explicit cleanup**

Update `metalModel.Close()` to walk model weights and free them. This depends on the internal model structure — each architecture (Gemma3, Qwen3) holds weight arrays that need freeing.

Add a `FreeWeights()` method to the internal `InternalModel` interface, implemented by each architecture to free all their `*Array` fields.

**Step 3: Run test**

```bash
go test ./internal/metal/ -run TestMemory -v -timeout 300s
# Expected: PASS — memory drops after Close()
```

**Step 4: Commit**

```bash
git add internal/metal/
git commit -m "fix(metal): deterministic memory cleanup in Close() and per-step freeIntermediates"
```

---

### Task 11: Integration test via public API

Test the full public surface: `mlx.LoadModel()` → `TextModel.Generate()`.

**Files:**
- Create: `mlx_test.go` (root package)

**Step 1: Write test**

```go
//go:build darwin && arm64

package mlx_test

import (
	"testing"

	"forge.lthn.ai/core/go-mlx"
)

func TestLoadModel_MetalBackend(t *testing.T) {
	if !mlx.MetalAvailable() {
		t.Skip("Metal not available")
	}

	const modelPath = "/Volumes/Data/lem/safetensors/gemma-3/"

	m, err := mlx.LoadModel(modelPath)
	if err != nil {
		t.Fatalf("LoadModel: %v", err)
	}
	defer m.Close()

	if m.ModelType() != "gemma3" {
		t.Errorf("ModelType = %q, want gemma3", m.ModelType())
	}

	ctx := context.Background()
	var count int
	for tok := range m.Generate(ctx, "What is 2+2?", mlx.WithMaxTokens(16)) {
		count++
		t.Logf("[%d] %q", tok.ID, tok.Text)
	}

	if count == 0 {
		t.Error("Generate produced no tokens")
	}
}

func TestLoadModel_NoBackend(t *testing.T) {
	// On darwin/arm64 this should succeed — metal is auto-registered.
	// This test verifies the registration mechanism works.
	_, err := mlx.LoadModel("/nonexistent/path")
	if err == nil {
		t.Error("expected error for nonexistent model path")
	}
}
```

**Step 2: Run test**

```bash
go test -run TestLoadModel -v -timeout 120s
# Expected: PASS
```

**Step 3: Commit**

```bash
git add mlx_test.go
git commit -m "test(api): integration tests for public LoadModel + Generate"
```

---

### Task 12: Update TODO.md and FINDINGS.md

Mark completed items, update the task queue to reflect the new structure.

**Files:**
- Modify: `TODO.md`
- Modify: `FINDINGS.md`
- Modify: `CLAUDE.md`

**Step 1: Update TODO.md**

Mark completed phase items. Add any new tasks discovered during implementation.

**Step 2: Update CLAUDE.md**

Update the architecture section to reflect the new `internal/metal/` structure and the public `TextModel` API.

**Step 3: Update FINDINGS.md**

Document: migration completed, new API surface, memory management approach, any issues found.

**Step 4: Commit**

```bash
git add TODO.md FINDINGS.md CLAUDE.md
git commit -m "docs: update project docs for backend abstraction"
```

---

## Summary

| Task | What | Commit message |
|------|------|---------------|
| 1 | Gitignore dist/ | `chore: gitignore dist/` |
| 2 | Public interfaces | `feat(api): define public TextModel, Backend, and options` |
| 3 | Move foundation (dtype, array, stream, metal) | `refactor(metal): move dtype, array, stream to internal/metal` |
| 4 | Move ops, slice, random, fast, compile | `refactor(metal): move ops, slice, random, fast, compile` |
| 5 | Move nn, io, grad, lora, optim | `refactor(metal): move nn, io, grad, lora, optim` |
| 6 | Flatten sub-packages | `refactor(metal): flatten model, tokenizer, sample, cache` |
| 7 | Move tests (148 must pass) | `refactor(metal): move all tests (148 passing)` |
| 8 | Clean root package | `refactor(api): clean root — interfaces only` |
| 9 | Generate loop | `feat(metal): autoregressive Generate with iter.Seq[Token]` |
| 10 | Memory fix | `fix(metal): deterministic memory cleanup` |
| 11 | Integration test | `test(api): integration tests for LoadModel + Generate` |
| 12 | Update docs | `docs: update project docs for backend abstraction` |

**Critical checkpoint:** After Task 7, all 148 tests must pass in `internal/metal/`. If they don't, stop and fix before continuing.

**Model-dependent tests** (Tasks 9, 10, 11) require Gemma3-1B at `/Volumes/Data/lem/safetensors/gemma-3/`. They use `t.Skip()` when the model isn't available.

## Virgil Review Items (folded in)

| Item | Where | Status |
|------|-------|--------|
| `context.Context` on Generate/Chat | Task 2 (textmodel.go), Task 9 (generate.go) | Integrated |
| `Err() error` on TextModel | Task 2 (textmodel.go), Task 9 (generate.go) | Integrated |
| `Chat()` on TextModel | Task 2 (textmodel.go), Task 9 (generate.go) | Integrated |
| Memory control functions at root | Task 8 (register_metal.go) | Integrated |
| Functional options convention | Confirmed by Virgil — no conflict with core/go | N/A |
| pkg/process for mlxlm | Confirmed by Virgil — no changes needed | N/A |

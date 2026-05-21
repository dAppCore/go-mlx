# Core Inference Contract Parity Design

Date: 2026-05-08
Owner: Core local inference suite
Anchor repo: `/Users/snider/Code/core/go-mlx`
Primary implementation repo: `/Users/snider/Code/core/go-inference`

## Purpose

The Core AI suite has grown enough local inference, training, probing, model
pack, benchmark, and OpenAI-compatible server features that backend-specific
packages must stop owning shared contract shapes. `go-inference` should become
the shared contract package for model-state work so `go-mlx`, `go-rocm`,
`go-ai`, `go-ml`, `api`, and `mcp` can compose without circular dependencies.

The design target is contract parity first, backend implementation parity
second. Backend packages should report the capabilities they truly support
instead of pretending every runtime can expose every model-state feature.

## Goals

- Make `go-inference` the dependency-safe home for shared structs and
  capability interfaces.
- Preserve `go-mlx` as the Apple-native model-state backend.
- Let `go-rocm` keep its current managed `llama-server` ROCm path while gaining
  the same public capability contracts where it can support them.
- Keep `go-ai` focused on "I am using AI" application flows.
- Keep `go-ml` focused on "I am building AI" evaluation, training, scoring, and
  research flows.
- Keep protocol surfaces in `api` and `mcp`, not in backend runtimes.
- Avoid new cgo unless a backend genuinely needs a native runtime boundary.

## Non-Goals

- Do not move MLX tensor, Metal, KV binary layout, prompt cache, or allocator
  internals into `go-inference`.
- Do not force `go-rocm` to fake stateful KV/probe/training capabilities while
  it is backed only by `llama-server`.
- Do not rebuild OpenAI-compatible HTTP or MCP protocol transformation inside
  `go-mlx` or `go-rocm`.
- Do not make `go-inference` depend on `go-mlx`, `go-rocm`, `go-ai`, `go-ml`,
  `api`, or `mcp`.

## Package Boundaries

`go-inference` owns shared contracts:

- `TextModel`, `Backend`, load options, generation options.
- Model, tokenizer, adapter, sampler, and runtime identity structs.
- State bundle metadata structs.
- Probe event structs and probe sink interfaces.
- Dataset stream, batch, and loss-mask contracts.
- Eval, benchmark, memory plan, model fit, and training result structs.
- Capability interfaces such as stateful, probeable, adapter-aware, evaluable,
  benchable, and trainable models.

`go-mlx` implements those contracts with MLX and Metal internals:

- Native model loading, generation, chat, batch, classify.
- KV snapshots, prompt cache, state bundles, and restore checks.
- Probe bus emission.
- SFT LoRA, distillation, GRPO, eval, benchmarking.
- Model packs, memory planning, merge, LoRA fuse, GGUF inspection, and
  quantization.

`go-rocm` implements those contracts in honest layers:

- Current managed `llama-server` path implements text generation, chat, model
  metadata, GGUF discovery, VRAM-aware fit planning, and basic benchmark
  reports where metrics are observable.
- It does not implement stateful KV, native probes, or native training until a
  native ROCm/HIP runtime exists.
- A future native ROCm path can implement additional interfaces without
  changing consumers.

`go-ml` consumes `go-inference` for building AI:

- Evals, scoring, quality probes, training runners, distillation orchestration,
  benchmark aggregation, and research output formats.

`go-ai` consumes `go-inference` for using AI:

- Chat, embeddings, simple app-facing generation, RAG wrappers, and task-level
  AI helpers.

`api` and `mcp` remain protocol surfaces:

- OpenAI-compatible HTTP, MCP tools, Anthropic/OpenAI transformation, SSE, and
  WebSocket transport route into `go-ai`, `go-ml`, or `go-inference`
  contracts, not backend internals.

## Core Contract Types

The first migration should add these backend-neutral structs to `go-inference`.
Where equivalent public structs already exist in `go-mlx`, `go-mlx` should
temporarily type-alias them to `inference` types.

```go
type ModelIdentity struct {
    ID              string
    Path            string
    Architecture    string
    Revision        string
    Hash            string
    QuantBits       int
    QuantGroup      int
    QuantType       string
    ContextLength   int
    NumLayers       int
    HiddenSize      int
    VocabSize       int
}

type TokenizerIdentity struct {
    Kind            string
    Path            string
    Hash            string
    ChatTemplate    string
    BOSID           int32
    EOSID           int32
    PADID           int32
}

type AdapterIdentity struct {
    Path            string
    Hash            string
    Format          string
    Rank            int
    Alpha           float32
    TargetKeys      []string
    BaseModelHash   string
}

type SamplerConfig struct {
    MaxTokens       int
    Temperature     float32
    TopK            int
    TopP            float32
    RepeatPenalty   float32
    StopTokens      []int32
    StopSequences   []string
}
```

Companion structs such as `RuntimeIdentity`, `StateRef`, `ProbeEvent`,
`DatasetStream`, `EvalConfig`, `BenchConfig`, and the training configs should
live in the same package and remain pure metadata or interfaces.

`StateBundle` should contain portable metadata and backend-owned references,
not raw backend tensors:

```go
type StateBundle struct {
    Version         string
    CreatedAtUnix  int64
    Model          ModelIdentity
    Tokenizer      TokenizerIdentity
    Adapter        AdapterIdentity
    Sampler        SamplerConfig
    PromptHash     string
    PromptTokens   int
    GeneratedTokens int
    Runtime        RuntimeIdentity
    KVRefs         []StateRef
    ProbeRefs      []StateRef
    StateRefs     []StateRef
    Labels         map[string]string
}
```

## Capability Interfaces

Capability interfaces keep feature parity explicit and prevent consumers from
needing backend-specific imports.

```go
type TokenizerModel interface {
    Encode(text string) []int32
    Decode(ids []int32) string
    ApplyChatTemplate(messages []Message) (string, error)
}

type AdapterModel interface {
    LoadAdapter(path string) (AdapterIdentity, error)
    UnloadAdapter() error
    ActiveAdapter() AdapterIdentity
}

type StatefulModel interface {
    CaptureState(ctx context.Context, prompt string, opts ...GenerateOption) (*StateBundle, error)
    RestoreState(ctx context.Context, bundle *StateBundle) error
}

type ProbeSink interface {
    EmitProbe(event ProbeEvent)
}

type ProbeableModel interface {
    SetProbeSink(sink ProbeSink)
}

type Evaluator interface {
    Evaluate(ctx context.Context, dataset DatasetStream, cfg EvalConfig) (*EvalReport, error)
}

type BenchableModel interface {
    Benchmark(ctx context.Context, cfg BenchConfig) (*BenchReport, error)
}
```

Training contracts should split orchestration from tensor execution:

- `go-inference` owns config, metadata, checkpoint, and result structs for SFT,
  distillation, and GRPO.
- Backend packages own tensor/autograd execution.
- `go-ml` orchestrates high-level workflows over the capability interfaces.

## Capability Matrix

| Capability | go-mlx now | go-rocm managed now | go-rocm native later |
|---|---:|---:|---:|
| Text generation | yes | yes | yes |
| Chat templates | yes | llama-server dependent | yes |
| Model identity | yes | yes | yes |
| Adapter identity | yes | partial if server exposes it | yes |
| Load/unload LoRA | yes | server dependent | yes |
| State bundle metadata | yes | metadata only | yes |
| KV snapshot/restore | yes | no | yes |
| Prompt cache | yes | no | yes |
| Probe events | yes | limited metrics only | yes |
| Dataset stream | yes | contract consumer | contract consumer |
| Eval reports | yes | yes through generation | yes |
| Bench reports | yes | yes for observable metrics | yes |
| Memory fit plan | yes | yes from GGUF + VRAM | yes |
| SFT LoRA training | yes | no | yes |
| Distillation | yes | teacher/student orchestration only | yes |
| GRPO | experimental | no | experimental |

## Migration Plan

1. Add contract structs to `go-inference`.
   - Start with identity, sampler, probe, state bundle metadata, dataset, eval,
     bench, memory fit, and training config/result structs.
   - Preserve JSON tags from existing `go-mlx` public structs where possible.
   - Add focused unit tests and examples for each public type.

2. Add capability interfaces to `go-inference`.
   - Keep interfaces small and opt-in.
   - Consumers must type-assert capabilities instead of assuming a backend can
     do everything.

3. Adapt `go-mlx`.
   - Type-alias moved public structs to `inference` equivalents.
   - Keep MLX-specific execution and storage internals private.
   - Add compile-time interface assertions for supported capabilities.

4. Adapt `go-rocm`.
   - Implement the shared metadata, fit, and benchmark contracts where the
     current managed path can do so honestly.
   - Return non-implementation by absence of interface support, not runtime
     "not implemented" errors.
   - Keep native ROCm/HIP work isolated behind future build tags and package
     boundaries.

5. Adapt consumers.
   - Move `go-ml` eval, probe, training, benchmark, and server code to consume
     `go-inference` shared structs.
   - Move the unfinished `go-ai` API provider routes onto `go-inference` and `go-ml`
     contracts.
   - Keep `api` and `mcp` as protocol adapters.

## Testing Strategy

- `go-inference`: pure Go unit tests and runnable examples, no GPU.
- `go-mlx`: existing normal tests plus opt-in native Metal tests.
- `go-rocm`: pure Go tests for discovery, contracts, GGUF metadata, and managed
  server request construction; opt-in ROCm tests behind explicit tags.
- `go-ml`: mock `inference.TextModel` and capability interfaces for orchestration
  tests.
- `go-ai`, `api`, and `mcp`: handler and transformer tests using fake contract
  implementations.

Each repo should continue to run with `GOWORK=off`. Contract changes should land
from the inside out: `go-inference` first, backend adapters second, consumers
last.

## Risks And Controls

- Risk: `go-inference` becomes a dumping ground.
  Control: it only owns portable data and narrow interfaces, never backend
  execution.

- Risk: shared contracts leak MLX-specific details.
  Control: backend-owned binary/tensor formats are stored as typed references
  and metadata, not raw implementation structs.

- Risk: ROCm parity is overstated.
  Control: capability interfaces are opt-in; managed ROCm exposes only what it
  can prove.

- Risk: consumers keep importing `go-mlx` directly.
  Control: move shared structs first, then add tests that exercise `go-ml` and
  `go-ai` through `go-inference` contracts.

- Risk: cgo spreads.
  Control: native boundaries stay in backend packages. Shared contracts remain
  pure Go.

## Acceptance Criteria

- `go-inference` owns all shared structs needed by model-state, eval, bench,
  dataset, and training orchestration.
- `go-inference` imports no backend or consumer package.
- `go-mlx` compiles after replacing duplicated public contracts with aliases or
  adapters.
- `go-rocm` reports a truthful capability matrix through interface support.
- `go-ml` can run eval/bench/training orchestration over `inference` contracts
  without importing backend-specific structs.
- `go-ai`, `api`, and `mcp` route through the shared contracts instead of
  backend internals.
- Normal repo gates pass with `GOWORK=off`.

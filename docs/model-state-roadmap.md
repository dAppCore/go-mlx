---
title: Model State Roadmap
description: Native Apple model-state runtime work needed to remove Python from local inference and training workflows.
---

# Model State Roadmap

go-mlx should own the Apple-native model runtime: Metal execution, tokenizer
binding, prompt/KV cache, low-level probes, LoRA training primitives, model
packing, and reproducible state artifacts. Higher packages should use those
capabilities rather than reimplement them.

## Package Boundaries

| Package | Boundary |
|---------|----------|
| `dappco.re/go/mlx` | Native Metal model loading, inference, KV/session state, probing hooks, LoRA execution, memory planning |
| `dappco.re/go/inference` | Stable backend interfaces and portable option/config contracts |
| `dappco.re/go/core/ml` | Scoring, capability probes, benchmark runners, and backend adapters |
| `dappco.re/go/core/ai` | Agent orchestration, MCP/RAG tooling, and operational event metrics |

This split keeps go-mlx useful as a standalone binary/library while letting
`go-ml` and `go-ai` consume richer native capabilities through narrow
interfaces.

## Roadmap

### 1. Restorable Session State

Add APIs that can put a saved KV snapshot back into a live `ModelSession`.
The target shape is:

- `Model.NewSessionFromKV(snapshot *KVSnapshot)`
- `Model.NewSessionFromBundle(bundle *StateBundle)`
- `ModelSession.RestoreKV(snapshot *KVSnapshot)`
- `ModelSession.LoadKV(path string)`
- `ModelSession.RestoreBundle(bundle *StateBundle)`
- `ModelSession.LoadBundle(path string)`

This turns saved KV from an observation artifact into reusable model state. It
also makes shared context, branching, and cold-start acceleration practical for
long local workflows.

### 2. State Bundle Format

Wrap KV data and metadata into a portable state bundle:

- model identity and architecture
- tokenizer hash and chat-template hash
- prompt hash and token range
- generation sampler config
- LoRA adapter identity
- KV snapshot reference or embedded KV payload
- SAMI/probe metrics
- memvid refs for cold storage

The bundle is versioned and hash-checked. Embedded KV payloads are validated on
load, and external KV paths are checked when `Snapshot()` resolves them.
`ModelSession.ExportBundle` captures the current live session into this schema,
while `StateBundleFileHash` can pin external tokenizer/model-pack files by
content hash.

### 3. Probe Bus

Expose a typed event stream around inference and training:

- token events and selected token IDs
- logits summaries and entropy
- KV/head/layer coherence
- router decisions for MoE models
- memory pressure and cache pressure
- training loss and gradient summaries

go-mlx should generate the low-level probe events. `go-ml` should score and
aggregate them.

### 4. Native LoRA Training Runner

Promote the existing LoRA, autograd, and AdamW primitives into a public Go
training runner:

- dataset/token stream
- prompt/completion masking
- gradient accumulation
- checkpoint/resume
- LoRA save and merge
- eval prompts after N steps
- optional probe emission during training

Start with SFT LoRA. Keep full fine-tuning and alternate optimizers as later
extensions.

### 5. Model Pack Tooling

Add validation and packaging for local model directories:

- `config.json` support check
- tokenizer and chat-template validation
- safetensors/GGUF shard detection
- quantization metadata
- context and cache recommendations
- architecture-specific warnings for Gemma/Qwen

This should support a no-Python install path: a user gets a model directory and
a go-mlx binary, then runs.

### 6. Memory Planner

Add an opinionated local hardware planner. It should inspect Apple GPU/unified
memory data and choose:

- context length
- cache mode and cache limits
- parallel slot count
- prefill/batch sizes
- memory hard limits
- prompt cache size

The defaults should target real local machines: M1 16 GB, M-series Pro/Max,
and M3/M4 Ultra class systems.

### 7. Fast Eval and Benchmark Harness

Add first-party local benchmarks for:

- prefill tokens/sec
- decode tokens/sec
- peak active Metal memory
- prompt-cache hit rate
- KV restore latency
- state-bundle round trip
- probe overhead
- small capability/eval suites

This gives beta testers a repeatable command that reports whether a machine,
model, and go-mlx build are behaving correctly.

## First Slice

Restorable session state and state bundles are the foundation for reusable
knowledge packs, exact repro artifacts, probe replay, and training checkpoints.
The remaining work is to extend them with tokenizer/chat-template hashes and a
store-backed `PrefillOrRestore` helper once model-pack metadata is first-class.

<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# go-mlx — documentation index

**Module**: `dappco.re/go/mlx`
**Role**: Native Apple Metal GPU inference + research-grade training pipeline. Implements the go-inference `Backend` + `TextModel` + `Session/Forker` contracts for darwin/arm64.

## Tetrad position

```
                    ┌──────────────────────────────┐
                    │      dappco.re/go (core)     │
                    └──────────────┬───────────────┘
                                   │
                    ┌──────────────┴────────────────┐
                    │     go-inference  (contract)  │
                    └──┬─────────────┬──────────────┘
                       │             │ register via init()
              ┌────────┴───┐  ┌──────┴────────┐
   you are here →  go-mlx  │  │  go-rocm /    │
                    │  darwin │  │  go-cuda      │
                    │  arm64  │  │  (planned)    │
                    └─────┬──┘  └───────────────┘
                          │ consumed by
                    ┌─────┴──────────┬────────────────┐
                    │  go-ml         │  go-ai          │
                    │  scoring/agent │  router/demos   │
                    └────────────────┘ └───────────────┘
```

## What this package owns

Five distinct areas, each with its own doc subtree:

| Area | Owns | Doc |
|------|------|-----|
| `runtime/` | Backend registration + adapter + Metal allocator | [runtime/README.md](runtime/README.md) |
| `memory/` | KV snapshots + bundles + memvid + Wake/Sleep/Fork/Fold | [memory/README.md](memory/README.md) |
| `moe/` | MiniMax M2 + JANG/JANGTQ + codebook VQ + expert residency | [moe/README.md](moe/README.md) |
| `training/` | SFT + GRPO + distillation + LoRA + eval + merge | [training/README.md](training/README.md) |
| `model/` | Model-pack validation + memory planning + GGUF | [model/README.md](model/README.md) |
| `inference/` | Scheduler + block cache + decode opt + parsers + thinking | [inference/README.md](inference/README.md) |
| `compute/` | Non-LLM Metal compute (pixel buffers, kernels, frame pipelines) | [compute/compute.md](compute/compute.md) |
| `observability/` | Probe emission (token / entropy / heads / router / cache / memory / training) | [observability/probe.md](observability/probe.md) |
| `cmd/` | Sidecar daemons | [cmd/violet.md](cmd/violet.md) |

## Mental model

```
                  ┌─────────────────────────────────┐
                  │  caller: inference.LoadModel    │
                  └──────────────┬──────────────────┘
                                 │
              ┌──────────────────┴───────────────────┐
              │      go-inference Default()           │
              │   picks "metal" → metalbackend        │
              └──────────────────┬───────────────────┘
                                 │
                    runtime/ (register_metal.go)
                                 │
                                 ▼
              ┌──────────────────────────────────────┐
              │ memory_plan → load weights via       │
              │ medium → metal.LoadAndInit → produce │
              │ &metaladapter wrapping metal.Model    │
              └──────────────────┬───────────────────┘
                                 │
        ┌────────────┬───────────┴────────┬──────────────┐
        ▼            ▼                    ▼              ▼
   inference/   memory/             training/       observability/
   (scheduler   (Wake/Sleep         (SFT/LoRA/      (probe events)
    cache       bundles             GRPO/distill/
    decode-opt  memvid)              eval)
    parsers
    thinking)

   moe/ adds MoE-specific paths into each area.
   compute/ runs alongside on the same Metal device.
```

## Status snapshot (2026-05-11)

**Production**: dense models (Gemma 3/4 dense, Qwen 2/3, Llama 3) — load, inference, scheduler, block cache, KV snapshots, agent memory wake/sleep/fork, SFT, LoRA, distillation, GRPO, eval, model pack validation, GGUF read+write, memory planning, frame compute. Qwen 3.6 model packs are recognised and planned through the `mlx_lm` fallback while native hybrid linear-attention kernels are pending.

**Phase 1 in flight** (vMLX parity sprint, started 2026-05-09): MiniMax M2/2.7 MoE forward, JANGTQ_K weight load, codebook VQ kernels, expert residency native path, disk-backed block cache.

**Planned**: speculative decoding (paired with Gemma 4 `-assistant`), prompt-lookup decoding, embeddings + rerank surfaces, OpenAI Responses handler, vision/audio (out-of-scope for core runner near-term).

## Repository layout

```
go-mlx/
├── go/                     Go module root (dappco.re/go/mlx)
│   ├── *.go                ← root package (80+ files, this is where docs land)
│   ├── internal/metal/     ← CGO bindings to mlx-c (44 files, internal)
│   ├── mlxlm/              ← CGO-free Python subprocess fallback
│   ├── cmd/violet/         ← Unix-socket sidecar daemon
│   ├── cmd/mlx/            ← CLI tool (built with `-o core-mlx`; consumers rename: lthn-mlx, etc.)
│   ├── pkg/daemon/         ← daemon implementation
│   ├── pkg/memvid/         ← QR-video knowledge-pack codec
│   └── tests/              ← integration tests
├── cpp/                    C++ companion (CLion-side)
├── docs/                   ← YOU ARE HERE
├── examples/               per-feature usage walkthroughs
├── external/               vendored core libraries
├── lib/mlx/                upstream MLX submodule (v0.31.1)
└── patches/                local patches to lib/mlx
```

## Where to start

- **Caller (loading a model)** → [`runtime/register_metal.md`](runtime/register_metal.md) + [`runtime/adapter.md`](runtime/adapter.md)
- **Local setup / autotune UI** → [`runtime/local_autotune.md`](runtime/local_autotune.md)
- **Agent memory / book state** → [`memory/agent_memory.md`](memory/agent_memory.md)
- **LTHN project context seed** → [`memory/agentic_project_seed.md`](memory/agentic_project_seed.md)
- **Training Vi or a custom model** → [`training/README.md`](training/README.md) → [`training/sft.md`](training/sft.md) → [`training/distill.md`](training/distill.md)
- **Understanding the vMLX parity work** → [`moe/README.md`](moe/README.md) + `docs/vmlx-feature-gap-report.md`
- **Serving many requests** → [`inference/scheduler.md`](inference/scheduler.md)
- **Frame compute (emulator UIs)** → [`compute/compute.md`](compute/compute.md)
- **Sidecar deployment** → [`cmd/violet.md`](cmd/violet.md)

## Legacy docs

The flat docs in this folder (`architecture.md`, `compute.md`, `distillation.md`, `grpo.md`, `models.md`, `training.md`, `eval.md`, `model-operations.md`, `model-state-roadmap.md`, `build.md`, `development.md`, `history.md`, `index.md`, `vmlx-feature-gap-report.md`, `superpowers/plans/2026-05-09-vmlx-feature-parity.md`) pre-date this per-file pass and may rot. Keep `vmlx-feature-gap-report.md` and the parity plan (they're active references). Fold the rest into the per-package READMEs over time.

## Measured

| Operation | Bundle / model | Latency |
|-----------|----------------|---------|
| Wake — chapter (warm) | ~500MB | 998ms |
| Wake — full book (warm) | ~10.5GB | 2.15s |
| Wake — full book (cold runner) | ~10.5GB | 55.2s |
| Sleep — incremental, parent-reuse | 200-token delta | <1s |
| Gemma 4 E2B inference (M3 Ultra) | dense | ~80 tok/s decode |
| Gemma 4 26B inference (M3 Ultra) | dense | ~25 tok/s decode |

## Standards

- UK English in code, comments, docs (colour, organisation, licence, serialise)
- SPDX header on every new file: `// SPDX-Licence-Identifier: EUPL-1.2`
- Conventional commits: `type(scope): description` — scopes per package + `metal`, `api`, `mlxlm`, `repo`, `deps`
- Test triplets: `_Good` / `_Bad` / `_Ugly` + `*_example_test.go` runnable examples
- Error wrapping via `core.E(scope, msg, cause)`
- Co-Author: `Co-Authored-By: Virgil <virgil@lethean.io>`
- Native files: `//go:build darwin && arm64` (or `&& !nomlx`); stubs return false on `MetalAvailable()`
- CGO confined to `go/internal/metal/`

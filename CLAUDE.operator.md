# CLAUDE.operator.md

Operator-facing guidance for **running** `lthn-mlx` in production. Companion to `CLAUDE.md` (developer-facing — architecture, build, contribute). If you arrived here mid-session needing to deploy, troubleshoot, or reason about distribution, you're in the right doc. If you arrived needing to add a model decoder or change cgo bindings, go to `CLAUDE.md`.

The operator audience is a future Cladius / Athena / Hephaestus session, *or* a human operator (Snider, ops-side) doing a deploy. Same mental model serves both — the difference is just whether the reader can edit code on the spot.

## Read order

1. **This file**, skim through "Operating principles" — calibrates what the binary is and isn't.
2. **`docs/operator/deployment.md`** — what you ship, how it runs, what to bind to.
3. **`docs/operator/metallib-and-variants.md`** — the variant question, the bundling strategy, the active CWD-resolution panic.
4. **`docs/operator/troubleshooting.md`** — the failure modes in lifecycle order, with fixes.
5. **`docs/operator/index.md`** — the full operator doc set + what's planned.

If you have ~3 minutes, read this file. If you have ~30 minutes, read all five.

## What lthn-mlx is

A single-process boundary that wraps native Apple Metal GPU inference (via mlx-c CGO bindings) and serves it as OpenAI / Anthropic / Ollama-compatible HTTP. Snider's framing, made explicit on 2026-05-25:

> **"The actual model is the binary, the rest is package."**

This is the load-bearing architecture decision. Everything that wants inference — `lthn` desktop, `pkg/lemma` in lthn/desktop, providers in `go-ai`, any OpenAI-compatible Python / TypeScript / curl client — talks to `lthn-mlx` over HTTP. There is no in-process library substitute for production. The binary is the boundary.

**One process. One model. One HTTP listener.** That's the unit. Multi-model deployments mean multiple processes on different ports plus a router in front (the `pkg/lemma` client is the canonical Go-side router).

The binary is built from `dappco.re/go/mlx/cmd/mlx`, default output name `core-mlx`, consumers rename to `lthn-mlx`. Module path is `dappco.re/go/mlx`.

## Operating principles

These are the load-bearing facts an operator needs in working memory. Each one shapes a deployment decision.

### 1. Apple Silicon only

`darwin/arm64`. No Linux. No Intel macOS. The CGO files carry `//go:build darwin && arm64`; a stub returns `MetalAvailable() = false` everywhere else. M1 / M2 / M3 / M4, any chip class, any deployment macOS ≥13 — one binary serves them all (modulo the metallib variant matrix; see point 5).

If the deployment target isn't Apple Silicon, you don't want `lthn-mlx` — you want a different go-inference backend (`go-rocm` for AMD GPUs, or the CGO-free `mlxlm` subprocess backend bundled in the same repo for Python-on-anything).

### 2. The binary needs the metallib

`mlx.metallib` (~107 MB, MetalLib v1.2.9, the compiled GPU kernel archive) must be findable at runtime. Today, until the bundling work lands, this means **setting `MLX_METALLIB_PATH` to an absolute path** before invoking. Not setting it is the single most common deployment failure — the binary starts, `/v1/health` passes, then panics inside `mlx_metal_load_library` on the first GPU dispatch.

```bash
export MLX_METALLIB_PATH=/opt/lthn-mlx/lib/mlx.metallib
lthn-mlx serve --model /opt/lthn-mlx/models/lemer-lite --addr :11434
```

The permanent fix is Path B bundling (embed via `//go:embed`, load via `MTLDevice newLibraryWithData:`). Until that ships, treat the env var as mandatory deployment config. See `docs/operator/metallib-and-variants.md` for the why and `docs/operator/troubleshooting.md` for the panic signature.

### 3. Model loads lazily

`lthn-mlx serve` starts in under a second. The model loads on the **first request that needs it**, not at process start. This means:

- Liveness probes against `/v1/health` pass before the model is loaded. They are not readiness probes.
- The first inference request after start takes 2-15 seconds depending on model size and storage speed.
- For consistent first-request latency, pre-warm in the service manager's post-start hook with a one-token completion (see deployment.md).

There is no on-disk lock, no PID file, no recovery state. Restart is safe; the new process starts cold and lazy-loads. The service manager is responsible for single-instance enforcement.

### 4. HTTP surface is trusted-network only

`lthn-mlx serve` has no authentication, no rate limiting, no TLS. Default bind is `:11434` (matches Ollama). Bind to `127.0.0.1:11434` for same-machine, `0.0.0.0:11434` for LAN. **Production LAN exposure sits behind a reverse proxy** that handles auth and TLS (Caddy, nginx).

If you need authenticated remote access, that lives in `pkg/lemma` (the Go client) plus a tunnel / proxy / auth-gateway — not in `lthn-mlx` itself. Don't try to add auth to the serve binary; it would violate the boundary rule and duplicate work already done one layer up.

### 5. Variants matter at the toolchain axis, not the chip axis

Snider's question of 2026-05-25: "if the lib is different for different apple versions, we need to know the variants that need building." The chip family (M1/M2/M3/M4) is **not** a variant axis — Apple's Metal driver handles forward-compatibility from a single archive. What actually varies is the build-host toolchain: Metal language version ≥4.0 + macOS SDK ≥26.2 (Xcode 26+) unlocks the NAX kernel family for M4-class tensor coprocessors.

**Practical ship matrix:**

| Variant | Build host | Runs on | Use case |
|---------|------------|---------|----------|
| `mlx-baseline.metallib` | Any modern Xcode, deployment-min 13 | M1-M4 on macOS 13+ | Default ship today |
| `mlx-nax.metallib` | Xcode 26+, deployment-min 26 | M4-class on macOS 26+ only | Deferred to M4 optimisation lane |

Ship the baseline. The NAX variant is a future M4 fast-path optimisation, not a today-decision. Full evidence and the open questions (driver-side load behaviour for higher `min`, NAX dispatch gating on non-M4) in `docs/operator/metallib-and-variants.md`.

### 6. Unified memory is the budget

On Apple Silicon there is no separate VRAM line item — the GPU and CPU share unified memory. The process budget includes: model weights, KV cache (scales linearly with `--context`), MLX allocator cache, plus everything else macOS is doing. A 7B model in 4-bit needs ~5 GB resident; the default 131k context can add several more.

Tuning knobs live in `dappco.re/go/mlx` at the package level (`SetMemoryLimit`, `SetCacheLimit`, `SetWiredLimit`, `ClearCache`, `GetActiveMemory`, `GetPeakMemory`). They are **not** exposed as `serve` flags today — if you need them on the bundled CLI, file a feature ticket against `cmd/mlx/serve.go`. For now, custom integrations on top of `openai.NewMuxWithAdmin` can wire them directly.

Activity Monitor's "Memory" column is the right place to watch the process. `/v1/cache/stats` reports MLX's allocator view.

### 7. Graceful shutdown is signal-driven

SIGINT and SIGTERM both trigger `http.Server.Shutdown` with `--shutdown-timeout` (default 10s) as the drain deadline. After the deadline, the process exits. There is no explicit model-unload step — the OS reclaims Metal allocations on exit.

If you have long-running generations and need them to drain cleanly on bounce, raise `--shutdown-timeout` (30s-60s). If you need explicit teardown for an exotic daemon scenario, wire the `Sleep` admin callback in a custom integration.

## Mental model in one paragraph

`lthn-mlx serve` is a stateless OpenAI-compatible HTTP server backed by Apple Metal GPU inference, single-model per process, lazy-load on first request, signal-driven graceful shutdown, requires a findable `mlx.metallib` (env var until bundling lands), no built-in auth or TLS, designed for trusted-network use, with a `pkg/lemma`-shaped routing layer one level up for multi-model or remote-access patterns. The architecture insists on the binary as the only process boundary — everything else is packages talking to it over HTTP.

That paragraph plus the seven principles is the working mental model. Everything else in `docs/operator/` fills in the operator's view of specific concerns.

## What this doc does not cover

- **How the inference works inside.** That's `docs/architecture.md`, `docs/runtime/`, `docs/memory/`. Developer-side.
- **How to add a model architecture.** That's a decoder under `go/internal/metal/`. Developer-side.
- **How training works.** That's `docs/training.md`, `docs/distillation.md`, `docs/grpo.md`. Production-bench / research-side.
- **GOAL.md production-bench lane.** Separate concern with its own canonical brief.
- **Memory limits & cache tuning as a knob set.** Stubbed in `docs/operator/performance-tuning.md` — not yet written. Source of truth meanwhile: `go/internal/metal/backend.go:10-12` and the `mlx.Set*` package surface.

## When the docs and reality disagree

This doc and `docs/operator/*` describe behaviour. Behaviour changes. If you find a discrepancy between what `lthn-mlx serve` actually does and what these docs claim, **the code is right and the docs are wrong**. Fix the doc, or PR a comment-block on the responsible source file referencing this directory.

The maintenance discipline lives in `docs/operator/index.md` under "Maintenance discipline." Read it if you're about to merge a PR that touches `cmd/mlx/serve.go`, `go/openai/openai.go`, `go/openai/admin.go`, or `go/internal/metal/backend.go` — those four files are the operator-visible surface.

## Files this directory ships

- `CLAUDE.operator.md` (this file) — operator mental model
- `docs/operator/index.md` — operator doc index + planned slots
- `docs/operator/deployment.md` — what you ship + how it runs
- `docs/operator/metallib-and-variants.md` — bundling strategy + variant matrix
- `docs/operator/troubleshooting.md` — lifecycle-phase failure modes

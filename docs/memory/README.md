<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# memory/ — KV snapshots, bundles, agent memory

**Package**: `dappco.re/go/mlx` (these files live in the root)

## What this area owns

Everything that turns **live runtime state** into **durable bytes** and back. This is the production implementation of the `inference/state.Session` and `state.Forker` contracts plus the go-mlx folded-state handoff for exhausted windows — the surface that delivers AI-cognition-as-filesystem-object.

```
                  Live metal.Model
                        │
                        ▼
        ┌─────────────────────────────┐
        │ CaptureKVSnapshot →         │ kv_snapshot.go
        │   K/V bytes per layer       │
        └─────────────────────────────┘
                        │
                        ▼
        ┌─────────────────────────────┐
        │ Chunk to blocks             │ kv_snapshot_blocks.go
        │   256-token spans + hashes  │
        └─────────────────────────────┘
                        │
                        ▼
        ┌─────────────────────────────┐
        │ Wrap in Bundle envelope     │ state_bundle.go
        │   ModelID + TokID + refs    │
        └─────────────────────────────┘
                        │
                        ▼
        ┌─────────────────────────────┐
        │ Index into BundleIndex      │ kv_snapshot_index.go
        │   URI → entry → blocks      │
        └─────────────────────────────┘
                        │
                        ▼
        ┌─────────────────────────────┐
        │ Encode + write to Store     │ kv_snapshot_state.go
        │   (State video / file / mem)     │ medium.go
        └─────────────────────────────┘

        ▲                            ▼
        └── Wake reverses ─── Sleep/Fold return
            the same chain          Bundle
            (session_agent.go)
```

## File map

| File | Doc | Role |
|------|-----|------|
| `session_agent.go` | [agent_memory.md](agent_memory.md) | Wake / Sleep / Fork / Fold — the lifecycle entry |
| `kv_snapshot.go` | [kv_snapshot.md](kv_snapshot.md) | Snapshot binary format (magic, version, encoding) |
| `kv_snapshot_blocks.go` | [kv_snapshot_blocks.md](kv_snapshot_blocks.md) | Chunk strategy + block hashing |
| `kv_snapshot_index.go` | [kv_snapshot_index.md](kv_snapshot_index.md) | Bundle index across entries + parents |
| `kv_snapshot_state.go` | [kv_snapshot_state.md](kv_snapshot_state.md) | State video integration |
| `state_bundle.go` | [state_bundle.md](state_bundle.md) | JSON envelope encode/decode |
| LTHN project seed | [agentic_project_seed.md](agentic_project_seed.md) | Agentic wake/reload/compact workflow |
| `medium.go` | [medium.md](medium.md) | Load model files via io.Medium (S3 / local / State video / …) |
| `kv_analysis.go` | (planned) | KV inspection utilities — entropy, layer balance |
| `kv_cache_bench.go` | (planned) | KV cache benchmark harness |
| `state_chapter_smoke.go` | (planned) | Smoke test fixtures for State bundles |
| `small_model_smoke.go` | (planned) | Smoke test fixtures for compact bundles |

## Why this area exists at all

The thesis: a model's **runtime state IS a filesystem object**. Once the KV cache + sampler + tokenizer state is durable, you can:

- Sleep an agent's session, walk away for a week, wake it, continue — no re-prompt.
- Mass-distribute a knowledge pack as a `.mp4` — phones can scan it; HTTP can stream it; YouTube can host it.
- Fork an agent into 100 divergent continuations from one parent — no re-prefill of the shared prefix.
- Fold an exhausted window into a fresh summary-plus-tail state while keeping
  the exact checkpoint for audit/replay.
- Train one base model + 50 personality bundles → users wake whichever persona fits the task.
- Seed a project agent with operator + repository memory, then checkpoint only
  the new suffix after each task.

Every file in this directory exists to make that thesis cheap, fast, and portable.

## Measured

- Wake (warm cache, chapter) — 998ms
- Wake (warm cache, full book ~10.5GB) — 2.15s
- Wake (cold runner, full book) — 55.2s (first-time decode included)
- Sleep (incremental, 200-token delta, parent-reuse on) — <1s

See [`agent_memory.md`](agent_memory.md) for context on what's being measured.

## Related contracts

- `../../../go-inference/docs/state/` — portable shape this implements
- `../../../go-inference/docs/state/agent_memory.md` — the Session + Forker interfaces
- `../../../go-inference/docs/state/identity.md` — Bundle DTO
- `../../../go-inference/docs/state/store.md` — Store / Resolver / Writer interfaces
- [`agentic_project_seed.md`](agentic_project_seed.md) — LTHN app/CLI workflow for project context seeds
- `cmd/violet/` — Unix-socket sidecar exposing wake/sleep over IPC
- `pkg/memvid/` (deprecated compatibility path) — the QR-video codec

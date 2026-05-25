---
title: Operator docs for lthn-mlx
description: Index for the operator-facing documentation set. Complementary to docs/index.md (developer-facing). Read CLAUDE.operator.md at the repo root first.
---

# Operator docs for lthn-mlx

Documentation for **running** `lthn-mlx` in production — not for hacking on its internals. Complementary to the developer-facing material at [`docs/index.md`](../index.md). If you arrived here looking for "how do I add a new model architecture" or "how does lazy evaluation work," go there instead.

Start at the repo root: [`CLAUDE.operator.md`](../../CLAUDE.operator.md) — the operator mental model in one document.

## What's here

### Shipped

- [Metallib & variants](metallib-and-variants.md) — what `mlx.metallib` is, the variant matrix (chip family doesn't matter; toolchain does), the bundling strategy (Path A → Path B), the active CWD-resolution panic and its env-var workaround.
- [Deployment](deployment.md) — what files you ship, the `serve` command surface, the HTTP route catalogue, graceful shutdown, launchd patterns, resource expectations.
- [Troubleshooting](troubleshooting.md) — failure modes grouped by lifecycle phase. Each is shaped: symptom → cause → fix. The active blockers are flagged.

### Planned (not yet written)

These slots exist in the operator mental model but aren't drafted yet. If you reach for one and it isn't here, look at the source-of-truth pointer in the row, then either inline the answer for now or PR a doc to this directory.

| Doc | Source of truth in the meantime | Why it's worth writing |
|-----|---------------------------------|------------------------|
| `performance-tuning.md` | `go/internal/metal/backend.go:10-12` (defaults), `docs/memory/*` | The Metal cache, memory limits, parallel-slots, prompt-cache-min-tokens knobs need a unified operator view. Today they're spread across the developer docs and the source. |
| `version-cascade.md` | Snider's manual squash workflow (`project_forge_squash_workflow.md`) | The discipline for cascading a tagged go-mlx release through downstream consumers (`pkg/lemma`, `lthn/desktop`, `go-ai` providers). Includes the metallib-rebuild-on-MLX-bump rule. |
| `multi-model-routing.md` | `pkg/lemma` in lthn/desktop (consumer side); `cmd/mlx/serve.go` (server side, single-model only) | The pattern for running multiple `lthn-mlx` instances on different ports for different models, and the lemma-side routing that picks between them. |
| `observability.md` | `docs/observability/probe.md`, `/v1/cache/stats`, `mlx.GetActiveMemory`, `mlx.GetPeakMemory` | What to log, what to scrape, what alarms to set. Cache hit rate, generation latency p50/p95, memory peaks. |
| `model-management.md` | `docs/model/`, `docs/model-operations.md` | The lifecycle from HuggingFace download → quantisation → on-disk layout → ready-to-load. Includes the `pack` and `gguf-quantize` CLI subcommands. |
| `upgrade-runbook.md` | The deployment doc + this index | Step-by-step for replacing a running `lthn-mlx` binary in place: which file to replace first, when to bounce, how to roll back if the new binary panics. |
| `hardware-matrix.md` | The serve binary's published baselines, plus per-chip-family observed numbers | What to expect on M1 / M2 / M3 / M4 (base / Pro / Max / Ultra) for the common model sizes. Operators provisioning hardware need this. |

Author convention for new operator docs: lead with the operator's question, not the system's structure. "How do I tune memory" beats "Memory architecture overview." If you find yourself writing a long lead-in before getting to the answer, the doc shape is wrong.

## Maintenance discipline

These docs describe behaviour. Behaviour changes. When `cmd/mlx/serve.go` gains a flag, when a default in `internal/metal/backend.go` shifts, when an HTTP route is added or removed, **the operator docs lag by a session at most**. The forcing function: every PR touching `serve.go`, `openai/openai.go`, `openai/admin.go`, or `internal/metal/backend.go` should grep this folder for the changed symbol and update or PR-comment.

The two failure modes to avoid:

1. **Stale-by-omission** — a route exists but isn't in `deployment.md`. Operator hits it via curl and there's no documented behaviour to compare against.
2. **Stale-by-error** — a route used to behave one way, now behaves differently, and the doc still says the old thing. Worse than absent; operator trusts the doc and misdiagnoses.

If you spot drift, fix it in the same PR as the behaviour change. If you spot drift in a PR that's not yours, comment-block until either the author fixes it or files a Mantis ticket against this doc.

## Cross-references

- [`CLAUDE.operator.md`](../../CLAUDE.operator.md) — start here for the mental model
- [`docs/index.md`](../index.md) — developer-facing index (architecture, build, contribute)
- [`docs/runtime/`](../runtime/) — runtime internals (developer-side, not operator-side)
- [`docs/memory/`](../memory/) — KV cache, snapshots, state bundles (developer-side, but the memory limits are operator concerns)
- [`docs/observability/probe.md`](../observability/probe.md) — probe surface, not yet operator-shaped

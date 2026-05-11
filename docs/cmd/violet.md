<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# cmd/violet — local-native inference sidecar

**Package**: `dappco.re/go/mlx/cmd/violet`
**Files**: `cmd/violet/main.go` (entry) + `pkg/daemon/` (server)

## What this is

The **Violet sidecar daemon** — a long-running process exposing inference + agent memory over a Unix socket. Lets local processes (CoreAgent, IDE, ml lab) call into a hot, model-loaded mlx runtime without each spawning their own.

Violet is what Cladius posts to instead of burning Anthropic tokens for routine inference. It's the local substrate that survives Codex's uncertain status (per `project_codex_status_uncertain.md`) and the budget pressure (per `project_go_mlx_research_grade.md`).

## Why a daemon

Three reasons one shared process beats N short-lived processes:

1. **Model load cost.** Loading Gemma 4 26B takes 30-60s on first touch. The daemon pays it once.
2. **KV cache locality.** Sessions retain their KV across requests; a fresh process can't.
3. **Memory budget.** Two LLM processes don't fit on a 96GB Ultra; one daemon serving many clients does.

## Transport

Unix domain socket — fast, secure-by-default (filesystem permissions), no TCP overhead.

```bash
violet --socket /var/run/violet/violet.sock --config /etc/violet.toml
```

Request envelope is line-delimited JSON over the socket; responses likewise (or SSE-like multi-line for streaming).

## Surface

Per-request operations (subset, more land as parity sprint completes):

- `Generate` / `Chat` — text generation
- `Classify` / `BatchGenerate`
- `WakeState` / `SleepState` / `ForkState` — agent memory
- `CacheStats` / `WarmCache` / `ClearCache` — prompt cache
- `CapabilityReport` — what this daemon supports right now
- `LoadModel` / `UnloadModel` — admin (default off, opt-in via config)

## Config

```toml
# /etc/violet.toml

[runtime]
socket = "/var/run/violet/violet.sock"
default_model = "gemma-4-e2b"

[models.gemma-4-e2b]
path = "/Volumes/Data/models/gemma-4-e2b/"
context_length = 32768

[models.qwen-3-coding]
path = "/Volumes/Data/models/qwen-3-coding-30b/"
context_length = 16384

[memory]
bundles_dir = "/var/lib/violet/bundles"
codec = "memvid"           # or "file"

[scheduler]
max_concurrent = 4
max_queue      = 32

[probe]
log_dir = "/var/log/violet/probes"
```

The daemon pre-loads `default_model` at startup. Other models load lazily on first reference.

## Lifecycle

```
violet starts
   ↓
read config + open socket
   ↓
pre-load default model
   ↓
warm prompt cache from on-disk seeds (if configured)
   ↓
serve requests until SIGINT/SIGTERM
   ↓
flush in-flight bundles to durable storage
   ↓
unload models cleanly
   ↓
close socket
```

## Used by

- **Cladius's local-inference skills** — `mattermost`, `wiki`, code summarise — call violet for batch text processing instead of round-tripping Anthropic
- **CoreAgent / core/ide** — chat-with-local-model surface
- **Vi training pipeline** — distillation teacher endpoint
- **LARQL vindex inspection** — pre/post-SFT model inference for diff

## Status

Production. Used in daily Cladius workflow (the wikis + mattermost + code-summarise skills route through it).

## Related

- `pkg/daemon/` — server implementation (planned dedicated doc)
- `../memory/agent_memory.md` — Wake/Sleep exposed over the socket
- `../inference/scheduler.md` — the scheduler that admits violet requests
- `../runtime/register_metal.md` — Violet boots the metal backend
- `project_local_inference_topology.md` — measured topology
- `project_go_mlx_research_grade.md` — the substrate this is part of

---
title: Deploying lthn-mlx
description: What lthn-mlx is as a deployed artefact, what files it needs alongside it, the serve command surface, health checks, graceful shutdown, and the canonical systemd / launchd patterns.
---

# Deploying lthn-mlx

`lthn-mlx` is the single process boundary in the Lethean local-inference stack. Snider's framing (2026-05-25): **"the actual model is the binary, the rest is package."** Everything that wants inference — `lthn` desktop, `pkg/lemma`, providers in `go-ai`, any OpenAI-compatible client — talks to this process over HTTP. There is no in-process library substitute for production deployments; the binary is the boundary.

This doc covers what you actually deploy, how to invoke it, what to expect at runtime, and how to wire it into the host service manager.

## What you ship

Until the metallib-bundling work lands (see [metallib-and-variants](metallib-and-variants.md)), a deployment is **two files plus the model directory**:

```
/opt/lthn-mlx/
├── bin/lthn-mlx              # the Go binary, ~25 MB
├── lib/mlx.metallib          # ~107 MB, see metallib-and-variants.md
└── models/                   # one or more model directories
    └── lemer-lite/
        ├── config.json
        ├── tokenizer.model
        ├── model.safetensors      # or *.gguf
        └── …
```

Once Path B bundling lands, the metallib disappears into the binary and you ship one file plus the model directory. Until then, the metallib is mandatory and its path is supplied via env var.

### What the binary is

`lthn-mlx` is `dappco.re/go/mlx/cmd/mlx` built and renamed. Default upstream output name is `core-mlx`; consumers (this includes the desktop app, this includes ops-side deployments) build with `-o lthn-mlx`. The binary embeds the full MLX runtime via cgo: 187 `mlx_*.cpp` files vendored at `go/internal/metal/` are compiled inline during `go build`, so the lthn-mlx executable has **zero non-system runtime dependencies** — `otool -L bin/lthn-mlx` shows only macOS frameworks (Foundation, Metal, Accelerate, QuartzCore, libSystem, libc++). The metallib is the only external file the binary needs at runtime today; Path B (Mantis #1779) folds it into the binary as well.

### Platform requirement

**darwin/arm64 only, macOS 26.0+.** Apple Silicon M1/M2/M3/M4/M5. The CGO files carry `//go:build darwin && arm64`. The 26.0 operating-system floor is intentional: the native path is built against the Metal 4 API generation shipped with macOS Tahoe 26, including the lower-overhead command API, explicit compilation API, tensors, and machine-learning passes documented by Apple. On any other platform the binary will not build, and pre-built `lthn-mlx` artefacts are not produced for Linux or Intel macOS. If you need inference on a non-Apple host, you want a different backend (e.g. `go-rocm` for AMD GPUs); the surface is the same go-inference interfaces.

References: [macOS Tahoe 26 release notes](https://developer.apple.com/documentation/macos-release-notes/macos-26-release-notes), [What's new in Metal](https://developer.apple.com/metal/whats-new/), [Understanding the Metal 4 core API](https://developer.apple.com/documentation/metal/understanding-the-metal-4-core-api), [Using the Metal 4 compilation API](https://developer.apple.com/documentation/metal/using-the-metal-4-compilation-api), and [Metal machine learning passes](https://developer.apple.com/documentation/metal/machine-learning-passes).

## The serve command

```
lthn-mlx serve --model <path> [--addr :11434] [--context N]
              [--read-timeout 30s] [--write-timeout 5m] [--shutdown-timeout 10s]
```

Reference: `go/cmd/mlx/serve.go`. The defaults are chosen to mirror Ollama's port (`11434`) so existing tooling pointed at `http://localhost:11434` works without reconfiguration.

| Flag | Default | What it does |
|------|---------|--------------|
| `--model` | *(required)* | Absolute path to a model directory containing `config.json`. HuggingFace safetensors layout or GGUF both supported. |
| `--addr` | `:11434` | TCP listen address. Use `127.0.0.1:11434` if you do not want LAN reach. |
| `--context` | `0` (model default) | Override the model's context length. Set explicitly if you know the workload doesn't need the full window — saves KV cache memory. |
| `--read-timeout` | `30s` | HTTP read-header timeout. Long enough for slow clients; not for inference. |
| `--write-timeout` | `5m` | HTTP write timeout, covering the full streaming response. The default accommodates long generations; raise if you serve very long outputs. |
| `--shutdown-timeout` | `10s` | Time the process gives in-flight requests to complete after SIGINT / SIGTERM before forcing exit. |

### Invocation, with the metallib workaround

```bash
export MLX_METALLIB_PATH=/opt/lthn-mlx/lib/mlx.metallib
lthn-mlx serve --model /opt/lthn-mlx/models/lemer-lite --addr 127.0.0.1:11434
```

The env-var set is **mandatory until bundling lands** — see [metallib-and-variants](metallib-and-variants.md) for why. Without it, `lthn-mlx` panics on first GPU dispatch as soon as a chat completion arrives.

### What "loaded" means

`lthn-mlx serve` does **not** load the model at process start. The model loads lazily on the first request that needs it, through the `openai.Resolver` constructed at `serve.go:68`. This is intentional: process startup stays sub-second, and admin endpoints (`/v1/health`, `/v1/runtime/sleep`, `/v1/runtime/wake`) respond immediately even when no model is mapped into VRAM yet.

The trade-off is **the first inference request after start takes the load cost** (typically 2-15 seconds depending on model size and storage speed). Pre-warming options:

1. **Hit `/v1/chat/completions` once at boot** with a one-token prompt before exposing the listener to traffic. Crude but effective.
2. **Wire to `/v1/runtime/wake`** if the admin handlers are configured with a Wake callback (the default serve invocation does not configure one — `serve.go:69-78` sets only `Health`). Pre-warm requires a custom integration on top of `openai.NewMuxWithAdmin`, not the bundled CLI.

If consistent first-request latency matters, do (1) in your service manager's `ExecStartPost`.

## The HTTP surface

The mux mounted by `openai.NewMuxWithAdmin` exposes three families of endpoints, all under the same listen address. Source of truth: `go/openai/openai.go:65-78` and `go/openai/admin.go:61-64`.

### OpenAI-compatible

| Path | Method | Purpose |
|------|--------|---------|
| `/v1/chat/completions` | POST | Standard chat completion. SSE streaming via `stream: true`. |
| `/v1/responses` | POST | OpenAI Responses API. |
| `/v1/embeddings` | POST | Embedding generation. |
| `/v1/rerank` | POST | Document reranking. |
| `/v1/models/capabilities` | GET | Reports what the loaded model supports (context length, modalities, etc). |
| `/v1/cancel` | POST | Cancel an in-flight stream. |

### Anthropic-compatible

| Path | Method | Purpose |
|------|--------|---------|
| `/v1/messages` | POST | Anthropic Messages API. |

### Ollama-compatible

| Path | Method | Purpose |
|------|--------|---------|
| `/api/chat` | POST | Ollama chat protocol. |
| `/api/generate` | POST | Ollama generate protocol. |
| `/api/tags` | GET | List available models (in this single-binary deploy, just the one loaded). |
| `/api/show` | POST | Model metadata. |

### Admin + cache

| Path | Method | Purpose |
|------|--------|---------|
| `/v1/health` | GET | Health probe. Returns the static struct populated at startup — confirms the process is up, not that the model is loaded. |
| `/v1/runtime/wake` | POST | If `AdminConfig.Wake` is wired, invokes the callback. Default serve: no-op. |
| `/v1/runtime/sleep` | POST | If `AdminConfig.Sleep` is wired, invokes the callback. Default serve: no-op. |
| `/v1/cache/entries` | GET | List cache block refs. |
| `/v1/cache/stats` | GET | KV cache statistics. |
| `/v1/cache/warm` | POST | Warm a cache entry. |
| `/v1/cache/clear` | POST | Clear cache state. |

### Health-check pattern

The bundled `/v1/health` is **liveness only** — it reports the runtime is up. It does NOT verify the model loads. A real readiness probe needs to issue a one-token chat completion:

```bash
curl -sf http://127.0.0.1:11434/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"lemer-lite","messages":[{"role":"user","content":"hi"}],"max_tokens":1}' \
  > /dev/null && echo READY
```

If you need a readiness probe in a service manager that distinguishes liveness from readiness (Kubernetes-style), point liveness at `/v1/health` and readiness at the above. For systemd or launchd, the one-shot test in `ExecStartPost` is usually enough.

## Graceful shutdown

The serve loop handles SIGINT and SIGTERM via the `signal.NotifyContext` set up in `main.go:32-34`. When a signal arrives:

1. `http.Server.Shutdown(ctx)` is called with `--shutdown-timeout` as the deadline.
2. Existing requests are given that long to drain.
3. After the deadline, the process exits with status 0 if drain succeeded, 1 if `Shutdown` returned an error.

There is **no model-unload step** in the shutdown path — the process exits and the OS reclaims the Metal allocations. If you have a long-running daemon scenario that needs explicit teardown (rare), wire the `Sleep` admin callback.

### Restart safety

The serve binary is stateless beyond the loaded model weights — there is no on-disk lock, no PID file, no recovery state. Restarting is safe; the new process starts cold and lazy-loads the model on the next request. **Two `lthn-mlx serve` processes on the same listen address will collide on `bind()` — the second will exit 1.** Use the service manager to enforce single-instance, don't rely on the binary.

## Service-manager patterns

### launchd (macOS, recommended)

Install the binary + metallib at `/opt/lthn-mlx/`, then create `~/Library/LaunchAgents/sh.lthn.mlx.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>sh.lthn.mlx</string>
    <key>ProgramArguments</key>
    <array>
        <string>/opt/lthn-mlx/bin/lthn-mlx</string>
        <string>serve</string>
        <string>--model</string><string>/opt/lthn-mlx/models/lemer-lite</string>
        <string>--addr</string><string>127.0.0.1:11434</string>
    </array>
    <key>EnvironmentVariables</key>
    <dict>
        <key>MLX_METALLIB_PATH</key>
        <string>/opt/lthn-mlx/lib/mlx.metallib</string>
    </dict>
    <key>RunAtLoad</key><true/>
    <key>KeepAlive</key>
    <dict>
        <key>SuccessfulExit</key><false/>
    </dict>
    <key>StandardOutPath</key><string>/opt/lthn-mlx/log/stdout.log</string>
    <key>StandardErrorPath</key><string>/opt/lthn-mlx/log/stderr.log</string>
</dict>
</plist>
```

Load: `launchctl load ~/Library/LaunchAgents/sh.lthn.mlx.plist`. Bounce: `launchctl kickstart -k gui/$UID/sh.lthn.mlx`. The `KeepAlive.SuccessfulExit=false` keeps the process up on crash but lets you stop it cleanly with `launchctl unload`.

### Foreground for development

```bash
MLX_METALLIB_PATH=$PWD/dist/lib/mlx.metallib \
  ./bin/lthn-mlx serve --model /Volumes/Data/models/lemer-lite --addr :11434
```

`Ctrl-C` triggers the graceful shutdown path.

## What to bind to

`127.0.0.1:11434` is the safe default — same-machine access only. Bind to `0.0.0.0:11434` if you want LAN reach, but note that **the serve binary has no authentication, no rate limiting, no TLS**. It is designed for trusted-network use: same machine, or a private LAN behind a firewall. Production LAN exposure should sit behind a reverse proxy (Caddy, nginx) that handles auth and TLS.

If you need authenticated remote access, that lives one layer up — the `pkg/lemma` client in `lthn/desktop` is the canonical Go-side consumer, and a tunnel / proxy / auth-gateway sits between lemma and a non-local `lthn-mlx`.

## Resource expectations

Measured on M3 Ultra (60-core GPU, 96 GB unified memory). Numbers will be lower on M1/M2 base chips with shared memory.

| Aspect | Observation |
|--------|-------------|
| Cold start (no model loaded) | <500 ms |
| First-request load (Gemma3-1B 4-bit) | ~2-3 s |
| First-request load (Llama 3.1 8B 4-bit) | ~5-7 s |
| Steady-state RAM (Gemma3-1B 4-bit, loaded) | ~1.5 GB |
| Steady-state RAM (DeepSeek R1 7B 4-bit) | ~5 GB |
| Process count | 1 |
| Threads | varies by request concurrency; typically 4-16 |

The model lives in unified memory — there is no separate "VRAM" line item on Apple Silicon. Activity Monitor's "Memory" column is the right place to watch; the Metal allocator reports its own numbers via `mlx.GetActiveMemory()` and the `/v1/cache/stats` endpoint.

For tuning the Metal cache and memory limits (the runtime-side knobs that affect serving behaviour), see [performance-tuning](performance-tuning.md).

## Sources

- `go/cmd/mlx/serve.go` — the serve command source
- `go/cmd/mlx/main.go` — signal handling + command dispatch
- `go/openai/openai.go:65-78` — mounted OpenAI/Anthropic/Ollama routes
- `go/openai/admin.go:16-65` — admin + health route definitions
- `go/internal/metal/backend.go:10-12` — default context length, parallel slots
- [macOS Tahoe 26 release notes](https://developer.apple.com/documentation/macos-release-notes/macos-26-release-notes)
- [What's new in macOS 26](https://developer.apple.com/macos/whats-new/)
- [What's new in Metal](https://developer.apple.com/metal/whats-new/)
- [Understanding the Metal 4 core API](https://developer.apple.com/documentation/metal/understanding-the-metal-4-core-api)
- [Using the Metal 4 compilation API](https://developer.apple.com/documentation/metal/using-the-metal-4-compilation-api)
- [Metal machine learning passes](https://developer.apple.com/documentation/metal/machine-learning-passes)

## Cross-references

- [Metallib & variants](metallib-and-variants.md) — what the env var workaround is buying you
- [Troubleshooting](troubleshooting.md) — panic signatures, model-load failures, port collisions
- [Performance tuning](performance-tuning.md) — Metal cache, memory limits, parallel slots

---
title: Troubleshooting lthn-mlx
description: The runtime failure modes you will actually hit, what they look like in the logs, and the specific fix for each. Grouped by where in the lifecycle they fire.
---

# Troubleshooting lthn-mlx

This doc catalogues the runtime failure modes for `lthn-mlx serve`. Each entry is shaped: **symptom → cause → fix**. Grouped by lifecycle phase: process start, model load, request handling, shutdown. The active blockers (the ones you will hit on a fresh deploy today) are flagged.

## Process-start failures

### Panic: "failed to load metallib" / segfault on first GPU touch

**ACTIVE BLOCKER until metallib-bundling lands.**

**Symptom.** Process starts cleanly, `/v1/health` returns 200. First chat completion request triggers an immediate panic or hard segfault. The MLX C++ side throws an exception that surfaces as a Go panic mentioning `mlx_metal_load_library` or `newLibraryWithURL`.

**Cause.** `MLX_METALLIB_PATH` is unset *and* the binary's CWD walk (`go/internal/metal/metal.go:204-224`) didn't find a `dist/lib/mlx.metallib` anywhere within five parent directories of CWD. The fallback returned the bare string `"mlx.metallib"`, which MLX resolved as a relative path against CWD and failed.

**Fix.** Set `MLX_METALLIB_PATH` to an absolute path before invoking:

```bash
export MLX_METALLIB_PATH=/opt/lthn-mlx/lib/mlx.metallib
lthn-mlx serve --model /opt/lthn-mlx/models/lemer-lite --addr :11434
```

This panic does not surface at process start — it waits until the first request hits the GPU. Liveness probes against `/v1/health` will pass; readiness probes that issue an actual completion will catch it. See [deployment.md](deployment.md) for the recommended readiness pattern.

**Permanent fix.** Path B bundling (embed via `//go:embed`, load via `MTLDevice newLibraryWithData:`). See [metallib-and-variants.md](metallib-and-variants.md). Once that lands, the env var becomes a dev override and is no longer required for production.

### "bind: address already in use" on start

**Symptom.** `lthn-mlx serve: listen failed: listen tcp :11434: bind: address already in use`. Process exits status 1.

**Cause.** Another process holds the listen port. Most commonly another `lthn-mlx serve` instance, or Ollama (default port also 11434), or a previous instance that didn't shut down cleanly.

**Fix.** Find and stop the holder:

```bash
lsof -i :11434
# kill the holder, or pick a different --addr
```

If you're running Ollama alongside `lthn-mlx` deliberately, give `lthn-mlx` a different port (e.g. `--addr :11435`).

### "--model is required" / exit code 2

**Symptom.** `lthn-mlx serve: --model is required` on the stderr, process exits 2.

**Cause.** The `--model` flag was missing or empty. The serve subcommand requires an explicit model path; there is no default.

**Fix.** Supply `--model /abs/path/to/model/dir`. The path must be a directory containing `config.json` (HuggingFace layout) or a `.gguf` file path.

### "dyld: Library not loaded: libmlx.dylib"

**Symptom.** Process fails to start with a dyld error pointing at `libmlx.dylib` or `libmlxc.dylib`.

**Cause.** The binary was built against the locally-built dylibs at `dist/lib/`, and was then copied somewhere else without those dylibs being available at the install-time linker search path. **This should not normally happen** — the build pipeline statically links these into the binary. If you see this, the binary was built with a non-default configuration that left them as dynamic dependencies.

**Fix.** Rebuild with the standard pipeline (`go generate ./... && go build -o lthn-mlx ./go/cmd/mlx`). If you must run a dynamic-link build, either:

1. `install_name_tool -change` the dylib paths to point at where they live on the target host, or
2. Set `DYLD_LIBRARY_PATH=/opt/lthn-mlx/lib` before invoking (fragile; not recommended).

## Model-load failures

### "no such file or directory: config.json"

**Symptom.** First request fails. Stderr shows a path-not-found error for `config.json` inside the `--model` directory.

**Cause.** The `--model` path either doesn't exist or doesn't contain a HuggingFace-style model directory. The loader expects either:

- A directory containing `config.json` + `tokenizer.model` (or `tokenizer.json`) + one or more `*.safetensors` files, or
- A single `*.gguf` file path.

**Fix.** Verify the path:

```bash
ls /path/to/model/
# Should show config.json + model.safetensors (or shards) + tokenizer files
```

If you have a GGUF, pass the file path directly:

```bash
lthn-mlx serve --model /path/to/model.gguf --addr :11434
```

### "unsupported model_type: X"

**Symptom.** First request fails. Stderr names a `model_type` from `config.json` that go-mlx doesn't recognise.

**Cause.** The model architecture isn't in the supported set. Currently supported (from `docs/index.md` and the `internal/metal/` decoder files):

| Family | `model_type` values |
|--------|---------------------|
| Gemma 3 | `gemma3`, `gemma3_text`, `gemma2` |
| Gemma 4 | `gemma4`, `gemma4_text` |
| Qwen 2/3 | `qwen3`, `qwen2` |
| Llama 3 | `llama` |

**Fix.** Either pick a model in the supported list, or open a Mantis ticket for the new architecture — adding a decoder is a defined extension point (`go/internal/metal/{gemma3,gemma4,qwen3,llama}.go` are the templates).

### Out-of-memory at model load

**Symptom.** First request fails, stderr shows a Metal allocator error or the process is killed by the OS OOM handler.

**Cause.** Model weights don't fit in unified memory. The whole-process budget on Apple Silicon includes the model weights, the KV cache (scales with `--context`), MLX's allocator cache, and everything else macOS is running. A 7B model in 4-bit needs ~5 GB resident; a 70B model needs ~40 GB.

**Fix.** Pick one or more:

1. **Use a smaller / more-quantised model.** 4-bit is the default for "fits comfortably"; 8-bit doubles the weight budget.
2. **Lower `--context`.** The KV cache scales linearly with context length. A 131k context (the default) on a 7B model can add several GB on top of the weights.
3. **Set Metal memory limits explicitly** at the binary call site if you have a custom integration:
   ```go
   mlx.SetMemoryLimit(32 << 30) // 32 GB hard cap
   mlx.SetCacheLimit(4 << 30)   // 4 GB allocator cache
   ```
   These knobs are not exposed as serve flags today. If you need them on the bundled CLI, that's a feature ticket against `cmd/mlx/serve.go`.
4. **Reboot.** macOS unified memory pressure persists across previous processes; a fresh boot gives the cleanest baseline.

See [performance-tuning.md](performance-tuning.md) for the memory-controls surface in detail.

## Request-handling failures

### Hang on the first request, no error

**Symptom.** First chat completion hangs for 10-30 seconds before producing a response.

**Cause.** Lazy model load — this is expected, not a failure. `lthn-mlx serve` does not load the model at process start; the first request triggers the load. See "What 'loaded' means" in [deployment.md](deployment.md).

**Fix.** Pre-warm at boot with a one-token completion before exposing the listener:

```bash
curl -sf http://127.0.0.1:11434/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"lemer-lite","messages":[{"role":"user","content":"hi"}],"max_tokens":1}' \
  > /dev/null
```

Wire this into the service manager's post-start hook.

### "context deadline exceeded" mid-stream

**Symptom.** A streaming completion cuts off partway through; client sees a connection close. Server log shows `http: write timeout`.

**Cause.** `--write-timeout` (default 5 min) elapsed before the stream finished. Either the prompt asked for an unusually long generation, or the model is slow on this hardware.

**Fix.** Raise the write timeout:

```bash
lthn-mlx serve --model … --addr … --write-timeout 15m
```

If you regularly hit this, the longer-term fix is to keep the connection alive at the protocol level (server-sent events with heartbeat) — a feature ticket against `openai.NewMuxWithAdmin`, not a config knob today.

### "model X not found" in the response

**Symptom.** Request succeeds with a 4xx response body referencing a model name mismatch.

**Cause.** The OpenAI/Anthropic/Ollama protocols all require a `model` field in the request. The serve binary loads exactly one model (the `--model` path). The model's reported name comes from `config.json` — typically the basename of the model directory, but architecture-dependent. Requesting any other name returns the mismatch.

**Fix.** Either:

1. Use the model name the server actually loaded — check via `GET /v1/models/capabilities` or `GET /api/tags`.
2. Send any string and rely on the resolver's single-model fallback (works in some protocol paths but not others — protocol-dependent, so verify per-client).

For a multi-model deployment, run multiple `lthn-mlx serve` instances on different ports, and put a router in front (the `pkg/lemma` client in lthn/desktop does this). Single binary, single model is the current shape.

### Streaming responses arrive whole, not chunked

**Symptom.** Client requested `stream: true` but the response arrives as one complete body.

**Cause.** Almost always a reverse-proxy buffering issue, not a server bug. nginx in particular buffers SSE by default.

**Fix.** Disable proxy buffering for the route. For nginx:

```nginx
location /v1/chat/completions {
    proxy_pass http://127.0.0.1:11434;
    proxy_buffering off;
    proxy_cache off;
    proxy_set_header X-Accel-Buffering no;
}
```

For Caddy, set `flush_interval -1` on the reverse_proxy directive.

### High latency / low tokens-per-second

**Symptom.** Inference works but is slower than the published baseline (e.g. 30 tok/s for Llama 3.1 8B 4-bit on M3 Ultra).

**Causes, in order of likelihood:**

1. **Model loaded on CPU not GPU.** Check log lines at startup; if you see `set cpu default device` without a corresponding successful Metal init, the load fell back to CPU. Usually because of a missing or wrong metallib (see "Process-start failures").
2. **Memory pressure forcing the allocator into churn.** Other processes are using unified memory; the MLX allocator is constantly evicting and re-allocating. Free up memory or set lower `SetCacheLimit` to make the eviction behaviour predictable.
3. **First-request latency mistaken for steady-state.** The first request after load includes prefill compilation cost; subsequent requests reuse compiled kernels. Measure on the second or third request.
4. **Thermal throttling.** Sustained inference loads can hit thermal limits on the chassis-constrained chips (MacBook Air; M2 Pro Mini in poor airflow). `pmset -g thermlog` reports thermal state.

See [performance-tuning.md](performance-tuning.md) for the levers that actually move steady-state throughput.

## Shutdown / restart failures

### Process doesn't exit on Ctrl-C

**Symptom.** First Ctrl-C is acknowledged in the log but the process hangs. Second Ctrl-C kills it.

**Cause.** The graceful shutdown path (`serve.go:107-114`) is waiting for in-flight requests to finish, bounded by `--shutdown-timeout` (default 10s). If a long generation is mid-stream when you Ctrl-C, the shutdown waits.

**Fix.** Either wait the 10 seconds, or send SIGKILL (`kill -9`) to force exit. For service-manager-driven restarts, bump `--shutdown-timeout` higher (30s-60s) if you have long-running generations and want them to complete cleanly.

### Restart leaves model state behind / next start is slow

**Symptom.** Restarting the process and the first post-restart request is slow again.

**Cause.** Lazy load — there is no model state to preserve across process boundaries (the model lives in MLX's allocator, which the OS reclaims on process exit). Every restart pays the cold-load cost on the next request.

**Fix.** Pre-warm post-restart (same pattern as cold start). If restart frequency is the actual problem, look at why you're restarting — `lthn-mlx serve` is designed to be a long-running daemon, not a request-per-process FastCGI-style worker.

### Two processes bound to the same model directory

**Symptom.** Two `lthn-mlx serve` processes running fine, each on a different port, both pointed at the same `--model`.

**Cause.** Not actually a failure — the model files are read-only at runtime. Both processes can map the same safetensors. There is no on-disk lock.

**Note.** Memory cost doubles — each process maps its own copy of the weights. If you want one set of weights serving two ports, you want one process serving requests at high concurrency, not two processes. The serve binary handles concurrent requests via Go's standard `net/http` goroutine-per-request; the only ceiling is `DefaultLocalParallelSlots` (currently 1 — see `backend.go:11`), which limits parallel GPU dispatches.

## Discovering what's actually wrong

When the failure doesn't match any of the above:

### Read the C++ side errors

MLX errors surface via `lastError()` in `metal.go:308-330`. Most are wrapped into the returned Go error and logged through `core.Error`. If a panic doesn't include a useful message, the C++ error handler may have caught and logged separately — check stderr for `mlx:` prefixed lines.

### Verify Metal availability

```go
// In your own test binary
import _ "dappco.re/go/mlx"
import "dappco.re/go/inference"

func main() {
    backend, _ := inference.GetBackend("metal")
    fmt.Println(backend.Available()) // false => Metal is the problem, not the model
}
```

If `Available()` returns false, the metallib + device init never completed cleanly. Check stderr for setup errors at process start.

### Get the device info

`mlx.GetDeviceInfo()` reports the Metal device the runtime selected. If you see a CPU device on a Mac you know has GPU, the GPU init failed silently — the runtime fell back to CPU and is decoding at single-digit tok/s. This is the most common "everything works but is dog-slow" cause.

## Where to file what you find

- **New failure mode not in this doc:** add an entry here in a PR, or file a Mantis ticket against `core` with the lifecycle phase + reproducer.
- **Panic deep in MLX C++:** file against `core` with the full stderr trace. May need an upstream MLX bug too — check `lib/mlx` issues.
- **Wrong recommendation in this doc:** PR the fix; this doc is supposed to be the operator's first stop, accuracy beats completeness.

## Cross-references

- [Deployment](deployment.md) — the happy-path setup these failure modes deviate from
- [Metallib & variants](metallib-and-variants.md) — the bundling work that resolves the process-start panic
- [Performance tuning](performance-tuning.md) — the levers for the slow-but-working class of problems

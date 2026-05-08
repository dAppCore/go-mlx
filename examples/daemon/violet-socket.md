# Violet Unix-Socket Sidecar

Violet is the local-only generation route for harnesses that don't need an OpenAI-compatible HTTP server. One JSON frame per line over a Unix socket, no HTTP stack, no networking — minimal overhead, no port conflicts, kernel-enforced trust boundary.

## Configure

```toml
# violet.toml
[models]
default = "/models/qwen3-8b"
classify = "/models/gemma-3-1b"

[runtime]
context_length = 131072
parallel_slots = 1
```

Multiple model paths can be loaded; clients select by name in each request.

## Run The Daemon

```bash
violet --config violet.toml --socket /tmp/violet.sock
```

Models are loaded lazily on first use and kept resident until the daemon exits. The `runtime` block sets the same defaults as `mlx.LoadModel` (GPU device, 131k bounded context, one active native slot, exact-token-prefix prompt cache enabled).

## Talking To It

JSON Lines protocol — one request per line, one response per line.

### Generate (raw prompt)

Request:
```json
{"action":"generate","model":"default","prompt":"What is 2+2?","max_tokens":64}
```

Response:
```json
{"text":"4","tokens":[19],"finish_reason":"eos","prompt_tokens":7,"completion_tokens":1}
```

### Generate (chat)

Request:
```json
{"action":"generate","model":"default","messages":[{"role":"system","content":"Be direct."},{"role":"user","content":"What is 2+2?"}],"max_tokens":64}
```

Response:
```json
{"text":"4","tokens":[19],"finish_reason":"eos"}
```

The chat template is auto-detected from the model's `config.json` (`gemma3`, `gemma4`, `qwen3`, `llama`).

### Classify

Request:
```json
{"action":"classify","model":"classify","prompts":["refund my order","what's your SLA"],"labels":["hostile","business","casual","technical"]}
```

Response:
```json
{"results":[{"top_label":"hostile","top_score":0.92},{"top_label":"business","top_score":0.78}]}
```

## Talking From Go

```go
import (
    "encoding/json"
    "net"
)

conn, err := net.Dial("unix", "/tmp/violet.sock")
if err != nil { log.Fatal(err) }
defer conn.Close()

req := map[string]any{
    "action":     "generate",
    "model":      "default",
    "prompt":     "What is 2+2?",
    "max_tokens": 64,
}
if err := json.NewEncoder(conn).Encode(req); err != nil { log.Fatal(err) }

var resp map[string]any
if err := json.NewDecoder(conn).Decode(&resp); err != nil { log.Fatal(err) }
fmt.Println(resp["text"])
```

## Talking From The Shell

```bash
echo '{"action":"generate","prompt":"What is 2+2?","max_tokens":64}' | nc -U /tmp/violet.sock
```

## Why Use This Instead Of HTTP

- **No port conflicts** — Unix sockets are filesystem objects, not network ports
- **Filesystem-level access control** — chmod the socket, you control who can talk to it
- **Lower overhead** — no HTTP framing, TLS handshake, or keep-alive bookkeeping
- **Async-friendly** — JSON Lines is trivially streamable; you can pipeline requests on one connection if the daemon supports it (currently single-flight per connection)

## When To Use The OpenAI HTTP Server Instead

If you need:
- Cross-machine access (Violet is unix-socket only)
- OpenAI-compatible tooling (function calling, structured output, third-party clients)
- Multiple concurrent clients

Then bring up the standard go-inference HTTP server in front of the metal backend. Violet is for the local research / agent-harness case where you own both ends of the wire.

## See Also

- [docs/index.md — Violet Native Route](../../docs/index.md#violet-native-route) — config reference
- [Streaming inference](../inference/streaming.md) — same generation path, in-process

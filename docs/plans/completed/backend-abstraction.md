# Backend Abstraction — Completion Summary

**Completed:** 19 February 2026
**Module:** `forge.lthn.ai/core/go-mlx`
**Status:** Complete — shared go-inference interfaces, Metal auto-registration

## What Was Built

Migrated go-mlx to implement shared `go-inference` interfaces so it
plugs into the unified ML backend system alongside HTTP and Llama backends.

### Key changes

- `InferenceAdapter` implements `inference.Backend` interface
- Metal backend auto-registers via `init()` when CGo is available
- `Result` struct carries text + `Metrics` (tokens, latency, tokens/sec)
- Model loading, tokenization, and generation all behind interface methods

### Architecture

```
go-ml (orchestrator)
  → go-inference (interfaces)
    → go-mlx (Metal/MLX backend, auto-registered)
    → llama (llama.cpp backend)
    → http (Ollama/OpenAI backend)
```

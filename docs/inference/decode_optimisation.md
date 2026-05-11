<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# decode_optimisation.go — speculative + prompt-lookup decoding

**Package**: `dappco.re/go/mlx`
**File**: `go/decode_optimisation.go`
**Status**: experimental — harness present, kernels pending

## What this is

The **hooks for speculative decoding** and **prompt-lookup decoding** — two optimisation techniques that accelerate autoregressive generation by parallelising the work that's normally serial.

This file owns the test/measurement harness; the actual native acceleration lives in `internal/metal/` once the kernels land.

## Speculative decoding

A small **draft model** generates K candidate tokens; the main model verifies all K in parallel (one forward pass at length K instead of K passes at length 1). When the draft and main agree, K tokens land per forward — net speedup ~2-3x for chat-style workloads where the small model usually matches.

Gemma 4 ships an `-assistant` drafter checkpoint specifically for this (see `project_gemma4_mtp_assistant_shipped.md`) — measured up to 3x decode speedup with zero quality loss.

## Prompt-lookup decoding

Inspect the prompt for repeated N-grams. When a token sequence already appearing in the prompt becomes a candidate continuation, parallel-verify the next K tokens against the prompt match. Common in retrieval-augmented workflows where the answer cribs from the context — saves the autoregressive walk through the rebuild-already-said-text part.

## DecodeGenerateFunc

```go
type DecodeGenerateFunc func(
    context.Context,
    string,                  // prompt
    GenerateConfig,
) (DecodeGeneration, error)
```

The small hook the harness uses to measure decode optimisation. Returns tokens (so accepted-vs-rejected can be counted) without binding to a concrete kernel.

## DecodeGeneration

```go
type DecodeGeneration struct {
    Tokens    []Token
    Accepted  int     // out of K candidates
    Rejected  int
    LatencyMs float64
}
```

Used to compute acceptance rate over a batch — the headline metric for both techniques.

## Status

| Technique | Harness | Kernel | Eval |
|-----------|---------|--------|------|
| Speculative | done | in flight (Phase 1) | suite ready |
| Prompt-lookup | done | planned | suite ready |

The Gemma 4 `-assistant` drafter integration is the immediate target — gives 2-3x decode on Gemma 4 dense models without re-training.

## Related

- [scheduler.md](scheduler.md) — scheduler decides per-request whether to use draft path
- [block_cache.md](block_cache.md) — cache misses on draft+main share the same block hashes
- `project_gemma4_mtp_assistant_shipped.md` — Gemma 4 drafter context
- `../../../go-inference/docs/inference/capability.md` — `CapabilitySpeculativeDecode` + `CapabilityPromptLookupDecode`
- `docs/vmlx-feature-gap-report.md` — vMLX claims; gap closing

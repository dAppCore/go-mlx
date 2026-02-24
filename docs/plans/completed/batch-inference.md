# Batch Inference — Completion Summary

**Completed:** 19 February 2026
**Module:** `forge.lthn.ai/core/go-mlx`
**Status:** Complete — 5K sentences/sec classification, integrated with go-i18n

## What Was Built

Added batch inference capabilities to the MLX backend for high-throughput
classification and generation.

### Components

- **`Classify()`** — prefill-only mode for single-token classification
  (domain labelling). No autoregressive generation needed.
- **`BatchGenerate()`** — autoregressive batch generation with attention
  masking for padded sequences in variable-length batches.
- **Attention masking** — correct handling of padded batches so shorter
  sequences don't attend to padding tokens.

### Performance

- 5,000 sentences/sec for classification on M3 Ultra (prefill-only)
- Native Metal execution via Go→CGo→mlx-c pipeline

### Integration

Used by go-i18n 1B Pre-Sort Pipeline (Phase 2a) to batch-classify 88K
seeds through Gemma3-1B at 80 prompts/sec (constrained by prompt
construction, not inference).

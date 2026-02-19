# Batch Inference API Design

**Goal:** Enable batched inference (multiple prompts in one forward pass) for both classification (prefill-only) and generation (autoregressive).

**Primary consumer:** go-i18n Phase 2a — ~5K sentences/sec domain classification through Gemma3-1B.

## API Surface (go-inference)

Two new methods on `TextModel`, two new result types:

```go
type ClassifyResult struct {
    Token  Token      // Sampled token at last prompt position
    Logits []float32  // Raw vocab-sized logits (when WithLogits() set)
}

type BatchResult struct {
    Tokens []Token // All generated tokens for this prompt
    Err    error   // Per-prompt error (context cancel, OOM, etc.)
}

// New methods on TextModel:
Classify(ctx context.Context, prompts []string, opts ...GenerateOption) ([]ClassifyResult, error)
BatchGenerate(ctx context.Context, prompts []string, opts ...GenerateOption) ([]BatchResult, error)
```

New option: `WithLogits()` — include raw logits in ClassifyResult (off by default to save memory).

## Architecture

### Classify (prefill-only, fast path)

1. Encode all prompts via tokenizer
2. Pad to max sequence length with pad token (0)
3. Build attention mask: `[N, 1, maxLen, maxLen]` combining causal + padding
4. Single forward pass: `[N, maxLen]` → `[N, maxLen, vocab]`
5. Gather logits at each prompt's last real token position
6. Sample (greedy/temperature) per prompt
7. Return `[]ClassifyResult`

No KV caches needed — single pass, no autoregressive decode.

### BatchGenerate (autoregressive)

1. **Prefill**: same as Classify steps 1-5
2. **Decode loop** (up to MaxTokens iterations):
   - Sample next token for all active sequences
   - Check stop conditions per sequence (EOS, stop tokens)
   - Mark finished sequences; continue until all done
   - Append `[N, 1]` next tokens, forward through model with KV caches
3. Collect per-sequence token lists into `[]BatchResult`

Simple batch management: keep all sequences in the batch even after some finish (pad with zeros). This wastes some compute but avoids complex batch compaction.

### Attention Mask

For padded batches, SDPA needs a combined causal + padding mask:

```
mask[b, 0, i, j] = 0.0    if j <= i AND j < promptLen[b]   (attend)
mask[b, 0, i, j] = -inf   otherwise                        (don't attend)
```

Shape: `[N, 1, maxLen, maxLen]` — broadcasts across attention heads.

Uses existing `ScaledDotProductAttentionWithMask`.

### Model Forward Changes

Add `ForwardMasked(tokens, mask *Array, caches []Cache) *Array` to `InternalModel`. Threads the mask through to each attention layer. Existing `Forward` unchanged (causal-only, no mask).

Changes to attention layers in gemma3.go and qwen3.go:
- Accept optional mask parameter
- Use `ScaledDotProductAttentionWithMask` when mask is provided
- Fall back to `ScaledDotProductAttention` with causal bool when nil

### Padding and Tokenisation

- Pad token: 0 (standard for most models, verified for Gemma3/Qwen/Llama)
- Left-pad or right-pad? **Right-pad** — simpler, causal mask handles it.
  Last real token position = `promptLen[b] - 1`.
- Sequences sorted by length descending before batching (reduces padding waste)

## Implementation Location

- `go-inference`: types (`ClassifyResult`, `BatchResult`), option (`WithLogits`), interface methods
- `internal/metal/batch.go`: `BatchClassify`, `BatchGenerate` implementation
- `internal/metal/generate.go`: wire through `Model.Classify`, `Model.BatchGenerate`
- `internal/metal/gemma3.go`, `qwen3.go`: add `ForwardMasked` method

## Testing

1. Unit: mask construction (causal + padding) with known shapes
2. Classify: batch of 4 prompts through Gemma3-1B, verify each gets a sensible token
3. BatchGenerate: batch of 2-3 prompts, verify coherent output per prompt
4. Throughput: benchmark Classify at batch sizes 1, 8, 32, measure sentences/sec

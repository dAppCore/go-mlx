---
title: Models
description: Model loading, supported architectures, tokenisation, and chat templates.
---

# Models

go-mlx loads transformer models from either HuggingFace safetensors shards or GGUF checkpoints. For safetensors directories, architecture is detected from the `model_type` field in `config.json`; for direct `.gguf` loads via `inference.LoadModel`, it is detected from checkpoint metadata.

## Loading a Model

```go
import (
    "dappco.re/go/inference"
    _ "dappco.re/go/mlx"
)

m, err := inference.LoadModel("/path/to/model/")
if err != nil {
    log.Fatal(err)
}
defer m.Close()
```

The model path may be either a model directory or an explicit `.gguf` file path.

When loading a directory, it must contain:

- `config.json` -- model configuration (architecture, dimensions, quantisation)
- `tokenizer.json` -- HuggingFace BPE tokeniser
- Weights in exactly one of these formats:
  - One or more `*.safetensors` files (multi-shard supported)
  - Exactly one `*.gguf` file

### Load Options

```go
m, err := inference.LoadModel("/path/to/model/",
    inference.WithContextLen(4096),           // bounded KV cache (default: unbounded)
    inference.WithAdapterPath("/path/to/lora/"), // load LoRA adapter at init
)
```

| Option | Effect |
|--------|--------|
| `WithContextLen(n)` | Replaces unbounded KV caches with `RotatingKVCache(n)` |
| `WithAdapterPath(dir)` | Loads a trained LoRA adapter from the given directory |
| `WithGPULayers(n)` | Ignored with a warning -- Metal always uses full GPU offload |

## Supported Architectures

### Gemma 3

**Config values:** `gemma3`, `gemma3_text`, `gemma2`

Decoder structure per layer (pre-norm with four norm points):

```
input -> InputNorm -> Attention -> PostAttnNorm -> residual add
      -> PreFFNorm -> MLP       -> PostFFNorm  -> residual add
```

Key features:

- **Q/K RMS normalisation** -- separate `QNorm` and `KNorm` modules per attention layer
- **Alternating attention** -- sliding window and global attention alternate based on `sliding_window_pattern` (default 6). Sliding layers use theta 10000; global layers use theta 1000000.
- **Grouped-query attention (GQA)** -- K/V heads repeated via `RepeatKV` when `num_kv_heads < num_attention_heads`
- **Gemma-style normalisation** -- weights are `(1 + weight)` scaled, precomputed at load time to avoid repeated addition during inference
- **Embedding scale** -- hidden states are multiplied by `sqrt(hidden_size)` after embedding lookup
- **MLP** -- GELU-based gate with tanh approximation, compiled via `CompileShapeless` as a singleton
- **Output head** -- typically tied to `embed_tokens`; uses a separate `lm_head.weight` if present in the safetensors

### Gemma 4

**Config values:** `gemma4`, `gemma4_text`

Gemma 4 uses a dedicated loader (`LoadGemma4`) with several architecture-specific behaviours:

- **Mixed attention head sizes** -- sliding layers use `head_dim`, full-attention layers use `global_head_dim`
- **Per-layer RoPE** -- sliding attention defaults to theta 10000 and full attention to theta 1000000 with partial rotary
- **Shared KV cache** -- the tail of the network can reuse KV state from earlier same-type layers to reduce memory use
- **K-equals-V support** -- full-attention layers can reuse the K projection for V
- **Value normalisation** -- values pass through `RMSNormNoScale` before caching
- **MoE routing** -- router projections stay quantised at 8-bit and sparse experts dispatch through `gather_mm` / `gather_qmm`
- **Weight sanitisation** -- multimodal tower weights are stripped and `experts.gate_up_proj` tensors are split into separate gate/up weights

Gemma 4 chat formatting follows the same turn template as Gemma 3.

### Qwen 3 / Qwen 2 / Llama 3

**Config values:** `qwen3`, `qwen2`, `llama`

These three architectures share one loader (`LoadQwen3`) and one decoder implementation. Decoder structure per layer (standard pre-norm):

```
input -> InputNorm    -> Attention -> residual add
      -> PostAttnNorm -> MLP       -> residual add
```

| Feature | Qwen 3 | Qwen 2 | Llama 3 |
|---------|--------|--------|---------|
| Q/K norm | Yes | No | No |
| MLP type | SwiGLU | SwiGLU | SwiGLU |
| Output head | Separate `lm_head` | Separate `lm_head` | Separate `lm_head` |

MLP: SwiGLU gate -- `down(silu(gate(x)) * up(x))`.

Qwen 2 vs Qwen 3 detection: if `model_type` is absent, the presence of `model.layers.0.self_attn.q_norm.weight` in the weights distinguishes Qwen 3 (present) from Qwen 2 (absent).

## Weight Loading

The loader performs these steps:

1. Reads `config.json` for model configuration
2. Loads `tokenizer.json` for the tokeniser
3. Loads weights from either all `*.safetensors` shards or a single `.gguf` file
4. Resolves weights by name, with automatic `language_model.` prefix fallback
5. Constructs `Linear` layers as quantised or dense based on presence of `scales` tensors
6. Calls `Materialize()` on all weight arrays to commit them to GPU memory

### Quantisation

Quantisation is transparent. Quantised models store packed weights alongside `scales` and `biases` tensors. The `Linear.Forward()` method dispatches to `QuantizedMatmul` (MLX grouped quantisation kernel) when these tensors are present. Quantisation parameters (`bits`, `group_size`) are read from top-level `config.json`.

### Head Dimension Inference

If `head_dim` is absent from `config.json` (common in some Gemma 3 variants), it is computed from the Q projection weight shape: `q_proj.weight[0] / num_attention_heads`.

## Tokeniser

`Tokenizer` reads a `tokenizer.json` file and supports two BPE variants, auto-detected at load time.

### SentencePiece BPE (Gemma 3 / Gemma 4)

- Prefixes each segment with `\u2581` (Unicode U+2581, the SentencePiece space marker)
- Splits into characters
- Applies BPE merges via a rank-sorted lookup table
- Looks up merged symbols in the vocabulary

Detection: absence of `Gthe` (GPT-2 space+the encoding) in the vocabulary. The check uses `Gthe` rather than bare `G` because large SentencePiece vocabularies may contain the character incidentally.

### GPT-2 Byte-Level BPE (Qwen, Llama, DeepSeek)

- Maps all 256 bytes to printable Unicode characters (GPT-2 convention)
- Printable ASCII (33-126) and Latin-1 Supplement (161-172, 174-255) map to themselves
- Control characters, space, DEL, and gaps map to U+0100 onwards
- Applies BPE merges in this Unicode representation
- Decodes back to raw bytes via the inverse map

Detection: presence of `Gthe` in the vocabulary.

### BPE Merge Algorithm

Standard greedy algorithm:

1. Build merge rank table from the merges field (O(1) lookup by `"a b"` key)
2. Scan all adjacent pairs; find the pair with the lowest rank
3. Merge that pair into a single symbol
4. Repeat until no more merges apply

Merges are parsed from both `["a b", ...]` and `[["a","b"], ...]` JSON formats.

### Special Tokens

Special tokens are matched before BPE encoding. Each architecture uses different stop tokens:

| Family | BOS | EOS / Stop |
|--------|-----|-----------|
| Gemma 3 / 4 | `<bos>` | `<end_of_turn>` |
| Qwen 2/3 | `<\|im_start\|>` | `<\|im_end\|>` |
| Llama 3 | `<\|begin_of_text\|>` | `<\|eot_id\|>` |

## Generation

### Streaming

`Generate` returns `iter.Seq[Token]` (Go 1.23+ range-over-func):

```go
ctx := context.Background()
for tok := range m.Generate(ctx, "What is 2+2?", inference.WithMaxTokens(64)) {
    fmt.Print(tok.Text)
}
if err := m.Err(); err != nil {
    log.Fatal(err)
}
```

### Chat

`Chat` applies the model's native template before generating:

```go
for tok := range m.Chat(ctx, []inference.Message{
    {Role: "system", Content: "You are a helpful assistant."},
    {Role: "user", Content: "Translate 'hello' to French."},
}, inference.WithMaxTokens(64)) {
    fmt.Print(tok.Text)
}
```

Chat templates by architecture:

| Architecture | Format |
|-------------|--------|
| Gemma 3 / 4 | `<start_of_turn>role\ncontent<end_of_turn>\n` |
| Qwen 2/3 | `<\|im_start\|>role\ncontent<\|im_end\|>\n` |
| Llama 3 | `<\|start_header_id\|>role<\|end_header_id\|>\n\ncontent<\|eot_id\|>` |

### Generation Options

```go
inference.WithMaxTokens(128)      // maximum tokens to generate
inference.WithTemperature(0.7)    // sampling temperature (0 = greedy)
inference.WithTopK(40)            // top-K sampling
inference.WithTopP(0.9)           // nucleus sampling
inference.WithRepeatPenalty(1.1)  // repetition penalty
inference.WithStopTokens(1, 2)   // additional stop token IDs
```

The direct root API adds `mlx.WithMinP(0.05)` for minimum-probability sampling.

When combined, sampling options are applied in this order: temperature, then top-p, then top-k, then min-p.

### Context Cancellation

Pass a cancellable context to stop generation early:

```go
ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
defer cancel()

for tok := range m.Generate(ctx, prompt, inference.WithMaxTokens(1000)) {
    fmt.Print(tok.Text)
}
// m.Err() returns context.Canceled or context.DeadlineExceeded
```

## Batch Inference

### Classify (Prefill-Only)

Single forward pass per batch -- no decode loop. Prompts are right-padded to the longest length:

```go
results, err := m.Classify(ctx, []string{
    "Bonjour, comment allez-vous?",
    "The quarterly report shows growth.",
}, inference.WithTemperature(0))

for i, r := range results {
    fmt.Printf("prompt %d -> token %d %q\n", i, r.Token.ID, r.Token.Text)
}
```

Use `inference.WithLogits()` to return the full logit vector alongside the sampled token.

### BatchGenerate (Autoregressive)

Full autoregressive generation for multiple prompts in parallel:

```go
results, err := m.BatchGenerate(ctx, []string{
    "The capital of France is",
    "Water boils at",
}, inference.WithMaxTokens(32))

for i, r := range results {
    for _, tok := range r.Tokens {
        fmt.Print(tok.Text)
    }
    fmt.Println()
}
```

## Model Discovery

Scan a directory tree for available models:

```go
models, err := inference.Discover("/path/to/models/")
for _, d := range models {
    fmt.Printf("%s (%s, %d-bit, %d files)\n",
        d.Path, d.ModelType, d.QuantBits, d.NumFiles)
}
```

## Model Info and Metrics

```go
info := m.Info()
fmt.Printf("%s: %d layers, %d vocab, %d hidden, %d-bit quant\n",
    info.Architecture, info.NumLayers, info.VocabSize,
    info.HiddenSize, info.QuantBits)

// After generation:
met := m.Metrics()
fmt.Printf("prefill: %.0f tok/s, decode: %.1f tok/s, peak GPU: %d MB\n",
    met.PrefillTokensPerSec, met.DecodeTokensPerSec,
    met.PeakMemoryBytes/1024/1024)
```

## Attention Inspection

The Metal adapter implements `inference.AttentionInspector`, enabling extraction of post-RoPE K vectors from the KV cache:

```go
inspector, ok := m.(inference.AttentionInspector)
snap, err := inspector.InspectAttention(ctx, "What is kindness?")

// snap.Keys[layer][head] -> flat float32 of len seq_len * head_dim
fmt.Printf("layers=%d heads=%d seq=%d dim=%d\n",
    snap.NumLayers, snap.NumHeads, snap.SeqLen, snap.HeadDim)
```

The K tensors are post-RoPE -- rotary position embeddings have already been applied. For GQA models, `NumHeads` reflects the KV head count, not the query head count.

## Adding a New Architecture

1. Create `internal/metal/{name}.go` with `//go:build darwin && arm64`
2. Implement the `InternalModel` interface (Forward, ForwardMasked, NewCache, NumLayers, Tokenizer, ModelType, ApplyLoRA)
3. Add a case in `model.go`:`loadModel` for the new `model_type` value
4. Add a `close{Name}` helper in `close.go` for deterministic weight cleanup
5. Add `format{Name}Chat` in `generate.go` for the chat template
6. Add BOS/EOS detection in `tokenizer.go`:`LoadTokenizer`
7. Write tests: config parsing, missing weights, end-to-end inference

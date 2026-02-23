# Architecture

Module: `forge.lthn.ai/core/go-mlx`

Native Apple Metal GPU inference via mlx-c bindings, implementing the `inference.Backend` interface from `forge.lthn.ai/core/go-inference` for Apple Silicon (M1-M4).

---

## Package Layout

```
go-mlx/
├── mlx.go                 — Package doc + go:generate CMake directives
├── mlx_stub.go            — !darwin || !arm64: MetalAvailable() = false
├── register_metal.go      — darwin && arm64: registers "metal" backend via init()
├── mlx_test.go            — Integration tests (public API via go-inference)
│
├── internal/metal/        — All CGO code (darwin && arm64 only)
│   ├── metal.go           — Init, error handler, Eval/EvalAsync/Materialize
│   ├── array.go           — Array type, creation, data access, Iter()
│   ├── dtype.go           — DType constants
│   ├── stream.go          — Metal stream/queue, memory controls
│   ├── ops.go             — Element-wise, reduction, matrix, shape ops
│   ├── fast.go            — Fused Metal kernels: RMSNorm, LayerNorm, RoPE, SDPA
│   ├── nn.go              — Linear, Embedding, RMSNormModule, RepeatKV
│   ├── compile.go         — CompiledFunc (shapeless function compilation)
│   ├── slice.go           — Array slicing, update-in-place
│   ├── random.go          — RandomCategorical, RandomUniform, RandomNormal
│   ├── io.go              — Safetensors load/save
│   ├── model.go           — InternalModel interface + architecture dispatch
│   ├── gemma3.go          — Gemma 3 decoder
│   ├── qwen3.go           — Qwen 2/3 and Llama 3 decoder
│   ├── cache.go           — KVCache + RotatingKVCache
│   ├── sample.go          — Sampling chain: greedy, temperature, TopK, TopP, MinP
│   ├── tokenizer.go       — BPE tokenizer (SentencePiece + GPT-2 byte-level)
│   ├── grad.go            — VJP, JVP, ValueAndGrad, Checkpoint, loss functions
│   ├── lora.go            — LoRA adapters, random normal, safetensors save
│   ├── optim.go           — AdamW optimiser
│   ├── generate.go        — Model, Generate, Chat, batch inference, metrics
│   ├── close.go           — Deterministic weight cleanup
│   └── backend.go         — LoadAndInit entry point
│
└── mlxlm/                 — Python subprocess backend
    ├── backend.go          — mlxlmBackend implementing inference.Backend
    └── bridge.py           — Python script (embedded via //go:embed)
```

---

## CGO / mlx-c Binding

### Build Chain

The native layer depends on mlx-c v0.4.1, a C API wrapping Apple's MLX C++ framework. `go generate ./...` fetches and builds it via CMake:

```
go:generate cmake -S . -B build -DCMAKE_INSTALL_PREFIX=dist ...
go:generate cmake --build build --parallel
go:generate cmake --install build
```

CMake installs headers to `dist/include/` and shared libraries to `dist/lib/`. The `#cgo` directives in `internal/metal/metal.go` reference those paths:

```
CPPFLAGS: -I${SRCDIR}/../../dist/include
LDFLAGS:  -L${SRCDIR}/../../dist/lib -lmlxc -lmlx
darwin:   -framework Foundation -framework Metal -framework Accelerate
          -Wl,-rpath,${SRCDIR}/../../dist/lib
```

Every Go source file in `internal/metal/` carries `//go:build darwin && arm64`. The root package compiles on all platforms; only the blank import of `_ "forge.lthn.ai/core/go-mlx"` triggers the Metal backend on supported hardware.

### Error Handling

mlx-c reports errors through a registered C callback. The handler stores the error string in a C atomic variable using `atomic_store_explicit` with release ordering. `lastError()` reads and atomically clears it with acquire ordering, returning a Go error. `Eval()` checks the mlx return code and calls `lastError()` to surface real MLX messages. `Materialize()` wraps `Eval()` and logs on error without returning; callers that need propagation call `Eval()` directly.

### Evaluation Model

MLX uses lazy evaluation: operations build a computation graph without executing. Execution is triggered by `mlx_eval` or `mlx_async_eval`, which dispatch the graph to the Metal GPU. Go wrappers:

- `Eval(...*Array) error` — synchronous, returns error
- `EvalAsync(...*Array) error` — queues for async execution
- `Materialize(...*Array)` — synchronous, logs error (used in test helpers and weight loading)

---

## Array Type

`Array` wraps an `mlx_array` (a C-side opaque handle). Arrays are reference-counted on the C side; Go uses `runtime.SetFinalizer` to call `mlx_array_free` when the Go object is collected. Go 1.26's Green Tea GC reduces finaliser latency under sustained inference.

Key operations:

- Creation: `newArray()`, `FromValue()`, `FromValues()`, `Zeros()`, `Ones()`
- Data access: `Floats()`, `DataInt32()`, `Int()` — all call `ensureContiguous()` first to handle view arrays (transpose, broadcast, slice views) that have non-contiguous physical layouts. Previously, reading views returned silently incorrect data.
- Shape: `Shape()`, `Dim()`, `Size()`
- Iteration: `Array.Iter() iter.Seq[float32]` — range-over-func (stable since Go 1.23), handles non-contiguous arrays

---

## Memory Management

The Metal allocator (separate from system RAM) is controlled via functions exposed at the root package level:

| Function | Purpose |
|----------|---------|
| `SetCacheLimit(bytes)` | Soft limit on allocator cache |
| `SetMemoryLimit(bytes)` | Hard limit |
| `SetWiredLimit(bytes)` | Wired memory limit |
| `GetActiveMemory()` | Current live allocations |
| `GetPeakMemory()` | High-water mark since last reset |
| `GetCacheMemory()` | Cached (not yet freed) memory |
| `ClearCache()` | Release cached memory to OS |
| `ResetPeakMemory()` | Reset high-water mark |

`Model.Close()` walks the full model tree and explicitly frees all weight arrays via `Free()`, without relying on GC finalisers. Tied output weights (shared with the embedding table) are detected and skipped to prevent double-free. `Close()` is idempotent.

During generation, each call allocates fresh KV caches that are released to GC at iterator completion. Call `ClearCache()` between multi-turn chat turns for prompt reclaim rather than waiting for GC.

---

## Model Architectures

All architectures implement the `InternalModel` interface:

```go
type InternalModel interface {
    Forward(tokens *Array, caches []Cache) *Array
    ForwardMasked(tokens *Array, mask *Array, caches []Cache) *Array
    NewCache() []Cache
    NumLayers() int
    Tokenizer() *Tokenizer
    ModelType() string
    ApplyLoRA(cfg LoRAConfig) *LoRAAdapter
}
```

Architecture is detected from `config.json` → `model_type` field:

| `model_type` values | Loader | Notes |
|---------------------|--------|-------|
| `gemma3`, `gemma3_text`, `gemma2` | `LoadGemma3` | Gemma 3 decoder |
| `qwen3`, `qwen2`, `llama` | `LoadQwen3` | Shared decoder, variant-specific features |

### Gemma 3

Decoder structure per layer (pre-norm with four norm points per block):

```
input → InputNorm → Attention → PostAttnNorm → residual add
      → PreFFNorm  → MLP       → PostFFNorm  → residual add
```

Attention specifics:
- Q/K RMS normalisation (separate `QNorm`, `KNorm` modules)
- Alternating sliding window / global attention: sliding layers use `RopeLocalBaseFreq` (10000), global layers use `RopeTheta` (1000000). Pattern period determined by `sliding_window_pattern` (default 6)
- Rotary embeddings via fused RoPE Metal kernel with per-layer theta
- Grouped-query attention (GQA): K/V heads repeated via `RepeatKV` when `num_kv_heads < num_attention_heads`

MLP: GELU-based gate using tanh approximation. The GELU function is compiled via `CompileShapeless` (shapeless function compilation) as a singleton to avoid recompilation across calls.

Normalisation: Gemma uses `(1 + weight) * RMSNorm(x)` — the `(1 + weight)` factor is precomputed at load time (`precomputeScaledWeights`) for all seven norm points per layer to avoid repeated additions during inference.

Embedding scale: hidden states are multiplied by `sqrt(hidden_size)` after embedding lookup (Gemma-specific convention). Qwen and Llama do not apply this scale.

Output head: Gemma 3 typically ties `lm_head` weights to `embed_tokens`. If a separate `lm_head.weight` is present in the safetensors, it is used as an independent output projection.

### Qwen 3 / Qwen 2 / Llama 3

These three architectures share one loader (`LoadQwen3`) and one decoder implementation. Distinctions:

| Feature | Qwen 3 | Qwen 2 | Llama 3 |
|---------|--------|--------|---------|
| Q/K norm | Yes | No | No |
| Sliding window | No | No | No |
| EOS token | `<\|im_end\|>` | `<\|im_end\|>` | `<\|eot_id\|>` |
| BOS token | `<\|im_start\|>` | `<\|im_start\|>` | `<\|begin_of_text\|>` |

Qwen 2 detection: if `model_type` is absent from config, weight presence of `model.layers.0.self_attn.q_norm.weight` distinguishes Qwen 3 (present) from Qwen 2 (absent).

Decoder structure per layer (standard pre-norm):

```
input → InputNorm    → Attention → residual add
      → PostAttnNorm → MLP       → residual add
```

MLP: SwiGLU gate — `down(silu(gate(x)) * up(x))`.

Output head: always a separate `lm_head.weight` (Qwen 3 has `tie_word_embeddings=false`).

### Weight Loading

All architectures load from HuggingFace safetensors format (not GGUF). The loader:

1. Reads `config.json` for model configuration
2. Loads `tokenizer.json` for the tokeniser
3. Glob-matches all `*.safetensors` files in the directory (multi-shard support)
4. Calls `LoadSafetensors` per shard; checks `lastError()` after each
5. Resolves weights by name, with automatic `language_model.` prefix fallback via `resolveWeight()`
6. Constructs `Linear` layers as quantised or dense based on presence of `scales` tensors
7. Calls `Materialize()` on all weight arrays to commit them to GPU memory

Quantisation is transparent: `NewQuantizedLinear` stores packed weights with `scales` and `biases`, dispatching to `QuantizedMatmul` (mlx-c grouped quantisation) in `Forward`. Quantisation parameters (`bits`, `group_size`) are read from top-level `config.json`.

Head dimension inference: if `head_dim` is absent from `config.json` (as with some Gemma 3 variants), it is inferred from `q_proj.weight[0]` / `num_attention_heads`.

---

## Attention Mechanism

### Virtual Transpose

Linear projections produce `[B, L, H*D]`. The reshape to `[B, H, L, D]` is implemented via `AsStrided` — a zero-copy stride manipulation that avoids a physical copy:

```
shape:   [B, H,   L, D]
strides: [L*H*D, D, H*D, 1]
```

- Batch stride: `L*H*D` (jump entire sequence)
- Head stride: `D` (adjacent heads are contiguous in memory)
- Sequence stride: `H*D` (jump one full row of heads)
- Element stride: `1` (contiguous within head)

The result is a non-contiguous view used for RoPE and SDPA calls.

### Rotary Position Embeddings (RoPE)

Applied via the fused `mlx_fast_rope` Metal kernel. Parameters:

- `dims`: head dimension
- `traditional`: false (standard non-interleaved layout)
- `base`: theta (varies by layer type in Gemma 3; single value for Qwen/Llama)
- `scale`: 1.0 (no frequency scaling)
- `offset`: current KV cache offset (enables continuation from cached position)

### Scaled Dot-Product Attention (SDPA)

Implemented via the fused `mlx_fast_scaled_dot_product_attention` kernel with two variants:

- `ScaledDotProductAttention(q, k, v, scale, causal)` — causal masking handled internally by the kernel
- `ScaledDotProductAttentionWithMask(q, k, v, mask, scale)` — explicit additive mask (0 = attend, -inf = ignore), used for batched inference with padding

Scale = `1/sqrt(head_dim)`, precomputed at load time.

After SDPA, output is transposed from `[B, H, L, D]` back to `[B, L, H*D]` via `Reshape(Transpose(out, 0, 2, 1, 3), ...)` for the output projection.

---

## KV Cache

The `Cache` interface provides `Update(k, v *Array, seqLen int) (*Array, *Array)`, returning the full accumulated K/V to pass to SDPA. `Offset()` tracks total tokens processed for RoPE continuation.

### KVCache (Unbounded)

Pre-allocates in 256-token chunks, growing as needed. On each decode step:

1. Checks whether the current buffer capacity is sufficient
2. If not, allocates a new chunk and concatenates it
3. Writes the new K/V via `SliceUpdateInplace`
4. Returns a slice view `[0:offset]` of the buffer

This amortises allocation cost while keeping the returned slice valid for the SDPA call.

### RotatingKVCache (Sliding Window)

Bounded to `maxSize` tokens. Two update paths:

- Prefill (`seqLen > 1`): concatenate, then trim the leading tokens that fall outside the window
- Decode (`seqLen == 1`): write in-place at circular index `idx % maxSize`

Used for Gemma 3 sliding-window attention layers (window size from `sliding_window` config field). Qwen and Llama use only unbounded caches.

---

## Tokeniser

`Tokenizer` supports two BPE variants detected at load time from `tokenizer.json`:

### SentencePiece BPE (Gemma 3)

- Prefix each segment with `▁` (Unicode U+2581, the SentencePiece space marker)
- Split into characters
- Apply BPE merges via `bpeMerge()` using a rank-sorted lookup table
- Look up merged symbols in the vocabulary

Detection: checks for absence of `Ġthe` in the vocabulary. Large SentencePiece vocabularies (Gemma 3 at 262K entries) may contain `Ġ` as an unrelated character, so the detection checks `Ġthe` rather than bare `Ġ`.

### GPT-2 Byte-Level BPE (Qwen, Llama, DeepSeek)

- Maps all 256 bytes to printable Unicode via `buildGPT2ByteMaps()`
- Printable ASCII (33–126) and Latin-1 Supplement (161–172, 174–255) map to themselves
- Control characters, space (32), DEL (127), and gap values (0–32, 127–160, 173) map to U+0100 onwards
- Apply BPE merges in this Unicode representation, then look up in vocabulary

Detection: presence of `Ġthe` in the vocabulary.

### BPE Merge Algorithm

`bpeMerge()` implements the standard greedy algorithm:

1. Build merge rank table from `tokenizer.json` merges field (O(1) lookup by `"a b"` key)
2. Scan all adjacent pairs; find the pair with the lowest rank
3. Merge that pair into a single symbol
4. Repeat until no merge can be applied

Merges are parsed from both `["a b", ...]` and `[["a","b"], ...]` JSON formats.

### Special Token Handling

Special tokens (BOS, EOS, chat delimiters) are matched before BPE encoding. Each architecture family uses different stop tokens:

| Family | BOS | EOS / Stop |
|--------|-----|-----------|
| Gemma 3 | `<bos>` | `<end_of_turn>` |
| Qwen 2/3 | `<\|im_start\|>` | `<\|im_end\|>` |
| Llama 3 | `<\|begin_of_text\|>` | `<\|eot_id\|>` |

---

## Generation Loop

`Model.Generate(ctx, prompt, cfg)` returns `iter.Seq[Token]` (range-over-func). The iterator:

1. Encodes the prompt via `Tokenizer.Encode()`
2. Allocates per-layer KV caches via `newCaches()`
3. Prefill: runs `model.Forward(tokens, caches)` on the full prompt in one pass; records prefill timing
4. Decode loop, up to `MaxTokens`:
   - Checks `ctx.Done()`; sets `m.lastErr = ctx.Err()` and returns on cancellation
   - Slices last-position logits via `SliceAxis`
   - Applies `applyRepeatPenalty` if `RepeatPenalty > 1.0`
   - Samples via the configured `Sampler` chain; calls `Eval()` and propagates any GPU error
   - Checks EOS token and `StopTokens` IDs
   - Yields `Token{ID, Text}` to the consumer; stops if `yield` returns false
   - Runs `model.Forward({next_token}, caches)` with the single new token
5. Records decode timing and memory metrics in `m.lastMetrics` via deferred closure

`Model.Err()` returns the error from the most recent `Generate` or `Chat` call.

### Repeat Penalty

`applyRepeatPenalty(logits, history, penalty)` deduplicates the history, gathers logits at those positions, then applies:
- Positive logits: divide by `penalty` (reduces probability)
- Negative logits: multiply by `penalty` (increases magnitude, reducing probability further)

### Chat Templates

`Model.Chat()` formats messages through `formatChat()` before calling `Generate()`:

| Architecture | Format |
|-------------|--------|
| Gemma 3 | `<start_of_turn>role\ncontent<end_of_turn>\n` |
| Qwen 2/3 | `<\|im_start\|>role\ncontent<\|im_end\|>\n` |
| Llama 3 | `<\|start_header_id\|>role<\|end_header_id\|>\n\ncontent<\|eot_id\|>` |

---

## Batch Inference

### Classify (Prefill-Only)

`Model.Classify(ctx, prompts, cfg, returnLogits)` runs a single forward pass per batch — no decode loop. The batch is right-padded to the length of the longest prompt:

1. Tokenise all prompts
2. Sort by descending token count
3. Build a `[N, 1, L, L]` attention mask combining causal masking and padding (0 = attend, -inf = ignore)
4. Run `ForwardMasked(tokens, mask, caches)` on the padded batch
5. Extract last-position logits for each prompt
6. Sample or return raw logits per the configuration

Measured throughput on M3 Ultra: 152 prompts/s for 4-prompt batches (Gemma3-1B 4-bit).

### BatchGenerate (Autoregressive Batches)

`Model.BatchGenerate(ctx, prompts, cfg)` runs full autoregressive generation for multiple prompts using the same masking approach as `Classify`. Each decode step processes the entire batch in one `ForwardMasked` call. Returns `[]BatchResult`, each holding the generated tokens and any per-prompt error.

---

## Sampling Chain

`newSampler(temp, topP, minP, topK)` builds a composable pipeline:

```
TopP -> MinP -> TopK -> Temperature -> RandomCategorical
```

If `temp == 0`, the chain collapses to greedy (argmax). Otherwise, each filter stage masks logits before the final categorical sample.

- **Greedy**: `Argmax(logits, -1)`
- **Temperature**: multiply logits by `1/temp`
- **TopK**: mask all but the K highest logits with `-inf`
- **TopP (nucleus)**: keep the smallest set with cumulative probability exceeding `p`; implemented via argsort, cumsum, and `PutAlongAxis` scatter back to original positions
- **MinP**: mask tokens whose probability falls below `min_p * max_probability`

---

## Training Pipeline

### LoRA Fine-Tuning

`InternalModel.ApplyLoRA(cfg)` wraps target projection layers in-place. The `LoRALinear` struct:

```go
type LoRALinear struct {
    Base  *Linear // frozen base weights (may be quantised)
    A     *Array  // [rank, in_features] — Kaiming normal initialisation
    B     *Array  // [out_features, rank] — zero initialisation
    Scale float32 // alpha / rank
}
```

Forward pass: `base(x) + scale * (x @ A^T) @ B^T`

B is zero-initialised so LoRA starts as the identity transformation (no change to base output).

`LoRAAdapter` collects all `LoRALinear` instances by weight path key. `AllTrainableParams()` returns A and B arrays in deterministic sorted order for use with `ValueAndGrad`. `LoRAAdapter.Save(path)` writes only the A and B matrices to safetensors (not the frozen base weights).

### Gradient Computation

Three autodiff interfaces via mlx-c:

- `VJP(fn, primals, cotangents)` — reverse mode (backward pass)
- `JVP(fn, primals, tangents)` — forward mode (directional derivative)
- `ValueAndGrad(fn, argnums)` — returns a `GradFn` that computes both value and gradients in one call

Go functions are registered as mlx-c closures via `goGradFunc` (exported CGO callback) using an atomic ID registry (`gradNextID atomic.Uintptr`).

### Gradient Checkpointing

`Checkpoint(fn)` wraps a function using `mlx_checkpoint`, which recomputes intermediate activations during the backward pass rather than storing them. Trades compute for GPU memory — useful for large models on constrained hardware.

### Mixed Precision

`LoRAConfig.DType` selects the dtype for A and B matrices. `DTypeBFloat16` halves parameter memory with accuracy matching Float32 in practice (validated: loss 7.15→6.29 in 5 steps). MLX auto-promotes operands for cross-dtype operations.

### AdamW Optimiser

Standard AdamW with decoupled weight decay:

```
m = beta1*m + (1-beta1)*grad
v = beta2*v + (1-beta2)*grad^2
param = param*(1 - lr*wd) - lr * m_hat / (sqrt(v_hat) + eps)
```

Defaults: `lr=1e-5`, `beta1=0.9`, `beta2=0.999`, `eps=1e-8`, `weight_decay=0.01`.

### Loss Functions

- `CrossEntropyLoss(logits, targets)` — numerically stable via logsumexp; averaged over all positions
- `MaskedCrossEntropyLoss(logits, targets, mask)` — averaged over masked positions only
- `MSELoss(predictions, targets)` — mean squared error

---

## Fused Metal Kernels

`internal/metal/fast.go` wraps four mlx-c fused kernels:

| Kernel | Go function | Notes |
|--------|------------|-------|
| `mlx_fast_rms_norm` | `RMSNorm(x, weight, eps)` | Gemma uses pre-scaled `(1+weight)` |
| `mlx_fast_layer_norm` | `LayerNorm(x, weight, bias, eps)` | Standard layer norm |
| `mlx_fast_rope` | `RoPE(x, dims, traditional, base, scale, offset)` | Rotary position embeddings |
| `mlx_fast_scaled_dot_product_attention` | `ScaledDotProductAttention(...)` | Causal or explicit mask |

These bypass the general MLX computation graph, dispatching directly to optimised Metal compute shaders.

---

## go-inference Integration

The public API is provided entirely by `forge.lthn.ai/core/go-inference`. go-mlx exports only Metal-specific controls:

- `MetalAvailable() bool` — hardware check
- `SetCacheLimit`, `SetMemoryLimit`, `GetActiveMemory`, `GetPeakMemory`, `ClearCache`, `GetCacheMemory`, `ResetPeakMemory`, `SetWiredLimit`, `GetDeviceInfo`

`register_metal.go` auto-registers `metalBackend` via `init()` on darwin/arm64. The adapter (`metalAdapter`) converts between `inference.*` types and `metal.*` types, implementing: `Generate`, `Chat`, `Classify`, `BatchGenerate`, `Metrics`, `Info`, `ModelType`, `Err`, `Close`.

Consumer pattern:

```go
import (
    "forge.lthn.ai/core/go-inference"
    _ "forge.lthn.ai/core/go-mlx"
)

m, err := inference.LoadModel("/path/to/model/")
for tok := range m.Generate(ctx, "prompt", inference.WithMaxTokens(128)) {
    fmt.Print(tok.Text)
}
```

`inference.LoadConfig` options understood by the Metal backend:
- `ContextLen` — replaces unbounded `KVCache` with `RotatingKVCache(contextLen)` for all layers
- `GPULayers` — logged as a warning if set to 0 (Metal always uses full GPU offload)

### AttentionInspector (Q/K Bone Orientation)

`metalAdapter` implements the optional `inference.AttentionInspector` interface, enabling Q/K Bone Orientation analysis from the KV cache.

```go
inspector, ok := model.(inference.AttentionInspector)
snap, err := inspector.InspectAttention(ctx, "What is kindness?")
// snap.Keys[layer][head] → post-RoPE K vectors as flat float32
```

**How it works:**

1. The prompt is tokenised and a single prefill pass populates all layer KV caches
2. For each layer, `cache.State()[0]` returns the K tensor with shape `[1, num_kv_heads, seq_alloc, head_dim]`
3. The tensor is sliced to valid token positions (cache may pre-allocate padding beyond `seq_len`)
4. K vectors are copied to CPU float32 slices via `.Floats()` and reshaped to `[head][seq_len * head_dim]`
5. GPU arrays are freed immediately after extraction

The K tensors are post-RoPE — rotary position embeddings have already been applied during the attention forward pass. This is the same data the model uses for attention scoring, making it suitable for coherence analysis.

For GQA models (Gemma3), `num_kv_heads` may be 1 per layer while `num_query_heads` is 8+. The returned snapshot reflects the KV head count, not query heads.

---

## mlxlm Subprocess Backend

`mlxlm/` provides a second backend (`"mlx_lm"`) that does not require CGO. It spawns a Python 3 process running the embedded `bridge.py` script and communicates via JSON Lines over stdin/stdout.

### Protocol

Commands sent to stdin (newline-delimited JSON):

| Command | Request fields | Response |
|---------|---------------|----------|
| `load` | `path` | `{ok, model_type, vocab_size}` or `{error}` |
| `generate` | `prompt, max_tokens, temperature?, top_k?, top_p?` | stream of `{token, token_id}`, then `{done}` |
| `chat` | `messages, max_tokens, ...` | same as generate |
| `info` | — | `{model_type, vocab_size, layers, hidden_size}` |
| `cancel` | — | subprocess drains and returns `{done}` |
| `quit` | — | subprocess exits cleanly |

Concurrent `Generate`/`Chat` calls are serialised via `sync.Mutex` (one generation at a time per subprocess instance).

### bridge.py

Embedded via `//go:embed bridge.py`, extracted to a temp file on first use via `sync.Once`. Uses `mlx_lm.load()` and `mlx_lm.stream_generate()` from the `mlx-lm` Python package. Flushes stdout after every line (critical for streaming).

### Limitations

- `Classify` and `BatchGenerate` are not supported (return error directing caller to use the native Metal backend)
- No inference metrics (`Metrics()` returns zero values)
- Requires Python 3 and `mlx-lm` installed in the Python environment
- Build tag `nomlxlm` removes the backend entirely

---

## Downstream Consumers

| Package | Role |
|---------|------|
| `forge.lthn.ai/core/go-ml` | Imports go-inference + go-mlx for Metal backend |
| `forge.lthn.ai/core/go-i18n` | Phase 2a: Gemma3-1B domain classification |
| `forge.lthn.ai/core/go-rocm` | Sibling AMD GPU backend, same go-inference interfaces |

---

## Performance Baseline (M3 Ultra, 60-core GPU, 96 GB unified memory)

| Operation | Throughput |
|-----------|-----------|
| Gemma3-1B 4-bit prefill | 246 tok/s |
| Gemma3-1B 4-bit decode | 82 tok/s |
| Gemma3-1B 4-bit classify (4 prompts) | 152 prompts/s |
| DeepSeek R1 7B 4-bit decode | 27 tok/s |
| Llama 3.1 8B 4-bit decode | 30 tok/s |

CGO call overhead floors at approximately 170 µs per operation (Metal command buffer + CGO boundary). MatMul scales well: 128² to 4096² is roughly 55× slower for 1024× more work. Full sampling chain (TopP+MinP+TopK) adds approximately 560 µs over greedy per token.

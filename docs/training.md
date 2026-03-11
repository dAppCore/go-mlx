---
title: Training
description: LoRA fine-tuning, gradient computation, AdamW optimiser, and loss functions.
---

# Training

go-mlx provides a complete LoRA fine-tuning pipeline on Metal: low-rank adapters, reverse-mode autodiff, the AdamW optimiser, loss functions, and gradient checkpointing. The training primitives are re-exported from `internal/metal/` at the root package level for use by downstream consumers like `go-ml`.

## LoRA Fine-Tuning

Low-Rank Adaptation (LoRA) injects small trainable matrices into frozen model layers. Only the LoRA parameters are updated during training -- the base model weights remain unchanged.

### How It Works

Each targeted `Linear` layer is wrapped with a `LoRALinear` struct:

```go
type LoRALinear struct {
    Base  *Linear // frozen base weights (may be quantised)
    A     *Array  // [rank, in_features] -- Kaiming normal initialisation
    B     *Array  // [out_features, rank] -- zero initialisation
    Scale float32 // alpha / rank
}
```

The forward pass computes: `base(x) + scale * (x @ A^T) @ B^T`

B is zero-initialised, so LoRA starts as the identity transformation -- no change to the base output until training begins.

### Applying LoRA

Through the `go-inference` `TrainableModel` interface:

```go
m, err := inference.LoadModel("/path/to/model/")
trainable := m.(inference.TrainableModel)

adapter := trainable.ApplyLoRA(inference.LoRAConfig{
    Rank:       8,
    Alpha:      16,
    TargetKeys: []string{"q_proj", "v_proj"},
    BFloat16:   true, // use BFloat16 for A/B matrices
})
```

Or directly via the Metal types:

```go
concreteAdapter := mlx.ConcreteAdapter(adapter)
fmt.Printf("LoRA params: %d\n", concreteAdapter.TotalParams())
```

### Configuration

```go
type LoRAConfig struct {
    Rank       int      // decomposition rank (default 8)
    Alpha      float32  // scaling factor (default 16)
    TargetKeys []string // weight name suffixes to target (default: q_proj, v_proj)
    DType      DType    // training dtype for A/B (default Float32; BFloat16 for mixed precision)
}
```

`DefaultLoRAConfig()` returns `{Rank: 8, Alpha: 16, TargetKeys: ["q_proj", "v_proj"], DType: Float32}`.

Common target keys: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`.

### Saving and Loading Adapters

Save trained adapter weights (only A and B matrices, not base weights):

```go
concreteAdapter := mlx.ConcreteAdapter(adapter)
err := concreteAdapter.Save("/path/to/adapter.safetensors")
```

Load a pre-trained adapter at model load time:

```go
m, err := inference.LoadModel("/path/to/model/",
    inference.WithAdapterPath("/path/to/adapter/"),
)
```

The adapter directory must contain:
- `adapter_config.json` -- rank, alpha, target layers
- One or more `*.safetensors` files -- adapter weights

The loader parses weight names like `layers.0.self_attn.q_proj.lora_a` to inject each A/B pair into the correct model layer. This is compatible with adapters trained by `mlx-lm`.

## Gradient Computation

### ValueAndGrad

The primary API for training loops. Creates a reusable `GradFn` that computes both the function value and gradients:

```go
lossFn := func(params []*mlx.Array) []*mlx.Array {
    // Forward pass + loss computation
    return []*mlx.Array{loss}
}

grad := mlx.ValueAndGrad(lossFn, 0) // differentiate w.r.t. first argument

// In the training loop:
values, grads, err := grad.Apply(adapterParams...)
```

The `argnums` parameter specifies which arguments to differentiate with respect to. Default is `{0}` (first argument only).

### VJP (Reverse Mode)

Low-level backward pass for custom gradient computation:

```go
outputs, vjps, err := metal.VJP(fn, primals, cotangents)
```

Given a function, input primals, and output cotangents (upstream gradients), returns (outputs, gradients) where gradients are with respect to the primals.

### JVP (Forward Mode)

Directional derivative computation:

```go
outputs, jvps, err := metal.JVP(fn, primals, tangents)
```

Useful for Hessian-vector products and directional derivatives.

### Closure Registration

Go functions are registered as mlx-c closures via an exported CGO callback (`goGradFunc`). Each closure gets a unique ID from an atomic counter (`gradNextID`), stored in a `sync.Map` for concurrent access. A C-side destructor (`goGradDestructor`) cleans up the registry entry when the closure is freed.

## AdamW Optimiser

Standard AdamW with decoupled weight decay:

```go
opt := mlx.NewAdamW(1e-4) // learning rate
```

Update rule per parameter per step:

```
m = beta1 * m + (1 - beta1) * grad
v = beta2 * v + (1 - beta2) * grad^2
m_hat = m / (1 - beta1^t)       // bias correction
v_hat = v / (1 - beta2^t)
param = param * (1 - lr * wd) - lr * m_hat / (sqrt(v_hat) + eps)
```

### Hyperparameters

| Field | Default | Description |
|-------|---------|-------------|
| `LR` | 1e-5 | Learning rate |
| `Beta1` | 0.9 | First moment decay |
| `Beta2` | 0.999 | Second moment decay |
| `Eps` | 1e-8 | Numerical stability |
| `WeightDecay` | 0.01 | Decoupled weight decay |

### Usage

```go
opt := mlx.NewAdamW(1e-4)
opt.WeightDecay = 0.01

// Training loop
params := adapter.AllTrainableParams()
for step := range numSteps {
    values, grads, err := gradFn.Apply(params...)
    if err != nil {
        log.Fatal(err)
    }

    loss := values[0]
    mlx.Materialize(loss)
    fmt.Printf("step %d: loss = %.4f\n", step, loss.Float())

    params = opt.Step(params, grads)
    adapter.SetAllParams(params)
}
```

`Step()` returns new parameter arrays. The moment estimates are maintained internally and grow lazily on first use. Call `Reset()` to clear all optimiser state.

## Loss Functions

### CrossEntropyLoss

```go
loss := mlx.CrossEntropyLoss(logits, targets)
```

Numerically stable via logsumexp: `loss_i = logsumexp(logits_i) - logits_i[target_i]`. Averaged over all positions.

- `logits`: `[..., V]` (raw model output, pre-softmax, last dim = vocab size)
- `targets`: `[...]` (integer token IDs, same shape minus last dim)

### MaskedCrossEntropyLoss

```go
loss := mlx.MaskedCrossEntropyLoss(logits, targets, mask)
```

Same as `CrossEntropyLoss` but only computes loss at masked positions. The mask has shape `[B, L]` with values 1.0 (compute loss) or 0.0 (ignore). Averaged over masked positions only.

Use this for training on chat data where system/user tokens should be excluded from the loss.

### MSELoss

```go
loss := metal.MSELoss(predictions, targets)
```

Mean squared error: `mean((predictions - targets)^2)`.

## Gradient Checkpointing

`Checkpoint` wraps a function so that during the backward pass, intermediate activations are recomputed rather than stored. This trades compute time for GPU memory:

```go
checkpointedFn := mlx.Checkpoint(func(params []*mlx.Array) []*mlx.Array {
    // Forward pass (intermediates discarded after forward, recomputed on backward)
    return []*mlx.Array{loss}
})
```

Use this for memory-constrained training with large models. The checkpointed function is wrapped via `mlx_checkpoint` on the C side, with a Go finaliser managing the closure lifetime.

## Mixed Precision

`LoRAConfig.DType` selects the dtype for A and B matrices:

- `DTypeFloat32` (default) -- full precision training
- `DTypeBFloat16` -- halves parameter memory with accuracy matching Float32 in practice

```go
adapter := trainable.ApplyLoRA(inference.LoRAConfig{
    Rank:     8,
    Alpha:    16,
    BFloat16: true,
})
```

MLX auto-promotes operands for cross-dtype operations (e.g. BFloat16 LoRA matrices multiplied with Float16 base weights), so no manual casting is needed in the training loop.

## Training Type Exports

The root `mlx` package re-exports training types from `internal/metal/` for use by downstream consumers:

```go
type Array     = metal.Array
type LoRAAdapter = metal.LoRAAdapter
type LoRAConfig  = metal.LoRAConfig
type GradFn    = metal.GradFn
type AdamW     = metal.AdamW
type Cache     = metal.Cache
type DType     = metal.DType
type InternalModel = metal.InternalModel
```

Exported functions:

| Function | Purpose |
|----------|---------|
| `ValueAndGrad(fn, argnums...)` | Create a GradFn for combined value + gradient computation |
| `NewAdamW(lr)` | Create an AdamW optimiser |
| `CrossEntropyLoss(logits, targets)` | Standard cross-entropy loss |
| `MaskedCrossEntropyLoss(logits, targets, mask)` | Masked cross-entropy loss |
| `Checkpoint(fn)` | Memory-efficient gradient recomputation |
| `FromValues(slice, shape...)` | Create a Metal Array from a Go slice |
| `Materialize(arrays...)` | Force GPU evaluation |
| `Free(arrays...)` | Release Metal arrays immediately |
| `Zeros(shape, dtype)` | Create a zero-filled array |
| `ConcreteAdapter(a)` | Extract `*LoRAAdapter` from an `inference.Adapter` |
| `TrainingModel(tm)` | Extract `InternalModel` from a `TrainableModel` for direct `Forward()` access |

## Training via go-ml

The typical training workflow uses `go-ml`, which orchestrates the training loop on top of go-mlx primitives:

```go
// go-ml loads a TrainableModel via go-inference + go-mlx
tm, err := inference.LoadTrainable("/path/to/model/")

// Apply LoRA
adapter := tm.ApplyLoRA(inference.LoRAConfig{Rank: 8, Alpha: 16})

// Get direct model access for Forward()
model := mlx.TrainingModel(tm)

// Build training loop with ValueAndGrad, AdamW, etc.
// See go-ml for the full implementation.
```

The `InternalModel` interface provides `Forward(tokens, caches)` and `NewCache()` for direct control over the forward pass in training, bypassing the generation loop.

## InternalModel Interface

For training code that needs direct forward pass access:

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

`Forward` returns logits of shape `[B, L, V]`. `ForwardMasked` accepts an explicit attention mask for batched training with padded sequences.

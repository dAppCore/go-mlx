# Batch Inference

Two batch entry points serve different workloads:

| API | Workload | Output |
|-----|----------|--------|
| `Classify` | Multiple prompts, one logit step each | Per-prompt top-k tokens or label probabilities |
| `BatchGenerate` | Multiple prompts, autoregressive decode | Per-prompt full completions |

`Classify` is prefill-only — much faster than running N independent `Generate` calls because the GPU stays saturated.

## Classify (Prefill-Only)

Useful for label-style problems: domain classifiers, sentiment, intent routing.

```go
package main

import (
    "context"
    "fmt"

    "dappco.re/go/inference"
    _ "dappco.re/go/mlx"
)

func main() {
    model, err := inference.LoadModel("/Volumes/Data/lem/safetensors/gemma-3-1b/")
    if err != nil {
        panic(err)
    }
    defer model.Close()

    prompts := []string{
        "Refund my order immediately, this is unacceptable.",
        "Could you clarify the SLA on the enterprise tier?",
        "Lol nvm",
        "DROP TABLE users; --",
    }

    results, err := model.Classify(context.Background(), prompts,
        inference.WithLabels([]string{"hostile", "business", "casual", "technical"}),
        inference.WithTopK(1),
    )
    if err != nil {
        panic(err)
    }

    for i, r := range results {
        fmt.Printf("[%s] %s\n", r.TopLabel, prompts[i])
    }
}
```

On an M3 Ultra, Gemma3-1B 4-bit classifies ~150 prompts/second.

## BatchGenerate (Autoregressive)

Use when each prompt needs a full streamed reply, not just a label. The implementation pads prompts to the longest in the batch, runs prefill once, then decodes step-by-step until each row hits its EOS or `MaxTokens`.

```go
prompts := []string{
    "Summarise: The quick brown fox jumped over the lazy dog.",
    "Translate to French: Where is the bathroom?",
    "Write one haiku about silicon.",
}

results, err := model.BatchGenerate(context.Background(), prompts,
    inference.WithMaxTokens(128),
    inference.WithTemperature(0.7),
)
if err != nil {
    panic(err)
}

for i, r := range results {
    fmt.Printf("=== prompt %d ===\n%s\n%s\n\n", i, prompts[i], r.Text)
}
```

Each result carries `.Text` (the generated completion), `.Tokens` (the token IDs), and `.FinishReason` (`"eos"`, `"max_tokens"`, or `"stop_token"`).

## Performance Notes

- Padding is memory-bound, not compute-bound — short prompts in a batch with long ones still pay for the long one's KV. Keep batch members similar in length when possible.
- For mixed-length batches, the autoregressive loop short-circuits decoded rows once they hit EOS, so total wall-time is dominated by the longest reply, not the sum.
- Batch mask is `[N, 1, L, L]` additive, built in Go and pushed to Metal on the first prefill step.
- `Classify` has no decode loop and is therefore the fastest possible per-prompt path.

## Performance Baseline (M3 Ultra, Gemma3-1B 4-bit)

| Workload | Throughput |
|----------|-----------|
| `Classify`, batch=4 | 152 prompts/s |
| `BatchGenerate`, batch=4, 128 tokens | ~30 completions/s |
| `Generate`, single, 128 tokens | 46 tok/s |

# Streaming Inference

Token-by-token generation via the `Generate` iterator. The iterator yields one `Token` per decoded step until the model emits EOS or `MaxTokens` is reached.

## Two API Styles

### Via `go-inference` (interface-driven)

```go
package main

import (
    "context"
    "fmt"

    "dappco.re/go/inference"
    _ "dappco.re/go/mlx" // registers "metal" backend via init()
)

func main() {
    model, err := inference.LoadModel("/Volumes/Data/lem/safetensors/gemma-3-1b/")
    if err != nil {
        panic(err)
    }
    defer model.Close()

    ctx := context.Background()
    for tok := range model.Generate(ctx, "Why is the sky blue?", inference.WithMaxTokens(256)) {
        fmt.Print(tok.Text)
    }
    if err := model.Err(); err != nil {
        panic(err)
    }
    fmt.Println()
}
```

### Via the root `mlx` API (direct)

```go
package main

import (
    "fmt"

    mlx "dappco.re/go/mlx"
)

func main() {
    model, err := mlx.LoadModel("/Volumes/Data/lem/safetensors/qwen3-8b/",
        mlx.WithContextLength(8192),
        mlx.WithDevice("gpu"),
    )
    if err != nil {
        panic(err)
    }
    defer model.Close()

    for tok := range model.GenerateStream("Explain Gemma 4 shared KV layers.", mlx.WithMaxTokens(256)) {
        fmt.Print(tok.Text)
    }
    if err := model.Err(); err != nil {
        panic(err)
    }
}
```

## Cancellation

Generation respects context cancellation. Cancelling stops the loop at the next decoded token; partial output up to that point is still valid:

```go
ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
defer cancel()

for tok := range model.Generate(ctx, prompt, inference.WithMaxTokens(2048)) {
    fmt.Print(tok.Text)
}
```

## Sampler Options

```go
model.Generate(ctx, prompt,
    inference.WithMaxTokens(512),
    inference.WithTemperature(0.7),
    inference.WithTopP(0.95),     // currently a stub — falls back to plain sampling
    inference.WithStopTokens([]string{"User:", "</s>"}),
)
```

## Notes

- Tokens carry both `Text` (decoded UTF-8) and `ID` (vocab index) — useful for logging or downstream logit analysis
- The first call to `Generate` warms the prompt cache; subsequent calls with the same prefix replay from cache
- For multi-turn chat, prefer the `Chat` API ([chat.md](chat.md)) — it handles role templates and KV reuse correctly
- Call `model.Close()` to release Metal resources; the finaliser will eventually free them but explicit close is recommended in long-lived processes

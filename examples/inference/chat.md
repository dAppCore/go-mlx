# Multi-Turn Chat

`Chat` consumes a `[]Message` history and generates an assistant reply. The model's chat template (Gemma 3, Gemma 4, Qwen 2/3, Llama 3) is applied automatically — you don't render the template yourself.

## Single Turn

```go
package main

import (
    "context"
    "fmt"

    "dappco.re/go/inference"
    _ "dappco.re/go/mlx"
)

func main() {
    model, err := inference.LoadModel("/Volumes/Data/lem/safetensors/qwen3-8b/")
    if err != nil {
        panic(err)
    }
    defer model.Close()

    messages := []inference.Message{
        {Role: "system", Content: "You are a precise, terse assistant."},
        {Role: "user", Content: "What's 17 * 23?"},
    }

    for tok := range model.Chat(context.Background(), messages, inference.WithMaxTokens(64)) {
        fmt.Print(tok.Text)
    }
    fmt.Println()
}
```

## Multi-Turn With KV Reuse

To keep the conversation efficient, append turns to the same `messages` slice and reuse the model. The prompt cache replays the unchanged prefix on each call:

```go
messages := []inference.Message{
    {Role: "system", Content: "You are a precise, terse assistant."},
}

addTurn := func(user string) string {
    messages = append(messages, inference.Message{Role: "user", Content: user})
    var reply strings.Builder
    for tok := range model.Chat(context.Background(), messages, inference.WithMaxTokens(256)) {
        reply.WriteString(tok.Text)
    }
    messages = append(messages, inference.Message{Role: "assistant", Content: reply.String()})
    return reply.String()
}

fmt.Println(addTurn("What's 17 * 23?"))
fmt.Println(addTurn("And the square of that?"))
```

## Warming the Prompt Cache

For a stable system prompt that doesn't change between sessions, warm it once at start:

```go
const stableSystem = `You are a domain classifier.
Respond with one of: technical, business, casual, hostile.`

if err := model.WarmPromptCache(stableSystem); err != nil {
    panic(err)
}
```

The warmed prefix is reused on every subsequent `Chat` whose first message starts with it — no re-prefilling.

## Stopping Conditions

Chat models often need explicit stop tokens to avoid bleeding into the next role's turn:

```go
model.Chat(ctx, messages,
    inference.WithMaxTokens(512),
    inference.WithStopTokens([]string{"<|im_end|>", "</s>"}),
)
```

Stop tokens are matched on the decoded text after each step.

## Architecture-Specific Templates

| Model family | Template format |
|--------------|----------------|
| Gemma 3 / 4 | `<start_of_turn>role\ncontent<end_of_turn>` |
| Qwen 2 / 3 | `<|im_start|>role\ncontent<|im_end|>` |
| Llama 3 | `<|start_header_id|>role<|end_header_id|>\n\ncontent<|eot_id|>` |

`Chat` applies the right template based on `config.json`'s `model_type`. If you need the rendered text for inspection or to feed it into `Generate`, use `model.RenderChatTemplate(messages)` (returns the string the tokenizer will see).

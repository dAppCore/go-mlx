<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# parser_registry.go — model-family output parser registry

**Package**: `dappco.re/go/mlx`
**File**: `go/parser_registry.go`

## What this is

The **registry** for model-family-specific output parsers. Different models emit reasoning channels and tool-calls in different formats; the registry maps a model-family / architecture id to a parser that knows how to extract them.

Each parser implements both `inference.ReasoningParser` (`<think>...</think>` channels) and `inference.ToolParser` (structured tool calls) — they share output stream parsing logic, so co-locating them avoids duplicate state.

## ModelOutputParser

```go
type ModelOutputParser interface {
    ParserID() string
    inference.ReasoningParser  // ParseReasoning(tokens, text) (ReasoningParseResult, error)
    inference.ToolParser       // ParseTools(tokens, text) (ToolParseResult, error)
}
```

## ParserRegistry

```go
type ParserRegistry struct {
    parsers map[string]ModelOutputParser
    // …
}

reg := mlx.NewParserRegistry()
reg.Register("qwen-think", qwenParser)
reg.Register("gemma-think", gemmaParser)
reg.Register("deepseek-r1", deepseekParser)
reg.Register("minimax-tools", minimaxParser)
// …
parser, ok := reg.Get("qwen-think")
```

Registration happens at package init time (and at LoadModel time when the pack's JANG capabilities declare which parsers it expects).

## Parsers shipped

| ID | Reasoning channel | Tool call format |
|----|-------------------|------------------|
| `qwen-think` | `<think>...</think>` | Qwen JSON in `<tool_call>...</tool_call>` |
| `gemma-think` | `<think>...</think>` (Gemma 4 thinking) | Gemma function-call JSON |
| `deepseek-r1` | `<think>...</think>` (R1 style) | n/a |
| `minimax-tools` | (no reasoning) | MiniMax tool-call JSON |
| `default` | `<thinking>...</thinking>` fallback | OpenAI function-call JSON |

The default lane handles any model that doesn't declare a parser in its JANG capabilities — best-effort, doesn't always work.

## How a backend uses this

```go
// In register_metal_parser.go:
reg := getParserRegistry()
parser, ok := reg.Get(model.GetCapability().ReasoningParser)
if ok {
    adapter.reasoningParser = parser
    adapter.toolParser      = parser
}
```

A loaded `metaladapter` then satisfies `ReasoningParser` + `ToolParser` if the registry had a match for its pack's declared parser. Consumers probe via type assertion.

## Why a registry not hard-coded

Model families evolve. New reasoning notations appear (e.g., Gemma 4's thinking channel differs from Gemma 3's). The registry decouples parser identity from architecture so:

- New parsers ship without touching existing model paths
- A model pack can declare which parser via its JANG sidecar without code change
- Third-party packs can register their own parser at import time

## Related

- [thinking.md](thinking.md) — reasoning channel detection and mode policy
- `../../../go-inference/docs/inference/contracts.md` — `ReasoningParser` + `ToolParser` interfaces
- [../moe/jang.md](../moe/jang.md) — JANGCapabilities declares which parser to load
- `../openai/responses.md` — Responses API exposes reasoning channels separately

<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# thinking.go — reasoning channel mode policy

**Package**: `dappco.re/go/mlx`
**File**: `go/thinking.go`

## What this is

The **policy layer** for reasoning channels — given a model that emits `<think>...</think>` (or family-specific equivalent) blocks, what does the runtime do with them?

Three modes:

```go
ThinkingShow    // leave model output untouched (compat default)
ThinkingHide    // strip thinking text from visible output
ThinkingCapture // strip from visible + emit captured chunks separately
```

The actual parsing lives in `parser_registry.go`; this file owns "what does the runtime promise to do once parsed?"

## ThinkingChunk

```go
type ThinkingChunk struct {
    Text       string             // captured reasoning text
    TokenRange [2]int              // start/end token index
    Tag        string              // parser-specific tag (e.g. "<think>")
    Labels     map[string]string
}
```

When `ThinkingCapture` is set, generation emits chunks alongside the visible text — caller can render them separately, log them, or train against them.

## Usage

```go
result, err := adapter.Generate(ctx, prompt, mlx.GenOpts{
    MaxTokens: 1024,
    Thinking:  mlx.ThinkingCapture,
})

// result.Text         = visible answer only
// result.Thinking[]   = captured reasoning chunks
```

## ThinkingShow (default)

The compatibility mode. Output passes through verbatim. Used by:

- Legacy callers that don't know about thinking channels
- Models without thinking channels (default is harmless on them)
- Tests against full output

## ThinkingHide

Visible output strips `<think>...</think>` blocks but doesn't expose them. Used by:

- Production chat UI showing user-friendly answers
- Tool-use loops where reasoning is internal-only

## ThinkingCapture

Visible output strips reasoning; captured chunks delivered alongside. Used by:

- `core/ide` reasoning inspector panel
- GRPO training (capture the reasoning to score)
- Distillation cascades (capture teacher reasoning for student supervision)

## Channel-aware streaming

For streaming generation, the thinking mode affects how tokens are categorised mid-flight:

```
ThinkingShow:    every token → visible stream
ThinkingHide:    inside-block tokens → /dev/null; outside-block tokens → visible
ThinkingCapture: inside-block tokens → captured stream; outside-block tokens → visible
```

The Responses API streaming events (`response.thinking.delta` vs `response.output.delta`) line up with this — see [`responses.md`](../../../go-inference/docs/openai/responses.md).

## Why a policy layer not just "always show"

Different consumers want different things from the same model output. A test wants raw. A user UI wants clean. A reasoning panel wants both. A training loop wants the reasoning isolated. One model, four consumers — the mode lets each get what it needs from one Generate call.

## Related

- [parser_registry.md](parser_registry.md) — parses the actual `<think>` tags
- `../../../go-inference/docs/inference/contracts.md` — `ReasoningSegment` / `ReasoningParseResult` DTOs
- `../../../go-inference/docs/openai/responses.md` — Responses API surfaces thinking as a separate channel
- [../training/grpo.md](../training/grpo.md) — reasoning training that captures `<think>` blocks

// SPDX-Licence-Identifier: EUPL-1.2

// Package gemma4chat renders the Gemma 4 chat prompt — the <|turn> / <turn|>
// turn structure with the <|think|> system block, the assistant thought-channel
// strip, the consecutive-assistant-turn continuation, and the large-variant
// thought-channel suppressor. It is the gemma4 family's faithful distillation of
// the model's declared chat_template.jinja turn structure.
//
// It is pure Go (no metal/cgo import) so the SPOR builder is reachable from both
// the cgo serve path and the cgo-free training/dataset path. It registers itself
// with the neutral chat dispatcher from init(); a blank import wires it in.
package gemma4chat

import (
	core "dappco.re/go"
	"dappco.re/go/mlx/chat"
)

func init() {
	chat.RegisterFormatter("gemma4", Format)
}

// Format renders messages as a Gemma 4 chat prompt.
//
//	text := gemma4chat.Format(messages, chat.Config{EnableThinking: true})
func Format(messages []chat.Message, cfg chat.Config) string {
	builder := core.NewBuilder()
	builder.Grow(chat.FormatCapacity(messages, 17, 13, true) + len("<bos><turn|>\n"))

	start := 0
	if cfg.Continuation {
		// The session's retained state ends inside an open model turn —
		// generation stops on the end-of-turn token without retaining it — so
		// a continuation closes that turn and renders only the new turns.
		builder.WriteString("<turn|>\n")
	} else {
		builder.WriteString("<bos>")
		if cfg.EnableThinking || initialSystemRole(messages) {
			builder.WriteString("<|turn>system\n")
			if cfg.EnableThinking {
				builder.WriteString("<|think|>\n")
			}
			if len(messages) > 0 {
				role := gemmaRole(messages[0].Role)
				if role == "system" {
					builder.WriteString(core.Trim(messages[0].Content))
					start = 1
				}
			}
			builder.WriteString("<turn|>\n")
		}
	}

	prevNonToolRole := ""
	for _, msg := range messages[start:] {
		normalisedRole := chat.NormaliseRole(msg.Role)
		role := roleFromNormalised(normalisedRole)
		if role == "" {
			continue
		}
		content := core.Trim(msg.Content)
		if role == "model" {
			content = stripThinking(content)
		}
		continueSameModelTurn := role == "model" && prevNonToolRole == "assistant"
		if !continueSameModelTurn {
			builder.WriteString("<|turn>")
			builder.WriteString(role)
			builder.WriteString("\n")
		}
		builder.WriteString(content)
		builder.WriteString("<turn|>\n")
		prevNonToolRole = normalisedRole
	}
	if !cfg.NoGenerationPrompt {
		builder.WriteString("<|turn>model\n")
		if !cfg.EnableThinking && cfg.LargeVariant {
			// 12B/26B/31B ghost an empty thought channel when thinking is off; the
			// empty suppressor (per chat_template.jinja) makes them answer directly.
			builder.WriteString("<|channel>thought\n<channel|>")
		}
	}
	return builder.String()
}

func initialSystemRole(messages []chat.Message) bool {
	if len(messages) == 0 {
		return false
	}
	return gemmaRole(messages[0].Role) == "system"
}

func gemmaRole(role string) string {
	return roleFromNormalised(chat.NormaliseRole(role))
}

func roleFromNormalised(role string) string {
	switch role {
	case "assistant":
		return "model"
	case "system":
		return "system"
	case "developer":
		return "system"
	case "user":
		return "user"
	default:
		return ""
	}
}

const (
	channelOpen  = "<|channel>"
	channelClose = "<channel|>"
)

// stripThinking removes the assistant's private thought channels —
// every <|channel>…<channel|> span — and returns the trimmed visible
// remainder. An unclosed <|channel> (no matching <channel|>) drops the
// rest of the message, matching the model's own template behaviour.
//
//	stripThinking("<|channel>thought<channel|>answer") // "answer"
func stripThinking(text string) string {
	open := core.Index(text, channelOpen)
	if text == "" || open < 0 {
		// No thought channel: the visible text is the whole message.
		// core.Trim returns a sub-slice, so this path never allocates.
		return core.Trim(text)
	}

	// Single-block, no-leading-text fast path — the dominant production
	// shape "<|channel>thought…<channel|>answer". The visible remainder
	// is one contiguous suffix, so Trim hands back a sub-slice with zero
	// allocation and no Builder. Guarded on there being no further
	// <|channel> after the close so multi-block messages take the loop.
	if open == 0 {
		rest := text[len(channelOpen):]
		if closeIdx := core.Index(rest, channelClose); closeIdx >= 0 {
			after := rest[closeIdx+len(channelClose):]
			if core.Index(after, channelOpen) < 0 {
				return core.Trim(after)
			}
		}
	}

	// General case: stitch the visible segments. Output is always shorter
	// than the input (we only ever drop spans), so one Grow(len(text))
	// sizes the Builder exactly once and the loop adds no per-iteration
	// slice allocation — core.Index walks the string in place.
	out := core.NewBuilder()
	out.Grow(len(text))
	for {
		i := core.Index(text, channelOpen)
		if i < 0 {
			out.WriteString(text)
			break
		}
		out.WriteString(text[:i])
		text = text[i+len(channelOpen):]
		j := core.Index(text, channelClose)
		if j < 0 {
			// Unclosed channel: drop the remainder.
			break
		}
		text = text[j+len(channelClose):]
	}
	return core.Trim(out.String())
}

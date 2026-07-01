// SPDX-Licence-Identifier: EUPL-1.2

// Package gemma4chat renders the Gemma 4 chat prompt: the <|turn> / <turn|>
// turn structure with the <|think|> system block, assistant thought-channel
// stripping, consecutive-assistant-turn continuation, and the large-variant
// thought-channel suppressor. It is pure Go and registers with the neutral
// chat dispatcher from init(); a blank import wires it in.
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
		// The retained state ends inside an open model turn; continuation closes
		// it and renders only the new turns.
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
			// Large Gemma 4 variants ghost an empty thought channel when thinking
			// is off; the empty suppressor makes them answer directly.
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

// stripThinking removes the assistant's private thought channels and returns
// the trimmed visible remainder. An unclosed channel drops the rest of the
// message, matching the model template behaviour.
func stripThinking(text string) string {
	open := core.Index(text, channelOpen)
	if text == "" || open < 0 {
		return core.Trim(text)
	}

	if open == 0 {
		rest := text[len(channelOpen):]
		if closeIdx := core.Index(rest, channelClose); closeIdx >= 0 {
			after := rest[closeIdx+len(channelClose):]
			if core.Index(after, channelOpen) < 0 {
				return core.Trim(after)
			}
		}
	}

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
			break
		}
		text = text[j+len(channelClose):]
	}
	return core.Trim(out.String())
}

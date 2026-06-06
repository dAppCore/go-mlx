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
	builder.Grow(chat.FormatCapacity(messages, 17, 13, true) + len("<bos>"))
	builder.WriteString("<bos>")

	start := 0
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
			// 26B/31B ghost an empty thought channel when thinking is off; the
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

func stripThinking(text string) string {
	if text == "" || !core.Contains(text, "<|channel>") {
		return core.Trim(text)
	}
	out := core.NewBuilder()
	for {
		parts := core.SplitN(text, "<|channel>", 2)
		out.WriteString(parts[0])
		if len(parts) != 2 {
			break
		}
		after := core.SplitN(parts[1], "<channel|>", 2)
		if len(after) != 2 {
			break
		}
		text = after[1]
	}
	return core.Trim(out.String())
}

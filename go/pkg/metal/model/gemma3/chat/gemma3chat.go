// SPDX-Licence-Identifier: EUPL-1.2

// Package gemma3chat renders the Gemma chat prompt — the <start_of_turn> /
// <end_of_turn> turn structure with fixed user/model tags and the system message
// folded into the first user turn. It is the gemma (Gemma 1/2/3) family's
// faithful distillation of the model's declared chat_template.
//
// It is pure Go (no metal/cgo import) so the SPOR builder is reachable from both
// the cgo serve path and the cgo-free training/dataset path. It registers itself
// with the neutral chat dispatcher from init(); a blank import wires it in.
package gemma3chat

import (
	core "dappco.re/go"
	"dappco.re/go/mlx/chat"
)

func init() {
	chat.RegisterFormatter("gemma", Format)
}

// Format renders messages as a Gemma chat prompt.
//
//	text := gemma3chat.Format(messages, chat.Config{})
func Format(messages []chat.Message, cfg chat.Config) string {
	builder := core.NewBuilder()
	// Gemma writes fixed "user" / "model" tags — role is not emitted
	// per-message, so the capacity calc skips role overhead.
	builder.Grow(chat.FormatCapacity(messages, 34, 22, false) + len("<bos>"))
	builder.WriteString("<bos>")
	firstUserPrefix := ""
	start := 0
	if len(messages) > 0 && chat.NormaliseRole(messages[0].Role) == "system" {
		firstUserPrefix = core.Trim(messages[0].Content)
		start = 1
	}
	for _, msg := range messages[start:] {
		role := chat.NormaliseRole(msg.Role)
		switch role {
		case "assistant":
			builder.WriteString("<start_of_turn>model\n")
			builder.WriteString(core.Trim(msg.Content))
			builder.WriteString("<end_of_turn>\n")
		case "system", "user":
			builder.WriteString("<start_of_turn>user\n")
			if firstUserPrefix != "" {
				builder.WriteString(firstUserPrefix)
				builder.WriteString("\n\n")
				firstUserPrefix = ""
			}
			builder.WriteString(core.Trim(msg.Content))
			builder.WriteString("<end_of_turn>\n")
		}
	}
	if !cfg.NoGenerationPrompt {
		builder.WriteString("<start_of_turn>model\n")
	}
	return builder.String()
}

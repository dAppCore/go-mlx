// SPDX-Licence-Identifier: EUPL-1.2

// Package qwen3chat renders the chat conventions of the models the qwen3 (dense)
// loader serves: ChatML (Format — <|im_start|>role … <|im_end|>, used by Qwen)
// and the Llama header convention (FormatLlama — <|begin_of_text|> /
// <|start_header_id|>, used by the llama/mistral/granite/phi/glm archs that load
// through the same dense path).
//
// It is pure Go (no metal/cgo import) so the SPOR builders are reachable from
// both the cgo serve path and the cgo-free training/dataset path. It registers
// itself with the neutral chat dispatcher from init(); a blank import wires it
// in.
package qwen3chat

import (
	core "dappco.re/go"
	"dappco.re/go/mlx/chat"
)

func init() {
	chat.RegisterFormatter("qwen", Format)
	chat.RegisterFormatter("llama", FormatLlama)
}

// Format renders messages as a ChatML prompt.
//
//	text := qwen3chat.Format(messages, chat.Config{})
func Format(messages []chat.Message, cfg chat.Config) string {
	builder := core.NewBuilder()
	builder.Grow(chat.FormatCapacity(messages, 24, 23, true))
	for _, msg := range messages {
		role := chat.NormaliseRole(msg.Role)
		if role == "" {
			continue
		}
		builder.WriteString("<|im_start|>")
		builder.WriteString(role)
		builder.WriteString("\n")
		builder.WriteString(msg.Content)
		builder.WriteString("<|im_end|>\n")
	}
	if !cfg.NoGenerationPrompt {
		builder.WriteString("<|im_start|>assistant\n")
	}
	return builder.String()
}

// FormatLlama renders messages as a Llama-header prompt, the convention the
// llama/mistral/granite/phi/glm archs the dense loader serves declare.
//
//	text := qwen3chat.FormatLlama(messages, chat.Config{})
func FormatLlama(messages []chat.Message, cfg chat.Config) string {
	builder := core.NewBuilder()
	builder.Grow(chat.FormatCapacity(messages, 52, 43, true) + len("<|begin_of_text|>"))
	builder.WriteString("<|begin_of_text|>")
	for _, msg := range messages {
		role := chat.NormaliseRole(msg.Role)
		if role == "" {
			continue
		}
		builder.WriteString("<|start_header_id|>")
		builder.WriteString(role)
		builder.WriteString("<|end_header_id|>\n\n")
		builder.WriteString(msg.Content)
		builder.WriteString("<|eot_id|>")
	}
	if !cfg.NoGenerationPrompt {
		builder.WriteString("<|start_header_id|>assistant<|end_header_id|>\n\n")
	}
	return builder.String()
}

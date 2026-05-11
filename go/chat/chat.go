// SPDX-Licence-Identifier: EUPL-1.2

// Package chat is the driver-neutral chat-template formatter. It maps
// inference.Message lists to architecture-specific tokenised text using
// the native chat template for each model family (Gemma, Gemma 4, Qwen,
// Llama, plain).
//
//	text := chat.Format(messages, chat.Config{Architecture: "qwen3"})
package chat

import (
	core "dappco.re/go"
	"dappco.re/go/inference"
)

// Message is the chat message envelope, aliased from the inference
// contract so callers do not need to import inference directly.
type Message = inference.Message

// Config selects the chat template used to render a message list.
// Architecture is consulted when Template is empty; Template overrides.
// NoGenerationPrompt suppresses the trailing assistant cue so the
// rendered text is suitable for offline storage rather than live
// generation.
type Config struct {
	Architecture       string
	Template           string
	NoGenerationPrompt bool
}

// Format applies a native model-family chat template.
//
//	text := chat.Format(messages, chat.Config{Architecture: "gemma4_text"})
func Format(messages []Message, cfg Config) string {
	template := templateName(cfg)
	switch template {
	case "gemma4":
		return formatGemma4(messages, cfg)
	case "gemma":
		return formatGemma(messages, cfg)
	case "qwen":
		return formatQwen(messages, cfg)
	case "llama":
		return formatLlama(messages, cfg)
	default:
		return formatPlain(messages, cfg)
	}
}

func formatGemma(messages []Message, cfg Config) string {
	builder := core.NewBuilder()
	for _, msg := range messages {
		role := normaliseRole(msg.Role)
		switch role {
		case "assistant":
			builder.WriteString("<start_of_turn>model\n" + msg.Content + "<end_of_turn>\n")
		case "system", "user":
			builder.WriteString("<start_of_turn>user\n" + msg.Content + "<end_of_turn>\n")
		}
	}
	if !cfg.NoGenerationPrompt {
		builder.WriteString("<start_of_turn>model\n")
	}
	return builder.String()
}

func formatGemma4(messages []Message, cfg Config) string {
	builder := core.NewBuilder()
	builder.WriteString("<bos>")
	for _, msg := range messages {
		role := normaliseRole(msg.Role)
		switch role {
		case "assistant":
			role = "model"
		case "system", "user":
		default:
			continue
		}
		builder.WriteString("<|turn>" + role + "\n" + core.Trim(msg.Content) + "<turn|>\n")
	}
	if !cfg.NoGenerationPrompt {
		builder.WriteString("<|turn>model\n")
	}
	return builder.String()
}

func formatQwen(messages []Message, cfg Config) string {
	builder := core.NewBuilder()
	for _, msg := range messages {
		role := normaliseRole(msg.Role)
		if role == "" {
			continue
		}
		builder.WriteString("<|im_start|>" + role + "\n" + msg.Content + "<|im_end|>\n")
	}
	if !cfg.NoGenerationPrompt {
		builder.WriteString("<|im_start|>assistant\n")
	}
	return builder.String()
}

func formatLlama(messages []Message, cfg Config) string {
	builder := core.NewBuilder()
	builder.WriteString("<|begin_of_text|>")
	for _, msg := range messages {
		role := normaliseRole(msg.Role)
		if role == "" {
			continue
		}
		builder.WriteString("<|start_header_id|>" + role + "<|end_header_id|>\n\n" + msg.Content + "<|eot_id|>")
	}
	if !cfg.NoGenerationPrompt {
		builder.WriteString("<|start_header_id|>assistant<|end_header_id|>\n\n")
	}
	return builder.String()
}

func formatPlain(messages []Message, cfg Config) string {
	builder := core.NewBuilder()
	for _, msg := range messages {
		if msg.Content == "" {
			continue
		}
		builder.WriteString(msg.Content + "\n")
	}
	if !cfg.NoGenerationPrompt {
		builder.WriteString("")
	}
	return builder.String()
}

// TemplateName returns the canonical template id selected by cfg. Used
// by callers that need to branch on template family before rendering.
//
//	switch chat.TemplateName(cfg) { case "gemma4": … }
func TemplateName(cfg Config) string {
	return templateName(cfg)
}

func templateName(cfg Config) string {
	template := core.Lower(core.Trim(cfg.Template))
	if template != "" {
		return template
	}
	switch core.Lower(core.Trim(cfg.Architecture)) {
	case "gemma4", "gemma4_text":
		return "gemma4"
	case "gemma", "gemma2", "gemma3", "gemma3_text":
		return "gemma"
	case "qwen", "qwen2", "qwen3", "qwen3_moe", "qwen3_next":
		return "qwen"
	case "llama", "llama3", "llama4":
		return "llama"
	default:
		return ""
	}
}

// NormaliseRole canonicalises chat role names across the HF / ShareGPT
// / Llama / Gemma variations. Empty input returns empty string.
//
//	role := chat.NormaliseRole("gpt") // → "assistant"
func NormaliseRole(role string) string {
	return normaliseRole(role)
}

func normaliseRole(role string) string {
	switch core.Lower(core.Trim(role)) {
	case "human", "user":
		return "user"
	case "gpt", "bot", "assistant", "model":
		return "assistant"
	case "system":
		return "system"
	default:
		return core.Lower(core.Trim(role))
	}
}

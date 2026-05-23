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
	// Gemma writes fixed "user" / "model" tags — role is not emitted
	// per-message, so the capacity calc skips role overhead.
	builder.Grow(chatFormatCapacity(messages, 34, 22, false))
	for _, msg := range messages {
		role := normaliseRole(msg.Role)
		switch role {
		case "assistant":
			builder.WriteString("<start_of_turn>model\n")
			builder.WriteString(msg.Content)
			builder.WriteString("<end_of_turn>\n")
		case "system", "user":
			builder.WriteString("<start_of_turn>user\n")
			builder.WriteString(msg.Content)
			builder.WriteString("<end_of_turn>\n")
		}
	}
	if !cfg.NoGenerationPrompt {
		builder.WriteString("<start_of_turn>model\n")
	}
	return builder.String()
}

func formatGemma4(messages []Message, cfg Config) string {
	builder := core.NewBuilder()
	builder.Grow(chatFormatCapacity(messages, 17, 37, true) + len("<bos>"))
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
		builder.WriteString("<|turn>")
		builder.WriteString(role)
		builder.WriteString("\n")
		builder.WriteString(core.Trim(msg.Content))
		builder.WriteString("<turn|>\n")
	}
	if !cfg.NoGenerationPrompt {
		builder.WriteString("<|turn>model\n")
		builder.WriteString("<|channel>thought\n<channel|>")
	}
	return builder.String()
}

func formatQwen(messages []Message, cfg Config) string {
	builder := core.NewBuilder()
	builder.Grow(chatFormatCapacity(messages, 24, 23, true))
	for _, msg := range messages {
		role := normaliseRole(msg.Role)
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

func formatLlama(messages []Message, cfg Config) string {
	builder := core.NewBuilder()
	builder.Grow(chatFormatCapacity(messages, 52, 43, true) + len("<|begin_of_text|>"))
	builder.WriteString("<|begin_of_text|>")
	for _, msg := range messages {
		role := normaliseRole(msg.Role)
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

func formatPlain(messages []Message, cfg Config) string {
	// Plain has no generation prompt suffix — the historic
	// builder.WriteString("") tail was a no-op that still cost
	// a function call + length branch per Format(). The cfg arg
	// is retained to keep the formatX signatures uniform.
	_ = cfg
	builder := core.NewBuilder()
	// Plain emits only the content + "\n" per message — no role.
	builder.Grow(chatFormatCapacity(messages, 1, 0, false))
	for _, msg := range messages {
		if msg.Content == "" {
			continue
		}
		builder.WriteString(msg.Content)
		builder.WriteString("\n")
	}
	return builder.String()
}

// maxNormalisedRoleLen is len("assistant") — the longest role string
// any template ever writes after normaliseRole expands aliases. Used
// in place of len(msg.Role) when sizing the Builder so aliased roles
// (gpt/bot/model → assistant) cannot under-allocate and trigger a
// silent realloc.
const maxNormalisedRoleLen = 9

func chatFormatCapacity(messages []Message, perMessageOverhead, generationPromptOverhead int, writesRole bool) int {
	// Templates that emit role per-message must reserve up to
	// maxNormalisedRoleLen — using len(msg.Role) would under-allocate
	// when normaliseRole expands aliases (gpt→assistant, etc) and
	// trigger a silent Builder realloc. Templates that don't emit
	// role skip the term entirely.
	roleOverhead := 0
	if writesRole {
		roleOverhead = maxNormalisedRoleLen
	}
	total := generationPromptOverhead
	for _, msg := range messages {
		total += len(msg.Content) + perMessageOverhead + roleOverhead
	}
	return total
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
	case "qwen", "qwen2", "qwen3", "qwen3_moe", "qwen3_next", "qwen3_6", "qwen3_6_moe":
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
	// Capture the canonicalised role once — the previous default
	// branch re-ran core.Lower(core.Trim(role)), doubling the work
	// for unknown roles (the common case once a wire handler passes
	// through any non-canonical custom role).
	r := core.Lower(core.Trim(role))
	switch r {
	case "human", "user":
		return "user"
	case "gpt", "bot", "assistant", "model":
		return "assistant"
	case "system":
		return "system"
	default:
		return r
	}
}

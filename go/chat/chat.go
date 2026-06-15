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
	"dappco.re/go/mlx/profile"
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
	EnableThinking     bool
	// LargeVariant marks a large Gemma 4 (12B/26B/31B, num_attention_heads>=16).
	// With thinking off, the shipped chat_template.jinja for those models appends
	// an empty <|channel>thought\n<channel|> after the model turn to suppress a
	// ghost thought channel; E2B/E4B do not. Ignored by other architectures.
	LargeVariant bool
	// Continuation renders messages as an append to a session whose retained
	// state ends inside an open model turn (generation stops on the
	// end-of-turn token without retaining it): the family template closes
	// that turn, skips the BOS/system opening, renders only the new turns,
	// and reopens the generation header. Session consumers prefill a normal
	// Format for turn one and a Continuation render for every later turn.
	Continuation bool
}

// Format applies a native model-family chat template.
//
//	text := chat.Format(messages, chat.Config{Architecture: "gemma4_text"})
//
// ConfigForArchitecture derives the chat-template config for a model
// architecture: the family default for thinking plus the large-variant
// gate (12B/26B/31B ghost-suppressor heads check).
//
//	cfg := chat.ConfigForArchitecture(info.Architecture, info.NumHeads)
func ConfigForArchitecture(architecture string, numHeads int) Config {
	return Config{
		Architecture:   architecture,
		EnableThinking: profile.DefaultThinkingEnabled(architecture),
		LargeVariant:   profile.IsGemma4LargeVariant(architecture, numHeads),
	}
}

func Format(messages []Message, cfg Config) string {
	if fn := formatters[templateName(cfg)]; fn != nil {
		return fn(messages, cfg)
	}
	// No family formatter registered for this template → plain text. Family
	// formatters live in their model packages (pkg/metal/model/{family}/chat)
	// and register themselves; plain is the neutral built-in fallback.
	return formatPlain(messages, cfg)
}

func formatPlain(messages []Message, cfg Config) string {
	// Plain has no generation prompt suffix — the historic
	// builder.WriteString("") tail was a no-op that still cost
	// a function call + length branch per Format(). The cfg arg
	// is retained to keep the formatX signatures uniform.
	_ = cfg
	builder := core.NewBuilder()
	// Plain emits only the content + "\n" per message — no role.
	builder.Grow(FormatCapacity(messages, 1, 0, false))
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

// FormatCapacity sizes a Builder for a chat template: the sum of message
// content plus per-message and generation-prompt overhead, reserving role
// width when the template emits a role per message. Family chat packages call
// it to Grow their Builder before writing.
//
//	b.Grow(chat.FormatCapacity(messages, 17, 13, true) + len("<bos>"))
func FormatCapacity(messages []Message, perMessageOverhead, generationPromptOverhead int, writesRole bool) int {
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

// templateName resolves the chat-template name for cfg: an explicit cfg.Template
// wins, otherwise the architecture's registry-advertised name
// (profile.ChatTemplateName). The name is metadata; whether a formatter exists
// for it is decided by the registry in Format — an unregistered name renders as
// plain text. The neutral chat package thus carries no family-name list.
func templateName(cfg Config) string {
	if template := core.Lower(core.Trim(cfg.Template)); template != "" {
		return template
	}
	return profile.ChatTemplateName(cfg.Architecture)
}

// NormaliseRole canonicalises chat role names across the HF / ShareGPT
// / Llama / Gemma variations. Empty input returns empty string.
//
//	role := chat.NormaliseRole("gpt") // → "assistant"
func NormaliseRole(role string) string {
	return normaliseRole(role)
}

func normaliseRole(role string) string {
	// Canonical fast path. The common Format flow (bench, every wire
	// handler that built its messages with the canonical role names)
	// hits this — no Lower/Trim/switch table walk needed, and the
	// branch is small enough to inline into the caller.
	switch role {
	case "user", "assistant", "system":
		return role
	}
	return normaliseRoleSlow(role)
}

func normaliseRoleSlow(role string) string {
	// Trim is alloc-free (it returns a sub-slice of role). Match the known
	// aliases case-insensitively on the trimmed form via EqualFold — every
	// known-alias branch returns a compile-time literal, so lowering the
	// input first (core.Lower → strings.ToLower) would allocate a string
	// purely to drive the switch and then discard it. Only the unknown-role
	// fallthrough actually returns the canonicalised input, so that is the
	// one branch that pays for core.Lower; an already-lowercase unknown role
	// (the common custom-role case) stays alloc-free because Lower no-ops it.
	trimmed := core.Trim(role)
	switch {
	case core.EqualFold(trimmed, "human"), core.EqualFold(trimmed, "user"):
		return "user"
	case core.EqualFold(trimmed, "gpt"), core.EqualFold(trimmed, "bot"),
		core.EqualFold(trimmed, "assistant"), core.EqualFold(trimmed, "model"):
		return "assistant"
	case core.EqualFold(trimmed, "system"), core.EqualFold(trimmed, "developer"):
		return "system"
	default:
		return core.Lower(trimmed)
	}
}

// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

// Chat-prompt formatting for the served model families. The engine holds no
// per-family template logic: every family renders through the shared,
// jinja-faithful chat.Format (SPOR — one builder for training + serve), which
// dispatches on the architecture's registered chat template. The engine only
// builds the chat.Config (thinking + large-variant) and chunks the output.

import (
	"iter"

	"dappco.re/go/mlx/chat"
	"dappco.re/go/mlx/profile"
)

// resolveThinkingEnabled resolves the Gemma 4 reasoning toggle: an explicit
// per-call EnableThinking wins, otherwise the architecture's registry default
// (profile.DefaultThinkingEnabled) — the single home for the thinking default,
// shared with the mlx serve adapter so the two render the same prompt.
func resolveThinkingEnabled(architecture string, cfg []GenerateConfig) bool {
	if len(cfg) == 0 || cfg[0].EnableThinking == nil {
		return profile.DefaultThinkingEnabled(architecture)
	}
	return *cfg[0].EnableThinking
}

// needsThoughtChannelSuppressor reports whether the loaded model declares it
// needs the empty thought-channel suppressor when reasoning is off
// (ThoughtChannelSuppressorModel — the large Gemma 4 variants). nil-safe for
// bare/unloaded Models; false for models that do not declare the capability.
func (m *Model) needsThoughtChannelSuppressor() bool {
	if m == nil || m.model == nil {
		return false
	}
	if suppressor, ok := m.model.(ThoughtChannelSuppressorModel); ok {
		return suppressor.NeedsThoughtChannelSuppressor()
	}
	return false
}

// toChatMessages converts metal chat turns to the shared chat package's type so
// all prompt building flows through the single jinja-faithful builder
// (chat.Format) — no reroll between training (dataset) and serve (SPOR).
func toChatMessages(messages []ChatMessage) []chat.Message {
	out := make([]chat.Message, len(messages))
	for i, msg := range messages {
		out[i] = chat.Message{Role: msg.Role, Content: msg.Content}
	}
	return out
}

// chatConfig builds the shared chat.Config for the loaded model. EnableThinking
// and LargeVariant are Gemma-4 controls; chat.Format ignores them for other
// architectures.
func (m *Model) chatConfig(cfg []GenerateConfig) chat.Config {
	return chat.Config{
		Architecture:   m.modelType,
		EnableThinking: resolveThinkingEnabled(m.modelType, cfg),
		LargeVariant:   m.needsThoughtChannelSuppressor(),
	}
}

// formatChat applies the model's native chat template via the shared builder.
func (m *Model) formatChat(messages []ChatMessage, cfg ...GenerateConfig) string {
	return chat.Format(toChatMessages(messages), m.chatConfig(cfg))
}

// formatChatChunks streams formatChat's output in chunkBytes-sized pieces; their
// concatenation equals the non-chunked prompt.
func (m *Model) formatChatChunks(messages []ChatMessage, chunkBytes int, cfg ...GenerateConfig) iter.Seq[string] {
	prompt := m.formatChat(messages, cfg...)
	return func(yield func(string) bool) {
		if chunkBytes <= 0 {
			yield(prompt)
			return
		}
		for i := 0; i < len(prompt); i += chunkBytes {
			end := min(i+chunkBytes, len(prompt))
			if !yield(prompt[i:end]) {
				return
			}
		}
	}
}

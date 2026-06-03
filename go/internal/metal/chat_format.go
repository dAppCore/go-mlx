// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

// Chat-prompt formatting for the served model families. Split out of generate.go
// (the de-rot, go-mlx #45). Gemma 4 delegates to the shared, jinja-faithful
// chat.Format (SPOR — one builder for training + serve); gemma/qwen/llama still
// have local builders pending the same Format(type,data) collapse.

import (
	"iter"

	"dappco.re/go"
	"dappco.re/go/mlx/chat"
)

// gemma4ThinkingEnabled resolves the Gemma 4 reasoning toggle from the optional
// per-call config: absent or nil EnableThinking means model default (on).
func gemma4ThinkingEnabled(cfg []GenerateConfig) bool {
	if len(cfg) == 0 || cfg[0].EnableThinking == nil {
		return true
	}
	return *cfg[0].EnableThinking
}

// gemma4LargeVariant reports whether the loaded model is a large Gemma 4
// (26B/31B, num_attention_heads>=16) that ghosts an empty thought channel when
// thinking is off and so needs the suppressor. nil-safe for bare/unloaded Models.
func (m *Model) gemma4LargeVariant() bool {
	if m == nil || m.model == nil {
		return false
	}
	return m.Info().NumHeads >= 16
}

// formatChat applies the model's native chat template.
func (m *Model) formatChat(messages []ChatMessage, cfg ...GenerateConfig) string {
	switch m.modelType {
	case "gemma4", "gemma4_text":
		return formatGemma4Chat(messages, gemma4ThinkingEnabled(cfg), m.gemma4LargeVariant())
	case "gemma2", "gemma3", "gemma3_text":
		return formatGemmaChat(messages)
	case "qwen2", "qwen3":
		return formatQwenChat(messages)
	case "llama":
		return formatLlamaChat(messages)
	default:
		builder := core.NewBuilder()
		for _, msg := range messages {
			builder.WriteString(msg.Content + "\n")
		}
		return builder.String()
	}
}

func (m *Model) formatChatChunks(messages []ChatMessage, chunkBytes int, cfg ...GenerateConfig) iter.Seq[string] {
	return func(yield func(string) bool) {
		switch m.modelType {
		case "gemma4", "gemma4_text":
			formatGemma4ChatChunks(messages, chunkBytes, gemma4ThinkingEnabled(cfg), m.gemma4LargeVariant(), yield)
		case "gemma2", "gemma3", "gemma3_text":
			formatGemmaChatChunks(messages, chunkBytes, yield)
		case "qwen2", "qwen3":
			formatQwenChatChunks(messages, chunkBytes, yield)
		case "llama":
			formatLlamaChatChunks(messages, chunkBytes, yield)
		default:
			for _, msg := range messages {
				if !yieldChatTextChunks(yield, msg.Content+"\n", chunkBytes) {
					return
				}
			}
		}
	}
}

func yieldChatTextChunks(yield func(string) bool, text string, chunkBytes int) bool {
	if text == "" {
		return true
	}
	if chunkBytes <= 0 || len(text) <= chunkBytes {
		return yield(text)
	}
	start := 0
	for index := range text {
		if index == start || index-start < chunkBytes {
			continue
		}
		if !yield(text[start:index]) {
			return false
		}
		start = index
	}
	if start < len(text) {
		return yield(text[start:])
	}
	return true
}

// toChatMessages converts metal chat turns to the shared chat package's type so
// all Gemma 4 prompt building flows through the single jinja-faithful builder
// (chat.Format) — no reroll between training (dataset) and serve (SPOR).
func toChatMessages(messages []ChatMessage) []chat.Message {
	out := make([]chat.Message, len(messages))
	for i, msg := range messages {
		out[i] = chat.Message{Role: msg.Role, Content: msg.Content}
	}
	return out
}

// formatGemma4Chat delegates to the shared chat.Format — the single
// jinja-faithful Gemma 4 builder. enableThinking toggles reasoning (<|think|>\n
// in the system turn); largeVariant (26B/31B, heads>=16) adds the off-mode
// <|channel>thought\n<channel|> ghost suppressor. See go/chat for the template.
func formatGemma4Chat(messages []ChatMessage, enableThinking, largeVariant bool) string {
	return chat.Format(toChatMessages(messages), chat.Config{
		Architecture:   "gemma4_text",
		EnableThinking: enableThinking,
		LargeVariant:   largeVariant,
	})
}

// formatGemma4ChatChunks streams formatGemma4Chat's output in chunkBytes-sized
// pieces; their concatenation equals the non-chunked prompt.
func formatGemma4ChatChunks(messages []ChatMessage, chunkBytes int, enableThinking, largeVariant bool, yield func(string) bool) {
	prompt := formatGemma4Chat(messages, enableThinking, largeVariant)
	if chunkBytes <= 0 {
		yield(prompt)
		return
	}
	for i := 0; i < len(prompt); i += chunkBytes {
		end := i + chunkBytes
		if end > len(prompt) {
			end = len(prompt)
		}
		if !yield(prompt[i:end]) {
			return
		}
	}
}

func formatGemmaChat(messages []ChatMessage) string {
	builder := core.NewBuilder()
	builder.WriteString("<bos>")
	firstUserPrefix := ""
	start := 0
	if len(messages) > 0 && core.Lower(core.Trim(messages[0].Role)) == "system" {
		firstUserPrefix = core.Trim(messages[0].Content)
		start = 1
	}
	for _, msg := range messages[start:] {
		switch core.Lower(core.Trim(msg.Role)) {
		case "system", "user", "human":
			builder.WriteString("<start_of_turn>user\n")
			if firstUserPrefix != "" {
				builder.WriteString(firstUserPrefix)
				builder.WriteString("\n\n")
				firstUserPrefix = ""
			}
			builder.WriteString(core.Trim(msg.Content))
			builder.WriteString("<end_of_turn>\n")
		case "assistant", "model":
			builder.WriteString("<start_of_turn>model\n")
			builder.WriteString(core.Trim(msg.Content))
			builder.WriteString("<end_of_turn>\n")
		}
	}
	builder.WriteString("<start_of_turn>model\n")
	return builder.String()
}

func formatGemmaChatChunks(messages []ChatMessage, chunkBytes int, yield func(string) bool) {
	if !yield("<bos>") {
		return
	}
	firstUserPrefix := ""
	start := 0
	if len(messages) > 0 && core.Lower(core.Trim(messages[0].Role)) == "system" {
		firstUserPrefix = core.Trim(messages[0].Content)
		start = 1
	}
	for _, msg := range messages[start:] {
		switch core.Lower(core.Trim(msg.Role)) {
		case "system", "user", "human":
			if !yield("<start_of_turn>user\n") {
				return
			}
			if firstUserPrefix != "" {
				if !yieldChatTextChunks(yield, firstUserPrefix, chunkBytes) || !yield("\n\n") {
					return
				}
				firstUserPrefix = ""
			}
			if !yieldChatTextChunks(yield, core.Trim(msg.Content), chunkBytes) || !yield("<end_of_turn>\n") {
				return
			}
		case "assistant", "model":
			if !yield("<start_of_turn>model\n") || !yieldChatTextChunks(yield, core.Trim(msg.Content), chunkBytes) || !yield("<end_of_turn>\n") {
				return
			}
		}
	}
	yield("<start_of_turn>model\n")
}

func formatQwenChat(messages []ChatMessage) string {
	builder := core.NewBuilder()
	for _, msg := range messages {
		builder.WriteString("<|im_start|>" + msg.Role + "\n" + msg.Content + "<|im_end|>\n")
	}
	builder.WriteString("<|im_start|>assistant\n")
	return builder.String()
}

func formatQwenChatChunks(messages []ChatMessage, chunkBytes int, yield func(string) bool) {
	for _, msg := range messages {
		if !yield("<|im_start|>"+msg.Role+"\n") || !yieldChatTextChunks(yield, msg.Content, chunkBytes) || !yield("<|im_end|>\n") {
			return
		}
	}
	yield("<|im_start|>assistant\n")
}

func formatLlamaChat(messages []ChatMessage) string {
	builder := core.NewBuilder()
	builder.WriteString("<|begin_of_text|>")
	for _, msg := range messages {
		builder.WriteString("<|start_header_id|>" + msg.Role + "<|end_header_id|>\n\n" + msg.Content + "<|eot_id|>")
	}
	builder.WriteString("<|start_header_id|>assistant<|end_header_id|>\n\n")
	return builder.String()
}

func formatLlamaChatChunks(messages []ChatMessage, chunkBytes int, yield func(string) bool) {
	if !yield("<|begin_of_text|>") {
		return
	}
	for _, msg := range messages {
		if !yield("<|start_header_id|>"+msg.Role+"<|end_header_id|>\n\n") || !yieldChatTextChunks(yield, msg.Content, chunkBytes) || !yield("<|eot_id|>") {
			return
		}
	}
	yield("<|start_header_id|>assistant<|end_header_id|>\n\n")
}

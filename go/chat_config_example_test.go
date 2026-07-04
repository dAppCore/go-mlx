// SPDX-Licence-Identifier: EUPL-1.2

// Usage-in-situ for chat_config.go — the session-prefill formatters a
// continuity consumer calls. FormatChatPrompt renders turn one (with the
// generation header); FormatChatContinuation appends later turns to a
// session whose retained state ends inside an open model turn, so it
// closes that turn before reopening the header. Both route through the
// private formatChatTurns with the model's per-family chat config.

package mlx

import (
	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/mlx/pkg/metal"
)

// FormatChatPrompt renders a conversation opening in the model's family
// template — here gemma4_text — exactly as Chat prefills it internally.
func ExampleModel_FormatChatPrompt() {
	model := &Model{model: &fakeNativeModel{info: metal.ModelInfo{Architecture: "gemma4_text"}}}

	prompt := model.FormatChatPrompt([]inference.Message{{Role: "user", Content: "hi"}})

	core.Println(core.Sprintf("%q", prompt))
	// Output: "<bos><|turn>system\n<|think|>\n<turn|>\n<|turn>user\nhi<turn|>\n<|turn>model\n"
}

// FormatChatContinuation closes the open model turn left by the retained
// session state, renders only the new turns, and reopens the generation
// header — so it never re-emits the <bos> opening that FormatChatPrompt does.
func ExampleModel_FormatChatContinuation() {
	model := &Model{model: &fakeNativeModel{info: metal.ModelInfo{Architecture: "gemma4_text"}}}

	continuation := model.FormatChatContinuation([]inference.Message{{Role: "user", Content: "hi"}})

	core.Println(core.Sprintf("%q", continuation))
	// Output: "<turn|>\n<|turn>user\nhi<turn|>\n<|turn>model\n"
}

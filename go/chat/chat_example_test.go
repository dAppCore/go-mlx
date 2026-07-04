// SPDX-Licence-Identifier: EUPL-1.2

package chat

import core "dappco.re/go"

// ExampleConfigForArchitecture derives the chat-template config for a model
// architecture: the family thinking default plus the large-variant gate.
func ExampleConfigForArchitecture() {
	cfg := ConfigForArchitecture("gemma4_text", 16)
	core.Println(cfg.Architecture)
	core.Println(cfg.EnableThinking)
	core.Println(cfg.LargeVariant)
	// Output:
	// gemma4_text
	// true
	// true
}

// ExampleFormat renders a message list with the plain template — content per
// message, no role, with the generation prompt suppressed for offline storage.
func ExampleFormat() {
	text := Format([]Message{
		{Role: "user", Content: "hello"},
		{Role: "assistant", Content: "hi there"},
	}, Config{Template: "plain", NoGenerationPrompt: true})
	core.Println(core.Sprintf("%q", text))
	// Output:
	// "hello\nhi there\n"
}

// ExampleFormatCapacity sizes a Builder for a chat template: the sum of content
// length plus per-message and generation-prompt overhead, reserving role width
// when the template emits a role per message.
func ExampleFormatCapacity() {
	messages := []Message{
		{Role: "user", Content: "abcde"},   // content len 5
		{Role: "assistant", Content: "xy"}, // content len 2
	}
	core.Println(FormatCapacity(messages, 3, 7, false))
	// Output:
	// 20
}

// ExampleTemplateName resolves the canonical template id for a config: an
// explicit Template wins, otherwise the architecture's advertised family name.
func ExampleTemplateName() {
	core.Println(TemplateName(Config{Architecture: "Gemma4ForConditionalGeneration"}))
	core.Println(TemplateName(Config{Architecture: "gemma3", Template: "qwen"}))
	// Output:
	// gemma4
	// qwen
}

// ExampleNormaliseRole canonicalises chat role names across the HF / ShareGPT /
// Llama / Gemma variations.
func ExampleNormaliseRole() {
	core.Println(NormaliseRole("gpt"))
	core.Println(NormaliseRole("developer"))
	// Output:
	// assistant
	// system
}

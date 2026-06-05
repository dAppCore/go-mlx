// SPDX-Licence-Identifier: EUPL-1.2

package chat

import core "dappco.re/go"

func ExampleFormat() {
	rendered := Format(
		[]Message{{Role: "user", Content: " hi "}},
		Config{Architecture: "gemma4_text", LargeVariant: true},
	)
	core.Println(rendered)
	// Output:
	// <bos><|turn>user
	// hi<turn|>
	// <|turn>model
	// <|channel>thought
	// <channel|>
}

func ExampleTemplateName() {
	core.Println(TemplateName(Config{Architecture: "Gemma4ForConditionalGeneration"}))
	core.Println(TemplateName(Config{Architecture: "gemma3", Template: "qwen"}))
	// Output:
	// gemma4
	// qwen
}

func ExampleNormaliseRole() {
	core.Println(NormaliseRole("gpt"))
	core.Println(NormaliseRole("developer"))
	// Output:
	// assistant
	// system
}

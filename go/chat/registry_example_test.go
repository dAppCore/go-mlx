// SPDX-Licence-Identifier: EUPL-1.2

package chat

import core "dappco.re/go"

// ExampleRegisterFormatter binds a chat-template name to its formatter. Family
// chat packages call this from init() so a blank import wires the family in;
// Format then dispatches to it by the resolved template name.
func ExampleRegisterFormatter() {
	RegisterFormatter("example-fmt", func(messages []Message, _ Config) string {
		return "rendered:" + messages[0].Content
	})
	core.Println(Format([]Message{{Role: "user", Content: "hi"}}, Config{Template: "example-fmt"}))
	// Output:
	// rendered:hi
}

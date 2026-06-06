// SPDX-Licence-Identifier: EUPL-1.2

package chat

// Formatter renders a message list into a model-family chat prompt. A family's
// own chat package (e.g. pkg/metal/model/gemma4/chat) registers its formatter
// under the template name it serves, so the neutral chat package dispatches by
// name and never carries family-specific prompt logic.
type Formatter func(messages []Message, cfg Config) string

// formatters maps a chat-template name (the value profile.ChatTemplateName
// advertises, e.g. "gemma4") to the formatter that renders it. Populated from
// family chat packages' init(); read by Format.
var formatters = map[string]Formatter{}

// RegisterFormatter binds a chat-template name to its formatter. Family chat
// packages call this from init(); a blank import of the package wires it in.
// Re-registering a name overwrites it (idempotent for the same function).
//
//	func init() { chat.RegisterFormatter("gemma4", Format) }
func RegisterFormatter(name string, fn Formatter) {
	formatters[name] = fn
}

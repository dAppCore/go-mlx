// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"dappco.re/go/inference"
	"dappco.re/go/mlx/chat"
	"dappco.re/go/mlx/dataset"
)

// DatasetConfigForModel returns the JSONL chat-template config that matches
// the loaded model metadata.
func DatasetConfigForModel(info ModelInfo) dataset.Config {
	return dataset.Config{ChatTemplate: modelChatConfig(info)}
}

func modelChatConfig(info ModelInfo) chat.Config {
	return modelChatConfigForArchitecture(info.Architecture, info.NumHeads)
}

func modelChatConfigForArchitecture(architecture string, numHeads int) chat.Config {
	return chat.ConfigForArchitecture(architecture, numHeads)
}

// FormatChatPrompt renders a conversation opening in the model's chat
// template, including the generation header — the same text Chat prefills
// internally. Session consumers (serve continuity, the state CLI) prefill
// this for turn one.
//
//	sess.Prefill(m.FormatChatPrompt(messages))
func (m *Model) FormatChatPrompt(messages []inference.Message) string {
	return m.formatChatTurns(messages, nil, false)
}

// formatChatTurns renders messages with the model's chat config, honouring a
// request-level thinking override (nil = model default) and the continuation
// form. The conversation-continuity manager formats every turn through this.
func (m *Model) formatChatTurns(messages []inference.Message, thinking *bool, continuation bool) string {
	cfg := modelChatConfig(m.Info())
	if thinking != nil {
		cfg.EnableThinking = *thinking
	}
	cfg.Continuation = continuation
	return chat.Format(messages, cfg)
}

// FormatChatContinuation renders messages as an append to a session whose
// retained state ends inside an open model turn: the family template closes
// that turn, renders only the new turns, and reopens the generation header.
// Session consumers append this for every turn after the first.
//
//	sess.AppendPrompt(m.FormatChatContinuation(newTurns))
func (m *Model) FormatChatContinuation(messages []inference.Message) string {
	return m.formatChatTurns(messages, nil, true)
}

// FormatChatPromptThinking is FormatChatPrompt with an explicit thinking
// override (nil = model default) — the state CLI wires -think through here
// so a small token budget is not consumed inside the thought channel by a
// template that defaults thinking on.
//
//	off := false
//	sess.Prefill(m.FormatChatPromptThinking(messages, &off))
func (m *Model) FormatChatPromptThinking(messages []inference.Message, thinking *bool) string {
	return m.formatChatTurns(messages, thinking, false)
}

// FormatChatContinuationThinking is FormatChatContinuation with an explicit
// thinking override (nil = model default).
//
//	off := false
//	sess.AppendPrompt(m.FormatChatContinuationThinking(newTurns, &off))
func (m *Model) FormatChatContinuationThinking(messages []inference.Message, thinking *bool) string {
	return m.formatChatTurns(messages, thinking, true)
}

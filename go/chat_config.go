// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"dappco.re/go/inference"
	"dappco.re/go/mlx/chat"
	"dappco.re/go/mlx/dataset"
	"dappco.re/go/mlx/profile"
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
	return chat.Config{
		Architecture:   architecture,
		EnableThinking: profile.DefaultThinkingEnabled(architecture),
		LargeVariant:   profile.IsGemma4LargeVariant(architecture, numHeads),
	}
}

func sftEvalPromptForModel(prompt string, info ModelInfo) string {
	if !isGemma4ModelArchitecture(info.Architecture) {
		return prompt
	}
	return chat.Format([]chat.Message{{Role: "user", Content: prompt}}, modelChatConfig(info))
}

// FormatChatPrompt renders a conversation opening in the model's chat
// template, including the generation header — the same text Chat prefills
// internally. Session consumers (serve continuity, the state CLI) prefill
// this for turn one.
//
//	sess.Prefill(m.FormatChatPrompt(messages))
func (m *Model) FormatChatPrompt(messages []inference.Message) string {
	return chat.Format(messages, modelChatConfig(m.Info()))
}

// FormatChatContinuation renders messages as an append to a session whose
// retained state ends inside an open model turn: the family template closes
// that turn, renders only the new turns, and reopens the generation header.
// Session consumers append this for every turn after the first.
//
//	sess.AppendPrompt(m.FormatChatContinuation(newTurns))
func (m *Model) FormatChatContinuation(messages []inference.Message) string {
	cfg := modelChatConfig(m.Info())
	cfg.Continuation = true
	return chat.Format(messages, cfg)
}

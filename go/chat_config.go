// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
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

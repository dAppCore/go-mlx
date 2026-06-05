// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
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
	return chat.Config{
		Architecture: architecture,
		LargeVariant: sftGemma4Architecture(architecture) && numHeads >= 16,
	}
}

func sftEvalPromptForModel(prompt string, info ModelInfo) string {
	if !sftGemma4Architecture(info.Architecture) {
		return prompt
	}
	return chat.Format([]chat.Message{{Role: "user", Content: prompt}}, modelChatConfig(info))
}

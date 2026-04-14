//go:build darwin && arm64 && !nomlx

package mlx

import (
	"dappco.re/go/core/inference"
	"dappco.re/go/core/mlx/internal/metal"
)

func inferenceGenerateConfigToMetal(cfg inference.GenerateConfig) metal.GenerateConfig {
	return metal.GenerateConfig{
		MaxTokens:     cfg.MaxTokens,
		Temperature:   cfg.Temperature,
		TopK:          cfg.TopK,
		TopP:          cfg.TopP,
		StopTokens:    cfg.StopTokens,
		RepeatPenalty: cfg.RepeatPenalty,
	}
}

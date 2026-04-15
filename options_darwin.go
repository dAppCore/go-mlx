//go:build darwin && arm64 && !nomlx

package mlx

import (
	"reflect"

	"dappco.re/go/core/inference"
	"dappco.re/go/mlx/internal/metal"
)

func inferenceGenerateConfigToMetal(cfg inference.GenerateConfig) metal.GenerateConfig {
	out := metal.GenerateConfig{
		MaxTokens:     cfg.MaxTokens,
		Temperature:   cfg.Temperature,
		TopK:          cfg.TopK,
		TopP:          cfg.TopP,
		StopTokens:    cfg.StopTokens,
		RepeatPenalty: cfg.RepeatPenalty,
	}
	// Keep go-mlx forward-compatible with inference.GenerateConfig versions that
	// expose MinP without requiring a synchronized dependency update here.
	if field := reflect.ValueOf(cfg).FieldByName("MinP"); field.IsValid() {
		switch field.Kind() {
		case reflect.Float32, reflect.Float64:
			out.MinP = float32(field.Float())
		}
	}
	return out
}

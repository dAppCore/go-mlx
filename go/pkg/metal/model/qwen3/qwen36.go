// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package qwen3

import (
	"dappco.re/go/mlx/pkg/metal"
	"dappco.re/go/mlx/profile"
)

// Qwen 3.6 family detection — the model owns it. These were defined in package
// metal but only ever consumed here; the engine names no family.

// isQwen36Architecture reports whether an architecture string names the
// Qwen3.5/3.6 hybrid-attention family, resolved through the shared registry.
func isQwen36Architecture(value string) bool {
	switch profile.ArchitectureID(value) {
	case "qwen3_6", "qwen3_6_moe":
		return true
	default:
		return false
	}
}

// isQwen36HybridConfig reports whether a dense config is a Qwen 3.6 hybrid
// (linear-attention / partial-rotary) variant the native Go loader does not
// implement yet.
func isQwen36HybridConfig(cfg *metal.DenseConfig) bool {
	if cfg == nil {
		return false
	}
	switch metal.NormalizeProbeModelType(cfg.ModelType) {
	case "qwen3_6", "qwen3_6_moe":
		return true
	}
	for _, layerType := range cfg.LayerTypes {
		if metal.NormalizeDenseLayerType(layerType) == "linear_attention" {
			return true
		}
	}
	return cfg.PartialRotaryFactor > 0 && cfg.PartialRotaryFactor < 1
}

// qwen36NativeGuardMessage keeps the staged Qwen 3.6 diagnostic consistent
// across the dense and MoE loaders.
func qwen36NativeGuardMessage(modelType string) string {
	if metal.NormalizeProbeModelType(modelType) == "qwen3_6_moe" {
		return "qwen3_6_moe hybrid linear attention and sparse expert routing are not implemented in the native Go loader yet"
	}
	return "qwen3_6 hybrid linear attention is not implemented in the native Go loader yet"
}

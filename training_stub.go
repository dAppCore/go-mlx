//go:build !(darwin && arm64) || nomlx

package mlx

// LoRAAdapter is a stub on unsupported builds.
type LoRAAdapter struct{}

// LoRAConfig is a stub on unsupported builds.
type LoRAConfig struct{}

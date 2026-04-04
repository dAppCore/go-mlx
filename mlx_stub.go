//go:build !(darwin && arm64)

// Package mlx provides Go bindings for Apple's MLX framework via mlx-c.
package mlx

// MetalAvailable reports whether Metal GPU is available.
//
//	mlx.MetalAvailable() // → false on non-Apple Silicon
func MetalAvailable() bool { return false }

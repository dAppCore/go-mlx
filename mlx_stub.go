// SPDX-Licence-Identifier: EUPL-1.2

//go:build !(darwin && arm64) || nomlx

// Package mlx provides Go bindings for Apple's MLX framework via mlx-c.
package mlx

// MetalAvailable reports whether Metal GPU is available.
//
//	mlx.MetalAvailable() // → false on non-Apple Silicon
func MetalAvailable() bool { return false }

// Available reports whether native MLX support is available in this build.
func Available() bool { return MetalAvailable() }

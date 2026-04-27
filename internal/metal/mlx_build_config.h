// mlx_build_config.h — Shared build configuration for MLX source compilation
#pragma once
#define ACCELERATE_NEW_LAPACK 1
#define FMT_HEADER_ONLY 1
#define MLX_BUILD_GGUF 1
#define MLX_USE_ACCELERATE 1
#define MLX_VERSION "0.30.1"

// METAL_PATH is not used when building via CGo. The device.cpp copy in
// this package resolves the metallib path at runtime using __FILE__.
// This fallback is kept for non-CGo builds.
#ifndef METAL_PATH
#define METAL_PATH "mlx.metallib"
#endif

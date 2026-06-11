// mlx_build_config.h — Shared build configuration for MLX source compilation
#pragma once

// MLX_SOURCE_REV: d02cc10b
// ^ The lib/mlx commit this build compiled against. The forwarding shims
// (mlx_mlx_*.cpp) #include lib/mlx sources, but cgo hashes only the shim
// files — NOT their include targets — so a lib/mlx checkout switch reuses
// stale cached objects (mixed-version binaries; undefined or wrong-thread
// symbols). This header is force-included into every TU via -include, so
// bumping the rev here busts the cache for the whole package. Update it
// whenever lib/mlx moves (cmake configure regenerates dist; this is the
// cgo-side counterpart).
#define ACCELERATE_NEW_LAPACK 1
#define FMT_HEADER_ONLY 1
#define MLX_BUILD_GGUF 1
#ifndef MLX_ENABLE_DISTRIBUTED
#define MLX_ENABLE_DISTRIBUTED 1
#endif
#define MLX_USE_ACCELERATE 1
#define MLX_VERSION "0.31.2"

#ifdef __cplusplus
#include <exception>
#if __cplusplus < 202302L
#error "go-mlx native bridge requires C++23 or newer"
#endif
#endif

// METAL_PATH is not used when building via CGo. The device.cpp copy in
// this package resolves the metallib path at runtime using __FILE__.
// This fallback is kept for non-CGo builds.
#ifndef METAL_PATH
#define METAL_PATH "mlx.metallib"
#endif

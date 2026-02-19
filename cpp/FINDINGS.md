# FINDINGS.md — go-mlx C++ Research & Discovery

Record findings about mlx-c internals, API quirks, and architectural decisions.

---

## 2026-02-19: Initial Setup (GoLand Claude)

### mlx-c v0.4.1 Source Stats
- **24 .cpp files**, **27 .h headers** in `build/_deps/mlx-c-src/mlx/c/`
- Pure C API wrapping C++ MLX framework
- Each header typically has: type definition, `_new()`, `_free()`, operation functions

### Installed Artifacts (dist/)
- `dist/lib/libmlxc.dylib` — C API shared library
- `dist/lib/libmlx.dylib` — MLX C++ framework shared library
- `dist/include/mlx/c/` — 27 public headers

### Build Verified
- CMake configure + build + install completes cleanly
- Go tests pass (29/29) after build
- AppleClang 17.0.0, macOS SDK 26.2

### Go Binding Coverage (preliminary)
The Go side currently uses functions from these headers:
- `array.h` — creation, data access, shape, dtype
- `ops.h` — add, multiply, divide, matmul, softmax, etc.
- `fast.h` — rms_norm, layer_norm, rope, scaled_dot_product_attention
- `transforms.h` — eval, compile, vjp, jvp
- `io.h` — safetensors load/save
- `random.h` — categorical
- `stream.h` — default_gpu_stream
- `metal.h` — is_available
- `error.h` — set_error_handler
- `vector.h` — vector_array operations
- `closure.h` — compiled function closures

**Likely unused**: `fft.h`, `linalg.h`, `distributed.h`, `distributed_group.h`, `export.h`, `map.h`, `memory.h`

---

*Add new findings below as work progresses.*

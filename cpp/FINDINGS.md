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

## 2026-02-19: Full API Research (CLion Claude — Session 1)

### CRITICAL: `mlx_contiguous` EXISTS — Fixes Floats() Bug

**Location**: `ops.h:220` / `ops.cpp:690`

```c
int mlx_contiguous(
    mlx_array* res,
    const mlx_array a,
    bool allow_col_major,  // false = force row-major
    const mlx_stream s);
```

Wraps `mlx::core::contiguous()`. This is the correct fix for the Floats()/DataInt32() non-contiguous bug.

**GoLand Claude action required**: Bind `mlx_contiguous` and call it before `mlx_array_data_*` when the array is non-contiguous. Use `allow_col_major = false` to guarantee row-major layout.

Additionally, there's a **contiguity check** function:

```c
// Internal but available — checks flags().contiguous
int _mlx_array_is_contiguous(bool* res, const mlx_array arr);
int _mlx_array_is_row_contiguous(bool* res, const mlx_array arr);
int _mlx_array_is_col_contiguous(bool* res, const mlx_array arr);
```

**Recommended pattern** for Go `Floats()`:
1. Call `_mlx_array_is_row_contiguous()` to check
2. If not row-contiguous, call `mlx_contiguous(res, arr, false, stream)` to get a contiguous copy
3. Then read via `mlx_array_data_float32()`

Also available but less ideal:
- `mlx_copy(res, a, s)` — copies array but may preserve non-contiguous layout
- `mlx_flatten(res, a, start_axis, end_axis, s)` — flattens dimensions, forces contiguous
- `mlx_reshape(res, a, shape, shape_num, s)` — Go's current workaround, works but semantically wrong

### CRITICAL: `mlx_cumsum` EXISTS — Unblocks TopP Sampling

**Location**: `ops.h:344`

```c
int mlx_cumsum(
    mlx_array* res,
    const mlx_array a,
    int axis,
    bool reverse,    // false = forward cumulative sum
    bool inclusive,   // true = include current element
    const mlx_stream s);
```

**GoLand Claude action required**: Bind `mlx_cumsum` and implement proper TopP (nucleus) sampling. For standard TopP: `axis=-1, reverse=false, inclusive=true`.

Related cumulative ops also available: `mlx_cumprod`, `mlx_cummax`, `mlx_cummin`, `mlx_logcumsumexp`.

### `mlx_array_data_*` Does NOT Auto-Evaluate

**Source**: `array.cpp:536` calls C++ `mlx_array_get_(arr).data<float>()`, which (`array.h:372`) just does pointer arithmetic into the raw buffer — no implicit evaluation.

The header comment says "Array must be evaluated, otherwise returns NULL" but this is misleading. The C++ `data<T>()` accesses `buffer().raw_ptr()` which will crash or return garbage if the buffer hasn't been allocated yet (i.e., the array is unscheduled).

**Contrast with `mlx_array_item_*`**: These call C++ `item<T>()` which **does** trigger evaluation internally (`array.h:564-569`).

**GoLand Claude action**: The current `Materialise()` call before data access is correct and essential. Never skip it. Consider adding an assertion using `_mlx_array_is_available()` as a safety check.

### Error Model — Free-form Strings Only

**Source**: `error.h` + `error.cpp`

The error handler signature is:
```c
typedef void (*mlx_error_handler_func)(const char* msg, void* data);
```

The message format is: `"<exception_message> at <file>:<line>"` (hardcoded in `_mlx_error`, `error.cpp:37-49`).

- **No error codes** — just a free-form string
- **No categories** — the string is whatever `e.what()` produces from C++ exceptions
- The `" at <file>:<line>"` suffix is always appended and could be parsed, but the file/line refers to the mlx-c wrapper code, not the original error site
- Default handler calls `printf()` then `exit(-1)` — the Go side MUST register a custom handler
- The `data` parameter with `dtor` allows attaching cleanup state (Go could pass a context pointer here)

**GoLand Claude note**: For the refactor from `checkError()` to proper error returns, the best approach is to have the error handler store the last error string (already done), and the Go wrapper functions check the return code (0=success, 1=error) to decide whether to read the stored error. No structured error info is available.

### Memory Management — Complete Picture

#### `mlx_array_free()` — Safe on Graph-Referenced Arrays

**Source**: `private/array.h:49-53`

```cpp
inline void mlx_array_free_(mlx_array d) {
  if (d.ctx) {
    delete static_cast<mlx::core::array*>(d.ctx);
  }
}
```

This deletes the C++ `mlx::core::array` object. But `mlx::core::array` uses `std::shared_ptr<ArrayDesc> array_desc_` (`array.h:522`) internally. So:

- **Graph safety**: Freeing a C handle just decrements the refcount. If the computation graph still holds references (via other arrays' input lists), the data survives. **Safe to free intermediates.**
- **Double-free is NOT safe**: Calling `mlx_array_free()` twice on the same handle calls `delete` twice on the same pointer — undefined behaviour. The Go finaliser must only run once per handle.
- **Free during async operations**: Safe because of refcounting. Async computation holds its own shared_ptr references.
- **NULL-safe**: Checks `d.ctx` before delete, so freeing an empty handle (ctx=NULL) is safe.

#### `mlx_clear_cache()` — Releases Allocator Pool

**Source**: `memory.cpp:11` wraps `mlx::core::clear_cache()`

This releases memory from the allocator's cache pool back to the system. It does NOT release active memory (arrays still in use). Safe to call mid-generation — it only frees allocations that are no longer referenced.

**GoLand Claude note**: Call `mlx_clear_cache()` periodically during generation to prevent memory growth. The allocator pool reuses freed allocations, so under sustained inference, memory should plateau even without explicit cache clears, but clearing helps when switching between different-sized operations.

#### Full Memory API

```c
int mlx_clear_cache(void);                          // Release cached memory to system
int mlx_get_active_memory(size_t* res);              // Currently allocated bytes
int mlx_get_cache_memory(size_t* res);               // Cached (reusable) bytes
int mlx_get_peak_memory(size_t* res);                // High-water mark
int mlx_reset_peak_memory(void);                     // Reset high-water mark
int mlx_set_cache_limit(size_t* res, size_t limit);  // Max cache size (returns previous)
int mlx_set_memory_limit(size_t* res, size_t limit); // Max total memory (returns previous)
int mlx_set_wired_limit(size_t* res, size_t limit);  // Max wired memory (returns previous)
```

**GoLand Claude action**: The Go side should bind `mlx_get_active_memory`, `mlx_get_cache_memory`, `mlx_get_peak_memory`, `mlx_reset_peak_memory`, and `mlx_set_wired_limit` — these are all useful for memory diagnostics. `mlx_set_cache_limit` and `mlx_set_memory_limit` appear to already be bound.

### Metal Device Info

**Location**: `metal.h:31-37`

```c
typedef struct mlx_metal_device_info_t_ {
  char architecture[256];
  size_t max_buffer_length;
  size_t max_recommended_working_set_size;
  size_t memory_size;
} mlx_metal_device_info_t;
mlx_metal_device_info_t mlx_metal_device_info(void);
```

Returns GPU hardware info — architecture name, max buffer size, recommended working set, total memory. Useful for model loading decisions (e.g., choosing model size based on available memory).

**GoLand Claude action**: Consider binding `mlx_metal_device_info()` for automatic model selection.

### Stream API Notes

The Go side uses `mlx_default_gpu_stream_new()`. Additional available:
- `mlx_synchronize(stream)` — block until all ops on stream complete
- `mlx_stream_new_device(dev)` — create stream on specific device
- `mlx_default_cpu_stream_new()` — for CPU fallback ops

### Complete API Surface Map

**Currently bound by Go side**:
- `array.h` — creation, data, shape, dtype, array evaluation
- `ops.h` — partial (~40 of ~180 functions)
- `fast.h` — rms_norm, layer_norm, rope, sdpa
- `transforms.h` — evaluation, compile, vjp, jvp
- `io.h` — safetensors load/save
- `random.h` — categorical only
- `stream.h` — default_gpu_stream
- `metal.h` — is_available
- `error.h` — set_error_handler
- `vector.h`, `closure.h` — support types
- `memory.h` — set_cache_limit, set_memory_limit, clear_cache (partial)

**High-priority unbound functions** (needed for current bugs/features):
1. `mlx_contiguous` — **CRITICAL**: fix Floats() bug
2. `mlx_cumsum` — **CRITICAL**: unblock TopP sampling
3. `_mlx_array_is_contiguous` / `_mlx_array_is_row_contiguous` — optimise data access
4. `mlx_get_active_memory` / `mlx_get_cache_memory` / `mlx_get_peak_memory` — memory diagnostics
5. `mlx_reset_peak_memory` — memory diagnostics
6. `mlx_set_wired_limit` — memory control
7. `mlx_metal_device_info` — hardware detection
8. `mlx_synchronize` — explicit stream sync

**Useful but lower priority**:
- `mlx_cumprod`, `mlx_cummax`, `mlx_cummin` — cumulative ops
- `mlx_topk` / `mlx_topk_axis` — could optimise TopK sampling (currently using argsort)
- `mlx_where` — conditional selection
- `mlx_sort_axis` / `mlx_argsort_axis` — already partially bound?
- `mlx_clip` — clamp values
- `mlx_async_eval` — async evaluation
- `mlx_metal_start_capture` / `mlx_metal_stop_capture` — GPU debugging

**Not needed for inference**:
- `fft.h` — Fast Fourier Transform
- `linalg.h` — linear algebra (inverse, solve, etc.)
- `distributed.h` / `distributed_group.h` — multi-device
- `export.h` — model export
- `map.h` — already used indirectly via safetensors

---

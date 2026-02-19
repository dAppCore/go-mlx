# TODO.md — go-mlx C++ Task Queue

Tasks for the CLion Claude session. Written by GoLand Claude or Virgil.

---

## Orientation (First Session)

- [x] **Map the mlx-c API surface** — Read all 27 headers. Full map in FINDINGS.md: ~180 ops functions, Go binds ~40. Identified 8 high-priority unbound functions. *(Done 2026-02-19)*
- [x] **Understand the error model** — Free-form strings only (`"<msg> at <file>:<line>"`). No error codes or categories. Handler stores string, Go checks return code. Details in FINDINGS.md. *(Done 2026-02-19)*
- [x] **Check memory management patterns** — Arrays are refcounted via `shared_ptr<ArrayDesc>`. Double-free is UB. Free during async is safe. NULL-free is safe. Details in FINDINGS.md. *(Done 2026-02-19)*

## Priority Tasks (from GoLand Claude)

- [x] **Find `mlx_contiguous` or equivalent** — **FOUND**: `mlx_contiguous(res, a, allow_col_major, stream)` at `ops.h:220`. Plus `_mlx_array_is_row_contiguous()` for checking. GoLand Claude: see FINDINGS.md for recommended pattern. *(Done 2026-02-19)*
- [x] **Verify `mlx_array_data_*` eval semantics** — Does NOT auto-evaluate. Returns raw buffer pointer (crash/garbage if unevaluated). `Materialise()` before data access is essential. `item()` auto-evaluates but `data()` does not. *(Done 2026-02-19)*
- [x] **Check if `mlx_cumsum` exists** — **FOUND**: `mlx_cumsum(res, a, axis, reverse, inclusive, stream)` at `ops.h:344`. GoLand Claude can now implement proper TopP sampling. *(Done 2026-02-19)*
- [x] **Survey `mlx_contiguous` / `mlx_flatten` / `mlx_copy`** — All three exist. `mlx_contiguous` is the correct tool (forces row-major). `mlx_copy` may preserve non-contiguous layout. `mlx_flatten` works but changes shape semantics. *(Done 2026-02-19)*

## Memory Management Research (from Backend Abstraction Design)

- [x] **What does `mlx_clear_cache()` release?** — Releases allocator pool cache back to system. Does NOT touch active memory. Safe mid-generation. *(Done 2026-02-19)*
- [x] **Is `mlx_array_free()` safe on graph-referenced arrays?** — Yes, safe. Arrays use `shared_ptr<ArrayDesc>`. Freeing the C handle just decrements refcount. Graph computation continues normally. *(Done 2026-02-19)*
- [x] **MLX allocator pool behaviour** — `mlx_array_free()` returns memory to internal pool (not system). Pool reuses allocations. Under sustained inference, memory should plateau. Call `mlx_clear_cache()` to release pool to system if needed. *(Done 2026-02-19)*
- [x] **Research structured error info** — No structured info available. Free-form string only. Format is stable: `"<message> at <file>:<line>"`. GoLand Claude should use return code (0/1) + stored error string pattern. *(Done 2026-02-19)*

## Standing Tasks

- [ ] **API gap analysis** — When the GoLand Claude needs a C function that isn't exposed by mlx-c, document the gap here and research if upstream mlx-c supports it or if a patch is needed.

## On-Demand Tasks (activate when needed)

- [ ] **mlx-c version bump validation** — When upstream mlx-c releases v0.4.2+ or v0.5.0, update `CMakeLists.txt` GIT_TAG, rebuild, and document any API changes, additions, or breaking changes. Check if new ops could benefit go-mlx (e.g. fused attention variants, new quantisation modes).
- [ ] **Batch evaluation patterns** — Research how mlx-c handles multiple independent forward passes. Does `mlx_eval` with a vector of arrays from separate graphs batch them? Or does MLX need explicit batching at the tensor level? Needed for Phase 5 batch inference API.
- [ ] **GPU profiling/capture** — Document `mlx_metal_start_capture()` / `mlx_metal_stop_capture()` usage for GPU debugging. Research how to generate Metal GPU traces for performance analysis of the inference pipeline.
- [ ] **Quantised matmul variants** — Survey all quantisation-related functions in ops.h beyond what Go currently binds. Document supported bit widths (2-bit, 3-bit?), group sizes, and affine vs symmetric modes. Relevant for model quantisation awareness (Phase 5).
- [ ] **Streaming/async patterns** — Research `mlx_async_eval` and multi-stream patterns. Can separate encode/decode streams overlap? Does MLX support concurrent GPU work from multiple goroutines via separate streams?

---

## Workflow

1. GoLand Claude or Virgil writes tasks here
2. Pick up in order, mark `[x]` when done
3. newArray findings → `cpp/FINDINGS.md`
4. If Go changes needed → note in FINDINGS.md for GoLand Claude

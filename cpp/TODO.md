# TODO.md — go-mlx C++ Task Queue

Tasks for the CLion Claude session. Written by GoLand Claude or Virgil.

---

## Orientation (First Session)

- [ ] **Map the mlx-c API surface** — Read all 27 headers in `build/_deps/mlx-c-src/mlx/c/`. Document which functions the Go side currently binds (cross-reference with Go files) vs which are available but unused. Priority headers: `ops.h`, `fast.h`, `array.h`, `transforms.h`.
- [ ] **Understand the error model** — `error.h` provides `mlx_set_error_handler()`. The Go side registers a handler that logs to stderr. Research: can we get structured error info (error codes, categories)? Is the error string stable or does it vary?
- [ ] **Check memory management patterns** — `mlx_*_free()` functions exist for each type. Verify: is double-free safe? What happens if you free during async eval? Document for the Go finaliser integration.

## Priority Tasks (from GoLand Claude)

- [ ] **Find `mlx_contiguous` or equivalent** — `Floats()`/`DataInt32()` on non-contiguous arrays (transpose, broadcast, slice views) returns wrong data because `mlx_array_data_float32` returns the physical buffer, not the logical layout. Need a C function that copies a non-contiguous array to contiguous memory. Check if `mlx_contiguous` exists in mlx-c headers or if we need `mlx_reshape` to force a copy. This is a data correctness bug — see FINDINGS.md in project root.
- [ ] **Verify `mlx_array_data_*` eval semantics** — Does `mlx_array_data_float32()` trigger evaluation (like C++ `array::data()` does), or must we call `mlx_eval` first? The Go side calls `Materialize()` before data access but some code paths might skip it.
- [ ] **Check if `mlx_cumsum` exists** — The Go TopP (nucleus) sampler in `sample/sample.go` is a stub because it needs cumulative sum along an axis. Check if `mlx_cumsum` or equivalent is in `ops.h`. If so, the GoLand Claude can implement proper TopP sampling.
- [ ] **Survey `mlx_contiguous` / `mlx_flatten` / `mlx_copy`** — We need a way to force an array into contiguous row-major memory. Check all of: `mlx_contiguous`, `mlx_flatten`, `mlx_copy`, `mlx_as_contiguous`. Any of these would fix the Floats() bug.

## Standing Tasks

- [ ] **API gap analysis** — When the GoLand Claude needs a C function that isn't exposed by mlx-c, document the gap here and research if upstream mlx-c supports it or if a patch is needed.

---

## Workflow

1. GoLand Claude or Virgil writes tasks here
2. Pick up in order, mark `[x]` when done
3. New findings → `cpp/FINDINGS.md`
4. If Go changes needed → note in FINDINGS.md for GoLand Claude

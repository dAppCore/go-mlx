# TODO.md — go-mlx C++ Task Queue

Tasks for the CLion Claude session. Written by GoLand Claude or Virgil.

---

## Orientation (First Session)

- [ ] **Map the mlx-c API surface** — Read all 27 headers in `build/_deps/mlx-c-src/mlx/c/`. Document which functions the Go side currently binds (cross-reference with Go files) vs which are available but unused. Priority headers: `ops.h`, `fast.h`, `array.h`, `transforms.h`.
- [ ] **Understand the error model** — `error.h` provides `mlx_set_error_handler()`. The Go side registers a handler that logs to stderr. Research: can we get structured error info (error codes, categories)? Is the error string stable or does it vary?
- [ ] **Check memory management patterns** — `mlx_*_free()` functions exist for each type. Verify: is double-free safe? What happens if you free during async eval? Document for the Go finaliser integration.

## Standing Tasks

- [ ] **API gap analysis** — When the GoLand Claude needs a C function that isn't exposed by mlx-c, document the gap here and research if upstream mlx-c supports it or if a patch is needed.

---

## Workflow

1. GoLand Claude or Virgil writes tasks here
2. Pick up in order, mark `[x]` when done
3. New findings → `cpp/FINDINGS.md`
4. If Go changes needed → note in FINDINGS.md for GoLand Claude

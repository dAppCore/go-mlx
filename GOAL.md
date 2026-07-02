# Goal: Native Engine Replacement
Top rule: keep this file and tracker files as compact worklists only. Do not read or edit them as routine progress logs, and do not add, preserve, or update proof, benchmark, savings, changelog, or status-diary notes here unless the user explicitly asks for tracking changes. When a tracked task is done, remove the task line and report evidence in the turn or commit instead.

Updated 2026-07-02.
Contract: make `go/pkg/native` replace `go/pkg/metal` by copying proven engine behaviour, removing the CGO/MLX dependency, and keeping native CGO-free. Do not add gates or new settings.

Current direction: first-draft feature routes before benchmark polish.

- Use `runtime.Pinner` for Go-owned buffers handed to Metal when tests prove lifetime safety and resource savings.
- Use C++23 `std::mdspan` only for `pkg/metal` comparison, C++ reference tests, or prototypes; do not make it a `pkg/native` dependency.
- Prefer fused/mega-kernel and no-copy streaming work on Apple unified memory: resident weights, stable host views, fewer host round-trips, and command-buffer chains.

Remaining feature tasks:

- Rebase and repair the native/model surface after the incoming `go-inference` update.
- Harden the `pkg/model` + `pkg/native` block-diffusion first draft with focused correctness tests.
- Finish native runtime parity review for ICB, PLE, MTP, no-copy head/weights, and paged/sliding KV paths.
- Shift to benchmark and resource-reduction work only after first-draft feature parity is green.

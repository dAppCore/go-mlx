# Goal: Native Engine Replacement
Top rule: keep this file and tracker files as compact worklists only. Do not read or edit them as routine progress logs, and do not add, preserve, or update proof, benchmark, savings, changelog, or status-diary notes here unless the user explicitly asks for tracking changes. When a tracked task is done, remove the task line and report evidence in the turn or commit instead.

Updated 2026-07-02.
Contract: make `go/pkg/native` replace `go/pkg/metal` by copying proven engine behaviour, removing the CGO/MLX dependency, and keeping native CGO-free. Do not add gates or new settings.

Current direction: first-draft feature routes before benchmark polish.

- Use `runtime.Pinner` for Go-owned buffers handed to Metal when tests prove lifetime safety and resource savings.
- Use C++23 `std::mdspan` only for `pkg/metal` comparison, C++ reference tests, or prototypes; do not make it a `pkg/native` dependency.
- Prefer fused/mega-kernel and no-copy streaming work on Apple unified memory: resident weights, stable host views, fewer host round-trips, and command-buffer chains.

Remaining feature tasks:

- Harden the `pkg/model` + `pkg/native` block-diffusion first draft with focused correctness tests and one live-ish synthetic contract.
- Implement full native target-GGUF loading (assistant-GGUF drafting and mmap dense + Q4_0/Q8_0 dequant exist; the target-model load path is the remaining high-value gap vs `pkg/metal`).
- Make the native assistant-pair (MTP) decode lane production-usable: it engages and verifies but decodes an order below the plain native lane — profile the draft/verify loop and close the gap before calling the lane feature-complete.
- Fix greedy-compatibility gating for speculative decode: at `-temp 0` with default top_p/top_k the MTP output diverges from the plain greedy stream (both engines) — lossless verify must hold, or the drafter must stand down for that config.
- Close the native prefill gap vs `pkg/metal` (several-fold on small models; decode is close behind).
- Finish native runtime parity review for ICB, PLE, no-copy head/weights, and paged/sliding KV paths.
- Resolve `go vet`'s unsafe.Pointer report at `pkg/native/encsend.go:506`.
- Make metallib-dependent native tests skip (not fail) when `MLX_METALLIB_PATH` is unset.
- Decide and implement chat-template framing for the `generate -state` turn loop (today it is a raw completion loop; serve's conversation continuity is the chat-framed path).
- Shift to benchmark and resource-reduction work only after first-draft feature parity is green. Alloc hunt entry points, receipts in dev history 2026-07-02: VisionSDPA, SDPA-causal-BF16, DecodeLayerBatchedKV; the SessionStateRestoreKV and `*Into` paths are already zero-alloc.

<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# Native Engine TODO

Compact tracker for replacing `pkg/metal` with `pkg/native`. Remove completed task lines; do not record proof, benchmark notes, savings, status, or a play-by-play here.

Rules:

- Copy proven engine behaviour from `pkg/metal`; do not reinvent working contracts.
- Keep `go/pkg/native` CGO-free.
- Keep `go/pkg/native` ARCH-FREE: no model-named files, identifiers, or hardcoded model ids in the engine. Nothing is gemma4 — attention/rope/norms/embed-scale are DECLARED config (`model.Arch`), checkpoint formats go through the `pkg/model` registries (`model.RegisterArch`, `model.RegisterAssistant`), and model knowledge lives in `pkg/model/<family>/`. If a new capability seems to need a model name in the engine, it needs a declared field or a registry hook instead.
- Do not add gates or new settings.
- Prefer no-copy streaming, resident buffers, fused/mega kernels, and command-buffer chaining.

Active worklist:

- ICB/session replay: profile submit/readback overhead and make eligible dense/quant hot routes replay by default.
- MoE: remove router/expert host readbacks where practical while preserving byte parity.
- KV cache: close fixed/paged/rotating/sliding cache parity and raw cache restore helpers.
- Head/sampler: exact full-vocab device TopP ranked-prefix/top-mass; no fixed-window approximation.

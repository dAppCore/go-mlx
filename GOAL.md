# Goal: Native Engine Replacement
Updated 2026-06-28. Remove a remaining-task line when it is done; keep compact trackers compact.
Contract: make `go/pkg/native` replace `go/pkg/metal` by copying proven engine behaviour, removing the CGO/MLX dependency, and keeping native CGO-free. Do not add gates or new settings.

Current direction: first-draft feature routes before benchmark polish.

- Use `runtime.Pinner` for Go-owned buffers handed to Metal when tests prove lifetime safety and resource savings.
- Use C++23 `std::mdspan` only for `pkg/metal` comparison, C++ reference tests, or prototypes; do not make it a `pkg/native` dependency.
- Prefer fused/mega-kernel and no-copy streaming work on Apple unified memory: resident weights, stable host views, fewer host round-trips, and command-buffer chains.

Current proof:

- Focused DecodeLayerBatched scratch, AttentionBlock ICB scratch, DecodeForward ICB core scratch, RMS residual scratch, embed-gather scratch, retained-hidden, binary scratch, router host-scratch, VisionSDPA, SDPA, DecodeLayer, decode-step, attention, dense/quant PLE fallback, gate scratch, GPU PLE scratch, and native KV trusted-prefix/metadata/sliding-window boundary restore tests pass.
- Native coverage command and `go tool cover -func` both pass at `81.4%`.
- Root/model smoke tests pass: `go test ./go` `1.170s`; `go test ./go/pkg/model` `0.328s`.
- Coverage target remains `go/pkg/native >=95%`; not met.

Latest completed slice:

- AttentionBlock ICB scratch is keyed by `(dModel, qDim, nHeads, nKVHeads, headDim, kvLen)`; DecodeForward ICB core scratch is keyed by `(dModel, qDim, kvDim, dFF, nLayers)`; RMS residual scratch is keyed by `axisSize`; embed-gather scratch is keyed by `dModel`; binary float32 scratch is keyed by byte length.
- DecodeLayer residual, decode-step attention/MLP, and attention KV scratch pools are dimension-keyed, keeping scratch resident across alternating model/layer/cache shapes.
- Dense BF16 and quant `PerLayerInputs` fallbacks borrow pooled `plHostScratch` and run the fused projection chain when caller scratch is absent.
- `PerLayerInputGate` and GPU PLE input scratch pools are keyed by dimensions, keeping pinned/device scratch resident across alternating layer shapes.
- Borrowed scratch copies final returned bytes before returning scratch to the pool, preserving public API lifetime.
- Dense/quant PLE fallback resource deltas: dense `901-1003 us/op`, `12 allocs/op` -> `229-272 us/op`, `8 allocs/op`; quant `267-292 us/op`, `8 allocs/op`.
- Gate/GPU PLE alternating-shape `Into` benchmarks: `263-282 us/op` and `266-273 us/op`, both `1 allocs/op`.
- Attention benchmark: fixed `233.6-234.9 us/op`, `150-156 B/op`, `3 allocs/op`; alternating `229.6-232.8 us/op`, `148-150 B/op`, `3 allocs/op`.
- DecodeStep benchmark: fixed `281.4-311.6 us/op`, `2918-2921 B/op`, `35 allocs/op`; alternating `306.1-312.7 us/op`, `4004-4047 B/op`, `48-49 allocs/op`.
- DecodeLayer benchmark: fixed `259.8-275.3 us/op`, `165-166 B/op`, `4 allocs/op`; alternating `270.4-274.3 us/op`, `226-228 B/op`, `4 allocs/op`; DecodeLayerBatched fixed/alternating `41`/`36 allocs/op`.
- ICB/RMS/embed-gather/binary/router/SDPA/VisionSDPA benchmarks: AttentionBlock ICB `32-34 allocs/op`; DecodeForward ICB `504-506`; RMS, embed-gather, binary, and SDPA fixed/alternating stay at `2`; router host-scratch `5-6`; VisionSDPA fixed `14-15`, alternating `447-451`.
- Native `RestoreKVBlocks` grafts resident trusted-prefix tokens, restores suffix-only absolute blocks, and carries per-layer cache index/mode/max-size metadata through native block descriptors.
- Native state-block capture/restore maps post-cap sliding-window cache rows through their physical ring slots, splits streamed block boundaries at the live-window start like metal, preserves zero-copy contiguous views for full/pre-cap blocks, and keeps range streaming at zero allocations.

Remaining feature tasks:

- First-draft no-copy/fused routing into session/replay hot paths that still submit/read back per op.
- First-draft MoE router/expert GPU flow that removes host readbacks while preserving parity.
- Finish KV cache parity for fixed, paged, rotating/sliding, and raw restore helpers beyond trusted-prefix suffix restore.
- First-draft exact full-vocab TopP ranked-prefix/top-mass device path; no fixed-window approximation.

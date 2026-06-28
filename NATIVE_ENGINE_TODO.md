<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# Native Engine TODO

Compact tracker for replacing `pkg/metal` with `pkg/native`. Do not grow this into a play-by-play.

Rules:

- Copy proven engine behaviour from `pkg/metal`; do not reinvent working contracts.
- Keep `go/pkg/native` CGO-free.
- Do not add gates or new settings.
- Prefer no-copy streaming, resident buffers, fused/mega kernels, and command-buffer chaining.

Current proof:

- Focused DecodeLayerBatched scratch, AttentionBlock ICB scratch, DecodeForward ICB core scratch, RMS residual scratch, embed-gather scratch, retained-hidden, binary scratch, router host-scratch, VisionSDPA, SDPA, DecodeLayer, decode-step, attention, dense/quant PLE fallback, gate scratch, and GPU PLE scratch tests pass.
- Native package coverage passes at `81.4%`; target remains `>=95%` with real feature/edge tests.
- Root/model smoke passes in `1.170s`/`0.328s`.

Latest measured wins:

- Dense/quant `PerLayerInputs` without caller scratch borrow pooled `plHostScratch` and use the fused projection chain: dense `901-1003 us/op`, `855-867 B/op`, `12 allocs/op` -> `229-272 us/op`, `178-180 B/op`, `8 allocs/op`; quant `267-292 us/op`, `180-181 B/op`, `8 allocs/op`.
- `PerLayerInputGate` and GPU PLE input scratch are dimension-keyed; alternating `Into` paths stay at `1 allocs/op`.
- Attention KV and AttentionBlock ICB scratch are keyed by shape; Attention KV fixed/alternating stays at `3 allocs/op`; AttentionBlock ICB fixed `269.2-276.5 us/op`, `1824-1830 B/op`, `32 allocs/op`, alternating `268.1-270.5 us/op`, `1978-1997 B/op`, `34 allocs/op`.
- Decode-step attention/MLP and DecodeLayerBatched scratch are keyed by dimensions; DecodeStep fixed/alternating is `35`/`48-49 allocs/op`, batched fixed `488.3-525.0 us/op`, `21114-21172 B/op`, `41 allocs/op`, alternating `452.9-466.1 us/op`, `13726-13743 B/op`, `36 allocs/op`.
- DecodeLayer residual scratch is keyed by `dModel`; fixed DecodeLayer bench is `259.8-275.3 us/op`, `165-166 B/op`, `4 allocs/op`, alternating bench is `270.4-274.3 us/op`, `226-228 B/op`, `4 allocs/op`.
- SDPA and VisionSDPA scratch are keyed by attention shape; fixed/alternating SDPA stays at `2 allocs/op`, VisionSDPA fixed `14-15 allocs/op`, alternating `447-451 allocs/op`.
- Router host scratch is keyed by `(dModel, numExperts)`; fixed host-scratch router bench is `339.6-350.9 us/op`, `46 B/op`, `5 allocs/op`, alternating pooled bench is `343.0-347.7 us/op`, `109-127 B/op`, `6 allocs/op`.
- Binary float32 scratch is keyed by byte length; fixed Add bench is `172.9-181.4 us/op`, `4124-4127 B/op`, `2 allocs/op`, alternating Add bench is `181.6-183.4 us/op`, `6296-6335 B/op`, `2 allocs/op`.
- Embed-gather scratch is keyed by `dModel`; fixed quant gather bench is `178.6-187.2 us/op`, `1037-1040 B/op`, `2 allocs/op`, alternating `dModel` bench is `185.6-188.4 us/op`, `1552-1557 B/op`, `2 allocs/op`.
- RMS residual BF16 and DecodeForward ICB core scratch are shape-keyed; RMS fixed/alternating stays at `2 allocs/op`; ICB fixed `679.8-697.7 us/op`, `20810-20876 B/op`, `504 allocs/op`, alternating `686.8-690.1 us/op`, `20958-21053 B/op`, `505-506 allocs/op`.

Active worklist:

- ICB/session replay: profile submit/readback overhead and make eligible dense/quant hot routes replay by default.
- MoE: remove router/expert host readbacks where practical while preserving byte parity.
- KV cache: close fixed/paged/rotating/sliding cache parity and raw cache restore helpers.
- Head/sampler: exact full-vocab device TopP ranked-prefix/top-mass; no fixed-window approximation.

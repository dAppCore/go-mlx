<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# Codex Native Engine Work

Compact ledger for the active `pkg/native` replacement goal. Keep only the current slice, proof, and next handoff.

Current slice completed 2026-06-28:
- DecodeLayerBatchedKV scratch is keyed by `(dModel, qDim, kvDim, nHeads, dFF, K)`, so alternating MTP verify batch shapes keep pinned row/output scratch separate; no gate or setting was added.

Verification:
- Focused batched decode scratch/correctness/allocation tests passed; benchmarks: fixed `488.3-525.0 us/op`, `21114-21172 B/op`, `41 allocs/op`; alternating `452.9-466.1 us/op`, `13726-13743 B/op`, `36 allocs/op`.
- Full native coverage passed at `81.4%`; root `1.170s`; model `0.328s`.

Retained wins:
- Dense PLE fallback `901-1003 us/op`, `855-867 B/op`, `12 allocs/op` -> `229-272 us/op`, `178-180 B/op`, `8 allocs/op`.
- Quant PLE fallback `267-292 us/op`, `180-181 B/op`, `8 allocs/op`.
- Gate scratch and GPU PLE scratch are dimension-keyed; alternating `Into` paths stay at `1 allocs/op`.
- Attention, AttentionBlock ICB, DecodeForward ICB core, DecodeLayerBatched, DecodeStep, and DecodeLayer scratch pools are dimension-keyed; alternating paths stay at `3`, `34`, `505-506`, `36`, `48-49`, and `4 allocs/op` respectively.
- SDPA and VisionSDPA scratch pools are dimension-keyed; fixed/alternating residency is covered by benchmarks.
- Router host scratch is dimension-keyed; fixed and alternating pooled paths stay at `5-6 allocs/op`.
- RMS residual, binary float32, and embed-gather scratch are dimension-keyed; fixed and alternating RMS/Add/gather stay at `2 allocs/op`.

Next handoff:
- Continue replacing per-op submit/readback session hot paths with fused no-copy/replay routes.
- Exact large-vocab TopP still needs a full-vocab ranked-prefix/top-mass device design.

<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# Codex Native Engine Work

Created 2026-06-22; compacted 2026-06-23. Live ledger; `GOAL.md` holds contract.

Done: kernels/resources; resident heads/scratch; BF16/qmv greedy; MTP direct greedy; softcap; whole-tensor/fused MoE; MoE quant per-weight geometry; mmap MoE quant triple views; ICB/cache replay; prompt cache.

Proof: MoE router mixed-geometry red→green; router/load guards 0.724s; coverage 40.687s at 91.5%.

Resource: router quant top2 19,540 B/op; MTP direct loop 10,972,933 B/op, 251,720 allocs; prompt cache ~33% faster/~29% fewer bytes+allocs; MoE combine 611,060 -> 484,152 ns/op; head BF16 16,008 vs 23,168 B/op, quant 19,937 vs 21,433.

Rejected: direct q4 top-1 until fused-equivalent qmv proof handles near ties.

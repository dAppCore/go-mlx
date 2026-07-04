<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# MTP boost — the multi-token (small-L) fast decode path

**Status:** in progress. Slice 1 (batched quantised matvec) DONE + landed
(`d0ce8320`): verify 56→52 ms/call, 31B q4 MTP 0.75x→0.81x, plain unchanged,
greedy-exact, unit-tested. Slice 2 (multi-query fused attention) is the
remaining lever to cross 1x — the harder kernel, for a focused session.

## Why MTP is below 1× today (measured, not guessed)

`TestSpeculativeBoost_Repro` with the `split:` logging, 31B q4 target + q4 QAT
drafter, 200 tok, draftTokens=2:

```
draft  = 2.9 ms/block   (~1.5 ms/step)   ← cheap; the drafter is NOT the problem
verify = 56 ms/call                       ← the wall (92% of MTP wall time)
   layers 52 ms  (attn ~45% / MLP ~55%, Eval-barrier split) + output 3 ms
```

Per decoder layer: the verify (L=2-3) costs **~1.75× a single-token (L=1)
decode**, across BOTH attention and MLP. Cause: every fast decode kernel is
gated to `L==1` and the batched verify (L>1) bypasses all of them:

| fast path (L==1) | where | L>1 verify falls to |
|---|---|---|
| `NativeFixedSingleTokenAttention` (attn+cache+norm fused, 1 kernel) | `attention.go:86` | separate KProj/VProj/norms/RoPE + `c.Update` + `ScaledDotProductAttention` (fast op, but un-fused, ~8 ops/layer) |
| `QuantizedDenseMatVec` (proj matvec) | `dense_matvec.go:108` requires `[1,1,in]` | `quantizedMatmulMode` (generic quantised GEMM) for QProj/OProj |
| `nativeMLPMatVec` (fused gate/up/down matvec) | requires `[1,1,in]` | the compiled `q4_g64_mlp_gelu` GEMM (better, but still not the L=1 fused matvec) |

The decode-time win of speculation is amortising the weight stream across k+1
tokens in ONE forward. We get that (verify is one forward), but we pay
**per-token generic compute** because the small batch misses the fused
single-token kernels — so the batched forward costs ~1.75× a single decode
instead of ~1×.

## The fix — a multi-token (L=2..4) fast decode path

Make the L∈[2..4] forward as bandwidth-bound as L=1 by giving the fused kernels
a small-batch mode (weights loaded once, reused across the L token-rows):

1. ✅ **DONE (`d0ce8320`) — Batched quantised matvec** (`dense_matvec.go`): row-loop
   in `QuantizedDenseMatVec` + `quantizedDenseGELUSplitGateUpMatVec` (weight word
   loaded once per `out_col`, fanned across L rows). `validateQuantizedDenseMatVec`
   accepts `[1,L,in]` for `L<=maxDecodeMatVecBatch` (8); q6 + non-contiguous
   decline. Covers QProj/OProj + the whole MLP. Result: verify 56→52 ms, MTP
   0.75x→0.81x. Smaller than hoped — the matmuls were ~GEMM-efficient already;
   the win is the explicit weight reuse. The bulk of the residual is NOT the
   matmuls.
2. **Multi-query fused attention** — the remaining lever (the verify is still
   ~1.6x a single-token decode). The L=1 path fuses attention+cache-update+norm
   into ONE kernel (`NativeFixedSingleTokenAttention`, attention.go:86); the L>1
   verify does ~8 separate ops/layer (KProj/VProj/norms/RoPE + `c.Update` +
   `ScaledDotProductAttention`). Need a small-L variant of the fused kernel: L
   query rows over the cache + the L new K/V rows, causal within the block,
   sliding-window aware. The hard kernel; focused session.
3. Wire `Gemma4Attention.forward` to prefer the fused multi-query path when
   `1 < L <= maxDecodeMatVecBatch`, else current behaviour.

Re-measure the attn-vs-mlp split AFTER slice 1 before building slice 2, to
confirm the residual is the un-fused attention/cache dispatch (it should be).

## Validation (the safety net makes this low-risk despite being kernel work)

- **Greedy-exact gate** (`TestSpeculativeBoost_Repro`): MTP output MUST equal the
  target's plain greedy. Output is target-determined, so a wrong kernel either
  fails this gate or tanks the accept rate — it CANNOT ship silent corruption.
- **`split:` logging**: watch `verify` ms/call drop from ~56 toward ~35.
- Per-step iteration is cheap (~18s/run; the 17GB target is mmap/disk-cached).
- Models cached: `gemma-4-31b-it-4bit` + `gemma-4-31B-it-qat-assistant-4bit`.

## Honest ceiling — read before investing

Even with a perfect multi-token verify (≈ single-token cost) + the matched QAT
target (`gemma-4-31b-it-qat-4bit`, accept ~0.475-0.6) + tuned draftTokens, the
math caps **31B at ~1.5-1.7× → ~45-51 tok/s** (up from 30). A speculative verify
is still ~one full target forward per ~2 emitted tokens; that ratio is the
floor.

- The `/goal` "100 tok/s on e2b/e4b/1b/26b/31b at q4 & q6" is **bandwidth-
  impossible above e2b** (31B q4 = 17 GB / 819 GB/s ≈ 48 sequential ceiling).
- "60-80 on 31B" exceeds even the speculative ceiling above.
- **30 → ~48 (≈1.6×) is the real, achievable prize.** Worth it, but it is not
  100 and not 60-80. Decide accordingly.

## Already landed (dev)
- `1cdf2f9f` go-mlx loads quantised (QAT) drafters (2 loader bugs fixed).
- `e8231616` the draft/verify `split:` diagnostic.
- Reverted dead ends: compile-the-draft-layer (wash), fast-per-position-output
  (no-op). The draft was never the bottleneck (that was an arithmetic error).

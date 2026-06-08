<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# AX-11 decode benchmark matrix — Gemma-4 (2026-06-08)

`BenchmarkGenerate_ContextGrowth` (`pkg/metal/generate_growth_bench_test.go`),
greedy, 512-token decode, `DefaultEngineFeatures().Apply()` (the serve's real
fast-path gates), rotating cache, M3 Ultra (~819 GB/s), dev `d0ce8320`.

Reproduce per model:

```
GO_MLX_BENCH_MODEL=mlx-community/<repo> \
GOWORK=$PWD/go.work MLX_METALLIB_PATH=$PWD/dist/lib/mlx.metallib \
go test -C go -tags 'metal_runtime model_eval' \
  -ldflags "-extldflags=-mmacosx-version-min=26.0" \
  -bench 'BenchmarkGenerate_ContextGrowth/greedy/tokens_512' -benchtime=1x -run '^$' \
  dappco.re/go/mlx/pkg/metal
```

## Decode tok/s

| model | q4 | q6 | q8 | bf16 |
|---|---:|---:|---:|---:|
| 1b (gemma-3) | **204.0** ✅ | **144.2** ✅ | — | — |
| e2b | **109.6** ✅ | 73.5 | 85.7 | 27.0 |
| e4b | 76.0 | 48.7 | — | — |
| 26b-a4b (MoE) | 54.3 | 45.1 | — | — |
| 31b (dense) | 30.3 | 14.2 | — | — |
| 12b unified | — | 36.4 | — | — |

The small models the fleet uses for *volume* coder work clear 100 at both q4 and
q6: **1b (204/144) and e2b q4 (110)**. The orchestrator-class models (e4b/26b/31b)
are bandwidth-bound below 100 — physics, not effort.

Single-sample (`-benchtime=1x`), so ~5-10% under a warm multi-sample run
(e.g. e2b q4 warms to ~117); directionally exact and reproducible.

## Against the /goal (100 tok/s q4 & q6 on e2b/e4b/1b/26b/31b; 50 tok/s q8/bf16)

| target | result |
|---|---|
| e2b q4 ≥ 100 | ✅ 109.6 |
| e2b q8 ≥ 50 | ✅ 85.7 |
| e2b q6 ≥ 100 | ❌ 73.5 (overhead-bound on the bitstream kernel) |
| e2b bf16 ≥ 50 | ❌ 27.0 (bf16 = 2 B/weight; even e2b can't hit 50) |
| e4b / 26b / 31b, any quant ≥ 100 | ❌ bandwidth-bound |

**Why everything above e2b misses 100: physics, not a bug.** Decode is
memory-bandwidth-bound — `tok/s ≈ 819 GB/s ÷ resident-weight-bytes`. 31B q4 ≈
17 GB → ~48 tok/s sequential ceiling; q6 has *more* bytes than q4 so its ceiling
is always lower. Only e2b is small enough to be overhead-bound rather than
bandwidth-bound, so only e2b q4 clears 100. No kernel beats the bandwidth wall.

**The only lever above e2b is speculative decode (MTP)** — and even a perfect MTP
is capped ~1.5-1.7× (31B 30 → ~48) by the verify-forward floor, not 100. See
`2026-06-07-mtp-batched-decode-kernel.md`. Slice 1 (batched matvec) landed
0.75×→0.81× on 31B; slice 2 (fused multi-query attention) is the remaining lever.

**Bottom line:** the matrix is the achievable ceiling on this hardware. "100
everywhere at q4 & q6" is bandwidth-impossible above e2b; the realistic target is
e2b-clears-100 (met) + push the big models toward their bandwidth ceilings via
MTP.

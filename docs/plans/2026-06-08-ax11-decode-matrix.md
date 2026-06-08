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

## Decode tok/s — plain greedy (current code, dev `4efd1b64`, 2026-06-08)

| model | q4 | q6 | q8 | bf16 |
|---|---:|---:|---:|---:|
| 1b (gemma-3) | **224.5** ✅ | **151.6** ✅ | — | — |
| e2b | **117.4** ✅ | 77.8 | **89.6** ✅ | 27.1 |
| e4b | 78.6 | 50.7 | — | — |
| 26b-a4b (MoE) | 54.4 | 46.2 | — | — |
| 31b (dense) | 30.3 | 14.4 | — | — |

`-benchtime=1x` single-sample (~5-10% under a warm run). No regression vs the
prior matrix; e2b q6 picked up the q6 fused-output commit (`9fc4709d`).

## Against the /goal (100 tok/s q4 & q6 on e2b/e4b/1b/26b/31b; 50 tok/s q8/bf16)

Plain decode meets: **1b q4/q6, e2b q4, e2b q8**. The rest need the MTP lever.

| cell | plain | with MTP (post-norm fix) | note |
|---|---:|---:|---|
| 1b q4/q6 | 224 / 152 | n/a | ✅ clears 100 plain |
| e2b q4 | 117 | n/a | ✅ clears 100 plain |
| e2b q8 | 90 | n/a | ✅ ≥ 50 |
| e2b q6 | 78 | **89** (1.15×) | MTP helps; short of 100 (accept 0.42 vs ref 0.70) |
| e4b q4/q6 | 79 / 51 | — | **no assistant cached** → no MTP |
| 26b q4/q6 | 54 / 46 | **39** (0.84×) | MTP *hurts* — MoE verify > accepted savings |
| 31b q4/q6 | 30 / 14 | ~34 (1.15×) | far from 100; verify-floor caps even perfect MTP |
| e2b bf16 | 27 | n/a | ❌ ≥ 50 (bf16 = 2 B/weight) |

## What the target surfaced (Snider: it's a diagnostic, not a hard limit)

Decode is **occupancy-bound** on single-token matvecs (~13% of peak BW; tok/s ×
bytes-per-weight ≈ const across q4/q6/bf16). No kernel tweak moves the q6 column
(custom Q6Group64 vs mx affine-q6 = wash). The lever above that wall is
speculative decode, and the MTP **machinery is efficient** (a 3-token batched
verify ≈ 1.1 plain-token-times on e2b) — so the speedup ceiling is
`accepted-per-round ÷ ~1.1`. At the ~0.70 acceptance reference impls get, e2b q6
→ ~150 (clears 100 with room).

**The wall was a BUG, not physics.** MTP acceptance was 0.19-0.33 across all
quants; root cause: the EAGLE head was seeded with the pre-final-norm hidden, not
the post-final-norm feature its LM head reads. Fixed (`4efd1b64`): e2b q6 accept
0.237→0.332, 1.03×→1.15×; generalises to the 26b MoE (0.24→0.40). Greedy-exact
holds throughout.

**Open:** acceptance is up but still 0.42 vs 0.70 — a 2nd draft-quality gap,
localised to the assistant's predicted FEATURE (output path / RoPE / shared-KV
all eliminated or by-design; see `project_go_mlx_perf_matrix_and_mtp_reality`
memory). Next move is a token-by-token diff against the reference EAGLE numerics.
Two structural levers remain for the matrix: (1) close acceptance → 0.70 (lifts
every MTP-eligible cell), (2) the **26b MoE verify** needs to be as batch-efficient
as e2b's before MTP can help it, and **e4b needs an assistant** at all. 31b is the
genuine outlier — even 2× MTP gives ~60, so it wants a faster orchestrator path,
not just MTP.

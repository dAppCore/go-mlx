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

## Re-validation — dev `fc26e518` (2026-06-08)

Per-token phase tracer (`TestTrace_DecodePhaseBreakdown_Diag`, 160-token
steady-state — runs ~5-8% over the 512-token `ContextGrowth` bench above because
it carries less KV-context growth). Confirms the matrix above and the two
conclusions that drive it.

| model | q4 | q6 | q8 |
|---|---:|---:|---:|
| 1b (gemma-3) | 221.3 | 158.0 | — |
| e2b | 123.5 | 81.4 | 100.4 |
| e4b | 86.0 | 54.2 | — |
| 12b (dense) | ~56* | 39.0 | — |
| 26b-a4b (MoE) | 57.2 | 49.7 | — |
| 31b (dense) | 31.5 | ~14 | — |

`*` 12b q4 not cached locally; estimated from the q6→q4 ~1.45× ratio the other
models show. 31b q6 from the prior 512-bench.

## Target (Snider, 2026-06-08, revised from "100 on all five")

Tiered, **plain decode, no MTP** — MTP is a boost on top, not the baseline:

- **< 12B (1b, e2b, e4b): 100+ tok/s**
- **≥ 12B (12b, 26b, 31b): 50+ tok/s**

| model | q4 | q6 | tier | plain verdict |
|---|---:|---:|---|---|
| 1b | 221 | 158 | 100+ | ✅ ✅ |
| e2b | 123 | 81 | 100+ | ✅ · ✗ (q6 at the ~83 6-bit ceiling) |
| e4b | 86 | 54 | 100+ | ✗ · ✗ |
| 12b | ~56 | 39 | 50+ | ~✅ · ✗ |
| 26b-a4b | 57 | 50 | 50+ | ✅ · ✅ |
| 31b | 31.5 | ~14 | 50+ | ✗ · ✗ |

Baseline accepted as "good"; improve from here. Gaps to close, all on the shared
single-token occupancy lever (plain decode at ~1.6×–5× off the BW floor): **e4b
q4 86→100**, **12b q6 39→50**, **31b q4 31→50**. The q6/format-ceiling cells
(e2b q6, e4b q6, 31b q6) and 31b q4 are the ones MTP is meant to lift past their
plain numbers.

Two things landed/were re-proved this pass:

1. **e2b q6 regression fixed (`fc26e518`).** The unified-matvec commit
   (`87cbf91b`) had folded q6's main matvec (q/k/v/o + down) into the q4/q8
   word-coalesced straddle loop, dropping the group-64 bit-position precompute
   and costing q6. Restored the dedicated q6 Group64 kernel on the main matvec,
   symmetric with the GELU gate/up path that already kept it. e2b q6 78.9 → 81.4.
   Parity held (`TestDenseMatVec` q6 default + E2B-shape).

2. **"No kernel tweak moves the q6 column" re-proved, now both ways.** Routing
   the q6 layers through MLX-native `quantized_matmul` instead of the hand-rolled
   kernels gives **83.1** vs the hand-rolled **81.4** — a 2% wash, *not* a path to
   100. The win sits mostly in the fused GELU (gate-off-only, GELU still
   hand-rolled, is 81.9; full-native 83.1). Both land at the ~83 ceiling: Apple's
   own q6 kernel is also q6 < q8 (83 < 100), so 6-bit's non-byte-aligned packing
   is the limiter, **not** a go-mlx bug. The hand-rolled q6 kernels are kept (they
   tie native and keep the unified q4/q8 fast-path intact); a follow-up could
   delete them for native at +2% if the simplification is wanted.

**The universal shape:** q6 sits ~35% below q4 on *every* model (1b 158/221,
e2b 81/123, e4b 54/86, 26b 50/57) — the format cost is fixed, not model-specific.
Plain decode runs at ~1.6×–5× off the memory-bandwidth floor; the gap *shrinks*
with model size (31b only 1.6× off, e2b ~5× off) because larger matvecs occupy
the GPU better. So the single-token occupancy wall — and the MTP lever above it —
is exactly as the matrix states; nothing in the plain-decode kernels closes the
e2b-q6 / e4b cells to 100. The lever for those remains MTP acceptance (0.42→0.70).

## MTP lever VALIDATED — QAT matched pairs (2026-06-08)

The go-mlx MTP path is reference-correct (verified against llama.cpp PR #23398 on
every axis — see `project_go_mlx_mtp_acceptance_reference_verified`). The official
**QAT** matched pairs (`mlx-community/gemma-4-{SIZE}-it-qat-4bit` target +
`…-qat-assistant-4bit` drafter, "full MTP support") validate the mechanics:

| pair (q4 QAT) | plain (repro) | MTP peak | accept | tier | meets? |
|---|---:|---:|---:|---|---|
| e2b | ~98 | **114.5** (dt3, 1.14×) | 0.455 | 100 | ✅ |
| e4b | ~67 | 76 (dt2, 1.14×; ~98 trace-adj) | 0.324 | 100 | ~borderline |
| 12b | 44 | **50.4** (dt3, 1.14×) | 0.372 | 50 | ✅ |
| 26b-A4B | 56 | **75.4** (dt3, 1.35×) | 0.444 | 50 | ✅ |
| 31b | 21/31 | 25 (dt3, 1.17×; ~37 trace-adj) | 0.449 | 50 | ✗ (31B dense, BW-capped) |

(repro tok/s is prefill-diluted over 200 tokens; the ×speedup is the fair signal;
greedy-exact correctness gate green on every pair, incl. the unified drafters.)

**q6 QAT MTP (the q6 column of the goal):** e2b q6 = plain 86.1 → **MTP 100.0**
(1.16×) — **clears 100**, so e2b meets the 100-tier at BOTH q4 (114.5) and q6.
e4b q6 = 51.7 → 66.1 (1.28×), short (4B). So the small-model 100-tier is met by
**1b (plain) and e2b (q4+q6)**; e4b is the lone <12B model that stays under at
both quants. q8 clears 50 on plain alone (e2b q8 = 100); bf16 (2 B/weight) is
bandwidth-bound like 31b (e2b bf16 ≈ 27) — a physics miss, not a code gap.

**The 12b/26b/31b drafters are `gemma4_unified_assistant`** (unified-text variant)
which go-mlx didn't load — added that arch (commit `4ae6766e`), which is what
made the big-model MTP runnable at all. The bigger the target the better the
speedup (26b 1.35× > e2b 1.14×), matching the reference's "larger targets up to
3.94×". **Tier verdict: 1b/e2b/12b/26b clear; e4b borderline (~98); 31b is the
genuine outlier** — 31B dense is bandwidth-capped below 50 even with MTP. The
remaining lift (e4b over 100, the q6 cells) is drafter acceptance (0.32–0.45 vs
ref ~0.70) → a tree/multi-candidate draft strategy, the next improvement.

<!-- SPDX-Licence-Identifier: EUPL-1.2 -->
# Rival Inference-Engine Commit Watch

Daily digest of what shipped in rival open-source inference engines, filtered through the
go-mlx lens (temporally-aware, CONT/no-replay retained-state engine; KV/state persists and is
mounted via Wake/Sleep, not re-prefilled). Newest entry at the top.

Repos tracked: `ml-explore/mlx`, `ml-explore/mlx-lm`, `Blaizzy/mlx-vlm`,
`lmstudio-ai/mlx-engine`, `ggml-org/llama.cpp`, `vllm-project/vllm`.

---

## 2026-06-10 (07:16 UTC run) — window 2026-06-09 05:04 → 2026-06-10 07:04 UTC (~26h)

> ⚠️ **Feeds still blocked + quiet window.** The 18 Atom feeds remain unreachable through
> `web_fetch`'s provenance allowlist (unchanged; the hard-code-the-18-URLs task-file fix is still
> pending and still the right call — concrete URLs in the task message would enter provenance and
> end this whole dance). No out-of-policy fetch methods used (no curl/wget/python/MCP); browser
> offline. What *did* render this run: **mlx-lm's bare `/commits`** (WebSearch happened to surface
> the no-slash URL — the only commit stream observable) and the llama.cpp + vllm `/releases` pages.
> Re-confirmed the wall: `.atom` URLs aren't search-indexed; the branch-qualified
> `/commits/<branch>`(`/`) HTML view returns an empty shell (only the bare `/commits` redirect
> renders); links inside fetched page bodies do **not** enter provenance, only WebSearch
> *result-links* do. Key result this window: **llama.cpp's latest build is still b9568
> (08 Jun 21:10 UTC) — unchanged since yesterday's run, so no new in-window builds** (a real ~34h
> lull or paused CI). vllm `/releases` came back **stale-cached again** (v0.20.2 / 10 May shown as
> "latest"); deferring to the v0.22.0 (29 May) / v0.22.1 (5 Jun, unverified) anchors.

### ⭐ Worth a look for go-mlx

Quiet day — nothing actionable shipped inside the window. The Gemma 4 MTP / iSWA-mask thread
flagged the last two runs (llama.cpp b9549 [#23398](https://github.com/ggml-org/llama.cpp/pull/23398),
b9566 [#24294](https://github.com/ggml-org/llama.cpp/pull/24294),
b9568 [#24282](https://github.com/ggml-org/llama.cpp/pull/24282)) has now rolled just *outside* the
26h window with no follow-on builds. Still the live thread on go-mlx's path (Gemma 4 dense+MoE + the
MTP batched-decode kernel plan, `docs/plans/2026-06-07-mtp-batched-decode-kernel.md`), but nothing
new to diff today.

### Per repo

**ggml-org/llama.cpp** — `/releases` cache-fresh; **latest build unchanged at b9568 (08 Jun 21:10
UTC)** — no new tags in window (b9568 now sits ~8h before the window opens). Releases body identical
to yesterday's (b9557–b9568). — quiet this window.

**ml-explore/mlx-lm** — commits **observable this run** (bare `/commits` rendered): newest is
`Fix Gemma 4 sanitize() not stripping KV projections for shared layers`
([#1240](https://github.com/ml-explore/mlx-lm/commit/df1d3f3c9a7aae402dcbb8f41d4c36bcc13a50ae),
4 May) — nothing since. No in-window commits. — quiet. (Backlog below the tip is heavy on go-mlx's
exact path — `ArraysCache`/`BatchKVCache` extend fixes #1177/#1169/#1141, `LRUPromptCache` refactor
#1019, `PromptTrie` prefix-cache off-by-one #1078, spec-decode output-corruption fix #1109 — but
all April, well pre-window.)

**ml-explore/mlx** — commits not observable (only the empty-rendering `/commits/main` view). No
release in window; latest remains [v0.31.2](https://github.com/ml-explore/mlx/releases/tag/v0.31.2)
(22 Apr, re-confirmed fresh). Gap.

**Blaizzy/mlx-vlm** — commits not observable (`/activity` returned an empty JS shell). No release in
window; prior-verified anchor [v0.5.0](https://github.com/Blaizzy/mlx-vlm/releases/tag/v0.5.0)
(6 May) / 0.6.1 (3 Jun, unverified) — predates the window. Gap.

**lmstudio-ai/mlx-engine** — commits not observable; repo publishes no GitHub releases (ships via
the LM Studio app). Search reports the repo last updated 8 Jun (just before the window). Gap for the
window.

**vllm-project/vllm** — commits not observable (biggest blind spot; normally dozens of merges/day).
`/releases` **stale-cached again** (v0.20.2 / 10 May as "latest"); defer to v0.22.0 (29 May) /
v0.22.1 (5 Jun, unverified) — both predate the window. Gap.

### Gaps

- Atom feeds: all 18 unavailable (provenance restriction; task-file hard-code fix still pending).
- In-window commit content unknown for mlx, mlx-vlm, mlx-engine and vllm; mlx-lm *was* observable
  this run (quiet since 4 May).
- llama.cpp: no new build tags since b9568 (08 Jun 21:10) — read as a genuine lull, but a single
  `/releases` page only; can't fully rule out an unbuilt in-window master push.
- vllm `/releases` stale-cached (v0.20.2 shown as latest); v0.22.0/v0.22.1 anchors used instead.

---

## 2026-06-09 (07:04 UTC run) — window 2026-06-08 05:04 → 2026-06-09 07:04 UTC (~26h)

> ⚠️ **Feeds still blocked** — the 18 Atom feeds remain unreachable through `web_fetch`'s
> provenance allowlist (unchanged from the runs below; the hard-code-the-18-URLs task-file fix
> from the 00:09 entry is still pending and still the right one). Re-confirmed this run: `.atom`
> URLs are not search-indexed, same-origin `/commits.atom` is rejected even once the repo page is
> in the set, and the JS-rendered `/commits/<branch>` HTML view returns an empty shell via
> `web_fetch`. No out-of-policy fetch methods used (no curl/wget/python/MCP). Browser offline.
> **This run:** llama.cpp `/releases` came back cache-fresh and fully timestamped (best coverage
> yet — 12 builds with UTC times); but the **mlx-vlm and vllm `/releases` pages came back stale**
> (cached snapshots showing v0.4.0 / 7 Mar and v0.20.2 / 10 May as "latest", both older than
> previously-verified releases) — so for those two I defer to the safer prior anchors below rather
> than regress the log.

### ⭐ Worth a look for go-mlx

- **llama.cpp b9568 — `mtp: support for gemma-4 E2B and E4B assistants`
  ([#24282](https://github.com/ggml-org/llama.cpp/pull/24282))** (08 Jun 21:10 UTC, in window).
  Multi-token-prediction draft/assistant heads for Gemma 4 E2B/E4B (adds `masked_embd` tensors to
  the gemma4-assist arch + converter support). This **continues** last run's Gemma 4 MTP merge
  (b9549 / [#23398](https://github.com/ggml-org/llama.cpp/pull/23398), 7 Jun) — a sustained
  upstream push on exactly go-mlx's path: we ship Gemma 4 (dense + MoE) and have an MTP
  batched-decode kernel plan (`docs/plans/2026-06-07-mtp-batched-decode-kernel.md`). Worth diffing
  their assistant-head conversion + masked-embedding wiring against ours. (models, spec-decode) —
  https://github.com/ggml-org/llama.cpp/releases/tag/b9568
- **llama.cpp b9566 — `graph: guard iswa kq_mask on its own buffer`
  ([#24294](https://github.com/ggml-org/llama.cpp/pull/24294))** (08 Jun 18:07 UTC, in window).
  Interleaved sliding-window-attention (iSWA) KQ-mask moved onto its own buffer — a
  correctness/aliasing guard in the sliding-window path. Relevant to go-mlx's `RotatingKVCache`
  sliding-window masking; cheap to check whether our mask buffering has the same hazard.
  (KV/state, Metal-attention) — https://github.com/ggml-org/llama.cpp/releases/tag/b9566

Only llama.cpp had confirmed in-window activity, so the cross-repo highlight list is short by
necessity, not because the others were quiet — their commit streams were simply not observable
(see Gaps).

### Per repo

**ggml-org/llama.cpp** — only repo with confirmed in-window activity; `/releases` cache-fresh.
Per-merge build tags **b9557–b9568, all 08 Jun 14:17–21:10 UTC** (12 builds). Lens-relevant:
- b9568 `mtp: support for gemma-4 E2B and E4B assistants` (#24282) — 21:10 — models + MTP/spec-decode ⭐
- b9566 `graph: guard iswa kq_mask on its own buffer` (#24294) — 18:07 — sliding-window attn / KV mask ⭐
- b9562 `mtmd : add video input support` (#24269) — 16:41 — multimodal video; low relevance (go-mlx is text-only)

Noise (non-Metal / infra): b9567 server header-flush (#24281), b9565 + b9564 ggml-webgpu
(#24000, #24044), b9561 `sync : ggml`, b9559 cli spinner (#24283), b9558 vulkan cm2 mul_mat_id
(#23991), b9557 cuda context reset (#23935). **Partial-window caveat:** this is a single releases
page (14:17–21:10); in-window builds before 14:17 (back to ~05:04) and any after 21:10 sit on
adjacent pages not fetched.

**ml-explore/mlx** — commits not observable. No release in window; latest remains
[v0.31.2](https://github.com/ml-explore/mlx/releases/tag/v0.31.2) (22 Apr, re-confirmed fresh this run). Gap.

**ml-explore/mlx-lm** — commits not observable. No release in window; latest remains
[v0.31.3](https://github.com/ml-explore/mlx-lm/releases/tag/v0.31.3) (22 Apr, re-confirmed fresh this run). Gap.

**Blaizzy/mlx-vlm** — commits not observable. **Stale page this run** (returned v0.4.0 / 7 Mar as
"latest" — a cached pre-May snapshot); defer to the prior-verified anchor
[v0.5.0](https://github.com/Blaizzy/mlx-vlm/releases/tag/v0.5.0) (6 May), with 0.6.1 (3 Jun) a
still-unverified earlier hint. Either way predates the window. Gap.

**lmstudio-ai/mlx-engine** — commits not observable; repo publishes no GitHub releases (confirmed
fresh: "There aren't any releases here"). Ships via the LM Studio app. Gap for the window.

**vllm-project/vllm** — commits not observable (biggest blind spot; normally dozens of merges/day).
**Stale page this run** (returned v0.20.2 / 10 May as "latest" — a cached snapshot); defer to the
prior anchors v0.22.0 (29 May) / v0.22.1 (5 Jun, unverified). Either way predates the window. For
context only (NOT in window), that stale v0.20.2 note lists a DeepSeek-V4 sparse-attention MTP=1
hang fix and a gpt-oss MXFP4-under-`torch.compile` fix — relevant themes (quant, spec-decode) but
old. Gap.

### Gaps

- Atom feeds: all 18 unavailable (provenance restriction; task-file hard-code fix still pending).
- In-window commit content unknown for mlx, mlx-lm, mlx-vlm, mlx-engine and vllm.
- llama.cpp: only a single `/releases` page captured (b9557–b9568, 14:17–21:10 UTC); earlier
  in-window builds and any after 21:10 not retrieved.
- mlx-vlm and vllm `/releases` came back **stale-cached** this run (v0.4.0 / v0.20.2 shown as
  "latest"); treat the prior-verified v0.5.0 (6 May) / v0.22.0 (29 May) as the safer anchors.

---

## 2026-06-08 (11:23 UTC run) — window 2026-06-07 09:23 → 2026-06-08 11:23 UTC (~26h)

> ⚠️ **Feeds still blocked** — the 18 Atom feeds remain unreachable through `web_fetch`'s
> provenance allowlist. Re-confirmed the boundary this run: only URLs from the task message, a
> prior fetch *result*, or a WebSearch *result-link* enter the set — `.atom` URLs are not
> search-indexed, and links inside a fetched page body do **not** count (llama.cpp release-tag
> links lifted from the releasealert page were still rejected; even WebSearch prose URLs are
> rejected — only its structured result links count). The hard-code-the-18-URLs task-file fix
> from the 00:09 entry is still the right one. Browser offline (no extension connected). No
> out-of-policy fetch methods used (no curl/wget/python). Coverage below is search-derived plus a
> few server-rendered GitHub README/issue/changelog pages reached via search links; dates are
> coarse (often day-only).

### ⭐ Worth a look for go-mlx

- **llama.cpp b9549 — Gemma 4 MTP ([#23398](https://github.com/ggml-org/llama.cpp/pull/23398))**
  (7 Jun, in window). Adds multi-token-prediction / self-speculative draft heads for Gemma 4 —
  the one solidly in-window, lens-relevant merge today. Sits right on go-mlx's path: we ship
  Gemma 4 and have an MTP batched-decode kernel plan
  (`docs/plans/2026-06-07-mtp-batched-decode-kernel.md`). Worth diffing their draft-head wiring
  against ours. (models, spec-decode)
- **(watch, undated) llama.cpp NVFP4 + tensor-split ~4–5× perf regression** after the hparams
  refactor (#24060), tracked in [#24182](https://github.com/ggml-org/llama.cpp/issues/24182).
  Tied to a current refactor but not datable to the window. Flag if go-mlx ever uses their FP4
  numbers as a baseline. (quant)
- **(ecosystem, undated) TurboQuant quantised-KV-in-SDPA momentum across MLX** — open feature
  requests in mlx ([#3404](https://github.com/ml-explore/mlx/issues/3404)) and mlx-lm
  ([disc #1064](https://github.com/ml-explore/mlx-lm/discussions/1064),
  [#1060](https://github.com/ml-explore/mlx-lm/issues/1060)) plus fused-Metal-kernel POCs
  ([arozanov/turboquant-mlx](https://github.com/arozanov/turboquant-mlx)). Not merged upstream,
  but this is the exact intersection go-mlx lives in: KV/state + Metal + quant. Track as a
  candidate upstream KV-quant path. (KV/state, quant, Metal)

Inside the strict 26h window the only *confirmed* shipped activity is llama.cpp's per-merge build
stream (b9547–b9551, 7 Jun, continuing into 8 Jun). The other five repos' in-window commits were
not observable; their latest known releases all predate the window.

### Per repo

**ml-explore/mlx** — commits not observable. No release in window; latest remains
[v0.31.2](https://github.com/ml-explore/mlx/releases/tag/v0.31.2) (22 Apr). Only fresh signal is
the TurboQuant SDPA feature request [#3404](https://github.com/ml-explore/mlx/issues/3404)
(quantised KV in `mx.fast.scaled_dot_product_attention`) — an issue, not a merge. Gap.

**ml-explore/mlx-lm** — commits not observable. No release in window. Active community thread on
TurboQuant KV-cache compression (disc #1064, issue #1060, third-party PR #1067 with a fused Metal
kernel) — relevant but unmerged/unverified. Gap.

**Blaizzy/mlx-vlm** — commits not observable. No release in window. Search suggests latest =
0.6.1 (3 Jun, **unverified**; would supersede the v0.5.0/6 May seen on the 06-07 run) — either
way predates the window. Recent themes (≈early Jun): Gemma 4 MTP speculative-decoding drafter and
APC prompt caching with disk / warm-disk persistence for hybrid models — squarely go-mlx-adjacent
(persistent prompt cache ≈ our mounted-state model) but not datable to the window. Gap.

**lmstudio-ai/mlx-engine** — commits not observable; repo ships via the LM Studio app, not GitHub
releases. LM Studio changelog latest = 0.4.16 (4 Jun, outside window); the relevant mlx-engine
work landed earlier — v1.8.5 KV-cache checkpointing for long agentic contexts, v1.8.1 parallel
predictions for Qwen 3.5/3.6 + Gemma 4 (≤ 0.4.13, 13 May). Standing TurboQuant-KV request
[#296](https://github.com/lmstudio-ai/mlx-engine/issues/296) (opened 28 Mar). Gap for the window.

**ggml-org/llama.cpp** — only repo with confirmed in-window activity. Per-merge build tags
**b9547–b9551 all dated 7 Jun** (releasealert index), and the repo reports "last release ~4h ago"
so the stream continued into 8 Jun. Confirmed contents: **b9549 Gemma 4 MTP (#23398)** (highlight
above) and **b9548 vocab compatibility-check fix
([#24256](https://github.com/ggml-org/llama.cpp/pull/24256))**. b9547/b9550/b9551 titles not
retrievable (release-tag pages blocked by provenance). Day-only timestamps.

**vllm-project/vllm** — commits not observable (biggest blind spot; normally dozens of merges per
day). No release in window: search shows v0.22.0 (29 May) and a v0.22.1 (5 Jun, **search-derived,
still unverified** — the 06-07 run could only confirm v0.22.0 on a fresh page). Either predates
the window. Standing relevant capability set: NGram GPU speculative decoding (async-scheduler
compatible) and a broad quant matrix (MXFP4/NVFP4/GGUF/AWQ). Gap.

### Gaps

- Atom feeds: all 18 unavailable (provenance restriction; task-file fix still pending).
- In-window commit content unknown for mlx, mlx-lm, mlx-vlm, mlx-engine and vllm.
- llama.cpp: only b9548/b9549 contents confirmed; b9547/b9550/b9551 titles and exact UTC times
  not retrievable.
- mlx-vlm 0.6.1 and vLLM 0.22.1 are unverified search hints; treat the 06-07-verified v0.5.0 /
  v0.22.0 as the safer anchors.

---

## 2026-06-07 (07:04 UTC run) — window 2026-06-06 05:04 → 2026-06-07 07:04 UTC (~26h)

> ⚠️ **Feeds still blocked** — same `web_fetch` provenance allowlist as the two runs below;
> the hard-code-the-18-URLs fix in the 00:09 entry has not yet been applied to the task file
> and remains the right one. (Re-tested this run: URLs appearing in a *file read* do not enter
> the allowlist either — only the task message or a prior fetch result count.) Browser offline.
> No out-of-policy fetch methods used. **Upgrade on yesterday:** the llama.cpp `/releases`
> index was served cache-fresh this time, and since llama.cpp cuts one release per merged
> commit, its master stream is fully enumerable with timestamps — that repo is properly
> covered; the other five are still release-level only.

### ⭐ Worth a look for go-mlx

Quiet day — nothing actionable in the observable window (a weekend lull; only trivial
llama.cpp cleanups landed). One borderline item minutes before the window opened: llama.cpp
`context : fix off-by-one comparisons to n_gpu_layers`
([#24208](https://github.com/ggml-org/llama.cpp/pull/24208), b9537, 06 Jun 04:34 UTC) — minor
correctness fix in layer-offload logic; no go-mlx action. (serving)

### Per repo

**ml-explore/mlx** — quiet / commits not observable. No release in window; latest remains
[v0.31.2](https://github.com/ml-explore/mlx/releases/tag/v0.31.2) (22 Apr), confirmed on a
fresh releases page. Standing context while go-mlx pins mlx v0.31.1: v0.31.2 carried the Metal
split-K quantised matmul ([#3120](https://github.com/ml-explore/mlx/pull/3120)) and the SDPA
int16-overflow fix for KV sequences > 32K
([#3361](https://github.com/ml-explore/mlx/pull/3361)) — the latter matters for a
retained-state engine holding long mounted contexts. Old news, not in window.

**ml-explore/mlx-lm** — quiet / commits not observable. No release in window; latest remains
v0.31.3 (22 Apr).

**Blaizzy/mlx-vlm** — quiet / commits not observable. No release in window; latest remains
[v0.5.0](https://github.com/Blaizzy/mlx-vlm/releases/tag/v0.5.0) (6 May).

**lmstudio-ai/mlx-engine** — commits not observable; repo publishes no releases (confirmed on
a fresh releases page). Search metadata now shows "last updated **6 Jun 2026**" (was 5 Jun
yesterday), so there was likely in-window activity whose content could not be retrieved. Gap.

**ggml-org/llama.cpp** — fully enumerated via per-merge build releases. In window:
- b9542 — [`6b80c74`](https://github.com/ggml-org/llama.cpp/commit/6b80c74f285390368b3c99c5e750f19e9b096e98) —
  completion : remove useless statics ([#24226](https://github.com/ggml-org/llama.cpp/pull/24226)) — 06 Jun 10:47 UTC — noise.
- b9541 — [`588f0dc`](https://github.com/ggml-org/llama.cpp/commit/588f0dc2ce844f469797b5870e7876ddac654f6c) —
  completion : fix format specifier in LOG_INF ([#24213](https://github.com/ggml-org/llama.cpp/pull/24213)) — 06 Jun 09:54 UTC — noise.
- Just before window: b9538 `5343f45` model : rename local n_layer_all variable
  ([#24209](https://github.com/ggml-org/llama.cpp/pull/24209)) 04:56 UTC (noise); b9537
  `603300b` n_gpu_layers off-by-one fix (#24208, highlight above) 04:34 UTC.
- Caveat: tags b9539/b9540 have no release entries (likely failed CI builds), so one or two
  commits may be hidden; non-build-bumping commits (docs/CI) are invisible to this method.

**vllm-project/vllm** — commits not observable (the biggest blind spot; vLLM normally merges
dozens/day). No release in window; a **fresh** repo page shows latest =
[v0.22.0](https://github.com/vllm-project/vllm/releases), 29 May 2026 — which contradicts
yesterday's search-derived "v0.22.1" hint; treat v0.22.0 as the verified latest.

### Gaps

- Atom feeds: all 18 unavailable (provenance restriction; fix still pending in the task file).
- In-window commit content unknown for mlx, mlx-lm, mlx-vlm, mlx-engine and vllm.
- llama.cpp timestamps are release-publication times, trailing merges by minutes.

---

## 2026-06-06 (09:56 UTC run) — window 2026-06-05 07:56 → 2026-06-06 09:56 UTC (~26h)

> ⚠️ **Still a degraded run** — the 18 Atom feeds remain blocked by the `web_fetch`
> provenance allowlist (see the 00:09 entry below for the full explanation and the
> hard-code-the-URLs fix, which is still the right one). This run found a partial
> workaround — bare `/commits` and `/releases` GitHub HTML pages *do* render through
> `web_fetch` when reached via search-result links — but they are served from CDN caches
> **2 days to several weeks stale**, so the window sweep below is best-effort, not verified.
> Branch-qualified pages (`/commits/main`), Pulse, and PyPI are JS-only shells and unusable.
> Claude-in-Chrome was offline (extension not connected). No out-of-policy fetch methods used.

### ⭐ Worth a look for go-mlx

- **llama.cpp b9489 — `cuda: reserve space for quantize kv-cache at startup`
  ([#23907](https://github.com/ggml-org/llama.cpp/pull/23907))** (3 Jun, just outside window).
  Pre-allocating quantised-KV memory up front rather than on demand — directly relevant to
  go-mlx's retained-state model, where long-lived mounted KV makes fragmentation and
  late-allocation failure costlier than in replay engines. (KV/state, quant)
- **llama.cpp Gemma 4 unified hardening** (3 Jun): `mtmd: fix Gemma 4 unified FPE`
  ([#24088](https://github.com/ggml-org/llama.cpp/pull/24088)), `non-causal vision for
  gemma 4 unified` ([#24082](https://github.com/ggml-org/llama.cpp/pull/24082)), `allow skip
  build_vit()` ([#24077](https://github.com/ggml-org/llama.cpp/pull/24077)). Upstream Gemma 4
  multimodal path still shaking out bugs. (models)
- **Re-flag from mlx-lm (4 May, its newest visible commit): `Fix Gemma 4 sanitize() not
  stripping KV projections for shared layers`
  ([df1d3f3 / #1240](https://github.com/ml-explore/mlx-lm/commit/df1d3f3c9a7aae402dcbb8f41d4c36bcc13a50ae))**,
  following [#1158](https://github.com/ml-explore/mlx-lm/commit/4f5cbd2a4f8bcd2c6e702e60b1090c644e45b952)
  (unused projections on KV-shared layers). Worth cross-checking go-mlx's `gemma4.go` weight
  loading for the same shared-layer KV-projection bug family. NB: mlx-lm's #1240 is
  numerically adjacent to our own Mantis #1241 — don't cross wires when grepping. (KV/state, models)
- **vLLM v0.22.1** (recent; date unconfirmed, search-indexed ~a week ago): Mellum v2
  (JetBrains MoE code-gen), zentorch-accelerated quantised linear on AMD Zen CPUs, DeepSeek-V4
  init fix, model-loading regression fixes. (models, quant, serving)

Strictly inside the 26h window the only *confirmed* items are llama.cpp housekeeping builds —
effectively a quiet/blind day.

### Per repo

**ml-explore/mlx** — commit pages unreachable (JS-only). Latest release still
[v0.31.2](https://github.com/ml-explore/mlx/releases/tag/v0.31.2) (22 Apr). Window activity
unknown — gap.

**ml-explore/mlx-lm** — bare `/commits` page rendered (cache possibly ~1 month stale): newest
visible commit 4 May (df1d3f3, Gemma 4 KV sanitize fix, above). April was heavy on KV-cache
surface work: `ArraysCache.extend` fixes
([3cd9a52](https://github.com/ml-explore/mlx-lm/commit/3cd9a52df261edbcfd74ba8f72ca345380bb1bbd),
[a9856b4](https://github.com/ml-explore/mlx-lm/commit/a9856b485d7789ccdee1d40d4643e20a9f61f750)),
batch KV/rotating-cache extend ([62f38ae](https://github.com/ml-explore/mlx-lm/commit/62f38aeb51da77f595be7161ba7caa119ca5234a)),
`max-kv-size` back in batch generator
([d4eb136](https://github.com/ml-explore/mlx-lm/commit/d4eb136d4440439582e7c631b0e07453e04b65a3)).
Treat "quiet since 4 May" as unverified.

**Blaizzy/mlx-vlm** — commits unreachable. Latest release
[v0.5.0](https://github.com/Blaizzy/mlx-vlm/releases/tag/v0.5.0) (6 May); search snippets
mention undated recent work on thread-local generation streams, DFlash spec-decode fixes, and
Qwen3-VL / Cohere2-MoE support — cannot pin to window. Gap.

**lmstudio-ai/mlx-engine** — repo metadata shows last update **5 Jun 2026 (in window)** but the
commit content was not retrievable. No releases; 164 commits, 3 open PRs. Gap on content.

**ggml-org/llama.cpp** — confirmed in window: build b9528 tagged 5 Jun ~13:18 UTC ("UI: run npm
install when package-lock newer", #24171 — noise) and b9524 ("minor: fix lint issues" — noise).
**~30 builds (b9497–b9528) landed 3–5 Jun that could not be enumerated — worth a manual skim.**
Last enumerable day (3 Jun): b9496 Gemma 4 FPE fix; b9495 `qwen35: post-norm hidden state for
MTP` ([#24025](https://github.com/ggml-org/llama.cpp/pull/24025)); b9493/94 mtmd vision-path
changes; b9491 CUDA PDL race fix ([#24030](https://github.com/ggml-org/llama.cpp/pull/24030));
b9489 quantised-KV startup reservation (#23907, highlight above); rest noise.

**vllm-project/vllm** — commits unreachable; cached pages weeks stale (open-PR list rendered as
of ~17 Apr). v0.22.1 recent but unconfirmed for window (highlights above). Gap.

### Gaps

- Atom feeds: all 18 unavailable (provenance restriction — same root cause as the 00:09 run).
- HTML fallback is CDN-stale by days-to-weeks; in-window coverage essentially limited to
  llama.cpp tags and the mlx-engine "updated 5 Jun" signal.
- Fix remains: hard-code the 18 literal feed URLs into the task file (list in the entry below),
  or leave Claude-in-Chrome connected for scheduled runs.

---

## 2026-06-06 — window ~2026-06-04 22:09 → 2026-06-06 00:09 UTC (last ~26h)

> ⚠️ **Degraded run — Atom feeds could not be loaded.** The GitHub commit/release/tag Atom
> feeds were unreachable this run, so the per-commit detail below is **not** feed-derived.
> See "Why the feeds failed" and "Action required" at the foot of this entry. Nothing below
> should be treated as a verified commit list, and no commit hashes/PR numbers have been
> invented to fill the gap.

### ⭐ Worth a look for go-mlx

Cannot be compiled reliably this run — the feed pipeline that produces per-commit, in-window
items did not function (see below). Treating this as **"no verified actionable items"** rather
than risk surfacing fabricated or stale highlights.

The only low-confidence, search-derived hint worth flagging: `llama.cpp` cut at least one
tagged build on **5 Jun 2026** (its cadence is ~one release every few hours), so anything that
landed there — quant/k-quant, sampling, or Metal kernel work — would be the most likely place
to find something in-window. Needs the feed to confirm specifics. (KV/state, quant, Metal —
unverified.)

### Per repo

**ml-explore/mlx** — feed unavailable (fetch blocked). Verified out-of-band from the repo
landing page: latest *release* is **v0.31.2, dated 22 Apr 2026** — well outside the window, so
**no release in window**. Commit-level activity in window: unknown (feed required).

**ml-explore/mlx-lm** — feed unavailable. Search signal only: repo last updated ~**2 Jun 2026**
(outside the 26h window); PyPI still at **0.31.3 (22 Apr 2026)**. A recurring theme in recent
mlx-lm work is batch KV behaviour (e.g. defaulting to `BatchRotatingKVCache` in batch mode) —
relevant to go-mlx's KV/state surface — but **not confirmed in this window**. — quiet / unverified.

**Blaizzy/mlx-vlm** — feed unavailable. No reliable in-window signal. — unverified.

**lmstudio-ai/mlx-engine** — feed unavailable. No reliable in-window signal. — unverified.

**ggml-org/llama.cpp** — feed unavailable. Search signal only: at least one tagged build on
**5 Jun 2026** (within window); project releases roughly every few hours, so multiple commits
almost certainly landed in window. Specific titles/hashes/PRs **not verified** (feed required).
Likely-relevant areas to check once feeds work: GGUF/k-quant/imatrix, sampling, Metal kernels.

**vllm-project/vllm** — feed unavailable. Search returned inconsistent version data; no reliable
in-window signal. — unverified.

### Honest gaps

- **All six commit/release/tag Atom feeds: unavailable this run.** Not a GitHub outage — a
  sandbox constraint (below).
- Per-commit detail, exact timestamps, and short hashes/PR numbers are therefore **absent by
  design** (not fabricated).
- Release facts marked "verified" come from a successful fetch of the repo landing page; items
  marked "search signal" are fuzzy and may be stale.

### Why the feeds failed

The run is restricted to the `web_fetch` tool, which enforces a **URL-provenance allowlist**: it
will only retrieve a URL that has already appeared verbatim in the task/user message or in a
prior fetch result. The task file supplies the feed URLs as *templates*
(`https://github.com/<owner>/<repo>/commits.atom`), so the **literal** feed URLs (with real
owner/repo) never entered the allowlist, and every `*.atom` fetch returned
*"URL not in provenance set."* GitHub's Atom feed URLs are not surfaced by web search result
links or inside fetched HTML bodies (the `<link rel="alternate">` tags are stripped), so there
is no in-policy way to get them into provenance. The task forbids substituting another fetch
method (curl/wget/python/browser), so per its own fallback rule the feeds are reported as
unavailable rather than worked around.

### Action required (one-line fix for tomorrow's run)

List the **18 literal feed URLs** explicitly in the scheduled-task SKILL.md body (not as
`<owner>/<repo>` templates). Once the exact URLs appear in the task message they enter the
`web_fetch` provenance allowlist and the feed pipeline works unchanged. The URLs to hard-code:

```
https://github.com/ml-explore/mlx/commits.atom
https://github.com/ml-explore/mlx/releases.atom
https://github.com/ml-explore/mlx/tags.atom
https://github.com/ml-explore/mlx-lm/commits.atom
https://github.com/ml-explore/mlx-lm/releases.atom
https://github.com/ml-explore/mlx-lm/tags.atom
https://github.com/Blaizzy/mlx-vlm/commits.atom
https://github.com/Blaizzy/mlx-vlm/releases.atom
https://github.com/Blaizzy/mlx-vlm/tags.atom
https://github.com/lmstudio-ai/mlx-engine/commits.atom
https://github.com/lmstudio-ai/mlx-engine/releases.atom
https://github.com/lmstudio-ai/mlx-engine/tags.atom
https://github.com/ggml-org/llama.cpp/commits.atom
https://github.com/ggml-org/llama.cpp/releases.atom
https://github.com/ggml-org/llama.cpp/tags.atom
https://github.com/vllm-project/vllm/commits.atom
https://github.com/vllm-project/vllm/releases.atom
https://github.com/vllm-project/vllm/tags.atom
```

(Alternative, if you'd rather not bloat the task file: allow the run to fetch via the rendered
GitHub pages with the Claude-in-Chrome browser tool — but that contradicts the current
"web_fetch only / never substitute" rule, so the URL-listing fix above is the clean one.)

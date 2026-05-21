<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# 2026-05-20 Long-Context Gap Diagnosis

This note records the current answer to why go-mlx is still slower than
configured external runners on the accepted 100k retained workflow.

## Short Continuation Check

A current-source C006 regression check was built to
`/private/tmp/go-mlx-c006-regression/lthn-mlx` and run from `/private/tmp`
with the same C006 premise, `context=131072`, paged cache,
`prefill_chunk_size=512`, thinking enabled, and the accepted `512` visible-token
floor, but with `chapters=9`.

The run completed:

| Metric | Value |
| --- | ---: |
| Successful turns | `9/9` |
| Generated / visible tokens | `6851` |
| Total wall | `94.359181752s` |
| Average decode | `75.44102448821488 tok/s` |
| Average prefill | `2212.4547571311377 tok/s` |
| Active MLX memory | `3373521322` bytes |
| Cache memory | `6679911976` bytes |
| Process RSS | `3550920704` bytes |
| Process virtual reservation | `587977261056` bytes |
| Estimated energy at `100 W` | `9435.9181752 J` |

This does not reproduce a massive C006-path rollback. The nearby canonical
`92.814218749s` artefact was a stricter `chapter_min_tokens=640` neighbour that
reported `7` successful turns and failed on turn `8` because the model naturally
stopped at `563` visible tokens. The accepted `chapter_min_tokens=512` C006 run
completed `10/10` turns in `105.946990083s`.

## Production Gap

The slower path is the accepted 100k retained workflow, not the shorter C006
continuation lane. The first corrective change is now in the default fast lane:
hyper-long paged K/V caches use `1024`-token pages instead of the old `512`
default, and the CLI records that choice as
`GO_MLX_PAGED_KV_PAGE_SIZE=1024`. The next corrective change retains the
materialised full K/V handles produced by a full-attention owner layer so later
shared full-attention layers can reuse them instead of re-concatenating the
same paged state.

| Runner | Shape | Warm per-turn decode | First prefill | Restore |
| --- | --- | ---: | ---: | ---: |
| go-mlx current | `101005` prompt tokens, `10x1024` retained turns, paged K/V `1024`, shared full-K/V reuse | about `17.07s` per warm `1024` tokens, `60.040 tok/s` | `60.186s`, `1678.322 tok/s` | `0.368ms` average |
| go-mlx previous borrowed-page row | `101005` prompt tokens, `10x1024` retained turns, paged K/V `1024` | about `19.97s` per warm `1024` tokens, `51.310 tok/s` | `60.195s`, `1678.071 tok/s` | `0.372ms` average |
| go-mlx previous page-size row | `101005` prompt tokens, `10x1024` retained turns | about `23.4s` per `1024` tokens, `43.617 tok/s` | `157.168s`, `642.657 tok/s` | `2.116ms` average |
| llama.cpp server | `100926` prompt tokens, `10x1024` cached-prefix turns | about `12.5s` per `1024` tokens, `82.680 tok/s` | `89.122s`, `1132.450 tok/s` | `45.591ms` warm prompt work |
| `mlx_lm` | `100935` cached prompt tokens, `10x1024` turns | about `10.0s` per `1024` tokens, `103.971 tok/s` | about `18.5s`, `5465.549 tok/s` | cached prefix in-process |

The retained-state restore is already cheap enough that it is not the active
loss. The page-size correction improves the 100k row from `408.483s` to
`262.995s`, a `1.553x` wall/energy improvement. Borrowing full page handles
then improves the accepted row to `260.093s` / `51.293 tok/s`, and shared
full-K/V reuse improves it again to `231.109s` / `60.011 tok/s`. The active
loss is still the evaluated long-context graph and kernel path:

- go-mlx cold 100k prefill is now `1.48x` faster than llama.cpp but still
  `3.26x` slower than the configured `mlx_lm` harness.
- go-mlx warm 100k decode remains `1.38x` slower than llama.cpp and `1.73x`
  slower than `mlx_lm`.
- The current one-run token-phase trace records `59.957 tok/s` on the
  shared-full-K/V path. Go-side forward graph construction is only
  `1.251ms/token`; most of the wait still lands in `sample_eval` at
  `15.402ms/token`, which is where lazy MLX graph work synchronises in the
  normal run.

## Sustained Long-Turn Check

A follow-up `driver-profile` diagnostic kept the accepted `101005` token
prompt, `context=131072`, paged K/V `1024`, shared full-K/V reuse, and `12 GiB`
active/RSS guards, but raised the generation budget from `1024` to `5120`.
The prompt naturally stopped at `2489` generated/visible tokens per turn, so
this is not a true forced `5k` row. It does test a much larger real turn than
the accepted runner-anchor row.

| Metric | Value |
| --- | ---: |
| Successful runs | `10/10` |
| Generated / visible tokens | `24890` |
| Average decode | `59.94667601709725 tok/s` |
| Warm decode min / max | `59.926061615914335` / `60.00645786751182 tok/s` |
| Warm wall average | `41.525169310s` |
| Warm restore average | `0.36199ms` |
| Cold prefill | `1680.309200848654 tok/s` |
| Active MLX memory | `4000601698` bytes |
| Process RSS | `3383967744` bytes |
| Estimated energy at `100 W` | `47557.0868251 J` |

This bounds one suspected failure mode: large generated turns are not causing
decode collapse or host-memory growth on the current shared-full-K/V path. The
remaining gap is still the baseline 100k attention cost versus cached
llama.cpp/`mlx_lm`, not long-turn allocator growth. A future fairness row that
requires `5k+` visible tokens should change the prompt/task shape rather than
ignore model stop tokens.

## Working Explanation

go-mlx has the retained-prefix architecture working, and the old paged-cache
block geometry plus duplicate shared full-attention K/V materialisation were
real parts of the long-context loss. The remaining 100k decode path still
evaluates a heavier per-token MLX graph than llama.cpp or `mlx_lm`. The likely
live boundary is full-attention K/V access and mask/graph materialisation over a
very large retained context, combined with the paged-cache view/concat
attention path. The shorter C006 path stays near the useful `75-80 tok/s` band
because it does not carry a 100k prompt prefix through every generated token.

The next optimisation should target the 100k first-prefill and warm-decode
kernel path directly. Re-running small-context or short-output smokes will not
measure this boundary.

## Token-Phase Trace

A same-shape one-run trace was recorded with `GO_MLX_TRACE_FORWARD_EVAL=1` and
`driver-profile -trace-token-phases` on the accepted README-repeat 100k shape.
The raw trace is intentionally not tracked because it is about `17 MB`, but the
compact derived note is tracked at
`docs/runtime/2026-05-20-go-mlx-gemma4-e2b-4bit-100k-token-phase-trace-summary.md`.

The trace was refreshed after shared full-K/V reuse. The normal token-phase run
holds the current `60 tok/s` band, while the forced native-event variant slows
decode to `21.207 tok/s`; that variant is diagnostic rather than a replacement
for the current untraced `60.011 tok/s` row. The forced-materialisation bucket
split is still decisive: out of `48.283s` traced decode-loop time, `47.593s` is
forward materialisation. Native event totals rank attention first at `18.982s`,
then output at `10.317s`, FFN at `9.314s`, and attention residual at `7.137s`.

The expensive attention layers are exactly the full-attention owners in the
Gemma 4 local/full pattern. Shared full-K/V reuse moved later shared
full-attention layers `19`, `24`, `29`, and `34` down to about `1.03ms/token`.
Early owner layers `4`, `9`, and `14` remain near `1.96-1.98ms/token`, while
local sliding-attention layers sit near the `0.29-0.37ms` band. The next
implementation target should therefore stay focused on owner-layer
full-attention K/V work in the paged/global path, but not by simply retaining a
second MLX full-cache tensor via `slice_update`.

## Rejected 100k Branches

Nine same-shape `100k` / `1024` one-run probes now bound the obvious branches:

| Probe | Shape | Result | Verdict |
| --- | --- | ---: | --- |
| Paged K/V without fast concat | `100937` prompt tokens, paged K/V `1024`, accepted fast gates except `GO_MLX_ENABLE_PAGED_DECODE_FAST_CONCAT` | `106.324s` wall, `22.956 tok/s` decode, `1638.525 tok/s` prefill, `3.640 GiB` active MLX | Rejected. Avoiding the concat makes the per-page Go/MLX attention graph much slower than the accepted paged fast-concat lane. |
| Native C++ paged attention reduction | `100937` prompt tokens, paged K/V `1024`, accepted fast gates plus `GO_MLX_ENABLE_NATIVE_PAGED_ATTENTION`, no fast concat | `104.572s` wall, `23.448 tok/s` decode, `1660.523 tok/s` prefill, `3.640 GiB` active MLX | Rejected. Moving the same page-reduction graph behind one C++ call trims only a little overhead; the missing path is a fused/custom paged-attention kernel. |
| Native C++ paged attention without single-KV-head repeat | `100912` prompt tokens, paged K/V `1024`, accepted fast gates plus `GO_MLX_ENABLE_NATIVE_PAGED_ATTENTION`; C++23 wrapper broadcasts one-head K/V pages instead of materialising repeats | `103.696s` wall, `23.828 tok/s` decode, `1665.263 tok/s` prefill, `3.613 GiB` active MLX | Rejected. The no-repeat correction is valid and slightly better, but the page-reduction graph remains far below the accepted fast-concat path. |
| Larger `2048`-token pages | `101005` prompt tokens, paged K/V `2048`, accepted fast gates | `80.787s` wall, `49.984 tok/s` decode, `1678.261 tok/s` prefill, `3.710 GiB` active MLX | Rejected. Fewer pages do not improve the borrowed fast-concat path; cache memory rises and decode falls below the accepted `1024`-page row. |
| Preallocated `1024`-token pages | `101005` prompt tokens, paged K/V `1024`, `GO_MLX_ENABLE_PAGED_KV_PREALLOC=1`, accepted fast gates | `80.459s` wall, `50.743 tok/s` decode, `1679.677 tok/s` prefill, `3.747 GiB` active MLX | Rejected. In-place page updates do not beat the accepted concat-backed page append path at 100k and slightly increase active memory. |
| Materialised owner full K/V | `100932` prompt tokens, paged K/V `1024`, accepted fast gates plus `GO_MLX_ENABLE_PAGED_FULL_KV_MATERIALIZE=1` | `77.200s` wall, `59.855 tok/s` decode, `1682.696 tok/s` prefill, `4.385 GiB` active MLX | Rejected. Keeping a full backing tensor for the owner layers removes no visible decode cost and raises active/cache memory versus the accepted shared-full-K/V row. |
| Fixed cache with sliding layers bounded | `100937` prompt tokens, fixed Gemma 4 cache, shared mask, sliding cache bound, `12 GiB` active/RSS guards | Failed after `13` visible tokens; stream active memory hit `13748980782` bytes over the `12884901888` byte guard | Rejected. Hyper-long fixed cache is not the default path until a narrower global-only/native attention storage plan exists. |
| Right-sized fixed cache with sliding layers bounded | README repeat `46`, fixed cache size forced to `102400`, shared mask, sliding cache bound, `12 GiB` active/RSS guards | Failed after `13` visible tokens; stream active memory hit `13682988726` bytes over the `12884901888` byte guard | Rejected. Right-sizing below the full `131072` context does not bring active memory under the production guard. |
| Borrowed fixed-cache native state | README repeat `46`, fixed Gemma 4 cache, shared mask, sliding cache bound, borrowed full-capacity K/V handles for native fixed-attention paths, `12 GiB` active/RSS guards | Failed after `13` visible tokens; stream active memory hit `13660804802` bytes over the `12884901888` byte guard | Rejected. Avoiding fixed-state clones trims the obvious handle duplication but does not change the full fixed-cache attention graph footprint enough to make the branch viable. |

The current boundary is therefore narrower than "turn off concat" or "restore
fixed cache": go-mlx needs a fused native paged/global-attention path that
avoids both unnecessary full K/V rematerialisation and the active-memory
footprint of a full fixed cache. A C++ wrapper around the existing
page-reduction graph is not enough, larger page geometry does not help,
preallocated pages do not help, and a right-sized fixed cache is still too
memory-heavy on the guarded 100k lane. Borrowed fixed-state handles remove an
obvious clone path but leave the same active-memory cliff. The
materialised-owner probe also
rejects a pure MLX `slice_update` full-backing workaround; the next viable path
needs the lower-level zero-copy/fused global-attention storage shape described
in `IDEAS.md`, not another Go-orchestrated full-cache view.

## 2026-05-21 Zero-Copy / Threshold Probe

The latest probes treat `IDEAS.md` as the optimisation brief rather than a
suggestion list. The C++23/raw-byte side of the "Zero-Copy Graph Injection" is
already present in source: the raw bytes path uses Go `runtime.Pinner`, C++23
`std::mdspan`, and `mlx_array_new_data_managed_payload`/strided MLX arrays.
The new guarded paged-restore path wires that lower level into prompt-cache
restore by keeping streamed KV block pages as their incoming page arrays instead
of coalescing them into runtime-sized pages immediately.

The C++23 status is explicit: the bridge cgo flags build with `-std=gnu++23`,
the repo CMake entrypoints require C++23, `pinned_array_bridge.cpp` uses
`std::mdspan` plus multidimensional `view[i, j, k, l]` indexing for strided
view validation, and `decode_bridge.cpp` already uses `std::unreachable()` in
the exhaustive Gemma 4 native KV ownership switch. The next use of those tools
should be in the fused paged/global attention path, not scattered into cold
validation code where it cannot move decode.

| Probe | Result | Verdict |
| --- | ---: | --- |
| `context=65536`, fixed cache | `63625` prompt tokens, `46.976s` wall, `1985.425 tok/s` prefill, `68.909 tok/s` decode, `32.147s` first token, `7.175 GB` peak MLX, `5.312 GB` active MLX, `6.040 GB` MLX cache, `3.374 GB` RSS | Fixed remains faster at the threshold, but it is not the guarded 128Ki default path. |
| `context=65537`, paged fast-concat | `63625` prompt tokens, `51.053s` wall, `1970.214 tok/s` prefill, `54.847 tok/s` decode, `32.383s` first token, `7.023 GB` peak MLX, `3.942 GB` active MLX, `6.553 GB` MLX cache, `3.397 GB` RSS | A one-token cap increase flips fixed to paged and exposes the decode cliff. |
| `context=65537`, native paged attention | `74.078s` wall, `1970.895 tok/s` prefill, `24.555 tok/s` decode, `6.651 GB` MLX cache | Rejected. The current native page-list reduction is much slower than fast-concat. |
| `context=65537`, paged fast-concat plus clear-cache | `52.127s` wall, `1899.350 tok/s` prefill, `55.233 tok/s` decode, `4` bytes MLX cache, `3.369 GB` RSS | Memory hygiene only. It clears allocator cache without closing decode. |
| `context=131072`, paged fast-concat plus clear-cache | `100912` prompt tokens, `80.551s` wall, `1593.668 tok/s` prefill, `59.919 tok/s` decode, `63.463s` first token, `7.151 GB` peak MLX, `3.879 GB` active MLX, `4` bytes MLX cache, `3.368 GB` RSS | Stable memory at 128Ki, but speed remains in the current 100k band. |
| `context=65537`, typed paged K/V without query alignment | fp16 and bf16 K/V storage both land around `55.9s` wall, `1873-1877 tok/s` prefill, `46.7 tok/s` decode, and `6.832 GB` peak MLX | Rejected. Storing K/V narrower while leaving the attention query in the old dtype made SDPA slower and proved dtype alignment is part of the storage contract. |
| `context=65537`, typed paged K/V with query alignment | fp16 K/V records `44.294s` wall, `2076.372 tok/s` prefill, `75.012 tok/s` decode, `5.405 GB` peak MLX; bf16 K/V records `44.019s` wall, `2101.038 tok/s` prefill, `74.548 tok/s` decode, `5.415 GB` peak MLX | Positive cold/threshold probe. Query-aligned typed K/V beats both the paged clear-cache threshold and the `65536` fixed-cache threshold while lowering peak MLX memory. |
| `context=131072`, typed paged K/V with query alignment, one run | fp16 K/V records `68.922s` wall, `1820.807 tok/s` prefill, `75.848 tok/s` decode, `5.471 GB` peak MLX; bf16 K/V records `68.912s` wall, `1824.374 tok/s` prefill, `75.300 tok/s` decode, `5.481 GB` peak MLX | Positive cold 100k probe. It cuts peak memory versus the current shared-full-K/V row, but a one-run row is not the retained workflow acceptance measure. |
| `context=131072`, fp16 paged K/V with query alignment, 10 retained runs | `100912` prompt tokens, `240.453s` wall, `56.025 tok/s` average decode, first run `75.883 tok/s`, warm turns about `53.8 tok/s`, `5.471 GB` peak MLX, `3.467 GB` active MLX, `3.381 GB` RSS, and `4` bytes MLX cache | Rejected as the default retained workflow. It saves memory, but is slower than the accepted shared-full-K/V row at `231.109s` wall and `60.011 tok/s` average decode. |

The zero-copy stack is therefore split into three parts:

1. Raw bytes to pinned MLX arrays: implemented with Go `runtime.Pinner` and
   C++23 `std::mdspan`.
2. Restore-time paged state: now guarded by
   `GO_MLX_ENABLE_ZERO_COPY_PAGED_RESTORE=1` so incoming KV pages can be kept as
   pages instead of immediately re-coalesced.
3. Decode-time paged/global attention: still missing. The accepted 100k path
   still depends on paged fast-concat during attention, so it is streamier on
   restore than before but not yet streamy during the hot per-token attention
   path.

`GO_MLX_ENABLE_GENERATION_CLEAR_CACHE=1` and
`GO_MLX_GENERATION_CLEAR_CACHE_INTERVAL=256` are also useful, but they should be
read as allocator discipline, not throughput evidence. They keep MLX cache
memory flat during long runs and after chunked prefill, but they do not change
the underlying paged/global attention work enough to beat the current external
runner anchors.

`GO_MLX_KV_CACHE_DTYPE` is therefore kept as an explicit opt-in R&D gate. The
implementation is useful because it gives the cache layer a typed-storage
contract and exposes the query/K/V dtype alignment rule. It is not promoted into
the fast Gemma 4 defaults because the realistic retained 10-turn workflow loses
wall time and warm decode, even though the cold rows are much faster and use
less memory. The next production path still has to make the hot retained
paged/global attention path streamier rather than only narrowing stored K/V.

## Atomic-Chat Reference Notes

Atomic-Chat is useful as a reference because its Metal/Gemma 4 stack is making
the same architectural bets visible in `IDEAS.md`:

- Its MLX backend surface includes APC, warm-memory/warm-disk tiers,
  TurboQuant-style KV quantisation, and Gemma 4 MTP drafters.
- Its llama.cpp fork documents TurboQuant KV types `turbo2`, `turbo3`, and
  `turbo4`, with `turbo3` as the recommended default and a Metal TurboFlash
  decode kernel.
- Its Gemma 4 MTP design attaches the assistant to the target context instead
  of allocating a second tokenizer, context, sampler, or draft KV cache. The
  assistant reads the target K/V and uses the target's last hidden state.
- Its MLX extension maps quantised Gemma 4 targets to bf16 assistant drafters
  and treats mismatch as lower acceptance rate rather than output corruption,
  because verification stays greedy.

For go-mlx, this means TurboQuant K/V and MTP are valid follow-up R&D lanes, but
they must be labelled separately from no-draft raw decode. The immediate no-draft
gap remains the paged/global attention hot path: owner full-attention layers need
a lower-level fused or directly strided storage path, not more Go-side page
orchestration.

## Model-Native Cache Diagnostic

The obvious `mlx_lm` comparison raised one useful diagnostic branch: try the
existing `-cache-mode fp16` path, which leaves Gemma 4 closer to its model-native
`KVCache`/`RotatingKVCache` split instead of replacing everything with the
production paged cache. Before the fix, the 100k shape failed during chunked
prefill at chunk `1024:1536` with MLX's "Attempting to eval an array without a
primitive" error. Disabling last-logits prefill did not move the failure, so the
bug was cache state materialisation before detach, not logits slicing or
sampling.

`prefillTokenBlockOnce` now evaluates non-paged cache state before detaching
chunked prefill caches. Paged caches are intentionally excluded from this extra
eval so the accepted production lane does not gain a new synchronisation point.
Focused coverage is in
`TestPromptCache_EvalCachesBeforeDetachSkipsPagedCaches_Good` and
`TestPromptCache_EvalCachesBeforeDetachKeepsChunkedKVCacheEvaluable_Good`.

After that fix, the same `fp16`/rotating 100k diagnostic passed the old prefill
boundary but exposed a stronger active-memory cliff. The local E2B MLX config
declares `text_config.max_position_embeddings=131072`; this is the model's
`128Ki` context cap, not an over-context setting. The failing 100k diagnostic is
therefore under the model cap.

The current bounded ladder is:

| Shape | Result | Verdict |
| --- | ---: | --- |
| `28548` prompt tokens, `context=32768`, `fp16`/rotating | `10.886s` wall, `2631.245 tok/s` prefill, `4.702 GB` active MLX, `6.479 GB` peak MLX, `3.379 GB` RSS | Safe memory-slope row; generation stopped immediately, so it is not a decode row. |
| `52677` prompt tokens, `context=65536`, `fp16`/rotating | `24.690s` wall, `2143.889 tok/s` prefill, `43.955 tok/s` decode over two generated tokens, `6.199 GB` active MLX, `8.771 GB` peak MLX, `3.369 GB` RSS | Safe medium-context row. |
| `52677` prompt tokens, `context=131072`, `fp16`/rotating | `24.559s` wall, `2154.850 tok/s` prefill, `41.977 tok/s` decode over two generated tokens, `6.199 GB` active MLX, `8.771 GB` peak MLX, `3.383 GB` RSS | Confirms the configured context ceiling itself is not the memory cliff. |
| README repeat `36`, `context=131072`, `fp16`/rotating | failed after one visible token at `28808918294` active bytes over the `12 GiB` guard | Rejected. Active MLX memory jumps nonlinearly between about `52k` and `80k` prompt tokens. |
| Same `80k` shape with `-prefill-chunk-size 256` | failed after one visible token at `51768088226` active bytes | Rejected. Smaller prefill chunks worsen the cliff, so this is not a simple `chunk_len * key_len` scratch fix. |
| Same `80k` shape with an experimental full-attention prefill layer eval boundary | failed after one visible token at `28904937562` active bytes | Rejected and removed from source. Layer-level materialisation does not reduce the active allocator cliff. |
| README repeat `46`, `context=131072`, `fp16`/rotating | failed after one visible token at `64794744442` active bytes | Rejected. A rotating-cache copy-detach diagnostic was also byte-for-byte flat at `64794744526` active bytes and was removed from source. |

This rejects model-native `fp16`/rotating as a drop-in replacement for the paged
100k production lane. The active cliff is not caused by exceeding context, by
retained rotating-tail slices, by smaller prefill chunks, or by keeping the
whole prefill chunk graph lazy across full-attention layers. The current
optimisation target stays the paged/global-attention path: a lower-level fused
global attention or zero-copy state layout that avoids both full fixed-cache
residency and per-token page concat.

## Replay Harness

Use `scripts/gemma4_context_ramp.sh` for the next context-scaling pass. The
tracked harness now defaults to the current E2B q4 production snapshot and uses
`driver-profile -report-file` so each row is emitted by the runner rather than
by shell stdout redirection. Override `GO_MLX_MODEL` and `GO_MLX_MODEL_LABEL`
when comparing E4B, 26B, or future model snapshots.

The `5120` token-budget fairness pass has now been run at the accepted 100k
shape and is recorded as a sustained long-turn diagnostic. The next context
ladder should use a suffix that naturally demands `5k+` visible tokens if the
goal is to measure a full-budget turn rather than the model's natural stop.

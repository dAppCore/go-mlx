<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# go-mlx Agentic Memory Production Runner Goal

> **For agentic workers:** treat this file as the source of truth for the next
> go-mlx optimisation and agentic-memory lane. Implement task-by-task, keep the
> public Go API stable, and verify each performance claim with recorded command
> output.

## Goal

Make go-mlx the production Apple Silicon runtime for LTHN agentic workflows:

- Build and ship the `lthn-mlx` binary for the app, CLI, and server bundle.
- Wake a model from durable project/operator memory without replaying the whole
  prompt into the model.
- Reload with new runtime settings when compatibility allows it, or fall back to
  summary-plus-new-window when it does not.
- Compact an agent context into a new state file when the operator wants exact
  continuation, or into text memory when portability is more important.
- Support Gemma 4 plus the Qwen 2, Qwen 3, and Qwen 3.6 families through the
  same driver-facing contracts.
- Prove go-mlx is the best practical Apple Silicon runner for repeated agentic
  workflows. Raw decode should stay close enough to the fastest comparable
  runner that the delta is not user-visible, but the primary production metric
  is 10+ turn wall-clock time with retained state, restore cost, prefill
  avoided, estimated energy delta, and effective throughput clearly reported.
- Treat opencode-sized sessions as the primary interactive target: roughly
  `30k`-`40k` tokens on first wake, followed by retained append/generate turns.
  The `100k` lane remains a stress ceiling and degradation probe, not the normal
  pass/fail shape for day-to-day agent work.

## Current Status: Active Parity Gap; Production Path Not Yet Accepted

The current q4 retained-State lane works, but the production benchmark lane is
not accepted. The production path is paged retained State with no fixed-cache
default and no arbitrary context-family switch. Do not reintroduce a
context-length cutoff to choose K/V behaviour, fixed-cache sizing, or benchmark
acceptance. Historical threshold rows are archive evidence only. Likewise, do
not use older partial retained lanes as the default benchmark target. Runnable
harness defaults should use the production `100k` stress target or the model
context window, with shorter rows labelled as smoke or archive evidence.
Code correction, 2026-05-25: the active CLI regression suite no longer carries
the archived threshold value as a named context case or script guard. Guards
should assert the invariant directly: paged retained State, no fixed cache, and
no context-derived cache-family switch.
Code correction, 2026-05-24: profile commands no longer call a
`disableGemma4FixedCacheRuntimeGates` shim. Fixed-cache and fixed-wide
diagnostic env names are ignored as ambient profile input unless an explicit
in-process override sets them, so the production path does not touch the old
fixed-cache family at all.
Fresh 2026-05-24 evidence shows a real decode recovery, but go-mlx is still
behind llama.cpp on raw decode. The retained workflow wall-time comparison is
useful, but must be read with visible output counts, output-quality flags, and
memory figures beside the speed numbers rather than using any one metric as a
rescue. The old llama.cpp control-channel leakage remains relevant to
historical rows, but the current request-context comparator below no longer
leaks visible control markers.

Latest request-context parity row, 2026-05-24:
`/private/tmp/go-mlx-goal/reports/2026-05-24-state-ramp-request-context-sharedkv-move-go-mlx-gemma4-e2b-4bit-opencode-30k-r10-g1024.json`
and
`/private/tmp/go-mlx-goal/reports/2026-05-24-llamacpp-request-context-memory-gemma4-e2b-q4km-opencode-30k-r10-g1024.json`
use the same `30k` seed, `10` retained request-context turns, `1024`
max-token budget, Gemma 4 stop strings, `temperature=1.0`, `top_p=0.95`, and
`top_k=64`. go-mlx completes `10/10` turns, reaches `48712` live tokens,
generates `4292` visible tokens, records `71.334s` retained wall, `84.633`
raw decode tok/s, `72.744` effective turn tok/s, `3.054x` retained-vs-replay
speedup, `7.133 kJ` estimated energy at `100 W`, `9.947 GB`
active-plus-cache, `3.153 GiB` RSS, and `568.218 GiB` process virtual
reservation with no output-quality flags. This row includes the same-forward
shared-KV ownership move, replacing the previous owner-layer clone into
`intermediates` with a move so shared Gemma 4 layers consume the exact same
K/V handles during the current token. Against the previous clone-based
request-context row, the same output count improves raw decode by `0.751%`,
effective turn throughput by `0.654%`, wall by `0.549%`, and estimated energy
by `39.391 J` at `100 W`. The memory-capable llama.cpp
Q4_K_M anchor completes `10/10`, reaches `50037` live tokens, generates
`5617` tokens / `5607` visible tokens, records `72.915s` wall, `109.997`
raw decode tok/s from llama.cpp timings, `76.898` wall-visible tok/s,
`7.291 kJ`, `4.331 GiB` RSS, and `427.141 GiB` virtual, with no control-marker
leak but one `visible_prompt_analysis` flag on turn 1. Interpretation: go-mlx
is `1.581s` / `2.17%` faster on wall and estimated energy in this single
same-shape pair and uses less RSS, but llama.cpp is still `1.300x` faster on
raw decode and returns more visible content in roughly the same wall time.
This is useful retained-State evidence, not production acceptance.

Fresh seeded request-context refresh after retiring the 70k default,
2026-05-24:
`/private/tmp/go-mlx-goal/reports/2026-05-24-state-ramp-request-context-100k-seed240524-go-mlx-gemma4-e2b-4bit-opencode-30k-r10-g1024.json`
and
`/private/tmp/go-mlx-goal/reports/2026-05-24-llamacpp-request-context-100k-seed240524-gemma4-e2b-q4km-opencode-30k-r10-g1024.json`
use the same opencode request-context fixture, `30k` seed, `10` turns,
`1024` max-token budget, `seed=240524`, Gemma 4 thinking prompt, Gemma 4 stop
strings, `temperature=1.0`, `top_p=0.95`, `top_k=64`, and target `100000`.
The real request-context material only grows the live state to `49153` tokens
on the go-mlx row and `54616` on the llama.cpp row after ten turns, so this is
the primary interactive 10-turn comparison, not the 100k stress proof. go-mlx
completes `10/10` turns, generates `4733` visible tokens, records `74.732s`
wall, `87.420` raw decode tok/s, `75.821` effective turn tok/s,
`2.957x` retained-vs-replay speedup, `7.473 kJ`, `9.548 GiB`
active-plus-cache, `3.156 GiB` RSS, and `573.604 GiB` virtual memory, with
`fixed_caches=0`, `paged_caches=15`, `max_local_capacity=512`,
`max_global_capacity=131072`, and `local_window_leaked=false`. llama.cpp
Q4_K_M completes `10/10`, generates `10196` predicted tokens but only `5613`
visible tokens, records `118.432s` wall, `105.988` raw decode tok/s,
`47.394` visible wall tok/s, `11.843 kJ`, `4.736 GiB` RSS, `427.515 GiB`
virtual memory, and no output-quality flags or visible control markers. The
important reading is split: go-mlx is `1.585x` faster on wall/energy and
`1.336x` faster on total visible-token wall throughput for the same retained
workflow, but llama.cpp is still `1.212x` faster on raw decode. The raw decode
gap remains a real optimisation target; the retained-State wall win should not
be used to hide it.

Fresh 100k retained-State stress proof, 2026-05-24:
`/private/tmp/go-mlx-goal/reports/2026-05-24-state-ramp-request-context-to100k-seed240524-go-mlx-gemma4-e2b-4bit-opencode-30k-g1024.json`
removes the turn cap and lets the same request-context fixture repeat until
the live state crosses `100000` tokens. It completes `41/41` turns without
failure, reaches `100205` live tokens, appends `58786` tokens, generates
`11337` visible tokens, records `200.882s` wall, `78.251` raw decode tok/s,
`60.075` effective turn tok/s, `3.348` minutes retained wall versus a
`24.588` minute replay estimate, `7.344x` retained-vs-replay speedup, and
`127.443 kJ` estimated energy saved at `100 W`. The final cache profile still
shows paged/no-fixed state with `max_local_capacity=512`,
`max_global_tokens=100203`, `max_global_capacity=131072`, `fixed_caches=0`,
`paged_caches=15`, and `local_window_leaked=false`. Memory stays bounded in
resident terms at `3.158 GiB` RSS and `9.548 GiB` active-plus-cache, while
virtual reservation grows to `960.783 GiB`; treat that virtual reservation as
the next memory-accounting item to watch, not as proof of active RAM growth.
There is one `visible_prompt_analysis` output issue, so the row is a strong
state/memory proof and replay-savings proof, but not final production
acceptance.

Current no-cutoff paged-State correction, 2026-05-24: fixed Gemma 4 K/V is no
longer a default fast-lane gate. `driver-profile`, `chapter-profile`, and
`state-ramp-profile` now stay on paged K/V by default, and
`state-ramp-profile` no longer synthesises
`GO_MLX_FIXED_GEMMA4_CACHE_SIZE`; the profile and bench harnesses now block the
fixed-cache gates rather than offering a diagnostic shortcut back onto that
path. The rebuilt smoke
`/private/tmp/go-mlx-goal/reports/2026-05-24-state-ramp-smoke-paged-no-fixed-default.json`
records runtime gates `GO_MLX_ENABLE_ASYNC_DECODE_PREFETCH=1`,
`GO_MLX_ENABLE_DIRECT_GREEDY_TOKEN=1`,
`GO_MLX_ENABLE_EXPERT_ID_FUSED_ACTIVATION=1`,
`GO_MLX_ENABLE_EXPERT_ID_MATVEC=1`,
`GO_MLX_ENABLE_GENERATION_STREAM=1`,
`GO_MLX_ENABLE_NATIVE_GEMMA4_ROUTER_MATVEC=1`,
`GO_MLX_ENABLE_NATIVE_GEMMA4_ROUTER_TOPK=1`,
`GO_MLX_ENABLE_NATIVE_LINEAR_MATVEC=1`,
`GO_MLX_ENABLE_NATIVE_MLP_MATVEC=1`,
`GO_MLX_ENABLE_PAGED_DECODE_FAST_CONCAT=1`,
`GO_MLX_ENABLE_SORTED_EXPERT_PREFILL=1`, and
`GO_MLX_KV_CACHE_DTYPE=fp16`, with no `GO_MLX_ENABLE_FIXED_GEMMA4_*` gates and
no `GO_MLX_FIXED_GEMMA4_CACHE_SIZE`. Its cache profile records
`paged_caches=15`, `fixed_caches=0`, `max_local_tokens=512`,
`max_local_capacity=512`, `max_global_tokens=3298`,
`max_global_capacity=32768`, and `local_window_leaked=false`; short smoke
decode is `110.531 tok/s`. This is a default-path correction, not production
acceptance, and the next real comparator run must use this paged-only default.
Follow-up cutoff correction: `state-ramp-profile` no longer treats an unarmed
compaction threshold as the live-token stop condition. The benchmark target now
drives retained turn growth unless a fold store is configured, so a stale or
diagnostic threshold cannot truncate K/V at an arbitrary context boundary.
Overflow compaction still stops at the configured threshold when a fold store is
present, preserving the operator-driven compact path without making it a
benchmark default.
The first full request-context retry after this correction wrote
`/private/tmp/go-mlx-goal/reports/2026-05-24-state-ramp-request-context-default-paged-drainfix-go-mlx-gemma4-e2b-4bit-opencode-30k-r10-g1024.json`
but did not produce timing evidence because `metal.LoadAndInit` reported
`mlx: no usable Metal device available`; keep it as a gate-selection/error
record only. The failure was reproduced only under the sandboxed `env GOWORK=...`
or generic `env GO*=...` launch shape; the built runtime binary does not need
Go tool workspace variables, and the Codex benchmark lane should launch it with
`MLX_METALLIB_PATH` only so the process keeps native Metal access. The corrected
smoke
`/private/tmp/go-mlx-goal/reports/2026-05-24-state-ramp-smoke-paged-after-budget-removal-mlxenvonly.json`
records `paged_caches=15`, `fixed_caches=0`, `local_window_leaked=false`, and
`114.939 tok/s` decode.

Follow-up sticky-env guard, 2026-05-24: the profile/bench harness now actively
writes runtime `0` overrides for `GO_MLX_ENABLE_FIXED_GEMMA4_CACHE`,
`GO_MLX_ENABLE_FIXED_GEMMA4_SLIDING_CACHE_BOUND`,
`GO_MLX_ENABLE_FIXED_GEMMA4_SHARED_MASK`,
`GO_MLX_ENABLE_NATIVE_FIXED_SLIDING_ATTENTION`, and
`GO_MLX_FIXED_GEMMA4_CACHE_SIZE` for `driver-profile`, `state-ramp-profile`,
`state-wake-profile`, `chapter-profile`, and `bench`, including when
`-fast-gemma4-lane=false`; the same block covers the fixed-owner/model-greedy
native diagnostics and fixed wide-attention env gates. The old
`driver-profile` fixed-cache and fixed-owner flags are rejected instead of
acting as diagnostics. The native fixed Gemma 4 helpers also
let runtime `0` override package-init env values, so a sticky shell env can no
longer silently turn a paged production run back into the old fixed-cache
threshold path.
Regression coverage:
`go test ./go/internal/metal -run 'TestRuntimeGate_FixedGemma4ZeroOverrideWins|TestSample_(NewSamplerWithSuppression|NewSamplerWithSuppressionBeforeTopPTopK|SuppressTokenLogits|SuppressTokenLogitsThenTopPTopK|SuppressionGuard)'`,
`go test ./go/cmd/mlx -run 'TestRunCommand_(DriverProfileFastGemma4LaneCanDisable|DriverProfileGemma4DecodeGateFlags|DriverProfileRejectsFixedCacheFlags|DriverProfileFastGemma4LaneIgnoresFixedCacheEnv|StateRampProfileFastLaneIgnoresFixedCacheEnv)'`,
and `go test ./go/internal/metal ./go/cmd/mlx ./go` all pass. The related
suppress-token sampler cache benchmark records
`BenchmarkSampler_TopKThenTopPWithSuppression_Vocab262k` at `3 allocs/op` and
about `27 B/op`, down from the prior suppress-path `5 allocs/op` / `139 B/op`
shape.

Latest paged/no-fixed request-context row after removing hidden fixed-budget
synthesis, 2026-05-24:
`/private/tmp/go-mlx-goal/reports/2026-05-24-state-ramp-request-context-default-paged-after-budget-removal-go-mlx-gemma4-e2b-4bit-opencode-30k-r10-g1024.json`
uses the same `30k` seed, `10` request-context turns, `1024` max-token budget,
Gemma 4 stops, and `temperature=1.0`, `top_p=0.95`, `top_k=64` as the
llama.cpp anchor above. The run records no fixed Gemma 4 gates, no
`GO_MLX_FIXED_GEMMA4_CACHE_SIZE`, `cache_mode=paged`, `context_length=131072`,
`prefill_chunk_size=512`, and `GO_MLX_KV_CACHE_DTYPE=fp16`. It completes
`10/10` turns, reaches `48380` live tokens, generates `3960` visible tokens,
records `64.929s` retained wall, `88.001` raw decode tok/s, `75.103`
effective turn tok/s, `2458.685 tok/s` first prefill, `1864.735 tok/s`
average append/prefill, `3.219x` retained-vs-replay speedup estimate,
`6492.909 J` at `100 W`, `9.711 GB` active-plus-cache, `3.153 GiB` RSS, and
`507.388 GiB` virtual reservation. Cache profile stays bounded at
`paged_caches=15`, `fixed_caches=0`, `max_local_tokens=512`,
`max_local_capacity=512`, `max_global_tokens=32768`, and
`local_window_leaked=false`, with no output-quality flags. Against the same
llama.cpp Q4_K_M request-context anchor, go-mlx is `7.986s` / `10.95%` faster
on wall and estimated energy and uses `1.178 GiB` less RSS, but llama.cpp is
still `1.250x` faster on raw decode and returns `5607` visible tokens versus
go-mlx's `3960`. Effective visible turn throughput is close but still behind:
`75.103` versus llama.cpp's `76.898` wall-visible tok/s (`2.33%` gap). This is
the current production-path evidence row, not final acceptance.

Context planning correction, 2026-05-24: the row above still exposed a hidden
planner clamp. `WithContextLength(131072)` used the same value as the package
default, so the auto memory plan could silently restore the actual Metal K/V
cache cap to the planner's `32768` row while the CLI load report still printed
`131072`. `WithContextLength` now marks the context as explicit, and
`applyMemoryPlanToLoadConfig` only clamps implicit defaults. The smoke report
`/private/tmp/go-mlx-goal/reports/2026-05-24-state-ramp-context-explicit-smoke.json`
confirms `max_global_capacity=131072`, `max_local_capacity=512`, no fixed
caches, and `local_window_leaked=false`. The short request-context trace
`/private/tmp/go-mlx-goal/reports/2026-05-24-state-ramp-request-context-explicit-context-turn2-go-mlx-gemma4-e2b-4bit-opencode-30k-r2-g1024.json`
crosses the old `32768` cap and records `2/2` turns, `33728` final live tokens,
`1069` generated/visible tokens, `88.085` raw decode tok/s, `78.883` effective
turn tok/s, `9.711 GB` active-plus-cache, `3.151 GiB` RSS,
`max_global_tokens=33726`, and `max_global_capacity=131072`. This removes the
hidden context cutoff; it does not close the llama.cpp raw-decode gap.

Trace attribution update, 2026-05-24: `TraceTokenPhases` originally split async
prefetch into diagnostic `prefetch_logits` and `prefetch_cache` buckets while
leaving the production, non-trace prefetch path as one combined call. The smoke
report
`/private/tmp/go-mlx-goal/reports/2026-05-24-trace-prefetch-split-smoke.json`
keeps the fast lane paged (`fixed_caches=0`, `paged_caches=15`,
`local_window_leaked=false`, `context_length=4096`) and records
`prefetch_logits` as effectively the whole prefetch cost (`16.597 ms` of
`16.618 ms` across three non-final tokens), with dirty-cache prefetch only
`9.124 us`. That rules out the dirty K/V handoff as the current decode
bottleneck and keeps the next optimisation pointed at logits/forward graph
materialisation, not any archived context-cutoff or fixed-cache lane. Superseding
correction, 2026-05-25: the default trace path now uses the same combined
`EvalAsync(logits + dirty K/V)` boundary as production generation, so timing
rows no longer measure a split graph shape. The split helper remains only as an
internal diagnostic. Focused bench evidence records
`BenchmarkAsyncDecodePrefetchTrace_CombinedDirtyKV` at `179966 ns/op`,
`513 B/op`, and `1 alloc/op`, versus the diagnostic split row at
`162819 ns/op`, `560 B/op`, and `3 allocs/op`; this is a fidelity correction
rather than a speed claim. The same opencode request-context two-turn trace
`/private/tmp/go-mlx-goal/reports/2026-05-25-state-ramp-request-context-production-trace-prefetch-opencode-turn2-go-mlx-gemma4-e2b-4bit-opencode-30k-r2-g1024.json`
uses the real opencode seed and records `2/2` turns, `33825` final live tokens,
`1166` generated/visible tokens, `91.608` raw decode tok/s, `82.494` effective
turn tok/s, `9.861 GB` active-plus-cache, `3.404 GB` RSS, `518.254 GB`
virtual reservation, `fixed_caches=0`, `paged_caches=15`,
`max_local_capacity=512`, and `local_window_leaked=false`. Its token phases
show production-shaped `prefetch` at `6.093 ms/token`, `sample_eval` at
`3.398 ms/token`, and `forward` at `1.394 ms/token`; `prefetch_cache` is no
longer separately reported on the default trace because separating it changes
the eval boundary being benchmarked.

Empty SDPA handle cleanup, 2026-05-25: absent mask/sink inputs now pass the
zero-value `mlx_array` handle instead of allocating and freeing empty native
handles on every unmasked attention call. Focused attention tests pass, and the
same production-shaped two-turn trace at
`/private/tmp/go-mlx-goal/reports/2026-05-25-state-ramp-request-context-zero-empty-sdpa-opencode-turn2-go-mlx-gemma4-e2b-4bit-opencode-30k-r2-g1024.json`
records `2/2` turns, `33825` final live tokens, `1166` generated/visible
tokens, `91.599` raw decode tok/s, `82.476` effective turn tok/s, `9.861 GB`
active-plus-cache, `3.401 GB` RSS, `fixed_caches=0`, `paged_caches=15`,
`max_local_capacity=512`, and `local_window_leaked=false`. This is retained as
a small native-handle cleanup only: `prefetch` moves from `6.093` to
`6.073 ms/token`, while `sample_eval` moves from `3.398` to
`3.413 ms/token`, so it is not a decode-parity claim. The next useful target
remains fused logits/materialisation or sampler/eval boundary work.

Concat parent-slice cleanup, 2026-05-25: `Concatenate` no longer builds a Go
`inputs` slice for `newArray`, because `newArray` no longer stores parent
references and MLX owns the graph edges through the native op handles. Focused
Metal benches moved `BenchmarkPromptCache_KVConcat_16Pages_256Each` from
`128 B/op` and `1 alloc/op` to `0 B/op` and `0 allocs/op`; the paged
fast-concat K+V benches moved from `2 allocs/op` (`128 B/op` at 8 pages and
`256 B/op` at 16 pages) to `0 B/op` and `0 allocs/op`. The timing stayed within
run noise, so this is a retained hot-path allocation cleanup, not a claim that
the owner-layer full-attention materialisation gap is closed.

Eval-vector cgo-boundary cleanup, 2026-05-25: `Eval` and `EvalAsync` now build
the MLX output vector through one native handoff from a pooled handle buffer
instead of calling `mlx_vector_array_append_value` once per output from Go. This
keeps the production `EvalAsync(logits + dirty K/V)` boundary intact while
removing per-output cgo calls. A stack-backed variant was rejected because cgo
forced the handle buffer to escape and regressed the sampler/prefetch
allocation profile. The retained pooled version keeps allocations flat:
`BenchmarkAsyncDecodePrefetchTrace_CombinedDirtyKV` moves from the pre-change
`160.024-179.131 us/op`, `512 B/op`, `1 alloc/op` band to
`164.487-165.937 us/op`, `513 B/op`, `1 alloc/op`; the Gemma-sized sampler
bench remains effectively neutral at `483.996-506.989 us/op`, `10-11 B/op`,
`1 alloc/op`. This is a cgo-boundary cleanup only; the next larger target
remains logits/materialisation fusion.

Prefetch benchmark-shape correction, 2026-05-25: the focused async prefetch
bench now keeps the cache slice outside the hot loop and adds a production
non-trace row beside the trace rows. The corrected Metal run
(`go test ./go/internal/metal -run '^$' -bench
'BenchmarkAsyncDecodePrefetch(_|Trace_)(CombinedDirtyKV|SplitDirtyKV)$'
-benchmem -benchtime=700ms`) records
`BenchmarkAsyncDecodePrefetch_CombinedDirtyKV` at `177.954 us/op`,
`512 B/op`, `1 alloc/op`; trace combined at `175.221 us/op`, `512 B/op`,
`1 alloc/op`; and trace split at `184.888 us/op`, `560 B/op`, `3 allocs/op`.
An internal slice-only `EvalAsync`/prefetch patch was rejected before commit:
the same combined trace row moved from `173.397 us/op` to `176.224 us/op` with
the same `512 B/op`, `1 alloc/op`. Interpretation: the remaining allocation is
not the benchmark cache-slice shape or the internal prefetch varargs hop; keep
the next optimisation aimed at the larger MLX logits/materialisation boundary.

Compiled sampler boundary cleanup, 2026-05-25: `CompiledFunc.CallOne` now
collapses one-input/one-output compiled closure invocation into a single C
helper that builds the input vector from a C-stack array, applies the closure,
checks the one-output contract, extracts the output handle, and frees both MLX
vectors before returning to Go. This preserves the public Go API while removing
the per-call Go-side `mlx_vector_array_new` / append / size / get sequence from
the compiled sampler path. The focused Metal bench moved
`BenchmarkSampler_CompiledTopKThenTopPCallOne_Vocab262k` from `496.546 us/op`,
`8 B/op`, `1 alloc/op` to `450.085 us/op`, `0 B/op`, `0 allocs/op`.
The production-shaped suppressed rows moved from the latest pre-change refresh
(`516.694`, `517.472`, `515.892`, and `532.456 us/op`, `16-17 B/op`,
`2 allocs/op`) to `486.107`, `483.077`, `475.959`, and `479.901 us/op`,
`7-8 B/op`, `1 alloc/op`. This is a real sampler/materialisation boundary
cleanup, but it is still a focused benchmark result; the next retained
request-context run must prove the wall-clock effect before treating it as a
parity milestone.
Retained proof: rebuilt `lthn-mlx` and reran the same full-output
request-context fixture at
`/private/tmp/go-mlx-goal/reports/2026-05-25-state-ramp-request-context-callone-helper-go-mlx-gemma4-e2b-4bit-opencode-30k-r10-g1024.json`.
The run keeps the exact comparator output shape (`10/10` turns, `48896` final
live tokens, `14400` appended tokens, `4476` generated/visible tokens, no
output issues) and the production cache invariants (`fixed_caches=0`,
`paged_caches=15`, `max_local_capacity=512`, `max_global_capacity=131072`,
`local_window_leaked=false`). Raw decode moves from the prior compiled-sampler
row's `87.48313854487908 tok/s` to `87.68683896696935 tok/s` (`+0.233%`);
effective turn throughput moves from `75.25731884731685` to
`75.38439382823918 tok/s` (`+0.169%`); wall drops only `16.075 ms` to
`71.710519835s`; estimated energy drops by `1.607 J` at `100 W`. Token phases
show the expected local effect (`sample_eval` down from `3.305ms/token` to
`3.274ms/token` and `forward` down from `1.402ms/token` to `1.361ms/token`),
while `prefetch_logits` remains dominant at `6.726ms/token`. Count this as an
accepted sampler-boundary cleanup, not a closed parity gate.

Concat2 boundary cleanup, 2026-05-25: the two-array `concatenate2` helper now
builds the temporary MLX vector on the C stack in one helper call instead of
crossing cgo for vector create, two appends, concatenate, and vector free. This
preserves the same MLX concatenate graph and is useful for token append, page
merge, and several prompt-cache/state edges. Focused Metal benches stayed
allocation-neutral and moved the 16-page fast-concat mixed-query row's median
from about `627.381 us/op` to `601.880 us/op`; the 16-page prompt-cache concat
median moved from about `238.422 us/op` to `236.052 us/op`. A broader multi-page
`mlx_vector_array_new_data` attempt was rejected before commit because passing a
Go handle array to C made it escape, regressing the same rows to `1152 B/op` and
`2305-2308 B/op`. Keep multi-page concat on the existing append-vector path until
there is a C-side page-list owner that avoids Go handle-array escape entirely.
Follow-up scalar page-list helpers with 64 and 32 C-side slots were also tested
and reverted. They preserved `0 allocs/op` and improved pure prompt-cache concat,
but the actual fast-concat SDPA rows were neutral-to-negative; the 32-slot helper
left the 16-page mixed-query fast-concat median around `623.972 us/op` versus the
accepted two-array helper's `601.880 us/op` row. Do not promote prompt-cache-only
concat wins into the retained decode path unless the SDPA fast-concat row moves
with it.

Dirty paged-State marker cleanup, 2026-05-25: `PagedKVCache` now marks the
two dirty K/V arrays with a fixed pair helper instead of routing the per-token
paged update through a variadic helper. This keeps the same dirty-state
dedupe/overflow semantics and removes the now-unused variadic path. Focused
Metal verification passed
`TestPagedKVCache_AppendDirtyStateOnlyRecentPage_Good`,
`TestPagedKVCache_BorrowedPageStateAvoidsFullPageClones_Good`, and
`TestPagedKVCache_SlidingWindowStaysSinglePage_Good`. The retained hot-path
bench remains allocation-stable while nudging
`BenchmarkPagedKVCache_UpdateBorrowedPages_To128` from the sweep's
`1129903 ns/op`, `43 B/op`, `5 allocs/op` to repeated rows around
`1072846-1077538 ns/op`, `44 B/op`, `5 allocs/op`. Treat this as small
graph-construction hygiene on the accepted paged State path, not raw-decode
parity closure.

Decode continuation input cleanup, 2026-05-25: single-token continuation paths
now construct the `[1,1]` int32 input array directly with a C-inline
`fromSingleInt32Matrix` helper instead of building a rank-1 token array and
reshaping it. This removes one reshape graph node from `Model.Generate`,
retained `ModelSession.Generate`, exact prompt-cache replay, split continuation,
and Gemma 4 assistant draft/verify continuation without changing K/V policy,
sampler ordering, or paged-State semantics. Focused verification:
`go test ./go/internal/metal -run
'TestArray_FromSingleInt32Matrix_Good|TestModel_Generate_TraceTokenPhases_Good|TestModelSession_Generate_TraceTokenPhases_Good'
-count=1` and `go test ./go/internal/metal -run
'TestPromptCache_(MatchesExactNoLogitsByReplayingFinalToken_Good|RestoreFromKVBlocksZeroCopyPagedRestore_Good)|TestGemma4AssistantDecode_(DraftStep_Good|VerifyDraftBlock_Good)|TestGemma4AssistantGenerate_ReplaysLastTokenForKVOnlyPromptCache_Good|TestSplit_Qwen3SplitPrefillAndAttention_Good'
-count=1`. Hot-path check:
`BenchmarkFromSingleInt32_Reshape2_1x1` reports about `745-760 ns/op`,
`8 B/op`, and `1 alloc/op`; `BenchmarkFromSingleInt32Matrix` reports about
`310-319 ns/op`, `0 B/op`, and `0 allocs/op`. This is a contained handover-safe
decode-construction cleanup, not a new external-runner parity row.

Rejected adjacent probes, 2026-05-25: two superficially similar cleanups were
tested and reverted. First, passing a zero-value random key handle to
`mlx_random_categorical`/`mlx_random_uniform` is correct in focused tests, but
the matched request-context trace
`/private/tmp/go-mlx-goal/reports/2026-05-25-state-ramp-request-context-zero-random-key-opencode-turn2-go-mlx-gemma4-e2b-4bit-opencode-30k-r2-g1024.json`
regressed to `90.113` raw decode tok/s and `81.232` effective turn tok/s, with
`prefetch` at `6.190 ms/token` and `forward` at `1.449 ms/token`, so the random
key path keeps the explicit empty key handle. Follow-up direct bench coverage
now records `BenchmarkRandomCategorical_Vocab32k` and
`BenchmarkRandomCategorical_Vocab262k`; the local wrapper-only zero-key rows
were slightly faster, but the retained request-context regression remains the
production decision, so this benchmark is attribution only. Second, yielding retained-session
tokens after state advance but before async prefetch improved the first-token
field (`7.49 ms` on turn 1) but regressed the real throughput in
`/private/tmp/go-mlx-goal/reports/2026-05-25-state-ramp-request-context-yield-before-prefetch-opencode-turn2-go-mlx-gemma4-e2b-4bit-opencode-30k-r2-g1024.json`
to `88.045` raw decode tok/s and `79.482` effective turn tok/s, with
`prefetch` at `6.350 ms/token`. Keep prefetch before the stream callback unless
a future change preserves the current decode band.

Follow-up trace attribution, 2026-05-24: native event capture is now armed by
`-trace-token-phases` without requiring a `GO_MLX_*` environment variable. The
expensive forced-eval trace remains behind `GO_MLX_TRACE_FORWARD_EVAL=1`, but
normal token tracing can now record lightweight paged K/V concat events. Gemma 4
multi-page decode emits `paged_kv.fast_concat.global`,
`paged_kv.fast_concat.local`, or `paged_kv.contiguous.*` events with duration,
page count, and token count, and the profile summaries carry `max_pages` and
`max_tokens` for native event buckets. The next 100k boundary trace should use
that evidence to decide whether the fast-concat view construction or its later
lazy materialisation is the decode gap. The smoke report
`/private/tmp/go-mlx-goal/reports/2026-05-24-paged-concat-trace-smoke-state-ramp-gemma4-e2b-4bit.json`
proves the JSON surface: a 4-token retained turn records `95.495 tok/s`,
`prefetch_logits=8.221 ms` on the first token, `fixed_caches=0`, and native
event summaries for `paged_kv.fast_concat.local` (`max_pages=2`,
`max_tokens=512`) and `paged_kv.fast_concat.global` (`max_pages=2`,
`max_tokens=1568`).
Negative trace result, same date: disabling local-window fast concat and routing
local multi-page decode through `ScaledDotProductAttentionPaged` removed
`paged_kv.fast_concat.local` from the trace, but it was slower and did not
improve memory at the `100k` boundary. The report
`/private/tmp/go-mlx-goal/reports/2026-05-24-state-ramp-request-context-100k-boundary-global-fastconcat-only-seed240524-go-mlx-gemma4-e2b-4bit-g1024.json`
recorded `55.059 tok/s` raw decode versus the previous `63.247 tok/s`, with
`prefetch_logits` rising to `12.487 ms/token`. Keep local fast concat in the
current paged path; the next decode work should stay at the logits/materialise
boundary or a fused native paged-attention path, not a local concat removal.
Two related gate probes were rejected before changing defaults. First,
`GO_MLX_ENABLE_NATIVE_PAGED_ATTENTION=1` looked useful in microbenchmarks
(`BenchmarkNativePagedSingleToken_8Pages_Page256` around `339 us/op` versus
`BenchmarkSDPAPaged_8Pages_Page256_Q1_D128` around `409 us/op`), but the real
30k retained turn regressed to `42.745 tok/s` in
`/private/tmp/go-mlx-goal/reports/2026-05-24-state-ramp-native-paged-attention-enabled-seed240524-go-mlx-gemma4-e2b-4bit.json`
because `prefetch_logits` rose to `18.550 ms/token`. Second, forcing the
last-token logits path for single-token cached decode helped the one-turn smoke
slightly (`90.922 tok/s` default experiment versus `89.801 tok/s` disabled),
but the 10-turn request-context control was neutral to slightly worse:
`86.069 tok/s` and `74.795` effective tok/s in
`2026-05-24-state-ramp-request-context-single-token-last-logits-default-seed240524-go-mlx-gemma4-e2b-4bit-opencode-30k-r10-g1024.json`
versus `86.230 tok/s` and `74.909` effective tok/s with
`GO_MLX_ENABLE_LAST_LOGITS_PREFILL=0`. Keep both out of the production default
until a fused logits/materialisation change proves a 10-turn workflow win.

Strict eval-boundary cleanup, 2026-05-24: `Model.Generate` and retained
`ModelSession.Generate` now detach the evaluated logits array at the same
per-token boundary as the K/V caches after `Eval(next)` materialises the
sampled token. This follows the IDEAS.md graph-bloat guidance: the current
token's logits graph should not stay attached while the next one-token graph is
being built. This is a production-path graph-lifetime correction, not a new
acceptance row. The tiny retained-session smoke
`/private/tmp/go-mlx-goal/reports/2026-05-24-detach-logits-boundary-smoke.json`
is only a runtime sanity check; it records paged K/V (`fixed_caches=0`,
`paged_caches=15`), `max_local_capacity=512`, `max_global_capacity=131072`,
and `local_window_leaked=false`. The next performance proof still needs the
matched request-context retained run against llama.cpp.

Default seed correction, 2026-05-24: the production lane and local profile
commands now use `mlx.DefaultNewSessionText` as the default prompt instead of
the old synthetic "retained model state" question. This lines up
`DefaultProductionLane`, `driver-profile`, and `state-ramp-profile` with the
Lemma new-session seed already used by the shared comparator scripts while
preserving explicit prompt overrides and the explicit empty-seed state-ramp
path. Verification: `go test ./go -run
'TestProductionLane_DefaultGemma4E2B|TestDefaultLemmaNewSessionText'`,
`go test ./go/cmd/mlx -run
'TestRunCommand_(StateRampProfileJSON|DriverProfileFastGemma4LaneDefault|StateRampProfileExplicitEmptySeedPrompt)'`,
and a grep check showing the old retained-state question is absent from the
production lane and CLI default sources.

Runtime correction, 2026-05-24: the rejected paged full-K/V materialise owner
path has now been physically retired from the runtime, not merely left unused
by benchmark flags. `GO_MLX_ENABLE_PAGED_FULL_KV_MATERIALIZE` is no longer a
known runtime/reporting gate, Gemma 4 single-token paged attention always
updates borrowed page state directly, and `PagedKVCache` no longer carries the
full-materialised backing arrays/helper path that previously made this easy to
re-enable. Focused verification: `go test ./go/internal/metal -run
'TestPagedKVCache_BorrowedPageState|TestGemma4_AttentionPagedDoesNotRetainFullMaterializedKV|TestRuntimeGate_KnownNativePagedAttention|TestRuntimeGate_KnownPagedKVPrealloc'`,
`go test ./go -run
'TestProductionLane|TestRunCommand_ChapterProfileFastLaneDefaults|TestStateRampProfileDefaultCompactionThresholdUsesModelContext'`,
and `go test ./go/internal/metal ./go/cmd/mlx ./go`. Hot-path check:
`BenchmarkPagedKVCache_UpdateBorrowedPages_To128` reports `1185060 ns/op`,
`40 B/op`, `5 allocs/op` on Apple M3 Ultra after the deletion.

Latest pinned State restore cleanup, 2026-05-24: the contiguous
`fromPinnedRawBytes` path no longer routes through the strided/mdspan wrapper
when the State page view exactly matches its storage layout. It now calls a
dedicated `go_mlx_array_new_pinned_data` bridge that validates one shape and
hands the pinned Go buffer directly to `mlx_array_new_data_managed_payload`;
`fromPinnedRawBytesStrided` still owns the C++23 mdspan subview path. Focused
verification: `go test ./go/internal/metal -run
'TestPinnedArray|TestRuntimeGate|TestPagedKVCache'` and
`go test ./go/internal/metal -run '^$' -bench
'BenchmarkPinnedArray_(NewFromGoSlice|VsCopyPath|Strided|PinSlice|ShapeElementCount|ContiguousStrides)'
-benchmem -benchtime=200ms`. The canonical pinned KV rows improve from the
previous same-machine band of about `3.9-5.1us/op` to `2.9-3.7us/op` while
staying at `56 B/op`; `BenchmarkPinnedArray_VsCopyPath_PinnedRaw_L4096`
records `3515 ns/op`, `56 B/op`, `2 allocs/op` versus the copy path at
`4206595 ns/op`, `8390354 B/op`, `3 allocs/op`. This is a State restore and
zero-copy layout win, not a raw decode acceptance row.

Latest retained decode phase correction, 2026-05-24: the accepted
`GO_MLX_ENABLE_ASYNC_DECODE_PREFETCH=1` fast-lane gate is now a real runtime
gate for both `Model.Generate` and retained `ModelSession.Generate`, not only a
reported CLI setting. The follow-up trace
`/private/tmp/go-mlx-goal/reports/2026-05-24-state-ramp-request-context-prefetchbucket-turn2-go-mlx-gemma4-e2b-4bit-opencode-30k-r2-g1024.json`
adds an explicit `prefetch` token-phase bucket around the async next-logits
materialisation boundary. It completes the same two-turn request-context shape
with `33728` final live tokens, `1069` visible/generated tokens,
`88.95376383688955 tok/s` raw decode, `79.58783725070474` effective turn
tok/s, `9710538338` active-plus-cache bytes, `3382902784` RSS bytes, no fixed
Gemma 4 caches, `max_local_tokens=512`, `max_global_capacity=131072`, and
`local_window_leaked=false`. The phase breakdown is now explicit: `prefetch`
averages `6332038 ns/token`, `sample_eval` averages `3278816 ns/token`,
`forward` averages `1560206 ns/token`, and the old catch-all `other` bucket
collapses to `2563 ns/token`. This proves the next decode target is not hidden
Go bookkeeping; it is the async MLX next-logits dispatch/materialisation
boundary that IDEAS.md calls the graph-compiler/eval-boundary problem. This is
instrumentation plus corrected gate behaviour, not final production acceptance.

Latest dirty-KV prefetch correction, 2026-05-24: retained decode now evaluates
the next logits together with only the K/V cache arrays touched by the most
recent token update. This follows the IDEAS.md eval-boundary guidance without
falling back to `PagedKVCache.AppendState`, which would re-evaluate every
historical page on every decode step. `PagedKVCache.AppendDirtyState` is covered
by `TestPagedKVCache_AppendDirtyStateOnlyRecentPage_Good` and the hot-path
benchmark records `BenchmarkPagedKVCache_AppendDirtyState_After128_PageSize256`
at `3.793 ns/op`, `0 B/op`, `0 allocs/op`, versus the same prepared full-state
access row at `4.787 ns/op`, `0 B/op`, `0 allocs/op`. The same two-turn traced
request-context shape writes
`/private/tmp/go-mlx-goal/reports/2026-05-24-state-ramp-request-context-dirtykv-prefetch-turn2-go-mlx-gemma4-e2b-4bit-opencode-30k-r2-g1024.json`;
with identical `33728` final live tokens and `1069` visible/generated tokens,
raw decode moves from `88.95376383688955` to `89.38593825405013 tok/s`, and
effective turn throughput moves from `79.58783725070474` to
`79.91675301645665 tok/s`. The full 10-turn retained workflow writes
`/private/tmp/go-mlx-goal/reports/2026-05-24-state-ramp-request-context-dirtykv-prefetch-go-mlx-gemma4-e2b-4bit-opencode-30k-r10-g1024.json`;
with the same `48712` final live tokens and `4292` visible/generated tokens as
the shared-KV baseline, raw decode improves from `84.63319127288695` to
`86.1254434039376 tok/s` (`+1.763%`), effective throughput improves from
`72.743662496295` to `73.83925639591638 tok/s` (`+1.506%`), wall time drops by
`0.967560791s`, and estimated energy drops by `96.7560791 J` at `100 W`.
Active-plus-cache memory is essentially flat (`+917560` bytes), RSS is
`+20398080` bytes, fixed caches remain absent, `paged_caches=15`,
`max_local_tokens=512`, `max_global_capacity=131072`, and
`local_window_leaked=false`. This is a small accepted production-path decode
win, not the final llama.cpp parity closure; the next target remains the larger
MLX graph/materialisation cost inside the `prefetch` and `sample_eval` buckets.

Latest packed-State wake proof, 2026-05-24: `state-wake-profile` now records
phase-local Go heap, MLX allocator, and process-memory deltas for store open
and wake. A same-state real wake comparison uses the existing folded C014
state, `658` prefix tokens, `3` native State blocks, `context=32768`,
`cache-mode=paged`, `max_tokens=64`, `temperature=1.0`, `top_p=0.95`, and
`top_k=64`. The raw `.mvlog` report
`/private/tmp/go-mlx-goal/reports/2026-05-24-state-wake-memorydelta-mvlog-c014-g64.json`
records `441.854083ms` wake, `49,452,400` wake-phase Go allocation bytes,
`2,580` wake mallocs, `23` generated/visible tokens, `104.87698882223789`
decode tok/s, and `759.881874ms` wake-plus-turn wall. The packed `.kv` report
`/private/tmp/go-mlx-goal/reports/2026-05-24-state-wake-memorydelta-kv-c014-g64.json`
opens the same State log as a Trix payload window at offset `705` with
`440,038,885` payload bytes and records `339.639375ms` wake, `157,344`
wake-phase Go allocation bytes, `2,635` wake mallocs, `23` generated/visible
tokens, `105.74402704288552` decode tok/s, and `653.837375ms`
wake-plus-turn wall. Interpretation: the packed `.kv` region path cuts the
wake heap allocation by about `99.68%`, saves `102.214708ms` of wake time, and
does not regress decode on this short continuation. Process RSS is effectively
neutral in this pair (`3,712,368,640` bytes for `.mvlog` versus
`3,712,090,112` bytes for `.kv`).

Follow-up State store-open fix, 2026-05-24: the `go-inference`
`state/filestore` index rebuild no longer preallocates index maps from raw file
byte size once the State payload is large. Large `.kv` containers often hold a
few huge records, so the old `(file_bytes / 128)` hint allocated hundreds of
MiB before wake could borrow mmap-backed blocks. The focused benchmark
`BenchmarkFilestoreCapacity_Open_SingleLargePayload` records `15856 ns/op`,
`1680 B/op`, and `10 allocs/op`, while
`BenchmarkFilestoreCapacity_Open_10000Records` keeps the small-record reopen
shape visible at `4793836 ns/op`, `2120132 B/op`, and `10075 allocs/op`.
The real packed `.kv` wake retry
`/private/tmp/go-mlx-goal/reports/2026-05-24-state-wake-memorydelta-kv-indexhint-rerun-g16.json`
opens the same `440,038,885` byte State payload and drops `store_open`
allocation from the earlier `481,103,232` total bytes / `309,535,144` live heap
bytes to `17,056` total bytes / `17,056` live heap bytes, with RSS delta down
from `285,851,648` bytes to `32,768` bytes. Decode remains in the same short
continuation band at `104.82051534023674 tok/s`, `fixed_caches=0`, and
`local_window_leaked=false`. The next hot path is therefore not State
store-open hydration; it is the retained decode graph/materialisation path
visible in the request-context `sample_eval` token phase.

While investigating that retry, the profile stream cancellation
path was corrected: `driver-profile`, `state-ramp-profile`, and
`chapter-profile` now cancel generation on live-memory/repetition/end-marker
guards but continue draining the token channel until the generator closes
before reading `model.Metrics()`. This prevents stale prompt/generated-token
counts, cache profiles, and memory figures in failed or guarded turns. Verified
with `TestDriverProfileGeneration_DrainsCancelledStreamBeforeMetrics_Good`,
`go test ./go/cmd/mlx -run 'TestDriverProfileGeneration_DrainsCancelledStreamBeforeMetrics|TestDriverProfileGeneration_ChatModeDoesNotStartRawStream|TestRunCommand_StateRampProfileTargetShapeStaysPaged' -count=1`,
`go test ./go/cmd/mlx -bench='BenchmarkStateRampProfile|BenchmarkDriverProfile' -benchmem -run='^$'`,
and `env MLX_METALLIB_PATH=/Users/snider/Code/core/go-mlx/dist/lib/mlx.metallib GOCACHE=/private/tmp/codex-go-mlx-cache go test ./go/... -count=1`.
Follow-up correction, 2026-05-24: `state-ramp-profile` no longer synthesises
`GO_MLX_FIXED_GEMMA4_CACHE_SIZE` from target tokens, compaction threshold, or
context window. The current optimisation lane does not use fixed Gemma 4 K/V;
profile and benchmark work must stay paged/no-fixed unless the user explicitly
asks to reproduce an archived diagnostic.

Superseded fixed-cache diagnostic, 2026-05-24: the `65536` context boundary was
removed as a cache-family switch, but the intermediate fix still used fixed K/V
by default. That diagnostic kept fixed K/V gates enabled and derived
`GO_MLX_FIXED_GEMMA4_CACHE_SIZE` from the requested run shape
(`target/compaction threshold + max tokens`, rounded to `32`) rather than from
the model context length. Follow-up code also stops treating `65536` as a
default or recommender boundary: `chapter-profile` now defaults to the
opencode-sized `32768` lane, the 64GB memory plan no longer selects `65536`,
the context ramp skips the `24:65536` step, and `kv.CompareModes` recommends
from estimated K/V bytes rather than a context-token cutoff. Two same-fixture
diagnostics validate the correction:
`2026-05-24-state-ramp-request-context-fixed70000-go-mlx-gemma4-e2b-4bit-opencode-30k-r10-g1024.json`
records `10/10`, `48712` final live tokens, `4292` generated/visible tokens,
`66.219s` wall, `94.091` raw decode tok/s, `79.667` effective turn tok/s,
`10055628170` active-plus-cache bytes, `3.177 GiB` RSS, and `508.415 GiB`
virtual reservation. The tighter
`2026-05-24-state-ramp-request-context-fixed54688-go-mlx-gemma4-e2b-4bit-opencode-30k-r10-g1024.json`
records the same output count at `66.180s` wall, `93.911` raw decode tok/s,
`79.525` effective turn tok/s, `9989449830` active-plus-cache bytes,
`3.166 GiB` RSS, and `510.477 GiB` virtual reservation. The rebuilt no-extra-env
default row,
`2026-05-24-state-ramp-request-context-default-fixedbudget-go-mlx-gemma4-e2b-4bit-opencode-30k-r10-g1024.json`,
keeps the same production shape and records runtime gates
`GO_MLX_ENABLE_FIXED_GEMMA4_CACHE=1`,
`GO_MLX_ENABLE_FIXED_GEMMA4_SLIDING_CACHE_BOUND=1`,
`GO_MLX_ENABLE_FIXED_GEMMA4_SHARED_MASK=1`,
`GO_MLX_FIXED_GEMMA4_CACHE_SIZE=71040`, and
`GO_MLX_KV_CACHE_DTYPE=fp16` without setting `GO_MLX_PAGED_KV_PAGE_SIZE`.
It completes `10/10`, reaches `48712` final live tokens, generates `4292`
visible tokens, records `66.165s` wall, `94.143` raw decode tok/s, `79.731`
effective turn tok/s, `3.212x` retained-vs-replay speedup estimate,
`6616.520 J` at `100 W`, `10048930954` active-plus-cache bytes, `3.166 GiB`
RSS, and `508.693 GiB` virtual reservation. Against the previous paged
request-context row, this recovers about `11%` raw decode and about `5.17s`
wall time while cutting process virtual reservation by about `59.5 GiB`.
Follow-up instrumentation now adds `metrics.cache_profile` to both one-shot and
retained generation reports. For Gemma 4 it records local-cache count,
global-owner count, shared-layer count, sliding-window tokens, max local/global
tokens, max local/global capacity, cache kind counts, max processed tokens, and
`local_window_leaked`. This makes the IDEAS.md local-layer leakage hypothesis
directly falsifiable in `state-ramp-profile` JSON instead of inferred from RSS
or raw tok/s. The hook is measured at `85.40 ns/op`, `176 B/op`, `1 alloc/op`
for the fixed Gemma 4 topology walk and root metrics conversion with a cache
profile at `52.14 ns/op`, `176 B/op`, `1 alloc/op`; the existing no-profile
root metrics path remains `25.79 ns/op`, `0 B/op`, `0 allocs/op`. The first
live 4096-context smoke with this metric exposed the remaining local-window
leak (`max_local_tokens=1283`, `max_local_capacity=1440`,
`local_window_leaked=true`) because `GO_MLX_ENABLE_FIXED_GEMMA4_SLIDING_CACHE_BOUND`
was still long-context-only. The diagnostic fixed-cache path then enabled the
fixed sliding bound and reran the same smoke at
`/private/tmp/go-mlx-goal/reports/2026-05-24-cache-profile-smoke-bounded.json`
records `GO_MLX_ENABLE_FIXED_GEMMA4_SLIDING_CACHE_BOUND=1`,
`max_local_tokens=512`, `max_local_capacity=512`, `max_global_tokens=1296`,
`max_global_capacity=1440`, and `local_window_leaked=false`, with the short
smoke decode at `110.929 tok/s`.

Latest request-context token-phase trace, 2026-05-24:
`/private/tmp/go-mlx-goal/reports/2026-05-24-state-ramp-request-context-current-trace-turn2-go-mlx-gemma4-e2b-4bit-opencode-30k-r2-g1024.json`
captures the same fixture with `-trace-token-phases` for two turns. It
completes `2/2` turns, generates `1069` visible tokens, and records
`87.814` raw decode tok/s. The phase summary shows steady token
`total` at `11.364ms` average, `sample_eval` at `9.804ms`, and next-token
`forward` graph construction at `1.514ms`. The `sample_eval` bucket is the
lazy MLX materialisation of the current one-token forward graph plus sampler,
not ordinary Go-side token sampling. This keeps the next optimisation target
on a stable/fused one-token graph boundary and KV slotting, not CLI streaming,
string handling, or visible-output accounting.

Follow-up sampler cleanup, 2026-05-24: the standard production sampling
configuration uses `temperature=1.0`, `top_p=0.95`, and `top_k=64`. The sampler
builder no longer inserts a `Temperature(1.0)` node before top-k/top-p because
that full-vocab `MulScalar(logits, 1)` is mathematically a no-op. Focused
bench evidence on the Gemma-sized vocab moves
`BenchmarkSampler_TopKThenTopP_Vocab262k` from `548272 ns/op`, `24 B/op`,
`3 allocs/op` to `512250 ns/op`, `24 B/op`, `3 allocs/op` (`~6.6%` faster).
The matched two-turn retained trace at
`/private/tmp/go-mlx-goal/reports/2026-05-24-state-ramp-request-context-unit-temp-skip-trace-turn2-go-mlx-gemma4-e2b-4bit-opencode-30k-r2-g1024.json`
keeps the same `1322` generated/visible tokens, no output-quality issues, and
bounded paged/no-fixed gates; it records `88.145` raw decode tok/s versus
`88.033` for the prior trace, `80.521` effective turn tok/s versus `80.451`,
and `9.758ms` average `sample_eval` versus `9.787ms`. This is a correct
production-path cleanup, not enough to close the llama.cpp raw-decode gap by
itself.

Q4 last-logits graph-path correction, 2026-05-25: the Gemma-sized isolated
tail bench rejects the native q4 last-token logits wrapper for production use.
`BenchmarkDecodeLoop_LastTokenOutputQ4Native_H2048_Vocab262k` repeats at
`726587`, `722748`, `716416`, `724500`, and `711984 ns/op`, while the MLX graph
path repeats at `700215`, `702024`, `704036`, `700512`, and `689999 ns/op`;
both paths report `0 B/op` and `0 allocs/op`, so the native wrapper is paying
execution cost rather than Go allocation cost. Production now keeps dense
last-token output on the native path, but leaves quantized q4 output on the MLX
graph path. The same-seed two-turn retained trace at
`/private/tmp/go-mlx-goal/reports/2026-05-25-state-ramp-request-context-q4-graph-last-logits-sameseed-trace-turn2-go-mlx-gemma4-e2b-4bit-opencode-30k-r2-g1024.json`
completes `2/2` turns with `local_window_leaked=false`, `1069` generated and
visible tokens, `90.256` raw decode tok/s, and `80.650` effective turn tok/s.
The average token phase moves from `11.327ms` total, `9.758ms` sample_eval, and
`1.523ms` prefetch_logits in the previous q4-native trace to `11.058ms` total,
`3.362ms` sample_eval, and `6.169ms` prefetch_logits. This is a narrow
production-path decode improvement; it does not replace the required full
10-turn request-context row against llama.cpp.
Full-row follow-up for the same q4 graph-path correction:
`/private/tmp/go-mlx-goal/reports/2026-05-25-state-ramp-request-context-q4-graph-last-logits-sameseed-go-mlx-gemma4-e2b-4bit-opencode-30k-r10-g1024.json`
uses the same `30k` opencode seed, `10` request-context turns, `1024`
max-token budget, `seed=240524`, paged K/V, and no fixed-cache gates. It
completes `10/10` turns, reaches `48712` live tokens, generates `4292`
visible tokens, records `70.031s` retained wall, `86.610` raw decode tok/s,
`74.211` effective turn tok/s, `3.074x` retained-vs-replay speedup,
`7003.057 J` at `100 W`, `9.259 GiB` active-plus-cache, `3.171 GiB` RSS, and
`568.230 GiB` process virtual reservation, with `local_window_leaked=false`.
Against the same-output dirty-K/V prefetch row, raw decode improves by
`0.563%`, effective throughput by `0.503%`, wall drops by `0.336s`, and
estimated energy drops by `33.622 J`. The current llama.cpp
Q4_K_M request-context anchor still leads raw decode at `105.988 tok/s`, so
the next optimisation remains the larger prefetch/logits materialisation
boundary rather than declaring parity from this small production-path win.

Last-token accessor cleanup, 2026-05-25: the normal single-token decode logits
shape no longer builds a no-op `SliceAxis` node before reshaping to `[1,vocab]`.
`BenchmarkDecodeLoop_LastTokenLogitsSingleStep_FastReshape_Vocab262k` repeats
at `21407`-`22023 ns/op`, `8 B/op`, `1 alloc/op` versus the legacy slice helper
at `22218`-`22759 ns/op`, `40 B/op`, `3 allocs/op`. The same two-turn
request-context trace shape writes
`/private/tmp/go-mlx-goal/reports/2026-05-25-state-ramp-request-context-last-token-reshape-turn2-go-mlx-gemma4-e2b-4bit-opencode-30k-r2-g1024.json`
with `1069` generated/visible tokens, `90.578` raw decode tok/s, `80.901`
effective tok/s, and `25.404s` wall. The `logits` phase drops from `9.124us`
to `4.121us` per token, while the dominant `prefetch_logits` and `sample_eval`
buckets remain the real parity target.

Scalar reshape cleanup, 2026-05-25: the remaining token input construction
paths now use the fixed-rank `Reshape2` helper instead of variadic `Reshape`
for `[1,len(tokens)]` and `[1,1]` token tensors. This covers retained
generation, prompt-cache replay/append, Gemma 4 assistant draft/verify, and the
Qwen split path without changing cache, sampling, or chat-template semantics.
The focused tests for prompt-cache, Gemma 4 assistant, split, last-token, and
`Reshape2` pass. A fresh `lthn-mlx` binary smoke at
`/private/tmp/go-mlx-goal/reports/2026-05-25-state-ramp-smoke-scalar-reshape-current.json`
uses the local Gemma 4 E2B 4bit pack, `context=4096`, `start=512`,
`target=1024`, `turns=1`, `turn_max_tokens=256`, paged K/V, and no fixed-cache
gates. It completes `1/1` retained turn with `1125` final live tokens, `99`
generated/visible tokens, `108.517` raw decode tok/s, `72.906` effective turn
tok/s, `3.978 GB` active-plus-cache, `3.390 GB` RSS, `465.540 GB` virtual
reservation, `paged_caches=15`, `fixed_caches=0`, `max_local_capacity=512`,
`max_global_capacity=4096`, and `local_window_leaked=false`. The phase summary
still points at the same real bottleneck: `prefetch_logits=4.730ms/token`,
`sample_eval=2.970ms/token`, and `forward=1.400ms/token`. Treat this as a
current-binary smoke and allocation/cgo-shape cleanup only, not a replacement
for the required 10-turn retained comparator against llama.cpp.

Current full-output request-context row, 2026-05-25:
`/private/tmp/go-mlx-goal/reports/2026-05-25-state-ramp-request-context-scalar-reshape-current-include-output-go-mlx-gemma4-e2b-4bit-opencode-30k-r10-g1024.json`
reruns the accepted `request-context` fixture with generated text captured in
the report. It uses the local Gemma 4 E2B 4bit pack, `30k` seed,
`context=131072`, `10` turns, `1024` max generated tokens per turn,
`append_tokens=8192`, `prefill_chunk_size=512`, `temperature=1.0`,
`top_p=0.95`, `top_k=64`, no visible-token floor, no forced compaction, paged
fp16 K/V, and the default fast Gemma 4 gates. It completes `10/10` turns with
`48896` final live tokens, `14400` appended tokens, `4476` generated and
visible tokens, `73.872368791s` wall, `84.06360150221701 tok/s` raw decode,
`72.64194131583837` effective turn tok/s, `2447.9658757787 tok/s` initial
prefill, `2.9776898258175146x` retained-vs-replay speedup, `7.3872368791 kJ`
estimated energy at `100 W`, and `14.6096632167 kJ` saved versus replayed
prefill. Memory is bounded on the real resident side: `3.746 GB` MLX peak,
`9.932 GB` active-plus-cache, `3.388 GB` process RSS, and `612.837 GB` process
virtual reservation. The final cache profile keeps the intended Gemma 4 shape:
`paged_caches=15`, `fixed_caches=0`, `local_caches=12`, `global_caches=3`,
`max_local_capacity=512`, `max_global_capacity=131072`, and
`local_window_leaked=false`. The captured text is topical for all ten turns and
has no harness-reported output issues, but turn `10` is concise (`116` visible
tokens) against its own `700`-`1000` token request, so this row is performance
evidence plus captured-output evidence rather than a closed quality gate. The
matched llama.cpp Q4_K_M request-context memory anchor still records
`109.99746968612104 tok/s` raw decode and `76.89775797091058` wall-visible
tok/s over `72.91499970806763s` wall, so go-mlx is only about `0.957s` slower
on total wall and uses about `1.262 GB` less RSS, but llama.cpp remains
`1.309x` faster on raw decode and `1.059x` faster on wall-visible throughput.
The trace keeps the next optimisation target unchanged:
`prefetch_logits=6.874ms/token`, `sample_eval=3.240ms/token`, and
`forward=1.700ms/token`.

Fused suppress-token sampler, 2026-05-25: the production Gemma 4 sampler shape
(`temperature=1.0`, `top_p=0.95`, `top_k=64`, non-empty control-token
suppression, no other sampler prefix) now folds suppression into the compiled
top-k/top-p sampler closure instead of materialising a separate prefix
`PutAlongAxis` graph before the compiled call. The unfused path remains for
temperature, min-p, non-top-k/top-p, and fallback shapes. Focused validation:
`go test ./go/internal/metal -run 'TestSample_|TestCompile_|TestModelSession_Generate|TestModel_Generate'`
passes, and the sampler benchmark
`go test ./go/internal/metal -run '^$' -bench 'BenchmarkSampler_TopKThenTopP(WithSuppression)?_Vocab262k|BenchmarkSampler_CompiledTopKThenTopPCallOne_Vocab262k' -benchmem -count 3`
keeps the production suppressed sampler at `495-503us/op`, `10 B/op`, and
`1 alloc/op`. The same full-output retained request-context row writes
`/private/tmp/go-mlx-goal/reports/2026-05-25-state-ramp-request-context-fused-suppress-sampler-go-mlx-gemma4-e2b-4bit-opencode-30k-r10-g1024.json`
with identical output/token shape to the current baseline: `10/10` turns,
`48896` final live tokens, `14400` appended tokens, and `4476` generated and
visible tokens. Wall drops from `73.872368791s` to `73.261458999s`
(`-0.82698%`), raw decode improves from `84.06360150221701` to
`85.01050148275976 tok/s` (`+1.12641%`), effective turn throughput improves
from `72.64194131583837` to `73.3508898684956` (`+0.97595%`), and estimated
energy drops by `61.0909792 J` at `100 W`. Cache invariants hold:
`paged_caches=15`, `fixed_caches=0`, `max_local_capacity=512`,
`max_global_capacity=131072`, and `local_window_leaked=false`. Phase timing
moves in the right direction but does not eliminate the boundary:
`prefetch_logits=6.839ms/token`, `sample_eval=3.239ms/token`, and
`forward=1.613ms/token`. Against the same llama.cpp Q4_K_M request-context
anchor, go-mlx is now only `0.346s` slower on wall and still uses less RSS, but
llama.cpp remains `1.294x` faster on raw decode and `1.048x` faster on
wall-visible throughput, so the production gate remains open.

Fresh llama.cpp anchor refresh, 2026-05-25: reran the same request-context
shape against `/opt/homebrew/bin/llama-server` version `9260 (3a6db741a)`,
built with AppleClang `21.0.0.21000099`, using the same
`gemma-4-E2B-it-Q4_K_M.gguf`, `30k` start tokens, `10` turns,
`target_tokens=100000`, `max_tokens=1024`, Gemma 4 stop strings,
`seed=240524`, `temperature=1.0`, `top_p=0.95`, `top_k=64`, and
`repeat_penalty=1.0`. Report:
`/private/tmp/go-mlx-goal/reports/2026-05-25-llamacpp-request-context-refresh-seed240524-gemma4-e2b-q4km-opencode-30k-r10-g1024.json`.
The refreshed llama.cpp row completes `10/10`, reaches `50248` final live
tokens, appends `14400` tokens, generates `5828` tokens / `5818` visible
tokens, records `75.161548416s` wall, `110.18737904534018` raw decode tok/s
from llama.cpp timings, `77.40660114915106` wall-visible tok/s,
`21.670089s` prompt timing, `7.516 kJ` estimated energy at `100 W`,
`5.068 GB` peak RSS, `459.112 GB` peak virtual, no output-quality flags, and
no visible control markers. Against the current fused-suppression go-mlx row
above, go-mlx is `1.900089417s` faster on retained workflow wall and saves
about `190.009 J` at `100 W`, while llama.cpp remains `1.29616197x` faster on
raw decode and `1.05529192x` faster on visible wall throughput because it
returns more visible content in the same shape. Interpretation: the retained
State wall/energy lane now beats the current llama.cpp server build on this
10-turn request-context row, but the production optimisation target remains
the raw decode/materialisation gap visible in go-mlx
`prefetch_logits=6.839ms/token`, `sample_eval=3.239ms/token`, and
`forward=1.613ms/token`.

Promoted paged K/V page geometry, 2026-05-25: the current retained
request-context path now defaults paged K/V blocks to `2048` tokens while local
Gemma 4 sliding-window layers still cap at their `512`-token window. The full
no-env default row
`/private/tmp/go-mlx-goal/reports/2026-05-25-state-ramp-request-context-default-page2048-go-mlx-gemma4-e2b-4bit-opencode-30k-r10-g1024.json`
uses only the normal fast-lane runtime gates plus `GO_MLX_KV_CACHE_DTYPE=fp16`;
it does not emit `GO_MLX_PAGED_KV_PAGE_SIZE`, proving the wider page geometry is
the code default rather than a hidden CLI/env override. It keeps the same output
shape as the fused-suppression baseline (`10/10`, `48896` final live tokens,
`14400` appended tokens, `4476` generated/visible tokens), drops wall from
`73.261458999s` to `71.73144004s` (`-2.088%`), improves raw decode from
`85.01050148275976` to `87.44275487305373 tok/s` (`+2.861%`), improves
effective turn throughput from `73.3508898684956` to
`75.21070749898786 tok/s` (`+2.536%`), and saves `153.0018959 J` at `100 W`.
RSS is slightly lower (`3.377 GB` versus `3.409 GB`) while virtual reservation
rises by about `16.40 GB`, so this is a retained-workflow speed/default cleanup
rather than a memory-only win. Native events report
`paged_kv.fast_concat.global` at `13428` calls, `24` max pages, and `48894`
max tokens; cache invariants remain `fixed_caches=0`, `paged_caches=15`,
`max_local_capacity=512`, `max_global_capacity=131072`, and
`local_window_leaked=false`. Against the refreshed llama.cpp Q4_K_M server row,
the no-env go-mlx default is `3.430108376s` faster on retained workflow wall and
saves `343.0108376 J`, while llama.cpp still leads raw decode by `1.2601x` and
visible wall throughput by `1.0292x`. The older archived 100k page-geometry
rejection remains useful historical evidence for the former path, but it does
not veto this current request-context default. The remaining raw-decode gap is
still the global owner attention materialisation/sampler-eval boundary, not a
fixed cache, hidden page-size flag, or context-cutoff problem.

Rejected wider-page follow-up, 2026-05-25: forcing
`GO_MLX_PAGED_KV_PAGE_SIZE=4096` on the same two-turn request-context shape
halves the global fast-concat page count (`17` max pages to `9`) but worsens
the real workflow row. The default 2048-token page report
`/private/tmp/go-mlx-goal/reports/2026-05-25-state-ramp-request-context-default-page2048-turn2-go-mlx-gemma4-e2b-4bit-opencode-30k-r2-g1024.json`
records `26.430020416s` wall, `91.0239815475048` raw decode tok/s,
`81.96795883694631` effective tok/s, `9827367654` active-plus-cache bytes,
`3389947904` RSS bytes, `522658332672` virtual bytes, and
`paged_kv.fast_concat.global` at `4047ns` average duration. The matched
4096-token diagnostic
`/private/tmp/go-mlx-goal/reports/2026-05-25-state-ramp-request-context-page4096-turn2-go-mlx-gemma4-e2b-4bit-opencode-30k-r2-g1024.json`
records the same `2/2` turns, `33825` final live tokens, and `1166`
generated/visible tokens, but regresses to `26.517627915s` wall,
`90.45554345018256` raw decode tok/s, `81.49816578484192` effective tok/s,
`9849196746` active-plus-cache bytes, `3391078400` RSS bytes, and
`522818568192` virtual bytes. Keep 2048 as the code default; larger pages are
not the next retained-decode fix even though the native concat micro-event gets
shorter.

Rejected flat-logits handle clone, 2026-05-25: replacing the normal
single-token `[1,vocab]` `lastTokenLogits` no-op `Reshape2` with a retained
handle clone looked attractive in isolation, and the new focused bench
`BenchmarkDecodeLoop_LastTokenLogitsAlreadyFlat_Vocab262k` records the flat
case explicitly. The real retained workflow rejected the runtime change. The
matched trace
`/private/tmp/go-mlx-goal/reports/2026-05-25-state-ramp-request-context-flat-lastlogits-turn2-go-mlx-gemma4-e2b-4bit-opencode-30k-r2-g1024.json`
keeps the same `2/2` turns, `33825` final live tokens, and `1166`
generated/visible tokens as the default 2048-token page row, but regresses wall
from `26.430020416s` to `26.808138414s`, raw decode from
`91.0239815475048` to `88.68742375156263 tok/s`, and effective throughput
from `81.96795883694631` to `80.03241840637767 tok/s`. The phase split shows
why this cannot be promoted: `sample_eval` improves slightly
(`3.291352ms/token` to `3.260448ms/token`), but `prefetch` worsens
(`6.219972ms/token` to `6.331789ms/token`), `forward` worsens
(`1.440422ms/token` to `1.618338ms/token`), and the native global concat event
average rises from `4047ns` to `5908ns`. Keep the existing `Reshape2` path;
the benchmark remains only to make this tempting flat-logits shape measurable.

Rejected follow-up probes, 2026-05-25: several small materialisation-boundary
cleanup ideas were measured and reverted because they did not improve the real
retained workflow. A rank-known Gemma 4 PLE view helper improved the isolated
PLE view microbench (`BenchmarkPLE_PerLayerInputViewsStreamedRank4_Graph` at
about `19.4-20.3us/op` versus the wrapper path at about `20.5-20.9us/op`), but
the matched two-turn retained trace
`/private/tmp/go-mlx-goal/reports/2026-05-25-state-ramp-request-context-ple-rank4-view-trace-turn2-go-mlx-gemma4-e2b-4bit-opencode-30k-r2-g1024.json`
fell to `88.597` raw decode tok/s and `79.277` effective tok/s versus the
accepted last-token-reshape trace at `90.578` / `80.901`. A host-side
64-candidate top-k/top-p sampler similarly improved the isolated sampler row
(`BenchmarkSampler_TopKThenTopP_Vocab262k` at about `461-481us/op` versus the
normal `545-566us/op` band) by moving top-p and categorical sampling out of the
MLX graph, but the retained trace
`/private/tmp/go-mlx-goal/reports/2026-05-25-state-ramp-request-context-host-topk-topp-trace-turn2-go-mlx-gemma4-e2b-4bit-opencode-30k-r2-g1024.json`
rejected it: `88.769` raw decode tok/s, `79.019` effective tok/s, larger
active-plus-cache memory, and `2` output-issue turns. The phase data was useful
but not a win: `sample_eval` collapsed to `308ns/token`, while `sample` grew to
`3.381ms/token`, proving the work merely moved buckets. Disabling the accepted
async prefetch gate was also slower (`88.645` raw decode tok/s with
`sample_eval=9.757ms/token`) than the same current-source default trace
(`89.712` raw decode tok/s). Keep the next optimisation on a fused/stable MLX
one-token graph boundary rather than host sampling, PLE rank checks, or
turning off async decode prefetch.

Local-window paged overflow cleanup, 2026-05-25: the bounded local Gemma 4
window path no longer appends a one-token second page, trims the first page,
then compacts both pages back into a single page after the 512-token cap is
full. The paged cache now handles the exact local-window single-token overflow
case directly as drop-first-plus-append, preserving temporal order and keeping
one visible K/V page. The focused bench
`BenchmarkPagedKVCache_BorrowedSlidingWindow512_SinglePage` moved from about
`10.8-11.1ms/op`, `32.9-33.0KB/op`, and `2061 allocs/op` to repeated rows
around `9.98-10.09ms/op`, `68-70 B/op`, and `7 allocs/op`. Correctness is
covered by `TestPagedKVCache_SlidingWindowStaysSinglePage_Good`, which now
checks token order after overflow, not just page count. Retained workflow
evidence classifies this as an allocation/GC-pressure cleanup, not a decode-gap
breakthrough: the same-seed two-turn trace
`/private/tmp/go-mlx-goal/reports/2026-05-25-state-ramp-request-context-local-window-fast-overflow-turn2-go-mlx-gemma4-e2b-4bit-opencode-30k-r2-g1024.json`
records `90.792` raw decode tok/s and `81.038` effective tok/s with
`local_window_leaked=false`, but the full rerun
`/private/tmp/go-mlx-goal/reports/2026-05-25-state-ramp-request-context-local-window-fast-overflow-rerun2-go-mlx-gemma4-e2b-4bit-opencode-30k-r10-g1024.json`
is effectively neutral against the accepted q4 graph row: `86.563` raw decode
tok/s, `74.140` effective tok/s, and `70.119s` wall versus `86.610`, `74.211`,
and `70.031s`. Keep the code for the sharply lower local-window allocation
surface and simpler state mutation, but do not count it as closing the
llama.cpp raw decode gap.

Compiled sampler cleanup, 2026-05-25: the default top-k/top-p sampler now uses a
per-generation compiled MLX closure for the bounded-candidate sampling graph and
`CompiledFunc.CallOne` for the one-input/one-output call shape. This avoids a
global compiled-closure mutex that would serialize parallel agents while still
removing the per-token variadic/output-slice allocation from the compiled call
path. The focused sampler bench moved the production `top_k=64`, `top_p=0.95`
shape into the compiled/CallOne band: `BenchmarkSampler_TopKThenTopP_Vocab262k`
records repeated rows around `462-492us/op`, `8 B/op`, and `1 alloc/op`, and
`BenchmarkSampler_TopKThenTopPWithSuppression_Vocab262k` records about
`466-485us/op`, `10 B/op`, and `1 alloc/op`, versus the previous uncompiled
rows in the `478-519us/op`, `24 B/op`, `3 alloc/op` band and suppressed rows
around `528-530us/op`, `26-27 B/op`, `3 alloc/op`. The retained request-context
proof
`/private/tmp/go-mlx-goal/reports/2026-05-25-state-ramp-request-context-compiled-sampler-go-mlx-gemma4-e2b-4bit-opencode-30k-r10-g1024.json`
keeps the production invariants (`fixed_caches=0`, `paged_caches=15`,
`max_local_capacity=512`, `max_global_capacity=131072`,
`local_window_leaked=false`) and records `87.483` raw decode tok/s plus
`75.257` effective turn tok/s over `10/10` turns. Against the previous
local-window cleanup row this is a `+1.063%` raw decode improvement and
`+1.506%` effective-throughput improvement, but not a wall-time win: the same
seed generated `4476` visible tokens instead of `4292`, so total wall rose to
`71.727s`. Keep this as a default sampler/runtime cleanup, not as production
completion or as a replacement for the remaining llama.cpp raw-decode parity
work.

Rejected native sampler fusion, 2026-05-25: moving suppress-token filtering,
top-k/top-p, and categorical sampling behind a new C++ `mlx::core::compile`
wrapper improved the suppressed sampler microbench only marginally
(`497510 ns/op` versus the normal compiled suppressed row around `466-485us/op`
and `0` visible Go allocs), while making the real retained decode path slower.
The matched two-turn request-context trace
`/private/tmp/go-mlx-goal/reports/2026-05-25-state-ramp-request-context-native-suppressed-topk-topp-opencode-turn2-go-mlx-gemma4-e2b-4bit-opencode-30k-r2-g1024.json`
kept the same `1166` generated/visible tokens and paged invariants
(`fixed_caches=0`, `paged_caches=15`, `max_local_capacity=512`,
`local_window_leaked=false`) but fell to `86.285` raw decode tok/s and
`77.998` effective turn tok/s versus the accepted zero-empty-SDPA row at
`91.599` raw and `82.476` effective. The phase summary also moved `forward`
from about `1.398ms/token` to `1.714ms/token` and `prefetch` from about
`6.073ms/token` to `6.397ms/token`. Do not revive this sampler shape as a
native boundary; the useful target remains a larger stable logits/eval boundary
that does not perturb the one-token forward graph.

Rejected sampled-token lookahead prefetch, 2026-05-25: a retained-session probe
tried to build the next sampled token immediately after next-logits construction
and include that token in the existing async prefetch/eval boundary, so the next
loop could consume a materialised token instead of paying `sample_eval`. The
gate-on trace
`/private/tmp/go-mlx-goal/reports/2026-05-25-state-ramp-request-context-prefetch-sampled-token-opencode-turn2-go-mlx-gemma4-e2b-4bit-opencode-30k-r2-g1024.json`
failed before speed was meaningful: turn 1 produced `empty_visible_output`,
`0` generated tokens, and stopped at `31186` live tokens. The same rebuilt
binary with the gate off completed the matched two-turn run at
`/private/tmp/go-mlx-goal/reports/2026-05-25-state-ramp-request-context-prefetch-sampled-token-gateoff-opencode-turn2-go-mlx-gemma4-e2b-4bit-opencode-30k-r2-g1024.json`
with `1166` generated/visible tokens, `89.023` raw decode tok/s, `80.311`
effective turn tok/s, and the same paged invariants. Do not ship sampled-token
lookahead without first proving token/RNG equivalence on the first sampled step;
the current production path stays on logits-only async prefetch plus the
accepted compiled sampler.
Follow-up guard, 2026-05-25: `TestSample_PrefetchTokenEvalParity_Good` now
seeds MLX, samples from lazy logits through the normal
`sampleTokenIDWithSuppressionGuard` path, then re-seeds and samples while
evaluating logits plus the sampled token together. This guards the first-token
token/RNG equivalence required before any future lookahead or fused sampler/eval
boundary can be benchmarked in retained State. Verified with
`GOCACHE=/private/tmp/codex-go-mlx-cache GO_MLX_RUN_METAL_TESTS=1 go test ./go/internal/metal -run 'TestSample_(PrefetchTokenEvalParity|NewSamplerWithSuppressionBeforeTopPTopK|NewSamplerSkipsUnitTemperature)'`
and the same focused command without `GO_MLX_RUN_METAL_TESTS`.
Retained-session follow-up guard, 2026-05-25:
`TestModelSession_PrefetchTokenStateAdvanceParity_Good` now extends that check
through the retained state-advance boundary. It compares normal two-token
`ModelSession.Generate` against a manual path that samples the first token,
calls `advanceTokenLocked`, then evaluates the next logits, next sampled token,
and paged dirty K/V handles together before reading the second token. This
proves the first retained-session state-advance shape needed for a future
lookahead experiment, without enabling lookahead in production. Verified with
`GOCACHE=/private/tmp/codex-go-mlx-cache GO_MLX_RUN_METAL_TESTS=1 go test ./go/internal/metal -run 'TestModelSession_(PrefetchTokenStateAdvanceParity|Generate_AsyncDecodePrefetch|Generate_TraceTokenPhases)|TestSample_PrefetchTokenEvalParity'`
and the same focused command without `GO_MLX_RUN_METAL_TESTS`.

Rejected scalar sampled-token sync, 2026-05-25: replacing the explicit
`Eval(next)` in the first guarded sampler path with direct `next.Int()` scalar
materialisation looked good in isolation. The focused Metal bench recorded
`BenchmarkSampler_TopKThenTopPTokenReadNoEvalChecked_Vocab262k` at
`483482 ns/op`, versus `BenchmarkSampler_TopKThenTopP_Vocab262k` at
`495797 ns/op` and the suppressed sampler row at `487873 ns/op`. The matched
two-turn retained request-context trace rejected the runtime change:
`/private/tmp/go-mlx-goal/reports/2026-05-25-state-ramp-request-context-scalar-token-read-turn2-go-mlx-gemma4-e2b-4bit-opencode-30k-r2-g1024.json`
kept `2/2` turns, `1166` visible/generated tokens, `fixed_caches=0`, and
`paged_caches=15`, but fell to `89.175` raw decode tok/s and `80.465`
effective turn tok/s versus the current default row at `91.024` raw and
`81.968` effective. The scalar-sync path also increased total token-phase
duration from `10.967ms/token` to `11.194ms/token` and prefetch from
`6.220ms/token` to `6.327ms/token`. Keep the benchmark as a hot-path probe, but
do not replace explicit sampled-token eval with scalar-read synchronisation in
the production retained path.

Sample/logits eval-boundary benchmark, 2026-05-25: the next safe lookahead
shape was measured as a benchmark-only probe before touching the retained
runtime loop. `BenchmarkSampler_PrefetchLogitsThenSampleEval_WithSuppression_Vocab262k`
models the current boundary of prefetching logits first, then evaluating the
sampled token; `BenchmarkSampler_CombinedLogitsSampleEval_WithSuppression_Vocab262k`
models building the sampled token before the eval boundary and prefetching
logits plus sampled token together. On Apple M3 Ultra these rows were
`516277 ns/op`, `18 B/op`, `2 allocs/op` versus `511315 ns/op`, `17 B/op`,
`2 allocs/op`. Adding a dirty paged K/V cache to match the retained production
prefetch boundary gives
`BenchmarkSampler_PrefetchLogitsDirtyThenSampleEval_WithSuppression_Vocab262k`
at `517691 ns/op`, `17 B/op`, `2 allocs/op` versus
`BenchmarkSampler_CombinedLogitsSampleDirtyEval_WithSuppression_Vocab262k` at
`515825 ns/op`, `18 B/op`, `2 allocs/op`. This is too small to justify another
runtime lookahead attempt after the previous retained trace failure; keep the
benchmark rows as boundary evidence and leave production on logits-only
prefetch plus explicit sampled-token eval.

Attention dtype-alignment probe, 2026-05-25: the accepted fp16 retained-KV path
keeps `attentionQueryForKV` casting float32 query tensors down to the K/V dtype
before SDPA. A correctness guard now proves MLX can evaluate mixed
`Q=float32`, `K/V=float16` directly:
`TestFast_ScaledDotProductAttentionMixedKVF16_Good`. The focused fast-concat
bench rejects removing the cast, though. On Apple M3 Ultra,
`BenchmarkSDPAPagedFastConcat_8Pages_Page1024_QF32KVF16_CastQ` records
`435944 ns/op` with `100946072 mlx_peak_B`, while the direct mixed row records
`640400 ns/op` with `235958424 mlx_peak_B`. At 16 pages the cast row records
`645359 ns/op` with `201875736 mlx_peak_B`, while mixed Q/KV records
`995736 ns/op` with `269508888 mlx_peak_B`. Keep the query cast: MLX supports
the mixed dtype shape, but it is slower and materially increases active-cache
pressure in the retained attention path.

Rejected local RoPE precompute probe, 2026-05-25: the IDEAS.md dual-RoPE note
suggested checking whether local/default Gemma 4 RoPE was still building
frequency state inside the decode path. A correctness guard now proves
`RoPEWithFreqs` using the default 10k frequency tensor matches the existing
base-driven local RoPE path at non-zero offset:
`TestFast_RoPE_DefaultFreqsMatchesBasePath_Good`. The focused bench rejects
using it as a runtime optimisation, though:
`BenchmarkRoPE_Decode_BaseLocal10k` stays in the `169-172us/op` band and
`BenchmarkRoPE_Decode_BaseLocal10k_WithFreqs` records the same `168-171us/op`
band, both at `0 allocs/op`. The p-RoPE global shape remains the fast explicit
frequency case (`BenchmarkRoPE_WithFreqs_Decode_D256` around `6.6us/op`), but
local/default RoPE does not get that benefit. Keep Gemma 4 runtime construction
on precomputed `RopeFreqs` only for proportional p-RoPE; do not add load-time
frequency tensors for local/default layers unless a future MLX kernel changes
this result.

Slow-vs-fast attention microbench follow-up, 2026-05-25: the new
`BenchmarkSDPAPaged*Page1024_Q1_D128(_F16)` rows pin down the known old
page-reduction path against the accepted fast-concat lane. With float32 pages,
fast-concat is only modestly faster (`8` pages: `560786 ns/op` to
`511595 ns/op`; `16` pages: `858594 ns/op` to `839743 ns/op`) and carries a
larger active-cache footprint. With the production retained `fp16` K/V shape,
the win is material: `8` pages moves from `616440 ns/op` to `402212 ns/op`, and
`16` pages moves from `966353 ns/op` to `606435 ns/op`, with `0 allocs/op` on
the old page path and `2 allocs/op` on the concat wrapper. This confirms the
current production choice is better than the old slow path for q4/fp16 retained
State, while also confirming the finite next target: keep fast-concat-like
runtime without paying the larger materialised active-cache footprint.
Native paged-attention follow-up, 2026-05-25: warmed standalone native C++
attention has the desired isolated shape but still rejects as a production
graph path. The same bench family now records warmed native rows at `401042
ns/op` for `8` float32 pages and `561197 ns/op` for `16`, both with
`0 allocs/op` and without the fast-concat active-cache footprint. On the
production retained `fp16` K/V shape, warmed native is also faster than
fast-concat: `8` pages records `366340 ns/op` versus `407679 ns/op`, and `16`
pages records `485718 ns/op` versus `610271 ns/op`, again at `0 allocs/op`.
The real retained run rejects flipping the gate:
`/private/tmp/go-mlx-goal/reports/2026-05-25-state-ramp-request-context-native-paged-attn-turn2-go-mlx-gemma4-e2b-4bit-opencode-30k-r2-g1024.json`
sets `GO_MLX_ENABLE_NATIVE_PAGED_ATTENTION=1`, completes `2/2` turns, reaches
`33963` live tokens, generates `1304` visible tokens, but falls to `53.200`
raw decode tok/s and `50.277` effective turn tok/s over `38.162s`. The matched
q4 graph-path trace generated `1069` visible tokens at `90.256` raw decode
tok/s and `80.650` effective tok/s over `25.443s`. Token phases explain the
rejection: native paged attention moves the retained path to `14.475ms/token`
average `prefetch_logits` versus `6.169ms/token` on the accepted q4 graph row,
while `forward` only moves from `1.470ms` to `1.787ms`. Interpretation: the
C++ native paged-attention closure is useful evidence for the target memory
shape, but using it as a separate compiled side graph breaks the larger lazy
decode boundary. The next implementation must keep this memory shape inside the
single-token model graph rather than replacing fast-concat with the current
native gate.
Shared-owner guard follow-up, 2026-05-25: the first native-paged retained
rejection was partly self-inflicted. When the native side graph handled a full
owner layer that later Gemma 4 shared-KV layers reused, it returned only the
page-state output and did not populate `kv.Keys`/`kv.Values`; the later shared
layers therefore lost the owner fast-concat handles and kept traversing pages.
The Go graph now threads a `materializePagedKVForReuse` bit from the
`PreviousKVs`/`sharedSources` layout into attention, so native paged attention
cannot steal an owner path that must publish reusable K/V handles. The guarded
diagnostic run
`/private/tmp/go-mlx-goal/reports/2026-05-25-state-ramp-request-context-native-paged-attn-shared-owner-guard-turn2-go-mlx-gemma4-e2b-4bit-opencode-30k-r2-g1024.json`
improves the native-paged opt-in lane from `53.200` to `78.105` raw decode
tok/s and from `50.277` to `70.542` effective turn tok/s, while reducing total
wall from `38.162s` to `26.885s`. It is still rejected for production because
the accepted q4 graph-path trace remains faster at `90.256` raw decode tok/s,
`80.650` effective tok/s, and `25.443s`; `prefetch_logits` is still
`7.860ms/token` with the native guard versus `6.169ms/token` on the accepted
path. Keep the guard because it fixes the diagnostic branch and encodes the
shared-KV invariant, but do not enable native paged attention by default.

Compiled-sampler diagnostic, 2026-05-24: MLX `CompileShapeless(..., true)`
cannot cover this top-k/top-p sampler graph (`Slice cannot infer output
shapes`). Shape-specific compile does run and is now tracked by
`BenchmarkSampler_CompiledTopKThenTopP_Vocab262k`; the repeated bench records
regular sampler rows at `547902`, `528375`, and `533011 ns/op` with `3 allocs`,
versus compiled diagnostic rows at `484221`, `485097`, and `496835 ns/op` with
`2 allocs`. A real two-turn retained trace at
`/private/tmp/go-mlx-goal/reports/2026-05-24-state-ramp-request-context-compiled-standard-sampler-trace-turn2-go-mlx-gemma4-e2b-4bit-opencode-30k-r2-g1024.json`
rejects promoting it by default: the same `1322` visible-token fixture records
`88.081` raw decode tok/s and `80.473` effective turn tok/s, below the
non-compiled sampler row despite a tiny `sample_eval` movement
(`9.754ms` versus `9.758ms`). Keep the benchmark as a diagnostic for the
IDEAS.md compile-first lane, but do not route production sampling through a
shape-specific compiled closure.

Prepared-sampler prefetch diagnostic, 2026-05-24: a retained-session experiment
split the deterministic top-k/top-p candidate work from the random categorical
draw and queued those candidate tensors in the existing async next-logits
prefetch. The microbench looked useful (`PreparedTopKThenTopPTokenOnly` at
`244001 ns/op`, `0 B/op`, `0 allocs/op` versus the normal top-k/top-p row at
`545400 ns/op`, `24 B/op`, `3 allocs/op`), but the real retained trace rejected
it. `/private/tmp/go-mlx-goal/reports/2026-05-24-state-ramp-request-context-prepared-sampler-trace-turn2-go-mlx-gemma4-e2b-4bit-opencode-30k-r2-g1024.json`
completed `2/2` turns with paged K/V, `fixed_caches=0`,
`local_window_leaked=false`, and `831` visible tokens, but raw decode fell to
`81.33817878691531 tok/s`; `prefetch` rose to `7352243 ns/token` and
`sample_eval` stayed high at `3370402 ns/token`. Interpretation: prefetching
the deterministic sampler candidate graph just moves more MLX work into the
same next-token materialisation boundary; it is not the larger stable graph
fix that IDEAS.md is pointing at. Do not keep this path in production code.

Latest prompt-contract note: do not promote output token-count floors into
acceptance criteria. If a fixture does not give the model enough real turn
content to continue for ten turns, that is a fixture failure, not a model or
runtime result. `scripts/state_ramp_fixture.py` now records structural fixture
facts (`section_count`, `unique_request_count`, dropped bytes, extraction
status, and retained context-excerpt bytes) and no longer derives a recommended
token floor. It can write either a thin `request-only` diagnostic stream or a
bounded `request-context` stream that keeps same-turn context excerpts without
reintroducing the old undifferentiated raw dump shape. The new
`scripts/gemma4_prompt_contract.py` compares the retained Gemma 4 seed plus
append-turn helpers against the local `chat_template.jinja` through
`AutoTokenizer.apply_chat_template(...)`; reference, direct, and direct plus
thinking mode all matched byte-for-byte against the local
`mlx-community/gemma-4-e2b-it-4bit` snapshot. Current short/early-stop rows
should therefore be investigated as fixture/content quality, sampling/state,
or runtime behaviour, not as a live Gemma 4 chat-template mismatch.

Latest local code note: a Gemma 4 shared-KV lifetime bug was fixed after the
native fixed-cache path could hand cache-owned K/V handles to shared layers and
later treat those handles as caller-owned intermediate state. The fix retains
only owner K/V handles that are read by later shared layers and marks native
fixed-cache handles as borrowed. A short rebuilt `driver-profile` smoke now
passes without the previous layer-6 shared-KV panic; treat it as a regression
guard, not a production benchmark row.

Latest prompt-template note: the Gemma 4 native prompt renderers were tightened
against the local model `chat_template.jinja`. `add_generation_prompt` is now
rendered as `<|turn>model\n` only; go-mlx no longer pre-seeds a synthetic empty
`<|channel>thought\n<channel|>` block for no-thinking mode. The Gemma 4
formatter also strips thought-channel content from assistant history before it
is replayed into a fresh prompt. This removes a real chat-template diff that
could bias short/zero visible-output probes and makes llama.cpp thinking leakage
an external comparator issue rather than a go-mlx prompt shape. Verification:
`go test ./go/... -count=1`, `git diff --check`,
`go test ./go/chat -bench 'BenchmarkChat_Format_Gemma4_5Turns|BenchmarkChat_TemplateName|BenchmarkChat_NormaliseRole' -benchmem -run '^$'`
(`BenchmarkChat_Format_Gemma4_5Turns`: `300.2 ns/op`, `2304 B/op`,
`1 alloc/op`), and focused state/chapter Gemma 4 prompt tests.

Comparator prompt-contract follow-up: the llama.cpp and `mlx_lm` opencode
workflow harnesses had drifted from the Go `state-ramp-profile` retained-turn
wrapper. They still used the older "retained project context" wrapper while
the Go path uses the stricter current prompt that suppresses scaffold output,
false completion claims, and reference continuation. Both Python comparator
harnesses now import `scripts/state_ramp_prompts.py`, sharing the retained
system prompt, Gemma 4 turn wrappers, and visible-control-channel stripping.
This does not close the raw decode gap by itself, but it removes a real
same-workload benchmark skew before the next llama.cpp rerun. Verification:
`python3 -m py_compile scripts/state_ramp_prompts.py scripts/llamacpp_opencode_workflow_bench.py scripts/mlx_lm_opencode_workflow_bench.py`
and `go test ./go/cmd/mlx -run 'TestStateRampProfileTurnPromptGemma4|TestStateRampProfileInitialPrompt' -count=1`.

Latest retained chat-template note: stop-token handling was still capable of
double-closing Gemma 4 assistant turns. `ModelSession.Generate` sampled
`<turn|>` as a stop token, advanced that token into retained KV state, then
`state-ramp-profile` appended the normal assistant close suffix
`<turn|>\n`, leaving `<turn|><turn|>\n` in live history. Retained sessions now
match the non-session generator: sampled EOS/stop tokens are withheld from the
visible stream and do not advance retained state, so callers append exactly one
template close suffix. The `mlx_lm` comparator was also tightened for the same
stateful-cache shape: when `stream_generate` has already consumed `<turn|>`,
the harness appends only the newline continuation instead of a second turn
marker. The checked BOS difference is not promoted as a bug: `llama-tokenize`
auto-adds BOS for the local Q4_K_M GGUF, so the llama.cpp comparator should not
also inject a literal `<bos>` unless tokenisation is forced with `--no-bos`.
Verification:
`go test ./go/internal/metal -run 'TestModelSession_Generate_(StopTokenDoesNotAdvanceRetainedState|GoodUsesLazyNativeGreedyState|TraceTokenPhases|AsyncDecodePrefetch)' -count=1`,
`go test ./go/cmd/mlx -run 'TestStateRampProfileTurnPromptGemma4|TestStateRampProfileInitialPrompt|TestRunCommand_DriverProfileFastGemma4Lane' -count=1`,
and `python3 -m py_compile scripts/mlx_lm_opencode_workflow_bench.py scripts/llamacpp_opencode_workflow_bench.py scripts/state_ramp_prompts.py`.

Latest chat-template parity check: the retained State prompt shape was compared
against the local Gemma 4 `chat_template.jinja`; the current state-ramp seed
and turn wrappers are valid native renderings for the message roles they use.
One remaining shared formatter diff was found and fixed: consecutive assistant
messages are now rendered as a continuation of the existing model turn, matching
the Jinja rule that suppresses a duplicate `<|turn>model\n` block. The
post-stop-fix retained workflow row
`/private/tmp/go-mlx-goal/reports/2026-05-24-state-ramp-after-stopfix-go-mlx-gemma4-e2b-4bit-opencode-delimited-30k-to-70k-r10-g1024.json`
completed `10/10` turns from `30k` to `61652` live tokens at `81.279 tok/s`
raw decode, `58.767 tok/s` effective turn throughput, `73.066s` wall time,
`3.834 GB` peak MLX memory, `10.046 GB` active-plus-cache, and an estimated
`3.395x` retained-vs-replayed speedup. It is not an acceptance row: turn `7`
returned only a Markdown fence, so `state-ramp-profile` now tags fence-only
visible output as `visible_fence_only` instead of letting that content-quality
failure hide behind a successful token stream. Focused verification:
`go test ./go/chat -run 'TestFormat_Gemma4Template' -count=1`,
`go test ./go/cmd/mlx -run 'TestStateRampProfileOutputIssues' -count=1`,
and hot-path checks showing `BenchmarkChat_Format_Gemma4_5Turns` at
`282.9-289.0 ns/op`, `2304 B/op`, `1 alloc/op`, and
`BenchmarkStateRampProfileOutputIssues_FullResponse` at `1943-1947 ns/op`,
`192 B/op`, `1 alloc/op`.

Latest benchmark-quality note: the same post-stop-fix row above was reclassified
with stricter output-quality accounting before the next acceptance rerun. The
old report carried `output_issues: null`, but the captured text shows `2`
prompt-analysis turns, `2` false-completion/success-claim turns, `6`
fence-prefixed turns despite the turn material saying "Do not output code
blocks", and `1` fence-only turn. `state-ramp-profile` now emits
`summary.output_issue_turns` and `summary.output_issue_counts`, and the
llama.cpp / `mlx_lm` comparator harnesses import the same shared detector from
`scripts/state_ramp_prompts.py`. Acceptance rows must report these counts
side-by-side with decode, wall time, memory, and energy; a faster row with
unexplained prompt-analysis or fence-only output is benchmark evidence, not
product evidence. Verification:
`go test ./go/cmd/mlx -run 'TestStateRampProfileOutputIssues|TestStateRampProfileSummary_OutputIssueCounts|TestStateRampProfileSummary_ReplayEstimate' -count=1`,
`python3 -m py_compile scripts/state_ramp_prompts.py scripts/llamacpp_opencode_workflow_bench.py scripts/mlx_lm_opencode_workflow_bench.py`,
and
`go test ./go/cmd/mlx -bench 'BenchmarkStateRampProfileOutputIssues_FullResponse' -benchmem -run '^$' -count=3`
(`2878-2892 ns/op`, `192 B/op`, `1 alloc/op`).

Comparator prompt-mode parity note: Go `state-ramp-profile` already exposes
`-turn-prompt-mode reference|direct`, and the Python `mlx_lm` / llama.cpp
opencode harnesses now expose the same flag through the shared
`gemma4_turn_prompt(..., mode)` helper. This is required before the next
quality-focused rerun: if the reference wrapper keeps eliciting prompt-analysis
or fenced-output artefacts, the direct mode can be tested against all runners
without changing any other benchmark dimension. Verification:
`python3 -m py_compile scripts/state_ramp_prompts.py scripts/llamacpp_opencode_workflow_bench.py scripts/mlx_lm_opencode_workflow_bench.py`
and a local direct/reference prompt render check.

Latest direct-mode quality rerun: the local Gemma 4 `chat_template.jinja` was
checked against the state-ramp retained seed shape and full replay shape; the
prompt template itself is not the current diff. A fresh direct-mode go-mlx row
`/private/tmp/go-mlx-goal/reports/2026-05-24-state-ramp-direct-after-quality-go-mlx-gemma4-e2b-4bit-opencode-delimited-30k-to-70k-r10-g1024.json`
completed `10/10` turns from `30k` to `62028` live tokens, generated `5495`
tokens, and records `82.262 tok/s` raw decode, `66.360 tok/s` effective turn
throughput, `95.142s` retained wall time, `2431.804 tok/s` cold prefill,
`1657.532 tok/s` average append/prefill, `9.996 GB` active-plus-cache memory,
and a `2.804x` retained-vs-replayed speedup estimate. It removes the previous
reference-wrapper prompt-analysis and code-fence artefacts, but it is still not
an acceptance row: turn `7` was asked for `700` to `1000` tokens of prose and
instead looped a table cell (`LLM`) to the token budget. Both Go and Python
quality accounting now tag this as `visible_repeated_table_cell`, so the row is
benchmark evidence for direct-mode throughput only, not product evidence.
Verification:
`go test ./go/cmd/mlx -run 'TestStateRampProfile(OutputIssues|InitialPromptGemma4|Summary_OutputIssueCounts)' -count=1`,
`python3 -m py_compile scripts/state_ramp_prompts.py scripts/llamacpp_opencode_workflow_bench.py scripts/mlx_lm_opencode_workflow_bench.py`,
`go test ./go/cmd/mlx -bench 'BenchmarkStateRampProfileOutputIssues_FullResponse' -benchmem -run '^$' -count=3`
(`3097-3194 ns/op`, `192 B/op`, `1 alloc/op`), `go test ./go/... -count=1`,
`git diff --check`, and `go build -o /private/tmp/go-mlx-goal/bin/lthn-mlx ./go/cmd/mlx`.

Aligned llama.cpp direct-mode anchor, 2026-05-24:
`/private/tmp/go-mlx-goal/reports/2026-05-24-llamacpp-direct-after-quality-gemma4-e2b-q4km-opencode-delimited-30k-to-70k-r10-g1024.json`
was run against the same prompt files, `30k -> 70k`, `10` turns, `1024`
token budget, sampling, direct Gemma 4 turn wrapper, and shared output-quality
detector. The row completed `10/10` clean turns with `0` output-issue turns,
`7586` generated tokens, `7576` visible tokens, `64119` final live tokens,
`104.894s` wall, `104.462 tok/s` decode from llama.cpp timings,
`72.226` wall visible tok/s, `31.647s` prompt/cache work, and `10489.356 J`
at the normalised `100 W` estimate. This shows the direct-mode table-cell loop
is not a generic prompt-shape failure: llama.cpp answered the same turn `7` as
prose and did not trip `visible_repeated_table_cell`. Against the go-mlx
direct row above, llama.cpp is `1.270x` faster on raw decode, while go-mlx is
`1.102x` faster on retained total wall for this row; because go-mlx turn `7`
is quality-rejected, that wall comparison is diagnostic only. The llama.cpp
script's internal `ps` memory probe is blocked by this sandbox, so the JSON
records unavailable memory; external `ps` polling during the run observed RSS
climbing to about `5.005 GB` and VSZ to about `448.343 GB`. The harness now
records the memory probe error explicitly on future sandboxed runs instead of
silently returning empty memory fields. Verification:
`python3 -m py_compile scripts/llamacpp_opencode_workflow_bench.py scripts/state_ramp_prompts.py scripts/mlx_lm_opencode_workflow_bench.py`
and a local probe check returning
`PermissionError: [Errno 1] Operation not permitted: 'ps'`.

Latest Gemma 4 stop-template finding, 2026-05-24: the literal retained/direct
prompt wrappers still match the local `chat_template.jinja`, but the retained
harness stop set did not match the model metadata. The local MLX pack declares
top-level `eos_token_id` as `[1, 106, 50]`, mapping to `<eos>`, `<turn|>`,
and `<|tool_response>`. go-mlx previously stopped only on `<turn|>` and
suppressed `<|tool_response>` as a forbidden visible control token. The
State/chapter token controls now stop on all three model-declared Gemma 4 EOS
markers and only suppress non-stop control/template tokens. Trace token phases
also record `token_id` / `token_text`, so an immediate no-visible-output turn
can identify the sampled stop token instead of leaving `sampled_token_ids`
empty. Diagnostic evidence:
`/private/tmp/go-mlx-goal/reports/2026-05-24-state-ramp-direct-after-stopset-trace-turn1-go-mlx-gemma4-e2b-4bit-opencode-delimited-30k-r1-g1024.json`
replays the seeded direct row's first turn and records sampled token `1`
(`<eos>`, empty decoded text) as the final token after `30,954` live tokens.
That means the older seeded direct row was not clean product evidence: it let
an empty EOS token flow into retained state instead of treating the turn as a
natural model stop. The same patch also tags the no-seed turn-7 repeated
`| **Verdict** | ... |` table-row stutter as
`visible_repeated_table_row_label`; the no-seed diagnostic remains rejected by
turn `10` `empty_visible_output`. Verification:
`go test ./go/cmd/mlx -run 'TestStateRampProfile(OutputIssues|Summary_OutputIssueCounts)|TestChapterProfileTemplateTokenControlsGemma4UsesAllModelStops' -count=1`,
`go test ./go/internal/metal -run 'TestModel_Generate_TraceTokenPhases|TestModelSession_Generate_(TraceTokenPhases|StopTokenDoesNotAdvanceRetainedState)' -count=1`,
`go test ./go/... -count=1`,
`go test ./go/cmd/mlx -bench 'BenchmarkStateRampProfileOutputIssues_FullResponse' -benchmem -run '^$' -count=3`
(`2872-2877 ns/op`, `192 B/op`, `1 alloc/op`),
`python3 -m py_compile scripts/state_ramp_prompts.py scripts/llamacpp_opencode_workflow_bench.py scripts/mlx_lm_opencode_workflow_bench.py`,
`git diff --check`, and
`go build -o /private/tmp/go-mlx-goal/bin/lthn-mlx ./go/cmd/mlx`.

Comparator stop-policy follow-up: the Python comparator harnesses now import
the same Gemma 4 stop/suppress token contract from `scripts/state_ramp_prompts.py`.
`GEMMA4_STOP_TOKEN_TEXTS` is `("<eos>", "<turn|>",
"<|tool_response>")`, resolving to `[1, 106, 50]` on the local
`mlx-community/gemma-4-e2b-it-4bit` tokenizer. `mlx_lm` no longer logit-biases
token `50` as suppressed while also loading the tokenizer with the model's EOS
list, and the llama.cpp server harness now sends the full stop-string list
instead of only `"<turn|>"`. Both comparator harnesses also mark empty visible
output as `empty_visible_output` rather than counting a zero-content stop as a
successful turn. Verification:
`python3 -m py_compile scripts/state_ramp_prompts.py scripts/mlx_lm_opencode_workflow_bench.py scripts/llamacpp_opencode_workflow_bench.py`,
local tokenizer helper check resolving stop IDs to `[1, 106, 50]` and proving
`50` is excluded from suppress IDs, and a row-label detector check returning
`['visible_repeated_table_row_label']`. A live one-turn `mlx_lm` rerun was not
accepted as evidence because the current Homebrew/Python path imports a broken
`mlx_lm` install (`ModuleNotFoundError: No module named 'mlx.utils'`); rerun
the comparator from the repaired parity environment before promoting a new
external row.

Chat-template diff follow-up: the immediate first-turn `<eos>` is not caused
by a retained Gemma 4 template mismatch. Rendering the same seed and first turn
through the local `chat_template.jinja` and through
`AutoTokenizer.apply_chat_template(..., add_generation_prompt=true)` produces
the exact byte stream used by the retained State prompt: one leading `<bos>`,
the retained system turn, `Ready.<turn|>`, then the incremental user turn and
`<|turn>model\n` suffix without a second BOS in the middle. Greedy diagnostics
show the old opencode direct fixture is the problem shape, not the wrapper:
the real first delimited section chooses token `1` (`<eos>`) immediately at
both `30k` and `4k` live context, and sanitising the two literal
`<|channel>` / `<channel|>` strings in the seed does not change that result.
A request-only counterfactual using the same retained seed generates `781`
visible tokens at `108.204 tok/s` on the `4k` diagnostic, while
`-turn-prompt-mode reference` avoids the EOS but produces
`visible_prompt_analysis`. Treat the old direct opencode fixture as rejected
for product evidence: the next retained workflow benchmark should use a clean
request-plus-context turn fixture that does not append truncated raw GOAL
chunks as undifferentiated user text after the actual request. Relevant
diagnostic artefacts:
`/private/tmp/go-mlx-goal/reports/2026-05-24-state-ramp-direct-after-stopset-greedy-trace-turn1-go-mlx-gemma4-e2b-4bit-opencode-delimited-30k-r1-g1024.json`,
`/private/tmp/go-mlx-goal/reports/2026-05-24-state-ramp-direct-after-stopset-greedy-trace-turn1-go-mlx-gemma4-e2b-4bit-opencode-delimited-4k-r1-g1024.json`,
`/private/tmp/go-mlx-goal/reports/2026-05-24-state-ramp-reference-after-stopset-greedy-trace-turn1-go-mlx-gemma4-e2b-4bit-opencode-delimited-30k-r1-g1024.json`,
and
`/private/tmp/go-mlx-goal/reports/2026-05-24-state-ramp-direct-simpleturn-greedy-trace-turn1-go-mlx-gemma4-e2b-4bit-opencode-4k-r1-g1024.json`.

Clean fixture correction, 2026-05-24: `scripts/state_ramp_fixture.py` can now
build either a thin `request-only` append stream or a bounded `request-context`
append stream from noisy opencode delimited material. The `request-only`
fixture is useful as a prompt-contract diagnostic, but it is not accepted
production material because it reduces `94,877` bytes of old mixed request/GOAL
chunks to `1,955` bytes of directives and can starve later turns of real
context. The new
`/private/tmp/go-mlx-goal/opencode-turns-request-context.txt` fixture extracts
the same `10` user requests while retaining up to `4096` bytes of same-turn
context per section; its metadata records `43,620` output bytes,
`39,445` context-excerpt bytes, and `8` truncated context sections. The prior
retained `30k` request-only state run completed `10/10` turns with no
control/fence/loop detector issues:
`/private/tmp/go-mlx-goal/reports/2026-05-24-state-ramp-request-only-go-mlx-gemma4-e2b-4bit-opencode-30k-r10-g1024.json`
records `36,667` final live tokens, `556` appended tokens, `6,091` generated
and visible tokens, `87.8565 tok/s` raw decode, `86.9605` effective turn
tok/s, `82.249s` wall, `9.863 GB` active-plus-cache, `3.387 GB` peak RSS, and
`2.373x` retained-vs-replay speedup. The aligned llama.cpp Q4_K_M row
`/private/tmp/go-mlx-goal/reports/2026-05-24-llamacpp-request-only-gemma4-e2b-q4km-opencode-30k-r10-g1024.json`
records `10/10` turns, `39,501` final live tokens, `8,925` generated tokens,
`8,914` visible tokens, `111.760 tok/s` raw decode from llama.cpp timings,
`96.107` wall visible tok/s, and `92.751s` wall. This row remains diagnostic,
not production acceptance: go-mlx is `1.128x` faster by wall time and saves
about `11.32%` wall-energy at the normalised `100 W`, but llama.cpp is
`1.272x` faster on raw decode and `1.105x` faster on wall-visible throughput.
Do not rescue or reject this row with a visible-token floor. The next accepted
row should use the richer `request-context` fixture, captured output, the shared
content-quality detectors, and a short human-readable note on whether each turn
actually answered its request.

Suppress-EOS diagnostic follow-up, same date: `-suppress-eos` now suppresses
the full effective Gemma 4 EOS/stop list instead of only the literal `<eos>`
token. The request-context trace
`/private/tmp/go-mlx-goal/reports/2026-05-24-state-ramp-request-context-suppresseos-eoslist-trace-turn2-go-mlx-gemma4-e2b-4bit-opencode-30k-r2-g1024.json`
shows the runtime suppress list includes `[1, 106, 50]` and the two-turn run no
longer fails with immediate empty output. This is not an accepted product row:
forcing all stop markers drove both turns into a repeated short-line
quote/paren cycle at the token budget. `state-ramp-profile` and the Python
comparator detector now tag that shape as
`visible_repeated_short_line_cycle`, so a forced-stop diagnostic cannot look
clean simply because it produced 1024 visible tokens. Verification:
`go test ./go/cmd/mlx -run 'Test(StateRampProfileEffectiveSuppressTokenIDsIncludesGemma4EOSList|ChapterProfileTemplateTokenControlsGemma4UsesAllModelStops|StateRampProfileOutputIssues)' -count=1`,
`python3 -m py_compile scripts/state_ramp_prompts.py scripts/llamacpp_opencode_workflow_bench.py scripts/mlx_lm_opencode_workflow_bench.py scripts/state_ramp_fixture.py`,
Python reclassification of the trace returning
`[['visible_repeated_short_line_cycle'], ['visible_repeated_short_line_cycle']]`,
and `go test ./go/cmd/mlx -bench 'BenchmarkStateRampProfileOutputIssues_FullResponse' -benchmem -run '^$' -count=3`
(`3571-3659 ns/op`, `192 B/op`, `1 alloc/op`).

Latest State continuity note: `state-ramp-profile` now treats `-fold-store` as
the append-only State log it claims to be. Folding opens an existing `.mvlog`
and appends checkpoint/folded records instead of truncating it; only a missing
path is created. Fold reports now include `fold.store_action` plus
`fold.compact_marker.{store_path,index_uri,entry_uri,bundle_uri,token_count}`
so the next process can wake from the same State file and compact marker.
`state-wake-profile -marker-file <state-ramp-report.json>` now reads either the
full ramp report or a standalone marker JSON, fills `-state-store` and
`-index-uri` from the marker when they are not explicitly supplied, and keeps
older reports usable by falling back to `fold.folded.index_uri`. This is a
code-path guard for cross-session continuity; it still needs a fresh end-to-end
retained run before being promoted to production benchmark evidence. The next
storage R&D step is a segment-aware State resolver where one compact marker can
live in a small main index file while referenced State blocks live in other
`.mvlog` segment files.

One-file cross-session continuity smoke, 2026-05-24:
`/private/tmp/go-mlx-goal/reports/2026-05-24-state-continuity-onefile-ramp.json`
folded a small `512 -> 700` retained state into
`/private/tmp/go-mlx-goal/state-continuity-onefile-20260524-smoke.mvlog`
(`78M`), emitted compact marker
`mlx://state-ramp/fold/1779612942781065000/folded/index`, and confirmed both
checkpoint and folded refs used that same `.mvlog` segment. A separate process
then ran
`/private/tmp/go-mlx-goal/reports/2026-05-24-state-continuity-onefile-wake.json`
with `state-wake-profile -marker-file <ramp-report>` and no manual
`-state-store`/`-index-uri`; it resolved the same State file, woke `206`
folded prefix tokens with `restore_strategy=folded-prefill`, and generated
`32` visible tokens at `95.790 tok/s`. Treat this as proof that one-file
compact markers survive a process boundary and can seed session 2 from session
1's State log. Do not promote it to content-quality evidence: the wake output
was marked `visible_prompt_analysis`, so the prompt/template still needs a
product-quality follow-up.

State `.kv` container bridge, 2026-05-24:
`state-pack -marker-file <ramp-report> -output
/private/tmp/go-mlx-goal/state-continuity-onefile-20260524-smoke.kv` now uses
`forge.lthn.ai/Snider/Enchantrix/pkg/trix` directly with magic `KVST`. The
resulting container stores the compact marker metadata in the JSON head
(`kind=go-mlx/state-kv`, folded index
`mlx://state-ramp/fold/1779612942781065000/folded/index`) and the raw `.mvlog`
State log as the binary tail. The smoke packed `81,857,007` State payload bytes
into an `81,857,631` byte `.kv` file. The first format proof used the old
in-memory `Payload []byte` helper; the current code path now uses the streaming
`trix.EncodeStream` / `ReadHeaderInfo` helpers so production packs do not load
the full State payload into a Go slice.
Follow-up direct `.kv` wake now works as a bridge:
`/private/tmp/go-mlx-goal/reports/2026-05-24-state-continuity-onefile-kv-wake.json`
ran `state-wake-profile -marker-file
/private/tmp/go-mlx-goal/state-continuity-onefile-20260524-smoke.kv` and no
manual `-state-store`/`-index-uri`. The wake resolved the folded index from the
Trix header, opened the State segment at
`/private/tmp/go-mlx-goal/state-continuity-onefile-20260524-smoke.mvlog`, read
`206` folded prefix tokens with `restore_strategy=folded-prefill`, appended
`204` prompt tokens, and generated `32` visible tokens at `104.331 tok/s`
decode. The next rebuild replaced path restoration with an opt-in
go-inference filestore segment alias:
`/private/tmp/go-mlx-goal/reports/2026-05-24-state-continuity-onefile-kv-wake-alias.json`
materialized the `.kv` binary tail to a temporary State file, opened it with
`state_store_segment_alias=/private/tmp/go-mlx-goal/state-continuity-onefile-20260524-smoke.mvlog`,
confirmed the temp payload was removed after wake, restored the same `206`
folded prefix tokens, appended `204` prompt tokens, and generated `32` visible
tokens at `104.801 tok/s` decode. This is now relocatable at the filestore API
level while preserving strict segment validation.

Code update, same date: `state-wake-profile -marker-file <session.kv>` now
supersedes the temp-materialized bridge. It reads the Trix header only, passes
`state_store_payload_offset` and `state_store_payload_bytes` through the CLI
report/config, and opens the `.kv` file itself with
`filestore.OpenRegionWithSegmentAlias`. The State refs keep their original
`.mvlog` segment as an alias, but payload reads map to
`payload_offset + frame_offset` inside the container and the embedded region is
read-only. Focused tests cover aliased refs, physical refs, wrong-segment
rejection, URI lookup, and write rejection, and the broad Go lane passes on
`go1.26.3`. The new region benchmarks record `7016 ns/op` for 64 KiB
`ResolveRefBytes`, `658.8 ns/op` for a 1000-record 64-byte ref read, and
`4.346 ms/op` for a 10k-record region open. Remaining production work is the
true zero-copy/mmap/pinned handoff from this payload window into MLX-ready
State vectors.

Second code update, same date: go-inference dev `41a48af` now exposes
`BorrowBytes` / `BorrowRefBytes` and the read-only filestore region path
services borrows from an mmap of the embedded `.kv` State payload. `go-mlx` raw
State block loading now asks for borrowed bytes first, so native-encoded KV
tensor slices parsed from a `.kv` wake can flow into the existing
`core.PinnedView` / `mlx_array_new_data` restore path without the old per-block
heap copy. The
focused region benchmark now records `BorrowRefBytes` at `29.71 ns/op`,
`0 B/op`, `0 allocs/op` for 64 KiB blocks versus copied `ResolveRefBytes` at
`6666 ns/op`, `65536 B/op`, `1 alloc/op`; the 1000-record 64-byte row is
`31.61 ns/op`, `0 B/op`, `0 allocs/op` versus `650.2 ns/op`, `64 B/op`,
`1 alloc/op`.

Third State restore code update, same date: partial-prefix
`LoadPrefixFromStateBlocksWithOptions` now stream-assembles the covering State
blocks instead of first retaining a `[]Block` and all per-block snapshots for
`AssembleBlocks`. When the requested prefix lands inside the final covering
block, that block is sliced before append, so the wake path does not copy the
over-covering K/V bytes only to discard them in a second assembled snapshot
slice. Focused hot-path deltas on the Apple M3 Ultra:
`BenchmarkMultiblock_LoadPrefix_HalfBlocks` moved from `23802 ns/op`,
`101632 B/op`, `39 allocs/op` to `19197 ns/op`, `78064 B/op`,
`37 allocs/op`; `BenchmarkMultiblock_LoadPrefix_ThreeQuarterBlocks` moved from
`30271 ns/op`, `139798 B/op`, `46 allocs/op` to `26940 ns/op`,
`105430 B/op`, `44 allocs/op`; and the mixed save/load/slice/save lifecycle now
records `53698 ns/op`, `193201 B/op`, `103 allocs/op`. This is a restore-path
memory/copy reduction, not the final true mmap-to-MLX zero-copy handoff.

The content caveat remains: the short wake output is prompt-analysis text, so
this is format/continuity evidence only.

### Methodology Correction

Do not use arbitrary visible-token floors as benchmark acceptance criteria.
`-turn-min-tokens` and `-chapter-min-tokens` are debug guards for catching
broken decoders or empty output only; rows that were judged by a `256`, `512`,
`768`, or similar minimum visible-token floor are diagnostic, not production
sign-off evidence. Natural model stops are valid if the content is non-empty,
not a repeated-token loop, not a control/thinking-channel leak, and coherent
for the supplied prompt.

The production comparison must be one default runner path versus external
runner anchors on the same natural workload. Record wall time, prefill/append
time, raw decode, active MLX memory, MLX allocator cache, active-plus-cache,
process RSS/virtual memory, generated/visible token counts, stop reason, and a
short content note. Do not add new env gates or CLI switches to make a row pass;
temporary diagnostics must either be promoted into the default path or removed.

Memory is a cost curve, not a standalone win condition. A higher active
footprint during live inference is acceptable when it is bounded, explained, and
buying retained-State wall time, especially if it is a fixed full-context cost
around the model plus cache. The memory blockers are runaway growth, duplicate
K/V materialisation, allocator-cache pressure that hides real active use, and
virtual-memory explosions that make long agent sessions fragile.

Fresh working evidence lives under `/private/tmp/go-mlx-goal/reports/` until the
next canonical runtime report set is regenerated:

- `2026-05-24-state-kv-warm-after-kv-slab.json`: rebuilt `lthn-mlx` smoke after
  making default zero-copy paged State restore explicit and tightening native
  layer-slab State assembly for single-head slabs. This is not production
  acceptance because the baseline README prompt naturally stops after one token,
  but it confirms the current default State path still works and writes clean
  JSON: `6` State blocks, `2765` restored/avoided prompt tokens, `238920119`
  State-store bytes, `108.517ms` State K/V restore, `8.469x` restore speedup
  over the measured `918.985ms` prefill, `102.649 tok/s` warmed decode for the
  `256` token State-KV generation leg, `3420202578` bytes active MLX memory
  (`3.185 GiB`), and `3491881978` bytes peak MLX memory (`3.252 GiB`).
  External process polling during the run observed about `3.82 GiB` RSS and
  `459 GB` virtual reservation, roughly `100 GB` below the earlier problematic
  virtual-reservation class. Treat this
  as a default-path smoke and memory-direction check, not a same-shape runner
  comparison.
- `2026-05-24` in-process State restore micro evidence: session-owned paged
  cache restore now transfers locally owned page arrays into the live
  `PagedKVCache` instead of cloning them and then freeing the streamed entry.
  `BenchmarkSession_RestorePagedCaches_Copy_8x512` measured `11439 ns/op`,
  `950 B/op`, `22 allocs/op`; `BenchmarkSession_RestorePagedCaches_Transfer_8x512`
  measured `7965 ns/op`, `944 B/op`, `28 allocs/op`. This is a narrow ownership
  benchmark, not a runner score, but it validates the wake/fork State path is
  removing a Metal-array copy where page ownership is local.
- `2026-05-24-state-kv-warm-transfer-smoke-ctx32768.json`: rebuilt
  `lthn-mlx` smoke after the paged-State transfer path and fixed-sliding
  Gemma 4 prefill chunk cap. The first attempt with the default `4096`
  context was correctly rejected as an invalid restore shape because the
  prompt was `4960` tokens, so this row uses `-context 32768`. It completes a
  full `256` token generation without the previous chunked-prefill panic:
  `4960` prompt tokens, `11` State blocks, `172670094` State-store bytes,
  `20.157x` restore speedup, `4960` prompt tokens avoided,
  `105.215 tok/s` State-warmed decode, `105.124 tok/s` baseline decode,
  `7273829970` bytes active MLX memory, and `7333642190` bytes peak MLX
  memory in the warmed leg. Treat this as a holistic State-path regression
  guard for prompt sizes above the old default context, not as a same-shape
  llama.cpp comparison.
- `2026-05-24-state-ramp-lighthouse-distractor-c10.json`: retained-State
  coherence proof-of-work using a `10000` token seed arc and `10` later turns
  that each carried a different distractor prompt for entropy. The first
  entropy attempt was rejected as a prompt-shape failure because the model
  treated each distractor as the new chapter topic; the tightened row makes the
  seed arc explicit as the only plot and marks distractors as imagery/style
  pressure only. The accepted row completes `10/10` turns, `1781` generated and
  visible tokens, `14088` final live tokens, `95.563 tok/s` average decode,
  `89.370 tok/s` effective turn throughput, `23.529s` total turn wall time,
  `7.468 GiB` peak MLX memory, `10.209 GiB` active-plus-cache, about
  `3.163 GiB` process RSS, and `507.893 GB` process virtual reservation. Most
  importantly, chapter 10 resolves the original lighthouse keeper, signalling
  light, and deep-ocean presence instead of drifting into the final island
  distractor. The readable book artefact is
  `/private/tmp/go-mlx-goal/books/2026-05-24-lighthouse-signal.md`. Treat this
  as content-coherence evidence for retained State under distractor entropy,
  not as a llama.cpp comparison row.
- `scripts/state_book_from_phase0.py`: repeatable retained-State book generator
  for `/Users/snider/Code/lthn/LEM/training/lem/creative/phase0.json`. It picks
  one seed prompt as the only book arc, picks random distractor prompts for
  later chapters, writes replayable seed/turn material, runs
  `state-ramp-profile`, and extracts a readable `book.md` from the JSON report.
  Dry-run validation with `--random-seed 4242` writes deterministic material and
  the exact command without launching MLX. A short escalated Metal smoke with
  the same seed completed `3/3` turns for `C027_STORY_INHERITANCE` at
  `100.310 tok/s` decode and `97.622 tok/s` effective turn throughput, writing
  `/private/tmp/go-mlx-goal/books/2026-05-24-c027-story-inheritance-seed4242.md`.
  A full random `10`-chapter run with `--random-seed 20260524` picked
  `C014_METAPHOR_SEASONS`, completed `10/10` turns, `3071` visible tokens,
  `16004` final live tokens, `95.384 tok/s` decode, `91.085 tok/s` effective
  turn throughput, `10.048 GiB` active-plus-cache, and about `3.180 GiB`
  process RSS, writing
  `/private/tmp/go-mlx-goal/books/2026-05-24-c014-metaphor-seasons-seed20260524.md`.
  The script now also supports `--count N` batch generation with per-book
  deterministic seeds and an append-only `manifest.jsonl` for later collation;
  `--dry-run --count 2 --random-seed 9000 --turns 2` wrote two distinct
  seed/distractor material sets and manifest rows under
  `/private/tmp/go-mlx-goal/book-runs-batch-dry/` and
  `/private/tmp/go-mlx-goal/books-batch-dry/` without launching MLX. A real
  batch mechanics smoke with `--count 2 --random-seed 9100 --turns 2` then wrote
  two actual `book.md` files and manifest rows under
  `/private/tmp/go-mlx-goal/books-batch-smoke/`: `C003_FICTION_MEMORY` completed
  `2/2` turns at `102.367 tok/s` decode and `99.694 tok/s` effective turn
  throughput, and `C048_FICTION_MIRROR` completed `2/2` turns at
  `102.565 tok/s` decode and `99.963 tok/s` effective turn throughput. This
  smoke used only `512` generated tokens per turn to validate batch output
  plumbing, so do not promote it to performance evidence. The nested Python
  launch needs the same unsandboxed Metal access as other model runs; direct
  dry-run/material generation works without it. Treat this as a reproducible
  content-coherence corpus harness, not as runner-anchor parity.
- Historical `2026-05-24-c014-metaphor-seasons-seed20260524` two-stage book
  detour is retained only as R&D evidence. The fixed-turn compact trigger has
  been removed from the runner and book harness: compaction is an
  overflow/degradation tool for the user-defined context window, not a benchmark
  interval or session-close action. The deprecated `-fold-on-exhaustion` switch
  has also been removed; providing `-fold-store` is enough to enable the old
  overflow behaviour when the live window reaches its threshold. That removed
  detour generated chapters
  `1`-`5`, compacted at its fixed test boundary, wrote
  `/private/tmp/go-mlx-goal/book-runs-compact/2026-05-24-c014-metaphor-seasons-seed20260524.compact.mvlog`,
  and packed it into a `482M` `.kv`. Stage 2 then started from
  `-wake-marker-file ...compact.kv` and generated chapters `6`-`10`; the wake
  used `folded-prefill`, read `1490` compacted prefix tokens, opened the
  embedded State region in `54.3515ms`, and completed the wake in `580.137ms`.
  The combined book is
  `/private/tmp/go-mlx-goal/books-compact/2026-05-24-c014-metaphor-seasons-seed20260524.md`.
  Stage 1 recorded `5/5` turns, `2562` visible tokens, `96.248 tok/s` decode,
  `93.604 tok/s` effective turn throughput, `10.074 GiB` active-plus-cache,
  about `3.165 GiB` RSS, and `495.826 GB` virtual. Stage 2 recorded `5/5`
  turns, `4136` visible tokens, `101.191 tok/s` decode, `99.412 tok/s`
  effective turn throughput, but a poor `34.776 GiB` active-plus-cache,
  about `4.688 GiB` RSS, and `543.264 GB` virtual. Mechanically this proves
  a chapter-5 compact marker can cross a `.kv` process boundary and still
  finish chapter 10. Follow-up external reading accepted the row as a real
  cross-process continuity proof: chapter 6 carries the chapter-1 "fifth
  direction" motif forward into the new cadence/material frame even though the
  visible post-wake prompt does not name that motif, and the same voice and
  boundary/structure vocabulary survive the wake boundary. Treat the doubled
  active memory as a fixable implementation cost, not a proof failure. The
  caveat is now narrower and more product-shaped: the artefact leaked prompt-analysis
  scaffolding (`Constraint Checklist` / plan blocks), and the seasonal-form
  seed lost form adherence because continuity pressure dominated the requested
  autumn/winter/spring/summer register switch. Treat this as state-continuity
  evidence, not final `book.md` polish. The retained-turn prompt was tightened
  afterwards to stop forcing creative material into engineering-analysis mode,
  and the output issue detector now flags `this is an engineering session`,
  `seed prompt to preserve`, `this request asks`, `based on the retained
  context`, and checklist/plan scaffolds as `visible_prompt_analysis`.
- `2026-05-24` scheduling correction: `state-ramp-profile` now resolves the
  default compaction threshold from the configured/model context window, not
  the benchmark `target-tokens`. With the Gemma 4 fast lane this keeps the
  default overflow boundary at `131072` tokens, so a `100000` token benchmark
  target can stop normally without creating a folded State. Explicit lower
  `-compaction-threshold-tokens` values still set the overflow boundary for
  diagnostics. Regression coverage:
  `TestRunCommand_StateRampProfileJSON_Good`,
  `TestRunCommand_StateRampProfileTurnForcedCompactionRemoved_Bad`,
  `TestStateRampProfileContextLifecycle_TargetBelowWindowDoesNotFold_Good`,
  and `TestStateRampProfileDefaultCompactionThresholdUsesModelContext_Good`.
- Production folded-summary path, 2026-05-24: `state-ramp-profile` now exposes
  `-fold-summary-generate`, `-fold-summary-prompt[-file]`, and
  `-fold-summary-max-tokens`. When enabled, the live session generates a
  durable continuation brief at the compact boundary and the fresh folded State
  is built from that model-generated summary plus recent tail. Fold reports
  include `fold.summary_mode=generated`, summary prompt/max-token fields, and a
  `fold.summary_generation` turn so compaction cost is visible instead of being
  hidden inside decode throughput. Empty visible outputs in `state-ramp-profile`
  now fail the turn with `empty_visible_output` instead of being counted as
  successful turns. Follow-up hardening removed the hard-coded
  "opencode-style engineering session" seed from retained chat-template
  preambles and replaced it with the shared Lemma new-session default exposed
  as `mlx.DefaultLemmaNewSessionText` / `mlx.DefaultNewSessionText`. The
  go-mlx, llama.cpp, and mlx_lm workflow harnesses now use that same text, so
  creative compact runs no longer start from an engineering-session scaffold
  and runner anchors stay prompt-matched. Explicit empty seed contexts are now
  valid with `-prompt "" -start-tokens 0`, letting frameworks lead with a
  blank/new-session pack or use the first real user prompt instead of a
  synthetic retained context. Generated folded summaries now fail the fold when
  the summary turn carries non-debug output issues such as prompt analysis or
  visible control tokens, preventing a bad summary from being accepted as a
  clean compact State. This is the production path for compacting into a new
  State file; raw cross-session continuation from the old live window remains
  an R&D lane.
- Generated-summary compact-book smoke, same date:
  `/private/tmp/go-mlx-goal/book-runs-prodsummary-seedtext/2026-05-24-c001-story-perspective-seed20260524.*`
  uses `C001_STORY_PERSPECTIVE`, Gemma 4 chat template wrapping, a
  model-generated folded summary, `.kv` packing, and a stage-2 command with no
  seed prompt replay. Stage 1 records `5/5` turns, `3986` generated/visible
  tokens, `98.007 tok/s` decode, `95.880 tok/s` effective turn throughput,
  `10.065 GB` active-plus-cache, about `3.409 GB` RSS, and a generated summary
  of `345` visible tokens. The generated folded prompt is `12130` bytes and
  the fold lifecycle is `4.946s`. Stage 2 wakes from the `.kv` with
  `restore_strategy=folded-prefill` in `896.781ms`, then records `5/5` turns,
  `762` generated/visible tokens, `103.681 tok/s` decode, `95.104 tok/s`
  effective turn throughput, `13.147 GB` active-plus-cache, about `4.432 GB`
  RSS, and `498.287 GB` virtual. This proves the generated-summary folded
  State path works mechanically with better bounded memory than the raw
  high-water compact detour. Do not promote this row as final content quality:
  stage-1 visible prompt analysis still appears in the artefact and stage-2
  distractor pressure remains stronger than desired.
- Lemma-family book research, same date: the book harness now has an opt-in
  direct turn mode (`state-ramp-profile -turn-prompt-mode direct`, exposed as
  `scripts/state_book_from_phase0.py --turn-prompt-mode direct`) so creative
  turns can use the native chat wrapper without the reference-material scaffold
  that smaller models may copy. While checking the `lthn/LEM-Gemma3-1B` zero
  output, the native Gemma chat formatter was corrected to match the model's
  `chat_template.jinja`: emit the BOS marker and fold a leading system message
  into the first user turn instead of creating consecutive user turns. The
  fixed template did not make the `C001_STORY_PERSPECTIVE` retained-book smoke
  generate visible output: it still stops at turn 1 with
  `empty_visible_output`, `0` generated tokens, about `5.84 GB`
  active-plus-cache, and about `3.00 GB` RSS. A neutral warm-state probe on the
  same model does generate normally (`109` visible tokens at `60.154 tok/s`,
  about `5.24 GB` active-plus-cache), so the 0-token book stop is
  seed/context-sensitive model behaviour rather than a general loader or chat
  template failure. The local `lthn/lemer-lite` q4 Gemma 4-family snapshot is
  the first readable Lemma-family retained book pass: the 10-turn direct run at
  `/private/tmp/go-mlx-goal/book-runs-lemer-lite-direct/2026-05-24-c001-story-perspective-seed2026052404.json`
  produced the readable book
  `/private/tmp/go-mlx-goal/books-lemer-lite-direct/2026-05-24-c001-story-perspective-seed2026052404.md`
  with `10/10` successful turns, `3139` generated/visible tokens,
  `100.508 tok/s` decode, `97.003 tok/s` effective turn throughput, `7999`
  initial prefill tokens, `13156` final live tokens, `8.995 GB`
  active-plus-cache, and about `3.05 GB` RSS. Content preserves the lighthouse,
  light, and deep-ocean signal arc across all ten turns, with distractors
  acting mostly as pressure rather than replacing the plot.
- `2026-05-24-default-after-native-sliding-reject-go-mlx-gemma4-e2b-4bit-opencode-delimited-30k-to-70k-r10-g1024.json`:
  current no-floor default retained-State row after rejecting native fixed
  sliding attention as a production default. It completes `10/10` retained
  turns from a `30000` token first context, `63971` final live tokens, `27943`
  appended tokens, `6000` generated/visible tokens, `95.053s` workload wall
  time, `16.974s` append time, `91.146 tok/s` raw decode, `72.456 tok/s`
  effective turn throughput, `2450.267 tok/s` first prefill, `1646.264 tok/s`
  average append/prefill, `4.756 GiB` peak MLX memory, `9.365 GiB`
  active-plus-cache, about `3.168 GiB` process RSS, `535.504 GiB` process
  virtual reservation, and `9505.252 J` estimated at `100 W`. The runtime gate
  capture intentionally does not include
  `GO_MLX_ENABLE_NATIVE_FIXED_SLIDING_ATTENTION`; the explicit diagnostic gate
  is retained for R&D only. Content is non-empty and coherent, but the first
  turn still exposes visible self-correction/plan scaffolding, so this row is a
  clean performance/default-path row rather than final product-quality sign-off.
  The same small repro shape proves why the native sliding helper is rejected:
  the default fast lane succeeds at `109.8 tok/s` decode in
  `2026-05-24-diagnostic-state-ramp-2k-to-5k-g16-default-after-native-sliding-reject.json`,
  while the same run with native fixed sliding enabled fails at decode step `0`
  with `mlx.lastError: expected a non-empty mlx_array`. Explicit runtime-gate
  `0` values now win over fast-lane defaults so single-gate diagnostics can be
  isolated without disabling the whole lane.
- `2026-05-24-default-native-linear-go-mlx-gemma4-e2b-4bit-opencode-delimited-30k-to-70k-r10-g1024.json`:
  current rebuilt default retained-State run after promoting
  `GO_MLX_ENABLE_NATIVE_LINEAR_MATVEC=1` into the fast lane. This is the best
  current primary interactive row: `10/10` retained turns from a `30000` token
  first context, `63671` final live tokens, `28363` appended tokens, `5280`
  visible/generated tokens, `84.311s` workload wall time, `16.060s` append
  time, `92.057 tok/s` raw decode, `71.911 tok/s` effective turn throughput,
  `4.517 GiB` peak MLX memory, `6.031 GiB` cache memory, `3.165 GiB` process
  RSS, and `8431.112 J` estimated at `100 W`. Treat process RSS as an
  incomplete memory figure for this runner: the comparable active footprint is
  the MLX allocator pressure, with active-plus-cache around `10.247 GiB`. Versus
  the fresh same-shape llama.cpp anchor below, llama.cpp still leads raw decode
  (`103.143 / 92.057 = 1.120x`), while go-mlx wins workload wall time
  (`84.311s` versus `129.275s`) and estimated energy at the normalised
  `100 W` draw. Memory is not a go-mlx win: llama-server was observed by
  external `ps` at about `5.25 GiB` RSS at the end of the run, while go-mlx
  reports about `10.247 GiB` active-plus-cache. The comparison is still not a
  production sign-off because llama.cpp leaks control/thinking channel text and
  consumes more of the `1024` token budget than the intended go-mlx answer
  stream.
- `state-ramp-profile -trace-token-phases`: retained-State workflow traces can
  now carry the same per-token phase and native-event buckets that
  `driver-profile` already exposed. This is instrumentation for the real
  repeated-workflow lane, not a decode-speed claim: the focused tests pass, and
  `BenchmarkSummariseStateRampProfileTurns_LongRampWithTrace` measures
  `12509 ns/op`, `816 B/op`, and `12 allocs/op` after replacing native-event
  string splitting with a prefix/dot scan. The no-trace long-ramp summary stays
  allocation-free at `3597 ns/op`, `0 B/op`, `0 allocs/op`. Use this flag on
  future 30k-to-70k and 30k-to-100k retained runs when diagnosing whether
  long-context time is still hidden in lazy MLX materialisation, but keep it
  out of default production rows unless a trace row is explicitly requested.
- `2026-05-24-state-ramp-trace-session-phases-go-mlx-gemma4-e2b-4bit-opencode-delimited-30k-to-70k-r10-g1024.json`:
  first full retained-State trace row after teaching `ModelSession.Generate` to
  retain `TokenPhases` in `model.Metrics()`. It completes the same `30k` to
  `70k` opencode-shaped workload at `10/10` turns, `64558` final live tokens,
  `27943` appended tokens, `6587` generated/visible tokens, `102.121s` total
  wall, `17.056s` append time, `90.447 tok/s` raw decode,
  `73.269 tok/s` effective turn throughput, `4.401 GiB` peak MLX memory,
  `9.361 GiB` active-plus-cache, about `3.184 GiB` process RSS, and
  `10212.052 J` estimated at `100 W`. The trace has `6596` per-token phase
  samples. The dominant bucket is `sample` at `60.180s` total and `9.124ms`
  average per token, followed by `forward` at `12.398s` total and `1.880ms`
  average; text decode, yield, token read, and reporting are microsecond-scale.
  For retained stochastic turns this `sample` bucket includes the lazy logits
  materialisation plus top-k/top-p sampling, so the next raw-decode target is
  still MLX eval/sampling graph work, not Go output handling. Native-event
  buckets remain empty unless `GO_MLX_TRACE_FORWARD_EVAL=1` is also enabled.
- `2026-05-24-state-ramp-trace-split-sample-eval-smoke-go-mlx-gemma4-e2b-4bit-opencode-delimited-30k-r1-g1024.json`:
  follow-up smoke row after splitting retained stochastic trace accounting so
  sampler graph build, `Eval` materialisation, and sampled-token readback are
  no longer collapsed into one `sample` bucket. This is not a benchmark row; it
  is a one-turn instrumentation check over the same `30k` seed and
  opencode-delimited append stream. It completed `1/1` turn at `32123` final
  live tokens, `1024` generated tokens, `95.228 tok/s` raw decode, and
  `90.303 tok/s` effective turn throughput. The split shows `sample_eval` as
  the real dominant bucket at `8.824s` total / `8.618ms` per token, `forward`
  graph construction at `1.856s` total / `1.812ms` per token, and sampler graph
  build at only `43.466ms` total / `42.447us` per token. This confirms the
  earlier full-row `sample` finding was MLX lazy materialisation pressure, not
  Go string/output handling or sampler-construction overhead. A focused
  sampler-only microbench reinforces the same conclusion:
  `BenchmarkSampler_TopKThenTopP_Vocab262k` is only `529389 ns/op`,
  `24 B/op`, and `3 allocs/op` on the current machine, versus
  `997718 ns/op` for the rejected legacy full-vocab top-p-then-top-k order.
  The retained `8.6ms/token` bucket is therefore model/logit graph evaluation
  flowing through the sampled token, not the bounded top-k/top-p sampler by
  itself.
- `2026-05-24-state-ramp-session-async-control-seed240524-suppresseos-go-mlx-gemma4-e2b-4bit-opencode-delimited-30k-r1-g1024.json`
  and
  `2026-05-24-state-ramp-session-async-prefetch-seed240524-suppresseos-go-mlx-gemma4-e2b-4bit-opencode-delimited-30k-r1-g1024.json`:
  retained-session eval-boundary A/B after wiring `ModelSession.Generate` into
  the existing `GO_MLX_ENABLE_ASYNC_DECODE_PREFETCH` path. The seeded,
  EOS-suppressed one-turn shape generated the same `1024` tokens in both rows.
  Async prefetch improved raw decode from `93.577 tok/s` to `96.152 tok/s`,
  effective turn throughput from `88.831 tok/s` to `91.191 tok/s`, wall from
  `23.772s` to `23.483s`, and estimated energy at `100 W` from `2377.210 J`
  to `2348.262 J`. Trace attribution moved the materialisation wait out of
  `sample_eval`: `sample_eval` fell from `8.640ms/token` to `3.278ms/token`,
  while the async wait showed up in `other` at `5.234ms/token`. This is a real
  retained-session boundary improvement, not sampler math.
- `2026-05-24-state-ramp-current-control-seed240524-suppresseos-go-mlx-gemma4-e2b-4bit-opencode-delimited-30k-to-70k-r10-g1024.json`
  and
  `2026-05-24-state-ramp-current-async-default-seed240524-suppresseos-go-mlx-gemma4-e2b-4bit-opencode-delimited-30k-to-70k-r10-g1024.json`:
  same-binary, same-seed, no-trace full retained workflow check over `10`
  turns. Both rows completed `10/10` turns with identical `63456` final live
  tokens, `27903` appended tokens, and `5526` generated/visible tokens. Async
  retained prefetch improved raw decode from `90.481 tok/s` to
  `91.964 tok/s`, effective turn throughput from `70.731 tok/s` to
  `71.674 tok/s`, wall from `90.371s` to `89.343s`, and estimated energy at
  `100 W` from `9037.052 J` to `8934.274 J`. Active-plus-cache also edged down
  from `9.719 GiB` to `9.669 GiB`. This is now promoted into
  `DefaultGemma4FastRuntimeGates()` as
  `GO_MLX_ENABLE_ASYNC_DECODE_PREFETCH=1`; the rebuilt default smoke
  `2026-05-24-state-ramp-default-async-promoted-smoke-go-mlx-gemma4-e2b-4bit-opencode-delimited-30k-r1-g1024.json`
  confirms the gate appears without an env override and completes the seeded
  `1024` token turn at `95.894 tok/s` raw decode, `90.937 tok/s` effective
  turn throughput, and `2346.068 J` estimated energy.
- `2026-05-24-state-ramp-default-repeat-history-cleanup-smoke-go-mlx-gemma4-e2b-4bit-opencode-delimited-30k-r1-g1024.json`:
  rebuilt `lthn-mlx` after aligning retained `ModelSession.Generate` with
  `Model.Generate` so repeat-penalty history is not copied or appended when
  `repeat_penalty=1`. The same seeded, EOS-suppressed default one-turn smoke
  completes `1024` generated tokens at `96.403 tok/s` raw decode,
  `91.383 tok/s` effective turn throughput, `23.537s` wall time, and
  `2353.682 J` estimated at `100 W`, with
  `9716531922` bytes active-plus-cache and `492307447808` bytes process
  virtual reservation. Treat this as a small hot-path hygiene/regression row:
  it removes avoidable per-token slice growth in the default sampling shape,
  but the wall/energy result is within the existing async smoke noise band and
  does not change the open llama.cpp decode gap.
- Host-side retained append now streams wrapped repeated-source spans into
  `ModelSession.AppendTokens` instead of first building a copied token slice.
  The focused benchmark records the old wrapped helper at `3378 ns/op`,
  `16384 B/op`, `1 alloc/op`, while
  `BenchmarkForEachRepeatedStateRampTokenSpan_Append4096Wrapped` records
  `4.504 ns/op`, `0 B/op`, and `0 allocs/op`. The rebuilt default delimited
  smoke
  `2026-05-24-state-ramp-default-streamed-append-smoke-go-mlx-gemma4-e2b-4bit-opencode-delimited-30k-r1-g1024.json`
  remains clean at `95.712 tok/s` raw decode, `90.765 tok/s` effective turn
  throughput, `23.512s` wall time, `2351.161 J` estimated at `100 W`,
  `9670627890` bytes active-plus-cache, and `492284395520` bytes process
  virtual reservation. This is a lower-memory/lower-power host-path cleanup
  for wrapped-source long ramps; it is not claimed as a Metal decode fix.
- Gemma 4 per-layer input views now stream from the combined PLE/projection
  tensor one layer at a time instead of prebuilding and retaining all layer
  views for the forward pass. The first version used generic `SliceAxis` and
  was correctly rejected by the benchmark as allocation-neutral/noisy. The
  corrected path uses rank-specific `Slice4` plus the new scalar-pass
  `Reshape3`: the current
  `BenchmarkPLE_PerLayerInputViewsSplitAll_Graph` rerun records
  `27063 ns/op`, `833 B/op`, and `52 allocs/op`, while
  `BenchmarkPLE_PerLayerInputViewsStreamed_Graph` records `21354 ns/op`,
  `0 B/op`, and `0 allocs/op`. The retained all-views splitter now uses the
  same scalar view helper and records `22471 ns/op`, `208 B/op`, and
  `1 alloc/op` in `BenchmarkPLE_SplitPerLayerInputTensor_Graph`. Focused
  Gemma 4 PLE correctness tests pass.
  The rebuilt seeded one-turn retained smoke
  `2026-05-24-state-ramp-default-ple-slice4-streamed-view-smoke-go-mlx-gemma4-e2b-4bit-opencode-delimited-30k-r1-g1024.json`
  completes `1024` generated/visible tokens at `95.936 tok/s` raw decode,
  `90.967 tok/s` effective turn throughput, `23.577s` wall time, and
  `2357.747 J` estimated at `100 W`, with `9640460118` bytes
  active-plus-cache and `492263161856` bytes process virtual reservation.
  The full corrected `10`-turn retained workflow row
  `2026-05-24-state-ramp-default-ple-slice4-streamed-view-c10-go-mlx-gemma4-e2b-4bit-opencode-delimited-30k-to-70k-r10-g1024.json`
  completes `10/10` turns, `63456` final live tokens, `27903` appended tokens,
  and `5526` generated/visible tokens at `92.472 tok/s` raw decode,
  `72.025 tok/s` effective turn throughput, `88.930s` wall time, and
  `8892.954 J` estimated at `100 W`, with `10235431210` bytes
  active-plus-cache and `576399851520` bytes process virtual reservation.
  This is accepted as cumulative streaming/lifetime cleanup: it keeps the
  workflow inside the healthy `90+ tok/s` band and improves the retained
  effective throughput slightly versus the earlier native-linear row, but its
  memory movement is neutral/noisy rather than a standalone memory win.
- The `30k` to `100k` retained build-up now has a current folded-State
  lifecycle row after the PLE view cleanup and hyper-long default correction.
  The first same-binary folded probe,
  `2026-05-24-state-ramp-default-ple-slice4-delimited-folded-30k-to-100k-g1024.json`,
  is retained as the rejected A/B: the state-ramp path had re-enabled full
  fixed Gemma 4 cache for the `100k` target, reached only `67040` live tokens
  after `11` successful turns, and then failed the active-memory guard on turn
  `12` (`92261571038 > 92261063065` bytes). Process RSS stayed bounded around
  `3404316672` bytes, but the fixed-cache active allocator spike prevented
  fold handoff.
  This fixed-cache failure row is now superseded by the paged/no-fixed
  correction above: the default retained path should not switch strategies at
  the long-form chapter boundary, and fixed cache stays a manual diagnostic
  option only. The historical rebuilt default folded row
  `2026-05-24-state-ramp-default-paged-after-fixed-threshold-30k-to-100k-folded-g1024.json`
  completes with no error: `23/23` retained turns, `103187` final live tokens,
  `63973` appended tokens, `9148` generated/visible tokens, `77.509 tok/s`
  raw decode, `56.692 tok/s` effective turn throughput, `173.735s` wall time,
  and `17373.509 J` estimated at `100 W`. Peak MLX memory is
  `3930481958` bytes, active MLX is `3391510954` bytes, active-plus-cache is
  `10040041690` bytes, process virtual reservation is `761543933952` bytes,
  and process RSS is `3390570496` bytes. The fold lifecycle writes
  `/private/tmp/go-mlx-goal/state-fold-2026-05-24-default-paged-30k-to-100k.mvlog`
  (`920M`), checkpoints `103188` tokens, folds to a `175` token compacted
  state in `1.074s`, wakes it in `73.821ms`, and continues for `298` tokens at
  `107.889 tok/s`. This closes the immediate 60k-ish retained-memory cliff in
  the default path.
  The follow-up replay-estimate instrumentation first reproduced the old bad
  path in a smaller shape:
  `2026-05-24-state-ramp-replay-estimate-smoke-10k-to-20k-g1024.json` crossed
  the `20k` fold threshold with auto fixed-cache defaults still enabled and
  failed the active-memory guard on turn `3`
  (`92351224286 > 92261063065` bytes). That smoke reflects the pre-correction
  fixed-cache sizing bug, not current intended behaviour: the state-ramp fast
  lane now keeps fixed-cache gates out of the production defaults and no longer
  invents a fixed K/V budget from the run shape.
  The corrected smoke
  `2026-05-24-state-ramp-replay-estimate-smoke-paged-10k-to-20k-g1024.json`
  then completes `3/3` turns at `94.636 tok/s` raw decode,
  `85.506 tok/s` effective turn throughput, `39.645s` wall time, `3.206 GB`
  peak MLX active memory, about `3.285 GB` RSS, and emits a same-binary replay
  estimate of `48.867s` one-shot wall versus `39.645s` retained wall
  (`1.23x` retained speedup, `922.196 J` saved at `100 W`).
  The current full folded row with emitted replay estimates,
  `2026-05-24-state-ramp-current-paged-replay-estimate-30k-to-100k-folded-g1024.json`,
  completes `23/23` retained turns, `103187` final live tokens, `63973`
  appended tokens, `9148` generated/visible tokens, `77.778 tok/s` raw decode,
  `56.839 tok/s` effective turn throughput, and `173.173s` retained wall time.
  It reports `55535708706ns` retained setup (`30k` seed prefill plus retained
  appends) versus `757459197525ns` replay-prefill estimate and
  `875096629732ns` one-shot/replay wall estimate. The retained path therefore
  saves `701.923s`, is `5.053x` faster than same-binary replayed prefill, and
  saves an estimated `70192.349 J` at the labelled `100 W` assumption. Memory
  stays bounded in the useful sense: `3930481958` bytes peak MLX active,
  `10040111834` bytes active-plus-cache, `3388882944` bytes RSS, and
  `762191462400` bytes virtual reservation. The fold store is
  `/private/tmp/go-mlx-goal/state-fold-2026-05-24-current-paged-replay-estimate-30k-to-100k.mvlog`
  (`920M`), checkpoints `103188` tokens, folds to `175` tokens in `1.056s`,
  wakes in `73.678ms`, and continues for `282` visible tokens at
  `109.547 tok/s`. The retained `77.778 tok/s` raw decode and `56.839 tok/s`
  effective-turn figures exclude the fold lifecycle. Compact itself took
  `1.056165625s`; the full folded handoff was `3.800255584s` after adding
  wake, continue-append, and continue-generation. New reports now emit
  `fold.lifecycle_duration` and
  `fold.retained_total_with_lifecycle_duration` so the compaction cost stays
  explicit instead of being folded into decode throughput.
- `2026-05-24-state-ramp-model-greedy-smoke-go-mlx-gemma4-e2b-4bit-opencode-delimited-30k-r1-g1024.json`
  and
  `2026-05-24-state-ramp-model-greedy-go-mlx-gemma4-e2b-4bit-opencode-delimited-30k-to-70k-r10-g1024.json`:
  current-binary retest with `GO_MLX_ENABLE_NATIVE_GEMMA4_MODEL_GREEDY=1`
  present in the runtime-gate map. These rows are now recorded as
  inconclusive, not as model-wrapper speed evidence: `state-ramp-profile` uses
  retained stochastic sampling (`temperature=1.0`, `top_p=0.95`, `top_k=64`),
  and `ModelSession.Generate` therefore does not enter the direct greedy/model
  greedy token path. The one-turn row completes at `95.570 tok/s` and the full
  `30k` to `70k` row completes `10/10` turns at `91.065 tok/s` raw decode,
  `72.022 tok/s` effective turn throughput, `5871` generated/visible tokens,
  `93.746s` wall, and `10.049 GiB` active-plus-cache. Treat the deltas versus
  the default trace row as normal sampled-output variance and answer-length
  skew, not as a production default signal. The real retained decode target
  remains the sampled logits/materialisation path.
- `2026-05-24-state-ramp-native-events-split-smoke-go-mlx-gemma4-e2b-4bit-opencode-30k-r1-g64.json`:
  diagnostic-only retained-State native-event trace with
  `GO_MLX_TRACE_FORWARD_EVAL=1` after the sampler/eval split above. Forced
  intermediate materialisation slows the one-turn run to `24.135 tok/s`, so do
  not compare it as a production speed row. Its value is attribution: the
  hidden `sample_eval` bucket drops to `56.725ms` total / `0.886ms` per token,
  while `forward` rises to `2.590s` total / `40.467ms` per token. Ranked native
  buckets over `64` generated tokens are attention first (`738.598ms` over
  `2240` events), then layer output (`620.715ms`), FFN (`599.815ms`), and
  attention residual (`448.256ms`). This confirms the retained path is still
  eval/materialisation-bound at the Gemma 4 layer graph, not blocked on sampler
  graph construction, token readback, decode text, or yield overhead.
- `2026-05-24-state-ramp-native-event-details-go-mlx-gemma4-e2b-4bit-opencode-30k-r1-g64.json`:
  follow-up diagnostic after adding `summary.native_event_details` to retained
  State and driver profile reports. The coarse `native_events` buckets stay
  intact, while the new exact-name summary ranks `140` layer/event buckets
  without external `jq` scraping. The one-turn trace is diagnostic-only
  (`23.176 tok/s` under forced materialisation), but it identifies the current
  E2B attention target precisely: the largest exact events are
  `gemma4.layer.00.output` at `33.706ms`, then full-attention owner layers
  `04`, `14`, `09`, `19`, `24`, `29`, and `34` at about `28.701ms` to
  `32.694ms` over `64` generated tokens. That matches the Gemma 4 config's
  `4+5n` full-attention interleave and keeps the next implementation target on
  full/global owner attention materialisation and layer-output graph boundaries,
  not local sliding-mask construction or sampler work. The no-trace summary
  benchmark remains allocation-free; the trace-summary benchmark intentionally
  grows to `16008 ns/op`, `1224 B/op`, `18 allocs/op` because it preserves
  exact event names for diagnostics only.
- `2026-05-24-go-mlx-gemma4-e2b-4bit-opencode-delimited-30k-to-70k-r10-g1024-paged-no-fixed-clearcache.json`:
  diagnostic retained-State run with `GO_MLX_ENABLE_FIXED_GEMMA4_CACHE=0` and
  generation clear-cache enabled. This proves the coherent paged retained path
  still works on current code, but it is not yet the production answer:
  `10/10` turns, `66879` final live tokens, `28323` appended tokens, `8530`
  generated/visible tokens, `135.156s` workload wall time, `79.985 tok/s` raw
  decode, `68.932 tok/s` effective turn throughput, `3.434 GiB` peak MLX
  memory, `3.153 GiB` active MLX memory, `6.214 GiB` MLX cache memory, about
  `9.367 GiB` active-plus-cache, `3.179 GiB` process RSS, and `13515.578 J`
  estimated at `100 W`. Compared with the fixed-cache row, paged/no-fixed is
  memory-safer in active allocations but slower and still carries high allocator
  cache pressure. Treat this as confirmation that the next real win is true
  pinned State-page decode over local sliding tails plus global owner pages, not
  merely disabling fixed caches.
- `2026-05-24-fresh-llamacpp-gemma4-e2b-q4km-opencode-delimited-30k-to-70k-r10-g1024.json`:
  fresh llama.cpp server anchor against the same opencode-delimited prompt
  shape, excluding server startup from workload timing just as the go-mlx row
  excludes `load_duration`. Server startup to listen was about `1.50s`.
  The workload records `10/10` turns, `67190` final live tokens, `27303`
  appended tokens, `9867` generated tokens, `9865` visible tokens,
  `129.275s` wall time, `103.143 tok/s` raw decode from llama.cpp timings,
  `76.310` visible tok/s by wall, `32.948s` prompt work, `12927.452 J` at
  `100 W`, and `10` leaked control markers. The Python harness could not call
  `ps` from inside the sandbox, so its JSON process-memory fields are empty;
  external polling during the run observed llama-server RSS rising to about
  `5.25 GiB`.
- `2026-05-24-default-native-linear-go-mlx-gemma4-e2b-4bit-opencode-30k-to-100k-r10-g1024.json`:
  stress-only fixed-token append run with `8192` appended tokens per turn. It
  reproduced the suspected `60k`-`70k` memory bend without OOMing: the run
  reached `72155` live tokens on turn 5, held process RSS near `3.158 GiB`,
  but aborted on the live stream safety guard when MLX active memory spiked to
  `13033167410` bytes over the `12 GiB` cap. Treat this as evidence that the
  next optimisation target is transient MLX graph/cache lifetime or append
  materialisation under large append chunks, not resident process runaway.
- `2026-05-24-default-fixed-cache-go-mlx-gemma4-e2b-4bit-opencode-r10-g1024.json`:
  superseded rebuilt `lthn-mlx` retained-State run after making hyper-long
  `state-ramp-profile` use a bounded Gemma 4 fixed cache by default; `10/10`
  retained turns from a `30000` token first context, `64696` final live tokens,
  `28363` appended tokens, `6305` visible/generated tokens, `99.556s`
  workload wall time, `16.047s` append time, `86.949 tok/s` raw decode,
  `71.189 tok/s` effective turn throughput, `3.160 GiB` process RSS, and
  `9955.593 J` estimated at `100 W`. Runtime gates include
  `GO_MLX_ENABLE_FIXED_GEMMA4_CACHE=1`,
  `GO_MLX_ENABLE_FIXED_GEMMA4_SLIDING_CACHE_BOUND=1`,
  `GO_MLX_ENABLE_FIXED_GEMMA4_SHARED_MASK=1`, and
  `GO_MLX_FIXED_GEMMA4_CACHE_SIZE=70000`. It recovered about `11.8%` raw
  decode at the time, but is now replaced by the native-linear default row
  above. Historical visible-token floor pass/fail wording on neighbouring rows
  is now treated as debug-only evidence.
- `2026-05-24-sampler-only-go-mlx-gemma4-e2b-4bit-opencode-r10-g1024.json`:
  diagnostic run after changing sampled generation to apply top-k before top-p
  when both are configured. The matching hot-path benchmark
  `BenchmarkSampler_(LegacyTopPThenTopK|TopKThenTopP)_Vocab262k` records
  `1015783 ns/op` for the previous full-vocab top-p path versus
  `539522 ns/op` for top-k-then-top-p, with both paths at `24 B/op` and
  `3 allocs/op`. The retained workflow records `64526` final live tokens,
  `28363` appended tokens, `6136` visible/generated tokens, `95.457s` wall
  time, `89.483 tok/s` raw decode, `72.535 tok/s` effective turn throughput,
  `3.160 GiB` process RSS, and `9545.749 J` estimated at `100 W`. Treat this
  as a valid local optimisation delta, not a production-accepted row; the
  historical `256` visible-token floor on this row is now classified as a debug
  guard, not a scientific acceptance criterion.
- `2026-05-24-diagnostic-greedy-output-rmsnorm-sampler.json` and
  `2026-05-24-diagnostic-greedy-output-sampler-only.json`: rejected Gemma 4
  RMSNorm `(1 + weight)` pre-fold for the local `mlx-community` E2B 4bit
  snapshot. Adding `1` to every Gemma 4 norm scale kept speed flat but made
  temperature-zero output collapse into token noise. Inspecting the checkpoint
  showed direct-scale-looking norm tensors at load time
  (`input_layernorm.weight` values such as `6.625..83`, `q_norm.weight` around
  `0.984`), so `precomputeGemma4ScaledWeights` remains a direct copy for this
  MLX checkpoint family. This is a correctness guard against blindly applying
  the zero-centred Gemma 3 rule to already-converted Gemma 4 MLX weights.
- `2026-05-24-decode-trace-go-mlx-gemma4-e2b-4bit-opencode-p51k-g128.json`
  and `2026-05-24-decode-trace-go-mlx-gemma4-e2b-4bit-opencode-p51k-g128-fixed.json`:
  focused decode traces against a `51242` token prompt and `128` generated
  tokens. The paged hyper-long default measured `79.177 tok/s`; token phase
  timing showed `12.628ms` average token time with `11.142ms` in
  `sample_eval`, confirming the bottleneck is lazy MLX graph materialisation,
  not Go token/text handling. Enabling bounded fixed cache plus the sliding
  local-window cap measured `90.952 tok/s`, reducing average `sample_eval` to
  `9.396ms` and confirming the paged hyper-long cache layout was a decode
  slowdown. The current sampler-only build keeps the same temperature-zero
  shape at `90.556 tok/s`; non-final token phases average `11.098ms`, with
  `9.558ms` in lazy forward materialisation and `1.511ms` in next-token graph
  construction. This keeps the next raw-decode target on collapsing or
  compiling the per-token Gemma 4 forward graph, not on driver text handling or
  sampler allocations.
- `2026-05-24-decode-trace-go-mlx-gemma4-e2b-4bit-opencode-p51k-g128-default-post-keqv.json`:
  fresh rebuilt default trace after the compiled/native guard fixes below;
  `128/128` generated tokens, `51242` prompt tokens, `90.347 tok/s` raw decode,
  `2379.488 tok/s` prefill, `22.952s` total time including prefill,
  `3.164 GiB` process RSS, `4.650 GiB` peak MLX memory, and `5.778 GiB`
  reported cache memory. This is consistent with the previous fixed-cache
  default trace and confirms the stability guards did not regress the accepted
  default lane.
- `2026-05-24-decode-trace-go-mlx-gemma4-e2b-4bit-opencode-p51k-g128-default-after-full-gate.json`:
  current rebuilt default after the per-layer full-attention safety gate;
  `128/128` generated tokens, `51242` prompt tokens, `90.453 tok/s` raw decode,
  `2373.521 tok/s` prefill, `23.043s` total time including prefill, and
  `3.167 GiB` process RSS. Token phases still place almost all steady decode
  time in lazy MLX materialisation (`9.426ms` average `sample_eval`, which is
  `Eval(next)` materialising the forward graph in the greedy path), so the raw
  parity target remains graph/eval-boundary work rather than driver text or
  sampler allocation work.
- `2026-05-24-decode-trace-go-mlx-gemma4-e2b-4bit-opencode-p51k-g512-borrowed-suppress.json`:
  rebuilt after the direct-greedy suppression tensor was made generation-local
  instead of per-token and single-token Gemma 4 decode stopped allocating an
  unused runtime mask cache / heap `sharedKV` scratch. The longer trace
  generates `512/512` tokens from the same `51242` token prompt at
  `90.554 tok/s`, `2377.046 tok/s` prefill, `27.249s` total wall time,
  `3.157 GiB` process RSS, and empty stderr. The focused benchmark pair
  `BenchmarkDecodeLoop_LastTokenGreedySuppressed_(FreshArray|BorrowedArray)`
  records `233154 ns/op`, `72 B/op`, `2 allocs/op` for the old per-token
  suppress-array path versus `223576 ns/op`, `0 B/op`, `0 allocs/op` for the
  borrowed-array path. Keep the patch for long-output allocation pressure, but
  do not count it as a raw decode parity fix: token phases remain dominated by
  lazy forward materialisation at `9.427ms` average `sample_eval`.
- `2026-05-24-decode-trace-go-mlx-gemma4-e2b-4bit-opencode-p51k-g32-native-events-borrowed-suppress.json`:
  diagnostic-only `GO_MLX_TRACE_FORWARD_EVAL=1` rerun after the same cleanup.
  Forced materialisation slows decode to `24.172 tok/s`, but moves the hidden
  lazy work into the `forward` bucket and ranks the current evaluated graph
  costs as attention first (`396.509ms` over `1085` events), then layer output
  (`310.796ms`), FFN (`296.605ms`), and attention residual (`220.893ms`). This
  reconfirms the next material speed path is a fused/model-level Gemma 4
  forward boundary or attention/FFN kernel work, not more Go-side sampler or
  token text allocation cleanup.
- `2026-05-24-decode-trace-go-mlx-gemma4-e2b-4bit-opencode-p51k-g512-default-native-linear-rerun.json`:
  accepted local decode improvement after promoting
  `GO_MLX_ENABLE_NATIVE_LINEAR_MATVEC=1` into the Gemma 4 fast default gates
  and guarding the custom q4/q8 matvec kernels against partial final
  threadgroups. The rebuilt default lane report includes the native-linear
  gate without passing `-native-linear-matvec` explicitly and records `512/512`
  generated tokens from the `51242` token prompt at `91.650 tok/s`,
  `2375.876 tok/s` prefill, `27.154s` total time including prefill,
  `5.279 GiB` peak MLX memory, `5.788 GiB` cache memory, and `3.181 GiB`
  process RSS. The first default trace after changing the kernel source,
  `2026-05-24-decode-trace-go-mlx-gemma4-e2b-4bit-opencode-p51k-g512-default-native-linear.json`,
  measured only `87.875 tok/s` because token step 1 paid a one-time custom
  Metal kernel materialisation cost; the immediate rerun recovered to the
  accepted row above. Keep the gate as a decode win for warmed agent
  processes, but account for first-use kernel compilation in cold-start wall
  reports.
- Rejected construction-path probes after the borrowed suppression cleanup:
  an inline fixed-mask lookup cache measured a nice synthetic reuse path
  (`BenchmarkAttention_FixedMaskSet_ReuseInline` at `6.217 ns/op`,
  `0 B/op`, `0 allocs/op`), but the real `51242` prompt / `512` token trace
  regressed to `89.840 tok/s` and `1.632ms` average forward construction, so it
  was reverted. Hoisting the native fixed-attention scale scalar into a
  borrowed model array was also rejected before a real trace:
  `BenchmarkDecodeLoop_FixedSingleTokenAttention_FreshScale` measured
  `244653 ns/op` while the borrowed-scale variant measured `248218 ns/op`, both
  at `0 B/op`; this confirms the current `FromValue(scale)` path is not an
  allocation issue worth promoting.
- Additional rejected decode probes from the native-linear sweep:
  reusing the same Go `Array` wrapper for Gemma 4 K=V instead of cloning the
  raw K projection passed focused Metal tests but regressed the real
  `51242` prompt / `512` token trace to `88.747 tok/s`, so it was reverted.
  `-native-gemma4-fixed-owner-attention -native-gemma4-fixed-owner-attention-residual`
  measured `88.7 tok/s` on a `256` token probe and remains off. The narrower
  `-native-gemma4-attention-o-matvec` probe measured `89.7 tok/s` at `512`
  tokens, which is not enough to promote over the broader native-linear gate.
  The native-linear promotion is covered by
  `TestDenseMatVec_NativeLinearForwardMatchesQuantizedMatmul_Good`,
  `TestDenseMatVec_NativeMLPMatchesGoGraph_Good`, and the production-gate
  tests; the dense matvec tests now compare the custom kernels against a CPU
  q4 affine reference so tiny MLX fallback-kernel availability cannot mask
  custom-kernel regressions.
- Expert-ID native dispatch shape cleanup: the MoE helper path now passes
  stack-backed output-shape arrays into `MetalKernel.DispatchOne` instead of
  per-call slice literals. This does not remove the remaining tiny dispatch
  allocation (`8 B/op` on matvec/split gate-up and `4 B/op` on weighted sum),
  so it is not the evaluated-graph parity fix. It is still a valid local
  hot-path cleanup: same-session `BenchmarkExpertIDMatVec_Q4_Gemma4_26B`
  improved from `202203 ns/op` to `182995 ns/op`,
  `BenchmarkExpertIDMatVec_Q4_Tiny` from `180817` to `159975`,
  `BenchmarkExpertIDGELUSplitGateUpMatVec_Q4_Tiny` from `175390` to `164880`,
  and `BenchmarkExpertIDWeightedMatVecSum_Q4_Tiny` from `173990` to `147444`.
  Focused expert-ID correctness tests pass. Treat this as 26B MoE helper
  hygiene, not an E2B retained decode win.
- `2026-05-24-decode-trace-go-mlx-gemma4-e2b-4bit-opencode-p51k-g128-native-model-greedy-keqv.json`:
  model-level native greedy diagnostic after fixing Gemma 4 K=V handling in
  the compiled/native layer graph. It completes cleanly at `89.235 tok/s` for
  `128/128` generated tokens, but it is not faster than the default path. The
  follow-up
  `2026-05-24-decode-trace-go-mlx-gemma4-e2b-4bit-opencode-p51k-g128-native-model-greedy-pinner.json`
  moves its per-token C/Go argument buffers to normal-layer-count
  stack-backed scratch pinned with `runtime.Pinner` and reuses the borrowed
  suppression tensor; the real Metal tests pass and the diagnostic improves to
  `90.174 tok/s`, but it still trails the default `90.453 tok/s` control.
  The later retained-State rows that set
  `GO_MLX_ENABLE_NATIVE_GEMMA4_MODEL_GREEDY=1` are not valid evidence for this
  wrapper because retained `state-ramp-profile` uses stochastic sampling, so
  `ModelSession.Generate` never enters the greedy-token shortcut. Keep this as
  a driver-profile-only greedy diagnostic unless a true greedy retained lane is
  explicitly being tested.
- `2026-05-24-decode-trace-go-mlx-gemma4-e2b-4bit-opencode-p51k-g128-compiled-keqv.json`:
  per-layer compiled decode remains rejected. The K=V graph mismatch was fixed,
  output and K/V shape guards were added, and the previous panic path now fails
  as a controlled empty-logits report after 4 generated tokens instead of
  corrupting cache state. Do not use `-compiled-gemma4-layer` for acceptance
  until the full local/global head-dim and eval-boundary semantics are fixed.
- `2026-05-24-decode-trace-go-mlx-gemma4-e2b-4bit-opencode-p51k-g128-native-layer-gated2.json`:
  per-layer native decode remains rejected. The paged-cache boundary now skips
  before CGO when no valid page exists, removing the missing-`prev_keys` class
  from that path, but the opt-in layer wrapper still hits Gemma 4 local/global
  head-dimension mismatches such as `(1,1,256)` versus `(1,1,512)`. Do not
  promote `-native-gemma4-layer` / `-native-gemma4-moe-layer` as defaults.
- `2026-05-24-decode-trace-go-mlx-gemma4-e2b-4bit-opencode-p51k-g32-native-layer-layerlog.json`,
  `2026-05-24-decode-trace-go-mlx-gemma4-e2b-4bit-opencode-p51k-g128-native-layer-full-skip.json`,
  and
  `2026-05-24-decode-trace-go-mlx-gemma4-e2b-4bit-opencode-p51k-g128-compiled-layer-full-skip.json`:
  the layer-log trace identifies the first bad opt-in native layer as Gemma 4
  layer `9`, type `full_attention`, with the real E2B split
  `(head_dim=256, global_head_dim=512)`. The per-layer native/compiled wrappers
  now skip those full-attention global-head-dim layers before CGO; the guard is
  covered by `TestDecode_gemma4PerLayerDecodeLayerUnavailableReason_Good` and
  `BenchmarkGemma4PerLayerDecodeLayerUnavailableReason_FullGlobal`
  (`1.486 ns/op`, `0 B/op`, `0 allocs/op`). The opt-in lanes now complete
  instead of panicking or empty-logit aborting, but they are slower than the
  default: native-layer full-skip records `68.464 tok/s` and compiled-layer
  full-skip records `63.364 tok/s` on the same `51242` prompt / `128` generated
  token diagnostic. This is a safety and evidence fix only, not a production
  speed path.
- `2026-05-23-current-go-mlx-gemma4-e2b-4bit-opencode-r10-g1024.json`: fresh
  rebuilt `lthn-mlx` retained-State run against
  `mlx-community/gemma-4-e2b-it-4bit`; `10` retained turns from a `30000` token
  first context, `63323` final live tokens, `28363` appended tokens, `4931`
  visible/generated tokens, `91.224s` workload wall time excluding `1.176s`
  model load, `16.426s` append time, `2635.838 tok/s` initial prefill,
  `1726.700 tok/s` retained append, `77.761 tok/s` raw decode,
  `61.759 tok/s` effective turn throughput, `3.142 GiB` process RSS, and
  `9122.440 J` estimated at `100 W`. This is a fresh wall/energy win over the
  same llama.cpp harness, but it is not an accepted production row because it
  predates the current default lane and used a historical `256` visible-token
  debug floor.
- `2026-05-23-current-llamacpp-gemma4-e2b-q4km-opencode-r10-g1024.json`:
  fresh llama.cpp server anchor against
  `gemma-4-E2B-it-Q4_K_M.gguf`; `10/10` turns, `67563` final live tokens,
  `27303` appended tokens, `10240` generated tokens, `10238` visible tokens,
  `133.629s` workload wall time after the server was already healthy,
  `34.162s` prompt time, `98.807s` decode time, `103.636 tok/s` raw decode,
  `76.615` visible tok/s wall throughput, and `13362.879 J` estimated at
  `100 W`. This row remains the raw decode anchor, but not a clean
  answer-volume anchor: every turn contains a visible orphan `<channel|>`
  marker and uses the full generation budget.

- `2026-05-21-after-hotpaths-go-mlx-gemma4-e2b-4bit-opencode-r10-g1024.json`:
  `10` retained turns from a `30000` token first context, `64178` final live
  tokens, `28363` appended tokens, `5787` visible/generated tokens,
  `101.898s` total wall time, `16.070s` append time, `77.350 tok/s` raw decode,
  `63.669 tok/s` effective turn throughput, `3.535 GiB` process RSS, and
  `10189.769 J` estimated at `100 W`.
- `2026-05-21-cache-pageview-go-mlx-gemma4-e2b-4bit-opencode-r10-g1024.json`:
  diagnostic run after reducing paged K/V append churn; `9` turns ok and `1`
  debug visible-token annotation, `63640` final live tokens, `28363` appended
  tokens, `5249` visible/generated tokens, `94.851s` wall time, `16.096s`
  append time, `77.495 tok/s` raw decode, `62.607 tok/s` effective turn
  throughput, `3523 MB` process RSS, and `9485.066 J` estimated at `100 W`.
  This row is useful for local delta tracking but is not an accepted production
  row because it predates the corrected natural-output methodology.
- `2026-05-21-cache-shape-go-mlx-gemma4-e2b-4bit-opencode-r10-g1024.json`:
  diagnostic run after caching paged K/V page layout metadata; `10/10` retained
  turns, `63973` final live tokens, `28363` appended tokens, `5582`
  visible/generated tokens, `99.460s` wall time, `16.162s` append time,
  `77.221 tok/s` raw decode, `63.107 tok/s` effective turn throughput,
  `3529 MB` process RSS, and `9945.972 J` estimated at `100 W`. This row
  restores the expected output shape after the bookkeeping cleanup but still
  does not close raw decode.
- `2026-05-21-cache-scratch-go-mlx-gemma4-e2b-4bit-opencode-r10-g1024.json`:
  diagnostic run after reusing borrowed page-state slice backing arrays; `8`
  turns ok and `2` debug visible-token annotations, `62963` final live tokens,
  `28363` appended tokens, `4571` visible/generated tokens, `85.298s` wall
  time, `16.031s` append time, `78.521 tok/s` raw decode, `61.554 tok/s`
  effective turn throughput, `3510 MB` process RSS, and `8529.827 J`
  estimated at `100 W`. This is a useful historical local diagnostic, not a
  production row under the corrected natural-output methodology.
- `2026-05-21-current-llamacpp-gemma4-e2b-q4km-opencode-r10-g1024.json`:
  `10/10` llama.cpp turns, `67563` final live tokens, `27303` appended tokens,
  `10240` generated tokens, `10237` visible tokens, `131.912s` wall time,
  `34.036s` prompt time, `97.074s` decode time, `105.486 tok/s` raw decode,
  `77.605` visible tok/s wall throughput, and `13191.239 J` estimated at
  `100 W`. The token-count side of this row is skewed by leaked thinking
  channel content; keep it as a speed anchor, not as a clean answer-volume
  baseline.

Interpretation: go-mlx's wall time is lower in these pairs and the llama.cpp
extra output is expected because that comparator leaked thinking/control-channel
text. Do not reject the retained-State wall-time angle on token count alone:
the fresh 2026-05-24 default workload finished `34.073s` faster than the
2026-05-23 llama.cpp anchor (`25.50%` less wall time and estimated energy at
the same `100 W` assumption) while producing a clean `10/10` go-mlx row. The
remaining hard speed gap is raw decode: go-mlx is still about `1.19x` behind
llama.cpp (`103.636 / 86.949`). That is no longer the earlier `1.33x` gap, but
it is still too large to treat as a raw-decode production pass. The next
optimisation target is the native decode/eval boundary and long-context
attention layout described in `IDEAS.md`, not more short-output benchmark rows.

Latest local microbenchmark delta: `BenchmarkPagedKVCache_AppendSingleTokenPageConcat_128`
improved from about `53168 B/op` and `3833 allocs/op` to `17472 B/op` and
`1282 allocs/op` after avoiding exact-token page slices, lazy `Owned` state
allocation, repeated page-shape queries, and per-token borrowed-state slice
allocation. The prealloc variant also improved from about `85137 B/op` and
`6026 allocs/op` to `51408 B/op` and `3599 allocs/op`, but it still costs more
memory than concat and remains diagnostic rather than a default.
The previous intermediate row was `19504 B/op` and
`1536 allocs/op` after avoiding exact-token page slices, lazy `Owned` state
allocation, and repeated page-shape queries.

Latest native State restore source delta: `metalKVSnapshotBlockSource` no
longer allocates and copies a second `[]kv.StateBlockRef` manifest for every
native prompt-cache/session restore. It validates contiguous prefix coverage,
stores only the covering block count, and indexes the original bundle slice
from the per-block loader. `BenchmarkBackend_MetalKVSnapshotBlockSource_Construct96Blocks`
improved from `2165 ns/op`, `18528 B/op`, `2 allocs/op` to `96.87 ns/op`,
`96 B/op`, `1 alloc/op`. This is a restore-path allocation cleanup, not a raw
decode fix; it keeps warm State restore closer to the intended streaming
layout before the pinned/mmap handoff work.

Latest fixed-cache restore delta: fixed-cache snapshots already own exact
prefix arrays, but `appendRestoreFixedCacheSnapshot` was copying those arrays
through `cacheSnapshotFloatArrays` and then copying the prefix again into the
restored fixed cache. The fixed-cache branch now borrows the snapshot arrays for
the source read and only performs the destination-prefix copy; the same restore
also hoists the default stream through `Zeros4WithStream` and
`SliceUpdateInplace4WithStream`. The focused 26-cache Gemma 4 restore run moved
from `452718 ns/op`, `4171 B/op`, `54 allocs/op` to `419152 ns/op`,
`4171 B/op`, `54 allocs/op`; repeated runs remain noisy under MLX eval
(`428445` to `466049 ns/op`), so treat this as a small fixed-cache restore
cleanup, not a benchmark acceptance row.

Current open gates:

- [x] Retained State can wake, append, generate, and report wall/decode/append,
      memory, and estimated energy without replaying the full first context.
- [x] The benchmark harness can run a realistic opencode-shaped `30k` first
      context with `10` retained turns and compare it against a llama.cpp
      anchor.
- [ ] Same-workload retained workflow beats or matches llama.cpp on wall time,
      raw decode, and estimated energy, with visible output counts and known
      thinking-channel leakage reported side by side rather than used to hide
      the speed result.
- [ ] Raw decode is within the acceptable calibration band. The current gap is
      `1.260x` versus llama.cpp on the no-env default `2048`-page
      request-context retained row, so this remains the primary code gap even
      though go-mlx now wins wall/energy on that same-shape pair.
- [ ] The default CLI path uses the fastest safe settings without requiring
      hidden extra flags.
- [ ] Long-output story/book turns remain coherent with `max_tokens` in the
      thousands, not only diagnostic `128` token outputs.
- [x] The `30k` to `100k` warm build-up and folded-State lifecycle are rerun
      after the decode/eval-boundary fixes and compared against one-shot/replay
      behaviour. The retained folded lifecycle now passes on the default paged
      hyper-long path and the current report emits same-binary replay estimates:
      retained wall `173.173s` versus `875.097s` replay estimate, a `5.053x`
      retained speedup and `70192.349 J` estimated saved at `100 W`.
- [ ] The seven `mlx-community` Gemma 4 E2B formats (`mxfp4`, `mxfp8`, `4bit`,
      `5bit`, `6bit`, `8bit`, `bf16`) are listed with go-mlx support status and
      llama.cpp anchors where a comparable GGUF quant exists.
- [ ] Canonical benchmark artefacts are regenerated and indexed after the code
      stabilises. The old `docs/runtime/2026-*` report set is being removed from
      this commit candidate and must not be cited as current acceptance evidence.

Default CLI tightening, 2026-05-25: `driver-profile` now seeds its public flag
defaults from `DefaultProductionLane()` instead of the older smoke shape. A
plain fast-lane profile therefore runs the production descriptor's `128` token
budget, `3` runs, hidden output, and token-phase tracing by default. Explicit
flags still override each field, including `-include-output` for captured text.
This is a default-path correction only; it does not close the raw decode gap by
itself.

Treat `IDEAS.md` as the active optimisation brief. Its highest-priority path is
strict MLX eval boundaries / graph lifetime control first, then pinned State
memory and C++23 `std::mdspan` layout work. Gemma 4 local/global attention
windowing, PLE handling, and K/V layout must be verified against the actual code
before declaring memory or decode fixed.

Do not close this goal because a short-context decode number is healthy. The
production claim is repeated-workflow wall time and retained-State savings under
real output budgets, with runner anchors and energy assumptions exposed.

## Production Acceptance Criteria

1. **Production runner win:** on the M3 Ultra target machine, go-mlx must beat
   configured Python/Metal alternatives such as `mlx_lm` and vLLM on a realistic
   opencode-sized repeated agentic workflow, or document why an alternative
   could not run the same workload. The required report must include model,
   quantisation, prompt length, context, token budget, load policy,
   cache/restore policy, raw decode, wall-clock time, setup time, estimated
   power/energy assumptions, and effective throughput. Use `100k` as a stress
   and degradation lane after the `30k`-`40k` workflow is healthy.
2. **External calibration, not permanent chasing:** use llama.cpp, `mlx_lm`,
   and vLLM to calibrate the lane. A small raw decode deficit, such as roughly
   5%, does not block the goal if go-mlx wins the repeated workflow wall-clock
   and no faster configured external runner exists for the same model/task.
   Once go-mlx is faster than available configured systems, future optimisation
   rounds benchmark against the current go-mlx best artefact unless an external
   runner produces a new realistic workflow win.
3. **Metric honesty:** keep raw visible decode, prefill, restore, wall-clock,
   input+output throughput, and decode-equivalent effective tok/s separate.
   Derived effective tok/s can remove the old round-number `100 tok/s` floor
   only when the report proves real 10+ turn time savings over replayed prefill.
   Estimated power must be labelled as an estimate unless backed by a real
   sampler, and joule deltas must name the assumed wattage. Speculative/MTP
   lanes must be labelled separately from no-draft raw decode.
4. **Native hot path:** expensive repeated decode work belongs in
   `go/internal/metal` and the MLX C/C++ wrapper. Go should own stable APIs,
   lifecycle, orchestration, settings, and reporting; it should not be doing
   avoidable per-token work that can stay in native MLX closures.
5. **No prefill regression:** restored project memory must answer smoke
   questions from durable state without feeding the source text back into the
   prompt.
6. **Agentic flow works end-to-end:** seed, wake, append task context, generate
   or continue work, compact, sleep, reload, and continue from the selected state
   or summary path.
7. **Portable contracts stay portable:** improvements in go-mlx must preserve
   the driver boundaries used by `go-inference/state`, go-ai, and go-ml so ROCm,
   CUDA, and future drivers can implement the same state and split-execution
   ideas.

## Current Baseline

Recent local measurements show that small activation-only changes are not
enough:

| Path | Result |
| --- | ---: |
| Clean Gemma 4 E2B 4-bit go-mlx driver profile | `~40.72 tok/s` |
| MLX `CompileShapeless` plus Go-defined activation fusion | `~44.94 tok/s` |
| Plain C++ native activation wrapper without MLX compile | `~41.87 tok/s` |
| C++ wrapper with cached MLX compiled activation closures | `~45.62 tok/s` clean, `~47.11 tok/s` traced short run |
| Current exact Gemma 4 E2B target command with token traces | `~44.56 tok/s`; steady `sample_eval_duration` averages `~20.98ms/token` |
| Native greedy/session decode-tail rerun | `44.93695802859693 tok/s` |
| Gated last-token output projection rerun | `44.874611039475575 tok/s`; steady `sample_eval_duration` averages `~20.88ms/token` |
| Gated native MLP sub-block rerun | `43.10698466210642 tok/s`; disabled by default because it regresses |
| Native MLP gate-off default rerun | `44.89465488606482 tok/s`; steady `sample_eval_duration` averages `~20.81ms/token` |
| Resolved-load target rerun after host-memory planner fix | `46.50145764359926 tok/s`; default target command now reports `cache_mode=paged` |
| Gated Gemma 4 native phase trace | diagnostic only; `native_events` show the remaining work is evaluated graph time; the 26B FFN split trace attributes the largest sub-bucket to routed experts at `13.736ms/token` |
| Native layer gate-off control rerun | `47.054122991613305 tok/s`; current best default target rerun on rebuilt binary |
| Gated one-token Gemma 4 native layer wrapper | `44.54197676930399 tok/s`; disabled by default because eval time regresses |
| Gated MLX-compiled Gemma 4 layer attempt | fail-closed diagnostic; MLX compile rejects the growing cache broadcast shape and falls back |
| Experimental fixed-cache compiled Gemma 4 layer | best bucketed probe `47.03732918131478 tok/s` at 96 slots; full-context 4096-slot topology regresses to `39.88411733551154 tok/s` |
| Fixed-cache native bridge compiled Gemma 4 layer | full-context 4096-slot gated path `107.77701729520602 tok/s`; valid 3-run E2B target-capacity result, but not default and not the llama.cpp parity target |
| Gated direct greedy token projection | `44.27055794965946 tok/s`; disabled by default because it shifts the same lazy forward materialisation into `Eval(next)` and regresses |
| Dense linear transpose cache probe | `45.9393904182794 tok/s`; reverted because it regressed the default paged-cache band |
| Gated compiled Gemma 4 per-layer inputs | `46.93672879306734 tok/s`; disabled by default because same-binary gate-off was `46.9841490339839 tok/s` |
| Correctness-breaking disabled per-layer-input diagnostic | `114.9355811775564 tok/s`; diagnostic only because it omits required Gemma 4 per-layer inputs and produces invalid model semantics |
| Quantized embedding row-gather default path | `121.9379742475021 tok/s` on the exact Gemma 4 E2B target command; valid path, generated `[20,20,20]` tokens, peak memory `3166205126` bytes |
| Final Gemma 4 E2B no-thinking template row-gather rerun | `124.88170583124456 tok/s` on the exact target command; valid path, generated `[128,128,128]` tokens, peak memory `3177609258` bytes |
| Gemma 4 E2B mixed-quant loader revalidation | `121.19859628423075 tok/s` on the exact target command; valid path, generated `[128,128,128]`, peak memory `3177560106` bytes |
| Archived shared Gemma 4 31B q4 `mlx_lm.generate` datapoints | historical context only; no longer an active benchmark target |
| Shared Gemma 4 31B q4 go-mlx current default shared-snapshot rerun | `24.663669410625896 tok/s` across three no-thinking runs; retained as internal large-model evidence |
| Shared Gemma 4 31B q4 mixed-quant loader rerun | `24.971269037945117 tok/s` across three no-thinking runs; retained as internal large-model evidence |
| Shared Gemma 4 31B q4 sustained no-thinking shared-snapshot run | go-mlx `23.086428954337055 tok/s` across three full 128-token runs; retained as internal large-model evidence |
| Shared Gemma 4 31B q4 fixed-cache native bridge probe | full 4096-slot native bridge first exposed the missing 512-wide SDPA resource; guarded 160-slot fallback runs at `24.94401176949734 tok/s`; opt-in wide-head matmul bridge runs at `24.333176943291804 tok/s`; patched 512-wide SDPA runs cleanly at `24.70397262176645 tok/s`; shared host-fed mask is neutral at `24.904493509253538 tok/s` fallback and `24.767920780634018 tok/s` with SDPA512, so attention/mask alone is not the 31B large-model boundary |
| Shared Gemma 4 31B q4 gated native MLP rerun | `24.7143167044012 tok/s`; disabled because it regresses the mixed-quant default |
| Shared Gemma 4 31B q4 gated native GELU probe | `25.260023959706817 tok/s` for one run; disabled because it is not a stable default-path improvement |
| Shared Gemma 4 31B q4 direct greedy output probe | `23.2767195467288 tok/s` across three full 128-token runs; disabled because it regresses the sustained default |
| Shared Gemma 4 31B q4 async prefetch current-order probe | `24.41755011370027 tok/s` for one traced run; disabled because it only moves timing buckets |
| Gemma 4 26B A4B go-mlx q4 vs llama.cpp Q8 decode | go-mlx `55.96521969803896 tok/s`, llama.cpp `87.688525 tok/s`; llama.cpp is `1.57x` faster |
| Gemma 4 26B A4B go-mlx q4 vs llama.cpp Q8 long prefill | go-mlx `864.6062359771336 tok/s` at 2061 tokens, llama.cpp `2231.973259 tok/s` at 2048 tokens; llama.cpp is `2.58x` faster |
| Gemma 4 26B A4B go-mlx q4 fused expert gate/up plus auto last-token long prefill vs llama.cpp Q4_K_M decode | go-mlx `56.220244342267904 tok/s`, llama.cpp `89.000726 tok/s`; llama.cpp is `1.58x` faster |
| Gemma 4 26B A4B go-mlx q4 fused expert gate/up plus auto last-token long prefill vs llama.cpp Q4_K_M long prefill | go-mlx `903.0290085147915 tok/s` at 2061 tokens, llama.cpp `2184.109033 tok/s` at 2048 tokens; llama.cpp is `2.42x` faster |
| Gemma 4 26B A4B expert-ID fused activation diagnostic | same-binary default `56.21477992583666 tok/s`, expert-ID fused activation `56.295534088943356 tok/s`; only `+0.14%`, llama.cpp Q4_K_M still `1.5809x` faster |
| Gemma 4 26B A4B sorted expert prefill vs llama.cpp Q4_K_M long prefill | go-mlx `1914.0303789361128 tok/s` at 2204 tokens, llama.cpp `2184.109033 tok/s` at 2048 tokens; llama.cpp is `1.14x` faster |
| Gemma 4 26B A4B sorted prefill plus multi-page fast-concat decode vs llama.cpp Q4_K_M long-context decode | go-mlx `42.372384580120396 tok/s` decode at 2204-token context, llama.cpp `92.624334 tok/s` at `p2048`; llama.cpp is `2.19x` faster |
| Gemma 4 26B A4B sorted prefill plus fixed-cache compiled decode vs llama.cpp Q4_K_M long-context decode | go-mlx `48.93511098804883 tok/s` decode at 2204-token context, llama.cpp `92.624334 tok/s` at `p2048`; llama.cpp is `1.89x` faster |
| Gemma 4 26B A4B sorted prefill plus fixed-cache compiled direct-greedy decode vs llama.cpp Q4_K_M long-context decode | go-mlx `49.75515922842408 tok/s` 3-run decode at 2204-token context, llama.cpp `92.624334 tok/s` at `p2048`; llama.cpp is `1.86x` faster |
| Gemma 4 26B A4B sorted prefill plus expert-ID fused direct-greedy decode vs llama.cpp Q4_K_M long-context decode | go-mlx `49.973204322219345 tok/s` 3-run decode at 2204-token context, llama.cpp `92.624334 tok/s` at `p2048`; llama.cpp is `1.85x` faster |
| Same prompt length llama.cpp Q4_K_M check | go-mlx `1915.3373741969128 tok/s` prefill and `49.973204322219345 tok/s` decode at 2204-token context; llama.cpp `pp2204` is `2109.335561 tok/s` and `tg128` is `91.451031 tok/s`; llama.cpp is `1.10x` faster on prefill and `1.83x` faster on decode |
| Gemma 4 26B A4B fixed-cache sliding-window diagnostic | preserving the 1024-token sliding cache bound inside the fixed-cache lane completes after fixed-cache overflow correctness fixes, but regresses to `1806.8318924630082 tok/s` prefill, `40.76006207167587 tok/s` decode, and `71228950132` peak bytes; rejected as the active lane |
| Current restored fixed-uniform cache lane vs same-prompt llama.cpp Q4_K_M | go-mlx `1923.322483219664 tok/s` prefill and `49.71518402860789 tok/s` decode at 2204-token context; llama.cpp `pp2204` is `2109.335561 tok/s` and `tg128` is `91.451031 tok/s`; llama.cpp is `1.0967x` faster on prefill and `1.8395x` faster on decode |
| Gemma 4 26B A4B expert down two-column diagnostic | a llama.cpp-inspired two-output down matvec completed with empty stderr but regressed to `1732.6641621430529 tok/s` prefill and `48.4963971321882 tok/s` decode; reverted as a kernel-shape dead end |
| Current router-residual parity lane vs same-prompt llama.cpp Q4_K_M | go-mlx routes Gemma 4 MoE logits from the attention residual like llama.cpp, while experts still consume the pre-FFN2-normalised tensor; the 3-run prompt-file lane records `1933.6368792628773 tok/s` prefill and `50.23367760579547 tok/s` decode, leaving llama.cpp `1.0909x` faster on prefill and `1.8205x` faster on decode |
| Gemma 4 26B A4B active split expert-ID path vs same-prompt llama.cpp Q4_K_M | the active MLX safetensors store expert `gate_proj` and `up_proj` separately with BF16 sidecars, so the earlier fused-`gate_up` expert-ID gate had been falling back; the split expert-ID path records `1939.2172632050945 tok/s` prefill and `62.52025013199337 tok/s` decode, leaving llama.cpp `1.4628x` faster on decode |
| Gemma 4 26B A4B split fused-activation expert-ID path vs same-prompt llama.cpp Q4_K_M | the split path now fuses `GELU(gate) * up` in the custom expert-ID kernel and traces active `activation_split_id_matvec` plus `down_weighted_sum_id_matvec`; it records `1941.0884632916652 tok/s` prefill and `68.22675114228564 tok/s` decode, leaving llama.cpp `1.3404x` faster on decode |
| Current split fused-activation shared-input expert-ID lane vs same-prompt llama.cpp Q4_K_M | shared-input kernels avoid broadcasting the single hidden row to one row per routed expert; the 3-run README prompt-file lane records `1923.9974775252285 tok/s` prefill and `70.54498924012704 tok/s` decode, leaving llama.cpp `1.0963x` faster on prefill and `1.2964x` faster on decode |
| Current split fused-activation token-phase profile | same lane, one run with `-trace-token-phases`, records `71.59452329863376 tok/s`; steady tokens average `14.0596ms`, with `12.7249ms` in `Eval(next)` and `1.2977ms` in next-forward graph construction |
| Current split fused-activation native MLP probe | `GO_MLX_ENABLE_NATIVE_MLP_GELU=1` is neutral-to-negative on the active 26B A4B q4 lane at `71.44678366026884 tok/s`, so standalone dense MLP wrapping is not the next parity boundary |
| Current packed-column expert-ID lane vs same-prompt llama.cpp Q4_K_M | expert-ID q kernels now iterate packed q words instead of scalar input columns, avoiding repeated q4 word loads; the final 3-run README prompt-file lane records `1936.5495347431952 tok/s` prefill and `79.1105587686013 tok/s` decode, leaving llama.cpp `1.0892x` faster on prefill and `1.1560x` faster on decode |
| Current right-sized fixed-cache packed expert-ID lane vs same-prompt llama.cpp Q4_K_M | setting `GO_MLX_FIXED_GEMMA4_CACHE_SIZE=2336` for the 2204-token README prompt plus 128-token decode avoids making attention scan the full 4096-slot fixed cache; the 3-run lane records `1937.0948107149452 tok/s` prefill and `84.23477753697784 tok/s` decode, leaving llama.cpp `1.0889x` faster on prefill and `1.0857x` faster on decode |
| Superseded right-sized fixed-cache packed expert-ID diagnostic vs same-prompt llama.cpp Q4_K_M | the generation cache builder derived the fixed-cache size from `prompt_tokens + max_tokens`, rounded to 32, when the fixed Gemma 4 cache gate was enabled and `GO_MLX_FIXED_GEMMA4_CACHE_SIZE` was unset; the same README 3-run lane recorded `1935.3610403257746 tok/s` prefill and `84.01009717307203 tok/s` decode, leaving llama.cpp `1.0899x` faster on prefill and `1.0886x` faster on decode. This is retained as diagnostic history only; production retained state is paged/no-fixed by default |
| Agentic 10-run fixed-cache retained-prefix bench | on the active packed expert-ID lane, one cold README prompt prefill plus nine fixed-cache prompt-cache wakes records `84.98980513059084 tok/s` decode, `4.674699ms` average restore time for the 2204-token retained prefix, and `471474 tok/s` retained-prefix setup equivalent; compared with re-prefilling the same prefix every batch, prompt setup drops from `10.567751250s` to `1.098864083s` over ten batches |
| Rejected native router top-k probe on fixed-cache packed expert-ID lane | the gated single-token router top-k/softmax Metal kernel proves fixed-cache prompt restore works, with run 2/3 restoring the 2204-token prompt in about `4.7ms`, but decode averages only `83.54086813967548 tok/s`; llama.cpp remains `1.0947x` faster on decode, so this is not the active parity lane |
| Native fixed-owner attention boundary probe | `GO_MLX_ENABLE_NATIVE_GEMMA4_FIXED_OWNER_ATTENTION=1` moves Q/K/V projection, Q/K RMSNorm, RoPE, fixed-cache update, masked SDPA, and O projection behind a stable `go/internal/metal` C++ wrapper, with a q4 compiled branch for the active fixed-mask path. It is correct but neutral on the same README 3-run lane: same-binary gate-off records `84.59149676385168 tok/s`, gate-on q4-compiled records `84.75303439310541 tok/s`, and same-prompt llama.cpp Q4_K_M remains `1.0790x` faster at `91.451031 tok/s`; keep it gated rather than default |
| Rejected native residual-norm probe | `GO_MLX_ENABLE_NATIVE_GEMMA4_RESIDUAL_NORM=1` compiles the attention residual `residual + RMSNorm(attnOut)` bucket into a reusable native wrapper and passes focused Metal tests, but the active README lane regresses to `84.36852051087726 tok/s`; this confirms the residual bucket is not the next default-path fix |
| Rejected combined attention-residual probe | `GO_MLX_ENABLE_NATIVE_GEMMA4_FIXED_OWNER_ATTENTION_RESIDUAL=1` combines the fixed-owner attention wrapper with post-attention RMSNorm and residual add so the whole attention-residual section crosses the boundary together. Dense and q4 compiled Metal tests pass, but the active README lane records only `84.4324627031718 tok/s`, below the fixed-cache control band, so it stays diagnostic |
| Rejected generic native MoE full-layer probe | The expanded `GO_MLX_ENABLE_NATIVE_GEMMA4_LAYER=1` ABI now supports q4/q8 ordinary linears, optional per-layer inputs, fixed-cache K/V owners, and tied K/V attention, and the traced 26B README lane proves all 30 layers can emit `native_layer`. That path is slower: the 10-run ours-only bench records `51.70264804488751 tok/s` decode with empty stderr. The root cause is boundary shape, not context length: pinning `-context 4096` still records `51.72847744673013 tok/s`, while the same binary with the native layer gate off records `84.67834684564139 tok/s` over three runs. The production guard now skips MoE layers unless `GO_MLX_ENABLE_NATIVE_GEMMA4_MOE_LAYER=1` is explicitly set, preserving the faster expert-ID kernel path by default |
| MoE-gated native-layer guard rerun | After adding the separate MoE native-layer gate, a trace with `-native-gemma4-layer` but without `-native-gemma4-moe-layer` emits 30 `moe native layer is disabled` skip reasons and no stderr. The post-guard 10-run README lane records `425831.7097091192 tok/s` retained-prefix prefill, `84.8683681726259 tok/s` decode, `84.9427850414965 tok/s` warm decode, `4.658939ms` average restore, and empty stderr. This restores the prior active 85 tok/s band while documenting that a full production native boundary must preserve the custom packed expert-ID kernels rather than replacing them with generic switch-linear MLX graph work |
| Rejected q4 expert-ID unrolled shader probe | `GO_MLX_ENABLE_EXPERT_ID_UNROLLED_Q4=1` manually unrolls the active q4 packed inner loop for the split gate/up activation and weighted-down expert-ID kernels. Focused Metal tests pass and stderr stays empty, but the 10-run README lane records `84.73372132835443 tok/s` decode and `84.84637816824524 tok/s` warm decode, slightly below the MoE-gated guard lane, so this remains a diagnostic gate rather than the production path |
| Trace-name formatting hot-path cleanup | native phase trace names are now formatted only when `GO_MLX_TRACE_FORWARD_EVAL=1` is enabled, and the decode layer reads the trace gate once per forward. The one-run token-phase profile shows graph construction moving only slightly, but the normal 10-run README lane records `427000.78466006636 tok/s` retained-prefix setup, `85.22730571622206 tok/s` decode, `85.3267114104144 tok/s` warm decode, `4.646185ms` average restore, and empty stderr. This is a small default-path cleanup, still below the `>=100 tok/s` floor and llama.cpp Q4_K_M decode parity |
| Native router matvec plus top-k probe | `GO_MLX_ENABLE_NATIVE_GEMMA4_ROUTER_MATVEC=1` replaces the tiny q8 router projection with a custom Metal matvec; pairing it with the existing native router top-k gate gives a 10-run README lane at `425482.7192523824 tok/s` retained-prefix setup, `86.06590721922689 tok/s` decode, `86.15307046004646 tok/s` warm decode, `4.662805ms` average restore, and empty stderr. The token-phase profile records `83.45742599530926 tok/s`, steady `10.5825ms` eval and `1.4308ms` forward graph construction, so this is a real but small router win, still below the `>=100 tok/s` floor and llama.cpp Q4_K_M decode parity |
| Native router plus dense MLP matvec retained-prefix probe | adding `GO_MLX_ENABLE_NATIVE_MLP_MATVEC=1` on top of the router matvec/top-k lane gives the current best 10-run README lane at `423630.8407376839 tok/s` average prefix setup, `86.95798305515721 tok/s` decode, `87.13332867474983 tok/s` warm decode, `4.683662ms` average restore, and empty stderr. For ten 2204-token agentic batches, retained state reduces prompt setup from `10.53230291s` of replayed prefill to `1.09538325s`, a `9.615176158664102x` setup speedup while decode remains below the `>=100 tok/s` floor and llama.cpp Q4_K_M parity |
| Runtime-gate hot-path cleanup | hot runtime gates now cache `SetRuntimeGate` overrides in atomics so the active single-token decode path does not repeatedly take the generic runtime-gate lock/env path. The current README 10-run lane records `423698.49297158385 tok/s` average prefix setup, `87.05458770800922 tok/s` decode, `87.16243827560751 tok/s` warm decode, `4.683013ms` average restore, and empty stderr. This preserves the 87 tok/s band but is not a material parity move |
| Agentic effective 10-step retained-state rerun | fresh current-source 10-step ours-only README run records `87.15020057594002 tok/s` average raw decode and `87.995764012926 tok/s` warm raw decode with empty stderr. Against same-prompt llama.cpp Q4_K_M decode at `91.451031 tok/s`, warm raw decode is `3.7782701291514065%` behind, so the strict within-1% parity clause is not met. Retained prefix setup still saves `9.49244888s` over ten turns: replayed prefill would take `10.59383417s`, retained setup takes `1.10138529s`, warm restore averages `4.665569ms`, and warm restore is `227.06414094400918x` faster than the cold `1.059383417s` README prefill. Crediting the saved setup seconds as decode-equivalent work gives `128.6485922304177` effective visible tok/s, while input-plus-output agentic throughput is `1423.6841246167085 tok/s`; both are labelled derived metrics, not raw decode |
| Agentic 10-step energy-estimate rerun | `driver-profile -estimate-power-watts 100` now records an explicit estimated-energy block. The same retained-state README shape records `87.74067183813047 tok/s` raw decode, `87.84861155177613 tok/s` warm decode, `16.252888247s` total wall time, and empty stderr. At the normalised `100 W` assumption, the run is `1625.2888247 J` total, `1.269756894296875 J/visible-token`, and retained prefix setup saves `9.406740417s` or `940.6740417 J` versus replaying the cold prompt setup every turn. These joules are estimates and scale linearly with the assumed watts |
| Current fast-lane 10-step refresh | the rebuilt `-fast-gemma4-lane` shortcut is back in the same 87 tok/s band rather than the stale slower shortcut sample. Chat-mode README records `86.96995653092598 tok/s` average raw decode, `87.10762008324762 tok/s` warm raw decode, `16.413198251s` wall time, `1641.3198251 J` at the normalised `100 W` estimate, and empty stderr. Raw prompt mode records `87.18727600068239 tok/s` average raw decode, `87.28239963327297 tok/s` warm raw decode, `16.382709584s` wall time, `1638.2709584 J`, and empty stderr. This refresh narrows reporting drift, but go-mlx still trails the persistent in-process `mlx_lm` cached-prefix README workflow by about `1.53-1.56s` over ten turns including load |
| Accepted generation-stream fast-lane refresh | studying `mlx_lm` shows its generator builds on `mlx` `0.31.2` / `mlx_lm` `0.31.3`, uses a dedicated `mx.new_thread_local_stream(mx.default_device())`, and queues one-token-ahead `mx.async_eval`. The existing Go async prefetch gate regresses slightly on the current lane: `86.55268124366343 tok/s`, `16.496068705s`, and `1649.6068705 J` versus the refreshed control at `86.96995653092598 tok/s`, `16.413198251s`, and `1641.3198251 J`. A narrower Go generation-stream gate is positive and now included in `-fast-gemma4-lane`: the no-explicit-stream shortcut validation reports `GO_MLX_ENABLE_GENERATION_STREAM=1`, `87.50749912985658 tok/s`, `16.334514708s`, `1633.4514708 J`, and empty stderr; the explicit diagnostic sample reached `88.10704229468793 tok/s` and `16.239494334s`. This is superseded by the restored shared-mask balance row below |
| Restored short-context fast-lane balance | the current `-fast-gemma4-lane` default keeps the accepted shared-mask gate set and is back in the desired first-run shape before retained-state credit. The rebuilt default 3-run README profile records `GO_MLX_ENABLE_FIXED_GEMMA4_SHARED_MASK=1`, `88.5760834806412 tok/s` average decode, `87.87017208983966 tok/s` first-run decode, `2094.1931616252605 tok/s` first-run prefill, `5.971295375s` wall time, and empty stderr. The same-gate 10-run shared-mask sample records `88.50777967819847 tok/s` average decode, `88.61333712754153 tok/s` warm decode, `2100.679478883641 tok/s` first-run prefill, `16.146115667s` wall time, and `1614.6115667 J` at `100 W`. Against same-prompt llama.cpp Q4_K_M (`pp2204=2109.335561 tok/s`, `tg128=91.451031 tok/s`), go-mlx reaches `99.5896299158653%` of first-run prefill and `96.78160946944215%` of raw decode. The checked neighbours stay diagnostic: attention O-proj matvec is `88.53279331842275 tok/s`, row cache update is `86.57971461366179 tok/s`, and no-shared-mask is not a stable 10-run win |
| Rejected current-source `gather_qmm` decode control | disabling `-expert-id-matvec` and `-expert-id-fused-activation` while keeping fixed cache, shared mask, direct greedy, sorted prefill, native router matvec/top-k, and native MLP matvec on records only `54.02683426487331 tok/s` average decode and `54.10799458992597 tok/s` warm decode with empty stderr. The active expert-ID lane is about `62.4%` faster than this control, so MLX `gather_qmm` fallback is not the path to the `mlx_lm` raw-decode gap in the current Go stack |
| Rejected current-stack fixed-owner attention rerun | re-enabling `-native-gemma4-fixed-owner-attention` on top of the current expert-ID, fixed-cache, router, direct-greedy, sorted-prefill, and native-MLP stack records `85.20005681731622 tok/s` average decode, `16.718573375s` wall time, and empty stderr. The current control is `87.74067183813047 tok/s` and `16.252888247s`, so the fixed-owner attention gate regresses decode by `2.8956%`, adds `0.465685128s`, and costs about `46.5685128 J` at the normalised `100 W` estimate |
| Configured `mlx_lm` 26B q4 README calibration | repaired parity venv `mlx_lm.generate` loads the same MLX-community 26B A4B q4 snapshot with `--max-kv-size 2336`, README stdin, temp 0, and 128 generated tokens. It records `2207` prompt tokens at `1506.907 tok/s` and `128` generation tokens at `109.958 tok/s`, peak `15.739 GB`. This means Python MLX is faster than go-mlx on raw decode and remains the main external codebase to study before retiring the old round-number throughput target |
| Configured `mlx_lm` prompt-cache calibration | `mlx_lm.cache_prompt` processes the README prefix at a final `2197.23 tok/s` and writes a `243 MB` prompt cache; `mlx_lm.generate --prompt-cache-file` then processes a 5-token suffix at `27.813 tok/s` and generates at `109.325 tok/s`, peak `14.841 GB`. The CLI timing does not include model load or cache-file load, but it proves the Python MLX stack has a fast cached-prefix path as well as faster raw decode |
| Configured `mlx_lm` cached-prefix CLI 10-turn wall-clock calibration | ten `mlx_lm.generate --prompt-cache-file` turns against the already-created README cache record `36.98s` wall time while preserving fast per-run generation stats averaging `109.5251 tok/s`; this excludes cache creation, but includes per-turn process/model/cache load because that is the configured CLI runner shape. The matching go-mlx retained-state energy rerun is `16.252888247s`, so go-mlx is `2.2753x` faster wall-clock for this CLI workflow. At the normalised `100 W` estimate, the external CLI loop is `3698 J`, go-mlx is `1625.2888247 J`, and go-mlx saves `2072.7111753 J` over ten turns |
| Configured `mlx_lm` in-process cached-prefix 10-turn calibration | a persistent Python harness loading the same model and prompt cache once, then deep-copying the cache for ten 128-token turns, records `13.358959957957268s` generation wall time and `14.851929999887943s` including load. It averages `109.65707805632005 tok/s` generation and `86.18408516668592` wall visible tok/s including load. This is faster than the restored shared-mask go-mlx `-fast-gemma4-lane` retained-state run by `1.2941856671120566s` over ten turns including load; excluding Python load, the gap is about `2.787155709042733s`. At the same normalised `100 W` estimate, `mlx_lm` is `1485.1929999887943 J` including load versus go-mlx's `1614.6115667 J` restored shared-mask refresh. This remains useful calibration, but the active q4-first goal lane no longer blocks on the old short-context Python cached-prefix shape after the long-context/8k-return q4 evidence |
| Large-context retained-state diagnosis at 24k and 29k prompt tokens | repeating the README prompt to `24212` prompt tokens with `context=32768` records cold prefill `55.555967333s`, cache-hit restore about `0.5s`, but top-level cache-hit first-token time around `72-74s` because the full prompt string is still tokenised before the model metrics begin. The `28612` token opencode-shaped run makes the cliff clearer: cold prefill is `87.872341208s`, cache restore is `0.497940792s`, but run 2 still takes `115.383811292s` wall time with `111.082583667s` driver overhead. The state restore is working; the repeated giant string tokenisation is the large-context double-work boundary |
| Prefill chunk-size `1024` large-context probe | lowering model prefill chunks from `4096` to `1024` on the `28612` token prompt improves cold model prefill from `87.872341208s` to `70.193964333s`, but cache-hit wall time remains `110.010683625s` with `105.659096458s` driver overhead. Smaller model prefill chunks help ingestion shape, but they do not solve repeated-turn overhead while the driver still tokenises one giant prompt each turn |
| Raw chunked prompt stream large-context 10-turn probe | `driver-profile -chat=false -prompt-chunk-bytes 4096 -prefill-chunk-size 1024` feeds the same repeated README text as bounded prompt chunks. It records `28625` prompt tokens, `115.288840001s` total for ten 128-token turns, `33.48494955572712 tok/s` average raw decode, and empty stderr. The cold turn takes `78.403770292s`; warm turns are about `4.1s`, with restore averaging `280.517444ms` and warm driver overhead around `18ms` instead of `~105s`. At the normalised `100 W` estimate, the ten-turn run is `11528.8840001 J`, retained setup saves `626.183063256s` versus replayed cold prefill, and that setup saving is `62618.3063256 J`. This proves chunked prompt tokenisation removes the 29k repeated-turn cliff |
| Chat-mode chunked prompt stream large-context 10-turn probe | `driver-profile -prompt-chunk-bytes 4096 -prefill-chunk-size 1024` now chunks the native chat template path instead of requiring raw `-chat=false` mode. The opencode-shaped repeated README chat run records `28637` prompt tokens, `115.247971709s` total for ten 128-token turns, `33.58024749556697 tok/s` average raw decode, and empty stderr. The cold turn takes `78.4869145s`; warm turns remain about `4.08-4.10s`, restore averages `278.342120ms`, and warm driver overhead stays around `18-22ms`. At the normalised `100 W` estimate, the run is `11524.7971709 J`, retained setup saves `626.722864295s`, or `62672.2864295 J`, versus replayed cold prefill. This makes the chunked large-context fix apply to normal chat-mode diagnostics |
| Superseded Gemma 4 fast-lane shortcut with fixed-cache gates | the old `driver-profile -fast-gemma4-lane` shortcut applied expert-ID matvec, fused expert activation, sorted expert prefill, native MLP matvec, native router matvec/top-k, fixed Gemma 4 cache, shared fixed mask, direct greedy token, and the dedicated generation stream. That fixed-cache default is rejected: the current fast lane keeps fixed Gemma 4 K/V and shared fixed masks out of production defaults, keeps paged K/V as the retained-State default, and only keeps the older rows as diagnostic history. Rejected broad wrappers such as native full layer, native model greedy, fixed-owner attention, attention O-proj matvec, and generic native linear matvec remain excluded |
| Fast-lane long-context prefill-chunk sweep and default validation | the opencode-shaped `28637` token chat sweep with `-prompt-chunk-bytes 4096` records cold prefill `82.128389084s` at chunk `128`, `74.8167155s` at `256`, `67.631178917s` at `512`, `69.769200709s` at `1024`, `73.696338791s` at `2048`, and `85.410324s` at `4096`. The curve is not monotonic: `512` is the measured elbow where chunks are small enough for natural model ingestion but not so small that per-chunk overhead dominates. The first rebuilt no-explicit-chunk fast-lane validation recorded `load.prefill_chunk_size=512` and `prompt_chunk_bytes=4096` by default, with `84.995550583s` wall time, `33.22422183528957 tok/s` average raw decode, `298.090812ms` average restore, `8499.5550583 J` at the normalised `100 W` estimate, and empty stderr; it is now superseded by the promoted sliding-cache-bound long-context default. This supersedes the older `1024` default artefact, which took `86.433517249s` |
| Same-length 29k llama.cpp calibration | the Metal comparator must run outside the sandbox and should not force `GGML_METAL_DEVICES=0`, which filters the device out for this build; the working invocation uses the embedded Metal library and reports `MTL0: Apple M3 Ultra`. On the same local Q4_K_M GGUF, `llama-bench -p 28637 -n 1 -r 1 -ngl 99 -fa 1` records `1525.801226 tok/s` prefill in `18.768499791s`, while `-pg 28637,128` records pure `tg128` decode at `92.211737 tok/s` and combined `pp28637+tg128` throughput at `1398.527504 tok/s` over `20.568061709s`. Against the current go-mlx long-context retained-state artefact, cold prefill is `419.11716620820545 tok/s`, warm retained decode is `33.91056160965191 tok/s`, and the cold prompt-plus-decode run takes `76.811422833s`, leaving llama.cpp `3.64x` faster on same-length cold prefill, `2.72x` faster on raw decode, and `3.73x` faster on the comparable cold wall-clock. The retained-state workflow still removes repeated prefix replay, but the next performance boundary is long-context fixed-cache/attention scaling rather than another `512` vs `640` default tweak |
| Promoted sliding fixed-cache bound | `GO_MLX_ENABLE_FIXED_GEMMA4_SLIDING_CACHE_BOUND=1` keeps Gemma 4 sliding-attention fixed caches at their native window while full-attention layers remain request-sized. It was first promoted only for long-context `-fast-gemma4-lane` runs, but the 2026-05-24 `metrics.cache_profile` smoke proved the normal `4096` context shortcut still leaked local windows, so the gate is now part of the default Gemma 4 fast lane as well. The first diagnostic proved the performance shape but missed prompt-cache restore; after fixed-cache snapshots learned to store bounded tail state with the full logical prefix offset, the no-explicit-flag `context=32768` validation records `GO_MLX_ENABLE_FIXED_GEMMA4_SLIDING_CACHE_BOUND=1`, `prefill_chunk_size=512`, `prompt_chunk_bytes=4096`, `36.868437918s` total for three `28637` token turns, `62.51129327845945 tok/s` average decode, `62.63259219208622 tok/s` warm decode, `1094.4247968802333 tok/s` cold prefill, `21.757104ms` average restore, `3686.8437918 J` at `100 W`, and empty stderr. Compared with the previous long-context default this is `0.434x` the wall time and energy, `1.88x` raw decode, `1.85x` warm decode, `2.61x` cold prefill, and `13.70x` faster restore. The same-length llama.cpp gap shrinks to `1.39x` on cold prefill, `1.47x` on raw decode, and `1.59x` on cold prompt-plus-decode wall-clock |
| Long-context sliding-bound trace attribution | the promoted `32768` context fast-lane trace records `1096.311492962768 tok/s` prefill and `59.84070210617055 tok/s` decode with token phases enabled. Steady non-final tokens average `17.746205ms`, with `16.3555565ms` in `Eval(next)` and `1.346199ms` in forward graph construction. The diagnostic native-event trace is slower by design, but attributes materialised time to attention first (`73.077582ms` over 90 events), then local MLP (`23.520166ms`), split expert activation (`23.266755ms`), router (`22.603662ms`), attention residual (`21.01459ms`), and expert down (`20.881961ms`). This keeps the next large-context target in full-attention graph/kernel work rather than prompt-cache restore, chunk size, or Go driver orchestration |
| Rejected long-context fixed-owner attention reruns | re-enabling the original all-layer `-native-gemma4-fixed-owner-attention` on top of the promoted `32768` context shortcut records `36.44726s` wall time, `62.317460438377985 tok/s` average decode, `19.824229ms` average restore, and empty stderr. Narrowing that diagnostic to the five full-attention owner layers is cleaner but still flat at `36.426556958s`, `62.48077885938384 tok/s`, and `20.02152ms` average restore. It does not close the llama.cpp decode gap, so fixed-owner attention remains a diagnostic wrapper rather than a long-context default |
| Long-context shared-mask and dynamic-update diagnostics | manually omitting `GO_MLX_ENABLE_FIXED_GEMMA4_SHARED_MASK` from the same long-context gate set records `36.337556126s` wall time and `62.79482183164808 tok/s` decode, a small 29k-only gain that is not promoted because the short README lane previously needed the shared mask for the active band. A gated MLX dynamic `slice_update` experiment for fixed K/V writes records `36.582005083s` and `62.45483265128252 tok/s`, so replacing `put_along_axis` with that primitive is not the missing KV slot update fix |
| Rejected long-context wide-head attention diagnostics | forcing the existing 512-wide native SDPA diagnostic with `GO_MLX_ENABLE_FIXED_WIDE_SDPA_ATTENTION=1` on the promoted `32768` context shortcut records `36.764483458s` wall time and `62.147525173976284 tok/s`, slightly below the accepted default. Forcing the native wide matmul fallback with `GO_MLX_ENABLE_FIXED_WIDE_MATMUL_ATTENTION=1` regresses to `46.590511585s`, `23.67497555194655 tok/s`, and `21548513532` peak bytes. Both complete with empty stderr, but neither is the full-attention/KV slot fix; future `driver-profile` reports now include these env-only wide gates in `runtime_gates` when set |
| Rejected long-context row cache-update diagnostic | a llama.cpp-inspired fixed-cache write path now exists behind `GO_MLX_ENABLE_FIXED_ROW_CACHE_UPDATE=1` and reports the gate in `driver-profile` snapshots. Paired with `GO_MLX_ENABLE_FIXED_WIDE_SDPA_ATTENTION=1` on the promoted `32768` context shortcut, it records `36.570614625s`, `62.0477494292309 tok/s`, `1101.1801978656852 tok/s` cold prefill, `20.323458ms` average restore, `19884219328` peak bytes, and `3657.0614625 J` at `100 W`. The slight wall-clock movement comes with worse decode and higher memory than the accepted default, so it stays diagnostic |
| Initial 100k context ramp harness and first ladder | `driver-profile` now supports `-prompt-repeat N`, so the README-shaped long-context workload can grow without throwaway prompt files and each JSON records the repeat count. `scripts/gemma4_context_ramp.sh` now runs the accepted `-fast-gemma4-lane` over model-shaped repeat/context steps `1:4096`, `4:16384`, `8:32768`, `13:32768`, `24:131072`, and `46:131072`; it does not use the old 64Ki cache-family boundary as a ramp target. The first historical Metal-visible 128-token ladder recorded repeat `1`/`4096` at `88.69834535003041 tok/s` over `5.971431375s`, repeat `4`/`16384` at `74.33104068005494 tok/s` over `12.315293209s`, repeat `8`/`32768` at `69.48165669588239 tok/s` over `21.636779s`, repeat `13`/`32768` at `62.59204228638978 tok/s` over `36.263682833s`, and one rejected old-boundary repeat `24`/`65536` row at `50.656561535149365 tok/s` over `80.389911666s`, all with empty stderr. The first repeat `46`/`131072` attempt produced no successful runs because MLX could not load `sdpa_vector_2pass_1_float_512_256` from the local Metal library, so it is recorded as a kernel-coverage blocker rather than timing evidence. A later `5120` token-budget sustained-turn diagnostic at the accepted 100k shape completes cleanly and is recorded separately |
| Tracked E2B context ramp harness | `scripts/gemma4_context_ramp.sh` is now tracked and defaults to the current E2B q4 production snapshot plus `-report-file`, so replayed ramp rows write JSON through the runner instead of shell stdout redirection. The model can still be overridden with `GO_MLX_MODEL` and the artefact stem with `GO_MLX_MODEL_LABEL`; use `GO_MLX_RAMP_MAX_TOKENS=5120` when replaying the sustained-turn fairness lane |
| Current E2B 100k retained-state real-workload pass | The current guarded 100k E2B q4 pass supersedes the historical 128-token rows, the earlier `408.483s` retained row, the adaptive page-size row, and the borrowed-page row. It was launched from `/private/tmp` on the Metal path with active/RSS hard caps of `12 GiB`, process virtual memory recorded but not capped, `prompt_repeat=46`, `context=131072`, `prompt_tokens=101005`, `max_tokens=1024`, `10` retained-prefix runs, paged K/V cache mode, `1024`-token hyper-long pages, borrowed full page state, and retained materialised full K/V handles for shared full-attention layers. It records `10/10` success, `10240` generated tokens, `231.109s` wall time, `60.011 tok/s` average decode, `1678.322 tok/s` cold prefill, `0.368ms` average warm restore, `3.710 GiB` peak MLX active memory, `3.146 GiB` process peak RSS, and `683.451 GiB` process virtual reservation. At the normalised `100 W` estimate, the run costs `23110.937 J`, saves `541.636s` of prompt setup versus replayed prefill, and saves `54163.552 J` of prompt setup energy. This is `1.170x` faster on decode and `1.125x` faster by wall/energy than the borrowed-page row, but still not a production close because cached llama.cpp and `mlx_lm` remain faster. See `docs/runtime/2026-05-20-go-mlx-gemma4-e2b-4bit-current-100k-g1024-r10-shared-fullkv-energy100w.json` |
| E2B 100k sustained long-turn diagnostic | The accepted 100k retained workflow was rerun with `max_tokens=5120` to avoid another tiny-output smoke. The prompt naturally stops at `2489` generated and visible tokens per turn, so this is not a true forced `5k` row, but it is `2.43x` the accepted 1024-token output length and completes `10/10` retained turns under the same `12 GiB` active/RSS guards. It records `24890` visible tokens, `475.571s` wall time, `59.947 tok/s` average decode, `59.962 tok/s` warm decode, `1680.309 tok/s` cold prefill, `0.362ms` average warm restore, `3.726 GiB` peak MLX active memory, `3.152 GiB` process peak RSS, and `47557.087 J` at `100 W`. This bounds long-output allocator growth on the current shared-full-K/V path; the remaining gap is still baseline 100k attention cost versus cached llama.cpp and `mlx_lm`. A future full `5k+` row needs a prompt shape that naturally demands that much output. See `docs/runtime/2026-05-20-go-mlx-gemma4-e2b-4bit-current-100k-g5120-budget-r10-shared-fullkv-energy100w.json` |
| E2B 100k token-phase trace | The refreshed promoted fp16 paged-K/V `100k`/`1024` token-phase probe holds the `76 tok/s` band at `75.8589865749723 tok/s`; Go-side forward graph construction is only `1.181ms/token`, while lazy MLX work lands in `sample_eval` at `11.967ms/token`. The paired `GO_MLX_TRACE_FORWARD_EVAL=1` native-event run is diagnostic only because forced materialisation slows decode to `22.54113728696051 tok/s`, but it isolates the live bucket: out of `45.428s` traced decode-loop time, `44.710s` is forward materialisation. Native event totals rank attention first at `15.537s`, then output `10.387s`, FFN `9.658s`, and attention residual `7.416s`. fp16 K/V moved later full-attention layers `19`, `24`, `29`, and `34` down to about `0.625ms/token`; early owner layers `4`, `9`, and `14` are down from the old `1.96-1.98ms/token` band to about `1.38ms/token` but still dominate. This keeps the next implementation target on owner-layer full-attention K/V work in the paged/global path. See `docs/runtime/2026-05-20-go-mlx-gemma4-e2b-4bit-100k-token-phase-trace-summary.md` |
| Rejected E2B 100k materialised-owner and O-projection diagnostics | `GO_MLX_ENABLE_PAGED_FULL_KV_MATERIALIZE=1` keeps a full backing tensor for the early full-attention owner layers so later tokens can append with `slice_update` instead of rebuilding from pages. On the old shared-full-K/V one-run `100k`/`1024` traced lane it records `77.200s` wall time, `59.855 tok/s` decode, `1682.696 tok/s` prefill, `1.249ms/token` Go-side forward graph construction, `15.435ms/token` sample/eval, `4.385 GiB` active MLX memory, and `3.137 GiB` process RSS. Rechecking the same branch after the fp16 K/V promotion records `67.049s` wall, `75.56536931370188 tok/s` decode, `1891.664 tok/s` prefill, and raises active MLX memory to `3.875 GB` versus `3.472 GB` for the promoted trace row, so the gate remains opt-in diagnostic only and is not part of `-fast-gemma4-lane`. The existing `-native-gemma4-attention-o-matvec` path was also rechecked on the promoted 100k lane and records `75.78008273592174 tok/s`, flat against the normal `75.8589865749723 tok/s` row, so it also stays diagnostic. See `docs/runtime/2026-05-20-go-mlx-gemma4-e2b-4bit-100k-materialized-owner-g1024-r1-energy100w.json` and `docs/runtime/2026-05-20-go-mlx-gemma4-e2b-4bit-100k-token-phase-trace-summary.md` |
| Rejected E2B 100k paged-attention branch probes | One-run `100k`/`1024` probes now bound the obvious alternatives to the accepted paged fast-concat lane. Omitting `GO_MLX_ENABLE_PAGED_DECODE_FAST_CONCAT` while keeping the other accepted hyper-long fast gates records `100937` prompt tokens, `106.324s` wall time, `22.956 tok/s` decode, `1638.525 tok/s` prefill, and `3.640 GiB` active MLX memory, so page-by-page Go/MLX attention is much worse. The `GO_MLX_ENABLE_NATIVE_PAGED_ATTENTION` diagnostic moves the same page-reduction graph behind one C++ call and improves only to `104.572s`, `23.448 tok/s` decode, and `1660.523 tok/s` prefill, rejecting CGO loop overhead as the main loss. A C++23 no-repeat correction for single-KV-head pages is correct and retained, but its 100k probe still records only `103.696s`, `23.828 tok/s` decode, and `1665.263 tok/s` prefill, so page-reduction graph shape remains rejected. Turning fixed Gemma 4 cache back on with the shared fixed mask and sliding-layer bound fails the guarded run after `13` visible tokens because active memory reaches `13748980782` bytes over the `12 GiB` guard; forcing `GO_MLX_FIXED_GEMMA4_CACHE_SIZE=102400` still fails after `13` visible tokens at `13682988726` active bytes, so right-sizing below the full context is not enough. The borrowed fixed-state native-handle correction removes full-cache handle clones from opt-in fixed paths, but the same guarded 100k shape still fails after `13` visible tokens at `13660804802` active bytes. These reject "turn off concat", "wrap the existing page graph in C++", and "restore fixed cache" as the 100k production path; the remaining target is a fused native paged/global-attention kernel that avoids concat without full fixed-cache residency. See `docs/runtime/2026-05-20-go-mlx-gemma4-e2b-4bit-100k-no-fastconcat-g1024-r1-energy100w.json`, `docs/runtime/2026-05-20-go-mlx-gemma4-e2b-4bit-100k-native-paged-attention-g1024-r1-energy100w.json`, `docs/runtime/2026-05-21-go-mlx-gemma4-e2b-4bit-100k-native-paged-no-singlekv-repeat-g1024-r1-energy100w.json`, `docs/runtime/2026-05-20-go-mlx-gemma4-e2b-4bit-100k-fixed-sliding-g1024-r1-energy100w.json`, `docs/runtime/2026-05-20-go-mlx-gemma4-e2b-4bit-100k-fixed-sliding-rightsized102400-g1024-r1-energy100w.json`, `docs/runtime/2026-05-21-go-mlx-gemma4-e2b-4bit-100k-fixed-borrowed-g1024-r1-energy100w.json`, and `docs/runtime/2026-05-20-long-context-gap-diagnosis.md` |
| Rejected E2B 100k paged-cache geometry probes | Two further same-shape one-run probes reject simple page-geometry tuning as the long-context fix. Forcing `GO_MLX_PAGED_KV_PAGE_SIZE=2048` on the accepted 100k/1024-token lane records `80.787s` wall time, `49.984 tok/s` decode, `1678.261 tok/s` prefill, `3.710 GiB` active MLX memory, and higher cache memory than the accepted `1024`-page row. Keeping `1024` pages but enabling `GO_MLX_ENABLE_PAGED_KV_PREALLOC=1` records `80.459s` wall time, `50.743 tok/s` decode, `1679.677 tok/s` prefill, and `3.747 GiB` active MLX memory, still below the accepted first-run `51.148 tok/s` and warm `51.310 tok/s` band. The next target remains a fused/global attention storage path, not larger pages or preallocated page writes. See `docs/runtime/2026-05-20-go-mlx-gemma4-e2b-4bit-100k-page2048-g1024-r1-energy100w.json`, `docs/runtime/2026-05-20-go-mlx-gemma4-e2b-4bit-100k-paged-prealloc-g1024-r1-energy100w.json`, and `docs/runtime/2026-05-20-long-context-gap-diagnosis.md` |
| Historical rejected fixed-to-paged threshold probe | A controlled 1024-token generation probe at the same `63625` prompt tokens showed the old artificial cliff: `context=65536` kept the fixed lane and recorded `46.976s` wall, `1985.425 tok/s` prefill, `68.909 tok/s` decode, `7.175 GB` peak MLX, and `3.374 GB` RSS. Raising the cap by one token to `context=65537` forced the paged fast-concat lane and recorded `51.053s` wall, `1970.214 tok/s` prefill, `54.847 tok/s` decode, `7.023 GB` peak MLX, and `3.397 GB` RSS. The one-token cap change cost about `20.4%` raw decode, so this branch is now treated as evidence against context-length cutoffs rather than as current production behaviour. See `docs/runtime/2026-05-21-go-mlx-gemma4-e2b-4bit-threshold-c65536-r29-g1024-fixed-energy100w.json`, `docs/runtime/2026-05-21-go-mlx-gemma4-e2b-4bit-threshold-c65537-r29-g1024-paged-fastconcat-energy100w.json`, and `docs/runtime/2026-05-20-long-context-gap-diagnosis.md` |
| E2B zero-copy paged restore / generation clear-cache probes | `GO_MLX_ENABLE_ZERO_COPY_PAGED_RESTORE=1` now keeps restored KV block pages as incoming pages instead of coalescing them during prompt-cache restore, giving the first guarded link between the pinned raw-byte bridge and the paged `.mp4` state path. `GO_MLX_ENABLE_GENERATION_CLEAR_CACHE=1` plus `GO_MLX_GENERATION_CLEAR_CACHE_INTERVAL=256` clears MLX allocator cache after prefill chunks and during long generation. On the `65537` paged threshold row it records `52.127s` wall, `55.233 tok/s` decode, and `4` bytes cache memory; on the `128Ki` row it records `80.551s` wall, `1593.668 tok/s` prefill, `59.919 tok/s` decode, `7.151 GB` peak MLX, `3.368 GB` RSS, and `4` bytes cache memory. This is valuable memory hygiene and streaming-restore plumbing, but it does not close the external runner decode gap. See `docs/runtime/2026-05-21-go-mlx-gemma4-e2b-4bit-threshold-c65537-r29-g1024-paged-fastconcat-clearcache-energy100w.json`, `docs/runtime/2026-05-21-go-mlx-gemma4-e2b-4bit-128ki-r46-g1024-paged-fastconcat-clearcache-energy100w.json`, and `docs/runtime/2026-05-20-long-context-gap-diagnosis.md` |
| Promoted retained fp16 K/V storage | `GO_MLX_KV_CACHE_DTYPE=fp16` is now part of the retained `-fast-gemma4-lane` long-context defaults without using the old fixed-to-paged boundary. The code casts stored fixed and paged K/V pages to the requested storage dtype, preserves that storage dtype through prompt-cache/session restore, and aligns the attention query dtype for fp16/bf16 K/V before SDPA. Without query alignment the old threshold row regressed to about `46.7 tok/s`, and before restore preserved the storage dtype the 100k retained fp16 row regressed to `240.453s` / `56.025 tok/s` with warm turns around `53.8 tok/s`; both variants are rejected. With restore-typed storage fixed, the accepted 100k/1024x10 row records `10/10` success, `188.417s` wall, `76.018 tok/s` average decode, warm turns around `76 tok/s`, `1888.005 tok/s` cold prefill, `0.384ms` average restore, `5.471 GB` peak MLX, `3.451 GB` active MLX, `3.382 GB` RSS, and `18841.703 J` at `100 W`. This beats the previous go-mlx shared-full-K/V row (`231.109s`, `60.011 tok/s`, `7.151 GB` peak) and the llama.cpp cached server wall/energy row (`214.205s`) while still trailing the configured `mlx_lm` cached anchor (`119.866s`, `103.971 tok/s`). See `docs/runtime/2026-05-21-go-mlx-gemma4-e2b-4bit-100k-r46-g1024-paged-fp16kv-restoretyped-clearcache-r10-energy100w.json`, `docs/runtime/2026-05-21-go-mlx-gemma4-e2b-4bit-100k-r46-g1024-paged-fp16kv-restoretyped-clearcache-r3-energy100w.json`, `docs/runtime/2026-05-21-go-mlx-gemma4-e2b-4bit-threshold-c65537-r29-g1024-paged-fp16kv-qalign-clearcache-energy100w.json`, and `docs/runtime/2026-05-21-go-mlx-gemma4-e2b-4bit-100k-r46-g1024-paged-fp16kv-qalign-clearcache-r10-energy100w.json` |
| Current E2B 100k llama.cpp cold anchor | The local llama.cpp Q4_K_M comparator was run from `/private/tmp` against `unsloth/gemma-4-E2B-it-GGUF` with `llama-bench -pg 101005,1024 -r 1 -ngl 99 -fa 1`. It records `94.904s` for cold `pp101005+tg1024` at `1075.081 tok/s` combined throughput on `BLAS,MTL` with `MTL0 (Apple M3 Ultra)` visible in stderr. This is slower than go-mlx's current shared-full-K/V cold first retained-profile turn by wall time, and it is not a cached-prefix runner verdict; repeated cold replay would be roughly `949.035s` over ten turns versus go-mlx's measured `231.109s` retained-prefix wall time. The server cached-prefix row below supersedes this cold row for runner-anchor evidence. See `docs/runtime/2026-05-20-llamacpp-gemma4-e2b-q4-k-m-pg101005-1024-bench.json` |
| Current E2B 100k llama.cpp cached server anchor | The local llama.cpp server comparator now covers the same retained-prefix class rather than cold replay only. It uses `llama-server` build `b8990-660b1b4bd`, `unsloth/gemma-4-E2B-it-GGUF` `Q4_K_M`, `context=131072`, prompt bytes `325754`, llama.cpp-reported prompt tokens `100926`, `10` repeated requests, and `1024` generated tokens per request with `ignore_eos=true`. It records `10/10` success, `10240` generated tokens, `214.205s` total wall time, `82.680 tok/s` decode from llama.cpp timings, `1132.450 tok/s` first prefill, `45.591ms` average warm prompt work with `100921` cached prompt tokens, `4.435 GiB` peak RSS, `427.173 GiB` peak VSZ, and `21420.531 J` at `100 W`. This closes the same-shape llama.cpp runner-anchor gap, but it exposes a production blocker: llama.cpp is still `1.079x` faster than the current go-mlx row by wall/energy and `1.378x` faster by decode on this retained workflow. See `docs/runtime/2026-05-20-llamacpp-gemma4-e2b-100k-cached-server.md` and `docs/runtime/2026-05-20-llamacpp-gemma4-e2b-q4-k-m-100k-cached-server-r10-g1024-energy100w.json` |
| Current E2B 100k `mlx_lm` cached anchor | The configured `/private/tmp/go-mlx-mlx-lm-venv` runner uses `mlx_lm 0.31.3` and `mlx 0.31.2`. The stock strict CLI load still fails on unused Gemma 4 shared-K/V extra tensors, so the measured in-process harness uses MLX-LM `load_model(strict=false)` and records that override in JSON. On the same local `mlx-community/gemma-4-e2b-it-4bit` snapshot, README repeat `46`, the same agentic suffix, `100935` cache prompt tokens, `5` cached suffix tokens, `1024` max tokens, and `10` runs, it records `119.866s` wall time including load and 100k prefill, `103.971 tok/s` average decode, `5465.549 tok/s` prefill, `5.473 GB` MLX peak memory, `3.820 GB` peak RSS, and `11986.551 J` at the normalised `100 W` estimate. Compared with the current shared-full-K/V go-mlx retained row, `mlx_lm` is `1.928x` faster by wall time and energy, `1.733x` faster on decode, and `3.257x` faster on one-time 100k prefill. This remains the current optimisation boundary. See `docs/runtime/2026-05-20-mlx-lm-gemma4-e2b-4bit-100k-cached-workflow-r46-g1024-r10-energy100w.json` and `docs/runtime/2026-05-20-mlx-lm-gemma4-e2b-4bit-100k-strict-load-failure.stderr` |
| Rejected E2B 100k cache-only chunk prefill diagnostic | A go-mlx diagnostic now exists behind `GO_MLX_ENABLE_CACHE_ONLY_CHUNK_PREFILL=1` that evaluates cache state only for intermediate prefill chunks and delays logits materialisation until the final chunk, matching the broad MLX-LM prefill shape more closely. On the same 100k/1024x10 workload it improves cold prefill from `157.168s` / `642.657 tok/s` to `116.210s` / `869.159 tok/s`, but the run fails `10/10` on the repeated-sentence quality guard and decode remains around `43.8 tok/s`. The summed failed diagnostic wall time is `365.468s`, still far behind the `mlx_lm` cached row, so this path is gated off by default and remains R&D evidence only. See `docs/runtime/2026-05-20-go-mlx-gemma4-e2b-4bit-cacheonly-prefill-r46-ctx131072-g1024-r10-energy100w.json` |
| Rejected E2B model-native fp16/rotating 128Ki diagnostic | The local `mlx-community/gemma-4-e2b-it-4bit` config declares `text_config.max_position_embeddings=131072`, i.e. the model's `128Ki` cap, so the 100k prompt diagnostics are under the model limit. The model-native `fp16`/rotating cache path is safe at `28548` prompt tokens (`4.702 GB` active MLX) and `52677` prompt tokens (`6.199 GB` active MLX), including when the context ceiling is set to `131072`. It then fails the `12 GiB` active guard around the `80k` prompt-token shape at `28808918294` active bytes, and fails the 100k shape at `64794744442` active bytes. Smaller `256`-token prefill chunks worsen the 80k failure to `51768088226` active bytes; rotating cache copy-detach and full-attention layer eval-boundary diagnostics were flat and removed from source. This rejects model-native `fp16`/rotating as the 100k production shortcut; the viable target remains a fused paged/global-attention or zero-copy state layout. See `docs/runtime/2026-05-20-long-context-gap-diagnosis.md` |
| Current E2B 100k vLLM Metal attempt | The configured vLLM Metal runner (`vllm 0.20.0+cpu` with the Metal plugin active) was launched from `/private/tmp` with `vllm bench latency --max-model-len 131072 --input-len 100935 --output-len 1024 --batch-size 1 --num-iters 1 --num-iters-warmup 0`. It reaches `MLX device set to: Device(gpu, 0)` and enables chunked prefill at `16384`, then fails during MLX-LM strict model load on the same Gemma 4 shared-K/V extra parameter class. No latency JSON is written, so this remains a documented compatibility failure rather than a throughput datapoint. See `docs/runtime/2026-05-20-vllm-metal-gemma4-e2b-4bit-100k-latency-p100935-g1024.stdout` and `docs/runtime/2026-05-20-vllm-metal-gemma4-e2b-4bit-100k-latency-p100935-g1024.stderr` |
| Current E2B 100k retained 10-chapter book pass | `chapter-profile` now renders the Gemma 4 chat template directly for retained sessions, strips thinking before appending assistant history, records natural model stops, and rejects max-token exhaustion before a chapter marker. The current E2B q4 100k book run uses `context=131072`, `prompt_repeat=46`, `chapters=10`, `chapter_max_tokens=8192`, `chapter_min_tokens=768`, thinking enabled, `temperature=1.0`, `top_p=0.95`, and `top_k=64`. It records `10/10` successful turns, `11425` generated/visible tokens, chapter visible lengths from `979` to `1484`, `482.081s` wall time, `41.442 tok/s` average decode, `578.182 tok/s` average prefill, `4.261 GiB` peak MLX active memory, `5.771 GiB` peak process RSS, `6.546 GiB` process peak RSS, `953.339 GiB` process virtual reservation, and `48208.084 J` at the normalised `100 W` estimate, with empty stderr. The stricter `chapter_min_tokens=1024` probe is debug-only: chapter 2 improved from `803` to `936` visible tokens after the paragraph prompt fix but still naturally stopped below that artificial threshold. See `docs/runtime/2026-05-20-gemma4-e2b-current-100k-realwork.md` and the captured markdown at `docs/runtime/2026-05-20-go-mlx-gemma4-e2b-4bit-current-realbook-ctx131072-c10-g8192-min768-naturalstop-thinking-book.md` |
| Benchmark safety correction | The later 10-chapter full-book attempt invalidated the assumption that short retained-story smokes and post-run metrics were enough. E2B fresh-history runs degenerated into repeated tokens, and one run was killed by the OS before writing a complete report. `chapter-profile` now records `safety_limits`, derives default resident limits from the resolved memory plan plus a `30%` active-memory headroom for live-eval allocator transients, checks memory after load, during token streaming, after prefill, and after each turn, rejects max-token-truncated chapters before they can become accepted story context, cancels repeated sampled suppressed-token loops from the probe callback, rejects empty visible Gemma 4 turns, repeated visible lines/sentences, fragmented visible output, and meta-planning/outline output, exposes JSON-visible `repeat_penalty`, captures profile panics as JSON errors, and carries process virtual/resident peaks in the summary. Visible-token floors are debug guards only, not content-quality proof. `driver-profile` now has the same JSON-visible active/RSS memory guards, live stream memory checks, repeated sampled-token cancellation, sampled-token evidence, quality guards, panic capture, and failed-run memory retention; process virtual memory is recorded by default and enforced only when explicitly capped because absolute MLX virtual address-space reservation produced false failures on the paged 100k lane. The sampler now suppresses banned tokens before top-p/top-k so dominant special tokens cannot collapse sampling back to token `0`. See `docs/runtime/2026-05-20-chapter-profile-safety.md`. The raw compact 10-heading book at `docs/runtime/2026-05-20-go-mlx-gemma4-26b-a4b-q4-raw-unaccepted-c10-g128-rp105-book.md` remains explicitly not accepted benchmark evidence; the current accepted E2B 100k book evidence is recorded separately in `docs/runtime/2026-05-20-gemma4-e2b-current-100k-realwork.md` |
| Current C006 report-file full-book artifact | `chapter-profile` now accepts `-report-file` so long-form JSON evidence can be written directly by the runner instead of depending on shell redirection. The current C006 poetry/mathematics book run uses `mlx-community/gemma-4-e2b-it-4bit`, `context=131072`, `chapters=10`, `chapter_max_tokens=8192`, `chapter_min_tokens=512`, thinking enabled, `temperature=1.0`, `top_p=0.95`, `top_k=64`, `cache_mode=paged`, and a normalised `100 W` power estimate. It records `10/10` successful turns, `8201` generated/visible tokens, chapter visible lengths from `668` to `1351`, `105.947s` wall time, `80.343 tok/s` average decode, `2676.126 tok/s` average prefill, `3.396 GB` active MLX memory, `3.611 GB` process RSS, `638.946 GB` process virtual reservation, and `10594.699 J` estimated energy. Operator review accepted the prompt/template path because the final chapter ended with the requested silence and stayed on point, so this is the accepted default small-model continuation lane. The stricter report-file neighbour with `chapter_min_tokens=640` failed only because chapter 8 naturally stopped at `563` visible tokens; no OOM, repeated-token, or max-token-truncation failure occurred. See `docs/runtime/2026-05-20-gemma4-e2b-c006-report-file-book.md`, `docs/runtime/2026-05-20-go-mlx-gemma4-e2b-4bit-c006-book-ctx131072-c10-g8192-min512-thinking-current-energy100w.json`, and `docs/runtime/2026-05-20-go-mlx-gemma4-e2b-4bit-c006-book-ctx131072-c10-g8192-min512-thinking-current-book.md` |
| Archived production benchmark index | The old `docs/runtime/2026-05-20-production-benchmark-index.md` replay map is no longer present in the checked-in runtime docs. Treat the surrounding GOAL/TODO summaries and the referenced `/private/tmp/go-mlx-goal/reports` paths as historical handover notes only until a fresh accepted benchmark index is regenerated after the code stabilises. This does not close production: the remaining long-context runner gap and runtime-fragment cleanup stay open work |
| Current E2B seven-format go-mlx matrix refresh | `docs/runtime/2026-05-20-gemma4-e2b-quant-matrix.md` reruns all seven local `mlx-community` E2B formats with `driver-profile -report-file`, `README.md` through the Gemma 4 chat template, `2205` prompt tokens, `context=32768`, paged cache, `prefill_chunk_size=512`, `3x128` generated tokens, hidden output, and `100 W` normalised energy. The raw go-mlx side is now replay-grade: `4bit` records `107.914 tok/s`, `5bit` `76.489`, `6bit` `73.411`, `8bit` `78.326`, `bf16` `27.703`, `mxfp4` `84.282`, and `mxfp8` `74.631`. MXFP4 initially crashed in the host suppressed-token fallback; `Array.Floats()` now materialises lazy float32 arrays before `mlx_array_data_float32`, and the rerun completes. External rows are recorded separately |
| Current E2B seven-format external runner rows | `docs/runtime/2026-05-20-gemma4-e2b-external-quant-rows.md` refreshes the runner-anchor side of the short E2B matrix. `mlx_lm.generate` `0.31.3` on `mlx 0.31.2` fails all seven strict loads with extra shared-K/V tensor counts `100` for MXFP, `140` for affine quant, and `60` for BF16. vLLM Metal `0.20.0+cpu` with `vllm_metal 0.2.0` reaches `MLX device set to: Device(gpu, 0)`, fails quantised rows with `40`/`80` extra-tensor counts, and loads BF16 at `3.571706959s` for `2205+128`. llama.cpp build `660b1b4bd` records comparable GGUF anchors: `Q4_K_M` at `4294.342 tok/s` prefill / `143.952 tok/s` decode and `Q8_0` at `4460.410 tok/s` prefill / `122.513 tok/s` decode |
| mlx-community Gemma 4 E2B vs 26B q4 fast iteration | Both native MLX q4 snapshots are cached from `mlx-community`: `gemma-4-e2b-it-4bit` and `gemma-4-26b-a4b-it-4bit`. On the same current-binary `driver-profile -fast-gemma4-lane` README profile (`2204` prompt tokens, `128` generation tokens, three runs, hidden output, `100 W` normalised energy), E2B records `122.23205359983257 tok/s` decode, `4.532718042s` wall, `453.2718042 J`, and `4.523123664781451 GiB` peak memory. The matched 26B run records `88.18156398367199 tok/s` decode, `6.027796249s` wall, `602.7796249 J`, and `17.314671628177166 GiB` peak memory. E2B is `1.3861x` faster on raw decode and uses `0.7519x` the wall time and energy for this short iteration profile |
| mlx-community Gemma 4 E2B retained-story iteration | The same `chapter-profile` story harness on `mlx-community/gemma-4-e2b-it-4bit` completes two thinking-enabled retained turns at `context=65536` with empty stderr. It records `1767` generated tokens, `1087` visible tokens, `16.935350541s` total, `110.35789603546327 tok/s` average decode, `965.9831974768388 tok/s` average prefill, `1693.5350541 J`, and `4.489579644054174 GiB` peak memory. Against the 26B retained-story smoke above, E2B is `1.4932x` faster on average decode and uses `0.2942x` the wall time and energy while producing a comparable visible chapter artifact at `docs/runtime/2026-05-19-go-mlx-gemma4-e2b-q4-fresh-story-thinking-ctx65536-c2-g8192-book.md` |
| Q4-first goal bench policy | Goal benchmarks should use q4 as the primary production lane for E2B, E4B, 26B MoE, and the 31B dense-family scale-up, with BF16 kept as the quality/reference comparator rather than the throughput target. For E2B/E4B, `>100 tok/s` decode is an acceptable target when paired with q4 memory/energy savings; maintaining that band as context grows is the stronger acceptance signal. The 26B A4B MoE q4 lane remains usable in the restored `88 tok/s` band, but future optimisation should first protect the q4 small dense-family path and then compare BF16 for quality/regression checks |
| E2B q4 vs BF16 long-context 8k-return bench | A q4-first long-return profile now uses the opencode-sized README repeat shape plus a synthetic agentic operations suffix: `prompt_repeat=13`, `context=65536`, `prompt_tokens=28587`, `max_tokens=8192`, and one completed `8192` token generation. The cached `mlx-community/gemma-4-e2b-it-4bit` run records `94.92547697253806 tok/s` decode, `1396.6243790432902 tok/s` prefill, `111.006821417s` wall time, `11100.6821417 J`, and `5.134385833516717 GiB` peak memory. The cached `mlx-community/gemma-4-E2B-it-bf16` comparator records `26.59615320070758 tok/s` decode, `1304.3044170967798 tok/s` prefill, `334.4575525s` wall time, `33445.75525 J`, and `12.643188176676631 GiB` peak memory. Q4 is `3.569x` faster on decode, `3.013x` lower wall/energy, and uses `0.406x` the peak memory, even though the 29k-context/8k-return q4 decode rate lands slightly below the round `100 tok/s` line |
| E2B all-quant matrix plus 4bit/8bit runner anchors | `docs/runtime/2026-05-19-gemma4-e2b-quant-matrix.md` lists `mxfp4`, `mxfp8`, `4bit`, `5bit`, `6bit`, `8bit`, and `bf16` on the same README-shaped profile. go-mlx records `123.34573087131434 tok/s` for MLX 4bit and `101.26776527534014 tok/s` for MLX 8bit. The llama.cpp anchors use comparable GGUF formats only: `Q4_K_M` records `139.914221 tok/s`, and `Q8_0` records `122.098723 tok/s`. The same matrix records `mlx-lm 0.31.3` / `mlx 0.31.2` and vLLM Metal as E2B compatibility gaps because both reject the snapshots at load with extra attention K/V parameters |
| E4B MXFP8 native QMM support | `mlx-c` is bumped to `v0.6.0`, local patched MLX is aligned to `v0.31.1`, and CMake now forces `mlx-c` to build against the local `lib/mlx` submodule so the patched 512-wide SDPA resource and native MXFP8 QMM kernels ship together. The E4B MXFP8 native-QMM three-run README profile records `69.23950679870225 tok/s` decode, `821584.7669364832 tok/s` prefill, `7.22419575s` wall, `722.419575 J`, and about `9.21 GiB` peak memory. The old dense fallback records `14.800582374835564 tok/s`, `27.691197209s`, and about `20.31 GiB`; the q4 E4B row records `86.09288563808235 tok/s`, `6.115125667s`, and about `5.97 GiB` |
| Small-model first target posture | New E2B and E4B builds are the next optimisation targets before further 26B work. The E-range models are the fast small dense-family iteration targets, with 31B as the larger member of the same effective architecture family. The 26B A4B MoE q4 lane is considered passable in the restored `88 tok/s` band for quality-focused use, while the larger dense-family lane remains blocked on scale/runtime compatibility until the GELU/native-array failure seen in the `lthn/lemer-mlx` smoke is cleared |
| `lthn/lemer-mlx` retained-story smoke | the cached `lthn/lemer-mlx` chat template matches the Gemma 4 thinking system-turn shape. The earlier native runtime panic is fixed far enough to reach generation: the loader now validates K/V state and infers affine q4 group/bits from U32 packed weight/scale shapes when the pack has no quantization block. A one-turn no-fast smoke completes at roughly `2008 tok/s` prefill, `78 tok/s` decode, `3.76 GB` active MLX memory, and `4.17 GB` resident memory. The corrected full-book harness is still not accepted: fast thinking with `chapter_max_tokens=2048` accepts chapter 1, then rejects chapter 2 for stopping before `[[END_CHAPTER]]`; no-thinking still emits visible planning in chapter 1. This is now a prompt/model-quality blocker, not a native crash or OOM blocker |
| Current fast-lane token-phase profile | `driver-profile -fast-gemma4-lane -trace-token-phases` records `84.32951687301572 tok/s` on the 26B README prompt, with steady non-final tokens averaging about `10.406612ms` in `Eval(next)`, `1.461166ms` in forward graph construction, and `11.915181ms` total. This keeps the next native target in evaluated graph/kernel work, not driver overhead |
| Current driver-profile summary schema smoke | the refreshed fast-lane README smoke profile records summary prompt-token stats directly: `prompt_tokens_average=2204`, `prompt_tokens_min=2204`, and `prompt_tokens_max=2204`, alongside decode, wall-clock, memory, restore, and energy fields, with empty stderr. This keeps the report aligned with the acceptance requirement to name prompt length at the top level |
| Current fast-lane native-event summary smoke | `GO_MLX_TRACE_FORWARD_EVAL=1` is diagnostic, but the refreshed report now emits duration-ranked `summary.native_events` bucket totals without external jq. The largest current buckets are attention (`100.062542ms` over `210` events), local MLP (`54.313699ms`), router (`54.281834ms`), split expert activation (`50.886424ms`), and attention residual (`45.670918ms`). This confirms the remaining raw-decode work is evaluated attention/FFN graph time, not prompt handling or driver bookkeeping |
| Rejected fixed-owner attention native-event smoke | re-enabling `-native-gemma4-fixed-owner-attention` under the same traced fast-lane shortcut lowers diagnostic decode to `14.50847005479256 tok/s` and leaves the ranked attention bucket effectively unchanged at `100.305117ms` over `210` events. This current-source trace confirms the existing broad fixed-owner attention wrapper is not the next attention fix |
| Bounded attention O-projection matvec probe | `-native-gemma4-attention-o-matvec` routes only Gemma 4 attention `OProj` through the existing q4/q8 single-token matvec kernel. Focused runtime-gate and CLI tests pass, and the path falls back for non-single-token shapes. It stays opt-in: the paired 3-run README control records `85.85272086042305 tok/s`, while the gated run records `84.68415619194967 tok/s`; the longer 10-run pass is only slightly positive at `84.04525365609535 tok/s` versus `83.59564887907933 tok/s` control, with warm decode `84.10303328183633 tok/s` versus `83.75771763124862 tok/s` and empty stderr. At the normalised `100 W` estimate, the 10-run gated path costs `1699.7798417 J` versus `1710.686 J` for control, but this is not a material parity fix and is not included in `-fast-gemma4-lane` |
| vLLM Metal 26B q4 README-shape calibration | local vLLM Metal `bench latency` can load the same MLX-community 26B A4B q4 snapshot. Batch size 1, input length `2204`, output length `128`, max model length `4096`, and BF16 reports `3.8800909579731524s` latency, slower than go-mlx cold same-prompt `2.668634083s` and warm retained `1.4592862175555557s` turns. Batch size 8 reports `15.160140624968335s`, useful as capacity evidence but not a single-request parity figure |
| Current native-event attribution trace | diagnostic-only `GO_MLX_TRACE_FORWARD_EVAL=1` on the runtime-gate cleanup lane slows decode to `13.93212949012604 tok/s`, but current traced materialisation time is led by attention `192.906671ms`, expert activation `112.32357699999996ms`, expert down `96.85933999999999ms`, local MLP `121.76254400000002ms`, router `113.1861289999999ms`, and the FFN branch norms/final norm/output cluster around `85-99ms` each over 15 non-final traced tokens |
| Rejected generic native linear matvec probe | `GO_MLX_ENABLE_NATIVE_LINEAR_MATVEC=1` routes generic q4/q8 single-token `Linear.Forward` through the custom dense matvec kernel, mainly touching attention projections in the active lane. Focused correctness and CLI gate tests pass, but the active README 3-run lane regresses to `83.01185809523686 tok/s` decode and `86.78823747504326 tok/s` warm decode with empty stderr, so the specialised router/local-MLP matvec wins do not generalise to all attention linears |
| Rejected native FFN residual combine probe | `GO_MLX_ENABLE_NATIVE_GEMMA4_FFN_RESIDUAL=1` fuses the MoE branch post-norms, branch add, final FFN RMSNorm, and residual add into one Metal kernel. Focused correctness and CLI gate tests pass, but the active README 3-run lane regresses to `83.43718600332822 tok/s` decode with empty stderr, so this confirms the remaining gap is not solved by collapsing those elementwise FFN graph nodes alone |
| Rejected native model-level greedy fixed-cache corrected probe | `GO_MLX_ENABLE_NATIVE_GEMMA4_MODEL_GREEDY=1` collapses the fixed-cache greedy decode layer loop into one C++ call that returns the next token plus updated owner K/V arrays. The earlier availability probe missed `-native-gemma4-moe-layer`, and the production 26B A4B pack has no per-layer input tensors, so the wrapper first needed a nil per-layer-input fix. The corrected trace now emits seven `gemma4.model.greedy_token` events over an 8-token run, proving the wrapper fires, but the full README 3-run lane regresses to `50.56636111604209 tok/s` decode with empty stderr. The broad one-call wrapper currently materialises too much native graph work and is rejected as a production path |
| Rejected per-layer sliding fixed-cache overflow lane | preserving the 1024-token sliding-layer fixed capacity required a shape-stable native overflow update and records `2033.3865559253882 tok/s` prefill but only `73.05984177869179 tok/s` decode; the active 128-token lane keeps uniform request-sized fixed caches |
| Restored uniform request-sized fixed-cache lane after sliding probe | after restoring uniform 2336-slot fixed caches, the same README 3-run lane records `1925.9978025157088 tok/s` prefill and `83.59574625080806 tok/s` decode; the earlier automatic run remains the best verified sample at `84.01009717307203 tok/s` |
| Prefill chunk-size sweep on current fixed-cache packed expert-ID lane | `driver-profile -prefill-chunk-size 4096` records `2101.369627343361 tok/s` prefill and `83.74497136862215 tok/s` decode on the README prompt; same-prompt llama.cpp `pp2204` is only `1.0038x` faster on prefill, while decode remains `1.0920x` faster |
| Default wide-prefill planner rerun | the 64GB-class memory plan now selects `prefill_chunk_size=4096`; the no-override README 3-run lane records `2088.289027094623 tok/s` prefill and `83.09590032942343 tok/s` decode, leaving same-prompt llama.cpp `1.0101x` faster on prefill and `1.1005x` faster on decode |
| Current packed-column token-phase profile | same lane, one run with `-trace-token-phases`, records `78.66136991155207 tok/s`; steady tokens average `12.7941ms`, with `11.4613ms` in `Eval(next)` and `1.3014ms` in next-forward graph construction |
| Current right-sized fixed-cache token-phase profile | same packed lane with `GO_MLX_FIXED_GEMMA4_CACHE_SIZE=2336`, one run with `-trace-token-phases`, records `83.73000373542442 tok/s`; steady tokens average `12.0209ms`, with `10.6246ms` in `Eval(next)` and `1.3577ms` in next-forward graph construction |
| Packed-column native-event attribution trace | diagnostic-only `GO_MLX_TRACE_FORWARD_EVAL=1` run slows throughput by forcing intermediate materialisation, but attributes traced native time across attention `17.52%`, local MLP `11.87%`, router `10.47%`, expert activation `10.25%`, attention residual `8.98%`, expert down `8.81%`, and several norm/output buckets |
| Rejected packed-column scale-hoist probe | hoisting scale/bias loads for aligned q4 groups was correct but slower on the 3-run lane at `77.70903294390506 tok/s`, so it was reverted while keeping packed-column q iteration |
| Rejected packed-column compiled-layer probe | enabling `-compiled-gemma4-layer` on top of the packed expert-ID lane records `78.78857639506562 tok/s` in a one-run token-phase profile, slightly below the packed baseline and still `1.1607x` behind same-prompt llama.cpp decode |
| Rejected packed-column compiled per-layer-input probe | enabling `GO_MLX_ENABLE_COMPILED_GEMMA4_PER_LAYER_INPUTS=1` on the packed expert-ID lane records `77.0865964024348 tok/s`, slower than the packed baseline and `1.1863x` behind same-prompt llama.cpp decode |
| Rejected packed-column native MLP probe | enabling `GO_MLX_ENABLE_NATIVE_MLP_GELU=1` on the packed expert-ID lane records `77.96201603724107 tok/s`, slower than the packed baseline and `1.1730x` behind same-prompt llama.cpp decode |
| Rejected dynamic paged cache control | removing the fixed-cache gate on the packed expert-ID lane records only `50.412141409798174 tok/s`; fixed-cache graph stability is still required |
| Rejected right-sized fixed-cache no-shared-mask control | keeping `GO_MLX_FIXED_GEMMA4_CACHE_SIZE=2336` but disabling the shared fixed mask records `79.62987660090852 tok/s`, so the shared mask stays on |
| llama.cpp PR 23211 Gemma 4 26B assistant MTP diagnostic | upstream master cannot load `gemma4_assistant`, but unmerged PR `ggml-org/llama.cpp#23211` runs the 26B Q4_K_M assistant path; tuned `--spec-draft-n-max 2` records `100.2 tok/s` CLI visible generation and server-side `93.76822253543413 tok/s` with `75/101` draft tokens accepted |
| go-mlx native Gemma 4 26B A4B assistant MTP first bench | native target+assistant loop now completes on the local 26B safetensors pair; `draftTokens=2` records target-only `61.42236924451142 tok/s`, MTP visible `32.207918216043666 tok/s`, and `8/24` draft tokens accepted; `draftTokens=1` records target-only `60.756648029450965 tok/s`, MTP visible `34.89669623707289 tok/s`, and `6/16` accepted, so the first native loop is correct enough to benchmark but not yet a speed win |
| Same-short-prompt llama.cpp MTP comparator | on `In a future city, the engineer opened the notebook and`, llama.cpp PR 23211 target-only server records `88.79861030174878 tok/s`, MTP `n_max=2` server records `100.62260235205333 tok/s` with `9/12` draft tokens accepted, and CLI records target-only `92.0 tok/s`, MTP `n_max=1` `103.2 tok/s`, MTP `n_max=2` `118.2 tok/s`; this rejects the current go-mlx MTP loop as the production path because go-mlx native MTP is slower than both go-mlx target-only and llama.cpp MTP |

Treat these as evidence that the next optimisation boundary must be larger than
individual activations. The earlier E2B lane isolated a major per-layer-input
cost, and the row-gather fix now gathers packed embedding rows and scale/bias
rows before dequantising, avoiding full vocabulary-table materialisation for
single-token decode. The active Gemma 4 26B A4B q4 snapshot has no
`per_layer_*` tensors, so its remaining parity miss is in the normal decode
stack: fixed-cache attention, local MLP, and routed expert activation/down
kernels. Router projection/top-k and dense local-MLP matvecs now have small
native wins, but are not enough alone. Direct grouped-query attention already
avoids explicit K/V head expansion on Gemma 4 fast SDPA paths. The E2B
short-context q4 floor by itself is not production acceptance; the accepted
production benchmark lane is now the opencode-sized retained workflow plus
runner anchors, folded 100k stress lifecycle, full-book continuation, bounded
long-context degradation handoff, and strict manifest coverage.

## Architecture Rules

- Prefer a stable package API over CLI-only behaviour. CLI commands are the
  diagnostic and bundle surface, not the core design.
- Keep CGO and native MLX code under `go/internal/metal`.
- Keep Qwen and Gemma model-specific shape decisions close to the native model
  loaders.
- Use structured profiling data before choosing an optimisation target.
- Store all repeatable benchmark results as JSON or markdown under
  `docs/runtime/` so future agents can compare against real numbers.
- Do not revert unrelated dirty worktree changes. Patch narrowly.
- Use UK English in new docs and comments.

## Workstream 1: Build and Packaging

**Purpose:** make `lthn-mlx` a reliable binary for the LTHN app, CLI, and server
bundle.

- [x] Keep `Taskfile.yml` targets for `build:lthn`, `build:violet`, and
  `build:bundle` working from the repository root.
- [x] Keep the direct build command working for environments without Task:

  ```bash
  cd /Users/snider/Code/core/go-mlx
  env GOCACHE=/private/tmp/codex-go-mlx-cache go build -trimpath -o bin/lthn-mlx ./go/cmd/mlx
  ```

- [x] Document any required `MLX_METALLIB_PATH` override beside the benchmark
  output when the bundled MLX metallib cannot be found automatically.
- [x] Use the repository workspace for local verification. Do not set
  `GOWORK=off` for this goal lane unless a separate release gate explicitly asks
  for standalone module resolution.

## Workstream 2: Benchmark and Runner Calibration

**Purpose:** prove the production runner lane against configured alternatives
without changing workload semantics. Use llama.cpp, `mlx_lm`, and vLLM as
calibration systems, then benchmark future optimisation rounds against the
current go-mlx best artefact unless an external runner demonstrates a realistic
agentic workflow win.

- [x] Keep `lthn-mlx driver-profile` producing machine-readable JSON with
	  effective load settings, restore, first-token, decode, tok/s, optional
	  estimated energy, optional prompt/chat chunking, and optional per-token native
	  phase timings. The report now exposes first-class per-run and summary restore
	  timings from prompt-cache restore metrics, summary prompt-token min/max/average,
	  preserves nested decode counters, optional token phase traces, summary
	  native-event bucket totals for diagnostic traces, and records the resolved
	  planner cache mode
	  instead of only the CLI flags, can include `-estimate-power-watts` joule
	  deltas for retained-state versus replayed-prefill setup, and can use
	  `-prompt-chunk-bytes N` to avoid tokenising one giant prompt string during
	  large-context diagnostics. It also accepts `-prompt-repeat N` so the same
	  prompt can be grown into 29k, 32k, and 100k-class diagnostic contexts while
	  keeping the repeat count in the JSON report. `-fast-gemma4-lane` applies
	  the current accepted Gemma 4 fast runtime gate set without enabling
	  rejected broad native wrappers, defaults larger-than-4096 contexts to the
	  proven `512` token prefill chunk plus `4096` byte prompt chunk shape unless
	  the operator overrides it, keeps fixed Gemma 4 K/V out of retained
	  production defaults, and does not derive cache-family or fixed-cache size
	  from a context-length cutoff.
- [x] Add or preserve a parity report under `docs/runtime/` for every meaningful
  optimisation round.
- [x] Use this go-mlx command shape for the target Gemma 4 E2B lane:

  ```bash
  env MLX_METALLIB_PATH=/Users/snider/Code/core/go-mlx/dist/lib/mlx.metallib /Users/snider/Code/core/go-mlx/bin/lthn-mlx driver-profile -json -include-output=false -context 4096 -prompt "Answer in one short sentence: why does retained model state matter?" -max-tokens 128 -runs 3 -trace-token-phases /Users/snider/.cache/huggingface/hub/models--mlx-community--gemma-4-e2b-it-4bit/snapshots/99d9a53ff828d365a8ecae538e45f80a08d612cd
  ```

  2026-05-16 rerun: command returned JSON with `successful_runs: 3`,
  `decode_tokens_per_sec_average: 44.55943393415422`, `visible_tokens: 48`,
  `peak_memory_bytes: 8579334138`, and per-token phase traces. See
  `docs/runtime/2026-05-16-gemma4-e2b-driver-profile.md`.

- [x] Re-admit configured Python/Metal runners as calibration evidence. Earlier
  broken `mlx_lm` attempts remain historical, but the repaired parity venv and
  local vLLM Metal install now provide useful external baselines. Future
  calibration reports should still keep prefill, decode, cache policy, and
  repeated-workflow wall-clock separate.
- [x] Keep a llama.cpp parity report with prefill and decode. The closest local
  26B A4B q4 comparison records the current go-mlx fused expert gate/up plus
  automatic long-prompt last-token prefill path at `56.220244342267904 tok/s`
  decode and `903.0290085147915 tok/s` long prefill. The latest same-prompt
  automatic fixed-cache path records `1935.3610403257746 tok/s` prefill and
  `84.01009717307203 tok/s` decode with split/BF16 expert-ID fused activation,
  packed-column expert kernels, request-sized fixed cache, shared fixed mask,
  direct greedy, and sorted prefill enabled. A 2026-05-18 chunk-size sweep first
  proved that `driver-profile -prefill-chunk-size 4096` records
  `2101.369627343361 tok/s` prefill and `83.74497136862215 tok/s` decode on
  the same README prompt. The 64GB-class memory plan now selects that width by
  default; the no-override rerun records `2088.289027094623 tok/s` prefill and
  `83.09590032942343 tok/s` decode. The latest 10-run retained-prefix guard
  rerun with the generic native MoE layer disabled records
  `425831.7097091192 tok/s` restored-prefix setup and
  `84.8683681726259 tok/s` decode. The trace-name formatting cleanup
  rerun records `427000.78466006636 tok/s` restored-prefix setup and
  `85.22730571622206 tok/s` decode. The native router matvec plus top-k probe
  records `425482.7192523824 tok/s` restored-prefix setup and
  `86.06590721922689 tok/s` decode. The latest native router plus dense MLP
  matvec retained-prefix probe records `423630.8407376839 tok/s` average prefix
  setup, `86.95798305515721 tok/s` decode, and `87.13332867474983 tok/s` warm
  decode. The runtime-gate hot-path cleanup keeps the same band at
  `423698.49297158385 tok/s` average prefix setup, `87.05458770800922 tok/s`
  decode, and `87.16243827560751 tok/s` warm decode. The fresh current-source
  10-step retained-state rerun records `87.15020057594002 tok/s` average raw
  decode, `87.995764012926 tok/s` warm raw decode, `9.49244888s` saved setup
  over ten turns, and `128.6485922304177` decode-equivalent effective visible
  tok/s. Same-prompt-length
  llama.cpp `Q4_K_M`
  records
  `2109.335561 tok/s` at `pp2204` and `91.451031 tok/s` long-context decode.
  Prefill is now within `1.0%` of llama.cpp on the default planner path; decode
  remains the active external parity miss.
- [x] Evaluate Gemma 4 MTP/speculative decode as a separate visible-throughput
  lane, not as raw prefill evidence. Google ships Gemma 4 `-assistant`
  drafter checkpoints for speculative decode, and llama.cpp exposes
  `--spec-draft-model` plus `--spec-type draft-mtp`. For the current 26B A4B
  lane, the matching pair is `google/gemma-4-26B-A4B-it` plus
  `google/gemma-4-26B-A4B-it-assistant`; the E4B assistant belongs with the
  E4B target. Acceptance requires target-only and speculative runs on the same
  prompt, draft tokens proposed/accepted/rejected, effective visible tok/s,
  target verify throughput, and a llama.cpp speculative comparator when a
  comparable GGUF drafter exists. 2026-05-18 progress: the Homebrew llama.cpp
  build is too old for `draft-mtp`, upstream master exposes `draft-mtp` but
  cannot load `gemma4_assistant`, and unmerged PR `ggml-org/llama.cpp#23211`
  successfully runs the local 26B Q4_K_M assistant GGUF. The best PR CLI
  sample is `100.2 tok/s` at `--spec-draft-n-max 2`; the matching server run
  reports `93.76822253543413 tok/s` with `75/101` drafted tokens accepted
  (`74.257%`). This validates MTP as a separate visible-throughput route. The
  go-mlx package now has a target+draft `GenerateSpeculative` reference API,
  `LoadSpeculativePair` loads target and assistant models with tokenizer
  compatibility probes, and the fast-eval bench adapter returns token IDs into
  the shared `go-inference/decode` speculative and prompt-lookup harness, so
  acceptance metrics no longer collapse to text-only zero-token reports. The
  `bench` command also accepts `-speculative-draft-model` and
  `-speculative-draft-tokens`, and emits accepted/rejected token counts plus
  visible/target/draft tok/s in JSON when the drafter is a standalone model.
  A real E2B target+assistant bench attempt reached the previous native loader
  boundary and failed cleanly with `gemma4_assistant native MTP drafter loading
  is not implemented yet`; `gemma4_assistant` is recognised as metadata-only
  instead of being misloaded as ordinary `gemma4_text`. Follow-up progress:
  `go/internal/metal.LoadGemma4Assistant` now loads and validates Gemma 4
  assistant drafter tensors separately from `InternalModel`, including pre/post
  projections, four Q/O-only assistant layers, MLP tensors, optional
  ordered-embedding centroids/token ordering, and projection shape checks.
  Focused verification passed with
  `go test ./internal/metal -run 'TestGemma4Assistant' -count=1` under
  `GOWORK=/Users/snider/Code/core/go-mlx/go.work`, and optional local-pack
  smokes passed against both the E2B assistant safetensors pack and the 26B A4B
  assistant safetensors pack via `GO_MLX_GEMMA4_ASSISTANT_MODEL`. Follow-up:
  `go/internal/metal.LoadGemma4AssistantPair` now loads and validates a target
  Gemma 4 text runtime beside its attached assistant drafter, checking the
  shared backbone hidden size, vocabulary, tokenizer probes, target K/V stream
  layer types, and compatible attention head dimensions. Focused tests pass on
  synthetic target+assistant fixtures. The root package `mlx.LoadSpeculativePair`
  now recognises `gemma4_assistant` draft packs and routes them through that
  native attachment path instead of trying to load the assistant as a standalone
  `InternalModel`; `SpeculativePair.Generate` now calls the native Gemma 4
  assistant generation loop when the target runtime implements it.
  Optional local-pack smokes pass for
  both the E2B target+assistant pair and the 26B A4B target+assistant pair via
  `GO_MLX_GEMMA4_TARGET_MODEL` plus `GO_MLX_GEMMA4_ASSISTANT_MODEL`. Follow-up:
  `Gemma4AssistantPair.DraftStep` now runs one executable MTP assistant step
  over the target model's populated K/V caches. `Gemma4Model` now exposes
  `ForwardLastTokenLogitsAndHidden` so the assistant can consume the real
  target-backbone hidden state from the same target forward pass, plus the last
  token, and return draft logits, a greedy draft token, and the projected
  backbone hidden for a chained MTP step. `Gemma4AssistantPair.DraftBlock`
  chains those steps into a CPU-visible draft token block for the future
  verifier. It fails closed for ordered-embedding logits until that centroid
  path is implemented. Focused synthetic tests pass, and an optional E2B
  real-pack draft-step smoke passes with
  `GO_MLX_GEMMA4_TARGET_MODEL` plus `GO_MLX_GEMMA4_ASSISTANT_MODEL`. Follow-up:
  `Gemma4AssistantPair.VerifyDraftBlock` now performs greedy target-side
  accept/reject over a cloned target cache, returning accepted/rejected draft
  tokens, the target replacement token, and the accepted-boundary cache/logits
  state without polluting the live cache on rejection. Focused tests cover
  accepted and rejected draft blocks, source-cache preservation, and the E2B
  real-pack smoke now verifies one accepted target token. Follow-up:
  `Model.GenerateGemma4Assistant` wires the draft/verify primitives into a
  conservative greedy native MTP generation loop, and the root
  `SpeculativePair.Generate` path now reaches that loop for attached
  `gemma4_assistant` pairs. The MTP prefill path is hidden-aware: native MTP
  prompt-cache entries store the final target hidden state, while KV-only
  restored memory entries replay only the final suffix token needed to recover
  hidden instead of replaying the whole memory prefix. A real 26B target+
  assistant bench now completes, and it exposed the current next bottleneck:
  visible MTP decode is slower than target-only because acceptance is low and
  the assistant/verify loop adds more target calls than it saves. Same-prompt
  llama.cpp PR 23211 runs on the short prompt used for the go-mlx bench reject
  the current native MTP loop as the production path: llama.cpp target-only
  server records `88.79861030174878 tok/s`, llama.cpp MTP `n_max=2` server
  records `100.62260235205333 tok/s` with `9/12` draft tokens accepted, while
  go-mlx MTP is only `32.207918216043666 tok/s` with `8/24` accepted. Keep the
  code as an R&D lane, but return the production parity work to raw target
  decode. See `docs/runtime/2026-05-18-gemma4-mtp-speculative-decode.md`.

## Workstream 3: Native Decode Hot Path

**Purpose:** move enough repeated decode work into native MLX to cross the
100 tok/s floor.

- [x] Profile one-token decode with `-trace-token-phases` and identify the
  largest recurring bucket. The exact Gemma 4 E2B target command produced
  45 steady token-phase samples where `sample_eval_duration` averages
  `~20.98ms/token`; this bucket materialises the lazy full-token forward plus
  sampling evaluation and dominates the microsecond-scale Go orchestration
  fields.
- [x] Move the chosen recurring bucket into `go/internal/metal` as a stable
  C/C++ wrapper API. 2026-05-16 progress: `go/internal/metal/decode.go` and
  `go/internal/metal/decode_bridge.cpp` now route deterministic single-step
  greedy decode through a native C++ wrapper for both one-shot generation and
  retained `ModelSession` generation. 2026-05-17 progress: the gated
  last-token output projection wrapper (`GO_MLX_ENABLE_LAST_LOGITS_PREFILL=1`)
  was benchmarked and produced `44.874611039475575 tok/s`, slightly below the
  previous native-greedy rerun. The native GELU MLP sub-block wrapper
  (`GO_MLX_ENABLE_NATIVE_MLP_GELU=1`) was also benchmarked and produced
  `43.10698466210642 tok/s`, so it remains disabled by default. A gated
  one-token Gemma 4 layer wrapper (`GO_MLX_ENABLE_NATIVE_GEMMA4_LAYER=1`) now
  covers the conservative E2B q4 decode shape: no MoE, no LoRA, single-token
  decode, no cache trim, paged cache with at most one page, attention, MLP,
  residuals, per-layer input injection, layer scalar, and native cache page
  handoff. It lowered Go-side forward construction time (`~0.99ms` to
  `~0.60ms/token`) but increased MLX eval time (`~20.21ms` to
  `~21.77ms/token`), producing `44.54197676930399 tok/s` versus the same
  rebuilt binary's gate-off control at `47.054122991613305 tok/s`. It remains
  disabled by default. A follow-up MLX-compiled layer closure
  (`GO_MLX_ENABLE_COMPILED_GEMMA4_LAYER=1`) adds dynamic RoPE offset support
  and fails closed on the real E2B path: MLX compile cannot reuse the closure
  across the growing K/V length and reports a broadcast mismatch between
  `(...,24,head_dim)` and `(...,23,head_dim)`. The fail-closed smoke generated
  normally through fallback at `44.437334470929095 tok/s` for one run. The
  positive full materialisation boundary remains open and likely needs a
  lower-level dynamic cache/block-table kernel rather than MLX compile over the
  existing growing-cache graph. `/private/tmp/llama.cpp` was cloned and
  inspected at commit `1a68ec9`; its Metal path reinforces that the next
  useful boundary is stable graph topology plus host-updated decode inputs, not
  another wrapper around the current growing MLX arrays. Relevant patterns:
  graph reuse when topology parameters match, host-fed K/V index and KQ-mask
  tensors, cache-slot planning before graph input update, flash attention for
  quantized V cache, and asynchronous Metal command-buffer submission. The
  default activation helper was also restored after a native activation-wrapper
  probe dropped the gate-off control to `40.956652070193485 tok/s`; the
  restored control is `46.37096822259417 tok/s` with binary SHA-256
  `0c4c9ec67aa16964b270fd349f3ce1bfea18680857f80d52f86b6c0e51d78f03`. See
  `docs/runtime/2026-05-17-gemma4-parity-and-last-logits.md`. 2026-05-17
  follow-up: the first fixed-shape decode-input primitive now exists and is
  verified by focused tests. `singleTokenCausalMask` builds an offset-fed mask,
  `singleTokenCacheUpdate` writes one K/V token into a fixed-capacity cache
  tensor via dynamic indices, and `fixedSingleTokenAttention` combines update,
  mask, and masked SDPA inside a reusable compiled closure. It proves MLX
  compile can reuse the closure across changing offsets when K/V shapes stay
  fixed, which is the concrete next step implied by the `llama.cpp` reference
  pass. A follow-up native bridge now exposes the same shape as
  `go_mlx_compiled_fixed_single_token_attention` in
  `go/internal/metal/decode_bridge.cpp`, so the host-fed offset plus fixed-K/V
  update path has a stable C++ wrapper API instead of only a Go-authored MLX
  graph primitive. It is wired into the gated fixed-cache compiled-layer path,
  and into `Gemma4Attention.forward` when the gated fixed-cache owner path can
  keep full-capacity K/V tensors, with fallback to the Go-authored graph if the
  native wrapper rejects a shape.
  Focused verification passed with
  `go test ./internal/metal -run 'TestGemma4_AttentionFixedCacheUsesNativeBridge_Good|TestDecode_(nativeFixedSingleTokenAttention|compiledGemma4DecodeLayer_FixedCacheGood)|TestFast_(fixedSingleTokenAttention_CompiledGood|singleTokenCacheUpdate_CompiledGood|singleTokenCausalMask_Good)' -count=1`.
  The full-context gated target rerun with binary SHA-256
  `be3983cfb67edcc7b784df38500a0350f6013a5f35692a38e7aa55ab8a1b7c6d`
  records `decode_tokens_per_sec_average: 107.77701729520602`, with three full
  128-token runs at `95.07907894498449`, `116.20241438731288`, and
  `112.0495585533207`, prefill at `844.1085014532886 tok/s`, and peak memory
  `3327392930` bytes. This turns the fixed-cache topology from a negative
  full-context probe into a gated positive E2B path, while leaving default
  selection and large-model throughput as separate open decisions. The same bridge
  was then probed on shared Gemma 4 31B q4. The unguarded fixed-cache native
  bridge aborts after one token because the current bundled metallib cannot
  load `sdpa_vector_float_512_512` for the 512-wide attention head path and
  reports `kIOGPUCommandBufferCallbackErrorInvalidResource`; the bridge guard
  now rejects 512-wide heads and falls back instead of crashing. The guarded
  160-slot run, which covers the 29-token prompt plus 128 generated tokens,
  completes at `24.94401176949734 tok/s` with runs
  `25.24160351823528`, `24.74238342491899`, and `24.848048365337757`,
  still below the archived `34.893 tok/s` Python-runner datapoint. See
  `docs/runtime/2026-05-17-go-mlx-gemma4-31b-q4-fixed-cache160-native-bridge-longdecode.json`
  for the failing unguarded 512-wide attempt and
  `docs/runtime/2026-05-17-go-mlx-gemma4-31b-q4-fixed-cache160-native-bridge-guarded-longdecode.json`
  for the guarded fallback result. A native matmul-softmax fallback for
  512-wide fixed single-token attention now exists behind
  `GO_MLX_ENABLE_FIXED_WIDE_MATMUL_ATTENTION=1` and is covered by a
  Metal-enabled grouped-query test, but the three-run 31B diagnostic benchmark
  records only `24.333176943291804 tok/s` with binary SHA-256
  `e5860c064f2a831db1a6a0afaab18c5cfc4d6b28b98c4a3131e0a35e0b29da5d`.
  It is slower than the guarded fallback, so it remains diagnostic only rather
  than the default 512-wide path. See
  `docs/runtime/2026-05-17-go-mlx-gemma4-31b-q4-fixed-cache160-native-matmul-longdecode.json`.
  The lower-level MLX source confirms the bundled metallib only instantiates
  SDPA vector heads through `256`. `patches/mlx-sdpa-vector-512.patch` records
  the minimal upstream MLX experiment to instantiate 512-wide vector SDPA and
  mark 512 as a supported vector head dimension; the patch has now been applied
  to `lib/mlx`, rebuilt into `dist/lib/mlx.metallib`, and benchmarked on the
  shared-31B longdecode lane. The fused SDPA512 run is clean but still negative:
  `24.70397262176645 tok/s` versus the guarded fallback's
  `24.94401176949734 tok/s`. This moves the 31B blocker from "missing 512-wide kernel" to
  "the one-token eval/materialisation path around attention is still doing too
  much work". A follow-up llama.cpp-style shared-mask gate
  (`GO_MLX_ENABLE_FIXED_GEMMA4_SHARED_MASK=1`) host-feeds one fixed-cache mask
  per token instead of building the same mask inside every layer. It is correct
  but neutral on the same 31B longdecode lane: `24.904493509253538 tok/s` when
  the 512-wide native SDPA path is still guarded off and
  `24.767920780634018 tok/s` when `GO_MLX_ENABLE_FIXED_WIDE_SDPA_ATTENTION=1`
  is enabled. The direct greedy output probe was also paired on 31B and
  regressed to `23.2767195467288 tok/s`, confirming output projection/argmax is
  not the missing boundary either.
  Follow-up: Gemma 4 now has an experimental fixed-cache compiled-layer
  lane behind `GO_MLX_ENABLE_FIXED_GEMMA4_CACHE=1`,
  `GO_MLX_ENABLE_COMPILED_GEMMA4_LAYER=1`, and optional
  `GO_MLX_FIXED_GEMMA4_CACHE_SIZE`. It validates the topology thesis but does
  not meet the performance target: full-context `4096` slots regressed to
  `39.88411733551154 tok/s`, `256` slots reached `43.18471280763444 tok/s`,
  `160` slots reached `45.95924162792853 tok/s`, `96` slots reached the best
  probe at `47.03732918131478 tok/s`, and `64` slots reached
  `46.870613364571796 tok/s`. The default post-change control remained
  `46.20225853209359 tok/s`. The result points to a lower-level attention/cache
  kernel rather than masked SDPA over unused fixed-cache cells. A final
  output-boundary probe (`GO_MLX_ENABLE_DIRECT_GREEDY_TOKEN=1`) fuses final
  RMSNorm, q4 output projection, and argmax when sampling is strictly greedy.
  It is also negative: the 3-run target rerun averaged
  `44.27055794965946 tok/s` because the same lazy one-token forward still
  materialises in `Eval(next)`. It remains disabled by default. A
  llama.cpp-inspired async command-submission probe
  (`GO_MLX_ENABLE_ASYNC_DECODE_PREFETCH=1`) starts `EvalAsync` on the next lazy
  decode value before the next sampling read. It is neutral rather than useful:
  the 3-run target rerun averaged `46.233006105790245 tok/s`, effectively the
  default paged-cache band, because the loop has little CPU-side work to overlap
  with Metal execution. That old non-session driver-profile result was later
  superseded for retained `ModelSession.Generate` by the seeded state-ramp rows
  above, where the same existing gate produced a measurable full-workflow win
  and was promoted into the Gemma 4 fast lane. The next cache probe
  attacked the local cache mismatch where go-mlx concatenated the last
  paged K/V block on every decode token. `GO_MLX_ENABLE_PAGED_KV_PREALLOC=1`
  keeps pages at fixed capacity and updates visible slices instead. It was
  clean but effectively neutral: same-binary gate-off averaged
  `46.50781893730525 tok/s`, while preallocated pages averaged
  `46.53706420697521 tok/s`. It remains disabled by default. A dense
  `Linear` transpose-cache probe matched the existing `SwitchLinear` pattern
  but was negative on the target (`45.9393904182794 tok/s`), likely because
  retaining the lazy transpose graph was more expensive than rebuilding the
  cheap transpose view around the dense call. That patch was reverted. The
  next layer-0 trace spike probe compiled Gemma 4 per-layer input construction
  behind `GO_MLX_ENABLE_COMPILED_GEMMA4_PER_LAYER_INPUTS=1`; it was also
  neutral/negative at `46.93672879306734 tok/s` versus the same-binary gate-off
  control at `46.9841490339839 tok/s`, so it remains disabled by default. A
  correctness-breaking diagnostic gate
  (`GO_MLX_DISABLE_GEMMA4_PER_LAYER_INPUTS=1`) then skipped that required
  Gemma 4 per-layer input construction entirely. It is not a valid model path,
  but it is a useful isolation proof: the same target run jumped to
  `114.9355811775564 tok/s` with full 128-token generations, steady eval around
  `7.890701744ms/token`, and peak memory `3835433982` bytes. The blocker is
  now concrete: preserve the per-layer semantics while avoiding repeated dense
  projection/materialisation of the per-token `[35,256]` side input. The
  correct fix landed in the quantized embedding path: `Embedding.Forward` now
  gathers packed token rows, scales, and biases before dequantising instead of
  dequantising the full vocabulary table and then taking a row. The exact E2B
  target command now reports `121.9379742475021 tok/s`, steady eval around
  `7.111331777777778ms/token`, and peak memory `3166205126` bytes on the
  default valid path. Final follow-up on the current no-thinking Gemma 4 chat
  template reports `124.88170583124456 tok/s` with three full 128-token E2B
  generations. The same pass removed explicit K/V head expansion from Gemma 4
  direct fast-SDPA paths after tests proved grouped-query, causal grouped-query,
  and masked grouped-query attention match the old repeated-K/V result. On the
  shared 31B q4 large-model lane the current default three-run sample records
  `24.663669410625896 tok/s`. The earlier no-thinking `mlx_lm.generate`
  comparison at `36.185 tok/s` is archived historical context only; it is no
  longer an active benchmark target.
  The gated native-layer direct-GQA probe remains disabled because it reports
  `24.85650433260677 tok/s`, below the default path. A gated native GELU
  gate-multiply probe reaches `25.260023959706817 tok/s` for one run and
  `25.084752484961715 tok/s` under tracing, but remains disabled because it is
  not a stable parity fix. The current-order async prefetch probe reports
  `24.41755011370027 tok/s` and confirms that async submission mostly moves
  work into the unaccounted bucket on this CLI workload.
- [x] Cache compiled MLX closures when shape-compatible. Do not rebuild native
  functions per token. `compiled_greedy_decode_token()` is a static MLX
  compiled closure and the generator only uses it once logits are already
  single-step, leaving variable-shape prefill logits on the existing path.
- [x] Record the native-boundary decision for the broad one-call wrapper.
  Go still owns architecture-level one-token forward orchestration, and the
  broad `GO_MLX_ENABLE_NATIVE_GEMMA4_MODEL_GREEDY=1` wrapper remains rejected
  because it regresses the 26B A4B q4 lane into the `50 tok/s` band. This
  resolves one rejected native-boundary branch; it does not complete the
  production goal. The current q4-first candidate keeps the proven native
  sub-blocks in `go/internal/metal` while the live production gates remain the
  100k retained-state rerun, accepted long-form workflow evidence, long-context
  decode bounds, and external runner anchors. The full one-token native
  boundary remains future R&D under the candidate boundary list below.
  Historical audit, now superseded as completion proof:
  `docs/runtime/2026-05-19-goal-completion-audit.md`.
- [x] Re-run the benchmark command after every boundary change and record the
  before/after tok/s. The 2026-05-16 native-greedy/session rebuild produced
  `bin/lthn-mlx` SHA-256
  `878797bbecec3f9e7f2c1614233220d15f94aa180c7118567fd1f660b9daf8bb`;
  the exact profile rerun completed outside the sandbox with
  `decode_tokens_per_sec_average: 44.93695802859693` versus the prior
  `44.55943393415422` baseline (`+0.3775240944427125 tok/s`, `+0.847%`).
  See `docs/runtime/2026-05-16-gemma4-e2b-native-greedy-rerun.json`. The
  2026-05-17 last-token output projection rerun used `bin/lthn-mlx` SHA-256
  `5c8aeea06fece0b49683e1683e2204447266f1fedbe7f2a642622af6deccd979` and
  produced `decode_tokens_per_sec_average: 44.874611039475575`, so it is not a
  positive optimisation boundary. See
  `docs/runtime/2026-05-17-gemma4-e2b-last-logits-prefill-rerun.json`. The
  gated native MLP rerun used `bin/lthn-mlx` SHA-256
  `85443fb248abe47afb546ee720e661b8f7dbae292981d0b98b00263799b1380b` and
  produced `decode_tokens_per_sec_average: 43.10698466210642`; the gate-off
  default rerun produced `44.89465488606482`, so the MLP wrapper is a negative
  boundary probe rather than a default runtime path. The cache-mode diagnostic
  flag then confirmed the paged KV path is a real but insufficient positive
  boundary: a sequential `-cache-mode paged` confirmation rerun produced
  `decode_tokens_per_sec_average: 46.94074033007464` with the steady
  `sample_eval_duration` average at `20.309252947ms/token`. A follow-up
  resolved-load fix now lets the unmodified target command report the effective
  planner shape and select paged KV from host-reported Apple memory without
  requiring the full MLX device probe; the same target command now records
  `cache_mode: "paged"` and `decode_tokens_per_sec_average:
  46.50145764359926`. See
  `docs/runtime/2026-05-17-gemma4-e2b-native-mlp-rerun.json` and
  `docs/runtime/2026-05-17-gemma4-e2b-native-mlp-gated-default-rerun.json`,
  plus `docs/runtime/2026-05-17-gemma4-e2b-cache-paged-confirm-rerun.json`
  and `docs/runtime/2026-05-17-gemma4-e2b-resolved-load-rerun.json`. The
  gated native layer rerun used `bin/lthn-mlx` SHA-256
  `bfefdf9510dfc399a7018eaa12447c763395afe1adae949a4135c8befc21e3ff` and
  produced `decode_tokens_per_sec_average: 44.54197676930399`; the same binary
  with the layer gate off produced `47.054122991613305`, so the layer wrapper
  is a negative boundary probe rather than a default runtime path. See
  `docs/runtime/2026-05-17-gemma4-e2b-native-layer-rerun.json` and
  `docs/runtime/2026-05-17-gemma4-e2b-native-layer-gateoff-rerun.json`. The
  compiled-layer diagnostic used `bin/lthn-mlx` SHA-256
  `1b71031e4d379217b13654b955d1db3171408886d101ebeb3a0f12cd55161185`; the
  gate failed closed with the MLX compile broadcast error captured in
  `docs/runtime/2026-05-17-gemma4-e2b-compiled-layer-failclosed.stderr`, while
  the JSON profile recorded `decode_tokens_per_sec_average:
  44.437334470929095` through fallback. See
  `docs/runtime/2026-05-17-gemma4-e2b-compiled-layer-failclosed.json`. The
  async prefetch diagnostic used `bin/lthn-mlx` SHA-256
  `a0ccacd82285720cd5a7865d5d0cb5724519e5430f4aebe9b6e9b8940f89a487` and
  produced `decode_tokens_per_sec_average: 46.233006105790245`, with runs at
  `46.298560210152495`, `46.49208501310205`, and `45.908373094116186`. See
  `docs/runtime/2026-05-17-gemma4-e2b-async-prefetch-rerun.json`. The paged KV
  preallocation diagnostic used `bin/lthn-mlx` SHA-256
  `fb53bb00561040f6123966746969f157adedffea967777a1ef6fa9392c6ef590`; its
  gate-off control recorded `46.50781893730525`, while
  `GO_MLX_ENABLE_PAGED_KV_PREALLOC=1` recorded
  `46.53706420697521 tok/s`. See
  `docs/runtime/2026-05-17-gemma4-e2b-paged-kv-prealloc-gateoff-rerun.json`
  and `docs/runtime/2026-05-17-gemma4-e2b-paged-kv-prealloc-rerun.json`. The
  dense linear transpose-cache probe used `bin/lthn-mlx` SHA-256
  `0755991897c7165eda960010d5709d56a3aa956ea6c6c1bb05afce8cfc2c3e95` and
  produced `decode_tokens_per_sec_average: 45.9393904182794`, so it was
  reverted. See
  `docs/runtime/2026-05-17-gemma4-e2b-linear-transpose-cache-rerun.json`. The
  compiled per-layer-input diagnostic used `bin/lthn-mlx` SHA-256
  `900b2e041f103f767575c0ae544fc29fd6b48e6a9a81373158e5885a5f4aeebf`; the gate
  produced `decode_tokens_per_sec_average: 46.93672879306734`, while the
  same-binary gate-off control produced `46.9841490339839`. See
  `docs/runtime/2026-05-17-gemma4-e2b-compiled-per-layer-inputs-rerun.json`
  and
  `docs/runtime/2026-05-17-gemma4-e2b-compiled-per-layer-inputs-gateoff-rerun.json`.
  The disabled per-layer-input diagnostic used `bin/lthn-mlx` SHA-256
  `c097cb7612b7c402880fb0ba7a1bad7baad1494df43dceec059feeef9e99942d`;
  `GO_MLX_DISABLE_GEMMA4_PER_LAYER_INPUTS=1` produced
  `decode_tokens_per_sec_average: 114.9355811775564`, with runs at
  `117.0486414046229`, `117.46595644094181`, and `110.29214568710452`, and
  generated token counts `[128,128,128]`. See
  `docs/runtime/2026-05-17-gemma4-e2b-disable-per-layer-inputs-rerun.json`.
  The valid row-gather fix used `bin/lthn-mlx` SHA-256
  `c40c7566f3b746a8072ae7c8f83f3c50ac05a46ac8b08d658d92752ea37b0536`;
  the target command produced `decode_tokens_per_sec_average:
  121.9379742475021`, with runs at `120.35003784437026`,
  `123.6154742394561`, and `121.84841065867997`. See
  `docs/runtime/2026-05-17-gemma4-e2b-quantized-embedding-row-gather-rerun.json`.
  The final current default binary, SHA-256
  `3d720db7a77235104b48707d50e27170c6e8e7b97dd022cba32acaaa6f4673e9`,
  reports `124.88170583124456 tok/s` on the same E2B target command with
  three full 128-token runs. The same binary family records a shared-31B
  current-default sample of `24.663669410625896 tok/s` across three
  no-thinking runs, versus the secondary `36.185 tok/s` datapoint from
  the archived `mlx_lm.generate` measurement. See
  `docs/runtime/2026-05-17-gemma4-e2b-final-current-default-rerun.json` and
  `docs/runtime/2026-05-17-go-mlx-gemma4-31b-q4-final-current-default-3run-parity.json`.
  A llama.cpp comparison was then run against the closest local 26B A4B pair:
  go-mlx q4 MLX safetensors versus llama.cpp `Q8_0` GGUF. The comparison is
  not strict same-quant evidence, but it includes prefill: go-mlx records
  `447.6882783215051 tok/s` on a 29-token prompt and
  `55.96521969803896 tok/s` decode for 128 generated tokens; llama.cpp records
  `375.334002 tok/s` for `pp29`, `87.688525 tok/s` for `tg128`, and
  `2231.973259 tok/s` for `pp2048`. The run also fixed a Gemma 4 26B loader
  bug by inferring q8 dense MLP/router projections from packed weight and scale
  shapes under the default q4 quantisation block. See
  `docs/runtime/2026-05-17-llamacpp-prefill-comparison.md`.
  A cleaner llama.cpp `Q4_K_M` follow-up on the same GGUF repo records
  `468.942791 tok/s` for `pp29`, `89.000726 tok/s` for `tg128`, and
  `2184.109033 tok/s` for `pp2048`. Against go-mlx q4 this leaves a
  `1.59x` decode gap and a `2.53x` large-prefill gap.
  The next llama.cpp code read found that Gemma MoE keeps the expert
  `gate_up` projection fused when the tensor exists, whereas go-mlx had
  sanitised it into separate gate and up projections and then executed two
  expert-indexed projections. go-mlx now retains the fused
  `experts.switch_glu.gate_up_proj` tensors and uses them only for
  single-token decode. The ungated prefill use regressed long prefill, so the
  guard is intentionally decode-only. On rebuilt binary SHA-256
  `085e204e17aa0f4f1fe614efa090f8779832129de5c377bf8b570902b3172f7b`, the
  26B A4B q4 short-prompt run records `56.45505318098333 tok/s` decode and
  `449.18863738146 tok/s` prefill, while the clean long-prefill run records
  `862.5952429295362 tok/s`. This is a small decode-only win over the
  previous `55.96521969803896 tok/s` result and does not close the
  llama.cpp Q4_K_M gap.
  A follow-up long-prefill probe found another double-work boundary: default
  prefill materialised full `[sequence,vocab]` logits before slicing the last
  row. go-mlx now automatically uses the existing `ForwardLastTokenLogits`
  model path for long prompts at or above 512 tokens, while preserving the
  short-prompt full-logits path unless `GO_MLX_ENABLE_LAST_LOGITS_PREFILL=1`
  explicitly forces it. On rebuilt binary SHA-256
  `dd212338c1864b6acb630bb5f534986432d1c189d17e100ae8ab3a3ee230a352`, the
  same 26B A4B q4 short-prompt decode rerun records
  `56.220244342267904 tok/s` and the clean 2061-token long-prefill run records
  `903.0290085147915 tok/s`. This narrows the long-prefill gap from `2.53x` to
  `2.42x`, but llama.cpp still leads decisively. A tiny-tail chunk coalescing
  probe was rejected because one 2061-token prefill pass regressed to
  `862.4738054025554 tok/s`; keeping the `2048 + 13` chunk split is faster for
  this MLX path.
  A llama.cpp-style shared-KV last-token trim after the final KV-owning Gemma 4
  layer was also tested and rejected. It nudged one clean long-prefill run only
  to `911.1355151113232 tok/s` and regressed the 128-token decode check to
  `53.616341210113625 tok/s`; the code was reverted and the accepted binary
  remains SHA-256 `dd212338c1864b6acb630bb5f534986432d1c189d17e100ae8ab3a3ee230a352`.
  Fixed-cache compiled-layer probes on the same active 26B A4B q4 lane were
  also negative: full-context fixed cache recorded `48.211754489053696 tok/s`
  decode and a 160-slot fixed cache recorded `53.69079065280556 tok/s`, both
  below the accepted default. The llama.cpp-only traces now show the remaining
  gap is evaluated graph work rather than Go orchestration: default token-phase
  tracing averages `17.432ms/token` in `sample_eval_duration`, while forced
  native phase tracing points at FFN first (`~20.082ms/token`), then attention
  (`~12.393ms/token`). The follow-up FFN split trace records 270 gated native
  events/token and puts the largest sub-buckets at routed expert gather/down/sum
  (`13.736ms/token`), attention (`10.614ms/token`), local MLP
  (`8.354ms/token`), and router/top-k (`7.560ms/token`). See
  `docs/runtime/2026-05-17-go-mlx-gemma4-26b-a4b-q4-fixed-cache-compiled-layer-llamacpp-comparison-longdecode.json`,
  `docs/runtime/2026-05-17-go-mlx-gemma4-26b-a4b-q4-fixed-cache160-compiled-layer-llamacpp-comparison-longdecode.json`,
  `docs/runtime/2026-05-17-go-mlx-gemma4-26b-a4b-q4-default-token-phase-trace-llamacpp-comparison.json`,
  `docs/runtime/2026-05-17-go-mlx-gemma4-26b-a4b-q4-native-phase-trace-llamacpp-comparison.json`,
  and
  `docs/runtime/2026-05-17-go-mlx-gemma4-26b-a4b-q4-native-phase-ffn-split-trace-llamacpp-comparison.json`.
  A direct native fused-experts probe then moved `gate_up` gather, GELU, down
  gather, expert weighting, and top-k sum behind one opt-in wrapper. It was
  rejected because the real 26B A4B q4 lane regressed to
  `53.08901433576139 tok/s` decode and `431.27066684929787 tok/s` prefill
  across three full 128-token runs. The source was reverted; the diagnostic is
  kept in
  `docs/runtime/2026-05-17-go-mlx-gemma4-26b-a4b-q4-native-fused-experts-llamacpp-comparison-longdecode.json`.
  Revalidation on rebuilt binary SHA-256
  `c1034cf834b9c40d65c0e9bcf2652f5c2232965ef1715188c89fb5eff8abf141`
  keeps the exact E2B target safely above the floor at
  `121.19859628423075 tok/s`, with three full 128-token runs, and nudges the
  shared-31B throughput lane to `24.971269037945117 tok/s`. The active external
  miss is now llama.cpp Q4_K_M on the closest local 26B A4B comparison. See
  `docs/runtime/2026-05-17-gemma4-e2b-mixed-quant-loader-rerun.json` and
  `docs/runtime/2026-05-17-go-mlx-gemma4-31b-q4-mixed-quant-loader-3run-parity.json`.
  A sustained no-thinking 31B diagnostic prompt that forces all 128 generated
  tokens records go-mlx at `23.086428954337055 tok/s` across three runs. This
  is internal large-model evidence only; the implementation and benchmark model
  to copy is the llama.cpp stable graph and host-fed KV input path. See
  `docs/runtime/2026-05-17-go-mlx-gemma4-31b-q4-longdecode-3run-parity.json`.
  A gated native MLP rerun was measured directly on the shared-31B diagnostic lane
  because the native phase trace points at FFN work. It averaged
  `24.7143167044012 tok/s`, below the mixed-quant default, so the gate stays
  disabled. See
  `docs/runtime/2026-05-17-go-mlx-gemma4-31b-q4-native-mlp-mixed-quant-parity.json`.
- [x] Add a gated native phase trace before attempting a full layer wrapper.
  `GO_MLX_TRACE_FORWARD_EVAL=1` now records per-token `native_events` under
  `-trace-token-phases` and forces/detaches Gemma 4 attention,
  attention-residual, FFN, and layer-output boundaries. The diagnostic E2B run
  is intentionally slower (`18.09851769746586 tok/s`) but records 2,800 native
  events across one run. Excluding warmup and the final token, each decode step
  records 140 events (35 layers x 4 boundaries), with p50 per-boundary timings
  around `0.265ms` attention, `0.261ms` FFN, `0.222ms` output, and `0.168ms`
  attention-residual; `gemma4.layer.00.output` remains a large cumulative
  boundary at `~11.8ms` p50. This confirms the next useful implementation is a
  whole one-token layer/materialisation boundary, not another isolated MLP or
  output-projection wrapper. See
  `docs/runtime/2026-05-17-gemma4-e2b-native-phase-trace.json`.
  The 26B A4B q4 follow-up adds trace-only FFN sub-boundaries on the active
  llama.cpp lane. It is intentionally slower (`14.452280580872943 tok/s` under
  trace overhead), but across 29 steady samples it records 270 native
  events/token and attributes the largest totals to `ffn_experts`
  (`13.736ms/token`), attention (`10.614ms/token`), `ffn_local_mlp`
  (`8.354ms/token`), and `ffn_router` (`7.560ms/token`). The failed
  native fused-experts wrapper shows this is not solved by wrapping the same
  MLX gather graph; the useful next boundary is lower-level quantized MoE or a
  broader llama.cpp-style one-token block. See
  `docs/runtime/2026-05-17-go-mlx-gemma4-26b-a4b-q4-native-phase-ffn-split-trace-llamacpp-comparison.json`.
  Static MLX/llama.cpp kernel reading narrows the next MoE target further:
  go-mlx's `SwitchLinear` calls MLX `GatherQMM` with unsorted RHS expert
  indices; MLX only uses its batched `gather_qmm_rhs` path when indices are
  globally sorted and the batch is large enough (`M == 1`, `B >= 16`, and
  `B / E >= 4`). Single-token 26B decode is top-k 8 over 128 experts, so it
  falls to the vector gather path. llama.cpp lowers Gemma MoE to
  `GGML_OP_MUL_MAT_ID`, then uses `kernel_mul_mv_id` for small token counts and
  `kernel_mul_mm_id` plus an expert-ID map for batched work. This makes the
  next native target an ID-matvec/ID-matmul expert kernel, not just an MLX
  sorted-gather wrapper.
  The source now has trace-only subevents inside `Gemma4Experts.forward`
  (`ffn_expert.gate_up`, `activation`, `down`, `weighted`, `sum`) so the next
  Metal-available trace can split the routed expert bucket without changing the
  default runtime path.
  A first internal correctness scaffold now exists in
  `go/internal/metal/expert_id_matvec.go`: `quantizedExpertIDMatVec` consumes
  MLX affine-packed q2/q4/q8 expert rows plus route expert ids and matches a
  CPU q4 reference on small and multi-pack tensors. The scaffold now uses one
  SIMD group per routed output row, which is closer to llama.cpp's ID-matvec
  primitive than the first serial proof. The custom kernel handle is cached per
  shape, and the path is wired into Gemma 4 experts only behind
  `GO_MLX_ENABLE_EXPERT_ID_MATVEC=1`; a unit regression compares that opt-in
  path against the existing MLX `GatherQMM` route. The down-projection side now
  uses a weighted expert-ID matvec-sum kernel, folding route weighting and
  top-k summation into the down matvec instead of leaving them as separate MLX
  nodes. The default runtime is unchanged until the gate has llama.cpp-lane
  benchmark evidence. A first full 26B A4B q4 env-gated probe was attempted,
  but the local runtime failed before generation with `no usable Metal device
  available`, so that artefact is environment evidence only. `driver-profile`
  now records active native runtime gates in `runtime_gates`, and a diagnostic
  `-expert-id-matvec` flag enables the same internal gate without relying on a
  second environment variable. The valid three-run llama.cpp-lane diagnostic is
  negative: `55.98273536629838 tok/s` decode and `449.436848070603 tok/s`
  short prefill, below the accepted go-mlx decode control at
  `56.220244342267904 tok/s`. llama.cpp `Q4_K_M` still leads the gated path by
  `1.5898x` on decode. A narrower fused-activation variant moved
  `GELU(gate) * up` into the custom expert-ID gate_up kernel behind
  `GO_MLX_ENABLE_EXPERT_ID_FUSED_ACTIVATION=1`; same-binary controls record
  `56.21477992583666 tok/s` for default, `56.06328243808281 tok/s` for
  non-fused expert-ID matvec, and `56.295534088943356 tok/s` for the fused
  variant. That is only `+0.14%` over the same-binary default control and still
  leaves llama.cpp `Q4_K_M` `1.5809x` faster, so it remains diagnostic only.
  A larger prefill-specific follow-up now uses MLX's own sorted RHS
  `GatherQMM` path for Gemma 4 prefill. `driver-profile -prompt-file` keeps
  long prompt inputs out of shell-generated argv, and
  `driver-profile -sorted-expert-prefill` records
  `runtime_gates.GO_MLX_ENABLE_SORTED_EXPERT_PREFILL=1` while sorting flattened
  routes by expert id, running split gate/up/down gathers with `sorted=true`,
  and restoring route order before top-k weighting. On the same binary with
  `README.md` as a 2204-token prompt-file input, the default control is
  `914.0299819202297 tok/s` prefill and `31.048941804155767 tok/s` decode;
  the same-binary sorted prefill path is `1914.0303789361128 tok/s` prefill and
  `31.508051014734626 tok/s` decode. That is a `2.0940x` prefill speedup and
  puts go-mlx at `87.6%` of llama.cpp `Q4_K_M` `pp2048` throughput
  (`2184.109033 tok/s`). The next llama.cpp-only follow-up added
  `driver-profile -paged-decode-fast-concat` for
  `GO_MLX_ENABLE_PAGED_DECODE_FAST_CONCAT=1`: multi-page single-token decode
  concatenates the paged KV state once and calls the regular SDPA path instead
  of the hand-rolled paged attention loop. With sorted prefill plus fast concat,
  the prompt-file lane records `1909.1904478108413 tok/s` prefill and
  `42.372384580120396 tok/s` decode. That is a `1.3448x` decode speedup over
  the same-binary sorted-prefill-only control, but llama.cpp `Q4_K_M` `tg128`
  at `p2048` is still `92.624334 tok/s`, or `2.186x` faster. Prefill is now
  close; long-context decode remains the bad lane. A further
  `driver-profile` cleanup lets the existing fixed-cache and compiled Gemma 4
  decode diagnostics run through CLI runtime gates instead of env-only package
  init switches: `-fixed-gemma4-cache`, `-fixed-gemma4-shared-mask`, and
  `-compiled-gemma4-layer`. The same README prompt-file lane with sorted
  prefill plus those fixed-cache compiled gates records
  `1876.6924105183755 tok/s` prefill and `48.93511098804883 tok/s` decode.
  That is `1.5531x` over sorted-prefill-only decode and `1.1549x` over the
  paged fast-concat decode probe, but still leaves llama.cpp `Q4_K_M`
	  `1.8928x` faster on long-context decode. Adding `driver-profile
	  -direct-greedy-token` records a 3-run average of `1908.4658285603446 tok/s`
	  prefill and `49.75515922842408 tok/s` decode. That is only `1.0168x` over
	  the fixed-cache compiled probe and leaves llama.cpp `Q4_K_M` `1.8616x`
	  faster. A follow-up added MoE support inside the opt-in compiled Gemma 4
	  decode graph; the tiny MoE regression passes, but the full 26B A4B profile
	  remains in the same `49.6-49.8 tok/s` band, so simply compiling the existing
	  MoE graph is not the missing llama.cpp boundary. A later source read found
	  that llama.cpp routes Gemma 4 MoE logits from the attention residual, not
	  the pre-FFN2-normalised expert input; go-mlx now matches that boundary. The
	  current best
	  long-context go-mlx decode result is sorted prefill plus expert-ID fused
	  direct-greedy decode with router-residual parity at
	  `1933.6368792628773 tok/s` prefill and `50.23367760579547 tok/s` decode,
	  leaving same-prompt-length llama.cpp `Q4_K_M` `1.8205x` faster. The older
	  C++ `-native-gemma4-layer` gate was
	  dense-only because its ABI did not carry MoE router/expert tensors. A
	  later same-lane rebuild kept fixed-cache sizing uniform for the compiled
	  decode path and records `1923.322483219664 tok/s` prefill with
	  `49.71518402860789 tok/s` decode. The rejected sliding-window fixed-cache
	  diagnostic confirms the cache-size hypothesis is not enough by itself:
	  it drops decode to `40.76006207167587 tok/s` and pushes peak memory to
	  `71228950132` bytes. A llama.cpp-inspired two-column down-projection
	  matvec also regressed to `48.4963971321882 tok/s`, so the next kernel work
	  should target the full ID-matvec shape rather than this partial row-pair
	  variant. The follow-up trace found the real expert-ID miss: the active MLX
	  safetensors do not have a fused `gate_up_proj`; they store split
	  `gate_proj` and `up_proj` tensors, and their q4 scale/bias sidecars are
	  BF16. The earlier fused-activation expert-ID gate therefore fell back on
	  this model. The new split/BF16 expert-ID path is active on the 26B A4B q4
	  pack and records `62.52025013199337 tok/s`; the split fused-activation
	  kernel records `68.22675114228564 tok/s`; and the shared-input variant
	  avoids broadcasting the single hidden row across top-k routes, reaching
	  `70.54498924012704 tok/s` decode with empty stderr. Same-prompt-length
	  llama.cpp `Q4_K_M` still leads at `91.451031 tok/s`, so the remaining
	  external parity gap is `1.2964x`. A non-native token-phase profile on the
	  same lane records `71.59452329863376 tok/s`, with steady tokens averaging
	  `14.0596ms`: `12.7249ms` is still spent inside `Eval(next)` and only
	  `1.2977ms` constructing the next forward graph. Re-enabling the existing
	  native dense MLP GELU wrapper is neutral-to-negative at
	  `71.44678366026884 tok/s`, so the next optimisation should target a larger
	  eval/materialisation boundary such as output greedy argmax/projection or
	  broader stable graph reuse, not another standalone MLP wrapper. The next
	  kernel pass fixed a concrete q4 packing inefficiency: expert-ID kernels now
	  iterate packed `uint32` q words and unpack their lanes locally, instead of
	  having adjacent SIMD lanes reload the same packed word for each scalar
	  input column. The final packed-column 3-run lane records
	  `1936.5495347431952 tok/s` prefill and `79.1105587686013 tok/s` decode.
	  That is `1.1214x` faster than the prior shared-input expert-ID result and
	  reduces the same-prompt-length llama.cpp decode gap to `1.1560x`. It is
	  still below the `100 tok/s` floor by `1.2641x`. Right-sizing the fixed
	  Gemma 4 cache for the same 2204-token prompt plus 128-token decode then
	  reduced attention's fixed-capacity tax: `GO_MLX_FIXED_GEMMA4_CACHE_SIZE=2336`
	  records a 3-run average of `1937.0948107149452 tok/s` prefill and
	  `84.23477753697784 tok/s` decode. That is `1.0648x` faster than the
	  packed 4096-slot baseline, leaves same-prompt llama.cpp only `1.0857x`
	  faster on decode, and is still below the `100 tok/s` floor by `1.1872x`.
	  This is now encoded in the generation cache builder rather than requiring
	  that env var: with `GO_MLX_FIXED_GEMMA4_CACHE_SIZE` explicitly unset, the
	  same command derives a 2336-slot capacity from `prompt_tokens + max_tokens`
	  rounded to 32 and records `1935.3610403257746 tok/s` prefill and
	  `84.01009717307203 tok/s` decode. That is within `0.27%` of the manual
	  2336-slot sample and leaves same-prompt llama.cpp `1.0886x` faster on
	  decode. A follow-up tried restoring Gemma 4's 1024-token sliding-layer
	  cache capacity inside the fixed-cache lane. The native overflow updater is
	  now correct, but that per-layer cache shape regresses the same 3-run lane
	  to `73.05984177869179 tok/s` decode. The active path was restored to
	  uniform request-sized fixed caches and rerun at `83.59574625080806 tok/s`;
	  the earlier `84.01009717307203 tok/s` automatic sample remains the best
	  verified result.
	  A dynamic paged-cache control regresses to `50.412141409798174 tok/s`,
	  and the 2336-slot no-shared-mask control regresses to
	  `79.62987660090852 tok/s`, so the fast lane needs both fixed-cache graph
	  stability and the shared fixed mask. A diagnostic native-event
	  trace with forced intermediate materialisation is not a throughput result,
	  but it shows the remaining GPU work is distributed: attention `17.52%`,
	  local MLP `11.87%`, router `10.47%`, expert activation `10.25%`,
	  attention residual `8.98%`, expert down `8.81%`, and the rest across norm,
	  FFN residual, output, and bookkeeping buckets. A scale-hoist variant for
	  aligned q4 groups was also tested and rejected at `77.70903294390506
	  tok/s`, likely due to register pressure. Re-enabling the compiled Gemma 4
	  layer over the packed expert-ID path was also neutral-to-negative at
	  `78.78857639506562 tok/s`; the packed path stays faster without that gate,
	  and same-prompt llama.cpp still leads that compiled probe by `1.1607x`.
	  Re-enabling the compiled per-layer-input tensor gate was worse at
	  `77.0865964024348 tok/s`, so the remaining gap is not solved by the
	  existing per-layer-input compiled closure either. Rechecking the native
	  MLP GELU gate on the packed path was also slower at
	  `77.96201603724107 tok/s`. A single-token native router top-k/softmax
	  Metal kernel also failed the decode acceptance lane at
	  `83.54086813967548 tok/s`, even though it verified that fixed-cache prompt
	  restore drops repeated 2204-token prompt setup to about `4.7ms`.
	  The next stable C++ boundary moves fixed-cache owner attention into
	  `go_mlx_gemma4_fixed_owner_attention`: Q/K/V projection, Q/K RMSNorm,
	  RoPE, fixed-cache update, masked SDPA, and O projection now cross the
	  Go/native boundary as one gated call, with dense fallback coverage and a
	  q4 compiled branch for the active fixed-mask shape. Focused Metal tests
	  pass, but the 3-run README lane is effectively neutral: same-binary
	  gate-off
	  `docs/runtime/2026-05-18-go-mlx-gemma4-26b-a4b-q4-native-fixed-owner-attention-q4compiled-gateoff-3run-readme-llamacpp-comparison-longdecode.json`
	  records `84.59149676385168 tok/s`, while gate-on
	  `docs/runtime/2026-05-18-go-mlx-gemma4-26b-a4b-q4-native-fixed-owner-attention-q4compiled-3run-readme-llamacpp-comparison-longdecode.json`
	  records `84.75303439310541 tok/s`. Attention wrapping alone is therefore
	  not the remaining llama.cpp parity miss; the full one-token native
	  boundary remains open. A follow-up compiled residual-norm wrapper for
	  `residual + RMSNorm(attnOut)` is also rejected:
	  `docs/runtime/2026-05-18-go-mlx-gemma4-26b-a4b-q4-native-residual-norm-3run-readme-llamacpp-comparison-longdecode.json`
	  records `84.36852051087726 tok/s`, below the same-binary fixed-cache
	  control band. Combining the two ideas into
	  `GO_MLX_ENABLE_NATIVE_GEMMA4_FIXED_OWNER_ATTENTION_RESIDUAL=1` is also
	  rejected: the dense and q4 compiled Metal tests pass, but
	  `docs/runtime/2026-05-18-go-mlx-gemma4-26b-a4b-q4-native-fixed-owner-attention-residual-3run-readme-llamacpp-comparison-longdecode.json`
	  records only `84.4324627031718 tok/s`.
	  A follow-up extends the C++ `-native-gemma4-layer` ABI across the MoE
	  router, local MLP, routed expert projections, branch norms, per-layer input
	  gate/projection, and fixed-cache owner update. Focused Metal tests pass for
	  paged and fixed-cache MoE layer outputs, but the traced 26B README
	  prompt-file lane emits per-bucket `gemma4.layer.*` events rather than the
	  `native_layer` marker. The gate-set benchmark records
	  `85.02574071831692 tok/s` with empty stderr, so this remains ABI groundwork
	  until the production model satisfies the full-layer availability guard.
	  A model-level fixed-cache greedy follow-up then added a one-call C++ wrapper
	  with per-layer metadata, shared-KV routing, fixed masks, and final greedy
	  output projection. The first traced README lane did not emit the
	  `gemma4.model.greedy_token` marker because the gate set missed
	  `-native-gemma4-moe-layer`; after adding trace skip reasons, the real pack
	  showed another silent guard: `per-layer input metadata is incomplete`
	  with `got 0 want 30`. The production 26B A4B q4 pack has no per-layer input tensors, so
	  the wrapper now accepts nil per-layer inputs and passes nil per layer. The
	  corrected trace emits seven `gemma4.model.greedy_token` events over an
	  8-token run, proving the model-level wrapper fires. The throughput result is
	  negative: the full README 3-run lane records only `50.56636111604209 tok/s`
	  decode with empty stderr, so this broad one-call wrapper remains rejected
	  and the production lane stays on the faster packed expert-ID path.
- [x] Stop optimising an activation-only patch once the measured improvement is
  small; move to the next larger boundary instead. The disabled per-layer-input
  diagnostic correctly identified the side-input materialisation boundary, and
  the quantized embedding row-gather fix clears the E2B 100 tok/s floor. The
  next larger boundary is now llama.cpp parity, not another standalone
  activation wrapper, final output wrapper, isolated MLP sub-block wrapper,
  async scheduling tweak, or simple compiled closure around the old tensor
  construction.

Candidate native boundaries, in priority order. llama.cpp is the source to copy
for native graph, KV-cache shape, and benchmark comparison:

1. Close the 26B A4B q4/Q4_K_M llama.cpp decode and prefill gap using
   llama.cpp-style stable decode graph inputs and KV slotting. Sorted expert
   prefill cut the long-prefill gap from the old `2.4x` class to `1.14x`, and
	   multi-page fast concat plus expert-ID fused direct-greedy decode cut
	   the long-context decode miss from `2.94x` to about `1.82x`, so sustained decode
	   at real context length is now the
   highest-signal gap.
2. Full one-token layer block including attention, MLP, residual, and norm.
3. KV cache append/update and attention read path.
4. Output projection plus top-k/top-p/temperature sampling.
5. Batched multi-token prefill path for unavoidable new context, keeping the
   sorted expert route path as the current baseline.

## Workstream 4: Agentic State Lifecycle

**Purpose:** make project memory a durable runtime primitive, not a prompt
stuffing convention.

- [x] Seed project/operator context into a durable state entry. `SleepAgentMemory`
  streams session KV blocks, writes a bundle/index, and records model/tokenizer
  metadata in `TestAgentMemoryWakeSleep_Good`.
- [x] Wake the seed into a live session without replaying the whole seed text.
  `WakeAgentMemory` restores State KV blocks directly and the test generates
  from restored state without refeeding the seed prompt. The prompt-cache wake
  path also restores fixed-cache Gemma 4 generation buffers now, so the
  diagnostic fixed-cache decode lane can reuse durable KV state instead of
  falling back to a full prefix prefill. The router-topk probe run demonstrates
  the shape in a real driver profile: run 2/3 restored the 2204-token README
  prompt in about `4.7ms` instead of replaying the prefix through prefill. The
  follow-up 10-run agentic bench on the active lane recorded nine warm wakes at
  `4.674699ms` average and reduced repeated 2204-token prompt setup from a
  `10.567751250s` no-state estimate to `1.098864083s` actual over ten batches.
- [x] Append current task context and fresh repo observations. `AppendAndSleep`
  appends prompt material before persisting the child state, and the no-reply
	  test covers background observation appends. `ModelSession.PrefillChunks`,
	  `ModelSession.AppendPromptChunks`, `ModelSession.PrefillTokens`, and
	  `ModelSession.AppendTokens` now expose bounded and already-tokenised session
	  input APIs so agent workflows can seed or append large context without
	  rebuilding one giant prompt string or re-tokenising stored token segments;
	  `TestSessionPrefillChunks_Good`, `TestSessionAppendPromptChunks_Good`,
	  `TestSessionPrefillTokens_Good`, and `TestSessionAppendTokens_Good` cover the
	  root package surface, while native session chunk prefill/append reuses the same
	  chunked tokenisation path as `GenerateChunks`.
- [x] Sleep the updated session to a new state entry when exact continuation is
  wanted. The agent-memory test verifies parent/child entry metadata after
  append-and-sleep and generate-and-sleep.
- [x] Compact an exhausted live context into a folded state and continue from it.
  `Model.FoldAgentMemory` checkpoints the exhausted K/V state, prefills a fresh
  session from summary-plus-tail text, sleeps the folded State with parent
  lineage, then `TestFoldAgentMemory_CheckpointSummaryTail_Good` wakes the
  folded entry, appends the next turn without replaying the summary text, and
  generates from the restored folded State. The test now forces a multi-block
  folded State wake, and `kv.LoadPrefixTokensFromStateBlocksWithOptions` loads
  only token IDs for folded prefill so mixed block shapes cannot fail K/V
  assembly during compaction wake. `state-ramp-profile` exposes the same
  production handoff when an explicit fold store is supplied and the live state
  reaches the context exhaustion threshold: it writes the exhausted checkpoint
  and folded State, wakes the folded State with `restore_strategy=folded-prefill`,
  and records the optional folded wake/continue turn in the benchmark report.
- [x] Reuse the current seed plus text memory when the operator does not want a
  new state file. `TestProjectSeed_PlanContinuationModes_Good` verifies
  `ProjectSeedReuseCurrent` avoids a sleep request and keeps the current seed
  as the reusable text-memory anchor.
- [x] Fall back to summary-plus-new-window when model, tokenizer, adapter,
  quantisation, or context compatibility is unsafe.
  `TestWakeCompatibility_GoodBadUgly` now covers tokenizer, adapter, context,
  model hash/architecture, and quantisation blockers.
- [x] Smoke test a restored state by asking a question about retained content
  without including that content in the prompt. `TestAgentMemoryWakeSleep_Good`
  wakes retained KV state, appends a question that omits the retained answer
  text, and generates from the restored session.
- [x] Keep the no-reply workflow available: background agents may append
  findings and sleep state without producing a user-facing answer.
  `TestAppendAndSleepAgentMemory_NoReply_Good` asserts append-and-sleep does
  not call generation.

## Workstream 5: Discovery and Autotuning

**Purpose:** let users opt into a one-time local setup that finds good runtime
settings without requiring them to understand every model and hardware flag.

- [x] Keep machine discovery returning backend, Metal availability, device
  architecture, memory size, recommended working set, supported cache modes, and
  candidate model settings.
- [x] Keep tuning profiles serialisable and reloadable by `driver-profile`.
  `tune-run` writes `inference.TuningProfile` JSON, `tune-profile` decodes the
  same file without loading weights, and `driver-profile -profile` applies the
  saved candidate load settings before profiling. See
  `docs/runtime/local_autotune.md`.
- [x] Support model replacement quickly enough that the UI can test multiple
  local models and compare profiles. `replace-plan` compares two saved tuning
  profiles without loading weights and returns a portable `ModelReplacePlan`
  for state reuse, checkpoint, or summary-window fallback.
- [x] Report results in terms a non-expert can trust: correctness smoke result,
  load time, restore time, first-token time, steady tok/s, and memory pressure.
  Tuning measurements now carry load milliseconds, first-token milliseconds,
  restore milliseconds, decode tok/s, peak/active memory, and bench quality
  smoke pass/fail; saved profiles also copy the selected trust counters into
  UI-facing labels.
- [x] Never hide a slower profile behind a successful run. Persist the measured
  reason a profile won. `tune-run` now stores score, measurements, selection
  policy, selected score, successful/failed candidate counts, and runner-up
  score delta in the saved `TuningProfile` labels.

## Workstream 6: Model Coverage

**Purpose:** avoid locking the driver to the in-house Gemma path.

- [x] Keep Gemma 4 as the production lane. `DefaultProductionLane` pins the
  package-owned target to `mlx-community/gemma-4-e2b-it-4bit`,
  `gemma4_text`, q4, the retained-state prompt, 4096 context, 128 tokens,
  three runs, hidden output, and token-phase tracing; `TestProductionLane_DefaultGemma4E2B_Good`
  and `TestProductionLane_ArchitectureProfileNative_Good` guard that this lane
  stays native Gemma 4 chat/generation rather than drifting to a fallback.
- [x] Keep Qwen 2 and Qwen 3 loading and generating through the same public
  contracts. `TestRunSmallModelSmoke_GemmaQwenPublicContracts_Good` proves
  safe Gemma 4, Qwen 2, and Qwen 3 packs enter the same guarded `LoadModel`
  plus workload-bench generation path, while `TestPlanSmallModelSmoke_GemmaQwenCoverageMatrix_Good`
  keeps the metadata/load-shape planner shared across the three families.
- [x] Add Qwen 3.6 support with explicit config detection, tokenizer handling,
  layer shape handling, and smoke coverage. `TestInspectModelPack_Qwen36HybridMetadataOnly_Good`
  verifies Qwen 3.6 alias detection, text-config shape metadata, qwen chat
  template handling, quantisation metadata, and the explicit `mlx_lm` fallback
  boundary; `TestPlanSmallModelSmoke_Qwen36FallbackSkipsNativeLoad_Good`
  verifies the guarded native-load skip for the recognised fallback path.
- [x] Use the same driver-profile and state smoke tests across Gemma and Qwen
  where the model architecture allows it.
  `TestRunCommand_DriverProfileGemmaQwenMatrix_Good` exercises the same
  driver-profile command shape for Gemma 4, Qwen 2, and Qwen 3, while
  `TestPlanSmallModelSmoke_GemmaQwenCoverageMatrix_Good` verifies the same
  state-smoke planning path for the native-loadable Gemma/Qwen families.

## Workstream 7: Split and Power Path

**Purpose:** lower the device entry barrier for mobile and low-memory Apple
Silicon machines.

- [x] Keep split-execution APIs aligned with go-inference contracts.
  `TestInferenceContract_MetalBackendImplementsFitPlanner_Good`,
  `TestInferenceContract_MetalBackendPlanModelSlice_Good`, and
  `TestInferenceContract_MetalBackendPlanSplitInference_Good` assert that the
  metal backend implements the portable slice/split planner contracts.
- [x] Explore CPU weights plus GPU attention as the first local split target.
  `TestSplitExecutor_Generate_GoodRoutesAttentionAndFFNPerLayer`,
  `TestSplitExecutor_LoadSplitExecutor_GoodCPUFFNOptionMakesPlacementReady`,
  and the native split-local runtime tests cover the local Metal
  attention/logits side plus CPU FFN placement and memory reporting.
- [x] Measure memory, power, first-token time, and tok/s for split execution
  rather than judging it only by peak throughput. `SplitExecutor.Metrics`
  records prompt/generated token counts, first-token/prefill/decode timing,
  decode tok/s, Metal memory counters, CPU FFN residency, and optional power
  samples supplied through `WithSplitPowerMeter`; `TestSplitExecutor_Generate_GoodRecordsMetricsMemoryAndPower`
  verifies the measurement path without requiring a live Metal device.
- [x] Preserve the path for future network split execution, but optimise the
  local low-power split first. `NewRemoteSplitFFNExecutor`,
  `TestRemoteSplitFFNExecutor_ForwardFFN_Good`, and
  `TestSplitExecutor_Generate_GoodRoutesRemoteFFN` verify the HTTP FFN shard
  contract and the split executor's remote FFN routing while keeping the
  existing local split path first-class.
- [x] Preserve the research query path for comparing base and fine-tuned model
  weights so training deltas can be inspected rather than guessed.
  `merge.ComparePacks`, `TestComparePacks_BaseFineTunedSafetensors_Good`,
  `TestComparePacks_RequiresSafetensorsPacks_Bad`, and
  `TestComparePacks_ReportsShapeMismatch_Ugly` provide a chunked safetensors
  delta report with aggregate and per-tensor metrics.

## Workstream 8: Training-Pipeline Enablement

**Purpose:** unblock the lthn/desktop autocratic-cascade Phase A training loop
against go-mlx's exported training surface. The downstream chain (corpus
reader, sandwich builder, R₁ store, CL-BPL envelope detector, training
orchestrator, training-window UI) shipped 2026-05-20 in lthn/desktop. The
remaining bottleneck is on this side: training types and a `Runner`
implementation that the orchestrator can drive.

### Gemma 4 architecture and training audit (2026-05-20)

10 of 12 IDEAS.md architectural/training items are now resolved in Go:
hybrid 5:1 attention (`gemma4.go:631-637`), sliding window size config
(`gemma4.go:587`), dual RoPE bases 10k/1M (`defaultGemma4RopeParameters`),
cross-layer KV sharing (`sharedKV` + `CacheIndexByLayer`), per-layer
embeddings via `mlx_take`, MoE top-2 sparse routing
(`gemma4_router_topk.go`), PLE gradient isolation through the Gemma 4 LoRA
safe-target policy and opt-in extended-target guard tests, final-cache K=V
rejection with a guard test, packed AdamW moment
state for homogeneous matrix parameters, and Gemma4 assistant drafter +
speculative decode (`gemma4_assistant*.go`).

- [x] Record the updated IDEAS.md architecture/training audit in
      `docs/runtime/2026-05-20-gemma4-ideas-architecture-audit.md`.
- [x] Confirm p-RoPE is covered by the mlx-c side. Go precomputes the
      proportional frequency array and MLX's Metal RoPE kernels use the
      `rope_*freqs*` path when that array is supplied.
- [x] Confirm RMSNorm kernel semantics. The native kernel multiplies the
      supplied scale directly; Gemma 4 currently precomputes direct scale and
      has a test protecting that convention. Do not add `(1 + weight)` until
      the MLX-community Gemma 4 weight convention proves it is zero-centred.
- [x] Confirm the C++23/pinned-byte bridge baseline. The repo-local native
      build requires C++23, and the pinned raw byte bridge already uses
      `runtime.Pinner`, `std::mdspan`, and `mlx_array_new_data_managed_payload`.
- [x] Explicitly reject unified K=V/global-layer final cache storage.
      `attention_k_eq_v` shares the projection source with a ref-counted MLX
      handle, but final K and V diverge because K takes KNorm+RoPE while V
      takes value RMSNorm. `TestGemma4_AttentionKEqVDoesNotAliasFinalCache_Good`
      guards that final snapshot/restore state must keep separate key/value
      arrays unless a future raw-projection state format chooses to recompute
      final K/V on restore.
- [x] Implement packed AdamW moment state for LoRA-style matrix parameters.
      `DefaultAdamWConfig` enables packed state by default; homogeneous
      same-dtype parameter layouts keep `m`/`v` in contiguous MLX slabs with
      shaped views for the existing update math, while scalar/mixed-dtype
      parameters fall back to the prior per-parameter state. Guard coverage:
      `TestOptim_AdamW_PacksHomogeneousMatrixMoments_Good`,
      `TestOptim_AdamW_PackedStateCanBeDisabled_Bad`,
      `TestOptim_AdamW_PackedStateFallsBackForMixedDTypes_Ugly`, and
      `TestSFTAdamWConfig_UsesExplicitOptimizer_Bad`.
- [x] Design the LoRA State timeline after one real native LoRA runner step
      works end-to-end.
      The latest `IDEAS.md` addendum turns this into the next training-state
      design target, not an immediate bridge rewrite. The real-step proof now
      lives in `TestSFTNativeSmoke_OneLoRAStep_Good`, which loads the local
      `mlx-community/gemma-4-e2b-it-4bit` snapshot, runs one rank-2 `q_proj`
      LoRA SFT step, and verifies one finite-loss adapter update. Verified with:

      ```sh
      env GO_MLX_SFT_SMOKE_MODEL=/Users/snider/.cache/huggingface/hub/models--mlx-community--gemma-4-e2b-it-4bit/snapshots/99d9a53ff828d365a8ecae538e45f80a08d612cd \
        MLX_METALLIB_PATH=/Users/snider/Code/core/go-mlx/dist/lib/mlx.metallib \
        GOCACHE=/private/tmp/go-mlx-gocache \
        go test ./go -run TestSFTNativeSmoke_OneLoRAStep_Good -count=1 -v -timeout=10m
      ```

      Result: `ok dappco.re/go/mlx`, `PASS`,
      `TestSFTNativeSmoke_OneLoRAStep_Good` in `1.72s`. The resulting design is
      documented in `docs/training/lora_state_timeline.md`: append-only State
      manifest plus full post-step frames for LoRA A/B and AdamW m/v, with PLE
      kept static and rollback done by moving the active step pointer.
- [x] Defer MTP drafter co-training until target-model SFT is stable.
      This is not implemented in the production training path. MTP remains a
      valid decode-boost lane: llama.cpp already shows the upside, while the
      current native go-mlx assistant loop is still slower than target-only on
      the same short prompt. Keep MTP optimisation alive for decode, but do not
      co-train a drafter until target-model SFT is stable enough that the
      drafter has the right behaviour to imitate.

### Training types export

- [x] Map the current public training surface from `go-mlx/go` for downstream
      use. The root package already exports `LoRAConfig`, `LoRAAdapter`,
      `AdamW`, `AdamWConfig`, `Cache`, `Array`, `TrainingModel`,
      `Model.Tokenizer`, `NewLoRA`, and `Model.TrainSFT`; the internal model
      returned by `TrainingModel` exposes `Forward`, `NewCache`, `Tokenizer`,
      and `ApplyLoRA`.
- [x] Compile the lthn/desktop `gomlxrunner` against that surface and add only
      the thin wrapper names that the adapter proves necessary. A top-level
      `Tokenizer(model)` function is not available as named because the package
      already owns the exported `Tokenizer` type; prefer `Model.Tokenizer()`
      unless the downstream interface forces a different accessor name. Verified
      from `lthn/desktop` with:

      ```sh
      env GOWORK=/Users/snider/Code/lthn/desktop/go.work \
        GOCACHE=/private/tmp/codex-lthn-desktop-cache \
        MLX_METALLIB_PATH=/Users/snider/Code/core/go-mlx/dist/lib/mlx.metallib \
        CGO_CPPFLAGS=-I/Users/snider/Code/core/go-mlx/dist/include/metal_cpp \
        go test ./go/pkg/gomlxrunner ./go/pkg/training -count=1
      ```

      Result: `ok dappco.re/lthn/desktop/pkg/gomlxrunner` and
      `ok dappco.re/lthn/desktop/pkg/training`. The downstream workspace needs
      `external/mlx` at `1cefb03` and `external/inference` at `f0af335`; the
      compile uses the go-mlx Metal-cpp include directory until desktop's
      external/mlx checkout grows its own generated `dist/include/metal_cpp`
      artefact.
- [x] Tag a release version that the lthn/desktop go.mod can pin against,
      or wire workspace-mode build path so lthn/desktop picks up the export
      via `external/`. The active path is workspace mode:
      `lthn/desktop/go.work` includes `./external/mlx/go`, and
      `go/go.mod` requires `dappco.re/go/mlx v0.10.0` while resolving the live
      external during development.

### `gomlxrunner` adapter — the single concrete handoff

- [x] Build `gomlxrunner` as a thin Go package implementing the
      `training.Runner` interface from
      `dappco.re/lthn/desktop/pkg/training`. Live target likely
      `lthn/desktop/go/pkg/gomlxrunner/` so it depends on go-mlx but not the
      other way round. Required methods (signatures already locked in
      lthn/desktop):

      ```go
      type Runner interface {
          StepBatch(prompt, target string) core.Result // wraps Forward + LoRA grad step, returns loss
          GenerateResponse(prompt string) core.Result  // single-turn inference, returns text
          ModelID() string                              // canonical ID per production_lane.go
          Substrate() string                            // "CONT" or "TRAD"
          Tier() int                                    // 0..3 cascade tier
      }
      ```

      The package now provides `Config`, `New`, `NewFromModel`, `StepBatch`,
      `GenerateResponse`, `ModelID`, `Substrate`, `Tier`, and `Close`. It uses
      `Model.Tokenizer()`, `BuildSFTBatches`, `NewLoRA`, `AdamW`, and
      `Model.Generate` without adding root-package wrapper names to go-mlx.
- [x] Substrate switch on the runner. CONT is the production-default (KV
      mount, no re-prefill, matches the 2026-05-20 c006 corrected-window
      run). TRAD is the comparison condition (full re-prefill per turn). The
      substrate-shift experiment in `host-uk/core/plans/rfc/research/experiments/worf/`
      requires both conditions; both must produce identical token output
      under identical seeds when the model weights are unchanged.

      Mechanical switch progress: go-mlx now exposes `Model.ClearPromptCache()`
      so a preloaded runner can force a fresh prefill without unloading weights.
      The downstream `gomlxrunner` normalises `cont`/`trad`, appends
      `mlx.WithPromptCache(false)` for TRAD loads, and clears prompt cache
      before TRAD `GenerateResponse` calls. Verification from `lthn/desktop`
      after fast-forwarding `external/mlx` to `89d2dfb`:

      ```sh
      env GOWORK=/Users/snider/Code/lthn/desktop/go.work \
        GOCACHE=/private/tmp/codex-lthn-desktop-cache \
        MLX_METALLIB_PATH=/Users/snider/Code/core/go-mlx/dist/lib/mlx.metallib \
        CGO_CPPFLAGS=-I/Users/snider/Code/core/go-mlx/dist/include/metal_cpp \
        go test ./go/pkg/gomlxrunner ./go/pkg/training -count=1
      ```

      Real-model parity proof: `TestSubstrateParity_PromptCacheReplay_Good`
      runs only when `GO_MLX_SUBSTRATE_PARITY_MODEL` points at a local model
      pack. Against
      `mlx-community/gemma-4-e2b-it-4bit` snapshot
      `99d9a53ff828d365a8ecae538e45f80a08d612cd`, a cache miss, prompt-cache
      hit, and forced replay produced identical chat output under
      `WithSeed(42)`.

      ```sh
      env GO_MLX_SUBSTRATE_PARITY_MODEL=/Users/snider/.cache/huggingface/hub/models--mlx-community--gemma-4-e2b-it-4bit/snapshots/99d9a53ff828d365a8ecae538e45f80a08d612cd \
        MLX_METALLIB_PATH=/Users/snider/Code/core/go-mlx/dist/lib/mlx.metallib \
        GOCACHE=/private/tmp/go-mlx-gocache \
        go test ./go -run TestSubstrateParity_PromptCacheReplay_Good -count=1 -v -timeout=10m
      ```

      Result: `ok dappco.re/go/mlx`, `PASS`,
      `TestSubstrateParity_PromptCacheReplay_Good` in `3.25s`.

      Seed-control progress: go-mlx now exposes `SeedRandom(seed)` for
      run-level MLX RNG seeding plus `WithSeed(seed)` for single-call
      generation. The option forwards through the root API into the native
      `metal.GenerateConfig`, and native generation/session/batch paths call
      `mlx_random_seed` before sampling when it is set. Guard coverage:
      `TestRandom_SeedRandom_Good`, `TestModelGenerateStream_ForwardsOptions_Good`,
      and `TestAPIGenerateOptions_Good`.

      Condition-contract progress: `go/substrate` now defines the four
      pre-registered method conditions (`TRAD`, `CONT`, `TRAD-no-replay`,
      `CONT-with-gap`) plus canonical transition semantics for replay,
      retained-state use, artificial prefill gaps, and T_prefill measurement.
      Guard coverage: `TestCondition_Normalize_Good`,
      `TestCondition_TransitionSemantics_Good`, and AX-11 benchmarks
      `BenchmarkNormalize_ConditionAlias` (`12.63 ns/op`, `0 allocs`) and
      `BenchmarkConditionTransition_FourConditions` (`7.933 ns/op`, `0 allocs`).

      Downstream adapter progress: `lthn/desktop` `external/mlx` now
      fast-forwards to go-mlx `23c431a` and `external/inference` to
      `6cb95d7`. `go/pkg/gomlxrunner` imports `dappco.re/go/mlx/substrate`,
      exposes all four canonical labels, forwards `Config{Seed, SeedSet}` to
      `mlx.WithSeed`, keeps TRAD as the only prompt-cache replay condition, and
      uses `Config.PrefillGap` for artificial-gap controls. Verified with:

      ```sh
      env GOWORK=/Users/snider/Code/lthn/desktop/go.work \
        GOCACHE=/private/tmp/codex-lthn-desktop-cache \
        MLX_METALLIB_PATH=/Users/snider/Code/core/go-mlx/dist/lib/mlx.metallib \
        CGO_CPPFLAGS=-I/Users/snider/Code/core/go-mlx/dist/include/metal_cpp \
        go test ./go/pkg/gomlxrunner ./go/pkg/training -count=1
      ```

      Result: `ok dappco.re/lthn/desktop/pkg/gomlxrunner` and
      `ok dappco.re/lthn/desktop/pkg/training`.

### Per-turn capture for the substrate-shift experiment

- [x] A 180-run capture script (Go or Python) that wraps the Runner and
      produces the per-run JSONL the `stats.py` analyser expects:

      ```
      header line:  {"type":"run_meta", subject, probe, condition, seed, model, timestamp}
      10 turn rows: {"type":"turn", turn, text, features:{11 keys}, self_ref_count,
                     terminal_count, timing_ms, kv_norm}
      ```

      Format pinned in `host-uk/core/plans/rfc/research/experiments/worf/02-method.md` §6.
      Output tree at `~/Lethean/data/experiments/substrate-shift/<subject>/<probe>/<condition>/<seed>.jsonl`.
      `scripts/substrate_shift_capture.py` now owns the default 180-run matrix,
      reads the three subject seed corpora, emits the 11 feature keys,
      `self_ref_count`, `terminal_count`, `timing_ms`, and `kv_norm`, and
      delegates actual generation to a JSON stdin/stdout runner command.
      Verification:

      ```sh
      scripts/substrate_shift_capture.py --dry-run \
        --out-dir /private/tmp/go-mlx-substrate-capture-full-dryrun-20260521 \
        --overwrite
      find /private/tmp/go-mlx-substrate-capture-full-dryrun-20260521 \
        -name '*.jsonl' | wc -l
      python3 /Users/snider/Code/host-uk/core/plans/rfc/research/experiments/worf/scripts/stats.py \
        --data-dir /private/tmp/go-mlx-substrate-capture-full-dryrun-20260521 \
        --out /private/tmp/go-mlx-substrate-capture-full-dryrun-20260521-results.json
      ```

      Result: `180` JSONL files; `stats.py` loaded all `180` runs. This closes
      the capture-script deliverable only. Actual model data capture still
      depends on the open runner substrate-switch parity/control-condition item.

### Downstream chain (already shipped in lthn/desktop, no work here)

When the items above land, the full cascade fires without further changes
to lthn/desktop. For confidence:

- `pkg/seeds` — Hypnos corpus reader, 13 tests green
- `pkg/sandwich` — LEK-1 builder with SHA-256 pinned digest, 8 tests green
- `pkg/r1` — append-only JSONL corpus with `AtomicAppendLineLarge` write path,
  Tier + MaxTier filter for cascade reads, Wails surface, 40 tests green
- `pkg/clbpl` — envelope detector with `core.Mutex`-guarded WailsService,
  race-clean, 32 tests green
- `pkg/contentshield` — non-LLM tier-1 scoring (sycophancy + grammar imprint
  + differential + authority), 79 tests green
- `pkg/training` — Service + Runner interface + FixtureRunner + Phase A loop
  + ctx-cancellable Run + per-Service Mutex guard, 9 tests + 1 example
- `frontend/src/lit/ext/training-window.ts` — operator UI with fixture data
  shaped to match `pkg/r1` + `pkg/clbpl` surfaces, 8 vitest green
- `RFC.fork-tree.md` — Phase A rotation order locked (english → european →
  latam → russian → middle-east → chinese → african)

The lthn/desktop side is gated only on (a) the training types export, (b)
the `gomlxrunner` adapter, and (c) the substrate switch. Three small pieces
on this side unlock the entire Phase A training pipeline downstream.

## Verification Commands

Run these before claiming a production-gate candidate is ready for review:

```bash
cd /Users/snider/Code/core/go-mlx
env GOCACHE=/private/tmp/codex-go-mlx-cache MLX_METALLIB_PATH=/Users/snider/Code/core/go-mlx/dist/lib/mlx.metallib go test ./go/... -count=1
```

```bash
cd /Users/snider/Code/core/go-mlx
env GOCACHE=/private/tmp/codex-go-mlx-cache MLX_METALLIB_PATH=/Users/snider/Code/core/go-mlx/dist/lib/mlx.metallib go build -trimpath -o bin/lthn-mlx ./go/cmd/mlx
```

```bash
cd /Users/snider/Code/core/go-mlx
git diff --check
```

For performance claims, also run a `driver-profile` command with JSON output and
save the result under `docs/runtime/`.

## Production-Ready Means

This is the handoff gate, not a description of the current state:

- `bin/lthn-mlx` builds reproducibly from the workspace-aware command above.
- The agentic memory lifecycle works without prompt-prefilling retained source
  text, and the 10+ turn retained-state path is measured against replayed
  prefill.
- The accepted workload uses realistic output budgets: long chapter/workflow
  turns, not `max_tokens=8`, `32`, or `128` smoke-only shortcuts.
- go-mlx is the best practical runner for the target repeated agentic workflow,
  or any faster external runner has a documented command, version, metric gap,
  and next native boundary to attack.
- The old `>= 100 tok/s` round-number floor is retired only after go-mlx beats
  configured `mlx_lm`/vLLM style runners on the realistic workflow, or after a
  report proves raw decode is close enough and retained-state wall-clock wins
  decisively over a 10+ turn flow, including estimated energy saved when a
  wattage assumption is supplied.
- Long-context memory use stays bounded for the small-model lane; a 5 GB model
  must not reserve or report hundreds of GB during the accepted workflow.
- Tests, build, diff hygiene, benchmark artefacts, and state smoke evidence are
  all present in the repo.

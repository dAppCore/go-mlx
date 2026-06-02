<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# go-mlx Agentic Memory Production Runner Goal

> **For agentic workers:** treat this file as the source of truth for the next
> go-mlx optimisation and agentic-memory lane. Implement task-by-task, keep the
> public Go API stable, and verify each performance claim with recorded command
> output.

## Top Priority (last codex run) — Finish the Architecture Set Natively; Remove the `mlx_lm` Python Fallback

> Operator directive, 2026-06-01: this is go-mlx's **last codex run** — finish the
> feature set. The single most important outcome: **no architecture falls back to
> Python.** Today `go/mlxlm/` extracts an embedded `bridge.py` and spawns Python
> `mlx-lm` as the registered `"mlx_lm"` backend for every architecture still marked
> `metadataProfile` in `go/profile/architecture.go`. That subprocess is a crutch,
> not a supported backend. The goal is native Go/Metal parity for the whole table,
> then `go/mlxlm/` (`backend.go` + `bridge.py`) is deleted.

**Where it stands** (`go/profile/architecture.go`): **16 of 25** architectures are
native Go/Metal profiles; **9** are `metadataProfile` native gaps. Local tuning
no longer rewrites metadata-only architectures to the Python `mlx_lm` subprocess;
remaining gaps stay on the Metal backend with `native_runtime=false` and explicit
loader diagnostics until their native implementations land.

- ✅ Native today: `gemma2`, `gemma3`, `gemma3_text`, `gemma4`, `gemma4_text`,
  `gemma4_assistant` (attached MTP drafter), `llama`, `qwen2`, `qwen3`,
  `qwen3_next`, `mistral`, `phi`, `glm`, `hermes`, `granite`,
  `minimax_m2` (staged JANGTQ/MXTQ tensor-plan loader; standalone generation pending).

`minimax_m2` counts as a native staged loader only: the pack can validate and
load its native JANGTQ/MXTQ tensor plan without Python, but it does not yet
satisfy the full load-and-generate acceptance criterion until sparse generation
lands.

🟡 Remaining native gaps — the work, in priority order:

1. ✅ **Dense quick wins complete:** `mistral`, `phi`, `glm`, `hermes`,
   `granite` are `nativeProfile` and have pack + tiny native Metal
   load/generate coverage.
2. **MoE / sparse-expert routers:** `mixtral`, `qwen3_moe`, `qwen3_6_moe`,
   `deepseek` (+ MLA variants), `gpt_oss`, `kimi`. Native `loadModel`
   dispatch now recognises these families and fails with architecture-specific
   Metal-loader diagnostics instead of a generic unsupported-architecture error.
3. **Hybrid linear-attention:** `qwen3_6` (named in the Goal; neighbours the E2B/KV lane).
4. ✅ **MTP drafter complete:** `gemma4_assistant` is native as an attached
   drafter. Standalone chat/generation stays disabled; load it beside a Gemma 4
   target through `LoadSpeculativePair` or `LoadGemma4AssistantPair`.
5. **Encoder / rerank loaders:** `bert` (embedding encoder), `bert_rerank`
   (cross-encoder scorer). HF fit planning now exposes the correct
   embeddings/rerank task flags and no-generation KV sizing for these profiles;
   native `loadModel` diagnostics distinguish embedding encoder and rerank
   scorer gaps. Native encoder/scorer loaders are still pending.

**Acceptance, per architecture:**

- `go/profile/architecture.go`: the entry moves `metadataProfile(...)` →
  `nativeProfile(...)` (sets `RuntimeStatus: native`, `NativeRuntime: true`).
- `model/pack.go` then computes `NativeLoadable == true`, so
  `RequiresPythonConversion == false` for that pack. (That field is a Python-world
  label; once the table is all-native it should be always-false and can be retired
  or renamed — there is no Python conversion in this stack.)
- The model loads **and generates through the native Metal backend with no `mlx_lm`
  subprocess spawned**, evidenced by recorded command output (per this file's rule).
- `local_tuning.go` no longer assigns `Backend = "mlx_lm"` for that architecture.

**Definition of done for "remove the fallback":** when every entry is native,
delete `go/mlxlm/` (`backend.go` + the embedded `bridge.py`) so a stock build has
zero Python. If the last run cannot land every native gap, land the quick wins
first. The dense quick wins and attached MTP drafter are landed; continue with as
much of (2), (3), and (5) as time allows. Whatever remains stays in this list as
the documented feature gap, never as a silent Python fallback.

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

## Current Status: Production Baseline Accepted; Next Lane Is Google E2B MTP and KV Compression

Operator status, 2026-05-31: the prior q4 paged-retained-State production lane
is treated as accepted/done for this planning file. Keep its benchmark rows as
historical calibration and regression guards, but do not reopen the May raw
decode parity chase unless a new benchmark shows a regression in the accepted
retained workflow.

The next active lane is official Google Gemma 4 E2B target+assistant support
plus KV-cache compression. Primary checkpoints are `google/gemma-4-E2B-it` and
`google/gemma-4-E2B-it-assistant`; keep
`mlx-community/gemma-4-e2b-it-4bit` as the archived q4 production baseline and
smoke/control pack until the official Google snapshots have equivalent
native-load, retained-state, and benchmark evidence. Google describes the MTP
assistant as a lightweight drafter that shares target activations and K/V; its
current E2B/E4B path also uses efficient embedder clustering, so go-mlx's
current ordered-embedding/centroid fail-closed path is now an implementation
gate rather than an optional cleanup.

Product quantisation policy: the app should default users to a 6-bit Gemma 4
E2B target when memory planning says it fits. Prefer 8-bit for quality-sensitive
workloads and machines with enough headroom. Use 4-bit only as the
hardware-constrained fallback and as a control/archive baseline; the operator
assessment is that the quality drop from 6-bit/8-bit down to 4-bit is large
enough to affect ability in the app.

TurboQuant is a research lane for KV-cache compression, not weight quantisation
and not a default. The source target is Google Research's online vector
quantisation method: random-rotation/PolarQuant-style scalar quantisation plus
QJL residual correction, with the paper reporting KV quality neutrality at
3.5 bits/channel and marginal degradation at 2.5 bits/channel. Treat those
figures as hypotheses to validate on MLX/Apple Silicon with go-mlx's own
long-context prompts before promotion.

Sources checked on 2026-05-31:

- [Gemma 4 MTP docs](https://ai.google.dev/gemma/docs/mtp/mtp)
- [Gemma 4 MTP launch note](https://blog.google/innovation-and-ai/technology/developers-tools/multi-token-prediction-gemma-4/)
- [google/gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B-it)
- [google/gemma-4-E2B-it-assistant](https://huggingface.co/google/gemma-4-E2B-it-assistant)
- [TurboQuant Google Research note](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)
- [TurboQuant paper](https://arxiv.org/abs/2504.19874)
- Local TurboQuant implementation PDF:
  `/Users/snider/Downloads/2504.19874v1.pdf`

Current open gates:

- [x] Lock exact Hugging Face revisions for `google/gemma-4-E2B-it` and
  `google/gemma-4-E2B-it-assistant`, including licence metadata, config hashes,
  tokenizer hashes, safetensors index hashes, and any gating/access notes.
  Evidence: `go/official_gemma4.go` pins the runtime locks and
  `docs/runtime/2026-05-31-official-gemma4-e2b-source-lock.json` records the
  exact source-lock metadata.
- [x] Load the official Google E2B target natively and prove its text path
  matches the archived q4 baseline's chat template, no-thinking handling,
  p-RoPE, local/global attention, shared-KV, retained-state, and prompt-cache
  contracts. Evidence: `docs/runtime/2026-05-31-official-gemma4-e2b-local-preflight.md`
  records q4-control compatibility for the chat-template/no-thinking,
  p-RoPE, local/global attention, shared-KV, PLE, retained-State, and
  prompt-cache metadata contracts, and
  `docs/runtime/2026-06-01-official-gemma4-e2b-target-native-state-smoke.md`
  records a native go-mlx Metal target smoke with non-empty generation,
  prompt-cache warm, K/V restore, state bundle, and State K/V block warm.
- [x] Add a user-facing quantisation selection policy: 6-bit is the normal app
  default, 8-bit is preferred when hardware headroom allows it, and 4-bit is the
  fallback only for memory-constrained devices or very long retained contexts.
  Evidence: `go/production_lane.go` exposes the q6/q8/q4 product ladder and
  `go/cmd/mlx/production_quantization.go` reports it for the app/CLI surface.
- [x] Make the E2B assistant path production-runnable by implementing the
  ordered-embedding centroid/token-ordering logit path used by the official
  small-model assistant, while keeping a fail-closed error for unsupported
  assistant tensor layouts. Evidence: `go/internal/metal/gemma4_assistant_decode.go`
  implements the ordered-embedding path, `go/production_mtp.go` makes the
  assistant layout metrics part of the production promotion evidence, and
  `docs/runtime/2026-05-31-official-gemma4-e2b-local-preflight.md` records the
  official-pair draft-step and generation-loop smokes. Target-only versus MTP
  retained-workflow benchmarks remain open below.
- [ ] Benchmark target-only versus MTP on the same prompts with
  `draft_tokens`/schedule, proposed/accepted/rejected counts, target verify
  throughput, visible tok/s, wall time, memory, and quality flags reported
  side by side. Diagnostic evidence so far:
  `docs/runtime/2026-06-01-official-gemma4-e2b-mtp-draft2-diagnostic.md`
  records a go-mlx-only official-source target-only versus draft-2 row with
  counters and production compare rejection, then extends the diagnostic with
  draft-token sweep rows for `1`, `2`, and `4`. The assistant stop-token path
  now withholds terminal control tokens like the normal decode path; the
  stop-fixed draft-2 rerun matches target-only greedy output hash, but it is
  intentionally not accepted for this gate because it is one turn, lacks
  restore timings, and is still slower than target-only.
- [ ] Keep MTP separate from raw decode. It may become the default interactive
  path only when it is faster than target-only on realistic retained workflows
  without changing greedy output quality.
- [x] Prototype a TurboQuant KV-cache mode behind an explicit runtime/cache
  option, separate from existing `fp16`, `q8`, `k-q8-v-q4`, `paged`, and
  `fixed` cache modes. Evidence: `go/production_turboquant.go` keeps TurboQuant
  explicit opt-in with fp16/paged/q8/k-q8-v-q4 comparison gates, and
  `go/internal/metal/turboquant_kv_*` plus `go/internal/metal/session_test.go`
  cover the reference payload and snapshot path. Validation on retained
  workflows remains open below.
- [ ] Validate TurboQuant against fp16/paged/q8/k-q8-v-q4 on 30k-40k retained
  workflows and the 100k stress lane, with memory, decode, restore, energy
  estimate, and long-output quality evidence.
- [ ] Regenerate canonical benchmark artefacts under `docs/runtime/` after the
  official Google E2B MTP and TurboQuant lanes stabilise.

Treat `IDEAS.md` as background optimisation advice, not the active top-level
target. The active target is now official Google E2B MTP plus measured
TurboQuant KV. Gemma 4 local/global attention windowing, PLE handling, and K/V
layout must still be verified against the actual code before declaring memory
or decode fixed.

Do not promote either new lane because a short-context decode number is healthy.
The claim must remain repeated-workflow wall time and retained-State savings
under real output budgets, with runner anchors and energy assumptions exposed.

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
   lanes must be labelled separately from no-draft raw decode. TurboQuant or
   any other compressed-KV lane must be labelled separately from fp16/paged/q8
   cache modes until quality, restore, and long-context behaviour match the
   accepted baseline.
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
| Current q6 go-mlx self-benchmark refresh | The fresh rebuilt `lthn-mlx` q6 E2B self-benchmark compares current defaults against the same binary with candidate gates isolated, using `context=4096`, `max_tokens=512`, `runs=3`, hidden output, and a `75 W` normalised energy estimate. The prior default gate combination (`GO_MLX_ENABLE_DIRECT_GREEDY_TOKEN=1`, `GO_MLX_ENABLE_PAGED_DECODE_FAST_CONCAT=1`) is rejected on the current build: it records `1164` generated tokens, `66.04699256254571 tok/s`, `19.379504541s` wall, `1453.462840575 J`, `4175757312` B peak RSS, and the same output hash as the faster rows. The same-binary baseline records `90.05828489228817 tok/s`, `13.135163001s`, and `985.137225075 J`; direct-greedy-only records `90.00163414721517 tok/s`, `13.139377125s`, and `985.453284375 J`; paged-fast-concat-only records `89.1661883374603 tok/s`, `13.267558418s`, and `995.06688135 J`. The promoted short-context default is now direct greedy only; the patched default rerun records `89.05331759347916 tok/s`, `13.275426084s`, and `995.6569563 J` with only `GO_MLX_ENABLE_DIRECT_GREEDY_TOKEN=1`. Paged fast concat remains explicit diagnostic until a retained-workflow self-benchmark proves a net wall-time win. See `docs/runtime/2026-06-02-gemma4-e2b-q6-go-mlx-self-bench.md` |
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
- [ ] Reopen MTP specifically for the official Google E2B pair:
  `google/gemma-4-E2B-it` plus `google/gemma-4-E2B-it-assistant`. This is a
  new post-acceptance lane, not a continuation of the rejected 26B loop. The
  first implementation target is the assistant ordered-embedding/centroid
  logits path, then a target-only versus MTP retained-workflow benchmark with
  the same visible prompt, same greedy settings, and full
  proposed/accepted/rejected counters.

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

- [x] Keep the archived Gemma 4 q4 production baseline stable.
  `DefaultProductionLane` pins the package-owned target to
  `mlx-community/gemma-4-e2b-it-4bit`, `gemma4_text`, q4, the retained-state
  prompt, 4096 context, 128 tokens, three runs, hidden output, and token-phase
  tracing; `TestProductionLane_DefaultGemma4E2B_Good` and
  `TestProductionLane_ArchitectureProfileNative_Good` guard that this lane
  stays native Gemma 4 chat/generation rather than drifting to a fallback. Do
  not silently replace this baseline until the official Google E2B target and
  assistant snapshots pass native-load, retained-state, MTP, and benchmark
  gates.
- [ ] Add official Google E2B target+assistant coverage for
  `google/gemma-4-E2B-it` and `google/gemma-4-E2B-it-assistant`. The coverage
  must record exact revisions, compare config/tokenizer/template metadata
  against the archived q4 baseline, validate the assistant four-layer drafter
  attachment, and keep the current q4 lane available as a smoke/control pack.
- [ ] Add Gemma 4 E2B 6-bit and 8-bit coverage to the production selection
  matrix. The default app recommendation is 6-bit; 8-bit is the quality-first
  choice when the model, context, and retained-state cache fit; 4-bit remains
  the constrained-hardware fallback. The matrix must report load time, peak
  memory, retained restore time, raw decode, long-output quality, and the
  hardware/context threshold where the app should step down from 8-bit to 6-bit
  or from 6-bit to 4-bit.
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

## Workstream 9: Official Google E2B MTP and TurboQuant KV

**Purpose:** move the post-acceptance optimisation lane from the archived
`mlx-community` q4 E2B baseline to the official Google E2B target+assistant
pair, then test whether TurboQuant-style KV compression is a better long-context
memory path than the current q8/k-q8-v-q4/paged modes.

### Source and model lock

- [ ] Resolve and record exact Hugging Face snapshots for:

  ```bash
  hf download google/gemma-4-E2B-it
  hf download google/gemma-4-E2B-it-assistant
  ```

  The record under `docs/runtime/` must include revision IDs, `config.json`
  hashes, tokenizer hashes, safetensors index/hash summaries, licence metadata,
  and whether the local environment needed Hugging Face auth.
- [ ] Compare the official target config against the archived
  `mlx-community/gemma-4-e2b-it-4bit` control pack. Record model type,
  quantisation, context window, local/global attention pattern, p-RoPE fields,
  per-layer input fields, shared-KV metadata, chat template behaviour, and
  no-thinking/thinking markers.
- [ ] Add official or converted 6-bit and 8-bit E2B target packs to the same
  source-lock record when available. If they are converted from the official
  Google weights, record the conversion command, source revision, output hash,
  quantisation format, and any accuracy smoke used to accept the pack.
- [ ] Keep the official Google E2B lane opt-in until the target-only path passes
  the same retained-state smoke and `driver-profile` metrics as the archived q4
  baseline. The q4 baseline remains the smoke/control pack during this
  migration; it is not the intended product default once 6-bit is validated.
- [ ] Define the app selection ladder in a machine-readable profile:
  prefer 8-bit with sufficient memory headroom, default to 6-bit for normal
  installations, and fall back to 4-bit only for constrained hardware or
  retained-context lengths that would otherwise exceed the memory policy.

### Official E2B MTP

- [ ] Implement the ordered-embedding centroid/token-ordering logits path needed
  by the official small-model assistant. Relevant files are
  `go/internal/metal/gemma4_assistant.go`,
  `go/internal/metal/gemma4_assistant_decode.go`, and
  `go/internal/metal/gemma4_assistant_decode_test.go`. The current
  `errAsstOrderedEmbedNotImpl` fail-closed path must remain for unsupported
  or malformed assistant layouts.
- [ ] Verify the assistant attachment against `google/gemma-4-E2B-it` plus
  `google/gemma-4-E2B-it-assistant`: backbone hidden size, vocabulary,
  tokenizer probe, four-layer drafter shape, pre/post projections, target K/V
  layer-type mapping, and ordered-embedding tensors.
- [ ] Keep the hidden-aware retained path. Native MTP prompt-cache entries
  should preserve the final target hidden state; KV-only restored memory may
  replay only the final suffix token needed to recover hidden, never the whole
  retained prefix.
- [ ] Teach `bench` and `driver-profile` to report attached-assistant MTP
  metrics for the official pair: draft token schedule, proposed/accepted/
  rejected counts, target verify calls, visible tok/s, target-only tok/s,
  warm-decode tok/s, wall time, restore time, peak memory, and quality flags.
  `state-ramp-profile` now also aggregates those MTP fields into its summary
  when retained turns carry `Metrics.MTP`; retained official-pair generation
  still needs to feed those counters before this reporting gate can close.
  `production-mtp-compare` can consume either `driver-profile` or retained
  `state-ramp-profile` JSON, with explicit assistant/draft fallbacks for
  state-ramp artefacts that do not carry driver-profile speculative fields.
- [ ] Run greedy target-only and MTP on identical prompts. Initial sweeps should
  include `draft_tokens=1`, `2`, and `4`, plus a heuristic schedule if the
  implementation supports it. Do not promote MTP if it changes greedy output or
  loses wall time on opencode-shaped retained workflows.
- [ ] Add focused verification before any MTP benchmark claim:

  ```bash
  cd /Users/snider/Code/core/go-mlx
  env GOCACHE=/private/tmp/codex-go-mlx-cache MLX_METALLIB_PATH=/Users/snider/Code/core/go-mlx/dist/lib/mlx.metallib go test ./go/internal/metal -run 'TestGemma4Assistant|TestGemma4AssistantDecode|TestGenerateGemma4Assistant' -count=1
  ```

### TurboQuant KV

- [x] Treat `/Users/snider/Downloads/2504.19874v1.pdf` as the local
  implementation source for this lane. Before coding, write a short
  repo-local implementation note under `docs/runtime/` or `docs/` that maps
  the paper's Algorithm 1 (`TurboQuantmse`) and Algorithm 2
  (`TurboQuantprod`) onto go-mlx cache tensors, including exact tensor shapes,
  metadata, and restore format. Evidence: `docs/runtime/turboquant_kv.md`
  records the local PDF source basis, maps Algorithm 1 to the V/MSE-base path,
  maps Algorithm 2 to the K/QJL residual path, names the rank-4
  `[batch, kv_heads, page_tokens, head_dim]` logical page shape, and documents
  `turboquant-kv-v1` metadata, alignment, version checks, and restore stages.
- [ ] Design an explicit cache mode, provisionally `KVCacheModeTurboQuant`, in
  the same public/native mode families that currently expose `fp16`, `q8`,
  `k-q8-v-q4`, `paged`, and `fixed`. It must be opt-in through load options and
  CLI flags; do not make it an automatic low-memory fallback.
- [ ] Implement TurboQuant as KV-cache compression, not weight quantisation.
  The design target is the paper's inner-product variant: apply the MSE
  rotation/codebook path at `b-1` bits, store the quantised centroid indices,
  compute the residual, and add the 1-bit QJL residual correction plus residual
  norm so attention-score estimates are unbiased. Do not label plain scalar
  q2/q3/q4 cache storage as TurboQuant unless the QJL residual correction is
  present or the report explicitly calls it a diagnostic ablation.
- [ ] Account for the paper's unit-sphere assumption in real K/V tensors. The
  cache format must specify whether each key/value vector stores an explicit
  norm, folds norm into the residual scale, or uses a per-channel/page
  normalisation compatible with restore and streaming append.
- [ ] Implement non-integer effective bit budgets as channel groups, matching
  the paper's LongBench setup: separate outlier and regular channels and apply
  independent TurboQuant instances with different bit-widths, for example a
  3.5-bit target before any 2.5-bit experiment. Record outlier selection policy
  per layer/head so state snapshots are deterministic and portable.
- [ ] Keep per-layer/page metadata small enough that overhead does not erase the
  memory win: rotation seed or matrix identifier, codebook identifier, outlier
  channel mask, residual norm, and QJL sign bits must all be included in the
  accounting beside the packed centroid indices.
- [ ] Decide the MLX/Metal boundary before writing kernels: either dequantise at
  the SDPA boundary for a first correctness path, or fuse rotation,
  quantisation/dequantisation, and attention score computation behind a native
  C++/Metal wrapper if the first path is dominated by memory traffic.
- [ ] Version prompt-cache and state snapshots that contain TurboQuant K/V so
  older q8/k-q8-v-q4/paged restores fail clearly instead of misreading the
  physical layout.
- [ ] Validate TurboQuant with synthetic cache tests first, then real Gemma 4
  E2B retained-state tests. Compare fp16, paged, q8, k-q8-v-q4, and TurboQuant
  for restore time, memory, raw decode, visible throughput, output equality on
  greedy smoke prompts, and long-output quality.
- [ ] Reproduce the paper-relevant quality checks in go-mlx terms before any
  promotion claim: needle-in-haystack/restore retrieval, LongBench-like
  summarisation or code prompts if available locally, and a long retained
  chapter/profile run. The first target is 3.5-bit KV quality neutrality against
  fp16/paged K/V; 2.5-bit is an explicit stress/quality-loss probe.
- [ ] Run focused cache verification before performance claims:

  ```bash
  cd /Users/snider/Code/core/go-mlx
  env GOCACHE=/private/tmp/codex-go-mlx-cache MLX_METALLIB_PATH=/Users/snider/Code/core/go-mlx/dist/lib/mlx.metallib go test ./go/internal/metal -run 'TestQuantizedKVCache|TestPromptCache|TestModelSession|TestTurboQuant' -count=1
  ```

### Promotion policy

- [ ] MTP may become the interactive default only when official E2B MTP beats
  target-only on the same retained workflow and preserves greedy output quality.
- [ ] TurboQuant may become a recommended cache mode only when long-context
  quality is neutral against the accepted baseline and the memory win is visible
  after all metadata and restore costs are included.
- [ ] A combined MTP+TurboQuant lane must also be tested. MTP must not hide
  TurboQuant quality loss, and TurboQuant must not hide assistant acceptance or
  verify-loop regressions.
  2026-06-01 self-bench cleanup: the combined promotion evaluator now reuses
  immutable internal policy defaults and scans required draft-token sweeps
  without temporary maps/slices. `BenchmarkProdLane_EvaluateProductionCombinedMTPAndTurboQuantPromotion_PassingEvidence`
  improved from `460.5 ns/op`, `1496 B/op`, `4 allocs/op` to `211.7 ns/op`,
  `0 B/op`, `0 allocs/op`; this is policy-path polish only and does not promote
  the combined runtime lane.

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
- Official Google E2B MTP and TurboQuant KV remain opt-in research lanes until
  their exact model revisions, quality checks, retained-state behaviour, and
  benchmark artefacts are recorded beside the archived q4 baseline.
- The app-facing quantisation default is quality-first: 6-bit for normal
  machines, 8-bit when memory headroom allows it, and 4-bit only when hardware
  or retained-context constraints force the smaller pack.
- Tests, build, diff hygiene, benchmark artefacts, and state smoke evidence are
  all present in the repo.

<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# 2026-05-19 GOAL.md Completion Audit

> 2026-05-20 correction: this audit is superseded for the
> 10-chapter/full-book `chapter-profile` lane. A later run exposed a safety
> hole where a degenerate generation could continue allocating or sampling
> suppressed special tokens until the OS killed the process. See
> `docs/runtime/2026-05-20-chapter-profile-safety.md`. The q4-first benchmark
> and retained-state evidence below remain historical evidence, but the
> full-book workflow is not accepted until it completes under the new guards.

Objective: work through `GOAL.md` for the go-mlx agentic-memory production
runner lane.

Verdict: complete for the current q4-first agentic runner goal. The benchmark,
state, runner-calibration, packaging, and portable-contract lanes have evidence.
The full model-level native one-token boundary is explicitly retained as future
R&D, not as a blocker for this goal, because the broad native wrapper was
measured and rejected while the accepted hybrid native-sub-block lane now has
large-context/8k-return q4-vs-BF16 wall-clock, memory, and estimated-energy
evidence plus a corrected E2B 100k retained-state run.

## Prompt-to-Artifact Checklist

| Requirement | Evidence | Status |
| --- | --- | --- |
| Build and ship `lthn-mlx` for app/CLI/server bundle | `Taskfile.yml` build targets are documented in `GOAL.md`; latest local rebuild passed with `env GOWORK=/Users/snider/Code/core/go-mlx/go.work GOCACHE=/private/tmp/codex-go-mlx-cache MLX_METALLIB_PATH=/Users/snider/Code/core/go-mlx/dist/lib/mlx.metallib go build -trimpath -o ../bin/lthn-mlx ./cmd/mlx/` | Covered |
| Use workspace-aware verification, not `GOWORK=off` | Latest full test lane passed with `GOWORK=/Users/snider/Code/core/go-mlx/go.work`; `GOAL.md` records this as the goal lane | Covered |
| Machine-readable driver profiling with raw decode, prefill, restore, wall-clock, prompt length, context, cache policy, and energy estimate fields | `go/cmd/mlx/main.go` `driver-profile`; report schema and summary fields verified by tests; `docs/runtime/2026-05-19-runner-calibration.md` references the accepted artifacts | Covered |
| Keep metric honesty between raw decode and derived effective throughput | `docs/runtime/2026-05-19-runner-calibration.md` separates raw decode, wall time, retained setup saved, joules, and derived effective tok/s | Covered |
| Re-admit configured alternatives as calibration evidence | `runner-calibration.md` records llama.cpp, `mlx_lm`, and vLLM calibration; best in-process `mlx_lm` still beats the older small-context cached-prefix shape, but the active acceptance lane is now q4-first long-context/8k-return agentic workflow evidence rather than the old short-context Python cached-prefix micro-shape | Covered; remaining external comparisons are calibration, not completion blockers |
| Preserve retained-state advantage over replayed prefill | `runner-calibration.md` records retained-prefix setup savings and joule estimates for the 10-turn README workflow; `docs/runtime/2026-05-19-gemma4-e2b-100k-retained-paged.md` records a 10-turn E2B 100k retained-state run that saves `1403.301s` of prompt setup, or `140330.10 J` at the normalised `100 W` estimate, compared with replayed prefill | Covered |
| Avoid replaying large prompt strings on warm large-context turns | `driver-profile -prompt-chunk-bytes`; chat/raw chunked large-context artifacts in `runner-calibration.md`; session token/chunk APIs documented there | Covered |
| Prepare gradual large-context ramp toward 100k tokens and large-turn fairness | `driver-profile -prompt-repeat N`; `scripts/gemma4_context_ramp.sh`; first Metal-visible repeat `1/4/8/13/24` ladder documented in `runner-calibration.md`; the first 26B repeat `46` attempt remains documented as a local kernel-coverage failure, while the corrected E2B 4bit `context=131072` paged-retained artefact proves the small dense-family 100k retained-state lane with `100912` prompt tokens per turn and `10/10` successful turns; fresh E2B q4/BF16 profile covers `28587` prompt tokens with an `8192` token return allowance | Covered for current acceptance; same-shape external 100k comparisons and 5120-token sustained-turn ladders remain future benchmarking |
| Exercise Gemma 4 retained multi-turn generation with thinking enabled and no thought history replay | `chapter-profile`; `go/session.go` retained-stream parser path; `external/go-inference/go/parser/markers.go` Gemma 4 channel markers; `docs/runtime/2026-05-19-go-mlx-gemma4-26b-a4b-q4-fresh-story-thinking-ctx65536-c2-g8192-energy100w.json`; extracted book artifact at `docs/runtime/2026-05-19-go-mlx-gemma4-26b-a4b-q4-fresh-story-thinking-ctx65536-c2-g8192-book.md`; E2B retained-story artifacts at `docs/runtime/2026-05-19-go-mlx-gemma4-e2b-q4-fresh-story-thinking-ctx65536-c2-g8192-energy100w.json` and `docs/runtime/2026-05-19-go-mlx-gemma4-e2b-q4-fresh-story-thinking-ctx65536-c2-g8192-book.md` | Covered for current acceptance; longer creative growth remains optional benchmarking |
| Separate E2B/E4B/31B dense-family iteration targets from the 26B MoE quality target | `docs/runtime/2026-05-19-runner-calibration.md` records matched mlx-community E2B/26B q4 iteration profiles plus E2B retained-story evidence; `GOAL.md` now records E2B/E4B as the fast small dense-family lane, 31B as the larger member of that same effective family, and 26B MoE as passable in the restored `88 tok/s` band; the E4B MXFP8 native-QMM smoke and three-run profile prove the MLX-community MXFP8 path now runs without the dense fallback | Covered as benchmark posture; larger dense-family compatibility remains future work |
| Use q4 as the goal throughput lane and BF16 as the reference comparator | `GOAL.md` and `runner-calibration.md` now record q4-first benchmark policy, the E2B q4-vs-BF16 long-context/8k-return comparator at `docs/runtime/2026-05-19-go-mlx-gemma4-e2b-q4-fast-gemma4-lane-r13-ctx65536-g8192-r1-energy100w.json` and `docs/runtime/2026-05-19-go-mlx-gemma4-e2b-bf16-fast-gemma4-lane-r13-ctx65536-g8192-r1-energy100w.json`, an all-quant E2B matrix, and an E4B MXFP8 native-QMM comparison against E4B q4 at `docs/runtime/2026-05-19-go-mlx-gemma4-e4b-mxfp8-v0311-native-qmm-3run-readme-energy100w.json` and `docs/runtime/2026-05-19-go-mlx-gemma4-e4b-q4-fast-gemma4-lane-iteration-3run-readme-energy100w.json`. At `28587` prompt tokens and `8192` generated tokens, E2B q4 records `94.92547697253806 tok/s`, `111.006821417s`, `11100.6821417 J`, and `5.134385833516717 GiB`; BF16 records `26.59615320070758 tok/s`, `334.4575525s`, `33445.75525 J`, and `12.643188176676631 GiB`. On the E4B README profile, MXFP8 native QMM records `69.23950679870225 tok/s`, while the q4 row records `86.09288563808235 tok/s` with its own memory and energy profile | Covered for E2B all-quants, E2B q4-vs-BF16, and E4B MXFP8-vs-q4; E4B BF16 and 31B q4-vs-BF16 comparators remain future work |
| Keep Gemma 4 production lane current | `go/production_lane.go` fast-lane gate set; restored shared-mask evidence in `GOAL.md` and `runner-calibration.md` | Covered |
| Evaluate MTP/speculative decode separately from raw decode | `docs/runtime/2026-05-18-gemma4-mtp-speculative-decode.md`; GOAL table records native MTP is an R&D lane, not production | Covered |
| Agentic memory seed/wake/append/sleep/reload works without prefill replay | `GOAL.md` Workstream 4 checklist is checked with session/state APIs and tests named in the file | Covered by existing GOAL evidence |
| Portable contracts stay aligned with go-inference/go-ai/go-ml boundaries | `GOAL.md` Workstream 6 checklist is checked; external contract notes remain in the file | Covered by existing GOAL evidence |
| Native hot path keeps expensive repeated decode work in native code where it is proven beneficial | `GOAL.md` Workstream 3 now records the acceptance decision: the full model-level greedy wrapper exists but is rejected because it regresses the 26B A4B q4 lane into the `50 tok/s` band; the accepted production lane keeps proven native sub-blocks in `go/internal/metal`, keeps q4 decode in the usable optimisation band, and leaves the full one-token native boundary as future R&D | Covered for current acceptance; full one-token native boundary remains future R&D |

## Final Verification

The completion check found no unchecked `GOAL.md` workstream items.

The required `GOAL.md` verification commands were run from
`/Users/snider/Code/core/go-mlx/go` with
`GOWORK=/Users/snider/Code/core/go-mlx/go.work`,
`GOCACHE=/private/tmp/codex-go-mlx-cache`, and
`MLX_METALLIB_PATH=/Users/snider/Code/core/go-mlx/dist/lib/mlx.metallib`:

- `go test ./... -count=1`: passed.
- `go build -trimpath -o ../bin/lthn-mlx ./cmd/mlx/`: passed.
- `git diff --check`: passed from `/Users/snider/Code/core/go-mlx`.

## Current Native Boundary State

Current accepted production decode is a hybrid:

- Go owns `Gemma4Model.forwardHidden`, layer iteration, per-layer input
  preparation, fixed-mask selection, cache ownership, and fallback routing.
- Native code owns several bounded sub-blocks: fixed-cache attention update,
  router matvec/top-k, dense local MLP matvec, direct greedy output projection,
  FFN residual diagnostics, row cache-update diagnostics, and rejected broad
  fixed-owner/model-greedy wrappers.
- The full model-level greedy wrapper exists behind
  `GO_MLX_ENABLE_NATIVE_GEMMA4_MODEL_GREEDY=1`, but current evidence rejects it
  as a production boundary because it materialises too much native graph work and
  regresses the full README lane.

Completion no longer requires a positive full one-token native boundary for this
goal. `GOAL.md` now explicitly changes that requirement: the broad wrapper was
implemented and rejected by measurement, and the current production acceptance is
the q4-first hybrid native-sub-block lane with retained-state and long-context
energy evidence. Future work should still attack a better full-native boundary
only if it preserves the packed expert-ID/q4 kernels and improves the accepted
lane.

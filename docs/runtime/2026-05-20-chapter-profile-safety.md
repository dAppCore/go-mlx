<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# 2026-05-20 Benchmark Safety Correction

## Verdict

The previous 2-chapter retained-story evidence is still useful as a template and
parser smoke, but it is not enough to accept the requested 10-chapter/full-book
workflow. The later E2B fresh-history attempt exposed a runner safety bug: a bad
generation could keep allocating or keep sampling repeated/special tokens and
still look like a normal run until the OS killed it.

No 10-chapter/full-book report is accepted until it completes under the new
guards.

## Rejected Evidence

- The E2B fresh-history book artifact at
  `docs/runtime/2026-05-19-go-mlx-gemma4-e2b-4bit-fresh-history-c10-g1536-book.md`
  is rejected. It contains planning text and repeated-token degeneration rather
  than a usable book.
- The matching per-chapter JSON sequence is rejected as a benchmark source
  because the run was killed before a complete 10-turn report was written.
- The earlier 2-chapter 26B and E2B story artifacts remain parser/template
  smokes only. They do not prove the longer creative retained-state workflow.
- The compact 26B raw Markdown artifact at
  `docs/runtime/2026-05-20-go-mlx-gemma4-26b-a4b-q4-raw-unaccepted-c10-g128-rp105-book.md`
  is available to read, but is rejected as benchmark evidence. It reached ten
  chapter headings before the stricter guard was added, and later chapters
  degrade into fragments.
- The rebuilt stricter rerun at
  `docs/runtime/2026-05-20-go-mlx-gemma4-26b-a4b-q4-guarded-chapter-profile-nothink-ctx4096-c10-g128-rp105-energy100w.json`
  rejects the same shape at chapter 9 with a repeated visible-sentence failure.
- The first `lthn/lemer-mlx` run is rejected for this harness. It exposed a
  Gemma 4 attention nil-state panic; the rebuilt CLI now captures that as a JSON
  error instead of dumping a stack trace. The root cause was a no-config affine
  q4 pack whose U32 packed weights needed group/bits inference from the
  safetensors weight/scale shape.

## Code Change

`chapter-profile` now fails fast instead of silently accepting pathological
turns:

- JSON reports include `safety_limits`.
- Default active-memory limits are derived from the resolved MLX memory plan
  with `30%` headroom for live-eval allocator transients; resident-memory limits
  use the resolved plan directly.
- Process virtual memory is reported in every run, but no absolute virtual
  address-space cap is derived by default. MLX can reserve hundreds of GiB of
  virtual address space for a physically small paged-cache run; default hard
  memory guards therefore stay on MLX active memory and process resident
  memory. Operators can still enforce a hard virtual cap with
  `-max-process-virtual-memory-bytes`.
- Post-load metrics are checked before prefill so a bad model load cannot exceed
  the memory guard before the first turn.
- Initial prefill is checked immediately after it completes.
- Memory is checked inside the token probe callback during generation, not only
  after a turn finishes.
- Every generated chapter turn is checked again before it can be appended back
  into retained history.
- Repeated sampled suppressed-token loops are cancelled from the token probe
  callback, including special tokens filtered out of visible output.
- Repeated visible lines, repeated visible sentences, fragmented sentence
  outputs, and meta-planning/outline outputs are rejected before a turn is
  appended back into retained history.
- Empty visible Gemma 4 turns are rejected.
- `chapter-profile` exposes `-repeat-penalty` and records `repeat_penalty` in
  JSON so anti-loop sampling changes are visible in the artifact.
- `chapter-profile` now requires each accepted chapter to emit the
  `[[END_CHAPTER]]` marker. If a turn reaches `chapter_max_tokens` or stops
  without that marker, it is rejected and is not accepted as completed story
  context.
- `chapter-profile` and `driver-profile` now recover profile panics into JSON
  errors, so model-variant crashes do not masquerade as shell/runner failures.
- Chapter summaries now carry process virtual and resident memory peaks.

`driver-profile` now has matching benchmark guards:

- JSON reports include `safety_limits`.
- Default active-memory limits are derived from the resolved MLX memory plan
  with `30%` headroom for live-eval allocator transients, and resident-memory
  limits use the resolved plan directly. Process virtual memory is recorded by
  default and is only a hard failure when the operator passes
  `-max-process-virtual-memory-bytes`.
- Memory is checked inside the token probe callback during generation.
- Consecutive sampled-token loops are cancelled from the token probe callback.
- Repeated visible lines, repeated visible sentences, fragmented sentence
  outputs, and profile panics are rejected/captured in the same benchmark
  surface.
- The first sampled token IDs/texts are retained in each run for auditability.
- Failed runs still contribute peak memory, process virtual memory, resident
  memory, and peak resident memory to the summary.

## Verification

Focused no-model-generation tests passed:

```bash
env GOWORK=/Users/snider/Code/core/go-mlx/go.work \
  GOCACHE=/private/tmp/codex-go-mlx-cache \
  MLX_METALLIB_PATH=/Users/snider/Code/core/go-mlx/dist/lib/mlx.metallib \
  go test ./cmd/mlx \
  -run 'TestRunCommand_(DriverProfileSafetyFlags|DriverProfileRepeatedTokenLoopLimit|ChapterProfileSafetyFlags|ChapterProfileSuppressedTokenLoopLimit)|TestDriverProfile(SafetyLimits|RepeatedTokenLoop|RunSafety|MetricsSafety|Summary_IncludesFailedRunMemory)|TestChapterProfile(SafetyLimits|SuppressedTokenLoop|TurnSafety|MetricsSafety)' \
  -count=1
```

Result: passed.

The final focused run also covered the panic guards, repeated visible-line
guard, repeated visible-sentence guard, fragmented-output guard, meta-planning
guard, and `chapter-profile -repeat-penalty` validation. Result: passed.

Full workspace-aware Go verification also passed:

```bash
env GOWORK=/Users/snider/Code/core/go-mlx/go.work \
  GOCACHE=/private/tmp/codex-go-mlx-cache \
  MLX_METALLIB_PATH=/Users/snider/Code/core/go-mlx/dist/lib/mlx.metallib \
  go test ./... -count=1
```

The CLI rebuild also passed:

```bash
env GOWORK=/Users/snider/Code/core/go-mlx/go.work \
  GOCACHE=/private/tmp/codex-go-mlx-cache \
  MLX_METALLIB_PATH=/Users/snider/Code/core/go-mlx/dist/lib/mlx.metallib \
  go build -trimpath -o ../bin/lthn-mlx ./cmd/mlx/
```

## Latest Guarded Attempts

- E2B 4bit `context=8192`, `chapter_max_tokens=1024`: no OOM; stopped at
  chapter 5 on eight suppressed token IDs. Peak active MLX memory stayed around
  `6.45 GB`, resident memory around `3.45 GB`.
- 26B A4B q4 `context=4096`, `chapter_max_tokens=384`: stopped at chapter 9 on
  active-memory guard before an OS OOM.
- 26B A4B q4 `context=4096`, `chapter_max_tokens=256/192/128/96`: later turns
  degenerated into repeated sentences or fragments; the stricter guard now
  rejects these shapes instead of calling them successful books.
- `lthn/lemer-mlx`: the initial native attention panic is now captured as JSON,
  then fixed by validating K/V state and inferring affine q4 settings from U32
  packed weight/scale shapes. A one-turn smoke now completes with active MLX
  memory around `3.76 GB`, resident memory around `4.17 GB`, `~2008 tok/s`
  prefill, and `~78 tok/s` decode.
- The corrected 10-chapter `lthn/lemer-mlx` fast thinking run with
  `chapter_max_tokens=2048` and `[[END_CHAPTER]]` markers accepts chapter 1,
  then rejects chapter 2 because the model stops before the marker with only
  `This is Chapter 2.`. The no-thinking comparator still emits visible planning
  text in chapter 1. No `lthn/lemer-mlx` 10-chapter/full-book artifact is
  accepted yet.
- The sampler suppression order is fixed: suppressed tokens are now masked
  before top-p/top-k filtering, so a dominant suppressed token cannot collapse
  the candidate set and fall back to token `0`.

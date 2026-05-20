<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# 2026-05-19 Runner Calibration

This pass reframes the old round-number `100 tok/s` target around the real
agentic workload: repeated turns over a retained project prefix. External
runners calibrate the lane; future optimisation should benchmark against the
current go-mlx best unless an external runner wins the same workflow.

## go-mlx Current Best

Artifact:
`docs/runtime/2026-05-19-go-mlx-gemma4-26b-a4b-q4-effective-agentic-10step-readme-ctx4096-ours-only.json`

Energy estimate artifact:
`docs/runtime/2026-05-19-go-mlx-gemma4-26b-a4b-q4-effective-agentic-10step-readme-ctx4096-energy100w.json`

Current shortcut refresh artefacts:

- `docs/runtime/2026-05-19-go-mlx-gemma4-26b-a4b-q4-fast-gemma4-lane-current-10step-readme-chat-energy100w.json`
- `docs/runtime/2026-05-19-go-mlx-gemma4-26b-a4b-q4-fast-gemma4-lane-current-10step-readme-raw-energy100w.json`
- `docs/runtime/2026-05-19-go-mlx-gemma4-26b-a4b-q4-fast-gemma4-lane-generation-stream-10step-readme-energy100w.json`
- `docs/runtime/2026-05-19-go-mlx-gemma4-26b-a4b-q4-fast-gemma4-lane-default-generation-stream-10step-readme-energy100w.json`
- `docs/runtime/2026-05-19-go-mlx-gemma4-26b-a4b-q4-fast-gemma4-lane-rebalance-control-3run-readme-energy100w.json`
- `docs/runtime/2026-05-19-go-mlx-gemma4-26b-a4b-q4-fast-gemma4-lane-restored-shared-mask-default-3run-readme-energy100w.json`
- `docs/runtime/2026-05-19-go-mlx-gemma4-26b-a4b-q4-fast-gemma4-lane-explicit-shared-mask-post-rebalance-10run-readme-energy100w.json`

- Model: `mlx-community/gemma-4-26b-a4b-it-4bit`
- Prompt: repo `README.md`, `2204` prompt tokens
- Generation: `128` visible tokens per turn, `10` turns
- Cold turn: `2.668634083s` total, `1.059383417s` prefill,
  `1.609250583s` decode, `79.54012964306628 tok/s` decode
- Warm turns: `1.4592862175555557s` average total,
  `0.004666874777777778s` average retained-prefix setup,
  `1.4546192917777776s` average decode,
  `87.995764012926 tok/s` warm decode
- Ten-turn wall-clock: `16.380037957s`
- Setup saved versus replaying prefill every turn: `9.49244888s`
- Decode-equivalent effective visible throughput: `128.6485922304177 tok/s`

The energy-enabled rerun uses `-estimate-power-watts 100` as a normalised
active-power assumption, not a measured claim. It records:

- Raw decode: `87.74067183813047 tok/s`; warm raw decode:
  `87.84861155177613 tok/s`
- Ten-turn wall-clock: `16.252888247s`
- Estimated total energy at `100 W`: `1625.2888247 J`
- Estimated joules per visible token at `100 W`: `1.269756894296875 J/token`
- Retained-prefix setup saved versus replayed prefill: `9.406740417s`, or
  `940.6740417 J` at `100 W`

These estimates scale linearly with the wattage assumption. For example, a
`150 W` active-power assumption would make the retained-prefix setup saving
about `1411.01106255 J`.

The refreshed current shortcut run keeps the same accepted gate set and removes
the older slow shortcut sample as a decision point. Chat-mode
`-fast-gemma4-lane` records `86.96995653092598 tok/s` raw decode,
`87.10762008324762 tok/s` warm raw decode, `16.413198251s` wall time, and
`1641.3198251 J` at the normalised `100 W` estimate. Raw prompt mode records
`87.18727600068239 tok/s` raw decode, `87.28239963327297 tok/s` warm raw
decode, `16.382709584s` wall time, and `1638.2709584 J`. Both stderr files are
empty. These refreshes keep the current go-mlx small-context repeated workflow
within the same `87 tok/s` band, but they still do not beat persistent
in-process `mlx_lm` on the README cached-prefix workflow.

The follow-up `mlx_lm` source comparison showed that Python is running
`mlx` `0.31.2` / `mlx_lm` `0.31.3`, uses a dedicated
`mx.new_thread_local_stream(mx.default_device())`, and queues the next token
with `mx.async_eval`. The existing Go async prefetch gate did not explain the
gap: it records `86.55268124366343 tok/s`, `16.496068705s`, and
`1649.6068705 J`, slower than the refreshed chat control. A narrower Go
generation-stream gate is positive and is now part of `-fast-gemma4-lane`.
The explicit diagnostic records `88.10704229468793 tok/s`, `16.239494334s`,
and `1623.9494334 J`; the no-explicit-stream shortcut validation records
`GO_MLX_ENABLE_GENERATION_STREAM=1`, `87.50749912985658 tok/s`,
`16.334514708s`, and `1633.4514708 J`, with empty stderr. This was the
accepted shortcut number before the rebalance refresh below.

The rebalance refresh restores the best small-context first-run shape while
keeping the accepted gate set. The default `-fast-gemma4-lane` 3-run validation
records `GO_MLX_ENABLE_FIXED_GEMMA4_SHARED_MASK=1`, `88.5760834806412 tok/s`
average raw decode, `87.87017208983966 tok/s` first-run decode,
`2094.1931616252605 tok/s` first-run prefill, `5.971295375s` wall time, and
`597.1295375000001 J` at `100 W`, with empty stderr. A same-gate 10-run pass
records `88.50777967819847 tok/s` average raw decode,
`88.61333712754153 tok/s` warm raw decode, `2100.679478883641 tok/s`
first-run prefill, `16.146115667s` wall time, and `1614.6115667 J` at
`100 W`. Against the archived same-prompt llama.cpp Q4_K_M calibration
(`pp2204=2109.335561 tok/s`, `tg128=91.451031 tok/s`), go-mlx now reaches
`99.5896299158653%` of first-run prefill and `96.78160946944215%` of raw
decode on the 10-run evidence. The gap to the best configured in-process
`mlx_lm` cached-prefix workflow narrows to `1.2941856671120566s` including
load at the same `100 W` estimate.

## go-mlx Large Context

Artifacts:

- `docs/runtime/2026-05-19-go-mlx-gemma4-26b-a4b-q4-effective-agentic-3step-readme-x11-ctx32768-energy100w.json`
- `docs/runtime/2026-05-19-go-mlx-gemma4-26b-a4b-q4-effective-agentic-3step-readme-x13-ctx32768-energy100w.json`
- `docs/runtime/2026-05-19-go-mlx-gemma4-26b-a4b-q4-effective-agentic-2step-readme-x13-ctx32768-chunk1024-energy100w.json`
- `docs/runtime/2026-05-19-go-mlx-gemma4-26b-a4b-q4-effective-agentic-2step-readme-x13-ctx32768-promptchunk4096-prefill1024-energy100w.json`
- `docs/runtime/2026-05-19-go-mlx-gemma4-26b-a4b-q4-effective-agentic-10step-readme-x13-ctx32768-promptchunk4096-prefill1024-energy100w.json`
- `docs/runtime/2026-05-19-go-mlx-gemma4-26b-a4b-q4-effective-agentic-10step-readme-x13-chat-ctx32768-promptchunk4096-prefill1024-energy100w.json`
- `docs/runtime/2026-05-19-go-mlx-gemma4-26b-a4b-q4-fast-gemma4-lane-longctx-default-chunks-3run-readme-x13-energy100w.json`
- `docs/runtime/2026-05-19-go-mlx-gemma4-26b-a4b-q4-longctx-prefill-chunk384-promptchunk4096-max1-readme-x13.json`
- `docs/runtime/2026-05-19-go-mlx-gemma4-26b-a4b-q4-longctx-prefill-chunk128-promptchunk4096-max1-readme-x13.json`
- `docs/runtime/2026-05-19-go-mlx-gemma4-26b-a4b-q4-longctx-prefill-chunk256-promptchunk4096-max1-readme-x13.json`
- `docs/runtime/2026-05-19-go-mlx-gemma4-26b-a4b-q4-longctx-prefill-chunk512-promptchunk4096-max1-readme-x13.json`
- `docs/runtime/2026-05-19-go-mlx-gemma4-26b-a4b-q4-longctx-prefill-chunk640-promptchunk4096-max1-readme-x13.json`
- `docs/runtime/2026-05-19-go-mlx-gemma4-26b-a4b-q4-longctx-prefill-chunk768-promptchunk4096-max1-readme-x13.json`
- `docs/runtime/2026-05-19-go-mlx-gemma4-26b-a4b-q4-longctx-prefill-chunk1024-promptchunk4096-max1-readme-x13.json`
- `docs/runtime/2026-05-19-go-mlx-gemma4-26b-a4b-q4-longctx-prefill-chunk2048-promptchunk4096-max1-readme-x13.json`
- `docs/runtime/2026-05-19-go-mlx-gemma4-26b-a4b-q4-longctx-prefill-chunk4096-promptchunk4096-max1-readme-x13.json`
- `docs/runtime/2026-05-19-go-mlx-gemma4-26b-a4b-q4-fast-gemma4-lane-longctx-prefill512-promptchunk4096-3run-readme-x13-energy100w.json`
- `docs/runtime/2026-05-19-go-mlx-gemma4-26b-a4b-q4-fast-gemma4-lane-longctx-default512-chunks-3run-readme-x13-energy100w.json`
- `docs/runtime/2026-05-19-go-mlx-gemma4-26b-a4b-q4-fast-gemma4-lane-longctx-sliding-cache-bound-3run-readme-x13-energy100w.json`
- `docs/runtime/2026-05-19-go-mlx-gemma4-26b-a4b-q4-fast-gemma4-lane-longctx-sliding-cache-bound-restore-3run-readme-x13-energy100w.json`
- `docs/runtime/2026-05-19-go-mlx-gemma4-26b-a4b-q4-fast-gemma4-lane-longctx-default-sliding-cache-bound-3run-readme-x13-energy100w.json`
- `docs/runtime/2026-05-19-go-mlx-gemma4-26b-a4b-q4-fast-gemma4-lane-longctx-default-sliding-cache-bound-token-phases.json`
- `docs/runtime/2026-05-19-go-mlx-gemma4-26b-a4b-q4-fast-gemma4-lane-longctx-default-sliding-cache-bound-native-events.json`
- `docs/runtime/2026-05-19-go-mlx-gemma4-26b-a4b-q4-fast-gemma4-lane-longctx-fixed-owner-attention-3run-readme-x13-energy100w.json`
- `docs/runtime/2026-05-19-go-mlx-gemma4-26b-a4b-q4-fast-gemma4-lane-longctx-full-only-fixed-owner-attention-3run-readme-x13-energy100w.json`
- `docs/runtime/2026-05-19-go-mlx-gemma4-26b-a4b-q4-longctx-no-shared-mask-3run-readme-x13-energy100w.json`
- `docs/runtime/2026-05-19-go-mlx-gemma4-26b-a4b-q4-fast-gemma4-lane-longctx-dynamic-slice-update-3run-readme-x13-energy100w.json`
- `docs/runtime/2026-05-19-go-mlx-gemma4-26b-a4b-q4-fast-gemma4-lane-longctx-wide-sdpa-attention-3run-readme-x13-energy100w.json`
- `docs/runtime/2026-05-19-go-mlx-gemma4-26b-a4b-q4-fast-gemma4-lane-longctx-wide-matmul-attention-3run-readme-x13-energy100w.json`
- `docs/runtime/2026-05-19-go-mlx-gemma4-26b-a4b-q4-fast-gemma4-lane-longctx-row-cache-update-wide-sdpa-3run-readme-x13-energy100w.json`
- `docs/runtime/2026-05-19-llamacpp-gemma4-26b-a4b-q4-k-m-p28637-g1-metal-bench.json`
- `docs/runtime/2026-05-19-llamacpp-gemma4-26b-a4b-q4-k-m-p28637-g128-metal-bench.json`

100k ramp harness:

- `scripts/gemma4_context_ramp.sh`
- `docs/runtime/2026-05-19-go-mlx-gemma4-26b-a4b-q4-fast-gemma4-lane-context-ramp-repeat1-ctx4096-g128-r3-energy100w.json`
- `docs/runtime/2026-05-19-go-mlx-gemma4-26b-a4b-q4-fast-gemma4-lane-context-ramp-repeat4-ctx16384-g128-r3-energy100w.json`
- `docs/runtime/2026-05-19-go-mlx-gemma4-26b-a4b-q4-fast-gemma4-lane-context-ramp-repeat8-ctx32768-g128-r3-energy100w.json`
- `docs/runtime/2026-05-19-go-mlx-gemma4-26b-a4b-q4-fast-gemma4-lane-context-ramp-repeat13-ctx32768-g128-r3-energy100w.json`
- `docs/runtime/2026-05-19-go-mlx-gemma4-26b-a4b-q4-fast-gemma4-lane-context-ramp-repeat24-ctx65536-g128-r3-energy100w.json`

The ramp harness uses the accepted `-fast-gemma4-lane`, the repo `README.md`,
`-prompt-repeat`, chunked large-context defaults, and writes one JSON plus stderr
artefact per step under `docs/runtime/`. The default ladder is:

- repeat `1`, `context=4096`
- repeat `4`, `context=16384`
- repeat `8`, `context=32768`
- repeat `13`, `context=32768`
- repeat `24`, `context=65536`
- repeat `46`, `context=131072`

Since the README prompt is about `2204` tokens in the normal chat template, the
final step is the intended `~100k` prompt-token neighbourhood. Set
`GO_MLX_RAMP_MAX_TOKENS=5120` to run the sustained large-turn fairness lane
instead of the default `128` token latency lane. The output must be treated as
new evidence only when the JSON reports successful runs and a non-empty summary,
not when it only records a Metal availability error.

The first Metal-visible ladder pass ran the smaller `1/4/8` repeat steps with
`128` generated tokens and three runs per step. All stderr files are empty.

- repeat `1`, `context=4096`, `2204` prompt tokens:
  `88.69834535003041 tok/s`, `5.971431375s`, `597.1431375 J`,
  restore average `4.730271ms`
- repeat `4`, `context=16384`, `8785` prompt tokens:
  `74.33104068005494 tok/s`, `12.315293209s`, `1231.5293209 J`,
  restore average `2.124937ms`
- repeat `8`, `context=32768`, `17559` prompt tokens:
  `69.48165669588239 tok/s`, `21.636779s`, `2163.6779 J`,
  restore average `12.732479ms`
- repeat `13`, `context=32768`, `28528` prompt tokens:
  `62.59204228638978 tok/s`, `36.263682833s`, `3626.3682833 J`,
  restore average `21.270354ms`
- repeat `24`, `context=65536`, `52657` prompt tokens:
  `50.656561535149365 tok/s`, `80.389911666s`, `8038.991166600001 J`,
  restore average `44.504187ms`, retained setup saved `129.80999529s`

The first cliff appears before the old 29k opencode-shaped prompt: short
context remains in the `88 tok/s` band, while `8.8k` and `17.6k` prompts move
to about `74 tok/s` and `69 tok/s`. The repeat-13 step reproduces the promoted
29k band at about `62.6 tok/s`, and repeat `24` reaches `52.7k` prompt tokens
at about `50.7 tok/s` with warm restore still in the millisecond range. The
next ramp should continue with repeat `46`, then repeat the best shapes with
`GO_MLX_RAMP_MAX_TOKENS=5120`.

Retained-story chapter harness:

- `go/cmd/mlx chapter-profile`
- `docs/runtime/2026-05-19-go-mlx-gemma4-26b-a4b-q4-fresh-story-thinking-ctx65536-c2-g8192-energy100w.json`
- `docs/runtime/2026-05-19-go-mlx-gemma4-26b-a4b-q4-fresh-story-thinking-ctx65536-c2-g8192-book.md`

The chapter harness uses the model's Gemma 4 turn markers, enables thinking by
placing `<|think|>` at the top of the system turn, standardises sampling at
`temperature=1.0`, `top_p=0.95`, and `top_k=64`, and appends only stripped
visible assistant text back into the retained session state. The session
stream now runs the shared thinking parser, with Gemma 4
`<|channel>thought ... <channel|>` markers registered in the parser, so
thought blocks are hidden before history is appended. The first corrected
two-chapter run at `context=65536`, `chapter_max_tokens=8192`, and the
normalised `100 W` energy assumption records `2` successful turns,
`4171` generated tokens, `1033` visible tokens, `57.559931252s` total wall
time, `73.90526235355026 tok/s` average decode, `910.112139725012 tok/s`
average prefill, and `5755.9931252 J`. The extracted markdown has no retained
Gemma channel markers or leading `thought` text, and stderr is empty.

The same harness was probed against the cached `lthn/lemer-mlx` snapshot after
confirming its `chat_template.jinja` uses the same Gemma 4 thinking system-turn
shape. It did not reach generation. The default run wrote no JSON and panicked
inside the dense Gemma compiled GELU path; the retry with
`GO_MLX_ENABLE_NATIVE_GELU_GATE_MUL=1` also wrote no JSON and panicked with an
empty MLX array in the native GELU gate/mul bridge. Evidence is preserved in:

- `docs/runtime/2026-05-19-go-mlx-lthn-lemer-mlx-fresh-story-thinking-ctx65536-c2-g8192-energy100w.stderr`
- `docs/runtime/2026-05-19-go-mlx-lthn-lemer-mlx-native-gelu-fresh-story-thinking-ctx65536-c2-g8192-energy100w.stderr`

mlx-community E2B/26B q4 iteration posture:

- `docs/runtime/2026-05-19-go-mlx-gemma4-e2b-q4-fast-gemma4-lane-iteration-3run-readme-energy100w.json`
- `docs/runtime/2026-05-19-go-mlx-gemma4-26b-a4b-q4-fast-gemma4-lane-iteration-3run-readme-energy100w.json`
- `docs/runtime/2026-05-19-go-mlx-gemma4-e2b-q4-fresh-story-thinking-ctx65536-c2-g8192-energy100w.json`
- `docs/runtime/2026-05-19-go-mlx-gemma4-e2b-q4-fresh-story-thinking-ctx65536-c2-g8192-book.md`
- `docs/runtime/2026-05-19-go-mlx-gemma4-e2b-4bit-default-longform-c2-g8192-energy100w.json`

Both native MLX q4 snapshots are cached under the `mlx-community` namespace, so
the faster iteration lane does not need Python-format conversion. On the same
current-binary README profile (`2204` prompt tokens, `128` generated tokens,
three runs, hidden output, and the normalised `100 W` energy assumption), E2B
records `122.23205359983257 tok/s` decode, `4.532718042s` wall time,
`453.2718042 J`, and `4.523123664781451 GiB` peak memory. The matched 26B A4B
q4 run records `88.18156398367199 tok/s` decode, `6.027796249s` wall time,
`602.7796249 J`, and `17.314671628177166 GiB` peak memory. E2B is therefore
`1.3861x` faster on raw decode and uses `0.7519x` the wall time and energy on
this short iteration profile.

The retained-story harness shows the same direction but with a larger workflow
gap. E2B completes two thinking-enabled retained turns at `context=65536` with
`1767` generated tokens, `1087` visible tokens, `16.935350541s` wall time,
`110.35789603546327 tok/s` average decode, `965.9831974768388 tok/s` average
prefill, `1693.5350541 J`, and `4.489579644054174 GiB` peak memory. Compared
with the 26B A4B story smoke, E2B is `1.4932x` faster on average decode and
uses `0.2942x` the wall time and energy. This makes E2B/E4B the practical
small dense-family iteration lane, with 31B treated as the larger member of the
same effective architecture family rather than a different bucket. The 26B MoE
q4 path remains a passable quality lane at the restored `88 tok/s` band. The
larger dense-family lane still needs separate scale/runtime compatibility work
because the first `lthn/lemer-mlx` smoke blocked before generation in
GELU/native array handling.

The goal bench policy is q4-first. BF16 should be retained as a quality and
regression comparator, but the production throughput target is q4 for E2B,
E4B, 26B MoE, and the 31B dense-family scale-up. For the E2B/E4B iteration
lane, `>100 tok/s` decode is acceptable when the q4 profile also keeps the
memory and estimated-energy advantages; holding that band as context length
grows is the stronger result to optimise for next.

Long-context 8k-return E2B q4/BF16 comparator:

- `docs/runtime/2026-05-19-go-mlx-gemma4-e2b-q4-fast-gemma4-lane-r13-ctx65536-g8192-r1-energy100w.json`
- `docs/runtime/2026-05-19-go-mlx-gemma4-e2b-bf16-fast-gemma4-lane-r13-ctx65536-g8192-r1-energy100w.json`

The comparator uses the README repeat shape to approximate an opencode-sized
startup context and then appends a synthetic agentic operations-log request:
`prompt_repeat=13`, `context=65536`, `28587` prompt tokens, and
`max_tokens=8192`. Both q4 and BF16 completed the full `8192` token generation
with empty stderr. Q4 records `94.92547697253806 tok/s` decode,
`1396.6243790432902 tok/s` prefill, `111.006821417s` wall time,
`11100.6821417 J`, and `5.134385833516717 GiB` peak memory. BF16 records
`26.59615320070758 tok/s` decode, `1304.3044170967798 tok/s` prefill,
`334.4575525s` wall time, `33445.75525 J`, and `12.643188176676631 GiB` peak
memory. Q4 is `3.569x` faster on decode, `3.013x` lower wall time and energy,
and uses `0.406x` the peak memory on this shape. The q4 decode rate is slightly
under the round `100 tok/s` line at this 29k-context/8k-return shape; BF16 stays
recorded as the quality/reference comparator rather than collapsed into a speed
verdict.

Gemma 4 E2B all-quant matrix:

- `docs/runtime/2026-05-19-gemma4-e2b-quant-matrix.md`

The E2B matrix now lists `mxfp4`, `mxfp8`, `4bit`, `5bit`, `6bit`, `8bit`, and
`bf16` on the same README-shaped profile. Cross-runner anchors are limited to
4-bit and 8-bit, where llama.cpp has comparable GGUF formats. The matrix also
records the MLX-LM/vLLM Metal E2B compatibility gap: both current runners use
the MLX-LM loader surface and reject the local Gemma 4 E2B snapshots at load
with extra attention K/V parameters, so no MLX-LM or vLLM throughput number is
claimed for those E2B rows.

mlx-community E4B MXFP8 native QMM support:

- `docs/runtime/2026-05-19-go-mlx-gemma4-e4b-q4-fast-gemma4-lane-iteration-3run-readme-energy100w.json`
- `docs/runtime/2026-05-19-go-mlx-gemma4-e4b-mxfp8-fast-gemma4-lane-iteration-3run-readme-energy100w.json`
- `docs/runtime/2026-05-19-go-mlx-gemma4-e4b-mxfp8-v0311-native-qmm-smoke-g16-energy100w.json`
- `docs/runtime/2026-05-19-go-mlx-gemma4-e4b-mxfp8-v0311-native-qmm-3run-readme-energy100w.json`

After bumping `mlx-c` to `v0.6.0` and aligning the local patched MLX submodule
to the `v0.31.1` version used by that release, the rebuilt `dist/lib/mlx.metallib`
contains both the patched 512-wide SDPA resource and native MXFP8 QMM kernels.
The loader now preserves `quantization.mode`, accepts MLX-community
`affine`, `mxfp4`, `mxfp8`, and `nvfp4` config shapes, and keeps the old MXFP8
dense-dequantise fallback behind `GO_MLX_ENABLE_MXFP8_DENSE_FALLBACK=1`.

The old E4B MXFP8 diagnostic fallback completed but had a different runtime
profile: it recorded `14.800582374835564 tok/s` decode, `27.691197209s` wall time,
`2769.1197209 J`, and `20.31 GiB` peak memory on the README profile. The native
MXFP8 QMM path completes the same three-run profile at `69.23950679870225 tok/s`
decode, `821584.7669364832 tok/s` prefill, `7.22419575s` wall time,
`722.419575 J`, and about `9.21 GiB` peak memory. This proves the MLX-community
MXFP8 path is wired through the native kernel stack. The matched q4 profile
records a separate point in the matrix at
`86.09288563808235 tok/s`, `6.115125667s`, `611.5125667 J`, and about
`5.97 GiB` peak memory.

The opencode IDE startup shape is closer to `29k` prompt tokens than the
README-sized `2204` token calibration. Repeating the README text exposes a
separate large-context cost:

- `24212` prompt tokens, `context=32768`, default `4096` prefill chunks:
  cold model prefill is `55.555967333s`; cache-hit restore is about `0.5s`;
  cache-hit turns still spend roughly `72-74s` before the first token.
- `28612` prompt tokens, `context=32768`, default `4096` prefill chunks:
  cold model prefill is `87.872341208s`; run 2 restore is `0.497940792s`, but
  run 2 wall time is `115.383811292s` with `111.082583667s` driver overhead.
- Lowering model prefill chunks to `1024` improves the `28612` token cold
  prefill to `70.193964333s`, but run 2 still takes `110.010683625s` with
  `105.659096458s` driver overhead.

The cliff is therefore not KV restore. It is the driver feeding a giant prompt
string through tokenisation every turn before the model metrics begin.

The patched chunked prompt path adds `driver-profile -prompt-chunk-bytes` and
uses chunk-aware stream calls so the driver can feed bounded prompt chunks to
the native generator. Raw prompt mode uses `GenerateChunksStream`; chat mode
uses `ChatChunksStream`, which renders the native chat template and chunks the
message content before tokenisation.

With `-chat=false -prompt-chunk-bytes 4096 -prefill-chunk-size 1024`, the
`28625` token run records:

- Ten-turn wall-clock: `115.288840001s`
- Cold turn: `78.403770292s`; cold prefill: `69.856424834s`
- Warm turns: about `4.1s` each for `128` visible tokens
- Warm restore: `255-303ms`; restore average: `280.517444ms`
- Warm driver overhead: about `18-19ms`, down from `~105s`
- Raw decode: `33.48494955572712 tok/s`
- Estimated total energy at `100 W`: `11528.8840001 J`
- Retained setup saved versus replayed cold prefill: `626.183063256s`, or
  `62618.3063256 J` at `100 W`

Verdict: chunked prompt tokenisation removes the repeated-turn 29k wall-clock
cliff.

The normal chat-mode rerun with `-prompt-chunk-bytes 4096` records:

- Prompt tokens: `28637`
- Ten-turn wall-clock: `115.247971709s`
- Cold turn: `78.4869145s`; cold prefill: `69.914225167s`
- Warm turns: about `4.08-4.10s` each for `128` visible tokens
- Warm restore: `260-298ms`; restore average: `278.342120ms`
- Warm driver overhead: about `18-22ms`, down from `~105s`
- Raw decode: `33.58024749556697 tok/s`
- Estimated total energy at `100 W`: `11524.7971709 J`
- Retained setup saved versus replayed cold prefill: `626.722864295s`, or
  `62672.2864295 J` at `100 W`

Verdict: the chunked large-context fix now applies to normal chat-mode
diagnostics, not only raw prompt mode. The session API now also exposes
`ModelSession.PrefillChunks`, `ModelSession.AppendPromptChunks`,
`ModelSession.PrefillTokens`, and `ModelSession.AppendTokens`, so durable
agent-memory callers can wake retained KV state, append bounded context, or feed
already-stored model-native tokens without reconstructing one giant prompt string.
For opencode-sized `24k+` startup contexts, the serving shape should keep both
levers on: `-prompt-chunk-bytes 4096` prevents repeated giant-string
tokenisation on warm turns, and a smaller model prefill chunk gives the model
digestible ingestion work. The initial accepted run used
`-prefill-chunk-size 1024`, but the follow-up chunk sweep shows `512` is the
better automatic default on the `28637` token chat shape:

- `128`: cold prefill `82.128389084s`, total `86.586956875s`
- `256`: cold prefill `74.8167155s`, total `79.315089166s`
- `384`: cold prefill `70.790761667s`, total `75.108669459s`
- `512`: cold prefill `67.631178917s`, total `71.980500625s`
- `640`: cold prefill `68.351593667s`, total `72.921384708s`
- `768`: cold prefill `69.52491675s`, total `74.067976s`
- `1024`: cold prefill `69.769200709s`, total `74.183554584s`
- `2048`: cold prefill `73.696338791s`, total `78.285060625s`
- `4096`: cold prefill `85.410324s`, total `89.920771417s`

The curve is not monotonic: below `512`, per-chunk overhead dominates; above
`512`, the model ingests less naturally for this long prompt.

The no-explicit-chunk shortcut validation with the rebuilt CLI records
`load.prefill_chunk_size=512` and `prompt_chunk_bytes=4096` by default. Its
three 128-token chat runs record `28637` prompt tokens, `84.995550583s` wall
time, `33.22422183528957 tok/s` average raw decode, `298.090812ms` average
restore, `8499.5550583 J` at the normalised `100 W` estimate, and empty
stderr. Warm-turn driver overhead stays at `17.72925ms` and `20.881375ms`,
confirming that the shortcut now encodes the large-context chunking shape rather
than relying on manual benchmark flags. The remaining production work is wiring
higher-level agent state through those token/session APIs and benchmarking
changing-prompt workflows where only the new turn context should be appended.

The follow-up same-length llama.cpp calibration shows that the `29k` slowdown is
not only a bad chunk-size choice. The working Metal invocation must run outside
the sandbox and must not force `GGML_METAL_DEVICES=0`; with the embedded Metal
library it reports `MTL0: Apple M3 Ultra`. On the same local Q4_K_M GGUF,
`llama-bench -p 28637 -n 1 -r 1 -ngl 99 -fa 1` records `1525.801226 tok/s`
prefill in `18.768499791s`. The paired `-pg 28637,128` run records pure
`tg128` decode at `92.211737 tok/s` and combined `pp28637+tg128` throughput at
`1398.527504 tok/s` over `20.568061709s`. Against the current go-mlx
long-context retained-state artefact, the cold run prefill is
`419.11716620820545 tok/s`, warm retained decode averages
`33.91056160965191 tok/s`, and the cold run takes `76.811422833s`. That leaves
llama.cpp about `3.64x` faster on
same-length cold prefill, `2.72x` faster on raw decode, and `3.73x` faster on
the comparable cold prompt-plus-decode wall-clock. The retained-state workflow
still avoids replaying the `29k` prefix on warm turns, but the next native
performance boundary is long-context fixed-cache/attention scaling rather than
another `512` vs `640` prefill-chunk default tweak.

The long-context cache follow-up made that boundary concrete. The small
README-sized lane had previously rejected per-layer sliding fixed-cache bounds,
so the first change kept it opt-in behind
`GO_MLX_ENABLE_FIXED_GEMMA4_SLIDING_CACHE_BOUND` / CLI
`-fixed-gemma4-sliding-cache-bound`. In the `29k` context shape, preserving the
native 1024-token fixed capacity for sliding-attention layers while leaving
full-attention layers request-sized improved a manual diagnostic from `84.996s`
to `88.185s` overall only because prompt-cache restore still missed; the per-run
numbers nevertheless exposed the right shape: cold prefill rose from
`419.11716620820545 tok/s` to `1105.275329844354 tok/s`, and warm decode would
be about `62.86 tok/s` if the prefix could be restored.

The prompt-cache restore path now snapshots bounded fixed-cache tail state with
the full logical prefix offset and restores it back into a bounded fixed cache
when the sliding-bound gate is active. After that fix, the same manual
diagnostic records `36.742183291s` total for three turns,
`62.85654704339822 tok/s` average decode, `63.09018925356014 tok/s` warm
decode, `1098.4953035273882 tok/s` cold prefill, `21.839395ms` average
restore, and `3674.2183291 J` at `100 W`, with empty stderr.

This gate is now promoted only for `-fast-gemma4-lane` when the requested
context exceeds the normal `4096` production context. The no-explicit-flag
validation records `GO_MLX_ENABLE_FIXED_GEMMA4_SLIDING_CACHE_BOUND=1`,
`prefill_chunk_size=512`, and `prompt_chunk_bytes=4096` by default for
`context=32768`. It reports `36.868437918s` total, `62.51129327845945 tok/s`
average decode, `62.63259219208622 tok/s` warm decode,
`1094.4247968802333 tok/s` cold prefill, `21.757104ms` average restore,
`3686.8437918 J` at `100 W`, and empty stderr. Against the previous
long-context default this is `0.434x` the wall time and energy, `1.88x` the raw
decode, `1.85x` the warm decode, `2.61x` the cold prefill, and about `13.70x`
faster restore. Against same-length llama.cpp, the cold prefill gap shrinks from
about `3.64x` to `1.39x`, pure decode remains `1.47x` behind, and the cold
prompt-plus-decode wall-clock gap is now about `1.59x`.

The long-context token-phase and native-event traces keep the next boundary in
evaluated graph/kernel work. A one-run `-trace-token-phases` profile with
`max_tokens=16` records `1096.311492962768 tok/s` prefill and
`59.84070210617055 tok/s` decode; excluding the first token and final step, the
14 steady tokens average `17.746205ms` total, with `16.3555565ms` in
`Eval(next)` and `1.346199ms` in forward graph construction. A diagnostic
`GO_MLX_TRACE_FORWARD_EVAL=1` trace slows throughput, but the ranked native
buckets are still useful: attention leads at `73.077582ms` over 90 events,
followed by local MLP at `23.520166ms`, split expert activation at
`23.266755ms`, router at `22.603662ms`, attention residual at `21.01459ms`,
and expert down at `20.881961ms`. The full-attention layers are the visible
long-context spike; prompt-cache restore and chunk sizing are no longer the
main 29k bottleneck.

Five immediate attention/cache follow-ups did not justify a default change.
Re-enabling the original all-layer `-native-gemma4-fixed-owner-attention` on the
promoted 29k shortcut records `36.44726s` wall time and
`62.317460438377985 tok/s` decode. Narrowing that diagnostic so it only wraps
the five full-attention owner layers records `36.426556958s` and
`62.48077885938384 tok/s`, which is cleaner but still effectively flat against
the default `36.868437918s` / `62.51129327845945 tok/s` run. A manual same-gate
run without `GO_MLX_ENABLE_FIXED_GEMMA4_SHARED_MASK` records `36.337556126s` and
`62.79482183164808 tok/s`, which is only a marginal 29k gain and conflicts with
the earlier README-sized evidence where the shared mask was required for the
active band. A gated experiment that swapped fixed K/V updates from
`put_along_axis` to MLX dynamic `slice_update` records `36.582005083s` and
`62.45483265128252 tok/s`, so the suspected full-cache write-copy cost is not
solved by that primitive. A llama.cpp-inspired row-shaped cache-update
diagnostic records `36.570614625s`, `62.0477494292309 tok/s`, `20.323458ms`
average restore, and `19884219328` peak bytes. That is a tiny wall-clock shift
but worse decode and higher memory than the accepted default, so the row update
also remains a diagnostic gate.

## go-mlx Expert Path Control

Artifact:
`docs/runtime/2026-05-19-go-mlx-gemma4-26b-a4b-q4-gather-qmm-decode-control-10step-readme-ctx4096-ours-only.json`

Fixed-owner attention rerun artifact:
`docs/runtime/2026-05-19-go-mlx-gemma4-26b-a4b-q4-fixed-owner-attention-current-stack-10step-energy100w.json`

This control disables `-expert-id-matvec` and `-expert-id-fused-activation`
while keeping fixed cache, shared mask, direct greedy, sorted prefill, native
router matvec/top-k, and native MLP matvec on.

- Average raw decode: `54.02683426487331 tok/s`
- Warm raw decode: `54.10799458992597 tok/s`
- stderr: empty

Verdict: the active expert-ID path is about `62.4%` faster than this MLX
`gather_qmm` fallback control. Re-admitting `gather_qmm` for single-token decode
is not the next path to close the `mlx_lm` gap.

The current-stack fixed-owner attention gate is also rejected. Re-enabling
`-native-gemma4-fixed-owner-attention` on top of the active flags records
`85.20005681731622 tok/s` average decode and `16.718573375s` wall time, versus
the active energy rerun at `87.74067183813047 tok/s` and `16.252888247s`.
That is a `2.8956%` decode regression, `0.465685128s` more wall time, and about
`46.5685128 J` extra at the normalised `100 W` estimate.

## Native Model Greedy Probe

Artifacts:

- `docs/runtime/2026-05-19-go-mlx-gemma4-26b-a4b-q4-native-model-greedy-moe-gated-trace.json`
- `docs/runtime/2026-05-19-go-mlx-gemma4-26b-a4b-q4-native-model-greedy-moe-gated-3run-readme.json`

The earlier model-level greedy probe enabled `-native-gemma4-model-greedy` but
missed the MoE-native gate, so the production model never reached the wrapper.
The new trace skip reason exposed a second real-pack guard: the 26B A4B q4 pack
has no per-layer input tensors, so the wrapper now accepts nil per-layer inputs
and passes nil per layer.

- Corrected trace: seven `gemma4.model.greedy_token` events over an 8-token run
- Full README 3-run decode: `50.56636111604209 tok/s`
- Warm decode runs: `50.85608151751184` and `50.9117166606287 tok/s`
- stderr: empty

Verdict: the model-level wrapper now fires, but it is much slower than the active
packed expert-ID path. This rejects the broad one-call native wrapper as the next
production optimisation; the useful target is a narrower native boundary that
preserves the custom packed expert kernels instead of rebuilding the whole layer
graph inside one C++ call.

## Fast Gemma 4 Lane

Artifact:
`docs/runtime/2026-05-19-go-mlx-gemma4-26b-a4b-q4-fast-gemma4-lane-3run-readme.json`

Token-phase artifact:
`docs/runtime/2026-05-19-go-mlx-gemma4-26b-a4b-q4-fast-gemma4-lane-token-phases.json`

Report-summary smoke artifact:
`docs/runtime/2026-05-19-go-mlx-gemma4-26b-a4b-q4-fast-gemma4-lane-report-summary-fields-smoke.json`

Native-event smoke artifact:
`docs/runtime/2026-05-19-go-mlx-gemma4-26b-a4b-q4-fast-gemma4-lane-native-event-smoke.json`

Fixed-owner attention native-event smoke artifact:
`docs/runtime/2026-05-19-go-mlx-gemma4-26b-a4b-q4-fast-gemma4-lane-fixed-owner-attention-native-event-smoke.json`

Attention O-projection matvec artefacts:

- `docs/runtime/2026-05-19-go-mlx-gemma4-26b-a4b-q4-attention-o-matvec-control-3run-readme.json`
- `docs/runtime/2026-05-19-go-mlx-gemma4-26b-a4b-q4-attention-o-matvec-gated-3run-readme.json`
- `docs/runtime/2026-05-19-go-mlx-gemma4-26b-a4b-q4-attention-o-matvec-control-10run-readme-energy100w.json`
- `docs/runtime/2026-05-19-go-mlx-gemma4-26b-a4b-q4-attention-o-matvec-gated-10run-readme-energy100w.json`

10-step shortcut artefacts:

- `docs/runtime/2026-05-19-go-mlx-gemma4-26b-a4b-q4-fast-gemma4-lane-10step-readme-energy100w.json`
- `docs/runtime/2026-05-19-go-mlx-gemma4-26b-a4b-q4-fast-gemma4-lane-10step-readme-raw-energy100w.json`
- `docs/runtime/2026-05-19-go-mlx-gemma4-26b-a4b-q4-fast-gemma4-lane-current-10step-readme-chat-energy100w.json`
- `docs/runtime/2026-05-19-go-mlx-gemma4-26b-a4b-q4-fast-gemma4-lane-current-10step-readme-raw-energy100w.json`
- `docs/runtime/2026-05-19-go-mlx-gemma4-26b-a4b-q4-fast-gemma4-lane-async-prefetch-10step-readme-energy100w.json`
- `docs/runtime/2026-05-19-go-mlx-gemma4-26b-a4b-q4-fast-gemma4-lane-generation-stream-10step-readme-energy100w.json`
- `docs/runtime/2026-05-19-go-mlx-gemma4-26b-a4b-q4-fast-gemma4-lane-default-generation-stream-10step-readme-energy100w.json`
- `docs/runtime/2026-05-19-go-mlx-gemma4-26b-a4b-q4-fast-gemma4-lane-rebalance-control-3run-readme-energy100w.json`
- `docs/runtime/2026-05-19-go-mlx-gemma4-26b-a4b-q4-fast-gemma4-lane-rebalance-attention-o-matvec-3run-readme-energy100w.json`
- `docs/runtime/2026-05-19-go-mlx-gemma4-26b-a4b-q4-fast-gemma4-lane-rebalance-row-cache-update-3run-readme-energy100w.json`
- `docs/runtime/2026-05-19-go-mlx-gemma4-26b-a4b-q4-fast-gate-set-no-shared-mask-rebalance-3run-readme-energy100w.json`
- `docs/runtime/2026-05-19-go-mlx-gemma4-26b-a4b-q4-fast-gate-set-no-shared-mask-rebalance-10run-readme-energy100w.json`
- `docs/runtime/2026-05-19-go-mlx-gemma4-26b-a4b-q4-fast-gemma4-lane-explicit-shared-mask-post-rebalance-10run-readme-energy100w.json`
- `docs/runtime/2026-05-19-go-mlx-gemma4-26b-a4b-q4-fast-gemma4-lane-restored-shared-mask-default-3run-readme-energy100w.json`

Long-context shortcut artefacts:
`docs/runtime/2026-05-19-go-mlx-gemma4-26b-a4b-q4-fast-gemma4-lane-longctx-default-chunks-3run-readme-x13-energy100w.json`
`docs/runtime/2026-05-19-go-mlx-gemma4-26b-a4b-q4-fast-gemma4-lane-longctx-prefill512-promptchunk4096-3run-readme-x13-energy100w.json`
`docs/runtime/2026-05-19-go-mlx-gemma4-26b-a4b-q4-fast-gemma4-lane-longctx-default512-chunks-3run-readme-x13-energy100w.json`
`docs/runtime/2026-05-19-go-mlx-gemma4-26b-a4b-q4-fast-gemma4-lane-longctx-default-sliding-cache-bound-3run-readme-x13-energy100w.json`
`docs/runtime/2026-05-19-go-mlx-gemma4-26b-a4b-q4-fast-gemma4-lane-longctx-default-sliding-cache-bound-token-phases.json`
`docs/runtime/2026-05-19-go-mlx-gemma4-26b-a4b-q4-fast-gemma4-lane-longctx-default-sliding-cache-bound-native-events.json`
`docs/runtime/2026-05-19-go-mlx-gemma4-26b-a4b-q4-fast-gemma4-lane-longctx-fixed-owner-attention-3run-readme-x13-energy100w.json`
`docs/runtime/2026-05-19-go-mlx-gemma4-26b-a4b-q4-fast-gemma4-lane-longctx-full-only-fixed-owner-attention-3run-readme-x13-energy100w.json`
`docs/runtime/2026-05-19-go-mlx-gemma4-26b-a4b-q4-longctx-no-shared-mask-3run-readme-x13-energy100w.json`
`docs/runtime/2026-05-19-go-mlx-gemma4-26b-a4b-q4-fast-gemma4-lane-longctx-dynamic-slice-update-3run-readme-x13-energy100w.json`
`docs/runtime/2026-05-19-go-mlx-gemma4-26b-a4b-q4-fast-gemma4-lane-longctx-wide-sdpa-attention-3run-readme-x13-energy100w.json`
`docs/runtime/2026-05-19-go-mlx-gemma4-26b-a4b-q4-fast-gemma4-lane-longctx-wide-matmul-attention-3run-readme-x13-energy100w.json`
`docs/runtime/2026-05-19-go-mlx-gemma4-26b-a4b-q4-fast-gemma4-lane-longctx-row-cache-update-wide-sdpa-3run-readme-x13-energy100w.json`

`driver-profile -fast-gemma4-lane` now applies the accepted Gemma 4 gate set in
one switch: expert-ID matvec, fused expert activation, sorted expert prefill,
native MLP matvec, native router matvec/top-k, fixed Gemma 4 cache, shared fixed
mask, direct greedy token, and the dedicated generation stream. It also defaults
diagnostics to `cache_mode=paged` and `context=4096` unless those flags are
explicitly supplied. When the operator supplies a larger context, the shortcut
now defaults to the proven long-context shape, `-prefill-chunk-size 512` plus
`-prompt-chunk-bytes 4096`, unless those chunk flags are explicitly supplied.

Rejected broad wrappers are intentionally absent from this shortcut:
`GO_MLX_ENABLE_NATIVE_GEMMA4_LAYER`,
`GO_MLX_ENABLE_NATIVE_GEMMA4_MODEL_GREEDY`,
`GO_MLX_ENABLE_NATIVE_GEMMA4_FIXED_OWNER_ATTENTION`, and
`GO_MLX_ENABLE_NATIVE_LINEAR_MATVEC`.

The real 26B README 3-run shortcut validation records:

- Average decode: `85.45833951808704 tok/s`
- Warm decode runs: `85.1685322234809` and `86.19157159973682 tok/s`
- Average retained-prefix setup: `308502.11971190706 tok/s`
- Restore average: `4.772ms`
- stderr: empty

The 10-step retained-prefix shortcut reruns are lower than the earlier same-gate
energy artefact:

- Chat-mode shortcut: `78.73916236563421 tok/s`, `1808.0075749999999 J` at
  `100 W`, retained setup saved `964.2656999999999 J`, stderr empty
- Raw `-chat=false` shortcut: `83.71186949154026 tok/s`, `1717.8121293 J` at
  `100 W`, retained setup saved `1046.5401381 J`, stderr empty
- Older same-gate retained-state artefact:
  `87.74067183813047 tok/s`, `1625.2888247 J` at `100 W`

The current default shortcut also reports `GO_MLX_ENABLE_GENERATION_STREAM=1`.
The no-explicit-stream validation records `87.50749912985658 tok/s` raw decode,
`16.334514708s` wall time, and `1633.4514708 J` at the normalised `100 W`
estimate. That saves `0.078683543s` and `7.8683543 J` versus the refreshed
chat control. The explicit `-generation-stream` diagnostic sample is faster
again at `88.10704229468793 tok/s`, `16.239494334s`, and `1623.9494334 J`,
but the default shortcut number is the accepted-path evidence.

The latest rebalance pass confirms the right small-context combination is the
default fast lane with the shared fixed mask still enabled. The rebuilt default
3-run validation records `88.5760834806412 tok/s` average decode,
`87.87017208983966 tok/s` first-run decode, `2094.1931616252605 tok/s`
first-run prefill, and empty stderr. The same-binary 10-run shared-mask sample
records `88.50777967819847 tok/s` average decode,
`88.61333712754153 tok/s` warm decode, `2100.679478883641 tok/s` first-run
prefill, `16.146115667s` wall time, and `1614.6115667 J` at the normalised
`100 W` estimate. The checked neighbours do not beat that full balance:
attention O-proj matvec is `88.53279331842275 tok/s`, the row cache-update
gate is `86.57971461366179 tok/s`, and the no-shared-mask 10-run default
sample is `87.10676731805157 tok/s`.

Verdict: the shortcut applies the intended accepted gate set and load defaults,
and the generation stream is a small accepted default-path win. It still does
not close the stronger in-process `mlx_lm` cached-prefix workflow gap.

The current token-phase profile records `84.32951687301572 tok/s`. Steady
non-final tokens average about `10.406612ms` in `Eval(next)`, `1.461166ms` in
forward graph construction, and `11.915181ms` total. That keeps the next
raw-decode target in evaluated graph/kernel work rather than CLI driver
overhead.

The report-summary smoke validates the current JSON schema on a short real
profile: `summary.prompt_tokens_average`, `summary.prompt_tokens_min`, and
`summary.prompt_tokens_max` all report `2204` for the README prompt, while the
same summary keeps decode, wall-clock, memory, restore, and energy fields at the
top level.

The native-event smoke enables diagnostic materialisation with
`GO_MLX_TRACE_FORWARD_EVAL=1`, so its `15.080719570351203 tok/s` decode is not a
throughput claim. It is useful attribution: `summary.native_events` now groups
the per-layer trace into stable buckets. On the short README smoke, the largest
bucket is attention (`100.062542ms` over `210` events), followed by local MLP
(`54.313699ms`), router (`54.281834ms`), split expert activation
(`50.886424ms`), and attention residual (`45.670918ms`). The buckets are ranked
by total duration in the JSON summary, so future traces expose the hot path
without a separate jq aggregation. That keeps the next
raw-decode target in the evaluated attention/FFN graph rather than prompt
handling or driver orchestration.

Re-enabling `-native-gemma4-fixed-owner-attention` under the same traced
shortcut does not reduce the ranked attention bucket: decode falls to
`14.50847005479256 tok/s`, while attention remains `100.305117ms` over `210`
events. That confirms the existing fixed-owner wrapper is not the current
answer to the attention bucket; the next useful attention work has to be a
lower-level graph/kernel change rather than reusing that broad wrapper.

The narrower `-native-gemma4-attention-o-matvec` probe routes only the Gemma 4
attention output projection through the existing q4/q8 single-token matvec
kernel. It stays opt-in. The paired three-run README control records
`85.85272086042305 tok/s`, while the gated run records
`84.68415619194967 tok/s`; both have empty stderr. A longer ten-run pass is
slightly positive but too small to promote by itself: same-binary control is
`83.59564887907933 tok/s` average raw decode and
`83.75771763124862 tok/s` warm raw decode, while the gated path is
`84.04525365609535 tok/s` average raw decode and
`84.10303328183633 tok/s` warm raw decode. At the normalised `100 W` estimate,
the gated ten-run costs `1699.7798417 J` versus `1710.686 J` for control. Treat
this as a bounded diagnostic showing attention O-proj alone is not a material
parity fix.

The refreshed long-context shortcut default is `load.prefill_chunk_size=512`
plus `prompt_chunk_bytes=4096`, and now also enables
`GO_MLX_ENABLE_FIXED_GEMMA4_SLIDING_CACHE_BOUND=1` only for contexts above the
normal `4096` shortcut. The no-explicit-flag `32768` context chat profile
records `62.51129327845945 tok/s` average raw decode,
`62.63259219208622 tok/s` warm decode, `36.868437918s` wall time,
`1094.4247968802333 tok/s` cold prefill, `21.757104ms` average restore,
`3686.8437918 J` at the normalised `100 W` estimate, and empty stderr. The
previous `512`-chunk default without the sliding-cache bound is now superseded
at `84.995550583s`, and the earlier `1024` default remains superseded at
`86.433517249s`.

The current long-context attention diagnostics do not yet close the llama.cpp
decode gap. The fixed-owner attention diagnostic is now scoped to full-attention
owner layers, but remains flat (`62.48077885938384 tok/s`). Disabling the shared
fixed mask is only marginally positive on this 29k prompt
(`62.79482183164808 tok/s`) and is not promoted because the short-context lane
uses the shared mask, and dynamic `slice_update` for fixed K/V
updates is negative (`62.45483265128252 tok/s`). Enabling the existing
512-wide native SDPA diagnostic is also flat at `62.147525173976284 tok/s`,
while the wide matmul fallback regresses hard to `23.67497555194655 tok/s` and
raises peak memory to `21548513532` bytes. These wide-head reports were run
with `GO_MLX_ENABLE_FIXED_WIDE_SDPA_ATTENTION=1` and
`GO_MLX_ENABLE_FIXED_WIDE_MATMUL_ATTENTION=1` respectively; the source now
records both env-only diagnostics in future `runtime_gates` snapshots. A
row-shaped K/V cache update behind `GO_MLX_ENABLE_FIXED_ROW_CACHE_UPDATE=1`
also does not move decode: paired with the wide SDPA gate it records
`36.570614625s`, `62.0477494292309 tok/s`, `1101.1801978656852 tok/s` cold
prefill, `3657.0614625 J` at `100 W`, and `19884219328` peak bytes. The next
useful work is still a llama.cpp-style full-attention/KV slot path or
lower-level kernel change, not another wrapper around the current fixed-cache
SDPA graph.

## E2B 100k Retained-State

Detailed report:
`docs/runtime/2026-05-19-gemma4-e2b-100k-retained-paged.md`

The E2B 4bit 100k pass exposed two separate behaviours. The fixed retained
cache path can make warm setup look fast, but it is not acceptable at 100k:
the three-run probe reached `197.17 GiB` MLX active memory and `1232.02 GiB`
process virtual memory for a roughly 5 GiB quantised model. The accepted
100k lane is now paged retained cache with sliding-tail prompt-cache snapshots
and fixed Gemma 4 cache gates excluded above the long-context threshold.

The final accepted 10-turn run uses `100912` prompt tokens per turn,
`128` generated tokens per turn, `context=131072`, and `prefill_chunk_size=512`.
It records `10/10` success, `275.717s` total wall time, `12.34 tok/s` average
raw decode, `647.19 tok/s` cold prefill, `1.98ms` average warm restore,
`3.58 GiB` MLX active memory, `5.19 GiB` resident memory, and `734.41 GiB`
process virtual memory. Treating the retained prefix as logical work, the run
processes `1010400` logical tokens at `3664.63` effective logical tok/s and
saves `1403.301s` of prompt setup, or `140330.10 J` at the normalised `100 W`
estimate, compared with replaying prefill every turn.

Do not read this as a fresh 100k llama.cpp, `mlx_lm`, or vLLM parity claim.
It proves the corrected go-mlx retained-state lane and the fixed-cache failure
mode. External 100k runner comparison still needs a matched run with comparable
cache reuse semantics.

## mlx_lm

Artifacts:

- `docs/runtime/2026-05-19-mlx-lm-gemma4-26b-a4b-q4-readme-ctx2336-g128.txt`
- `docs/runtime/2026-05-19-mlx-lm-gemma4-26b-a4b-q4-readme-cache-prompt-ctx2336.txt`
- `docs/runtime/2026-05-19-mlx-lm-gemma4-26b-a4b-q4-readme-cache-generate-ctx2336-g128.txt`
- `docs/runtime/2026-05-19-mlx-lm-gemma4-26b-a4b-q4-readme-cache-generate-ctx2336-g128-10run-wall.stdout`
- `docs/runtime/2026-05-19-mlx-lm-gemma4-26b-a4b-q4-readme-cache-generate-ctx2336-g128-10run-wall.stderr`
- `docs/runtime/2026-05-19-mlx-lm-gemma4-26b-a4b-q4-readme-cache-inprocess-10run.json`
- `docs/runtime/2026-05-19-mlx-lm-gemma4-26b-a4b-q4-readme-cache-inprocess-10run.stderr`

Configured one-shot command used the repaired parity venv, same MLX-community
26B A4B q4 snapshot, README stdin, `--max-kv-size 2336`, temp `0`, top-p `1`,
and `128` generated tokens.

- One-shot prefill: `2207` tokens at `1506.907 tok/s`
- One-shot generation: `128` tokens at `109.958 tok/s`
- One-shot peak memory: `15.739 GB`
- Prompt-cache setup: final line `2202` tokens at `2197.23 tok/s`; cache file
  `/private/tmp/gemma4-26b-readme-mlx-lm-cache.safetensors` is `243 MB`
- Cached-prefix generate: 5-token suffix at `27.813 tok/s`, then `128`
  generation tokens at `109.325 tok/s`, peak `14.841 GB`
- Cached-prefix CLI 10-turn wall-clock: ten `mlx_lm.generate
  --prompt-cache-file` invocations against the already-created README cache take
  `36.98s` wall time. Per-run generation remains fast, averaging
  `109.5251 tok/s`, but the full CLI workflow only delivers
  `34.613304 visible tok/s` wall-clock because each turn pays process,
  model-load, and cache-load overhead.
- Cached-prefix in-process 10-turn wall-clock: a persistent Python harness loads
  the model and prompt cache once, then deep-copies the saved cache for each
  128-token turn. It records `13.358959957957268s` generation wall time, or
  `14.851929999887943s` including load, with average generation
  `109.65707805632005 tok/s`, peak `15.05557006 GB`, and empty stderr.

Verdict: `mlx_lm` is faster than go-mlx on raw decode today. go-mlx beats the
configured `mlx_lm` CLI cached-prefix loop, but it does not beat the stronger
persistent in-process Python cached-prefix workflow yet. Comparing the
in-process `14.851929999887943s` including load with the restored shared-mask
go-mlx shortcut at `16.146115667s`, go-mlx is `1.2941856671120566s` slower
over ten turns. At the same normalised `100 W` estimate, that is
`1485.1929999887943 J` for in-process `mlx_lm` versus `1614.6115667 J` for
go-mlx default generation-stream mode. The next native
optimisation lane should account for both the Python MLX `0.31.2` runtime
delta and its thread-local stream behaviour; the immediate production target is
about `1.29s` over this 10-turn workflow including load, or
`2.787155709042733s` against generation wall time alone.

## vLLM Metal

Artifacts:

- `docs/runtime/2026-05-19-vllm-metal-gemma4-26b-a4b-q4-readme-shape-b1-latency.json`
- `docs/runtime/2026-05-19-vllm-metal-gemma4-26b-a4b-q4-readme-shape-b1-latency.stdout`
- `docs/runtime/2026-05-19-vllm-metal-gemma4-26b-a4b-q4-readme-shape-latency.json`
- `docs/runtime/2026-05-19-vllm-metal-gemma4-26b-a4b-q4-readme-shape-latency.stdout`

Configured command used the same model directory, input length `2204`, output
length `128`, max model length `4096`, dtype `bfloat16`, and vLLM Metal.

- Batch size 1 latency: `3.8800909579731524s`
- Batch size 8 latency: `15.160140624968335s`

Verdict: vLLM Metal can load and run the model, but it is slower than go-mlx for
the single-request README shape. The batch-8 result is useful capacity evidence,
not a single-request parity number.

## Current Conclusion

The realistic production goal is now:

- Beat vLLM-style serving latency for this Apple Silicon local workflow.
- Preserve the retained-prefix 10-turn win against replay/CLI-style workflows
  and keep reporting derived effective throughput separately from raw decode.
- Use persistent in-process `mlx_lm` as the immediate wall-clock and raw-decode
  target; do not declare the old throughput floor retired until go-mlx closes
  that repeated-workflow gap or explains why the production embedding does not
  admit the Python in-process shape.
- Do not spend another round on the current broad native model greedy wrapper:
  after the corrected MoE/nil-per-layer-input run it fires, but only reaches
  `50.56636111604209 tok/s`.
- Use `driver-profile -fast-gemma4-lane` for future accepted-path Gemma 4
  comparisons, then add only the single diagnostic gate being tested. Refresh
  the 10-step retained-prefix number before claiming a new small-context best;
  the restored shared-mask shortcut is `88.50777967819847 tok/s` over
  `16.146115667s`, while the stronger persistent in-process `mlx_lm`
  cached-prefix workflow is still `14.851929999887943s` including load.
- Use `scripts/gemma4_context_ramp.sh` for the next large-context fairness pass.
  Run the default `128` token ladder first, then rerun the same ladder with
  `GO_MLX_RAMP_MAX_TOKENS=5120` once the best context/chunk shape is confirmed.
  Compare external runners only at matched prompt-token and generation-token
  shapes.
- For large-context IDE workflows, avoid feeding a full prompt string back
  through tokenisation each turn. The chat-mode chunked prompt probe proves that
  repeated 29k prompt handling can move from `~110s` cache-hit turns to `~4.1s`
  turns once tokenisation is chunked or bypassed, and the promoted sliding
  fixed-cache bound moves the same `28637` token shape to about `2.07s` warm
  turns with `62.63259219208622 tok/s` warm decode and `21.757104ms` restore.
  The session token APIs now give callers a direct bypass when they already own
  model-native token segments, but same-length llama.cpp still leads the cold
  prompt-plus-decode wall-clock by about `1.59x`.

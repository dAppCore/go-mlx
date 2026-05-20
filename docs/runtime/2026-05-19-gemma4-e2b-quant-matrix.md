<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# 2026-05-19 Gemma 4 E2B Quant Matrix

Shape: README prompt through the Gemma 4 chat template, `2282` prompt tokens,
`128` generated tokens per run, three go-mlx runs, and normalised `100 W`
energy estimates.

This matrix is a compatibility and short-latency smoke test. It is useful for
checking that each quant loads, that the fast path is active, and that small
decode does not regress. It is not the acceptance benchmark for agentic
workflows. Long-form generation and retained-state wall time are tracked below
and in `docs/runtime/2026-05-19-runner-calibration.md`.

## go-mlx MLX-community Quant Matrix

| Quant | Model | Status | Decode tok/s | Cold prefill tok/s | Summary prefill tok/s | Wall s | Peak GiB | J/visible token |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4bit | `mlx-community/gemma-4-e2b-it-4bit` | ok | `123.34573087131434` | `3724.2800578634306` | `1625456.9132217274` | `4.488069917` | `4.607094233855605` | `1.1687682075520833` |
| 5bit | `mlx-community/gemma-4-e2b-it-5bit` | ok | `110.24303206945446` | `3711.741979944603` | `1578098.0803308908` | `4.8832625` | `5.04675561375916` | `1.2716829427083332` |
| 6bit | `mlx-community/gemma-4-e2b-it-6bit` | ok | `103.05645453314004` | `3683.675031535051` | `1724852.2563665994` | `5.09656125` | `5.5862911362200975` | `1.3272294921874999` |
| 8bit | `mlx-community/gemma-4-e2b-it-8bit` | ok | `101.26776527534014` | `3728.023633539537` | `1706534.3508289002` | `5.154395667` | `6.6653621811419725` | `1.34229053828125` |
| BF16 | `mlx-community/gemma-4-E2B-it-bf16` | ok | `28.854437649593265` | `3594.3087972815256` | `1643867.5871782675` | `14.702114417` | `11.79025492630899` | `3.8286756294270834` |
| MXFP4 | `mlx-community/gemma-4-e2b-it-mxfp4` | ok after fix | `109.19709288036368` | `3735.077133148257` | `1656658.4588410568` | `4.915764375` | `5.139078916981816` | `1.28014697265625` |
| MXFP8 | `mlx-community/gemma-4-e2b-it-mxfp8` | ok | `102.75732486556983` | `3096.4599165672307` | `1717025.6883325065` | `5.215661584` | `6.515818418934941` | `1.3582452041666668` |

`Summary prefill tok/s` includes the two prompt-cache restore runs, so it is a
retained-state workflow metric. `Cold prefill tok/s` is run 1 model prefill.

## 4bit/8bit Runner Anchors

llama.cpp cannot run the MLX MXFP files directly, so the cross-runner anchors
use Unsloth GGUF files with the closest 4-bit and 8-bit formats.

| Anchor | go-mlx model | llama.cpp model | go-mlx decode tok/s | llama.cpp decode tok/s | go-mlx cold prefill tok/s | llama.cpp prefill tok/s | go/llama decode | go/llama prefill |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4-bit | MLX `4bit` | GGUF `Q4_K_M` | `123.34573087131434` | `139.914221` | `3724.2800578634306` | `4320.131793` | `0.8815810858233942` | `0.8620755653561217` |
| 8-bit | MLX `8bit` | GGUF `Q8_0` | `101.26776527534014` | `122.098723` | `3728.023633539537` | `4494.211153` | `0.829392501306833` | `0.8295167954115789` |

MLX-LM runner comparison was attempted with `mlx-lm 0.31.3` and `mlx 0.31.2`
against all seven local MLX-community E2B snapshots. That runner currently
fails at model load with extra Gemma 4 E2B attention K/V parameters, so it is
recorded as a compatibility gap rather than a throughput datapoint. vLLM Metal
uses the same MLX-LM loader surface for these E2B snapshots; the 4bit and 8bit
latency attempts fail at the same load boundary and are recorded as
compatibility artifacts.

## Long-Form Generation Anchors

These are the better production-shaped scores because they allow the model to
produce real text rather than stopping at a 128-token smoke return.

| Shape | Artifact | Result | Decode tok/s | Wall s | Peak GiB | Energy |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| E2B q4 default retained story, two thinking chapters | `docs/runtime/2026-05-19-go-mlx-gemma4-e2b-4bit-default-longform-c2-g8192-energy100w.json` | `1859` generated, `1121` visible | `100.3437506687683` | `19.275618251` | `6.277465732768178` | `1927.5618251 J` |
| E2B q4 retained story, two thinking chapters | `docs/runtime/2026-05-19-go-mlx-gemma4-e2b-q4-fresh-story-thinking-ctx65536-c2-g8192-energy100w.json` | `1767` generated, `1087` visible | `110.35789603546327` | `16.935350541` | `4.489579644054174` | `1693.5350541 J` |
| 26B A4B q4 retained story, two thinking chapters | `docs/runtime/2026-05-19-go-mlx-gemma4-26b-a4b-q4-fresh-story-thinking-ctx65536-c2-g8192-energy100w.json` | `4171` generated, `1033` visible | `73.90526235355026` | `57.559931252` | `20.62171307951212` | `5755.9931252 J` |
| E2B q4 29k-context 8k return | `docs/runtime/2026-05-19-go-mlx-gemma4-e2b-q4-fast-gemma4-lane-r13-ctx65536-g8192-r1-energy100w.json` | `28587` prompt, `8192` generated | `94.92547697253806` | `111.006821417` | `5.134385833516717` | `11100.6821417 J` |
| E2B BF16 29k-context 8k return | `docs/runtime/2026-05-19-go-mlx-gemma4-e2b-bf16-fast-gemma4-lane-r13-ctx65536-g8192-r1-energy100w.json` | `28587` prompt, `8192` generated | `26.59615320070758` | `334.4575525` | `12.643188176676631` | `33445.75525 J` |

The default retained-story row is the current no-extra-fast-flag CLI path:
`chapter-profile` defaults to the accepted Gemma 4 fast gates, `65536` context,
`8192` chapter token budget, paged cache mode, and `512` token prefill chunks.
On the real 8k-return profile, E2B q4 is `3.569x` faster on decode,
`3.013x` lower wall time and estimated energy, and uses `0.406x` the peak
memory versus BF16. On the retained-story profile, E2B q4 produces a comparable
two-chapter artifact `3.399x` faster wall-clock than the 26B A4B q4 story run,
at `0.294x` the estimated energy.

## Improvement Landed

MXFP4 initially panicked during prefill in the compiled GELU path because the
top-level quantization config said `mxfp4`, while each MLP projection carries a
per-weight affine 8-bit override shape. The loader now detects when a non-affine
default does not match a weight/scales tensor pair and infers the affine
group-64 override instead. The fixed MXFP4 README profile now completes at
`109.19709288036368 tok/s`.

Historical artefact names:

The metric table above is the current source for these short-latency numbers,
but the raw JSON/stderr files named below are not present in the current tree.
Recover or rerun them before treating this matrix as replay-grade evidence for
the production gate.

- `docs/runtime/2026-05-19-go-mlx-gemma4-e2b-mxfp4-v0311-quant-matrix-3run-readme-energy100w.json`
- `docs/runtime/2026-05-19-go-mlx-gemma4-e2b-mxfp8-v0311-quant-matrix-3run-readme-energy100w.json`
- `docs/runtime/2026-05-19-go-mlx-gemma4-e2b-4bit-v0311-quant-matrix-3run-readme-energy100w.json`
- `docs/runtime/2026-05-19-go-mlx-gemma4-e2b-5bit-v0311-quant-matrix-3run-readme-energy100w.json`
- `docs/runtime/2026-05-19-go-mlx-gemma4-e2b-6bit-v0311-quant-matrix-3run-readme-energy100w.json`
- `docs/runtime/2026-05-19-go-mlx-gemma4-e2b-8bit-v0311-quant-matrix-3run-readme-energy100w.json`
- `docs/runtime/2026-05-19-go-mlx-gemma4-e2b-bf16-v0311-quant-matrix-3run-readme-energy100w.json`
- `docs/runtime/2026-05-19-llamacpp-gemma4-e2b-q4-k-m-p2282-g128-bench.json`
- `docs/runtime/2026-05-19-llamacpp-gemma4-e2b-q8-0-p2282-g128-bench.json`
- `docs/runtime/2026-05-19-mlx-lm-gemma4-e2b-4bit-quant-matrix-readme-g128.stderr`
- `docs/runtime/2026-05-19-mlx-lm-gemma4-e2b-8bit-quant-matrix-readme-g128.stderr`
- `docs/runtime/2026-05-19-vllm-metal-gemma4-e2b-4bit-readme-shape-b1-latency.stderr`
- `docs/runtime/2026-05-19-vllm-metal-gemma4-e2b-8bit-readme-shape-b1-latency.stderr`

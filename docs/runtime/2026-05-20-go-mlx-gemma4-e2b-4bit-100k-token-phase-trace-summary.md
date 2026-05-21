<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# 100k Token-Phase Trace Summary

Date: 2026-05-20

This is a compact summary of two current shared-full-K/V trace probes:

- `/private/tmp/go-mlx-e2b-100k-shared-fullkv-token-phase-r1.json`, a normal
  `-trace-token-phases` run without forced native-event materialisation.
- `/private/tmp/go-mlx-e2b-100k-shared-fullkv-native-trace-r1.json`, a
  diagnostic `GO_MLX_TRACE_FORWARD_EVAL=1` run with per-layer native events.

The native-event raw JSON is about `17 MB` because it contains `1024`
per-token phase records with per-layer events, so this note records the replay
commands and derived buckets instead of adding the full trace to the production
manifest.

## Command

```sh
env MLX_METALLIB_PATH=/Users/snider/Code/core/go-mlx/dist/lib/mlx.metallib \
  /Users/snider/Code/core/go-mlx/bin/lthn-mlx driver-profile \
  -report-file /private/tmp/go-mlx-e2b-100k-shared-fullkv-token-phase-r1.json \
  -fast-gemma4-lane \
  -context 131072 \
  -prompt-file /Users/snider/Code/core/go-mlx/README.md \
  -prompt-repeat 46 \
  -prompt-suffix "\n\nContinue the agentic workflow with a concrete implementation step and preserve prior state." \
  -max-tokens 1024 \
  -runs 1 \
  -include-output=false \
  -estimate-power-watts 100 \
  -trace-token-phases \
  -max-active-memory-bytes 12884901888 \
  -max-process-resident-memory-bytes 12884901888 \
  /Users/snider/.cache/huggingface/hub/models--mlx-community--gemma-4-e2b-it-4bit/snapshots/99d9a53ff828d365a8ecae538e45f80a08d612cd
```

The native-event trace uses the same command with
`GO_MLX_TRACE_FORWARD_EVAL=1` and
`-report-file /private/tmp/go-mlx-e2b-100k-shared-fullkv-native-trace-r1.json`.

## Run Summary

The normal token-phase probe matches the current shared-full-K/V production
shape closely enough to preserve the accepted `60 tok/s` band. The native-event
trace is diagnostic only: forcing intermediate materialisation slows decode
materially, so the `21.207 tok/s` native-event number must not replace the
accepted untraced `60.011 tok/s` production row.

| Metric | Value |
| --- | ---: |
| Prompt tokens | `101005` |
| Generated tokens | `1024` |
| Normal token-phase total wall | `77.260729709s` |
| Normal first token / prefill | `60.180820375s` / `1682.068440 tok/s` |
| Normal decode throughput | `59.957460 tok/s` |
| Native-event total wall | `117.882639750s` |
| Native-event first token / prefill | `69.469968583s` / `1454.035227 tok/s` |
| Native-event decode throughput | `21.206863 tok/s` |
| Active MLX memory | `3984053838` bytes |
| Cache memory | `5801428840` bytes normal, `6248824400` bytes native-event |
| Process RSS | `3373875200` bytes normal, `3386048512` bytes native-event |
| Estimated energy at `100 W` | `7726.073 J` normal, `11788.264 J` native-event |

## Token-Phase Buckets

Derived from:

```sh
jq 'reduce .runs[0].metrics.token_phases[] as $p
  ({count:0,total_ns:0,forward_ns:0,sample_eval_ns:0,logits_ns:0,other_ns:0};
   .count += 1
   | .total_ns += ($p.total_duration // 0)
   | .forward_ns += ($p.forward_duration // 0)
   | .sample_eval_ns += ($p.sample_eval_duration // 0)
   | .logits_ns += ($p.logits_duration // 0)
   | .other_ns += ($p.other_duration // 0))' \
  /private/tmp/go-mlx-e2b-100k-shared-fullkv-token-phase-r1.json
```

| Bucket | Total |
| --- | ---: |
| Token phases | `1024` |
| Total normal decode-loop time | `17.078322332s` |
| Sample/eval | `15.771446303s` |
| Forward graph construction | `1.279341924s` |
| Next input | `0.013136146s` |
| Other | `0.001767183s` |

Without forced native-event tracing, Go-side forward graph construction is only
about `1.251ms/token`; the lazy graph synchronisation still lands in
`sample_eval` at about `15.402ms/token`.

With `GO_MLX_TRACE_FORWARD_EVAL=1`, the same shared-full-K/V shape records
`48.283068809s` traced decode-loop time. That splits into `47.592696279s`
forward materialisation (`46.523ms/token`) and `0.673812733s` sample/eval
(`0.658ms/token`). The trace overhead is intentional: it moves the hidden MLX
work out of `sample_eval` and into named native buckets.

## Native Event Buckets

| Bucket | Count | Total | Average |
| --- | ---: | ---: | ---: |
| Attention | `35805` | `18.981869088s` | `0.530145ms` |
| Output | `35805` | `10.317275666s` | `0.288151ms` |
| FFN | `35805` | `9.313775357s` | `0.260124ms` |
| Attention residual | `35805` | `7.136504981s` | `0.199315ms` |

## Attention Layer Split

The expensive attention layers are still the Gemma 4 full-attention owners. The
shared full-K/V reuse change is visible here: the later shared full-attention
layers now sit around `1.03ms/token`, while early owner layers remain near
`1.96-1.98ms/token`.

| Layer | Total | Average per generated token |
| --- | ---: | ---: |
| `gemma4.layer.04.attention` | `2.022539536s` | `1.977067ms` |
| `gemma4.layer.14.attention` | `2.012931386s` | `1.967675ms` |
| `gemma4.layer.09.attention` | `2.002039955s` | `1.957028ms` |
| `gemma4.layer.29.attention` | `1.059230046s` | `1.035415ms` |
| `gemma4.layer.34.attention` | `1.056698051s` | `1.032940ms` |
| `gemma4.layer.19.attention` | `1.053443280s` | `1.029759ms` |
| `gemma4.layer.24.attention` | `1.049440184s` | `1.025846ms` |

The next runtime target is therefore the full-attention paged/global K/V path,
not restore, token sampling, or broad CGO wrapper work. Local sliding-attention
layers are present in the trace but sit around the `0.29-0.37ms` band. The
remaining attention target is narrower than before: reduce owner-layer
full-attention K/V work for layers `4`, `9`, and `14` without reintroducing the
full fixed-cache active-memory blowout.

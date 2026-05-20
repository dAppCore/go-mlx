<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# 100k Token-Phase Trace Summary

Date: 2026-05-20

This is a compact summary of the raw trace generated at
`/private/tmp/go-mlx-e2b-100k-trace-g1024-r1.json`. The raw JSON is about
`17 MB` because it contains `1024` per-token phase records with per-layer native
events, so this note records the replay command and derived buckets instead of
adding the full trace to the production manifest.

## Command

```sh
env MLX_METALLIB_PATH=/Users/snider/Code/core/go-mlx/dist/lib/mlx.metallib \
  GO_MLX_TRACE_FORWARD_EVAL=1 \
  /Users/snider/Code/core/go-mlx/bin/lthn-mlx driver-profile \
  -report-file /private/tmp/go-mlx-e2b-100k-trace-g1024-r1.json \
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

## Run Summary

The trace run is diagnostic only. Trace hooks slow decode materially, so the
`19.026 tok/s` decode number must not replace the accepted untraced `51.293
tok/s` production baseline.

| Metric | Value |
| --- | ---: |
| Prompt tokens | `100932` |
| Generated tokens | `1024` |
| Total wall | `124.398033s` |
| First token / prefill | `70.578236s` / `70.459088s` |
| Decode duration | `53.821633s` |
| Decode throughput with trace overhead | `19.025807 tok/s` |
| Active MLX memory | `3902592590` bytes |
| Cache memory | `6637277800` bytes |
| Process RSS | `3366092800` bytes |
| Process virtual reservation | `602661699584` bytes |
| Estimated energy at `100 W` | `12439.8033 J` |

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
  /private/tmp/go-mlx-e2b-100k-trace-g1024-r1.json
```

| Bucket | Total |
| --- | ---: |
| Token phases | `1024` |
| Total traced decode-loop time | `53.816603233s` |
| Forward materialisation | `53.083827410s` |
| Sample/eval | `0.707828075s` |
| Logits | `0.000632015s` |
| Other | `0.003727168s` |

The decode loss is therefore not driver bookkeeping. It is almost entirely the
lazy forward materialisation that happens when each next token is forced.

## Native Event Buckets

| Bucket | Count | Total | Average |
| --- | ---: | ---: | ---: |
| Attention | `35805` | `22.745016951s` | `0.635247ms` |
| Output | `35805` | `10.642778362s` | `0.297243ms` |
| FFN | `35805` | `9.909272722s` | `0.276757ms` |
| Attention residual | `35805` | `7.816795192s` | `0.218316ms` |

## Attention Layer Split

The expensive attention layers are the Gemma 4 full-attention owners. They are
the every-fifth layers in the local/full pattern, and dominate the trace:

| Layer | Total | Average per generated token |
| --- | ---: | ---: |
| `gemma4.layer.04.attention` | `2.074647441s` | `2.028003ms` |
| `gemma4.layer.09.attention` | `2.054151433s` | `2.007968ms` |
| `gemma4.layer.14.attention` | `2.047648082s` | `2.001611ms` |
| `gemma4.layer.34.attention` | `1.883382378s` | `1.841038ms` |
| `gemma4.layer.19.attention` | `1.878529132s` | `1.836294ms` |
| `gemma4.layer.24.attention` | `1.878259219s` | `1.836031ms` |
| `gemma4.layer.29.attention` | `1.873139219s` | `1.831026ms` |

The next runtime target is therefore the full-attention paged/global K/V path,
not restore, token sampling, or broad CGO wrapper work. Local sliding-attention
layers are present in the trace but sit around the `0.3-0.4ms` band, while the
full-attention layers sit near `1.8-2.0ms` each under trace overhead.

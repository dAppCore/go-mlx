<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# 100k Token-Phase Trace Summary

Date: 2026-05-21

This is the refreshed compact trace for the promoted hyper-long fp16 paged-K/V
lane. It replaces the older shared-full-K/V-only trace while preserving the
same workload shape:

- `/private/tmp/go-mlx-e2b-100k-fp16kv-token-phase-r1.json`, a normal
  `-trace-token-phases` run without forced native-event materialisation.
- `/private/tmp/go-mlx-e2b-100k-fp16kv-native-trace-r1.json`, a diagnostic
  `GO_MLX_TRACE_FORWARD_EVAL=1` run with per-layer native events.

The native-event raw JSON is about `17 MB` because it contains `1024`
per-token phase records with per-layer events, so this note records the replay
commands and derived buckets instead of adding the full trace to the production
manifest.

## Command

```sh
env MLX_METALLIB_PATH=/Users/snider/Code/core/go-mlx/dist/lib/mlx.metallib \
  GOWORK=/Users/snider/Code/core/go-mlx/go.work \
  GOCACHE=/private/tmp/codex-go-mlx-cache \
  /private/tmp/go-mlx-current-trace/lthn-mlx driver-profile \
  -report-file /private/tmp/go-mlx-e2b-100k-fp16kv-token-phase-r1.json \
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
`-report-file /private/tmp/go-mlx-e2b-100k-fp16kv-native-trace-r1.json`.

## Run Summary

The normal token-phase probe matches the current promoted production shape:
hyper-long paged K/V uses `1024`-token pages and stores restored K/V as fp16.
The diagnostic native-event run is still slower because it intentionally forces
intermediate materialisation; it must not replace the accepted untraced
`76.018 tok/s` 10-run production row.

| Metric | Normal fp16 K/V | Native-event diagnostic |
| --- | ---: | ---: |
| Prompt tokens | `100932` | `100932` |
| Generated tokens | `1024` | `1024` |
| Total wall | `66.943334625s` | `107.568992750s` |
| First token / prefill | `53.445116166s` / `1892.571781 tok/s` | `62.141185917s` / `1627.587177 tok/s` |
| Decode throughput | `75.858987 tok/s` | `22.541137 tok/s` |
| Active MLX memory | `3472447054` bytes | `3472430670` bytes |
| Cache memory | `6549661092` bytes | `6360830576` bytes |
| Process RSS | `3398680576` bytes | `3365502976` bytes |
| Estimated energy at `100 W` | `6694.333 J` | `10756.899 J` |

## Token-Phase Buckets

Derived from:

```sh
jq 'reduce .runs[0].metrics.token_phases[] as $p
  ({count:0,total_ns:0,forward_ns:0,sample_eval_ns:0,next_input_ns:0,other_ns:0};
   .count += 1
   | .total_ns += ($p.total_duration // 0)
   | .forward_ns += ($p.forward_duration // 0)
   | .sample_eval_ns += ($p.sample_eval_duration // 0)
   | .next_input_ns += ($p.next_input_duration // 0)
   | .other_ns += ($p.other_duration // 0))' \
  /private/tmp/go-mlx-e2b-100k-fp16kv-token-phase-r1.json
```

| Bucket | Normal fp16 K/V | Native-event diagnostic |
| --- | ---: | ---: |
| Token phases | `1024` | `1024` |
| Total decode-loop time | `13.498352036s` | `45.427755330s` |
| Sample/eval | `12.253825634s` | `0.696081414s` |
| Forward graph construction/materialisation | `1.208567074s` | `44.709807077s` |
| Next input | `0.013075331s` | `0.008495334s` |
| Other | `0.001643749s` | `0.003111974s` |

Without forced native-event tracing, Go-side forward graph construction is
about `1.181ms/token`; the lazy MLX synchronisation still lands in
`sample_eval` at about `11.967ms/token`.

With `GO_MLX_TRACE_FORWARD_EVAL=1`, the same fp16 K/V shape records
`45.428s` traced decode-loop time. That splits into `44.710s` forward
materialisation (`43.705ms/token`) and `0.696s` sample/eval (`0.680ms/token`).
The trace overhead is intentional: it moves hidden MLX work out of
`sample_eval` and into named native buckets.

## Native Event Buckets

| Bucket | Count | Total | Average |
| --- | ---: | ---: | ---: |
| Attention | `35805` | `15.537483359s` | `0.433947ms` |
| Output | `35805` | `10.387081047s` | `0.290101ms` |
| FFN | `35805` | `9.657761730s` | `0.269732ms` |
| Attention residual | `35805` | `7.416089181s` | `0.207124ms` |

## Attention Layer Split

The expensive attention layers remain the Gemma 4 full-attention owners. The
fp16 K/V promotion moved the owner layers down from the older `1.96-1.98ms`
band to about `1.38ms/token`, and moved later shared full-attention layers down
from about `1.03ms/token` to about `0.625ms/token`. That is a real gain, but
the owner layers are still the dominant long-context attention cost.

| Layer | Total | Average per generated token |
| --- | ---: | ---: |
| `gemma4.layer.04.attention` | `1.418512132s` | `1.386620ms` |
| `gemma4.layer.14.attention` | `1.414508359s` | `1.382706ms` |
| `gemma4.layer.09.attention` | `1.413532095s` | `1.381752ms` |
| `gemma4.layer.34.attention` | `0.641025116s` | `0.626613ms` |
| `gemma4.layer.19.attention` | `0.640309167s` | `0.625913ms` |
| `gemma4.layer.24.attention` | `0.639849376s` | `0.625464ms` |
| `gemma4.layer.29.attention` | `0.639545913s` | `0.625167ms` |

The current next runtime target is still the full-attention owner paged/global
K/V path, not restore, token sampling, broad CGO wrapping, or short-context
matvec work. The refreshed diagnostics also rechecked two obvious branches on
the fp16 K/V lane:

- `GO_MLX_ENABLE_PAGED_FULL_KV_MATERIALIZE=1` records `75.565369 tok/s` and
  raises active MLX memory to `3875100238` bytes, so retaining a pure MLX full
  backing tensor for owner layers remains rejected.
- `-native-gemma4-attention-o-matvec` records `75.780083 tok/s`, which is flat
  against the normal `75.858987 tok/s` trace row, so attention O-projection
  matvec remains diagnostic and should not be promoted for the hyper-long lane.

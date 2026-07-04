<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# Gemma 4 6-bit Chapter Profile Baselines

Captured on 2026-06-05 with the go-mlx CLI and the downloaded
`mlx-community` 6-bit Gemma 4 family packs. These are `chapter-profile` runs,
not synthetic `driver-profile` prompt smokes.

## Runtime

- Binary: `/private/tmp/go-mlx-self/bin/lthn-mlx`
- Worktree: `/Users/snider/Code/core/go-mlx`
- Go workspace: `/Users/snider/Code/core/go-mlx/go.work`
- Go cache: `/private/tmp/go-mlx-self/gocache`
- Metal library: `/Users/snider/Code/core/go-mlx/dist/lib/mlx.metallib`
- Build flags: `-ldflags "-extldflags=-mmacosx-version-min=26.0"`
- Cache mode: `paged`
- Chapters: `1`
- Output: enabled through `-include-output` and `-output-file`

## Baselines

| Pack | Snapshot | Report | Generated tokens | Decode tok/s | Prefill tok/s | Active+cache bytes | Peak bytes | Cache profile |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| E2B q6 | `/Users/snider/.cache/huggingface/hub/models--mlx-community--gemma-4-e2b-it-6bit/snapshots/40d43b05f94ee798c0e40fe19fcd9ef49928486b` | `/private/tmp/go-mlx-self/reports/gemma4-e2b-q6-chapter-profile-uncapped-native-1.json` | 1,499 | 68.76 | 1108.38 | 9,400,629,338 | 4,028,025,290 | 15 caches, 12 local, 3 global, 20 shared layers, 512 local window, no local-window leak |
| E4B q6 | `/Users/snider/.cache/huggingface/hub/models--mlx-community--gemma-4-e4b-it-6bit/snapshots/d786394b6a0cfb1cebb74bac11d81fcb1b3ce8c8` | `/private/tmp/go-mlx-self/reports/gemma4-e4b-q6-chapter-profile-uncapped-native-1.json` | 1,495 | 47.09 | 452.81 | 12,927,586,884 | 6,411,030,952 | 24 caches, 20 local, 4 global, 18 shared layers, 512 local window, no local-window leak |
| 12B Unified q6 | `/Users/snider/.cache/huggingface/hub/models--mlx-community--gemma-4-12B-it-6bit/snapshots/f0d6f5d34239a612f695362750044905e6dd072c` | `/private/tmp/go-mlx-self/reports/gemma4-12b-it-q6-chapter-profile-uncapped-native-word-safe-1.json` | 2,019 | 33.04 | 635.54 | 19,239,393,780 | 12,757,909,568 | 48 caches, 40 local, 8 global, 1024 local window, no local-window leak |

These reports were captured before the 2026-06-05 cleanup that split the
user-facing `chapter_max_tokens` request from the internal backend generation
budget. They completed naturally before the backend budget, so the throughput
numbers remain useful as current baselines, but fresh accepted reports should
show `chapter_max_tokens: 0` when the command is run without
`-chapter-max-tokens`.

Fresh reports also include Go allocation deltas for the actual generation turn:
`memory_delta.go_total_alloc_delta_bytes`, `memory_delta.go_mallocs_delta`, and
summary-level `go_bytes_per_generated_token` /
`go_allocs_per_generated_token`. Record those with tok/s and MLX memory for the
next optimisation pass.

## Failed probes

| Pack | Report | Generated tokens | Decode tok/s | Active+cache bytes | Outcome |
| --- | --- | ---: | ---: | ---: | --- |
| 12B Unified q6 | `/private/tmp/go-mlx-self/reports/gemma4-12b-it-q6-chapter-profile-uncapped-native-1.json` | 16,000 | 30.45 | 19,698,793,748 | manually aborted after visible output collapsed into repeated `order-` / `0` runs |
| 12B Unified q6 | `/private/tmp/go-mlx-self/reports/gemma4-12b-it-q6-chapter-profile-uncapped-native-loop-safe-1.json` | 7,390 | 31.95 | 19,417,208,104 | manually aborted after visible output collapsed into repeated `neighbors`; token-id safety alone was insufficient |
| 31B q6 | `/private/tmp/go-mlx-self/reports/gemma4-31b-q6-chapter-profile-uncapped-native-word-safe-1.json` | 96 | 13.52 | 32,173,312,424 | stopped by repeated visible word `same`; load/generate worked, quality did not |
| 26B A4B MoE q6 | `/private/tmp/go-mlx-self/reports/gemma4-26b-a4b-q6-chapter-profile-uncapped-native-word-safe-1.json` | 841 | 38.53 | 27,781,603,808 | stopped by repeated visible word `termination`; load/generate worked, quality did not |
| E2B q6 post-cleanup | `/private/tmp/go-mlx-self/reports/gemma4-e2b-q6-chapter-profile-postfix-uncapped-request-1.json` | 0 | 0 | 0 | failed before load: `metal.LoadAndInit: select device: mlx: no usable Metal device available`; report confirms `chapter_max_tokens: 0`, but this is not a performance baseline |

## Gate Diagnostics

These are not chapter baselines. They are narrow off/on checks for cleanup
decisions around experimental runtime gates.

| Gate | Pack | Off report | On report | Generated tokens | Output token hash | Off decode tok/s | On decode tok/s | Off active+cache bytes | On active+cache bytes | Result |
| --- | --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| `NATIVE_GEMMA4_MODEL_GREEDY` | E2B q6 | `/private/tmp/go-mlx-self/reports/gemma4-e2b-q6-model-greedy-off.json` | `/private/tmp/go-mlx-self/reports/gemma4-e2b-q6-model-greedy-on.json` | 2,595 | `18ce8de9f6f972df6c916b362591ea6765a740fff258b4ffc25ee192a8c3dd87` | 71.130 | 71.101 | n/a | n/a | parity, no decode win; gate and branch deleted |
| `PAGED_KV_PREALLOC` | E2B q6 | `/private/tmp/go-mlx-self/reports/gemma4-e2b-q6-paged-kv-prealloc-off.json` | `/private/tmp/go-mlx-self/reports/gemma4-e2b-q6-paged-kv-prealloc-on.json` | 2,595 | `18ce8de9f6f972df6c916b362591ea6765a740fff258b4ffc25ee192a8c3dd87` | 71.416 | 70.433 | 5,576,000,330 | 4,308,684,758 | parity and lower MLX residency, but no decode win; reclassified as explicit memory-mode load option, not default |

## Commands

Baseline command shape:

```sh
env GOWORK=/Users/snider/Code/core/go-mlx/go.work GOCACHE=/private/tmp/go-mlx-self/gocache MLX_METALLIB_PATH=/Users/snider/Code/core/go-mlx/dist/lib/mlx.metallib /private/tmp/go-mlx-self/bin/lthn-mlx chapter-profile -json -chapters 1 -cache-mode paged -include-output -report-file REPORT.json -output-file OUTPUT.md MODEL_SNAPSHOT
```

Post-cleanup failed probe command:

```sh
env GOWORK=/Users/snider/Code/core/go-mlx/go.work GOCACHE=/private/tmp/go-mlx-self/gocache MLX_METALLIB_PATH=/Users/snider/Code/core/go-mlx/dist/lib/mlx.metallib /private/tmp/go-mlx-self/bin/lthn-mlx chapter-profile -json -chapters 1 -cache-mode paged -include-output -report-file /private/tmp/go-mlx-self/reports/gemma4-e2b-q6-chapter-profile-postfix-uncapped-request-1.json -output-file /private/tmp/go-mlx-self/reports/gemma4-e2b-q6-chapter-profile-postfix-uncapped-request-1.md /Users/snider/.cache/huggingface/hub/models--mlx-community--gemma-4-e2b-it-6bit/snapshots/40d43b05f94ee798c0e40fe19fcd9ef49928486b
```

Current runtime discovery after the failed probe:

```sh
env GOWORK=/Users/snider/Code/core/go-mlx/go.work GOCACHE=/private/tmp/go-mlx-self/gocache MLX_METALLIB_PATH=/Users/snider/Code/core/go-mlx/dist/lib/mlx.metallib /private/tmp/go-mlx-self/bin/lthn-mlx discover -json
```

Discovery saw `Apple M3 Ultra` but reported `load_available=false`; native
model load and benchmark capabilities were therefore unsupported at that moment.

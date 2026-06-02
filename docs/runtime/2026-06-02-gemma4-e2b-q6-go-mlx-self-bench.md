<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# Gemma 4 E2B q6 go-mlx Self-Benchmark

Date: 2026-06-02

Purpose: compare the current go-mlx q6 default gate set against the same binary
with candidate gates isolated. This is a go-mlx-vs-go-mlx regression guard, not
an external runner parity claim.

Model:

`mlx-community/gemma-4-e2b-it-6bit`
snapshot `40d43b05f94ee798c0e40fe19fcd9ef49928486b`

Command shape:

```bash
env MLX_METALLIB_PATH=/Users/snider/Code/core/go-mlx/dist/lib/mlx.metallib \
  /private/tmp/go-mlx-self/bin/lthn-mlx driver-profile \
  -report-file /private/tmp/go-mlx-self/q6-default-current.json \
  -include-output=false \
  -context 4096 \
  -prompt "Write a concise technical note explaining why retained model state matters for repeated agent workflows." \
  -max-tokens 512 \
  -runs 3 \
  -estimate-power-watts 75 \
  -trace-token-phases=false \
  /Users/snider/.cache/huggingface/hub/models--mlx-community--gemma-4-e2b-it-6bit/snapshots/40d43b05f94ee798c0e40fe19fcd9ef49928486b
```

The baseline adds `-fast-gemma4-lane=false`. Isolated gate rows also add one of
`-direct-greedy-token` or `-paged-decode-fast-concat` while keeping
`-fast-gemma4-lane=false`.

Raw JSON was written during the run to:

- `/private/tmp/go-mlx-self/q6-default-current.json`
- `/private/tmp/go-mlx-self/q6-baseline-fast-off-current.json`
- `/private/tmp/go-mlx-self/q6-direct-greedy-only-current.json`
- `/private/tmp/go-mlx-self/q6-paged-fast-concat-only-current.json`
- `/private/tmp/go-mlx-self/q6-default-patched-rerun.json`

## Results

| Lane | Runtime gates | Generated tokens | Decode tok/s | Wall time | Energy at 75 W | Peak RSS | Active+cache | Output hash |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Baseline | none | 1164 | 90.05828489228817 | 13.135163001s | 985.137225075 J | 4202987520 B | 4414100477 B | `ea621e942f414fde824380a89a39cd120283fe303e34a5930b7a046c950a6754` |
| Direct greedy only | `GO_MLX_ENABLE_DIRECT_GREEDY_TOKEN=1` | 1164 | 90.00163414721517 | 13.139377125s | 985.453284375 J | 4221779968 B | 4414034316 B | `ea621e942f414fde824380a89a39cd120283fe303e34a5930b7a046c950a6754` |
| Paged fast concat only | `GO_MLX_ENABLE_PAGED_DECODE_FAST_CONCAT=1` | 1164 | 89.1661883374603 | 13.267558418s | 995.06688135 J | 4203577344 B | 4413838333 B | `ea621e942f414fde824380a89a39cd120283fe303e34a5930b7a046c950a6754` |
| Previous default combination | `GO_MLX_ENABLE_DIRECT_GREEDY_TOKEN=1`, `GO_MLX_ENABLE_PAGED_DECODE_FAST_CONCAT=1` | 1164 | 66.04699256254571 | 19.379504541s | 1453.462840575 J | 4175757312 B | 4414659581 B | `ea621e942f414fde824380a89a39cd120283fe303e34a5930b7a046c950a6754` |
| Patched default rerun | `GO_MLX_ENABLE_DIRECT_GREEDY_TOKEN=1` | 1164 | 89.05331759347916 | 13.275426084s | 995.6569563 J | 4215341056 B | 4413571468 B | `ea621e942f414fde824380a89a39cd120283fe303e34a5930b7a046c950a6754` |

The current fresh binary rejects the combined short-context default. Each gate
is individually healthy on this q6 prompt shape, but the combination drops
decode to `0.7334x` of the baseline and raises wall time and estimated energy by
`47.54%`.

The promoted short-context default is therefore narrowed to
`GO_MLX_ENABLE_DIRECT_GREEDY_TOKEN=1`. The patched default rerun confirms the
CLI default itself now reports only that gate and returns to the baseline band.
`GO_MLX_ENABLE_PAGED_DECODE_FAST_CONCAT` remains available as an explicit
diagnostic flag and should only be promoted again after a retained-workflow
self-benchmark shows a net wall-time win.

All four rows produced the same generated token hash and `1164` total generated
tokens, so the gate decision is about runtime cost, not visible output drift.

## Refresh After External/Core Optimisation Pass

After the CoreGO, go-inference, and go-cgo optimisation pass, the same source
tree was rebuilt into `/private/tmp/go-mlx-self/bin/lthn-mlx` and the same q6
prompt shape was rerun as a go-mlx-vs-go-mlx self-benchmark. This refresh uses
the current binary only; it is not an external runner comparison.

| Lane | Runtime gates | Generated tokens | Decode tok/s | Wall time | Energy at 75 W | Peak RSS | Active+cache |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Current default refresh | `GO_MLX_ENABLE_DIRECT_GREEDY_TOKEN=1` | 1164 | `90.55655393185161` | `13.060853417s` | `979.564006275 J` | `4217913344 B` | `4413826045 B` |
| Fast lane off refresh | none | 1164 | `90.4228600340199` | `13.081736499s` | `981.130237425 J` | `4218142720 B` | `4414245885 B` |
| Forced old combined refresh | `GO_MLX_ENABLE_DIRECT_GREEDY_TOKEN=1`, `GO_MLX_ENABLE_PAGED_DECODE_FAST_CONCAT=1` | 1164 | `90.27815337757984` | `13.0991055s` | `982.4329125 J` | `4203757568 B` | `4413744125 B` |

The previous `66 tok/s` combined-gate failure no longer reproduces on the
current stack. The forced combined row is only `0.31%` slower than the current
default and `0.16%` slower than the fast-lane-off baseline on this short q6
shape, so the old failure appears fixed by later runtime/external changes.
This does not automatically promote `GO_MLX_ENABLE_PAGED_DECODE_FAST_CONCAT`
back into the default lane; promotion still needs a retained-workflow
self-benchmark where wall time, memory, and output parity all hold.

## Current Source Refresh

After the pack-report Python fallback cleanup and latest external refreshes, the
same q6 prompt shape was rebuilt into `/private/tmp/go-mlx-self/bin/lthn-mlx`
and rerun as a three-row go-mlx-vs-go-mlx self-benchmark.

| Lane | Runtime gates | Generated tokens | Decode tok/s | Wall time | Energy at 75 W | Peak RSS | Active+cache | Process virtual | Output hash |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Current default refresh | `GO_MLX_ENABLE_DIRECT_GREEDY_TOKEN=1` | 1164 | `89.8999819954319` | `13.155227709s` | `986.642078175 J` | `4021.5 MiB` | `4209.6 MiB` | `421.2 GiB` | `ea621e942f414fde824380a89a39cd120283fe303e34a5930b7a046c950a6754` |
| Fast lane off refresh | none | 1164 | `90.28679966071449` | `13.098565207s` | `982.392390525 J` | `4013.3 MiB` | `4209.3 MiB` | `421.2 GiB` | `ea621e942f414fde824380a89a39cd120283fe303e34a5930b7a046c950a6754` |
| Forced old combined refresh | `GO_MLX_ENABLE_DIRECT_GREEDY_TOKEN=1`, `GO_MLX_ENABLE_PAGED_DECODE_FAST_CONCAT=1` | 1164 | `90.70567768488039` | `13.039203415s` | `977.940256125 J` | `4013.2 MiB` | `4209.2 MiB` | `421.2 GiB` | `ea621e942f414fde824380a89a39cd120283fe303e34a5930b7a046c950a6754` |

The forced old combined gate is now the local winner on this short q6
self-bench: `0.90%` faster than the current default and `0.46%` faster than
fast-lane-off, with the same output hash and no memory increase. This remains a
short-context driver-profile result; the production decision still belongs to a
retained workflow self-benchmark because the goal optimises repeated stateful
turns, not a single isolated prompt.

## Retained Workflow Gate Check

The same source binary was then checked with a book-shaped retained workflow
using the `scripts/state_book_from_phase0.py` material generator's seed and turn
files, run directly through `state-ramp-profile` from `/private/tmp/go-mlx-self`
so Metal device selection matched the normal benchmark lane. Shape:

- model: `mlx-community/gemma-4-e2b-it-6bit`
- context: `32768`
- seed state: `4096` tokens
- turns: `10`
- append budget: `512` tokens per turn
- generation budget: `512` tokens per turn
- prompt materials: `/private/tmp/go-mlx-self/book-runs/2026-06-02-c002-poetry-time-seed60201.seed.txt`
  and `/private/tmp/go-mlx-self/book-runs/2026-06-02-c002-poetry-time-seed60201.turns.txt`

| Lane | Runtime gates | RNG seed | Successful turns | Generated tokens | Decode tok/s | Effective turn tok/s | Wall time | Replay estimate | Saved replay setup | Retained speedup | Energy at 75 W | Active+cache | Process virtual |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Current default retained | `GO_MLX_ENABLE_DIRECT_GREEDY_TOKEN=1` | `123` | 10 | 871 | `77.07749725770038` | `68.09373741617839` | `14.257772919s` | `33.878033324s` | `19.620260405s` | `2.3761x` | `1069.332968925 J` | `5805.3 MiB` | `422.8 GiB` |
| Forced old combined retained | `GO_MLX_ENABLE_DIRECT_GREEDY_TOKEN=1`, `GO_MLX_ENABLE_PAGED_DECODE_FAST_CONCAT=1` | `123` | 10 | 2575 | `79.9306857602304` | `76.31642807228945` | `35.206650004s` | `54.899240921s` | `19.692590917s` | `1.5593x` | `2640.4987503 J` | `9233.6 MiB` | `426.2 GiB` |
| Current default retained | `GO_MLX_ENABLE_DIRECT_GREEDY_TOKEN=1` | unset | 10 | 1507 | `76.58961441991917` | `71.03835202114963` | `22.679124753s` | `42.598209033s` | `19.91908428s` | `1.8783x` | `1700.934356475 J` | `5840.2 MiB` | `422.8 GiB` |
| Forced old combined retained | `GO_MLX_ENABLE_DIRECT_GREEDY_TOKEN=1`, `GO_MLX_ENABLE_PAGED_DECODE_FAST_CONCAT=1` | unset | 10 | 2693 | `79.15688298056841` | `75.65948907278459` | `37.055022793s` | `56.813276865s` | `19.758254072s` | `1.5332x` | `2779.126709475 J` | `10121.6 MiB` | `427.1 GiB` |

The retained workflow rejects promoting `GO_MLX_ENABLE_PAGED_DECODE_FAST_CONCAT`
back into the default lane. It improves raw decode by roughly `3-4%`, but it
changes the sampled retained generation length even with a fixed RNG seed,
raises active+cache memory by multiple GiB, and more than doubles wall time and
estimated energy on the seeded 10-turn run. The current q6 default remains the
production-safe retained setting until a candidate wins on retained wall time,
memory, and output shape rather than isolated short-prompt decode.

The first raw default attempt with the Lemma boot prompt failed on turn 1 with a
repeated-sentence guard, and the raw JSON phase0 attempt failed on turn 5 with a
repeated code-fence line cycle. Those rows are intentionally excluded from the
gate table because they measure prompt/content failure modes, not a valid
10-turn retained runtime comparison.

## Current Comparator Safety Refresh

After the retained state-ramp shape comparator fix, the binary was rebuilt from
commit `5945ad7` and the same short q6 self-benchmark shape was rerun. This is a
go-mlx-vs-go-mlx regression guard only; it does not replace the retained
workflow gate above.

Raw JSON was written during the run to:

- `/private/tmp/go-mlx-self/q6-5945ad7-default.json`
- `/private/tmp/go-mlx-self/q6-5945ad7-fast-off.json`
- `/private/tmp/go-mlx-self/q6-5945ad7-combined.json`

| Lane | Runtime gates | Generated tokens | Decode tok/s | Wall time | Energy at 75 W | Peak memory | Active+cache | Process virtual | Output hash |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Current default | `GO_MLX_ENABLE_DIRECT_GREEDY_TOKEN=1` | 1164 | `88.7358091797484` | `13.329282751s` | `999.696206325 J` | `3978597471 B` | `4414185868 B` | `452290723840 B` | `ea621e942f414fde824380a89a39cd120283fe303e34a5930b7a046c950a6754` |
| Fast lane off | none | 1164 | `87.97583586786008` | `13.439744291s` | `1007.980821825 J` | `3978449616 B` | `4413092861 B` | `452364533760 B` | `ea621e942f414fde824380a89a39cd120283fe303e34a5930b7a046c950a6754` |
| Forced old combined | `GO_MLX_ENABLE_DIRECT_GREEDY_TOKEN=1`, `GO_MLX_ENABLE_PAGED_DECODE_FAST_CONCAT=1` | 1164 | `87.82054036181437` | `13.478652334s` | `1010.89892505 J` | `3977990235 B` | `4414169484 B` | `452364779520 B` | `ea621e942f414fde824380a89a39cd120283fe303e34a5930b7a046c950a6754` |

The current default remains the short-shape winner on this refresh: `0.86%`
faster than fast-lane-off and `1.04%` faster than the forced old combined gate,
with identical generated token counts and output hashes. The retained-workflow
decision remains unchanged: `GO_MLX_ENABLE_PAGED_DECODE_FAST_CONCAT` is still a
diagnostic-only candidate until it wins on retained wall time, memory, and
output shape.

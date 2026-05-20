<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# Gemma 4 E2B 4bit C006 Report-File Book Run

This note records a current-source `chapter-profile` run that writes the JSON
report through the runner's native `-report-file` path instead of relying on
shell redirection. It is a canonical full-book artifact for the C006 creative
prompt, not a runner-anchor comparison row.

## Command

```sh
env MLX_METALLIB_PATH=/Users/snider/Code/core/go-mlx/dist/lib/mlx.metallib \
  /Users/snider/Code/core/go-mlx/bin/lthn-mlx chapter-profile \
  -report-file /Users/snider/Code/core/go-mlx/docs/runtime/2026-05-20-go-mlx-gemma4-e2b-4bit-c006-book-ctx131072-c10-g8192-min512-thinking-current-energy100w.json \
  -premise "Write a poem that is also a mathematical proof. The emotional arc should mirror the logical arc. The conclusion should be both mathematically inevitable and emotionally devastating." \
  -chapters 10 \
  -chapter-max-tokens 8192 \
  -chapter-min-tokens 512 \
  -output-file /Users/snider/Code/core/go-mlx/docs/runtime/2026-05-20-go-mlx-gemma4-e2b-4bit-c006-book-ctx131072-c10-g8192-min512-thinking-current-book.md \
  -enable-thinking \
  -temperature 1.0 \
  -top-p 0.95 \
  -top-k 64 \
  -context 131072 \
  -prefill-chunk-size 512 \
  -cache-mode paged \
  -estimate-power-watts 100 \
  /Users/snider/.cache/huggingface/hub/models--mlx-community--gemma-4-e2b-it-4bit/snapshots/99d9a53ff828d365a8ecae538e45f80a08d612cd
```

## Accepted Artifacts

- `docs/runtime/2026-05-20-go-mlx-gemma4-e2b-4bit-c006-book-ctx131072-c10-g8192-min512-thinking-current-energy100w.json`
- `docs/runtime/2026-05-20-go-mlx-gemma4-e2b-4bit-c006-book-ctx131072-c10-g8192-min512-thinking-current-book.md`

## Shape

- Model: `mlx-community/gemma-4-e2b-it-4bit`
- Snapshot:
  `/Users/snider/.cache/huggingface/hub/models--mlx-community--gemma-4-e2b-it-4bit/snapshots/99d9a53ff828d365a8ecae538e45f80a08d612cd`
- Prompt: C006 poetry/mathematics premise from
  `/Users/snider/Code/lthn/LEM/training/lem/creative/phase0.json`
- Context: `131072`
- Cache mode: `paged`
- Prefill chunk size: `512`
- Chapters: `10`
- Chapter max tokens: `8192`
- Accepted visible-token floor: `512`
- Thinking: enabled, hidden from appended assistant history
- Sampling: `temperature=1.0`, `top_p=0.95`, `top_k=64`
- Power estimate: normalised `100 W`, not measured power

## Result

| Metric | Value |
| --- | ---: |
| Successful turns | `10/10` |
| Generated / visible tokens | `8201` |
| Chapter visible-token range | `668` to `1351` |
| Total wall time | `105.947s` |
| Average decode | `80.343 tok/s` |
| Average prefill | `2676.126 tok/s` |
| Peak MLX memory | `3.587 GB` |
| Active MLX memory | `3.396 GB` |
| Cache memory | `6.680 GB` |
| Process RSS | `3.611 GB` |
| Process virtual reservation | `638.946 GB` |
| Estimated energy | `10594.699 J` |
| Estimated energy per visible token | `1.292 J/token` |

## Rejected Neighbor

The same report-file path also captured a stricter `chapter_min_tokens=640`
attempt:

- `docs/runtime/2026-05-20-go-mlx-gemma4-e2b-4bit-c006-book-ctx131072-c10-g8192-min640-thinking-current-energy100w.json`
- `docs/runtime/2026-05-20-go-mlx-gemma4-e2b-4bit-c006-book-ctx131072-c10-g8192-min640-thinking-current-book.md`

That run reached chapter 8 and failed only because chapter 8 naturally stopped
at `563` visible tokens, below the `640` floor. It did not fail from OOM,
special-token collapse, max-token truncation, or runner instability. The
accepted `512` floor still rejects tiny smoke responses while preserving a real
10-turn book workload.

<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# Gemma 4 E2B Driver Profile, 2026-05-16

This is the first persisted benchmark artefact for the GOAL.md 100 tok/s lane
after the `lthn-mlx` bundle binary and workspace-aware Taskfile build path were
restored.

## Environment

| Item | Value |
| --- | --- |
| Host | Apple M3 Ultra |
| macOS | 26.4.1, build 25E253 |
| Go | go1.26.2 darwin/arm64 |
| Python | 3.14.4 |
| System Python `mlx` package | 0.30.6 |
| System Python `mlx-lm` package | 0.31.2 |
| Temporary parity venv | `/private/tmp/go-mlx-mlx-lm-venv` |
| Temporary parity venv `mlx` package | 0.31.2 |
| Temporary parity venv `mlx-lm` package | 0.31.3 |
| `MLX_METALLIB_PATH` | `/Users/snider/Code/core/go-mlx/dist/lib/mlx.metallib` |
| Model snapshot | `/Users/snider/.cache/huggingface/hub/models--mlx-community--gemma-4-e2b-it-4bit/snapshots/99d9a53ff828d365a8ecae538e45f80a08d612cd` |

Built binaries:

| Binary | SHA-256 |
| --- | --- |
| `bin/lthn-mlx` | `736787e9a4fb4f9d470791f9df117f44516ed9b85aa142a387aab839a960d9f9` |
| `bin/violet` | `87e6a6df9ce62d2d04ede001fd9d13d0313be27216f4cc7bb576a41c741318d4` |

## Discovery Command

```bash
env MLX_METALLIB_PATH=/Users/snider/Code/core/go-mlx/dist/lib/mlx.metallib /Users/snider/Code/core/go-mlx/bin/lthn-mlx discover -json -probe-device
```

JSON output was saved to `docs/runtime/2026-05-16-metal-discovery.json`.
The discovery report now carries explicit load readiness:

```text
available: true
runtime.labels.load_available: true
model.load: supported
runtime.autotune: supported
benchmark: supported
```

The earlier no-device result was caused by running without the metallib
override in this process. With `MLX_METALLIB_PATH` set, the runtime reports
native load and generation support.

The Gemma 4 E2B metadata discovery command was also captured:

```bash
env MLX_METALLIB_PATH=/Users/snider/Code/core/go-mlx/dist/lib/mlx.metallib /Users/snider/Code/core/go-mlx/bin/lthn-mlx discover -json -probe-device -include-models -include-candidates -max-models 1 -model-dir /Users/snider/.cache/huggingface/hub/models--mlx-community--gemma-4-e2b-it-4bit/snapshots/99d9a53ff828d365a8ecae538e45f80a08d612cd
```

JSON output was saved to
`docs/runtime/2026-05-16-metal-discovery-gemma4.json`. It includes the model
pack metadata, supported cache modes, standard workloads, and first-pass tuning
candidates while labelling native model load, autotune, benchmark, and
generation as available in this process.

## go-mlx Command

```bash
env MLX_METALLIB_PATH=/Users/snider/Code/core/go-mlx/dist/lib/mlx.metallib /Users/snider/Code/core/go-mlx/bin/lthn-mlx driver-profile -json -include-output=false -context 4096 -prompt "Answer in one short sentence: why does retained model state matter?" -max-tokens 128 -runs 3 -trace-token-phases /Users/snider/.cache/huggingface/hub/models--mlx-community--gemma-4-e2b-it-4bit/snapshots/99d9a53ff828d365a8ecae538e45f80a08d612cd
```

JSON output was saved to
`docs/runtime/2026-05-16-gemma4-e2b-driver-profile.json`.

## Result

The native profile loaded and generated successfully:

```text
successful_runs: 3
generated_tokens: 48
visible_tokens: 48
decode_tokens_per_sec_average: 44.55943393415422
first_token_avg_duration: 92.270319ms
peak_memory_bytes: 8579334138
```

This is below the 100 tok/s floor, so the optimisation lane remains open.
`-trace-token-phases` captured the recurrent one-token decode bucket:

```text
steady token phase samples: 45
sample_eval_duration average: 20.979348955555555ms
sample_eval_duration min/max: 20.679375ms / 21.83775ms
forward_duration typical range: ~1.18ms to ~1.43ms
```

In this generator, `Eval(next)` materialises the lazy forward pass that produced
the current token logits. The largest repeated bucket is therefore the native
one-token forward materialisation plus sampling evaluation boundary, not the
small Go-side token read, text decode, or orchestration fields.

## Runner Parity Check

The system `mlx_lm.generate` comparison runner was not usable:

```text
ModuleNotFoundError: No module named 'mlx.utils'
```

The installed system Python package metadata reports `mlx==0.30.6` and
`mlx-lm==0.31.2`, but importing `mlx_lm` fails before a model can load.

A temporary parity runner environment was created without mutating the Homebrew
Python install:

```bash
python3 -m venv /private/tmp/go-mlx-mlx-lm-venv
/private/tmp/go-mlx-mlx-lm-venv/bin/python -m pip install --upgrade pip mlx mlx-lm
```

That environment installed `mlx==0.31.2` and `mlx-lm==0.31.3`, which clears the
old `mlx.utils` package mismatch. Inside the sandbox, the repaired runner still
cannot reach even `--help`, with or without the same `MLX_METALLIB_PATH`
override:

```bash
env MLX_METALLIB_PATH=/Users/snider/Code/core/go-mlx/dist/lib/mlx.metallib /private/tmp/go-mlx-mlx-lm-venv/bin/python -m mlx_lm.generate --help
```

```text
RuntimeError: [metal::load_device] No Metal device available. This typically occurs in headless, sandboxed, or virtualized macOS sessions where the GPU is not accessible.
```

Outside the sandbox, the same repaired runner can import and show help, but it
still cannot generate from the exact Gemma 4 E2B snapshot:

```bash
env MLX_METALLIB_PATH=/Users/snider/Code/core/go-mlx/dist/lib/mlx.metallib /private/tmp/go-mlx-mlx-lm-venv/bin/python -m mlx_lm.generate --model /Users/snider/.cache/huggingface/hub/models--mlx-community--gemma-4-e2b-it-4bit/snapshots/99d9a53ff828d365a8ecae538e45f80a08d612cd --prompt "Answer in one short sentence: why does retained model state matter?" --max-tokens 128 --temp 0 --verbose True
```

That run reaches `mlx_lm.utils.load_model` and then fails strict weight loading:

```text
ValueError: Received 140 parameters not in model
```

Full stderr is saved as
`docs/runtime/2026-05-16-mlx-lm-gemma4-e2b-parity-attempt.txt`. This is not a
parity pass and produces no reference tok/s. A valid comparison still needs an
MLX runner version or shared model snapshot that both runtimes can load with
the same prompt, context, sampling, and token budget.

## Native Greedy Decode-Tail Attempt

After the baseline profile above, the deterministic single-step greedy decode
tail was moved behind a native C++ wrapper in `go/internal/metal`:

- `decode_bridge.cpp` owns a static MLX compiled closure for last-token argmax.
- `decode.go` only enables it for unprobed greedy generation once logits are
  already single-step, so variable-shape prefill logits and non-greedy sampling
  stay on the existing path.
- `ModelSession.Generate` uses the same wrapper and keeps next-token logits
  lazy between retained-state decode steps.
- Go still owns model loading, lifecycle, compatibility checks, metrics, and
  reporting; the full one-token layer/materialisation boundary remains open.

The bundle was rebuilt after that boundary change:

| Binary | SHA-256 |
| --- | --- |
| `bin/lthn-mlx` | `878797bbecec3f9e7f2c1614233220d15f94aa180c7118567fd1f660b9daf8bb` |
| `bin/violet` | `cee610ae6228d17a0cd7cfd7c220fb9fa460111d9a57949087dda87c74ba7788` |

The exact Gemma 4 E2B profile command was rerun with the same
`MLX_METALLIB_PATH`, prompt, context, token budget, runs, and token phase trace
flags. The first sandboxed attempt failed before model load:

```text
metal.LoadAndInit: select device: mlx: no usable Metal device available; refusing native MLX load because CPU fallback can abort this MLX build
```

The same command completed outside the sandbox, where the Metal device was
visible. JSON output is saved as
`docs/runtime/2026-05-16-gemma4-e2b-native-greedy-rerun.json`.

```text
successful_runs: 3
generated_tokens: 48
visible_tokens: 48
decode_tokens_per_sec_average: 44.93695802859693
first_token_avg_duration: 92.981527ms
peak_memory_bytes: 8579365770
```

This is a small improvement over the baseline
`44.55943393415422` decode tok/s: `+0.3775240944427125 tok/s`, or roughly
`+0.847%`. The steady token phase bucket remains dominated by native
materialisation:

```text
steady token phase samples: 45
sample_eval_duration average: 20.77524171111111ms
sample_eval_duration min/max: 20.488208ms / 24.405208ms
forward_duration average: 1.3604814444444445ms
```

The result confirms that the compiled greedy decode tail is measurable but too
small to close the 100 tok/s lane. The full one-token layer/materialisation
boundary remains the next target.

## Next Boundary

The next native optimisation boundary is the full one-token layer block:
attention, MLP, residual, norm, lazy materialisation, and sampling evaluation.
Activation-only patches are not expected to close the gap because the traced
steady-state bucket is approximately 21ms/token while the named Go
orchestration phases are in microseconds and the recorded lazy `forward` setup
is roughly 1.2-1.4ms/token.

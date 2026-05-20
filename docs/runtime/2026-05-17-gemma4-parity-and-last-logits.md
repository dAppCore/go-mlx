<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# Gemma 4 Parity and Last-Logits Profile, 2026-05-17

This report records the follow-up evidence for `GOAL.md` after the native
last-token output projection wrapper landed behind
`GO_MLX_ENABLE_LAST_LOGITS_PREFILL=1`.

New external benchmark evidence in this report is llama.cpp-only. The
`mlx_lm.generate` entries below are archived historical context and should not
be rerun for the active parity lane.

## Environment

| Item | Value |
| --- | --- |
| Host | Apple M3 Ultra |
| go-mlx binary | `bin/lthn-mlx` |
| go-mlx SHA-256 after last-logits run | `5c8aeea06fece0b49683e1683e2204447266f1fedbe7f2a642622af6deccd979` |
| go-mlx SHA-256 for native-MLP benchmark | `85443fb248abe47afb546ee720e661b8f7dbae292981d0b98b00263799b1380b` |
| final verified go-mlx SHA-256 before layer probes | `9d9c8dc69f734c4ec45db952abae07b06cb8efb4bb3eedb1f9bbc303d8491341` |
| final verified go-mlx SHA-256 after default-path restore | `0c4c9ec67aa16964b270fd349f3ce1bfea18680857f80d52f86b6c0e51d78f03` |
| go-mlx SHA-256 for disabled per-layer-input diagnostic | `c097cb7612b7c402880fb0ba7a1bad7baad1494df43dceec059feeef9e99942d` |
| go-mlx SHA-256 for quantized embedding row-gather fix | `c40c7566f3b746a8072ae7c8f83f3c50ac05a46ac8b08d658d92752ea37b0536` |
| final go-mlx SHA-256 after direct-GQA and template alignment | `5aed4d4ede92e9e5e16958d018a984ac1d80fbebdb34cf1a0a8d406b276cc64d` |
| final current go-mlx SHA-256 after native GELU gate probe | `3d720db7a77235104b48707d50e27170c6e8e7b97dd022cba32acaaa6f4673e9` |
| go-mlx SHA-256 after SDPA512 rebuild | `1ba7ea769df0b48f39ec6f0581fa4b8bf0931b1a8944e7ad2e7ea911d43b6f49` |
| go-mlx SHA-256 after shared-mask gate | `fb0525b7fb411c978c6cc001af03d48517b04b9f8377613329b74ed8578b0e18` |
| go-mlx SHA-256 after decode-only fused expert gate/up | `085e204e17aa0f4f1fe614efa090f8779832129de5c377bf8b570902b3172f7b` |
| go-mlx SHA-256 after auto long-prompt last-token prefill | `dd212338c1864b6acb630bb5f534986432d1c189d17e100ae8ab3a3ee230a352` |
| go-mlx SHA-256 after FFN split trace instrumentation | `92a8ad92aa9fab6090aeb904540bba32c0afe37d5a037624b9109db8263fbc73` |
| go-mlx SHA-256 after expert-ID matvec scaffold | `f919eb75ab334887366acfc8e432b99c9d2fc7323d4dd0fe43ffb4fbfbf3d4cd` |
| go-mlx SHA-256 after expert-ID CLI gate diagnostic | `c094b241103db1099ebbf21a8950d599eb76cae487b43b840365dbda58fa0e9f` |
| go-mlx SHA-256 after expert-ID fused activation diagnostic | `374cdd7f4455b3dff5379281372ec6eb092146ec6f7a5acc4446aaf4d5afb958` |
| go-mlx SHA-256 after sorted prefill and paged fast-concat decode | `1eea3598b6265d5bf8326e00873ad6fd13877f471b778f739fed9213a3d3c286` |
| go-mlx SHA-256 after Gemma 4 decode runtime-gate CLI flags | `7fa565aa81715db5451771a1ecfa8e3aed730a1b7318aa237a9c27e8f9b7ffd5` |
| go-mlx SHA-256 after direct-greedy runtime-gate CLI flag | `088b423e65b088e5ff8d2e8d30e4e1edb8180f1888b68a568f32229a9dbc6631` |
| go-mlx SHA-256 after compiled Gemma 4 MoE graph support | `f45340c4c6d3f92a1f817a1096929652e1f08b86dd403a02078329f8772d2670` |
| go-mlx SHA-256 after native-layer MoE gate correction | `5686978954adac5941e48ae305ff875f33a507d81c7e07a8f8f6380e3812d09c` |
| `/private/tmp/lthn-mlx-split-expert-id` SHA-256 after split/BF16 expert-ID shared-input path | `dd9dfe917d073c4006b74e7ae7a42fbdefe96f3f74533607e46e5d7785923b1f` |
| llama.cpp Q4_K_M same-prompt-length artefact | `docs/runtime/2026-05-17-llamacpp-gemma4-26b-a4b-q4-k-m-p2204-g128-bench.json` |
| patched `libmlx.dylib` SHA-256 | `b9769e488037e3a4bdc3fdbded69068ae8b3d58a0d007cea7693223a76141790` |
| patched `mlx.metallib` SHA-256 | `627afba8939b38f13878eebdcaacc6d063225c2351516abdf6954b1f8ca557ce` |
| Archived Python runner env | `/private/tmp/go-mlx-mlx-lm-venv` |
| Archived Python runner `mlx` | `0.31.2` |
| Archived Python runner `mlx-lm` | `0.31.3` |
| `MLX_METALLIB_PATH` | `/Users/snider/Code/core/go-mlx/dist/lib/mlx.metallib` |
| `llama.cpp` reference clone | `/private/tmp/llama.cpp`, commit `1a68ec9` |

## Target E2B Last-Logits Rerun

The exact target command was rerun with the gated last-token output path:

```bash
env GO_MLX_ENABLE_LAST_LOGITS_PREFILL=1 MLX_METALLIB_PATH=/Users/snider/Code/core/go-mlx/dist/lib/mlx.metallib /Users/snider/Code/core/go-mlx/bin/lthn-mlx driver-profile -json -include-output=false -context 4096 -prompt "Answer in one short sentence: why does retained model state matter?" -max-tokens 128 -runs 3 -trace-token-phases /Users/snider/.cache/huggingface/hub/models--mlx-community--gemma-4-e2b-it-4bit/snapshots/99d9a53ff828d365a8ecae538e45f80a08d612cd
```

JSON output is saved as
`docs/runtime/2026-05-17-gemma4-e2b-last-logits-prefill-rerun.json`.

Result:

```text
successful_runs: 3
generated_tokens: 48
visible_tokens: 48
decode_tokens_per_sec_average: 44.874611039475575
first_token_avg_duration: 134.800944ms
peak_memory_bytes: 8579365766
steady sample_eval_duration average: 20.882495ms/token
steady forward_duration average: 1.322953ms/token
```

This is slightly below the previous native-greedy run
(`44.93695802859693 tok/s`, `-0.06234698912135883 tok/s`, `-0.1387%`).
The last-token output projection wrapper is therefore not the 100 tok/s
boundary. The recurrent materialisation bucket remains roughly 21 ms/token.

## Target E2B Native MLP Rerun

The dense GELU MLP sub-block was moved behind a native compiled wrapper for the
common no-bias path, including the q4/group-64 projection shape used by the
target E2B lane. Because the first measurement regressed, the path is gated by
`GO_MLX_ENABLE_NATIVE_MLP_GELU=1` and the default runtime leaves it disabled.

Gated command:

```bash
env GO_MLX_ENABLE_NATIVE_MLP_GELU=1 MLX_METALLIB_PATH=/Users/snider/Code/core/go-mlx/dist/lib/mlx.metallib /Users/snider/Code/core/go-mlx/bin/lthn-mlx driver-profile -json -include-output=false -context 4096 -prompt "Answer in one short sentence: why does retained model state matter?" -max-tokens 128 -runs 3 -trace-token-phases /Users/snider/.cache/huggingface/hub/models--mlx-community--gemma-4-e2b-it-4bit/snapshots/99d9a53ff828d365a8ecae538e45f80a08d612cd
```

JSON output is saved as
`docs/runtime/2026-05-17-gemma4-e2b-native-mlp-rerun.json`.

```text
successful_runs: 3
generated_tokens: 48
visible_tokens: 48
decode_tokens_per_sec_average: 43.10698466210642
steady sample_eval_duration average: 21.633695ms/token
peak_memory_bytes: 8579365786
```

This is slower than the prior native-greedy rerun by
`-1.82997336649051 tok/s`, so the native MLP wrapper is retained only as an
experimental boundary probe.

Default command, with the native MLP gate off:

```bash
env MLX_METALLIB_PATH=/Users/snider/Code/core/go-mlx/dist/lib/mlx.metallib /Users/snider/Code/core/go-mlx/bin/lthn-mlx driver-profile -json -include-output=false -context 4096 -prompt "Answer in one short sentence: why does retained model state matter?" -max-tokens 128 -runs 3 -trace-token-phases /Users/snider/.cache/huggingface/hub/models--mlx-community--gemma-4-e2b-it-4bit/snapshots/99d9a53ff828d365a8ecae538e45f80a08d612cd
```

JSON output is saved as
`docs/runtime/2026-05-17-gemma4-e2b-native-mlp-gated-default-rerun.json`.

```text
successful_runs: 3
generated_tokens: 48
visible_tokens: 48
decode_tokens_per_sec_average: 44.89465488606482
steady sample_eval_duration average: 20.805728ms/token
peak_memory_bytes: 8579365770
```

The default lane remains below the 100 tok/s floor and effectively unchanged
from the previous native-greedy profile.

## Target E2B Paged KV Rerun

`driver-profile` now accepts `-cache-mode` so the same target workload can
force the native KV cache storage mode without creating a separate tuning
profile. The confirmation run was sequential and used the paged KV path:

```bash
env MLX_METALLIB_PATH=/Users/snider/Code/core/go-mlx/dist/lib/mlx.metallib /Users/snider/Code/core/go-mlx/bin/lthn-mlx driver-profile -json -include-output=false -cache-mode paged -context 4096 -prompt "Answer in one short sentence: why does retained model state matter?" -max-tokens 128 -runs 3 -trace-token-phases /Users/snider/.cache/huggingface/hub/models--mlx-community--gemma-4-e2b-it-4bit/snapshots/99d9a53ff828d365a8ecae538e45f80a08d612cd
```

JSON output is saved as
`docs/runtime/2026-05-17-gemma4-e2b-cache-paged-confirm-rerun.json`.

```text
successful_runs: 3
generated_tokens: 60
visible_tokens: 60
load.cache_mode: paged
decode_tokens_per_sec_average: 46.94074033007464
steady sample_eval_duration average: 20.309252947ms/token
peak_memory_bytes: 8579365290
```

This is a positive cache-boundary result compared with the default gate-off
native MLP rerun (`44.89465488606482 tok/s`, `+2.04608544400982 tok/s`,
`+4.5575%`), but it still leaves the target path far below the 100 tok/s
floor. A later explicit fp16 cache rerun averaged
`45.065057937704864 tok/s`, below the resolved paged path. Earlier q8 and
asymmetric-cache JSON files from this date were launched concurrently with
another GPU run and are not acceptance evidence.

## Target E2B Resolved-Load Rerun

The next issue was that the default `driver-profile` report only showed
flag-provided load settings. The root loader also used the conservative unknown
machine-class plan unless callers opted into the full MLX device probe with
`GO_MLX_REPORT_DEVICE_INFO=1`, which made the target command resolve to q8 KV
on this machine. The loader now uses host-reported Apple memory for planning
without initialising MLX device probing, and the report records the effective
resolved load settings.

The unmodified target command was rerun after that fix, without `-cache-mode`:

```bash
env MLX_METALLIB_PATH=/Users/snider/Code/core/go-mlx/dist/lib/mlx.metallib /Users/snider/Code/core/go-mlx/bin/lthn-mlx driver-profile -json -include-output=false -context 4096 -prompt "Answer in one short sentence: why does retained model state matter?" -max-tokens 128 -runs 3 -trace-token-phases /Users/snider/.cache/huggingface/hub/models--mlx-community--gemma-4-e2b-it-4bit/snapshots/99d9a53ff828d365a8ecae538e45f80a08d612cd
```

JSON output is saved as
`docs/runtime/2026-05-17-gemma4-e2b-resolved-load-rerun.json`.

```text
load.cache_policy: rotating
load.cache_mode: paged
load.batch_size: 2
load.prefill_chunk_size: 2048
successful_runs: 3
generated_tokens: 60
visible_tokens: 60
decode_tokens_per_sec_average: 46.50145764359926
steady sample_eval_duration average: 20.443046053ms/token
peak_memory_bytes: 8579365290
```

This makes the measured paged-KV path the default target-command path on the
M3 Ultra-class machine. It is still not a completion result: the decode floor is
less than half of the 100 tok/s requirement.

## Target E2B Native Phase Trace

The native phase trace is diagnostic only. It is enabled with
`GO_MLX_TRACE_FORWARD_EVAL=1` and only records events when
`-trace-token-phases` arms token-level tracing. Under that gate Gemma 4 forces
and detaches four materialisation boundaries in each layer: attention,
attention residual, FFN, and layer output. This intentionally changes timing so
the result should not be compared as a throughput optimisation.

Command:

```bash
env GO_MLX_TRACE_FORWARD_EVAL=1 MLX_METALLIB_PATH=/Users/snider/Code/core/go-mlx/dist/lib/mlx.metallib /Users/snider/Code/core/go-mlx/bin/lthn-mlx driver-profile -json -include-output=false -context 4096 -prompt "Answer in one short sentence: why does retained model state matter?" -max-tokens 64 -runs 1 -trace-token-phases /Users/snider/.cache/huggingface/hub/models--mlx-community--gemma-4-e2b-it-4bit/snapshots/99d9a53ff828d365a8ecae538e45f80a08d612cd
```

JSON output is saved as
`docs/runtime/2026-05-17-gemma4-e2b-native-phase-trace.json`.

```text
successful_runs: 1
generated_tokens: 20
visible_tokens: 20
decode_tokens_per_sec_average: 18.09851769746586
token_phase_count: 21
native_event_count: 2800
steady events per token: 140
steady forward_duration average: 55.365661765ms/token
steady native_events total p50: 47.615249ms/token
steady sample_eval_duration average: 0.718654353ms/token
```

Boundary summary, excluding the first two decode steps and the final token:

```text
attention p50: 0.264542ms, p90: 0.558083ms
ffn p50: 0.260667ms, p90: 0.480500ms
output p50: 0.222458ms, p90: 0.495917ms
attention_residual p50: 0.168208ms, p90: 0.351042ms
gemma4.layer.00.output p50: 11.818917ms
gemma4.layer.00.attention p50: 2.211834ms
```

The trace does not identify another small wrapper like MLP, argmax, output
projection, or cache storage as sufficient. It points back to the full
one-token layer/materialisation boundary, with the first layer/output
materialisation standing out as the largest repeated cumulative boundary.

## Archived Exact E2B Python Runner Attempts

Archived attempts showed that the exact Gemma 4 E2B q4 target was unsupported
by the repaired `mlx_lm.generate` runner:

```bash
env MLX_METALLIB_PATH=/Users/snider/Code/core/go-mlx/dist/lib/mlx.metallib /private/tmp/go-mlx-mlx-lm-venv/bin/python -m mlx_lm.generate --model /Users/snider/.cache/huggingface/hub/models--mlx-community--gemma-4-e2b-it-4bit/snapshots/99d9a53ff828d365a8ecae538e45f80a08d612cd --prompt "Answer in one short sentence: why does retained model state matter?" --max-tokens 128 --temp 0 --verbose True
```

The failure is saved in
`docs/runtime/2026-05-16-mlx-lm-gemma4-e2b-parity-attempt.txt`:

```text
ValueError: Received 140 parameters not in model
```

The nearest E2B BF16 text snapshot fails in the same shared-KV area:

```bash
env MLX_METALLIB_PATH=/Users/snider/Code/core/go-mlx/dist/lib/mlx.metallib /private/tmp/go-mlx-mlx-lm-venv/bin/python -m mlx_lm.generate --model /Users/snider/.cache/huggingface/hub/models--mlx-community--gemma-4-E2B-it-bf16/snapshots/37cb2cef400fc8381f2b7d0e08482a6def6aaaaf --prompt "Answer in one short sentence: why does retained model state matter?" --max-tokens 128 --temp 0 --verbose True
```

Full output is saved as
`docs/runtime/2026-05-17-mlx-lm-gemma4-e2b-bf16-parity.txt`:

```text
ValueError: Received 60 parameters not in model
```

The assistant E2B BF16 snapshot was also not a comparison target for this
archived runner:

```bash
env MLX_METALLIB_PATH=/Users/snider/Code/core/go-mlx/dist/lib/mlx.metallib /private/tmp/go-mlx-mlx-lm-venv/bin/python -m mlx_lm.generate --model /Users/snider/.cache/huggingface/hub/models--mlx-community--gemma-4-E2B-it-assistant-bf16/snapshots/a7770799b560135ebdbfae8b7f468947415003bc --prompt "Answer in one short sentence: why does retained model state matter?" --max-tokens 128 --temp 0 --verbose True
```

Full output is saved as
`docs/runtime/2026-05-17-mlx-lm-gemma4-e2b-assistant-bf16-parity.txt`:

```text
ValueError: Model type gemma4_assistant not supported.
```

## Archived Shared Gemma 4 31B q4 Python Runner Evidence

The closest cached shared Gemma 4 q4 snapshot without the E2B shared-KV
loading blocker is:

```text
/Users/snider/.cache/huggingface/hub/models--mlx-community--gemma-4-31b-it-4bit/snapshots/dcb78c3f5d6becacbfce71cd4851ad98c4f08a05
```

Its config reports `model_type=gemma4`, `text_config.model_type=gemma4_text`,
`num_hidden_layers=60`, `num_kv_shared_layers=0`, `num_key_value_heads=16`,
and 4-bit affine quantisation.

### Archived `mlx_lm.generate`

```bash
env MLX_METALLIB_PATH=/Users/snider/Code/core/go-mlx/dist/lib/mlx.metallib /private/tmp/go-mlx-mlx-lm-venv/bin/python -m mlx_lm.generate --model /Users/snider/.cache/huggingface/hub/models--mlx-community--gemma-4-31b-it-4bit/snapshots/dcb78c3f5d6becacbfce71cd4851ad98c4f08a05 --prompt "Answer in one short sentence: why does retained model state matter?" --max-tokens 128 --temp 0 --verbose True
```

Output is saved as
`docs/runtime/2026-05-17-mlx-lm-gemma4-31b-q4-parity.txt`.

```text
Prompt: 29 tokens, 43.832 tokens-per-sec
Generation: 128 tokens, 34.702 tokens-per-sec
Peak memory: 17.560 GB
```

### go-mlx

```bash
env MLX_METALLIB_PATH=/Users/snider/Code/core/go-mlx/dist/lib/mlx.metallib /Users/snider/Code/core/go-mlx/bin/lthn-mlx driver-profile -json -include-output=false -context 4096 -prompt "Answer in one short sentence: why does retained model state matter?" -max-tokens 128 -runs 1 /Users/snider/.cache/huggingface/hub/models--mlx-community--gemma-4-31b-it-4bit/snapshots/dcb78c3f5d6becacbfce71cd4851ad98c4f08a05
```

JSON output is saved as
`docs/runtime/2026-05-17-go-mlx-gemma4-31b-q4-parity.json`.

```text
successful_runs: 1
generated_tokens: 20
visible_tokens: 18
decode_tokens_per_sec_average: 18.534762178149645
peak_memory_bytes: 21635473840
```

After the quantized embedding row-gather fix, the same go-mlx command was
rerun:

```text
successful_runs: 1
generated_tokens: 26
visible_tokens: 24
decode_tokens_per_sec_average: 21.086800870117965
prefill_tokens_per_sec_average: 111.28818410149346
peak_memory_bytes: 19078040792
```

JSON output is saved as
`docs/runtime/2026-05-17-go-mlx-gemma4-31b-q4-row-gather-parity.json`.

This archived Python-runner result is no longer an active parity target. It
remains useful as historical context for the shared Gemma 4 31B q4 snapshot:
the row-gather fix improved go-mlx and reduced peak memory, but the current
active external comparison moved to llama.cpp.

After matching the model's no-thinking chat-template cue and letting MLX fast
SDPA consume grouped-query K/V heads directly, the current default go-mlx binary
reports:

```text
go-mlx SHA-256: 5aed4d4ede92e9e5e16958d018a984ac1d80fbebdb34cf1a0a8d406b276cc64d
prompt_tokens: 26
successful_runs: 1
generated_tokens: 22
visible_tokens: 22
decode_tokens_per_sec_average: 25.50627418114353
prefill_tokens_per_sec_average: 146.52537585350962
peak_memory_bytes: 19062558400
active_memory_bytes: 18501830376
```

JSON output is saved as
`docs/runtime/2026-05-17-go-mlx-gemma4-31b-q4-final-direct-gqa-template-parity.json`.
The traced rerun is saved as
`docs/runtime/2026-05-17-go-mlx-gemma4-31b-q4-final-direct-gqa-template-trace.json`;
excluding the first two decode steps and the final stop token, it reports 20
steady samples with average `sample_eval_duration` `38.10032295ms/token`,
average `forward_duration` `1.6913334ms/token`, and average total
`39.8736084ms/token`.

For the same no-thinking chat-template lane, the archived `mlx_lm.generate`
runner was rerun with:

```bash
env MLX_METALLIB_PATH=/Users/snider/Code/core/go-mlx/dist/lib/mlx.metallib /private/tmp/go-mlx-mlx-lm-venv/bin/python -m mlx_lm.generate --model /Users/snider/.cache/huggingface/hub/models--mlx-community--gemma-4-31b-it-4bit/snapshots/dcb78c3f5d6becacbfce71cd4851ad98c4f08a05 --prompt "Answer in one short sentence: why does retained model state matter?" --max-tokens 128 --temp 0 --chat-template-config '{"enable_thinking": false}' --verbose True
```

Output is saved as
`docs/runtime/2026-05-17-mlx-lm-gemma4-31b-q4-no-thinking-parity.txt`.

```text
Prompt: 26 tokens, 76.733 tokens-per-sec
Generation: 23 tokens, 36.185 tokens-per-sec
Peak memory: 17.559 GB
```

The previous `mlx_lm.generate` result with 29 prompt tokens is the
thinking-enabled template lane (`enable_thinking=true`). These Python-runner
measurements remain useful as archived context only. They are no longer the
acceptance comparator for go-mlx throughput work.

The first go-mlx direct-GQA/template run above was a one-run result. The final
current default binary was rerun three times on the same no-thinking lane:

```text
go-mlx SHA-256: 3d720db7a77235104b48707d50e27170c6e8e7b97dd022cba32acaaa6f4673e9
prompt_tokens: 26
successful_runs: 3
generated_tokens: 66
visible_tokens: 66
decode_tokens_per_sec_average: 24.663669410625896
run tok/s: 24.662465213186447, 24.606634069565054, 24.721908949126185
prefill_tokens_per_sec_average: 153.73412997063005
peak_memory_bytes: 19076060876
active_memory_bytes: 18501830376
```

JSON output is saved as
`docs/runtime/2026-05-17-go-mlx-gemma4-31b-q4-final-current-default-3run-parity.json`.
The stderr file beside it is zero bytes. Against the archived no-thinking
Python-runner datapoint, this historical sample was roughly `1.47x` slower
(`36.185 / 24.663669...`), but that comparison is no longer an active
benchmark target.

Two follow-up probes did not close the 31B gap:

| Probe | Decode tok/s | Result |
| --- | ---: | --- |
| `GO_MLX_ENABLE_ASYNC_DECODE_PREFETCH=1`, current order | `24.41755011370027` | Negative; traced timing moved from `sample_eval_duration` into unaccounted work without raising throughput |
| `GO_MLX_ENABLE_NATIVE_GELU_GATE_MUL=1` | `25.260023959706817` untraced, `25.084752484961715` traced | Slight one-run uplift only; not a stable parity boundary and disabled by default |

The async-current-order JSON is saved as
`docs/runtime/2026-05-17-go-mlx-gemma4-31b-q4-async-prefetch-current-order-trace.json`.
The native GELU probe outputs are saved as
`docs/runtime/2026-05-17-go-mlx-gemma4-31b-q4-native-gelu-gate-parity.json` and
`docs/runtime/2026-05-17-go-mlx-gemma4-31b-q4-native-gelu-gate-trace.json`.

The 31B native phase trace is diagnostic because it forces materialisation at
layer boundaries. It reports `10.677002004607127 tok/s`, with 240 native events
per decode step (60 layers times 4 boundaries). Excluding warmup and the final
token, aggregate forced-boundary time is highest in the FFN family
(`250.267ms` total), then attention (`184.729ms`), layer output
(`90.987ms`), and attention residual (`88.420ms`). Isolated activation wrappers
therefore are not enough; the remaining gap is likely in the larger graph and
materialisation topology.

Raw-prompt reruns were also recorded to check template effects:

```bash
env MLX_METALLIB_PATH=/Users/snider/Code/core/go-mlx/dist/lib/mlx.metallib /private/tmp/go-mlx-mlx-lm-venv/bin/python -m mlx_lm.generate --model /Users/snider/.cache/huggingface/hub/models--mlx-community--gemma-4-31b-it-4bit/snapshots/dcb78c3f5d6becacbfce71cd4851ad98c4f08a05 --prompt "Answer in one short sentence: why does retained model state matter?" --max-tokens 128 --temp 0 --ignore-chat-template --verbose True
```

```text
Generation: 128 tokens, 34.881 tokens-per-sec
```

```bash
env MLX_METALLIB_PATH=/Users/snider/Code/core/go-mlx/dist/lib/mlx.metallib /Users/snider/Code/core/go-mlx/bin/lthn-mlx driver-profile -json -include-output=false -chat=false -context 4096 -prompt "Answer in one short sentence: why does retained model state matter?" -max-tokens 128 -runs 1 /Users/snider/.cache/huggingface/hub/models--mlx-community--gemma-4-31b-it-4bit/snapshots/dcb78c3f5d6becacbfce71cd4851ad98c4f08a05
```

```text
successful_runs: 1
generated_tokens: 0
decode_tokens_per_sec_average: 0
```

The raw-prompt path is therefore diagnostic only. It confirms that prompt
formatting materially changes stop behaviour and should not be used as a hidden
parity substitute for the default chat-template lane.

## Target E2B Native Layer Rerun

A conservative one-token Gemma 4 layer wrapper now exists behind:

```bash
GO_MLX_ENABLE_NATIVE_GEMMA4_LAYER=1
```

The wrapper is intentionally narrow: no MoE, no LoRA, single-token decode, no
cache trim, paged cache with at most one page, q4/dense linears, attention,
MLP, residuals, per-layer input injection, layer scalar, and native cache page
handoff. It is a boundary probe, not a default runtime path.

Gate-on command:

```bash
env GO_MLX_ENABLE_NATIVE_GEMMA4_LAYER=1 MLX_METALLIB_PATH=/Users/snider/Code/core/go-mlx/dist/lib/mlx.metallib /Users/snider/Code/core/go-mlx/bin/lthn-mlx driver-profile -json -include-output=false -context 4096 -prompt "Answer in one short sentence: why does retained model state matter?" -max-tokens 128 -runs 3 -trace-token-phases /Users/snider/.cache/huggingface/hub/models--mlx-community--gemma-4-e2b-it-4bit/snapshots/99d9a53ff828d365a8ecae538e45f80a08d612cd
```

JSON output is saved as
`docs/runtime/2026-05-17-gemma4-e2b-native-layer-rerun.json`.

```text
successful_runs: 3
generated_tokens: 60
visible_tokens: 60
decode_tokens_per_sec_average: 44.54197676930399
steady forward_duration average: 0.602300925925926ms/token
steady sample_eval_duration average: 21.77002551851852ms/token
```

Gate-off control on the same rebuilt binary:

```bash
env MLX_METALLIB_PATH=/Users/snider/Code/core/go-mlx/dist/lib/mlx.metallib /Users/snider/Code/core/go-mlx/bin/lthn-mlx driver-profile -json -include-output=false -context 4096 -prompt "Answer in one short sentence: why does retained model state matter?" -max-tokens 128 -runs 3 -trace-token-phases /Users/snider/.cache/huggingface/hub/models--mlx-community--gemma-4-e2b-it-4bit/snapshots/99d9a53ff828d365a8ecae538e45f80a08d612cd
```

JSON output is saved as
`docs/runtime/2026-05-17-gemma4-e2b-native-layer-gateoff-rerun.json`.

```text
bin/lthn-mlx SHA-256: bfefdf9510dfc399a7018eaa12447c763395afe1adae949a4135c8befc21e3ff
successful_runs: 3
generated_tokens: 60
visible_tokens: 60
decode_tokens_per_sec_average: 47.054122991613305
steady forward_duration average: 0.9899429074074074ms/token
steady sample_eval_duration average: 20.205370388888888ms/token
```

The native layer wrapper therefore reduces Go-side graph construction but
increases MLX eval time enough to regress throughput by
`-2.512146222309312 tok/s` against its gate-off control. It stays disabled by
default. The next positive boundary needs a compiled or lower-level whole
materialisation path rather than a non-compiled layer regrouping.

## Target E2B Compiled Layer Attempt

A follow-up experiment added dynamic RoPE offset support and a separate
fail-closed MLX-compiled layer gate:

```bash
GO_MLX_ENABLE_COMPILED_GEMMA4_LAYER=1
```

The focused tiny-layer tests pass, but the real E2B cache path is not reusable
through MLX compile because the K/V cache length changes each token.

```bash
env GO_MLX_ENABLE_COMPILED_GEMMA4_LAYER=1 MLX_METALLIB_PATH=/Users/snider/Code/core/go-mlx/dist/lib/mlx.metallib /Users/snider/Code/core/go-mlx/bin/lthn-mlx driver-profile -json -include-output=false -context 4096 -prompt "Answer in one short sentence: why does retained model state matter?" -max-tokens 128 -runs 1 -trace-token-phases /Users/snider/.cache/huggingface/hub/models--mlx-community--gemma-4-e2b-it-4bit/snapshots/99d9a53ff828d365a8ecae538e45f80a08d612cd
```

JSON output is saved as
`docs/runtime/2026-05-17-gemma4-e2b-compiled-layer-failclosed.json`, and stderr
is saved beside it as
`docs/runtime/2026-05-17-gemma4-e2b-compiled-layer-failclosed.stderr`.

```text
bin/lthn-mlx SHA-256: 1b71031e4d379217b13654b955d1db3171408886d101ebeb3a0f12cd55161185
successful_runs: 1
generated_tokens: 20
visible_tokens: 20
decode_tokens_per_sec_average: 44.437334470929095
steady forward_duration average: 1.022509111111111ms/token
steady sample_eval_duration average: 20.320287111111112ms/token
```

The repeated fallback error is:

```text
compiled closure failed: mlx.lastError: [broadcast_shapes] Shapes (1,1,1,24,256) and (1,1,8,23,256) cannot be broadcast.
```

Full-attention layers show the same failure with `head_dim=512`. The gate now
fails closed and falls back instead of panicking, but this route is not a
positive optimisation boundary. The next attempt needs a lower-level dynamic
cache/block-table materialisation path, not MLX compile over the current
growing-cache graph.

## Default-Path Restore After Native Activation Probe

The activation bridge added explicit native `GELUGateMul` and `SiLUGateMul`
primitives, but routing the default Gemma/Qwen helper through those wrappers
regressed the normal lane. The gate-off control temporarily fell to
`40.956652070193485 tok/s`; steady `forward_duration` rose from about
`0.99ms/token` to about `1.2ms/token` while `sample_eval_duration` stayed near
`20ms/token`. The default helper was restored to the original lazy graph shape:
compiled GELU or regular SiLU, then `Mul`.

Restored default command:

```bash
env -u GO_MLX_ENABLE_NATIVE_GEMMA4_LAYER -u GO_MLX_ENABLE_COMPILED_GEMMA4_LAYER MLX_METALLIB_PATH=/Users/snider/Code/core/go-mlx/dist/lib/mlx.metallib /Users/snider/Code/core/go-mlx/bin/lthn-mlx driver-profile -json -include-output=false -context 4096 -prompt "Answer in one short sentence: why does retained model state matter?" -max-tokens 128 -runs 3 -trace-token-phases /Users/snider/.cache/huggingface/hub/models--mlx-community--gemma-4-e2b-it-4bit/snapshots/99d9a53ff828d365a8ecae538e45f80a08d612cd
```

JSON output is saved as
`docs/runtime/2026-05-17-gemma4-e2b-compiled-layer-gateoff-rerun.json`.

```text
bin/lthn-mlx SHA-256: 0c4c9ec67aa16964b270fd349f3ce1bfea18680857f80d52f86b6c0e51d78f03
successful_runs: 3
generated_tokens: 60
visible_tokens: 60
decode_tokens_per_sec_average: 46.37096822259417
steady step-10 sample_eval_duration: ~20.2ms/token
steady step-10 forward_duration: ~1.15-1.25ms/token
```

The restoration keeps the native activation wrappers as directly tested
experiments but removes them from default model execution. The lane remains
below target, but the accidental default regression is gone.

## `llama.cpp` Metal Read

`llama.cpp` was cloned to `/private/tmp/llama.cpp` and inspected at commit
`1a68ec9` to compare the current go-mlx path against a high-throughput Metal
runtime.

Useful reference points:

- This is the native design and benchmark reference for the next optimisation
  pass. `mlx_lm.generate` measurements in this report are archived context only,
  not active benchmark targets.
- The Gemma MoE path keeps the expert `gate_up` projection fused when the
  tensor exists, then splits the projected result into gate and up halves.
  That avoids two expert-indexed projections during decode.
- `src/llama-context.cpp` reuses the previous graph when graph parameters still
  determine the same topology. `process_ubatch` calls `res->can_reuse(gparams)`,
  skips graph rebuild/allocation on a hit, updates only graph inputs, and then
  calls the scheduler.
- `src/llama-graph.cpp` builds attention inputs as explicit host-fed tensors:
  token positions, K/V cache indices, and KQ masks are inputs rather than
  rebuilt model constants. The reuse check validates mask shape compatibility
  with the current KV span.
- `src/llama-kv-cache.cpp` keeps a ring-like KV cell plan. `prepare` finds
  slots for ubatches first, `apply_ubatch` mutates cache metadata, and
  `set_input_k_idxs` / `set_input_v_idxs` fill host input tensors for the graph.
  That is a better match for a dynamic block table than concatenating growing
  K/V arrays into the graph.
- `src/llama-graph.cpp` routes the attention hot path through
  `ggml_flash_attn_ext` when flash attention is enabled. The context validation
  rejects quantized V cache without flash attention, which is the inverse of
  the current go-mlx experiment that tries to compile over a growing cache.
- `ggml/src/ggml-metal/ggml-metal-context.m` submits graph compute
  asynchronously: the first command buffer is encoded immediately, additional
  command buffers are encoded on a concurrent dispatch queue, and completion is
  not waited on unless capture/error handling requires it.

The portable lesson for this repo is not to add another layer wrapper around
the current MLX arrays. The next serious attempt should introduce a stable
single-token decode topology with host-updated inputs for offset/cache indices
and an in-place or block-table KV read/write path, then measure a flash-attn
compatible cache layout. That maps to the `llama.cpp` design and avoids the
compiled-layer broadcast failure from baking the previous K/V length into the
closure.

## Fixed-Shape Decode Input Primitive

The first reusable-topology primitive now exists in `go/internal/metal`:

- `singleTokenCausalMask(capacity, offset)` builds a `[1,1,1,capacity]` mask
  from an offset array, keeping positions `<= offset` visible and future cache
  cells masked.
- `singleTokenCacheUpdate(cache, token, offset)` writes one K/V token into a
  fixed-capacity cache tensor using `PutAlongAxis` with a dynamic offset input.
- `fixedSingleTokenAttention(...)` combines those pieces: update K/V, build the
  offset mask, and run masked SDPA over fixed-size cache tensors.
- `go_mlx_compiled_fixed_single_token_attention` now exposes the same boundary
  through `go/internal/metal/decode_bridge.cpp`, which gives the host-fed offset
  and fixed-K/V update path a stable native C++ wrapper API. The gated
  fixed-cache compiled Gemma 4 layer now uses this wrapper for owner K/V
  updates. `Gemma4Attention.forward` also uses it when the gated fixed-cache
  owner path can keep full-capacity K/V tensors. Both paths fall back to the
  Go-authored graph if the native shape guard or wrapper fails.

Focused verification:

```bash
cd /Users/snider/Code/core/go-mlx/go
env GOCACHE=/private/tmp/codex-go-mlx-cache go test ./internal/metal -run 'TestGemma4_AttentionFixedCacheUsesNativeBridge_Good|TestDecode_(nativeFixedSingleTokenAttention|compiledGemma4DecodeLayer_FixedCacheGood)|TestFast_(fixedSingleTokenAttention_CompiledGood|singleTokenCacheUpdate_CompiledGood|singleTokenCausalMask_Good)' -count=1
```

Result:

```text
ok  	dappco.re/go/mlx/internal/metal	0.529s
```

This is positive evidence for the next boundary: MLX compile can reuse a
closure across changing decode offsets when K/V tensor shapes stay fixed and
the offset is an input. That directly addresses the compiled-layer failure
mode, where the closure saw growing K/V lengths such as `(...,24,head_dim)`
versus `(...,23,head_dim)`.

The bridge was then wired into the gated fixed-cache owner path and benchmarked
on the full 4096-slot target capacity:

```bash
env GO_MLX_ENABLE_FIXED_GEMMA4_CACHE=1 GO_MLX_ENABLE_COMPILED_GEMMA4_LAYER=1 MLX_METALLIB_PATH=/Users/snider/Code/core/go-mlx/dist/lib/mlx.metallib /Users/snider/Code/core/go-mlx/bin/lthn-mlx driver-profile -json -include-output=false -context 4096 -prompt "Answer in one short sentence: why does retained model state matter?" -max-tokens 128 -runs 3 -trace-token-phases /Users/snider/.cache/huggingface/hub/models--mlx-community--gemma-4-e2b-it-4bit/snapshots/99d9a53ff828d365a8ecae538e45f80a08d612cd
```

Result:

```text
binary sha256: be3983cfb67edcc7b784df38500a0350f6013a5f35692a38e7aa55ab8a1b7c6d
decode_tokens_per_sec_average: 107.77701729520602
runs: 95.07907894498449, 116.20241438731288, 112.0495585533207
generated_tokens: 384
visible_tokens: 384
prefill_tokens_per_sec_average: 844.1085014532886
peak_memory_bytes: 3327392930
stderr_bytes: 0
```

This is the first valid full-context fixed-cache result above the E2B
`100 tok/s` floor. It is still gated and does not settle default selection or
large-model throughput.

The same native bridge was then tested on the shared Gemma 4 31B q4 longdecode
lane. The unguarded bridge is not valid for that model yet: the first attempt
aborted after one generated token with the current bundled metallib unable to
load `sdpa_vector_float_512_512`, followed by
`kIOGPUCommandBufferCallbackErrorInvalidResource`. The partial failure artifact
is
`docs/runtime/2026-05-17-go-mlx-gemma4-31b-q4-fixed-cache160-native-bridge-longdecode.json`,
with stderr in the matching `.stderr` file.

The bridge now rejects 512-wide single-token heads so the 31B path falls back
instead of aborting. A bounded 160-slot cache covers this 29-token prompt plus
128 generated tokens:

```bash
env GO_MLX_ENABLE_FIXED_GEMMA4_CACHE=1 GO_MLX_FIXED_GEMMA4_CACHE_SIZE=160 GO_MLX_ENABLE_COMPILED_GEMMA4_LAYER=1 MLX_METALLIB_PATH=/Users/snider/Code/core/go-mlx/dist/lib/mlx.metallib /Users/snider/Code/core/go-mlx/bin/lthn-mlx driver-profile -json -include-output=false -context 4096 -prompt "Write exactly 200 comma-separated integers, starting at 1." -max-tokens 128 -runs 3 -trace-token-phases /Users/snider/.cache/huggingface/hub/models--mlx-community--gemma-4-31b-it-4bit/snapshots/dcb78c3f5d6becacbfce71cd4851ad98c4f08a05
```

Result:

```text
binary sha256: 0ff44477bb93be16754e6b3a4b71f238d77ab0cab27d6145369b1d460d3092fc
decode_tokens_per_sec_average: 24.94401176949734
runs: 25.24160351823528, 24.74238342491899, 24.848048365337757
generated_tokens: 384
visible_tokens: 384
prefill_tokens_per_sec_average: 168.39024382897423
peak_memory_bytes: 19331029517
stderr_bytes: 0
```

That is a small improvement over the current-default sustained 31B result
(`23.086428954337055 tok/s`), but 31B is now internal evidence rather than the
active external benchmark target. At this point the concrete 31B blocker was the
missing 512-wide native SDPA/vector-kernel path.

An opt-in native matmul-softmax fallback was then added for 512-wide fixed
single-token attention. It uses the same host-fed offset and fixed K/V update
shape, but avoids the missing MLX SDPA vector kernel. It is gated because it is
diagnostic, not a speed win:

```bash
env GO_MLX_ENABLE_FIXED_GEMMA4_CACHE=1 GO_MLX_FIXED_GEMMA4_CACHE_SIZE=160 GO_MLX_ENABLE_COMPILED_GEMMA4_LAYER=1 GO_MLX_ENABLE_FIXED_WIDE_MATMUL_ATTENTION=1 MLX_METALLIB_PATH=/Users/snider/Code/core/go-mlx/dist/lib/mlx.metallib /Users/snider/Code/core/go-mlx/bin/lthn-mlx driver-profile -json -include-output=false -context 4096 -prompt "Write exactly 200 comma-separated integers, starting at 1." -max-tokens 128 -runs 3 -trace-token-phases /Users/snider/.cache/huggingface/hub/models--mlx-community--gemma-4-31b-it-4bit/snapshots/dcb78c3f5d6becacbfce71cd4851ad98c4f08a05
```

Result:

```text
binary sha256: e5860c064f2a831db1a6a0afaab18c5cfc4d6b28b98c4a3131e0a35e0b29da5d
decode_tokens_per_sec_average: 24.333176943291804
runs: 24.52948796672134, 24.23060627819461, 24.239436584959467
generated_tokens: 384
visible_tokens: 384
prefill_tokens_per_sec_average: 165.63513923761562
peak_memory_bytes: 19331029342
stderr_bytes: 0
```

This confirms that simply replacing missing 512-wide SDPA with compiled
matmul/softmax does not close the 31B gap. The default 512-wide path remains
guarded so the fixed-cache experiment falls back instead of selecting the
slower diagnostic bridge.

The lower-level source check shows why the original fixed-cache bridge failed:
`mlx/backend/metal/kernels/scaled_dot_product_attention.metal` instantiates
vector SDPA for 64, 96, 128, and 256 head dimensions only. The local patch
`patches/mlx-sdpa-vector-512.patch` records the minimal MLX experiment to add
`512` vector and aggregation instantiations and to mark 512 as a supported
vector head dimension in `scaled_dot_product_attention.cpp`. The forward apply
check passed before applying it, and `git -C lib/mlx apply -R --check
../../patches/mlx-sdpa-vector-512.patch` now passes, confirming the patch is
applied to the pinned `lib/mlx` submodule for the local rebuild.

The rebuild needed the standalone Metal Toolchain component:

```bash
xcodebuild -downloadComponent MetalToolchain
xcodebuild -runFirstLaunch
```

`xcrun metal` still did not resolve the installed component, but direct tools
under
`/private/var/run/com.apple.security.cryptexd/mnt/com.apple.MobileAsset.MetalToolchain-v17.5.188.0.MM2SNE/Metal.xctoolchain/usr/bin/`
worked. A temporary wrapper at `/private/tmp/go-mlx-xcrun/xcrun` redirected
only `metal` and `metallib` to that path while delegating all other `xcrun`
calls back to `/usr/bin/xcrun`. The successful build disabled ccache and
installed the patched libraries into `dist/lib/`:

```bash
cmake -S . -B /private/tmp/go-mlx-build-sdpa512-noccache -DCMAKE_INSTALL_PREFIX=/Users/snider/Code/core/go-mlx/dist -DCMAKE_BUILD_TYPE=Release -DMLX_USE_CCACHE=OFF -DFETCHCONTENT_SOURCE_DIR_MLX-C=/Users/snider/Code/core/go-mlx/lib/mlx-c -DFETCHCONTENT_SOURCE_DIR_MLX=/Users/snider/Code/core/go-mlx/lib/mlx
env PATH=/private/tmp/go-mlx-xcrun:$PATH cmake --build /private/tmp/go-mlx-build-sdpa512-noccache --target install --parallel
```

The rebuilt metallib contains `sdpa_vector_float_512_512`,
`sdpa_vector_float16_t_512_512`, and `sdpa_vector_bfloat16_t_512_512`.

The patched 512-wide SDPA path was then benchmarked on the same shared-31B
longdecode lane:

```bash
env GO_MLX_ENABLE_FIXED_GEMMA4_CACHE=1 GO_MLX_FIXED_GEMMA4_CACHE_SIZE=160 GO_MLX_ENABLE_COMPILED_GEMMA4_LAYER=1 GO_MLX_ENABLE_FIXED_WIDE_SDPA_ATTENTION=1 MLX_METALLIB_PATH=/Users/snider/Code/core/go-mlx/dist/lib/mlx.metallib /Users/snider/Code/core/go-mlx/bin/lthn-mlx driver-profile -json -include-output=false -context 4096 -prompt "Write exactly 200 comma-separated integers, starting at 1." -max-tokens 128 -runs 3 -trace-token-phases /Users/snider/.cache/huggingface/hub/models--mlx-community--gemma-4-31b-it-4bit/snapshots/dcb78c3f5d6becacbfce71cd4851ad98c4f08a05
```

Result:

```text
binary sha256: 1ba7ea769df0b48f39ec6f0581fa4b8bf0931b1a8944e7ad2e7ea911d43b6f49
libmlx.dylib sha256: b9769e488037e3a4bdc3fdbded69068ae8b3d58a0d007cea7693223a76141790
mlx.metallib sha256: 627afba8939b38f13878eebdcaacc6d063225c2351516abdf6954b1f8ca557ce
successful_runs: 3
generated_tokens: 384
visible_tokens: 384
decode_tokens_per_sec_average: 24.70397262176645
runs: 24.54956052082555, 24.799885029282997, 24.762472315190802
prefill_tokens_per_sec_average: 138.49735481596804
peak_memory_bytes: 19331029334
stderr_bytes: 0
```

JSON output is saved as
`docs/runtime/2026-05-17-go-mlx-gemma4-31b-q4-fixed-cache160-native-sdpa512-longdecode.json`.
The missing-kernel failure is solved, but the speed result is still negative:
patched SDPA512 is slower than the guarded fallback
(`24.94401176949734 tok/s`). The next native target remains the llama.cpp-shaped
stable one-token graph boundary with host-fed cache slots, masks, and less eval
materialisation around the attention result.

The next llama.cpp-shaped micro-probe was to host-feed a single fixed-cache
mask once per token instead of building the same offset mask inside every layer
closure. This is gated behind:

```bash
GO_MLX_ENABLE_FIXED_GEMMA4_SHARED_MASK=1
```

The paired 31B longdecode runs are clean but neutral:

| Path | Decode tok/s | Runs | Prefill tok/s | Notes |
| --- | ---: | --- | ---: | --- |
| Shared host mask, fallback attention | `24.904493509253538` | `24.817692762578993`, `25.061646800329598`, `24.834140964852022` | `168.69260898305686` | No SDPA512 gate; stderr `0` |
| Shared host mask, patched SDPA512 | `24.767920780634018` | `24.885272574903453`, `24.72823353070345`, `24.69025623629516` | `166.11163115294733` | `GO_MLX_ENABLE_FIXED_WIDE_SDPA_ATTENTION=1`; stderr `0` |

JSON output is saved as
`docs/runtime/2026-05-17-go-mlx-gemma4-31b-q4-fixed-cache160-shared-mask-fallback-longdecode.json`
and
`docs/runtime/2026-05-17-go-mlx-gemma4-31b-q4-fixed-cache160-native-shared-mask-longdecode.json`.
The shared host-fed mask removes a duplicated graph component, but it does not
beat the previous guarded fallback. Mask construction is not the dominant 31B
cost.

## Experimental Fixed-Cache Gemma 4 Wiring

The fixed-shape primitive is now wired into Gemma 4 behind two explicit gates:

```bash
GO_MLX_ENABLE_FIXED_GEMMA4_CACHE=1
GO_MLX_ENABLE_COMPILED_GEMMA4_LAYER=1
```

`-cache-mode paged` remains the CLI/API shape. With the fixed-cache gate set,
Gemma 4 paged caches are swapped internally for `FixedKVCache` only when a
bounded context is known. `GO_MLX_FIXED_GEMMA4_CACHE_SIZE` optionally narrows
the fixed bucket below `-context`; this is diagnostic only and must be large
enough for the prompt plus generated tokens before it can be treated as a real
target-capacity result.

Post-change target reruns:

| Path | Decode tok/s | Notes |
| --- | ---: | --- |
| Default post-change control | `46.20225853209359` | No fixed-cache or compiled-layer gates |
| Fixed cache, full `4096` slots before native bridge | `39.88411733551154` | Stable topology lost when cache update and mask remained Go-authored MLX graph pieces |
| Fixed cache, full `4096` slots with native bridge | `107.77701729520602` | Stable topology plus native host-fed offset/KV update; valid 3-run target-capacity result |
| Fixed cache, `256` slots | `43.18471280763444` | Still below default |
| Fixed cache, `160` slots | `45.95924162792853` | Nearly default, covers this prompt plus 128 requested tokens |
| Fixed cache, `96` slots | `47.03732918131478` | Best fixed bucket for this prompt/EOS behaviour, but not a general 128-token capacity claim |
| Fixed cache, `64` slots | `46.870613364571796` | Slightly below the 96-slot result |

Representative command:

```bash
env GO_MLX_ENABLE_FIXED_GEMMA4_CACHE=1 GO_MLX_FIXED_GEMMA4_CACHE_SIZE=96 GO_MLX_ENABLE_COMPILED_GEMMA4_LAYER=1 MLX_METALLIB_PATH=/Users/snider/Code/core/go-mlx/dist/lib/mlx.metallib /Users/snider/Code/core/go-mlx/bin/lthn-mlx driver-profile -json -include-output=false -context 4096 -cache-mode paged -prompt "Answer in one short sentence: why does retained model state matter?" -max-tokens 128 -runs 3 -trace-token-phases /Users/snider/.cache/huggingface/hub/models--mlx-community--gemma-4-e2b-it-4bit/snapshots/99d9a53ff828d365a8ecae538e45f80a08d612cd
```

The native bridge changes the read: the fixed topology is now sufficient for
the E2B throughput floor when the cache update and host-fed offset/mask path
are inside the native wrapper. The remaining decisions are whether to promote a
fixed-cache bucket automatically, and whether the same llama.cpp-shaped boundary
can close the shared-31B gap.

## Direct Greedy Token Probe

Gemma 4 also has a final-output shortcut behind:

```bash
GO_MLX_ENABLE_DIRECT_GREEDY_TOKEN=1
```

The gate only applies to strict greedy decoding: no probe sink, temperature
zero, top-p/min-p/top-k disabled, and no active repeat penalty after history is
present. For that shape, final logit softcapping is monotonic, so the path can
skip materialising the softcapped logits tensor and return the argmax token
directly from final RMSNorm plus output projection.

Target rerun:

| Path | Decode tok/s | Notes |
| --- | ---: | --- |
| Default post-change control | `46.20225853209359` | Same rebuilt binary, gate off |
| Direct greedy token gate | `44.27055794965946` | 3 runs: `46.79984606501032`, `45.70047978214544`, `40.311348001822616` |

Representative command:

```bash
env GO_MLX_ENABLE_DIRECT_GREEDY_TOKEN=1 MLX_METALLIB_PATH=/Users/snider/Code/core/go-mlx/dist/lib/mlx.metallib /Users/snider/Code/core/go-mlx/bin/lthn-mlx driver-profile -json -include-output=false -context 4096 -prompt "Answer in one short sentence: why does retained model state matter?" -max-tokens 128 -runs 3 -trace-token-phases /Users/snider/.cache/huggingface/hub/models--mlx-community--gemma-4-e2b-it-4bit/snapshots/99d9a53ff828d365a8ecae538e45f80a08d612cd
```

The shortcut is correct as a gated experiment, but it is not the missing
performance boundary. The token trace still shows roughly `20ms/token` under
`sample_eval_duration`; the lazy one-token forward is just materialised through
`Eval(next)` instead of through sampled logits. This confirms the same lesson as
the fixed-cache probe: the next useful implementation has to reduce the native
one-token materialisation work itself, not only change the final logits/token
API shape.

## Async Decode Prefetch Probe

The `llama.cpp` Metal read also highlighted asynchronous command-buffer
submission. go-mlx now has an explicit diagnostic gate:

```bash
GO_MLX_ENABLE_ASYNC_DECODE_PREFETCH=1
```

When enabled, generation starts `EvalAsync` on the next lazy decode value after
constructing it, then the normal next-loop sampling read still synchronises the
value before token selection. This keeps semantics unchanged and tests the
specific overlap opportunity without making it a default runtime path.

Target rerun:

| Path | Decode tok/s | Notes |
| --- | ---: | --- |
| Default post-change control | `46.20225853209359` | Same default paged-cache band as the fixed-cache control |
| Async decode prefetch gate | `46.233006105790245` | 3 runs: `46.298560210152495`, `46.49208501310205`, `45.908373094116186` |

Representative command:

```bash
env GO_MLX_ENABLE_ASYNC_DECODE_PREFETCH=1 MLX_METALLIB_PATH=/Users/snider/Code/core/go-mlx/dist/lib/mlx.metallib /Users/snider/Code/core/go-mlx/bin/lthn-mlx driver-profile -json -include-output=false -context 4096 -prompt "Answer in one short sentence: why does retained model state matter?" -max-tokens 128 -runs 3 -trace-token-phases /Users/snider/.cache/huggingface/hub/models--mlx-community--gemma-4-e2b-it-4bit/snapshots/99d9a53ff828d365a8ecae538e45f80a08d612cd
```

This is clean but not meaningful acceleration. The generation loop has almost
no CPU-side work between queuing the next lazy value and synchronising for the
token read, so async submission lands inside normal run noise. The result keeps
the same conclusion: the next useful path is not another host scheduling tweak,
but a lower-level attention/cache materialisation boundary with stable inputs.

## Paged KV Preallocation Probe

One local cache mismatch left in go-mlx was not fp16 versus paged storage. It
was that `PagedKVCache` appended decode tokens to the last page via
`Concatenate`, so the final page shape and graph changed every token. The new
diagnostic gate keeps each page at fixed capacity and uses slice updates while
returning visible slices to attention and snapshot readers:

```bash
GO_MLX_ENABLE_PAGED_KV_PREALLOC=1
```

Same-binary reruns:

| Path | Decode tok/s | Notes |
| --- | ---: | --- |
| Gate off | `46.50781893730525` | 3 runs: `46.480078202731576`, `46.64872177417628`, `46.394656835007915` |
| Paged KV prealloc gate | `46.53706420697521` | 3 runs: `46.515688942973505`, `46.52283947852047`, `46.57266419943166` |

Representative command:

```bash
env GO_MLX_ENABLE_PAGED_KV_PREALLOC=1 MLX_METALLIB_PATH=/Users/snider/Code/core/go-mlx/dist/lib/mlx.metallib /Users/snider/Code/core/go-mlx/bin/lthn-mlx driver-profile -json -include-output=false -context 4096 -prompt "Answer in one short sentence: why does retained model state matter?" -max-tokens 128 -runs 3 -trace-token-phases /Users/snider/.cache/huggingface/hub/models--mlx-community--gemma-4-e2b-it-4bit/snapshots/99d9a53ff828d365a8ecae538e45f80a08d612cd
```

The result is effectively neutral (`+0.02924526966996 tok/s`, about `+0.063%`).
It proves the page-concatenation mismatch was real in code but not the dominant
runtime cost on this target. The gate stays off by default.

## Dense Linear Transpose Cache Probe

One smaller mismatch with the local code was that `SwitchLinear` cached its
dense transposed weight, while `Linear` rebuilt a transpose view inside every
dense forward. The probe added a cached `WeightT` field to `Linear` and reused
it for dense matmuls.

Target rerun:

| Path | Decode tok/s | Notes |
| --- | ---: | --- |
| Dense linear transpose cache | `45.9393904182794` | 3 runs: `45.958544400246424`, `46.12575826364638`, `45.733868590945406` |

Representative command:

```bash
env MLX_METALLIB_PATH=/Users/snider/Code/core/go-mlx/dist/lib/mlx.metallib /Users/snider/Code/core/go-mlx/bin/lthn-mlx driver-profile -json -include-output=false -context 4096 -prompt "Answer in one short sentence: why does retained model state matter?" -max-tokens 128 -runs 3 -trace-token-phases /Users/snider/.cache/huggingface/hub/models--mlx-community--gemma-4-e2b-it-4bit/snapshots/99d9a53ff828d365a8ecae538e45f80a08d612cd
```

The patch was reverted. On this target the dense transpose view is not the
dominant cost, and retaining the lazy transposed handle made the default path
slower than the surrounding paged-cache controls.

## Compiled Per-Layer Inputs Probe

The native phase trace showed `gemma4.layer.00.output` as a large materialisation
point because the first per-layer gate consumes Gemma 4's lazily built
per-layer-input tensor. A diagnostic gate now wraps that tensor construction in
a cached shapeless MLX compiled closure:

```bash
GO_MLX_ENABLE_COMPILED_GEMMA4_PER_LAYER_INPUTS=1
```

Same-binary reruns:

| Path | Decode tok/s | Notes |
| --- | ---: | --- |
| Gate off | `46.9841490339839` | 3 runs: `46.84891284169694`, `47.10549942668368`, `46.998034833571076` |
| Compiled per-layer inputs | `46.93672879306734` | 3 runs: `46.88946529014483`, `47.06309143201619`, `46.857629657040995` |

Representative command:

```bash
env GO_MLX_ENABLE_COMPILED_GEMMA4_PER_LAYER_INPUTS=1 MLX_METALLIB_PATH=/Users/snider/Code/core/go-mlx/dist/lib/mlx.metallib /Users/snider/Code/core/go-mlx/bin/lthn-mlx driver-profile -json -include-output=false -context 4096 -prompt "Answer in one short sentence: why does retained model state matter?" -max-tokens 128 -runs 3 -trace-token-phases /Users/snider/.cache/huggingface/hub/models--mlx-community--gemma-4-e2b-it-4bit/snapshots/99d9a53ff828d365a8ecae538e45f80a08d612cd
```

This confirms the per-layer-input tensor is a real materialisation component,
but compiling it separately does not reduce the steady decode path. The gate is
disabled by default.

## Disabled Per-Layer Inputs Diagnostic

The previous trace and compiled-input probe pointed at the Gemma 4 per-layer
input tensor. A correctness-breaking diagnostic gate was added to skip
`computePerLayerInputs` entirely:

```bash
GO_MLX_DISABLE_GEMMA4_PER_LAYER_INPUTS=1
```

This is not a production path. Gemma 4 requires those per-layer side inputs, so
the generated logits are semantically invalid. The run is useful only because it
isolates the cost of the second stack.

Target rerun:

| Path | Decode tok/s | Notes |
| --- | ---: | --- |
| Per-layer inputs disabled | `114.9355811775564` | 3 runs: `117.0486414046229`, `117.46595644094181`, `110.29214568710452`; generated `[128,128,128]` tokens |

Representative command:

```bash
env GO_MLX_DISABLE_GEMMA4_PER_LAYER_INPUTS=1 MLX_METALLIB_PATH=/Users/snider/Code/core/go-mlx/dist/lib/mlx.metallib /Users/snider/Code/core/go-mlx/bin/lthn-mlx driver-profile -json -include-output=false -context 4096 -prompt "Answer in one short sentence: why does retained model state matter?" -max-tokens 128 -runs 3 -trace-token-phases /Users/snider/.cache/huggingface/hub/models--mlx-community--gemma-4-e2b-it-4bit/snapshots/99d9a53ff828d365a8ecae538e45f80a08d612cd
```

JSON output is saved as
`docs/runtime/2026-05-17-gemma4-e2b-disable-per-layer-inputs-rerun.json`.
Stderr is saved beside it with the same stem and `.stderr` suffix.

```text
successful_runs: 3
generated_tokens: 384
visible_tokens: 381
decode_tokens_per_sec_average: 114.9355811775564
prefill_tokens_per_sec_average: 718.891541170347
steady token phases after warmup: 375
steady sample_eval_duration average: 7.890701744ms/token
steady total_duration average: 8.771842768ms/token
peak_memory_bytes: 3835433982
active_memory_bytes: 2976142934
```

The corresponding E2B q4 tensor shapes explain why the delta looks like a
second model-side stack rather than small host overhead:

```text
language_model.model.per_layer_model_projection.weight: bf16 [8960,1536]
language_model.model.embed_tokens_per_layer.weight: q4-packed u32 [262144,1120]
language_model.model.embed_tokens_per_layer.scales: [262144,140]
language_model.model.embed_tokens_per_layer.biases: [262144,140]
```

The correct optimisation is therefore not to skip per-layer inputs. The next
valid boundary has to preserve the side-input semantics while avoiding repeated
full projection/materialisation of the per-token `[35,256]` tensor every decode
step, either by fusing the projection/norm/add/split path, pushing slices down
to layer consumption, or caching only cases that are provably token-id stable.

## Quantized Embedding Row-Gather Rerun

The diagnostic pointed at the right stack, but the concrete bug was more
specific: quantized `Embedding.Forward` dequantized the whole vocabulary table
before taking the requested token rows. For Gemma 4 E2B's per-layer embedding
table, that means the q4-packed `[262144,1120]` table can expand to the full
side-input table in the decode path. The valid fix gathers packed weight rows,
scale rows, and bias rows first, then dequantizes only those selected rows.

Target rerun on the default valid path:

| Path | Decode tok/s | Notes |
| --- | ---: | --- |
| Quantized embedding row gather | `121.9379742475021` | 3 runs: `120.35003784437026`, `123.6154742394561`, `121.84841065867997`; generated `[20,20,20]` tokens |

Representative command:

```bash
env MLX_METALLIB_PATH=/Users/snider/Code/core/go-mlx/dist/lib/mlx.metallib /Users/snider/Code/core/go-mlx/bin/lthn-mlx driver-profile -json -include-output=false -context 4096 -prompt "Answer in one short sentence: why does retained model state matter?" -max-tokens 128 -runs 3 -trace-token-phases /Users/snider/.cache/huggingface/hub/models--mlx-community--gemma-4-e2b-it-4bit/snapshots/99d9a53ff828d365a8ecae538e45f80a08d612cd
```

JSON output is saved as
`docs/runtime/2026-05-17-gemma4-e2b-quantized-embedding-row-gather-rerun.json`.
Stderr is saved beside it with the same stem and `.stderr` suffix.

```text
load.cache_mode: paged
successful_runs: 3
generated_tokens: 60
visible_tokens: 60
decode_tokens_per_sec_average: 121.9379742475021
prefill_tokens_per_sec_average: 747.9028788388396
steady token phases after warmup: 54
steady sample_eval_duration average: 7.111331777777778ms/token
steady total_duration average: 8.140010037037037ms/token
peak_memory_bytes: 3166205126
active_memory_bytes: 2971768406
```

Compared with the resolved-load baseline
(`46.50145764359926 tok/s`, peak `8579365290` bytes), this is a
`+75.43651660390284 tok/s` improvement and cuts peak memory by roughly
`5413160164` bytes. It also beats the correctness-breaking skip diagnostic on
this target command while keeping the required Gemma 4 side inputs.

## Current Blocker

The exact E2B q4 target path now clears the 100 tok/s floor on the default
valid path. The final current-default rerun reports `124.88170583124456 tok/s`
on the exact target command with three full 128-token runs; JSON is saved as
`docs/runtime/2026-05-17-gemma4-e2b-final-current-default-rerun.json`.

After the Gemma 4 mixed-quant loader fix for the 26B A4B comparison, the
current binary was rebuilt and the exact E2B command was rerun:

```text
go-mlx SHA-256: c1034cf834b9c40d65c0e9bcf2652f5c2232965ef1715188c89fb5eff8abf141
successful_runs: 3
generated_tokens: 384
visible_tokens: 384
decode_tokens_per_sec_average: 121.19859628423075
run tok/s: 124.45518442558254, 119.37332258565571, 119.767281841454
prefill_tokens_per_sec_average: 857.3137242568481
peak_memory_bytes: 3177560106
stderr_bytes: 0
```

JSON output is saved as
`docs/runtime/2026-05-17-gemma4-e2b-mixed-quant-loader-rerun.json`. This is
below the previous best by normal run variance but still safely above the
`100 tok/s` target.

The remaining external blocker in this report is llama.cpp parity, not
`mlx_lm`. The active comparator is the closest local Gemma 4 26B A4B q4 pair:
go-mlx q4 MLX safetensors versus llama.cpp `Q4_K_M` GGUF.

The llama.cpp MoE read exposed one concrete mismatch: its Gemma expert path
keeps `gate_up` fused when the tensor exists, while go-mlx had split the same
source tensor into `gate_proj` and `up_proj` and then executed both expert
projections. go-mlx now retains `experts.switch_glu.gate_up_proj` and uses the
fused projection only for single-token decode. The first ungated attempt also
used the fused path for prefill and regressed the long-prefill lane, so the
accepted implementation is deliberately decode-only.

Current evidence after the automatic long-prompt last-token prefill change:

```text
go-mlx SHA-256: dd212338c1864b6acb630bb5f534986432d1c189d17e100ae8ab3a3ee230a352
short prompt: 29 tokens
go-mlx decode: 56.220244342267904 tok/s
go-mlx prefill: 443.8939306138111 tok/s
go-mlx decode runs: 56.138136941728334, 56.25724605690424, 56.26535002817114
long prompt: 2061 tokens
go-mlx long prefill: 903.0290085147915 tok/s
llama.cpp Q4_K_M decode: 89.000726 tok/s
llama.cpp Q4_K_M long prefill: 2184.109033 tok/s
```

The decode-only fused expert path remains a small improvement over the earlier
`55.96521969803896 tok/s` go-mlx decode result. The long-prompt prefill path
now also avoids materialising full `[sequence,vocab]` logits before slicing the
last row: `prefillTokenBlockOnce` automatically uses
`ForwardLastTokenLogits` when the prompt chunk is at least 512 tokens, while
short prompts remain on the full-logits path unless
`GO_MLX_ENABLE_LAST_LOGITS_PREFILL=1` explicitly forces the old experiment.
This improves the clean 2061-token long-prefill run from
`862.5952429295362 tok/s` to `903.0290085147915 tok/s`, and reduces peak memory
from `19811354828` to `17974597848` bytes.

The change does not close parity: llama.cpp remains `1.58x` faster on decode
and `2.42x` faster on long prefill.
The short-prompt JSON is saved as
`docs/runtime/2026-05-17-go-mlx-gemma4-26b-a4b-q4-auto-last-logits-llamacpp-comparison-longdecode-rerun2.json`;
the long-prefill JSON is saved as
`docs/runtime/2026-05-17-go-mlx-gemma4-26b-a4b-q4-auto-last-logits-longprefill-one-run-llamacpp-comparison.json`.

A tiny-tail chunk coalescing probe was also tried because the 2061-token prompt
is chunked as `2048 + 13`. It was negative: forcing one 2061-token prefill pass
recorded only `862.4738054025554 tok/s` with the same model. That diagnostic
is saved as
`docs/runtime/2026-05-17-go-mlx-gemma4-26b-a4b-q4-auto-last-logits-tail-coalesce-longprefill-one-run-llamacpp-comparison.json`;
the code path was reverted.

A llama.cpp-shaped shared-KV last-token trim was then tested after the final
Gemma 4 KV-owning layer. It preserved the final token RoPE position and trimmed
sliding shared KV to the local window, but the result was not worth carrying:
one clean long-prefill run reached only `911.1355151113232 tok/s`, and the
short-prompt 128-token decode check fell to `53.616341210113625 tok/s`.
Those rejected diagnostics are saved as
`docs/runtime/2026-05-17-go-mlx-gemma4-26b-a4b-q4-shared-kv-last-token-trim-longprefill-one-run-llamacpp-comparison.json`
and
`docs/runtime/2026-05-17-go-mlx-gemma4-26b-a4b-q4-shared-kv-last-token-trim-llamacpp-comparison-longdecode.json`;
the source change was reverted.

The next active-lane probe tried the fixed-cache compiled Gemma 4 layer on the
same 26B A4B q4 versus llama.cpp Q4_K_M workload. Full-context fixed cache
regressed to `48.211754489053696 tok/s` decode and
`402.4998847052011 tok/s` prefill. A tighter 160-slot fixed cache improved to
`53.69079065280556 tok/s` decode and `433.71986471660057 tok/s` prefill, but
still missed the accepted default (`56.220244342267904 tok/s` decode). Both
stderr files are empty. The diagnostics are saved as
`docs/runtime/2026-05-17-go-mlx-gemma4-26b-a4b-q4-fixed-cache-compiled-layer-llamacpp-comparison-longdecode.json`
and
`docs/runtime/2026-05-17-go-mlx-gemma4-26b-a4b-q4-fixed-cache160-compiled-layer-llamacpp-comparison-longdecode.json`.

Two traces then narrowed the remaining 26B gap. The accepted default path under
`-trace-token-phases` records `53.24884702642772 tok/s` with trace overhead.
Excluding warmup and the final token, 125 steady samples average
`18.887ms/token`; `17.432ms` is `sample_eval_duration`, while forward
construction is only `1.414ms`. With `GO_MLX_TRACE_FORWARD_EVAL=1`, the trace
forces 120 native events per token on the 30-layer model. Across 29 steady
decode samples, forced-boundary totals are about `20.082ms/token` FFN,
`12.393ms/token` attention, `7.990ms/token` layer output, and
`7.398ms/token` attention residual. Those traces are saved as
`docs/runtime/2026-05-17-go-mlx-gemma4-26b-a4b-q4-default-token-phase-trace-llamacpp-comparison.json`
and
`docs/runtime/2026-05-17-go-mlx-gemma4-26b-a4b-q4-native-phase-trace-llamacpp-comparison.json`.
This points the next implementation at a broader llama.cpp-shaped one-token
block or native MoE/FFN boundary, not another isolated final-logits, tiny-tail,
shared-KV trim, or fixed-cache wrapper.

A native fused-experts bridge was then implemented as the direct MoE/FFN probe:
`gate_up` gather, GELU, down gather, expert weighting, and top-k sum moved
behind one opt-in native wrapper. It was correct on a dense unit test but
negative on the real 26B A4B q4 llama.cpp lane: three full 128-token runs
recorded `53.08901433576139 tok/s` decode and `431.27066684929787 tok/s`
short prefill, below the accepted default. Stderr was empty, and the source
change was reverted. The rejected diagnostic is saved as
`docs/runtime/2026-05-17-go-mlx-gemma4-26b-a4b-q4-native-fused-experts-llamacpp-comparison-longdecode.json`.
The follow-up FFN split trace keeps the active comparator on llama.cpp and adds
trace-only MoE sub-boundaries. One 32-token diagnostic run records
`14.452280580872943 tok/s` under trace overhead. Across 29 steady decode
samples it records 270 native events/token, with the largest totals in
`ffn_experts` (`13.736ms/token`), attention (`10.614ms/token`),
`ffn_local_mlp` (`8.354ms/token`), and `ffn_router` (`7.560ms/token`). The
trace is saved as
`docs/runtime/2026-05-17-go-mlx-gemma4-26b-a4b-q4-native-phase-ffn-split-trace-llamacpp-comparison.json`.
Together these rule out a small native MoE graph wrapper as the missing
`~1.58x` decode factor; the next attempt needs either a broader one-token block
or a lower-level quantized MoE kernel shaped closer to llama.cpp.

The static kernel read makes that more concrete. go-mlx currently reaches MLX
through `SwitchLinear.Forward`, which calls `GatherQMM` with RHS expert indices
and `sorted=false`. MLX's Metal `GatherQMM::eval_gpu` only uses the
specialised `gather_qmm_rhs` path when indices are globally sorted and the
batch is large enough (`M == 1`, `B >= 16`, `B / E >= 4`). Single-token Gemma 4
26B decode is top-k 8 over 128 experts, so it cannot use that batched RHS
kernel. llama.cpp lowers the same work to `GGML_OP_MUL_MAT_ID`, using
`kernel_mul_mv_id` for small token counts and `kernel_mul_mm_id` plus an
expert-ID map for larger batches, with Metal specialisations for quant types
and `n_expert_used`. The next go-mlx target is therefore an ID-matvec/ID-matmul
native boundary, not sorted MLX gather alone. The source now also emits
trace-only `ffn_expert.gate_up`, `activation`, `down`, `weighted`, and `sum`
events under `GO_MLX_TRACE_FORWARD_EVAL=1`; the next Metal-available trace can
split the routed expert bucket without affecting default execution.
The matching code-side scaffold is
`go/internal/metal/expert_id_matvec.go`: `quantizedExpertIDMatVec` consumes MLX
affine-packed q2/q4/q8 expert rows plus route expert ids and matches a CPU q4
reference on small and multi-pack tensors. One SIMD group now reduces each
routed output row, closer to the llama.cpp ID-matvec primitive than the first
serial proof. Gemma 4 can route through it only with
`GO_MLX_ENABLE_EXPERT_ID_MATVEC=1`, and the unit regression compares that
opt-in path against the existing MLX `GatherQMM` result. The custom kernel
handle is cached per shape so repeated decode calls do not rebuild it. The
down-projection side now uses a weighted expert-ID matvec-sum kernel, folding
route weighting and top-k summation into the down matvec instead of leaving
them as separate MLX nodes. It remains disabled by default until the
llama.cpp-lane benchmark shows it helps.

A full 26B A4B q4 env-gated model probe was attempted with the llama.cpp
comparison prompt, but the local runtime failed before any generation because
MLX reported no usable Metal device for native model load. A follow-up
`driver-profile -expert-id-matvec` diagnostic flag enables the same internal
gate without a second environment variable and records
`runtime_gates.GO_MLX_ENABLE_EXPERT_ID_MATVEC=1`. That profile is valid but
negative: `55.98273536629838 tok/s` decode and `449.436848070603 tok/s` short
prefill across three full 128-token runs. It is below the accepted go-mlx
decode control (`56.220244342267904 tok/s`), while llama.cpp `Q4_K_M` remains
`1.5898x` faster on decode. The failed env-gated JSON is saved as
`docs/runtime/2026-05-17-go-mlx-gemma4-26b-a4b-q4-expert-id-matvec-gated-llamacpp-comparison-longdecode.json`;
the valid negative diagnostic is saved as
`docs/runtime/2026-05-17-go-mlx-gemma4-26b-a4b-q4-expert-id-matvec-flag-llamacpp-comparison-longdecode.json`.
Neither replaces the accepted go-mlx or llama.cpp numbers.

A narrower fused-activation variant then moved `GELU(gate) * up` into the
custom expert-ID gate_up kernel behind
`driver-profile -expert-id-fused-activation`. It is valid but not meaningful:
same-binary controls record `56.21477992583666 tok/s` for the default path,
`56.06328243808281 tok/s` for non-fused expert-ID matvec, and
`56.295534088943356 tok/s` for expert-ID fused activation. The fused variant
is only `+0.14%` over the same-binary default control, while llama.cpp
`Q4_K_M` remains `1.5809x` faster. The diagnostic JSON is saved as
`docs/runtime/2026-05-17-go-mlx-gemma4-26b-a4b-q4-expert-id-fused-activation-llamacpp-comparison-longdecode.json`.

The next llama.cpp-only follow-up targeted the batched prefill side of that
same read. `driver-profile` now has `-prompt-file` for repeatable long-context
inputs and `-sorted-expert-prefill` for
`GO_MLX_ENABLE_SORTED_EXPERT_PREFILL=1` without adding a second environment
variable. The sorted path flattens Gemma 4 prefill routes, sorts them by
expert id, runs split gate/up/down `GatherQMM` with `sorted=true`, then
restores route order before weighting and summing. On the same binary and a
`README.md` prompt-file input (`2204` prompt tokens), the default control is
`914.0299819202297 tok/s` prefill and `31.048941804155767 tok/s` decode; the
same-binary sorted route path is `1914.0303789361128 tok/s` prefill and
`31.508051014734626 tok/s` decode. That is a `2.0940x` prefill speedup and
puts go-mlx at `87.6%` of the existing llama.cpp `Q4_K_M` `pp2048`
throughput (`2184.109033 tok/s`). The artefacts are:
`docs/runtime/2026-05-17-go-mlx-gemma4-26b-a4b-q4-readme-default-llamacpp-comparison-longdecode.json`
and
`docs/runtime/2026-05-17-go-mlx-gemma4-26b-a4b-q4-sorted-expert-prefill-readme-llamacpp-comparison-longdecode.json`.

The next llama.cpp-only follow-up targeted the long-context decode side.
`driver-profile -paged-decode-fast-concat` enables
`GO_MLX_ENABLE_PAGED_DECODE_FAST_CONCAT=1`; when single-token decode spans
multiple paged KV blocks, the path concatenates the paged state once and calls
regular SDPA instead of the hand-rolled paged attention loop. With sorted
prefill plus fast concat, the same prompt-file lane records
`1909.1904478108413 tok/s` prefill and `42.372384580120396 tok/s` decode.
This is a `1.3448x` decode speedup over the same-binary sorted-prefill-only
control, but llama.cpp `Q4_K_M` `tg128` at `p2048` is still
`92.624334 tok/s`, or `2.186x` faster. Prefill is now close to the llama.cpp
result; long-context decode remains the active parity miss. The artefact is
`docs/runtime/2026-05-17-go-mlx-gemma4-26b-a4b-q4-sorted-prefill-paged-fast-concat-readme-llamacpp-comparison-longdecode.json`.

The next probe moved the existing fixed-cache and compiled Gemma 4 decode
diagnostics onto CLI runtime gates so the llama.cpp lane no longer needs
env-only package-init switches. The command used `-cache-mode paged`,
`-fixed-gemma4-cache`, `-fixed-gemma4-shared-mask`,
`-compiled-gemma4-layer`, and `-sorted-expert-prefill` on the same
`README.md` prompt-file workload. It records `1876.6924105183755 tok/s`
prefill and `48.93511098804883 tok/s` decode. This is a `1.5531x` decode
speedup over sorted-prefill-only and `1.1549x` over the paged fast-concat
probe, but llama.cpp `Q4_K_M` `tg128` at `p2048` is still `1.8928x` faster.
The artefact is
`docs/runtime/2026-05-17-go-mlx-gemma4-26b-a4b-q4-sorted-prefill-fixed-compiled-readme-llamacpp-comparison-longdecode.json`.

Adding `driver-profile -direct-greedy-token` to the same fixed-cache compiled
lane records a 3-run average of `1908.4658285603446 tok/s` prefill and
`49.75515922842408 tok/s` decode. That is only `1.0168x` over the fixed-cache
compiled probe. llama.cpp `Q4_K_M` `tg128` at `p2048` remains `1.8616x`
faster. The artefact is
`docs/runtime/2026-05-17-go-mlx-gemma4-26b-a4b-q4-sorted-prefill-fixed-compiled-direct-greedy-3run-readme-llamacpp-comparison-longdecode.json`.

The compiled Gemma 4 decode graph was then extended to include MoE layers
instead of only dense MLP layers. The focused tiny-MoE regression passes, but
the full README prompt-file profile remains in the same band:
`1882.3003597479092 tok/s` prefill and `49.57330167871466 tok/s` decode for
one run. Adding `-expert-id-fused-activation` on top averaged
`49.705483987003994 tok/s` across three runs, below the direct-greedy 3-run
average. The evidence says MLX-compiling the current MoE graph is not enough;
the remaining llama.cpp gap still needs a lower-level MoE/KV/decode boundary.

A final same-lane probe removed `-compiled-gemma4-layer` and combined sorted
prefill, fixed-cache/shared-mask, direct greedy, and the expert-ID fused
activation path so the single-token decode branch can use the custom expert-ID
kernel instead of the compiled MoE graph. It records `1915.3373741969128 tok/s`
prefill and `49.973204322219345 tok/s` decode across three runs. That is the
current best go-mlx long-context decode result in this report, but it is only
`+0.44%` over the prior direct-greedy 3-run sample; llama.cpp `Q4_K_M` `tg128`
at `p2048` remains `1.8535x` faster. A same-prompt-length llama.cpp check records
`pp2204` at `2109.335561 tok/s` and `tg128` at `91.451031 tok/s`, leaving a
`1.1013x` prefill gap and a `1.8300x` decode gap. The go-mlx artefact is
`docs/runtime/2026-05-17-go-mlx-gemma4-26b-a4b-q4-sorted-prefill-expert-id-fused-direct-greedy-3run-readme-llamacpp-comparison-longdecode.json`.

While reviewing this path, the older C++ `-native-gemma4-layer` gate was also
narrowed back to dense-only layers. The Go/MLX compiled graph can represent
Gemma 4 MoE through `Gemma4Experts.forward`, but the C++ native-layer ABI does
not pass router or expert tensors, so allowing MoE there would be a correctness
bug rather than a speed path.

A follow-up cache-shape probe tested preserving Gemma 4's 1024-token sliding
cache bound inside the fixed-cache lane. That exposed and fixed two
`FixedKVCache` overflow correctness cases: multi-token prompt overflow must
return the full attention context while storing the bounded tail, and
single-token overflow must return the stored tail so post-eval `Detach()` does
not strip an unevaluated cache. The diagnostic itself is negative:

`docs/runtime/2026-05-17-go-mlx-gemma4-26b-a4b-q4-sliding-cache-bound-direct-greedy-3run-readme-llamacpp-comparison-longdecode.json`

```text
prefill: 1806.8318924630082 tok/s
decode: 40.76006207167587 tok/s
peak_memory_bytes: 71228950132
```

The active fixed-cache lane was restored to uniform context-sized fixed caches,
with non-fixed paged cache replacement still preserving inherited rotating-cache
bounds. The restored current-code same-lane run is:

`docs/runtime/2026-05-17-go-mlx-gemma4-26b-a4b-q4-fixed-uniform-cache-direct-greedy-3run-readme-llamacpp-comparison-longdecode.json`

```text
prefill: 1923.322483219664 tok/s
decode: 49.71518402860789 tok/s
peak_memory_bytes: 19212389680
bin/lthn-mlx SHA-256: 5a4081baa3c2cd9f492d333b01c04328f60ae2fe15d19015f35ddf68f2661e38
```

Against same-prompt-length llama.cpp `Q4_K_M`, that is `1.0967x` behind on
prefill and `1.8395x` behind on decode.

A follow-up llama.cpp source read found that Gemma 4 router logits come from the
post-attention residual stream, not the pre-FFN2-normalised expert input. The
Go graph and compiled decode graph now match that boundary while leaving the
expert input normalised. The same prompt-file lane records:

`docs/runtime/2026-05-17-go-mlx-gemma4-26b-a4b-q4-router-residual-fixed-uniform-direct-greedy-3run-readme-llamacpp-comparison-longdecode.json`

```text
prefill: 1933.6368792628773 tok/s
decode: 50.23367760579547 tok/s
peak_memory_bytes: 19212389680
```

Against same-prompt-length llama.cpp `Q4_K_M`, that is `1.0909x` behind on
prefill and `1.8205x` behind on decode. A two-output down-projection matvec
diagnostic regressed to `48.4963971321882 tok/s` decode and was reverted:
`docs/runtime/2026-05-17-go-mlx-gemma4-26b-a4b-q4-expert-down-two-col-fixed-uniform-direct-greedy-3run-readme-llamacpp-comparison-longdecode.json`.
No new `mlx_lm` measurements were taken.

### Split/BF16 Expert-ID Shared-Input Follow-Up

The active 26B A4B q4 MLX safetensors store expert `gate_proj` and `up_proj`
tensors separately, with BF16 q4 scale/bias sidecars. The previous
fused-`gate_up` expert-ID gate therefore fell back on this model. The new
expert-ID path handles split gate/up tensors, BF16/F16/F32 sidecars, fused
`GELU(gate) * up`, and one shared hidden row routed through multiple top-k
expert IDs.

Trace artefact:

`docs/runtime/2026-05-17-go-mlx-gemma4-26b-a4b-q4-split-fused-expert-id-shared-input-native-phase-trace.json`

```text
stderr_bytes: 0
native phases include activation_split_id_matvec and down_weighted_sum_id_matvec
```

Intermediate 3-run artefacts:

```text
split expert-ID active:
  prefill: 1939.2172632050945 tok/s
  decode: 62.52025013199337 tok/s

split expert-ID fused activation:
  prefill: 1941.0884632916652 tok/s
  decode: 68.22675114228564 tok/s
```

Current shared-input artefact:

`docs/runtime/2026-05-17-go-mlx-gemma4-26b-a4b-q4-split-fused-expert-id-shared-input-fixed-uniform-direct-greedy-3run-readme-llamacpp-comparison-longdecode.json`

```text
prompt_tokens: 2204
prefill: 1923.9974775252285 tok/s
decode: 70.54498924012704 tok/s
run decode tok/s: 69.91341816877653, 70.25276863828591, 71.46878091331867
peak_memory_bytes: 19212389664
active_memory_bytes: 17457260720
stderr_bytes: 0
```

Against same-prompt-length llama.cpp `Q4_K_M`
(`pp2204: 2109.335561 tok/s`, `tg128: 91.451031 tok/s`), this leaves a
`1.0963x` prefill gap and a `1.2964x` decode gap. The decode lane is now
`1.4043x` faster than the router-residual result, but still below the `100
tok/s` floor and behind llama.cpp.

The non-native token-phase profile for the same lane avoids the diagnostic
per-layer materialisations and records:

`docs/runtime/2026-05-17-go-mlx-gemma4-26b-a4b-q4-split-fused-expert-id-shared-input-token-phases.json`

```text
decode: 71.59452329863376 tok/s
steady token average: 14.05959232ms
steady Eval(next): 12.724946032ms
steady forward graph construction: 1.297721312ms
stderr_bytes: 0
```

A one-run native dense MLP GELU probe is neutral-to-negative:

`docs/runtime/2026-05-17-go-mlx-gemma4-26b-a4b-q4-split-fused-shared-input-native-mlp-probe.json`

```text
decode: 71.44678366026884 tok/s
prefill: 1927.4283286475602 tok/s
stderr_bytes: 0
```

That keeps the next candidate boundary on larger eval/materialisation work,
not another standalone MLP wrapper.

### Packed-Column Expert-ID Follow-Up

The expert-ID kernels were still walking q4-packed weights as scalar input
columns. In q4 this makes adjacent SIMD lanes reload the same packed `uint32`
word and extract one nibble each. The packed-column rewrite changes the loop so
each lane loads one packed word, unpacks its q values locally, and contributes
all of them before the SIMD reduction.

Final packed-column artefact:

`docs/runtime/2026-05-18-go-mlx-gemma4-26b-a4b-q4-packed-expert-id-final-fixed-uniform-direct-greedy-3run-readme-llamacpp-comparison-longdecode.json`

```text
prompt_tokens: 2204
prefill: 1936.5495347431952 tok/s
decode: 79.1105587686013 tok/s
run decode tok/s: 79.01523558809173, 79.17622090660484, 79.1402198111073
peak_memory_bytes: 19212389664
active_memory_bytes: 17457260720
stderr_bytes: 0
/private/tmp/lthn-mlx-packed-expert-id SHA-256: f6d8e3853c305fff69bf8d8c20fa4a885bbcc6875b29101181af1de4c0e86a77
```

Against same-prompt-length llama.cpp `Q4_K_M`
(`pp2204: 2109.335561 tok/s`, `tg128: 91.451031 tok/s`), this leaves a
`1.0892x` prefill gap and a `1.1560x` decode gap. It is `1.1214x` faster than
the prior shared-input split expert-ID result, but still `1.2641x` below the
`100 tok/s` floor.

Right-sizing the fixed Gemma 4 cache then exposed another concrete source of
extra attention work. The default fixed-cache lane keeps the graph stable by
allocating the full 4096-slot context, but this README prompt-file comparison
only needs about 2204 prompt tokens plus 128 decode tokens. Setting
`GO_MLX_FIXED_GEMMA4_CACHE_SIZE=2336` keeps the workload inside capacity while
avoiding the larger fixed attention scan.

Best 2336-slot fixed-cache artefact:

`docs/runtime/2026-05-18-go-mlx-gemma4-26b-a4b-q4-packed-expert-id-fixed-cache2336-3run-readme-llamacpp-comparison-longdecode.json`

```text
prompt_tokens: 2204
prefill: 1937.0948107149452 tok/s
decode: 84.23477753697784 tok/s
run decode tok/s: 84.1698833924705, 84.12789512233812, 84.4065540961249
peak_memory_bytes: 18419404064
active_memory_bytes: 16664275120
stderr_bytes: 0
bin/lthn-mlx SHA-256: f2a5f2d07239eb4c3e401047c20c6fa817d97f1a99975cf498be1daa5531a394
```

That is `1.0648x` faster than the packed 4096-slot baseline on decode and
reduces the same-prompt llama.cpp decode gap to `1.0857x`. It is still
`1.1872x` short of `100 tok/s`.

The same request-sized capacity is now derived automatically for one-shot
generation when `-fixed-gemma4-cache` is enabled and
`GO_MLX_FIXED_GEMMA4_CACHE_SIZE` is unset. The generation cache builder uses
`prompt_tokens + max_tokens`, rounded up to 32 slots, which selects 2336 for
this 2204-token README prompt plus 128-token decode.

Automatic right-sized fixed-cache artefact:

`docs/runtime/2026-05-18-go-mlx-gemma4-26b-a4b-q4-packed-expert-id-auto-fixed-cache-3run-readme-llamacpp-comparison-longdecode.json`

```text
prompt_tokens: 2204
prefill: 1935.3610403257746 tok/s
decode: 84.01009717307203 tok/s
run decode tok/s: 84.14374646177602, 84.27602963804662, 83.61051541939345
peak_memory_bytes: 18419404064
active_memory_bytes: 16664275120
stderr_bytes: 0
```

That is within `0.27%` of the manual 2336-slot sample and leaves same-prompt
llama.cpp `1.0886x` faster on decode. An earlier cold auto-sized process is
preserved as
`docs/runtime/2026-05-18-go-mlx-gemma4-26b-a4b-q4-packed-expert-id-auto-fixed-cache-cold-3run-readme-llamacpp-comparison-longdecode.json`;
its first run dipped to `78.8853520463259 tok/s`, while the second and third
runs returned to the `83-84 tok/s` band.

A follow-up tested preserving Gemma 4's 1024-token sliding-window capacity
inside the fixed-cache lane. The native overflow update now uses a compiled
`take` plus final-slot overwrite path because MLX compile cannot infer the
output shapes for `slice` or `roll` in that closure. Correctness is covered by
`TestDecode_nativeFixedSlidingSingleTokenAttention_Good`, but the benchmark is
negative:

`docs/runtime/2026-05-18-go-mlx-gemma4-26b-a4b-q4-sliding-fixed-cache-3run-readme-llamacpp-comparison-longdecode.json`

```text
prompt_tokens: 2204
prefill: 2033.3865559253882 tok/s
decode: 73.05984177869179 tok/s
peak_memory_bytes: 18318341380
active_memory_bytes: 16127004820
stderr_bytes: 0
```

That leaves same-prompt llama.cpp `1.2517x` faster on decode, so the active
lane was restored to uniform request-sized fixed caches. The restored rerun is:

`docs/runtime/2026-05-18-go-mlx-gemma4-26b-a4b-q4-restored-uniform-fixed-cache-3run-readme-llamacpp-comparison-longdecode.json`

```text
prompt_tokens: 2204
prefill: 1925.9978025157088 tok/s
decode: 83.59574625080806 tok/s
peak_memory_bytes: 18419404064
active_memory_bytes: 16664275120
stderr_bytes: 0
bin/lthn-mlx SHA-256: a634fc8418a2b7cf0494c889e4241df3aa55144d936f2782daf7364661cc4373
```

The restored code is within the established `83-84 tok/s` band, but it is not a
new best. The earlier automatic sample at `84.01009717307203 tok/s` remains the
best verified no-draft go-mlx result for this lane.

## Prefill Chunk-Size Sweep

`driver-profile` now accepts `-prefill-chunk-size` as a diagnostic load
override. The active 26B A4B q4 README prompt-file lane still uses sorted
expert prefill, the packed expert-ID fused-activation kernels, request-sized
fixed cache, shared fixed mask, and direct greedy decode.

Rebuilt binary:

```text
bin/lthn-mlx SHA-256: ff7363f29ad02dcb1da3204423ba9f121250c0d03cb0b41df22c3e9e2d292810
```

Three-run results:

| Prefill chunk | Prefill tok/s | Decode tok/s | Peak bytes | Artefact |
| ---: | ---: | ---: | ---: | --- |
| `1024` | `1658.2779108140055` | `83.31228694999267` | `18148762344` | `docs/runtime/2026-05-18-go-mlx-gemma4-26b-a4b-q4-prefill-chunk1024-3run-readme-sweep.json` |
| `2048` | `1933.0886541161783` | `83.86143957778368` | `18419404064` | `docs/runtime/2026-05-18-go-mlx-gemma4-26b-a4b-q4-prefill-chunk2048-3run-readme-sweep.json` |
| `4096` | `2101.369627343361` | `83.74497136862215` | `18591487096` | `docs/runtime/2026-05-18-go-mlx-gemma4-26b-a4b-q4-prefill-chunk4096-3run-readme-sweep.json` |

The result answers the chunking question directly: for this 2204-token prompt,
`2048` is a two-pass prefill shape, while `4096` keeps the prompt in one
prefill chunk and wins. The `4096` override is `1.0871x` faster than `2048`
prefill and reaches `99.62%` of same-prompt llama.cpp `Q4_K_M` prefill
(`2101.369627343361` versus `2109.335561 tok/s`). Decode does not materially
move, so the remaining same-prompt llama.cpp gap is still the `83-84 tok/s`
go-mlx decode band versus `91.451031 tok/s`.

The high-memory planner was then updated so the 64GB class selects `4096`
prefill chunks without a CLI override. The rebuilt default run confirms the
load setting and keeps prefill near parity:

`docs/runtime/2026-05-18-go-mlx-gemma4-26b-a4b-q4-default-wide-prefill-planner-3run-readme.json`

```text
load.prefill_chunk_size: 4096
prompt_tokens: 2204
prefill: 2088.289027094623 tok/s
run prefill tok/s: 2055.580173863937, 2104.0715909404157, 2105.2153164795163
decode: 83.09590032942343 tok/s
run decode tok/s: 82.67387547724431, 83.03889708276647, 83.5749284282595
peak_memory_bytes: 18591487096
active_memory_bytes: 16664275120
stderr_bytes: 0
```

The no-override planner path reaches `99.00%` of same-prompt llama.cpp prefill.
It does not solve decode: llama.cpp remains `1.1005x` faster on generation.

The 2336-slot token-phase profile is:

`docs/runtime/2026-05-18-go-mlx-gemma4-26b-a4b-q4-packed-expert-id-fixed-cache2336-token-phases.json`

```text
decode: 83.73000373542442 tok/s
steady token average: 12.020852016ms
steady Eval(next): 10.624570008ms
steady forward graph construction: 1.357705992ms
stderr_bytes: 0
```

Capacity controls:

```text
fixed 2560 slots: 82.54488235136516 tok/s
fixed 2368 slots: 82.59760436786303 tok/s
fixed 2336 slots: 83.73000373542442 tok/s one-run, 84.23477753697784 tok/s 3-run
automatic request-sized fixed cache: 84.01009717307203 tok/s 3-run
per-layer sliding fixed cache with native overflow update: 73.05984177869179 tok/s 3-run
restored uniform request-sized fixed cache: 83.59574625080806 tok/s 3-run
dynamic paged, no fixed cache: 50.412141409798174 tok/s
fixed 2336, no shared mask: 79.62987660090852 tok/s
fixed 2336, compiled layer: 81.00297503992995 tok/s
fixed 2336, no direct greedy: 82.58079828207372 tok/s
```

The fast lane therefore needs fixed-cache graph stability, the shared fixed
mask, direct greedy, and a workload-sized fixed-cache capacity. The compiled
layer remains slower even after right-sizing the cache.

Final token-phase artefact:

`docs/runtime/2026-05-18-go-mlx-gemma4-26b-a4b-q4-packed-expert-id-final-token-phases.json`

```text
decode: 78.66136991155207 tok/s
steady token average: 12.794125648ms
steady Eval(next): 11.461327984ms
steady forward graph construction: 1.301446032ms
stderr_bytes: 0
```

A scale-hoist variant for aligned q4 groups was correct but slower at
`77.70903294390506 tok/s`, so it was reverted while keeping the packed-column
iteration.

The packed path was also rechecked with `-compiled-gemma4-layer` enabled:

`docs/runtime/2026-05-18-go-mlx-gemma4-26b-a4b-q4-packed-expert-id-compiled-layer-token-phases.json`

```text
decode: 78.78857639506562 tok/s
prefill: 1928.2622708114843 tok/s
steady token average: 12.771735744ms
steady Eval(next): 11.381450264ms
steady forward graph construction: 1.358808696ms
stderr_bytes: 0
```

That is slightly below the packed 3-run baseline (`79.1105587686013 tok/s`) and
still leaves same-prompt llama.cpp `1.1607x` faster on decode, so the compiled
layer remains a rejected probe for this lane.

The existing compiled per-layer-input tensor gate was also rechecked on the
packed path:

`docs/runtime/2026-05-18-go-mlx-gemma4-26b-a4b-q4-packed-expert-id-compiled-per-layer-inputs-token-phases.json`

```text
decode: 77.0865964024348 tok/s
prefill: 1914.738466606945 tok/s
steady token average: 13.053710288ms
steady Eval(next): 11.575552296ms
steady forward graph construction: 1.43809028ms
stderr_bytes: 0
```

It is slower than the packed baseline and leaves same-prompt llama.cpp
`1.1863x` faster on decode, so it remains off for this lane.

The existing native MLP GELU wrapper was rechecked on the packed path too:

`docs/runtime/2026-05-18-go-mlx-gemma4-26b-a4b-q4-packed-expert-id-native-mlp-token-phases.json`

```text
decode: 77.96201603724107 tok/s
prefill: 1917.671369776293 tok/s
steady token average: 12.903903664ms
steady Eval(next): 11.517494352ms
steady forward graph construction: 1.353573288ms
stderr_bytes: 0
```

It is also slower than the packed baseline and leaves same-prompt llama.cpp
`1.1730x` faster on decode.

The native-event trace below was run with `GO_MLX_TRACE_FORWARD_EVAL=1`. It
forces intermediate materialisation and is therefore attribution-only, not a
throughput result:

`docs/runtime/2026-05-18-go-mlx-gemma4-26b-a4b-q4-packed-expert-id-native-event-trace.json`

```text
generated_tokens: 16
decode: 14.365261910718765 tok/s
stderr_bytes: 0
attention: 185.826367ms, 17.52%
ffn_local_mlp: 125.883954ms, 11.87%
ffn_router: 111.062662ms, 10.47%
ffn_expert.activation_split_id_matvec: 108.760886ms, 10.25%
attention_residual: 95.194334ms, 8.98%
ffn_expert.down_weighted_sum_id_matvec: 93.448827ms, 8.81%
```

That trace supports treating the remaining llama.cpp gap as a larger
graph/kernel scheduling problem rather than another sampler-only or
single-wrapper fix.

The shared Gemma 4 31B q4 results below remain useful internal large-model
evidence, but the `mlx_lm` comparisons are archived and should not be used for
new benchmark decisions. Active external benchmark decisions use llama.cpp.

The mixed-quant loader rebuild was also rerun on the shared-31B lane:

```text
successful_runs: 3
generated_tokens: 66
visible_tokens: 66
decode_tokens_per_sec_average: 24.971269037945117
run tok/s: 25.411423243755376, 24.919505974599943, 24.582877895480028
prefill_tokens_per_sec_average: 152.57561118762987
peak_memory_bytes: 19076060876
stderr_bytes: 0
```

JSON output is saved as
`docs/runtime/2026-05-17-go-mlx-gemma4-31b-q4-mixed-quant-loader-3run-parity.json`.
This is a small improvement over the prior `24.663669410625896 tok/s`
three-run sample, but it remains internal evidence only under the llama.cpp
benchmark policy.

The short no-thinking prompt only generates around 22-23 tokens, so a sustained
128-token diagnostic prompt was also run:

```bash
env MLX_METALLIB_PATH=/Users/snider/Code/core/go-mlx/dist/lib/mlx.metallib /Users/snider/Code/core/go-mlx/bin/lthn-mlx driver-profile -json -include-output=false -context 4096 -prompt "Write exactly 200 comma-separated integers, starting at 1." -max-tokens 128 -runs 3 /Users/snider/.cache/huggingface/hub/models--mlx-community--gemma-4-31b-it-4bit/snapshots/dcb78c3f5d6becacbfce71cd4851ad98c4f08a05
```

```text
successful_runs: 3
generated_tokens: 384
visible_tokens: 384
decode_tokens_per_sec_average: 23.086428954337055
run tok/s: 23.1032323325884, 22.935095047267012, 23.22095948315575
prefill_tokens_per_sec_average: 166.37095912885252
peak_memory_bytes: 19270082392
stderr_bytes: 0
```

JSON output is saved as
`docs/runtime/2026-05-17-go-mlx-gemma4-31b-q4-longdecode-3run-parity.json`.

Archived `mlx_lm.generate` no-thinking command:

```bash
env MLX_METALLIB_PATH=/Users/snider/Code/core/go-mlx/dist/lib/mlx.metallib /private/tmp/go-mlx-mlx-lm-venv/bin/python -m mlx_lm.generate --model /Users/snider/.cache/huggingface/hub/models--mlx-community--gemma-4-31b-it-4bit/snapshots/dcb78c3f5d6becacbfce71cd4851ad98c4f08a05 --prompt "Write exactly 200 comma-separated integers, starting at 1." --max-tokens 128 --temp 0 --chat-template-config '{"enable_thinking": false}' --verbose True
```

reports:

```text
Prompt: 29 tokens, 89.253 tokens-per-sec
Generation: 128 tokens, 34.893 tokens-per-sec
Peak memory: 17.560 GB
```

Full output is saved as
`docs/runtime/2026-05-17-mlx-lm-gemma4-31b-q4-longdecode-no-thinking-parity.txt`.
This is retained only to explain prior work; it is no longer the active
benchmark target.

The same rebuilt binary was also used for a gated native MLP rerun on the
shared-31B diagnostic lane because the native phase trace points at FFN work:

```text
successful_runs: 3
generated_tokens: 66
visible_tokens: 66
decode_tokens_per_sec_average: 24.7143167044012
prefill_tokens_per_sec_average: 151.59127450834528
peak_memory_bytes: 19089528524
stderr_bytes: 0
```

JSON output is saved as
`docs/runtime/2026-05-17-go-mlx-gemma4-31b-q4-native-mlp-mixed-quant-parity.json`.
This regresses the `24.971269037945117 tok/s` mixed-quant default, so the
native MLP gate remains disabled.

The later fixed-cache attention pass removed the concrete 512-wide SDPA kernel
blocker by applying `patches/mlx-sdpa-vector-512.patch`, rebuilding
`dist/lib/mlx.metallib`, and rerunning the shared-31B longdecode prompt with
`GO_MLX_ENABLE_FIXED_WIDE_SDPA_ATTENTION=1`:

```text
go-mlx SHA-256: 1ba7ea769df0b48f39ec6f0581fa4b8bf0931b1a8944e7ad2e7ea911d43b6f49
successful_runs: 3
generated_tokens: 384
visible_tokens: 384
decode_tokens_per_sec_average: 24.70397262176645
run tok/s: 24.54956052082555, 24.799885029282997, 24.762472315190802
prefill_tokens_per_sec_average: 138.49735481596804
peak_memory_bytes: 19331029334
stderr_bytes: 0
```

JSON output is saved as
`docs/runtime/2026-05-17-go-mlx-gemma4-31b-q4-fixed-cache160-native-sdpa512-longdecode.json`.
This changes the diagnosis: 512-wide SDPA support is no longer the primary
blocker. The patched attention path is clean but does not beat the guarded
fallback (`24.94401176949734 tok/s`), so the remaining 31B gap is still the
larger one-token native eval/materialisation boundary that llama.cpp avoids with
stable graph reuse and host-fed decode inputs.

Two paired follow-ups narrow that further. `GO_MLX_ENABLE_FIXED_GEMMA4_SHARED_MASK=1`
host-feeds one fixed-cache attention mask per decode token. It records
`24.904493509253538 tok/s` without the SDPA512 gate and
`24.767920780634018 tok/s` with the SDPA512 gate, both with three full
128-token runs and empty stderr. `GO_MLX_ENABLE_DIRECT_GREEDY_TOKEN=1` on the
same sustained 31B longdecode prompt records only `23.2767195467288 tok/s`, so
skipping final logits materialisation is also not the missing boundary on this
model.

## Gemma 4 Assistant MTP Diagnostic

The 2026-05-18 speculative-decode follow-up keeps MTP separate from raw
target-only parity. Homebrew llama.cpp build `8990`, commit `660b1b4bd`, rejects
`--spec-type draft-mtp`, and upstream master at `/private/tmp/llama.cpp`,
commit `1a68ec9`, exposes the flag but cannot load `gemma4_assistant`.

Unmerged PR `ggml-org/llama.cpp#23211`, cloned to
`/private/tmp/llama.cpp-pr23211`, does load the local 26B assistant GGUF:

```text
target: unsloth/gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf
assistant: AtomicChat/gemma-4-26B-A4B-it-assistant-GGUF/gemma-4-26B-A4B-it-assistant.Q4_K_M.gguf
assistant sha: 171ecca181ec00ed6ffacb573195aa7c644bbdc6
```

On the README prompt with 128 generated tokens, PR `llama-cli` target-only
records `2063.7 tok/s` prompt and `83.4 tok/s` generation. The same PR CLI with
`--spec-type draft-mtp --spec-draft-n-max 2` records `1615.7 tok/s` prompt and
`100.2 tok/s` generation. The server path reports `1562.0125388366318 tok/s`
prompt, `93.76822253543413 tok/s` generation, and `75/101` draft tokens
accepted. Full notes and artefacts are in
`docs/runtime/2026-05-18-gemma4-mtp-speculative-decode.md`.

<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# llama.cpp Prefill Comparison, 2026-05-17

This note records the local Apple M3 Ultra comparison requested after the
Gemma 4 E2B row-gather fix. It includes prefill and decode.

## Caveat

The closest local llama.cpp model is not bit-for-bit identical to the go-mlx
model:

| Runtime | Model | Format | Quantisation |
| --- | --- | --- | --- |
| go-mlx | `mlx-community/gemma-4-26b-a4b-it-4bit` | MLX safetensors | q4, with per-tensor q8 overrides |
| llama.cpp baseline | `unsloth/gemma-4-26B-A4B-it-GGUF` | GGUF | `Q8_0` via `Q8_K_XL` |
| llama.cpp q4 follow-up | `unsloth/gemma-4-26B-A4B-it-GGUF` | GGUF | `Q4_K_M` |

All rows are Gemma 4 26B A4B on the same M3 Ultra. The `Q4_K_M` follow-up is
the cleaner q4-family llama.cpp comparison, but it is still not bit-for-bit
identical to the MLX safetensors pack.

## llama.cpp

Binary:

```text
llama.cpp build 8990, commit 660b1b4bd
backends: BLAS, MTL
gpu: Apple M3 Ultra
flash_attn: true
n_gpu_layers: 99
KV cache: f16 K, f16 V
```

`Q8_K_XL` short prefill plus decode command:

```bash
llama-bench -m /Users/snider/.cache/huggingface/hub/models--unsloth--gemma-4-26B-A4B-it-GGUF/snapshots/b68961b3c96e42475123a39fe3f8aa149163cf8b/gemma-4-26B-A4B-it-UD-Q8_K_XL.gguf -p 29 -n 128 -r 3 -ngl 99 -fa 1 -o json
```

Output:

`docs/runtime/2026-05-17-llamacpp-gemma4-26b-a4b-q8-p29-g128-bench.json`

```text
pp29: 375.334002 tok/s, samples [376.739, 375.478, 373.785]
tg128: 87.688525 tok/s, samples [83.6194, 90.3844, 89.0618]
```

`Q8_K_XL` long prefill plus decode command:

```bash
llama-bench -m /Users/snider/.cache/huggingface/hub/models--unsloth--gemma-4-26B-A4B-it-GGUF/snapshots/b68961b3c96e42475123a39fe3f8aa149163cf8b/gemma-4-26B-A4B-it-UD-Q8_K_XL.gguf -p 2048 -n 128 -r 3 -ngl 99 -fa 1 -o json
```

Output:

`docs/runtime/2026-05-17-llamacpp-gemma4-26b-a4b-q8-p2048-g128-bench.json`

```text
pp2048: 2231.973259 tok/s, samples [2225.00, 2238.75, 2232.17]
tg128: 90.996302 tok/s, samples [90.8843, 90.9639, 91.1407]
```

`Q4_K_M` short prefill plus decode command:

```bash
llama-bench -m /Users/snider/.cache/huggingface/hub/models--unsloth--gemma-4-26B-A4B-it-GGUF/snapshots/3365c68df1a83799b846d05324ebfadbb8cc70b3/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf -p 29 -n 128 -r 3 -ngl 99 -fa 1 -o json
```

Output:

`docs/runtime/2026-05-17-llamacpp-gemma4-26b-a4b-q4-k-m-p29-g128-bench.json`

```text
pp29: 468.942791 tok/s, samples [467.316, 466.954, 472.558]
tg128: 89.000726 tok/s, samples [83.9378, 89.8643, 93.2001]
```

`Q4_K_M` long prefill plus decode command:

```bash
llama-bench -m /Users/snider/.cache/huggingface/hub/models--unsloth--gemma-4-26B-A4B-it-GGUF/snapshots/3365c68df1a83799b846d05324ebfadbb8cc70b3/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf -p 2048 -n 128 -r 3 -ngl 99 -fa 1 -o json
```

Output:

`docs/runtime/2026-05-17-llamacpp-gemma4-26b-a4b-q4-k-m-p2048-g128-bench.json`

```text
pp2048: 2184.109033 tok/s, samples [2177.44, 2189.5, 2185.39]
tg128: 92.624334 tok/s, samples [93.4653, 92.9257, 91.482]
```

`Q4_K_M` same-prompt-length prefill plus decode command for the go-mlx
`README.md` prompt-file lane:

```bash
llama-bench -m /Users/snider/.cache/huggingface/hub/models--unsloth--gemma-4-26B-A4B-it-GGUF/snapshots/3365c68df1a83799b846d05324ebfadbb8cc70b3/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf -p 2204 -n 128 -r 3 -ngl 99 -fa 1 -o json
```

Output:

`docs/runtime/2026-05-17-llamacpp-gemma4-26b-a4b-q4-k-m-p2204-g128-bench.json`

```text
pp2204: 2109.335561 tok/s, samples [2109.38, 2113.35, 2105.28]
tg128: 91.451031 tok/s, samples [91.2108, 91.3161, 91.8262]
```

## go-mlx

The first go-mlx 26B q4 run exposed a loader bug before it produced a
benchmark number: the model has q8 overrides for the dense MLP/router
projections under a default q4 quantisation block. The Gemma 4 loader now
infers the effective bit width from the packed weight and scale shapes before
constructing quantized linears. Focused coverage:

```bash
cd /Users/snider/Code/core/go-mlx/go
env GOCACHE=/private/tmp/codex-go-mlx-cache go test ./internal/metal -run 'TestGemma4_(Linear_Infers8BitOverrideFromScales|SwitchLinear_Preserves4BitWhenShapesMatchDefault|QuantPredicate_RouterForces8Bit|Linear_QuantizedWithoutConfig|SwitchLinear_QuantizedWithoutConfig)_Good' -count=1
```

Result:

```text
ok  	dappco.re/go/mlx/internal/metal	0.477s
```

Rebuilt binary:

```text
bin/lthn-mlx SHA-256: c1034cf834b9c40d65c0e9bcf2652f5c2232965ef1715188c89fb5eff8abf141
```

Short prefill plus full decode command:

```bash
env MLX_METALLIB_PATH=/Users/snider/Code/core/go-mlx/dist/lib/mlx.metallib /Users/snider/Code/core/go-mlx/bin/lthn-mlx driver-profile -json -include-output=false -context 4096 -prompt "Write exactly 200 comma-separated integers, starting at 1." -max-tokens 128 -runs 3 /Users/snider/.cache/huggingface/hub/models--mlx-community--gemma-4-26b-a4b-it-4bit/snapshots/695690b33533b1f8b0395c1d6b4f00dc411353ef
```

Output:

`docs/runtime/2026-05-17-go-mlx-gemma4-26b-a4b-q4-llamacpp-comparison-longdecode.json`

```text
prompt_tokens: 29
prefill: 447.6882783215051 tok/s, samples [407.4314083955457, 466.5826882184106, 469.05073835055885]
decode: 55.96521969803896 tok/s, samples [55.930446120682824, 56.058854506076614, 55.90635846735742]
generated_tokens: [128, 128, 128]
peak_memory_bytes: 16284290208
```

Long prefill command:

```bash
prompt=""; for i in {1..2048}; do prompt="${prompt}state "; done
env MLX_METALLIB_PATH=/Users/snider/Code/core/go-mlx/dist/lib/mlx.metallib /Users/snider/Code/core/go-mlx/bin/lthn-mlx driver-profile -json -include-output=false -context 4096 -prompt "$prompt" -max-tokens 1 -runs 1 /Users/snider/.cache/huggingface/hub/models--mlx-community--gemma-4-26b-a4b-it-4bit/snapshots/695690b33533b1f8b0395c1d6b4f00dc411353ef
```

Output:

`docs/runtime/2026-05-17-go-mlx-gemma4-26b-a4b-q4-longprefill-one-run-llamacpp-comparison.json`

```text
prompt_tokens: 2061
prefill: 864.6062359771336 tok/s
peak_memory_bytes: 20480346316
```

The three-run long-prefill file
`docs/runtime/2026-05-17-go-mlx-gemma4-26b-a4b-q4-longprefill-llamacpp-comparison.json`
is not used for average prefill because runs 2 and 3 hit the prompt cache.
The clean no-reuse long-prefill number is the one-run value above.

### Decode-only fused expert gate/up follow-up

A follow-up read of llama.cpp found that Gemma MoE keeps the expert
`gate_up` projection fused when the tensor exists, then splits the result into
gate and up halves. go-mlx had sanitised that source tensor into separate
`gate_proj` and `up_proj` weights and executed both expert-indexed projections.

go-mlx now retains `experts.switch_glu.gate_up_proj` and uses the fused
projection for single-token decode only. The first ungated attempt regressed
long prefill, so prefill deliberately stays on the split fallback path.

Rebuilt binary:

```text
bin/lthn-mlx SHA-256: 085e204e17aa0f4f1fe614efa090f8779832129de5c377bf8b570902b3172f7b
```

Short prefill plus full decode output:

`docs/runtime/2026-05-17-go-mlx-gemma4-26b-a4b-q4-fused-gate-up-decode-only-llamacpp-comparison-longdecode.json`

```text
prompt_tokens: 29
prefill: 449.18863738146 tok/s, samples [413.5639447651411, 466.3272865317299, 467.67468084750914]
decode: 56.45505318098333 tok/s, samples [56.42639515728892, 56.50928981909404, 56.42947456656704]
generated_tokens: [128, 128, 128]
peak_memory_bytes: 16126451615
```

Clean no-reuse long prefill output:

`docs/runtime/2026-05-17-go-mlx-gemma4-26b-a4b-q4-fused-gate-up-decode-only-longprefill-one-run-llamacpp-comparison.json`

```text
prompt_tokens: 2061
prefill: 862.5952429295362 tok/s
peak_memory_bytes: 19811354828
```

The change improves decode by `+0.4898334829443698 tok/s` over the previous
go-mlx comparison run. Long prefill is effectively neutral and remains far
behind llama.cpp.

### Automatic long-prompt last-token prefill follow-up

The next prefill-specific probe targeted another avoidable double-work pattern:
the default prefill path materialised full `[sequence,vocab]` logits and then
sliced the last row, even though generation consumes only the last-token logits.
go-mlx now automatically uses the existing `ForwardLastTokenLogits` path for
prompt chunks at or above 512 tokens. Short prompts stay on the full-logits
path unless `GO_MLX_ENABLE_LAST_LOGITS_PREFILL=1` explicitly forces the old
experiment.

Rebuilt binary:

```text
bin/lthn-mlx SHA-256: dd212338c1864b6acb630bb5f534986432d1c189d17e100ae8ab3a3ee230a352
```

Short prefill plus full decode rerun:

`docs/runtime/2026-05-17-go-mlx-gemma4-26b-a4b-q4-auto-last-logits-llamacpp-comparison-longdecode-rerun2.json`

```text
prompt_tokens: 29
prefill: 443.8939306138111 tok/s, samples [402.6365753676662, 466.478868708316, 462.5663477654512]
decode: 56.220244342267904 tok/s, samples [56.138136941728334, 56.25724605690424, 56.26535002817114]
generated_tokens: [128, 128, 128]
peak_memory_bytes: 16126451711
```

Clean no-reuse long prefill rerun:

`docs/runtime/2026-05-17-go-mlx-gemma4-26b-a4b-q4-auto-last-logits-longprefill-one-run-llamacpp-comparison.json`

```text
prompt_tokens: 2061
prefill: 903.0290085147915 tok/s
peak_memory_bytes: 17974597848
```

The long-prefill path improves by `+40.43376558525529 tok/s`
(`+4.687455201808732%`) versus the previous default run. A tiny-tail chunk
coalescing probe was also tried because this prompt splits as `2048 + 13`.
That was negative: one 2061-token prefill pass recorded only
`862.4738054025554 tok/s`, so the code path was reverted and the two-chunk
planner shape remains in place.

A llama.cpp-inspired shared-KV trim probe was also tested. It collapsed the
long last-logits prefill path to the final token after the last KV-owning
Gemma 4 layer, while preserving the final RoPE position and the sliding shared
KV window. The one-run long prefill rose only to `911.1355151113232 tok/s`,
and the 128-token decode check fell to `53.616341210113625 tok/s`, so the
source change was reverted. The rejected diagnostic artefacts are:
`docs/runtime/2026-05-17-go-mlx-gemma4-26b-a4b-q4-shared-kv-last-token-trim-longprefill-one-run-llamacpp-comparison.json`
and
`docs/runtime/2026-05-17-go-mlx-gemma4-26b-a4b-q4-shared-kv-last-token-trim-llamacpp-comparison-longdecode.json`.

Two fixed-cache compiled-layer probes were then run on the active 26B
Q4_K_M comparison lane. Both were negative against the accepted default:

```text
full-context fixed-cache compiled layer:
decode: 48.211754489053696 tok/s
prefill: 402.4998847052011 tok/s
artefact: docs/runtime/2026-05-17-go-mlx-gemma4-26b-a4b-q4-fixed-cache-compiled-layer-llamacpp-comparison-longdecode.json

fixed-cache compiled layer, 160 slots:
decode: 53.69079065280556 tok/s
prefill: 433.71986471660057 tok/s
artefact: docs/runtime/2026-05-17-go-mlx-gemma4-26b-a4b-q4-fixed-cache160-compiled-layer-llamacpp-comparison-longdecode.json
```

Both stderr files are empty. The fixed 160-slot path is closer, but still
below the accepted `56.220244342267904 tok/s` decode control, so this is not
the llama.cpp parity fix.

The follow-up traces point at evaluated Metal graph work, not Go orchestration.
With ordinary token-phase tracing on the accepted default path, a 128-token
single run records `53.24884702642772 tok/s` under trace overhead. Excluding
warmup and the final token, 125 steady samples average `18.887ms/token` total,
of which `17.432ms` is `sample_eval_duration` and only `1.414ms` is forward
construction. The trace is saved as
`docs/runtime/2026-05-17-go-mlx-gemma4-26b-a4b-q4-default-token-phase-trace-llamacpp-comparison.json`.

The native phase trace is intentionally slower because it forces per-layer
boundaries. It records 120 native events per token on the 30-layer 26B model.
Across 29 steady decode samples, the forced boundary totals are roughly
`20.082ms/token` in FFN, `12.393ms/token` in attention, `7.990ms/token` in
layer output, and `7.398ms/token` in attention residual. That diagnostic is
saved as
`docs/runtime/2026-05-17-go-mlx-gemma4-26b-a4b-q4-native-phase-trace-llamacpp-comparison.json`.

A native fused-experts bridge was then tried against that FFN/MoE suspicion.
It fused `gate_up` gather, GELU, down gather, expert weighting, and top-k sum
behind an opt-in native wrapper, but the real 26B A4B q4 run regressed:
`53.08901433576139 tok/s` decode and `431.27066684929787 tok/s` short
prefill, with three full 128-token runs and empty stderr. The source change was
reverted. The rejected diagnostic is saved as
`docs/runtime/2026-05-17-go-mlx-gemma4-26b-a4b-q4-native-fused-experts-llamacpp-comparison-longdecode.json`.

The follow-up FFN split trace keeps the same llama.cpp-only comparison lane and
adds trace-only sub-boundaries inside the MoE branch. It is diagnostic, not a
throughput result: one 32-token run records `14.452280580872943 tok/s` under
trace overhead. Across 29 steady decode samples it records 270 native events per
token. The largest totals are `ffn_experts` at `13.736ms/token`, attention at
`10.614ms/token`, `ffn_local_mlp` at `8.354ms/token`, and `ffn_router` at
`7.560ms/token`. The trace is saved as
`docs/runtime/2026-05-17-go-mlx-gemma4-26b-a4b-q4-native-phase-ffn-split-trace-llamacpp-comparison.json`.

The next useful implementation target is therefore a broader llama.cpp-shaped
one-token block or a lower-level quantized MoE kernel, not another wrapper
around the same MLX gather graph.

### MLX GatherQMM versus llama.cpp `mul_mat_id`

The follow-up static read explains why a small MLX flag change is unlikely to
close the decode gap. go-mlx routes expert projections through `SwitchLinear`,
which calls `GatherQMM(..., rhs_indices=topKIndices, sorted=false)`. MLX's
Metal `GatherQMM::eval_gpu` only enters the specialised `gather_qmm_rhs` path
when the RHS indices are globally sorted and there is enough batched work
(`M == 1`, `B >= 16`, and `B / E >= 4`). Single-token 26B decode presents top-k
8 work over 128 experts, so it cannot meet that batched RHS path. It falls back
to the vector gather path.

llama.cpp uses a different primitive boundary. Gemma MoE lowers to
`GGML_OP_MUL_MAT_ID`; Metal then chooses a dedicated `kernel_mul_mv_id` path for
small token counts and a `kernel_mul_mm_id` plus expert-ID map for larger
batches. The kernels are specialised for the quant type and `n_expert_used`,
including the top-k 8 case. That is the implementation shape go-mlx still
needs to copy for parity. go-mlx now has trace-only expert subevents under
`GO_MLX_TRACE_FORWARD_EVAL=1` so the next Metal-available run can split
`ffn_experts` into gate/up, activation, down, weighting, and sum buckets.
The first code-side scaffold for that shape is
`go/internal/metal/expert_id_matvec.go`: an internal q2/q4/q8
`quantizedExpertIDMatVec` helper that consumes MLX affine-packed expert rows
and expert ids, then matches a CPU q4 reference on small and multi-pack tensors.
One SIMD group now reduces each routed output row. Gemma 4 can route through it
only with `GO_MLX_ENABLE_EXPERT_ID_MATVEC=1`, and the unit regression compares
that opt-in path against the existing MLX `GatherQMM` result. The custom kernel
handle is cached per shape so repeated decode calls do not rebuild it. The
down-projection side now uses a weighted expert-ID matvec-sum kernel, folding
route weighting and top-k summation into the down matvec instead of leaving
them as separate MLX nodes. This is not benchmark evidence or a default Gemma 4
runtime path.

The first full 26B A4B q4 env-gated probe did not produce a throughput number:
native model load failed with `no usable Metal device available` before
generation. A follow-up added a `driver-profile -expert-id-matvec` diagnostic
flag so the gate can be enabled without a second environment variable, while
still recording `runtime_gates.GO_MLX_ENABLE_EXPERT_ID_MATVEC=1`. The compact
three-run profile is valid but negative: `55.98273536629838 tok/s` decode and
`449.436848070603 tok/s` short prefill. It trails the accepted go-mlx decode
control by `0.237509 tok/s`, and llama.cpp `Q4_K_M` is still `1.5898x` faster
on decode. The diagnostic artefacts are:
`docs/runtime/2026-05-17-go-mlx-gemma4-26b-a4b-q4-expert-id-matvec-gated-llamacpp-comparison-longdecode.json`
and
`docs/runtime/2026-05-17-go-mlx-gemma4-26b-a4b-q4-expert-id-matvec-flag-llamacpp-comparison-longdecode.json`.

A narrower fused-activation variant then moved `GELU(gate) * up` into the
custom expert-ID gate_up kernel behind
`driver-profile -expert-id-fused-activation`, which also records
`runtime_gates.GO_MLX_ENABLE_EXPERT_ID_FUSED_ACTIVATION=1`. Same-binary
controls show the effect is noise-scale, not a parity fix:

```text
default control: 56.21477992583666 tok/s decode
expert-ID matvec: 56.06328243808281 tok/s decode
expert-ID fused activation: 56.295534088943356 tok/s decode
```

The fused variant is only `+0.080754 tok/s` (`+0.14%`) over the same-binary
default control, while llama.cpp `Q4_K_M` remains `1.5809x` faster. The
diagnostic JSON is saved as
`docs/runtime/2026-05-17-go-mlx-gemma4-26b-a4b-q4-expert-id-fused-activation-llamacpp-comparison-longdecode.json`.

### Sorted expert prefill follow-up

The first change that lands on the large-prefill gap is the MLX sorted RHS
path. `driver-profile` now accepts `-prompt-file` so long-prompt benchmark
inputs do not need shell-generated prompt arguments, and
`-sorted-expert-prefill` enables `GO_MLX_ENABLE_SORTED_EXPERT_PREFILL=1`
without a second environment variable. The implementation sorts flattened
Gemma 4 prefill routes by expert id, runs split gate/up/down `GatherQMM` calls
with `sorted=true`, then restores route order before top-k weighting and sum.
It is prefill-only; single-token decode cannot satisfy MLX's batched RHS
condition.

Rebuilt binary:

```text
bin/lthn-mlx SHA-256: 1eea3598b6265d5bf8326e00873ad6fd13877f471b778f739fed9213a3d3c286
```

Same-binary sequential controls used `README.md` as a prompt file, which
tokenises to `2204` prompt tokens with chat templating.

Default control:

`docs/runtime/2026-05-17-go-mlx-gemma4-26b-a4b-q4-readme-default-llamacpp-comparison-longdecode.json`

```text
prompt_tokens: 2204
prefill: 914.0299819202297 tok/s
decode: 31.048941804155767 tok/s
peak_memory_bytes: 17974597848
```

Sorted expert prefill:

`docs/runtime/2026-05-17-go-mlx-gemma4-26b-a4b-q4-sorted-expert-prefill-readme-llamacpp-comparison-longdecode.json`

```text
prompt_tokens: 2204
prefill: 1914.0303789361128 tok/s
decode: 31.508051014734626 tok/s
peak_memory_bytes: 18306419992
```

That is a `2.0940x` prefill speedup over the default control. Against the
existing llama.cpp `Q4_K_M` `pp2048` result (`2184.109033 tok/s`), go-mlx is
now at `87.6%` of llama.cpp prefill throughput on this long-prompt lane,
leaving a `1.141x` prefill gap instead of the previous `2.4x` class gap.

### Multi-page decode fast-SDPA concat follow-up

The sorted prefill run still decoded slowly because the 2204-token prompt
spans more than one paged KV block. The default long-context decode path used
`ScaledDotProductAttentionPaged`, a page-by-page softmax written out of MLX
ops. `driver-profile -paged-decode-fast-concat` enables
`GO_MLX_ENABLE_PAGED_DECODE_FAST_CONCAT=1`: for multi-page single-token decode
it concatenates the visible K/V pages and uses MLX fast SDPA, matching the
one-page short-context attention primitive.

Sorted prefill plus paged fast concat:

`docs/runtime/2026-05-17-go-mlx-gemma4-26b-a4b-q4-sorted-prefill-paged-fast-concat-readme-llamacpp-comparison-longdecode.json`

```text
prompt_tokens: 2204
prefill: 1909.1904478108413 tok/s
decode: 42.372384580120396 tok/s
peak_memory_bytes: 18306419992
```

This is a `1.3448x` decode speedup over the same-binary sorted-prefill-only
control (`31.508051014734626 tok/s`). llama.cpp `Q4_K_M` `tg128` at `p2048`
is still `92.624334 tok/s`, so the remaining long-context decode gap is
`2.186x`. Prefill remains close: the fast-concat run is `87.4%` of the
llama.cpp `pp2048` prefill result.

### Fixed-cache compiled decode follow-up

The next llama.cpp-only comparison probe moved the existing fixed-cache and
compiled Gemma 4 decode diagnostics onto `driver-profile` CLI runtime gates:
`-fixed-gemma4-cache`, `-fixed-gemma4-shared-mask`, and
`-compiled-gemma4-layer`. The run keeps the same README prompt-file workload
and uses `-cache-mode paged` so the fixed-capacity Gemma 4 cache path owns the
decode cache shape.

Sorted prefill plus fixed-cache compiled decode:

`docs/runtime/2026-05-17-go-mlx-gemma4-26b-a4b-q4-sorted-prefill-fixed-compiled-readme-llamacpp-comparison-longdecode.json`

```text
prompt_tokens: 2204
prefill: 1876.6924105183755 tok/s
decode: 48.93511098804883 tok/s
peak_memory_bytes: 19212389664
```

This is a `1.5531x` decode speedup over sorted-prefill-only and a `1.1549x`
speedup over the paged fast-concat decode probe. It is still not parity:
llama.cpp `Q4_K_M` `tg128` at `p2048` is `92.624334 tok/s`, leaving a
`1.8928x` long-context decode gap.

Adding `driver-profile -direct-greedy-token` to the same fixed-cache compiled
lane records a 3-run sample:

`docs/runtime/2026-05-17-go-mlx-gemma4-26b-a4b-q4-sorted-prefill-fixed-compiled-direct-greedy-3run-readme-llamacpp-comparison-longdecode.json`

```text
prompt_tokens: 2204
prefill: 1908.4658285603446 tok/s
decode: 49.75515922842408 tok/s
peak_memory_bytes: 19212389680
```

That is only a `1.0168x` decode speedup over fixed-cache compiled decode, but
llama.cpp `Q4_K_M` `tg128` at `p2048` is still `1.8616x` faster.

The compiled Gemma 4 decode graph was also extended to cover MoE layers instead
of only dense MLP layers. A focused tiny-MoE regression passes, but the full
26B A4B profile stays in the same band: one run records
`49.57330167871466 tok/s`, and adding the expert-ID fused activation gate
averages `49.705483987003994 tok/s` over three runs. That is below the
direct-greedy 3-run sample, so MLX-compiling the current MoE graph is not the
missing llama.cpp boundary.

The direct expert-ID path was then measured without `-compiled-gemma4-layer`, so
single-token decode can take the custom expert-ID fused activation branch while
prefill still uses sorted expert routing:

`docs/runtime/2026-05-17-go-mlx-gemma4-26b-a4b-q4-sorted-prefill-expert-id-fused-direct-greedy-3run-readme-llamacpp-comparison-longdecode.json`

```text
prompt_tokens: 2204
prefill: 1915.3373741969128 tok/s
decode: 49.973204322219345 tok/s
peak_memory_bytes: 19212389680
```

This is the current best go-mlx long-context decode sample, but the gain is only
`+0.44%` over the fixed-cache compiled direct-greedy sample. llama.cpp `Q4_K_M`
`tg128` at `p2048` is still `1.8535x` faster. The same-prompt-length p2204
llama.cpp row is `1.1013x` faster on prefill and `1.8300x` faster on decode.
A code-side follow-up also keeps the older C++ `-native-gemma4-layer` gate
dense-only; its ABI does not carry MoE router/expert tensors, while the Go/MLX
compiled graph does.

The next cache-shape diagnostic tested the tempting hypothesis that the fixed
Gemma 4 lane should preserve the model's 1024-token sliding-window cache bound.
That required fixing `FixedKVCache` overflow semantics so multi-token prompt
chunks and single-token decode overflows survive the detach boundary. The
diagnostic completed, but it is not the active benchmark lane:

`docs/runtime/2026-05-17-go-mlx-gemma4-26b-a4b-q4-sliding-cache-bound-direct-greedy-3run-readme-llamacpp-comparison-longdecode.json`

```text
prompt_tokens: 2204
prefill: 1806.8318924630082 tok/s
decode: 40.76006207167587 tok/s
peak_memory_bytes: 71228950132
stderr_bytes: 0
```

The read is negative: bounding the fixed-cache sliding layers by itself
increases memory pressure and loses the fixed-shape decode advantage. The
default fixed-cache lane therefore keeps uniform context-sized fixed caches,
while non-fixed paged replacement preserves inherited rotating-cache bounds.
The restored current-code run is:

`docs/runtime/2026-05-17-go-mlx-gemma4-26b-a4b-q4-fixed-uniform-cache-direct-greedy-3run-readme-llamacpp-comparison-longdecode.json`

```text
prompt_tokens: 2204
prefill: 1923.322483219664 tok/s
decode: 49.71518402860789 tok/s
peak_memory_bytes: 19212389680
stderr_bytes: 0
bin/lthn-mlx SHA-256: 5a4081baa3c2cd9f492d333b01c04328f60ae2fe15d19015f35ddf68f2661e38
```

Against the same-prompt-length llama.cpp `Q4_K_M` row, that leaves a
`1.0967x` prefill gap and a `1.8395x` decode gap.

### Router residual source-parity follow-up

A follow-up read of llama.cpp's Gemma 4 graph found one remaining routing
shape mismatch. llama.cpp computes MoE router logits from the post-attention
residual stream, while the expert branch still consumes the pre-FFN2-normalised
tensor. go-mlx was routing from the pre-FFN2-normalised tensor too, so the router
input did not match the llama.cpp graph. The Go graph and compiled decode graph
now route from the attention residual while keeping the expert input unchanged.

The same README prompt-file lane now records:

`docs/runtime/2026-05-17-go-mlx-gemma4-26b-a4b-q4-router-residual-fixed-uniform-direct-greedy-3run-readme-llamacpp-comparison-longdecode.json`

```text
prompt_tokens: 2204
prefill: 1933.6368792628773 tok/s
decode: 50.23367760579547 tok/s
peak_memory_bytes: 19212389680
stderr_bytes: 0
```

Against same-prompt-length llama.cpp `Q4_K_M`, that leaves a `1.0909x` prefill
gap and a `1.8205x` decode gap.

A llama.cpp-inspired two-output down-projection matvec was also tested as a
kernel-shape diagnostic and rejected. It completed with empty stderr but
regressed to `1732.6641621430529 tok/s` prefill and `48.4963971321882 tok/s`
decode:

`docs/runtime/2026-05-17-go-mlx-gemma4-26b-a4b-q4-expert-down-two-col-fixed-uniform-direct-greedy-3run-readme-llamacpp-comparison-longdecode.json`

### Active split expert-ID follow-up

The next trace found that the active MLX safetensors do not expose a fused
`experts.switch_glu.gate_up_proj` tensor. They store split `gate_proj` and
`up_proj` expert tensors, and the q4 sidecar scales/biases are BF16. That meant
the earlier fused-`gate_up` expert-ID gate was falling back on this 26B A4B q4
pack instead of timing the intended custom kernel.

The split expert-ID path now accepts BF16/F16/F32 sidecars and supports both
split gate/up tensors and one shared hidden row for multiple top-k expert IDs.
The phase trace confirms active `activation_split_id_matvec` and
`down_weighted_sum_id_matvec` events in every MoE layer:

`docs/runtime/2026-05-17-go-mlx-gemma4-26b-a4b-q4-split-fused-expert-id-shared-input-native-phase-trace.json`

```text
stderr_bytes: 0
native phases: activation_split_id_matvec, down_weighted_sum_id_matvec
```

Intermediate 3-run evidence:

```text
split expert-ID, separate gate/up activation:
  prefill: 1939.2172632050945 tok/s
  decode: 62.52025013199337 tok/s
  llama.cpp decode gap: 1.4628x

split expert-ID, fused activation:
  prefill: 1941.0884632916652 tok/s
  decode: 68.22675114228564 tok/s
  llama.cpp decode gap: 1.3404x
```

Current shared-input split fused-activation output:

`docs/runtime/2026-05-17-go-mlx-gemma4-26b-a4b-q4-split-fused-expert-id-shared-input-fixed-uniform-direct-greedy-3run-readme-llamacpp-comparison-longdecode.json`

```text
prompt_tokens: 2204
prefill: 1923.9974775252285 tok/s, samples [1882.4987804692028, 1943.3438983553547, 1946.1497537511284]
decode: 70.54498924012704 tok/s, samples [69.91341816877653, 70.25276863828591, 71.46878091331867]
generated_tokens: [128, 128, 128]
peak_memory_bytes: 19212389664
active_memory_bytes: 17457260720
stderr_bytes: 0
/private/tmp/lthn-mlx-split-expert-id SHA-256: dd9dfe917d073c4006b74e7ae7a42fbdefe96f3f74533607e46e5d7785923b1f
```

Against same-prompt-length llama.cpp `Q4_K_M`, that leaves a `1.0963x` prefill
gap and a `1.2964x` decode gap. It is a material improvement over the
router-residual lane (`1.4043x` decode speedup), but it is still below both the
`100 tok/s` floor and llama.cpp's `91.451031 tok/s`.

The matching token-phase profile, without native event materialisation, is:

`docs/runtime/2026-05-17-go-mlx-gemma4-26b-a4b-q4-split-fused-expert-id-shared-input-token-phases.json`

```text
decode: 71.59452329863376 tok/s
steady token average: 14.05959232ms
steady Eval(next): 12.724946032ms
steady next-forward graph construction: 1.297721312ms
stderr_bytes: 0
```

Re-enabling the older native dense MLP GELU wrapper on this same lane is
neutral-to-negative:

`docs/runtime/2026-05-17-go-mlx-gemma4-26b-a4b-q4-split-fused-shared-input-native-mlp-probe.json`

```text
decode: 71.44678366026884 tok/s
prefill: 1927.4283286475602 tok/s
stderr_bytes: 0
```

That points the next optimisation away from another standalone MLP wrapper and
back toward the larger eval/materialisation boundary, especially final
projection/greedy argmax fusion or broader stable graph reuse.

### Packed-column expert-ID follow-up

The expert-ID kernels were still doing scalar-column work over q4-packed
weights. Adjacent SIMD lanes loaded the same packed `uint32` word and extracted
one q value each. The packed-column rewrite makes each lane load one packed word
and unpack its values locally before the SIMD reduction.

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

Against same-prompt-length llama.cpp `Q4_K_M`, that leaves a `1.0892x` prefill
gap and a `1.1560x` decode gap. It is `1.1214x` faster than the prior
shared-input split expert-ID lane, but still `1.2641x` short of the `100 tok/s`
floor.

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

A follow-up tested the visual "double work" hypothesis by preserving Gemma 4's
1024-token sliding-window capacity inside the fixed-cache lane. The native
overflow update now uses a compiled `take` plus final-slot overwrite path
because MLX compile cannot infer the output shapes for `slice` or `roll` in
that closure. Correctness is covered by
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

### Prefill chunk-size sweep

The default planner still reports `load.prefill_chunk_size: 2048`. To test
whether the 2204-token README prompt was paying an avoidable second prefill
chunk, `driver-profile` now accepts `-prefill-chunk-size` as a diagnostic load
override. The sweep kept the active fixed-cache packed expert-ID lane:
`-cache-mode paged`, `-expert-id-fused-activation`, `-sorted-expert-prefill`,
`-fixed-gemma4-cache`, `-fixed-gemma4-shared-mask`, and
`-direct-greedy-token`.

Three-run results:

| Prefill chunk | Prefill tok/s | Decode tok/s | Peak bytes | Artefact |
| ---: | ---: | ---: | ---: | --- |
| `1024` | `1658.2779108140055` | `83.31228694999267` | `18148762344` | `docs/runtime/2026-05-18-go-mlx-gemma4-26b-a4b-q4-prefill-chunk1024-3run-readme-sweep.json` |
| `2048` | `1933.0886541161783` | `83.86143957778368` | `18419404064` | `docs/runtime/2026-05-18-go-mlx-gemma4-26b-a4b-q4-prefill-chunk2048-3run-readme-sweep.json` |
| `4096` | `2101.369627343361` | `83.74497136862215` | `18591487096` | `docs/runtime/2026-05-18-go-mlx-gemma4-26b-a4b-q4-prefill-chunk4096-3run-readme-sweep.json` |

For this prompt, `4096` makes prefill effectively all-in-one and is the clear
winner. It is `1.0871x` faster than `2048` prefill and `1.2672x` faster than
`1024`, while costing about `172MB` more peak memory than `2048` and about
`443MB` more than `1024`. Against same-prompt llama.cpp `Q4_K_M`, `4096` is
within `0.38%` of prefill parity (`2101.369627343361` versus
`2109.335561 tok/s`). Decode stays in the same `83-84 tok/s` band, so this is
not the remaining llama.cpp decode fix.

The measured win was promoted into the high-memory planner by widening the
64GB-class default from `2048` to `4096`. The no-override rerun confirms the
default path now reports `load.prefill_chunk_size: 4096`:

`docs/runtime/2026-05-18-go-mlx-gemma4-26b-a4b-q4-default-wide-prefill-planner-3run-readme.json`

```text
prompt_tokens: 2204
prefill: 2088.289027094623 tok/s
run prefill tok/s: 2055.580173863937, 2104.0715909404157, 2105.2153164795163
decode: 83.09590032942343 tok/s
run decode tok/s: 82.67387547724431, 83.03889708276647, 83.5749284282595
peak_memory_bytes: 18591487096
active_memory_bytes: 16664275120
stderr_bytes: 0
bin/lthn-mlx SHA-256: 42d1dc76efbe75e61e833164c8fe8fc6193a29e56b1eb25c8b2e2b15e393c447
```

That default-planner run is `1.0803x` faster than the `2048` control on prefill
and reaches `99.00%` of same-prompt llama.cpp prefill. Decode remains slower:
same-prompt llama.cpp is still `1.1005x` faster on generation.

The 2336-slot token-phase profile is:

`docs/runtime/2026-05-18-go-mlx-gemma4-26b-a4b-q4-packed-expert-id-fixed-cache2336-token-phases.json`

```text
decode: 83.73000373542442 tok/s
steady token average: 12.020852016ms
steady Eval(next): 10.624570008ms
steady next-forward graph construction: 1.357705992ms
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

The final token-phase profile is:

`docs/runtime/2026-05-18-go-mlx-gemma4-26b-a4b-q4-packed-expert-id-final-token-phases.json`

```text
decode: 78.66136991155207 tok/s
steady token average: 12.794125648ms
steady Eval(next): 11.461327984ms
steady next-forward graph construction: 1.301446032ms
stderr_bytes: 0
```

A follow-up scale-hoist variant for aligned q4 groups was correct but slower:

`docs/runtime/2026-05-18-go-mlx-gemma4-26b-a4b-q4-packed-scale-hoist-expert-id-fixed-uniform-direct-greedy-3run-readme-llamacpp-comparison-longdecode.json`

```text
decode: 77.70903294390506 tok/s
prefill: 1939.4991106953985 tok/s
stderr_bytes: 0
```

That variant was reverted while keeping the packed-column q iteration.

The packed path was also rechecked with `-compiled-gemma4-layer` enabled:

`docs/runtime/2026-05-18-go-mlx-gemma4-26b-a4b-q4-packed-expert-id-compiled-layer-token-phases.json`

```text
decode: 78.78857639506562 tok/s
prefill: 1928.2622708114843 tok/s
steady token average: 12.771735744ms
steady Eval(next): 11.381450264ms
steady next-forward graph construction: 1.358808696ms
stderr_bytes: 0
```

That is slightly below the packed 3-run baseline (`79.1105587686013 tok/s`) and
still leaves same-prompt llama.cpp `1.1607x` faster on decode, so the compiled
layer stays a rejected probe for this lane.

The existing compiled per-layer-input tensor gate was also rechecked on the
packed path:

`docs/runtime/2026-05-18-go-mlx-gemma4-26b-a4b-q4-packed-expert-id-compiled-per-layer-inputs-token-phases.json`

```text
decode: 77.0865964024348 tok/s
prefill: 1914.738466606945 tok/s
steady token average: 13.053710288ms
steady Eval(next): 11.575552296ms
steady next-forward graph construction: 1.43809028ms
stderr_bytes: 0
```

It is slower than the packed baseline and leaves same-prompt llama.cpp
`1.1863x` faster on decode, so it stays off for this lane.

The existing native MLP GELU wrapper was rechecked on the packed path too:

`docs/runtime/2026-05-18-go-mlx-gemma4-26b-a4b-q4-packed-expert-id-native-mlp-token-phases.json`

```text
decode: 77.96201603724107 tok/s
prefill: 1917.671369776293 tok/s
steady token average: 12.903903664ms
steady Eval(next): 11.517494352ms
steady next-forward graph construction: 1.353573288ms
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

No new `mlx_lm` measurements were taken for this pass.

## Comparison

| Lane | go-mlx | llama.cpp `Q8_K_XL` | llama.cpp `Q4_K_M` | Read |
| --- | ---: | ---: | ---: | --- |
| Short prefill, ~29 tokens | `443.894 tok/s` | `375.334 tok/s` | `468.943 tok/s` | q4 llama.cpp is `1.06x` faster |
| Decode, 128 tokens | `56.220 tok/s` | `87.689 tok/s` | `89.001 tok/s` | q4 llama.cpp is `1.58x` faster |
| Long prefill, ~2k tokens | `903.029 tok/s` at 2061 tokens | `2231.973 tok/s` at 2048 tokens | `2184.109 tok/s` at 2048 tokens | q4 llama.cpp is `2.42x` faster |
| Sorted long prefill, prompt-file | `1914.030 tok/s` at 2204 tokens | `2231.973 tok/s` at 2048 tokens | `2184.109 tok/s` at 2048 tokens | q4 llama.cpp is now `1.14x` faster |
| Sorted prefill plus fast-concat decode, prompt-file | `42.372 tok/s` decode at 2204-token context | `90.996 tok/s` at 2048-token context | `92.624 tok/s` at 2048-token context | q4 llama.cpp is now `2.19x` faster |
| Sorted prefill plus fixed-cache compiled decode, prompt-file | `48.935 tok/s` decode at 2204-token context | `90.996 tok/s` at 2048-token context | `92.624 tok/s` at 2048-token context | q4 llama.cpp is now `1.89x` faster |
| Sorted prefill plus fixed-cache compiled direct-greedy decode, prompt-file | `49.755 tok/s` 3-run decode at 2204-token context | `90.996 tok/s` at 2048-token context | `92.624 tok/s` at 2048-token context | q4 llama.cpp is now `1.86x` faster |
| Sorted prefill plus expert-ID fused direct-greedy decode, prompt-file | `49.973 tok/s` 3-run decode at 2204-token context | `90.996 tok/s` at 2048-token context | `92.624 tok/s` at 2048-token context | q4 llama.cpp is now `1.85x` faster |
| Same prompt length, prompt-file | `1915.337 tok/s` prefill and `49.973 tok/s` decode at 2204-token context | n/a | `2109.336 tok/s` pp2204 and `91.451 tok/s` tg128 | q4 llama.cpp is `1.10x` faster on prefill and `1.83x` faster on decode |
| Fixed-cache sliding-window diagnostic, prompt-file | `1806.832 tok/s` prefill and `40.760 tok/s` decode at 2204-token context | n/a | `2109.336 tok/s` pp2204 and `91.451 tok/s` tg128 | rejected; q4 llama.cpp is `2.24x` faster on decode and memory rises to `71.2GB` |
| Current fixed-uniform cache lane, prompt-file | `1923.322 tok/s` prefill and `49.715 tok/s` decode at 2204-token context | n/a | `2109.336 tok/s` pp2204 and `91.451 tok/s` tg128 | q4 llama.cpp is `1.10x` faster on prefill and `1.84x` faster on decode |
| Router-residual source parity lane, prompt-file | `1933.637 tok/s` prefill and `50.234 tok/s` decode at 2204-token context | n/a | `2109.336 tok/s` pp2204 and `91.451 tok/s` tg128 | q4 llama.cpp is `1.09x` faster on prefill and `1.82x` faster on decode |
| Split/BF16 expert-ID fused activation with shared input, prompt-file | `1923.997 tok/s` prefill and `70.545 tok/s` decode at 2204-token context | n/a | `2109.336 tok/s` pp2204 and `91.451 tok/s` tg128 | q4 llama.cpp is `1.10x` faster on prefill and `1.30x` faster on decode |
| Packed-column expert-ID fused activation with shared input, prompt-file | `1936.550 tok/s` prefill and `79.111 tok/s` decode at 2204-token context | n/a | `2109.336 tok/s` pp2204 and `91.451 tok/s` tg128 | q4 llama.cpp is `1.09x` faster on prefill and `1.16x` faster on decode |
| Automatic request-sized fixed-cache packed expert-ID, prompt-file | `1935.361 tok/s` prefill and `84.010 tok/s` decode at 2204-token context | n/a | `2109.336 tok/s` pp2204 and `91.451 tok/s` tg128 | q4 llama.cpp is `1.09x` faster on prefill and `1.09x` faster on decode |
| Rejected native router top-k on fixed-cache packed expert-ID, prompt-file | `83.541 tok/s` decode; repeated prompt-cache restores average `4.694ms` for the 2204-token prefix | n/a | `2109.336 tok/s` pp2204 and `91.451 tok/s` tg128 | rejected for decode; q4 llama.cpp is `1.095x` faster, but durable fixed-cache wake avoids replaying the repeated prefix |
| Rejected per-layer sliding fixed-cache packed expert-ID, prompt-file | `2033.387 tok/s` prefill and `73.060 tok/s` decode at 2204-token context | n/a | `2109.336 tok/s` pp2204 and `91.451 tok/s` tg128 | rejected; q4 llama.cpp is `1.25x` faster on decode |
| Restored uniform request-sized fixed-cache packed expert-ID, prompt-file | `1925.998 tok/s` prefill and `83.596 tok/s` decode at 2204-token context | n/a | `2109.336 tok/s` pp2204 and `91.451 tok/s` tg128 | q4 llama.cpp is `1.09x` faster on decode |
| Prefill chunk-size `4096` override, prompt-file | `2101.370 tok/s` prefill and `83.745 tok/s` decode at 2204-token context | n/a | `2109.336 tok/s` pp2204 and `91.451 tok/s` tg128 | q4 llama.cpp is only `1.0038x` faster on prefill and `1.09x` faster on decode |
| Default 64GB-class wide-prefill planner, prompt-file | `2088.289 tok/s` prefill and `83.096 tok/s` decode at 2204-token context | n/a | `2109.336 tok/s` pp2204 and `91.451 tok/s` tg128 | q4 llama.cpp is `1.0101x` faster on prefill and `1.10x` faster on decode |
| llama.cpp PR 23211 assistant MTP `n_max=2`, CLI | n/a | n/a | `1615.7 tok/s` prompt and `100.2 tok/s` generation | unmerged llama.cpp PR path; visible speculative lane, not raw target-only parity |
| llama.cpp PR 23211 assistant MTP `n_max=2`, server | n/a | n/a | `1562.0125388366318 tok/s` prompt and `93.76822253543413 tok/s` generation | accepted `75/101` draft tokens; visible speculative lane, not raw target-only parity |

The useful signal is that the remaining gap is not uniform. go-mlx is fine on
small prompt setup after the mixed-q loader fix, and the fused expert gate/up
path trims only a little decode duplication. The automatic last-token
long-prefill path removed one full-logits materialisation waste, and sorted
expert prefill removes the first major MoE route-order waste. The fast-concat
paged decode probe removes one avoidable multi-page attention tax, and the
fixed-cache compiled direct-greedy decode probe removes another slice of
cache-shape and output-selection churn. The router-residual source-parity fix
removes a small graph-shape mismatch, while the two-column down matvec shows
that partial row-pairing is not the missing kernel boundary. The split/BF16
expert-ID path is the first large decode improvement in this lane because it
removes the silent fallback on the active safetensors and avoids shared-input
broadcast work. The packed-column follow-up then removes a lower-level q4 load
duplication inside those custom kernels. The q4 follow-up now says large
prefill is close enough to be a secondary problem, and the wide-prefill planner
now makes that explicit by putting this prompt within about `1.0%` of llama.cpp
prefill by default. The remaining primary gap is still decode at real context
length, where llama.cpp is getting more value from stable graph topology,
KV/cache layout, flash attention, and Metal command scheduling than go-mlx
currently gets from the MLX graph assembled per step.

The assistant MTP rows are deliberately kept out of raw target-only parity.
They show a viable visible-throughput lane if go-mlx adds the same target plus
assistant speculative API and the proposed/accepted/rejected token metrics. They
also confirm that larger draft windows are not automatically better on this
hardware: the same PR CLI path drops from `100.2 tok/s` at `n_max=2` to
`90.7 tok/s` at `n_max=4` and `61.5 tok/s` at `n_max=8`.

<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# go-mlx Production Goal

This file is the current operating target for go-mlx. Keep it concise: one
baseline, current gates only, and no play-by-play benchmark ledger. Detailed
evidence belongs in `docs/runtime/` or generated benchmark artefacts.

## Product Target

Make go-mlx the production Apple Silicon runtime for LTHN/Lemma agentic
workflows:

- Native Go/Metal model loading and generation, with no Python subprocess
  fallback in the production path.
- Native platform floor is known: [macOS Tahoe 26.0+](https://developer.apple.com/documentation/macos-release-notes/macos-26-release-notes)
  on Apple Silicon. Do not lower build/link targets to older macOS floors;
  earlier releases such as 11.x do not provide the Metal 4 APIs this lane uses.
- Durable retained `State` for repeated agent turns, avoiding replayed prefill
  when the state is compatible.
- Default model policy tuned for quality and speed: 6-bit first, 8-bit when
  headroom allows, 4-bit only for constrained hardware or archived controls.
- Practical throughput target: keep decode at or above the 100 tok/s production
  floor where the model/quant/context allows it, while proving retained-state
  wall time beats replay-first runners on real 10+ turn workflows.

## Single Baseline

Canonical baseline until replaced by a newer signed benchmark:

- Model lane: Gemma 4 E2B/E4B, target model first, MTP assistant as an optional
  sidecar.
- Quant lane: `mlx-community` 6-bit as the default baseline, with 8-bit and
  4-bit recorded as comparison variants.
- Workflow shape: empty/new session or opencode-sized first wake
  (`30k`-`40k` tokens), then retained append/generate turns. The `100k` lane is
  a stress and degradation probe, not the normal pass/fail shape.
- Claims table: compare go-mlx against itself first, then llama.cpp as the
  external anchor. Track decode tok/s, prefill/restore time, end-to-end wall
  time, peak active memory, virtual memory, and estimated energy from active
  wall time.

Do not add ad hoc rows here. Replace this baseline only when a new benchmark
run has artefacts and the older baseline is moved to runtime history.

## Current Architecture State

The profile table is native/staged across the production set. Former
metadata-only gaps now validate model/config/tokenizer metadata without Python.
Some families still stop at explicit native diagnostics until their generation
kernels are wired.

Native generation paths are established for the dense Gemma/Qwen/Llama/Mistral
style families. The remaining work is not another profile sweep; it is shared
kernel and forward-path completion for structures reused across model families.

## Open Gates

- Shared Metal primitives:
  - MoE router projection: hidden state to expert scores.
  - MoE top-k and route-weight normalisation.
  - Selected expert gate/up activation and weighted down projection.
  - Hybrid/linear-attention primitives for Qwen 3.6 style layers.
- Shared MoE full decode is wired for `qwen3_moe`, `mixtral`, `gpt_oss`, and
  `kimi`; finish the staged `qwen3_6_moe` and `deepseek` decode paths.
- DeepSeek staged loading now validates an MLA plan from config; finish MLA
  projection/cache kernels and Qwen 3.6 hybrid attention before treating those
  families as full generation-complete.
- Finish MiniMax M2 standalone generation after JANGTQ/MXTQ sparse primitives
  are validated.
- Validate official Google Gemma 4 E2B/E4B target and assistant snapshots
  across 6-bit/8-bit/4-bit packs.
- Keep TurboQuant/KV compression as an opt-in research lane until long-context
  output quality and retained-state memory are proven against fp16/q6/q8.

## Benchmark Rules

- No artificial smoke-token floors for production claims. Smoke tests only
  prove the binary and harness work.
- Separate restore/compact time from decode time. Compact is an overflow or
  operator-requested tool, not a normal retained-turn benchmark step.
- Retained-state benchmarks append only new turns; replay-first runs are marked
  as replay controls.
- Use realistic prompts with enough content to generate real chapters or agent
  work. Token count alone is not an output-quality metric.
- Keep one active run shape at a time when memory is under investigation.
- Use `GOWORK=/Users/snider/Code/core/go-mlx/go.work`; do not benchmark with
  `GOWORK=off`.

## Verification Gates

Focused gates for this lane:

```sh
env GOWORK=/Users/snider/Code/core/go-mlx/go.work GOCACHE=/private/tmp/go-mlx-self/gocache go test -ldflags "-extldflags=-mmacosx-version-min=26.0" ./go/internal/metal -run 'Test(MoERouter|Gemma4Router|Model_LoadModel_|Model_Generate_Qwen3MoEDiagnostic)' -count=1
env GOWORK=/Users/snider/Code/core/go-mlx/go.work GOCACHE=/private/tmp/go-mlx-self/gocache go test -ldflags "-extldflags=-mmacosx-version-min=26.0" ./go/profile ./go/cmd/mlx ./go -count=1
env GOWORK=/Users/snider/Code/core/go-mlx/go.work GOCACHE=/private/tmp/go-mlx-self/gocache go build -ldflags "-extldflags=-mmacosx-version-min=26.0" -o /private/tmp/go-mlx-self/bin/lthn-mlx ./go/cmd/mlx
```

Production benchmark artefacts should include the exact model path/revision,
quant pack, context shape, command, stderr, memory capture method, and output
sample or quality note.

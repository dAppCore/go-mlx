<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# go-mlx Goal

The production Apple Silicon runtime for LTHN/Lemma agentic + coder workflows.
Native Go/Metal model loading, generation, and training — **no Python in the
production path**. Platform floor: macOS Tahoe 26.0+ on Apple Silicon (Metal 4);
do not lower build/link targets. Build/link with
`-ldflags "-extldflags=-mmacosx-version-min=26.0"`; `GOWORK=…/go.work` (never off).

## Architecture — model↔runtime SDK (the shape to hold)

`pkg/metal` is the runtime + cgo + the **neutral algo features** every model
composes:

- `transformer.go` — `MLP` (GELU), `SiLUMLP`, activations
- `attention.go` — `GQAAttention`, `DenseDecoderLayer`, `DenseConfig`
- `moe.go` + `moe_*.go` — `MoERouter`, SwiGLU experts, routing
- caches, native fused kernels, the runtime-author API (`runtime_author.go`),
  and the capability interfaces (`model.go`)

A **model** is a thin pure-Go package `pkg/metal/model/{family}` that *composes*
those features, implements `metal.InternalModel` + capability interfaces,
self-registers from `init()` via `metal.RegisterModelLoader`, and is
blank-imported through `speculative.go`. A model package has **no cgo and names
no private metal symbol**. (`go/internal/metal` is gone — everything is `pkg/metal`.)

Extracted so far: bert staged/rerank, deepseek staged, gemma4, gemma3, mixtral,
kimi, gpt_oss, minimax_m2, qwen3 (dense + Qwen3.6 staged), qwen3_moe.
Design: `docs/RFC.model-sdk.md`.

## Active goals

1. **Finish model support on the split SDK.** The model split is complete at the
   current package level: `pkg/metal` owns the neutral runtime/features and each
   model composes them from `pkg/metal/model/{family}`. Next,
   complete the generation paths still stopping at diagnostics (shared MoE /
   Qwen3.6 hybrid-attention / DeepSeek MLA / MiniMax M2 sparse) and validate the
   official Gemma 4 packs. New models (e.g. Gemma4-12B: dense + multimodal + MTP)
   **compose the features — never a new monolith**.

2. **Native Simple Self-Distillation — `github.com/apple/ml-ssd`, no Python.**
   The three-step core already lives in `go/ssd.go` + `go/distill.go`:
   `RunSimpleSelfDistillation` samples a frozen model at non-unit temperature
   (`SampleGenerateConfig`), SFTs on the raw outputs, and decodes at a
   separately-tuned temperature (`DecodeGenerateConfig`). Close the gap to the
   pipeline ml-ssd actually ships:
   - **Data-gen post-process** — add `filter_shortest_percent` (drop the bottom
     length-decile of generations before SFT) and the `repetition_penalty`
     sampling knob to `SimpleSelfDistillationConfig`; both are in ml-ssd's
     `data_generation/config.yaml`.
   - **Eval harness** — port `evaluation/eval.py` + `benchmark.py`: a
     LiveCodeBench-v6 code-execution benchmark running each generated solution
     against its tests, with `n_repeat` and configurable sampling
     (temperature/top_p/top_k), results to an output path.
   - **Parity** — reproduce the SimpleSD-4B-instruct / -thinking / -30b-a3b
     recipes natively; artefacts in `docs/runtime/`.

3. **Native hierarchical-memory pretraining — `github.com/apple/ml-memory-pretraining`,
   no Python.** Implement the memory-augmented architecture in Go: a small anchor
   model + a hierarchical memory bank whose context-dependent blocks are retrieved
   (cluster-ID lookup) and added into the feed-forward layers — plus the offline
   build (hierarchical KMeans over an embedded corpus → centroids → memory bank →
   retriever). Lets a small local coder model punch above its parameter count from
   a memory bank (edge-efficient), on go-mlx primitives.

4. **Gemma 4 12B support — `huggingface.co/google/gemma-4-12B`.** Extend
   `pkg/metal/model/gemma4` (compose, never a new monolith) for the 12B
   "Unified" pack — 11.95B / 48 layers / 256K context / 262K vocab, BF16:
   - **Hybrid attention** — interleaved 1024-token local sliding-window + full
     global (final layer global), with unified KV and proportional RoPE (p-RoPE)
     on the global layers. Build on `gemma4/attention.go` + the hybrid-attention
     cache planning already hoisted.
   - **Encoder-free unified multimodal** — "Unified" = no dedicated encoders
     (unlike 31B/E4B). Raw image patches (variable aspect/resolution) and audio
     waveforms (≤30s) project directly into the embedding space through
     lightweight linear layers; video arrives as frame sequences (≤60s). Every
     modality flows into the one decoder-only transformer — implement the linear
     projection path, not encoder subgraphs.
   - **Decode** — `<|think|>` thinking (already wired) at the card's sampling
     defaults (temperature 1.0, top_p 0.95, top_k 64).
   Validate the official `gemma-4-12B-it` pack; artefacts (path+revision, quant,
   context shape, command, sample) in `docs/runtime/`.

5. **Intel AutoRound quantisation — `github.com/intel/auto-round`, no Python.**
   Add AutoRound as a native quant algorithm beside the existing GGUF/jang paths
   (`go/gguf/quantize.go`, `go/quant/jang`, `go/profile/algorithm.go`):
   - **Loader** — read AutoRound-exported packs: the `auto_round` native format
     and its `gguf` exports (e.g. `GGUF:Q4_K_M`), dequantising on Metal.
   - **Quantiser** — the SignRound signed-gradient-descent rounding optimisation
     (the `iters` tuning loop; `iters=0` = RTN baseline), weight-only schemes
     W4A16 / W2A16 / W8A16 with `group_size` 32/64/128 and sym/asym, plus the FP
     variants (MXFP4 / NVFP4 / FP8). Calibrate over a sample corpus (nsamples /
     seqlen knobs).
   - Expose `auto-round` / `auto-round-best` / `auto-round-light` as algorithm
     profiles; validate a quantised gemma-4-12B pack round-trips load + generate.

## Extraction recipe (proven on gemma4 / gemma3 / mixtral / kimi)

1. Rescue shared helpers: any helper in `<model>.go` used outside it → move to a
   neutral metal feature file + export.
2. `git mv <model>.go → pkg/metal/model/<name>/`; `package metal` → `package <name>`.
3. Qualify metal refs (`gofmt -r 'Sym -> metal.Sym'` per symbol; `goimports -w`).
4. Register in `init()` via `metal.RegisterModelLoader`; remove from metal's
   registry; blank-import in `speculative.go`.
5. Relocate or `t.Skip` test straddles — never export metal internals for a test.
6. Green: build `pkg/metal` + the new package + `cmd/mlx`; `go list -deps
   ./go/pkg/metal/` shows no `model/<name>` (no cycle); `go test ./go/...`.
   Commit per model.

## Rules (de-ralph discipline — do not regress)

- **Compose, don't monolith.** A model needing an algo another uses → hoist it
  neutral to a feature file; never copy it into the model.
- **Dedupe only if identical.** e.g. `SiLUMLP` ≠ `MLP` (SiLU vs GELU) — keep
  variants distinct; a lossy merge changes behaviour.
- Behaviour-preserving moves; no algo logic change during a split.
- **Port what upstream ships.** When a goal names an upstream repo (`ml-ssd`,
  `ml-memory-pretraining`), scope by what it actually implements — its config
  knobs, pipeline stages, eval — not by preference. A feature is in the repo
  because the method needs it; don't ethics-scope it out.
- Git: never `reset --hard` / `stash` / `checkout -- <path>` / `add -A`; explicit
  `git add` paths; `git mv` to move; commit per model/feature.
- UK English; `// SPDX-Licence-Identifier: EUPL-1.2` on every new file.

## Verification gates

```sh
env GOWORK=…/go.work GOCACHE=/private/tmp/go-mlx-self/gocache \
  go build -ldflags "-extldflags=-mmacosx-version-min=26.0" ./go/pkg/metal/...
go test ./go/...                       # green
go build -ldflags "-extldflags=-mmacosx-version-min=26.0" \
  -o /private/tmp/go-mlx-self/bin/lthn-mlx ./go/cmd/mlx   # binary links
```

Production-claim artefacts (model path+revision, quant, context shape, command,
stderr, memory method, output sample) go in `docs/runtime/`.

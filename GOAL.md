<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# go-mlx Goal

The production Apple Silicon runtime for agentic + coder workflows.
Native Go/Metal model loading, generation, and training — **no Python in the
production path**. Platform floor: macOS Tahoe 26.0+ on Apple Silicon (Metal 4);
do not lower build/link targets. Build/link with
`-ldflags "-extldflags=-mmacosx-version-min=26.0"`; `GOWORK=…/go.work` (never off).

## Active goals

1. **Finish model support on the split SDK.** The model split is complete at the
   current package level: `pkg/metal` owns the neutral runtime/features and each
   model composes them from `pkg/metal/model/{family}`. Next,
   complete the generation paths still stopping at diagnostics (shared MoE /
   Qwen3.6 hybrid-attention / DeepSeek MLA / MiniMax M2 sparse) and validate the
   official Gemma 4 packs. New models (e.g. Gemma4-12B: dense + multimodal + MTP)
   **compose the features — never a new monolith**.

2. **Native hierarchical-memory pretraining — `github.com/apple/ml-memory-pretraining`,
   no Python.** Implement the memory-augmented architecture in Go: a small anchor
   model + a hierarchical memory bank whose context-dependent blocks are retrieved
   (cluster-ID lookup) and added into the feed-forward layers — plus the offline
   build (hierarchical KMeans over an embedded corpus → centroids → memory bank →
   retriever). Lets a small local coder model punch above its parameter count from
   a memory bank (edge-efficient), on go-mlx primitives.

3. **Gemma 4 12B support — `huggingface.co/google/gemma-4-12B`.** Extend
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

4. **Benchmark decode throughput — target ≥100 tok/s.** Make throughput a tracked,
   gated objective, not an afterthought (AX-11). The harness already exists —
   `*_bench_test.go`, the `cmd/mlx` production-compare tools
   (`production_mtp_compare`, `production_turboquant_compare`, `auto_tune`), and
   go-inference's `GenerateMetrics` reporting prefill/decode tok/s + GPU memory.
   Current M3 Ultra decode: Gemma3-1B 4-bit 82, Gemma 4 E2B ~80, 26B ~25; the MTP
   speculative path already averages ~110. Goal: **sustained ≥100 tok/s decode**
   on the coder packs (E2B/E4B and the quantised mid-size) via MTP + quant tuning.
   Record per-pack throughput artefacts (model, quant, context, tok/s, GPU mem) in
   `docs/runtime/`; fail the production compare when a pack regresses below target.

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

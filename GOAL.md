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

Extracted so far: gemma4, gemma3, mixtral, kimi. Design: `docs/RFC.model-sdk.md`.

## Active goals

1. **Finish the model split + fill out model support.** Extract the remaining
   monoliths in `pkg/metal` — `gpt_oss`, `minimax_m2`, `qwen3` (+ `qwen3_moe`) —
   into their own packages by composing the features (recipe below). Then
   complete the generation paths still stopping at diagnostics (shared MoE /
   Qwen3.6 hybrid-attention / DeepSeek MLA / MiniMax M2 sparse) and validate the
   official Gemma 4 packs. New models (e.g. Gemma4-12B: dense + multimodal + MTP)
   **compose the features — never a new monolith**. Done when `pkg/metal` holds
   only the SDK + runtime and every model is a thin package.

2. **Native Simple Self-Distillation — `github.com/apple/ml-ssd`, no Python.**
   Implement the SSD loop in Go on go-mlx's own generation + training + LoRA
   primitives: (a) sample from a frozen model at non-unit temperature, (b)
   fine-tune on the raw unverified outputs with cross-entropy, (c) decode at a
   separately-tuned temperature. No verifier/teacher/RL. This is Lemma's
   self-improvement loop for the local coder models.

3. **Native hierarchical-memory pretraining — `github.com/apple/ml-memory-pretraining`,
   no Python.** Implement the memory-augmented architecture in Go: a small anchor
   model + a hierarchical memory bank whose context-dependent blocks are retrieved
   (cluster-ID lookup) and added into the feed-forward layers — plus the offline
   build (hierarchical KMeans over an embedded corpus → centroids → memory bank →
   retriever). Lets a small local coder model punch above its parameter count from
   a memory bank (edge-efficient), on go-mlx primitives.

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

<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# model_pack.go — model-pack validation + format detection

**Package**: `dappco.re/go/mlx`
**File**: `go/model_pack.go`

## What this is

The **pre-load validator** for model packs. Given a model directory, answers:

- What format is this? (safetensors / GGUF / future)
- What architecture? (Gemma 3 / 4, Qwen 2 / 3, Llama 3, MiniMax M2)
- What quantisation? (none / Q4/Q8 / JANG / VQ)
- What capabilities does it claim? (reasoning, tool-use, chat template, …)
- Is it loadable on this backend?

Returns an `inference.ModelPackInspection` — the portable shape from `go-inference/contracts.go`. Used by `LoadModel` for pre-flight checks, by the IDE model picker, and by `core/api` for the `/v1/models/capabilities` endpoint.

## ModelPackFormat

```go
type ModelPackFormat string

ModelPackFormatSafetensors = "safetensors"
ModelPackFormatGGUF        = "gguf"
```

Two formats today. Safetensors is the HuggingFace shape — `config.json` + `tokenizer.json` + `*.safetensors`. GGUF is the llama.cpp single-file shape.

## Inspection

```go
inspection := mlx.InspectModelPack(path)
```

Returns `*inference.ModelPackInspection`:

```go
type ModelPackInspection struct {
    Path         string
    Format       string                      // "safetensors" | "gguf"
    Model        ModelIdentity               // arch, quant, ctx, layers, vocab, hash
    Tokenizer    TokenizerIdentity           // kind, chat template, hash, BOS/EOS/PAD
    Supported    bool                        // can metal backend load this?
    Capabilities []Capability                // claimed feature surface
    Notes        []string                    // human-readable findings
    Labels       map[string]string
}
```

## Detection flow

```
ReadDir(path)
   ├── *.gguf present?  → ModelPackFormatGGUF
   │                        → readGGUFInfo(path)
   │                        → fill ModelIdentity from header
   │
   └── config.json present?  → ModelPackFormatSafetensors
                                → parseConfig
                                → detect arch (dense / MoE / JANG / VQ)
                                ├── IsMiniMaxM2Config? → minimax_m2 lane
                                ├── IsJANGModelPack?   → JANG quant lane
                                ├── IsCodebookPack?    → VQ quant lane
                                └── otherwise → standard safetensors
                                → check tokenizer.json present
                                → check chat_template.jinja (optional)
                                → check adapter_config.json (optional)
                                → compute pack hash
                                → emit ModelPackInspection
```

## Supported determination

A pack is `Supported: true` when:

- Format is recognised
- Architecture has a Metal forward implementation
- All required tensors are present per the architecture's shape contract
- Tokenizer is recognised (SentencePiece / GPT-2 BPE)
- Quantisation is one the runtime supports

Otherwise `Supported: false` with `Notes` describing why. The IDE picker filters supported packs; the audit pipeline records why unsupported ones aren't.

## Capabilities reported

Per-pack capabilities (vs per-backend or per-loaded-model):

- What chat template exists
- Whether tool-call / reasoning parsers are declared (from JANG sidecar)
- Whether the pack is quantised + which quant scheme
- Whether the pack carries adapter weights
- Architecture-specific flags (MoE expert count, MTP modules, etc.)

## Hash computation

The pack hash is SHA-256 of:

```
sorted(config.json + tokenizer.json + chat_template + adapter_config.json) + 
sorted(file_sizes_of(*.safetensors))
```

Lightweight — doesn't read tensor bytes. Captures everything that affects behaviour without forcing a full content scan. Tensor-bytes-changed-but-shape-unchanged: rare-and-suspicious case caught at first inference (KV restore hash mismatch).

## Used by

- `register_metal.go` LoadModel — pre-load validation
- `core/ide` model picker — "show only loadable models"
- `core/api` `/v1/models/capabilities` — list available + supported state
- Audit pipeline — inventory + freshness checks
- LARQL — model identity for cross-version diff

## Status

Dense models: production. MoE detection: in progress (JANGTQ + MiniMax lanes). VQ detection: metadata-aware.

## Related

- `../../../go-inference/docs/inference/contracts.md` — `ModelPackInspector` interface
- `../../../go-inference/docs/inference/discover.md` — `Discover()` finds packs to inspect
- `../../../go-inference/docs/inference/gguf.md` — GGUF metadata reader
- [../moe/minimax_m2.md](../moe/minimax_m2.md) — MiniMax detection
- [../moe/jang.md](../moe/jang.md) — JANG detection
- [../moe/codebook_vq.md](../moe/codebook_vq.md) — VQ detection

<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# codebook_vq.go — VQ codebook quantisation metadata

**Package**: `dappco.re/go/mlx`
**File**: `go/codebook_vq.go` (plus `internal/metal/codebook_vq.go` for Metal-side kernels)
**Status**: experimental (vMLX parity Phase 1)

## What this is

Metadata for **vector-quantised** tensors — a quantisation family adjacent to JANG/JANGTQ but distinct in shape. Where JANG quantises element-wise with per-tensor-class bit budgets, VQ quantises **vector-wise**: each row chunk is replaced by an index into a learned codebook of representative vectors.

VQ is common in:

- Some MiniMax pack variants
- Recent Qwen experiments
- Various third-party MLX quant repacks

## Constants

```go
CodebookQuantizationType = "codebook"
CodebookFormatVQ         = "vq"
```

These match the sidecar JSON values — `"type": "codebook"`, `"format": "vq"` in the pack's `*_codebook.json`.

## CodebookQuantizationProfile

```go
type CodebookQuantizationProfile struct {
    Type         string  // "codebook"
    Format       string  // "vq" | (future formats)
    CodebookSize int     // number of vectors in the book
    CodeDim      int     // dimension of each vector
    IndexBits    int     // bits per index (4 | 8 | 12 typical)
    Source       string  // upstream training source
    Tensors      []CodebookTensorDescriptor
}
```

## CodebookTensorDescriptor

```go
type CodebookTensorDescriptor struct {
    Name          string    // tensor name (e.g. "model.layers.0.mlp.gate_proj.weight")
    Format        string    // "vq" — must match parent format
    Shape         []uint64  // reconstructed tensor shape
    CodebookName  string    // which codebook to use (multi-codebook packs)
    IndexTensor   string    // *.safetensors key for the index stream
    CodebookTensor string   // *.safetensors key for the codebook itself
    // …
}
```

Each VQ-compressed tensor is paired:

- One **index stream** (per-row codebook indices, packed at IndexBits each)
- One **codebook** (CodebookSize × CodeDim float32 — or quantised further)

Reconstruction: `weight[row,col] = codebook[index[row]][col]`.

## Why VQ separately from JANG

JANG quantises *elements*. VQ quantises *vectors*. They can coexist in one model pack:

- JANG handles attention projections (element-wise tolerance high)
- VQ handles FFN expert weights (vectors clustered by training pattern, VQ exploits that)

The validator (this file) ensures the two schemes don't claim the same tensor.

## Native kernels

The actual VQ dequant + matmul kernels live in `internal/metal/codebook_vq.go`. From config side (this file), we plan and validate; from runtime side, we dispatch the right Metal kernel per tensor.

## Status

Metadata + validation: done. Native dequant: in progress. Codebook-aware matmul: planned (current path dequants to f32, then runs standard matmul — works but loses the VQ speed benefit).

## Related

- [jang.md](jang.md) — sibling element-wise quant scheme
- [minimax_m2.md](minimax_m2.md) — MiniMax packs sometimes use VQ for routed experts
- `../../../go-inference/docs/inference/capability.md` — `CapabilityCodebookVQ` flag
- `internal/metal/codebook_vq.go` — Metal-side dequant kernel
- `docs/vmlx-feature-gap-report.md` — origin context

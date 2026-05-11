<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# kv_snapshot.go — portable KV cache encode/decode

**Package**: `dappco.re/go/mlx`
**File**: `go/kv_snapshot.go`

## What this is

The on-disk binary format for one KV cache snapshot. Captures the K/V tensors from a live `metal.Model` into a portable byte stream that can be saved, transported, decoded later, and restored into a fresh model with the same architecture.

This file owns the **format spec** (magic, version, encoding enum, save/load/capture options) and the marshal/unmarshal. Block chunking lives in `kv_snapshot_blocks.go`; bundle indexing lives in `kv_snapshot_index.go`; memvid integration lives in `kv_snapshot_memvid.go`.

## Format

```
+-----------------------------------------------------+
| magic = "MLXKV001"            (8 bytes)             |
| version = 3                   (4 bytes uint32)      |
| encoding flag                 (1 byte)              |
| reserved                      (3 bytes)             |
| layer count                   (4 bytes uint32)      |
+-----------------------------------------------------+
| per-layer K/V tensors                               |
|  - layer header                                     |
|  - K tensor bytes                                   |
|  - V tensor bytes                                   |
+-----------------------------------------------------+
```

`KVSnapshotVersion = 3`. Older snapshots are not auto-upgraded — `LoadKVSnapshot` returns an error and the caller decides whether to re-capture.

## Encoding

```go
type KVSnapshotEncoding string

KVSnapshotEncodingFloat32 = "float32"   // exact float32 K/V — largest on disk
KVSnapshotEncodingQ8      = "q8"        // symmetric int8 + scale per tile — ~4x smaller, lossy
KVSnapshotEncodingNative  = "native"    // preserve captured dtype when available (bf16/fp16)
```

Native is the default for newly captured snapshots — Metal already holds K/V in the model's native dtype, so encoding it back into float32 just to satisfy old loaders wastes bytes and adds a round-trip lossless-but-pointless conversion.

## Options

```go
type KVSnapshotSaveOptions struct {
    KVEncoding KVSnapshotEncoding   // float32 | q8 | native
}

type KVSnapshotLoadOptions struct {
    RawKVOnly bool                  // skip float32 side decode — for raw-byte transport
}

type KVSnapshotCaptureOptions struct {
    RawKVOnly bool                  // capture native bytes only — skip float32 mirror
}
```

`RawKVOnly` is the "I'm forwarding this to a peer, don't decode" path used by the disaggregated inference layer (LARQL + memvid in `design_disaggregated_inference_lethean.md`).

## Public API

```go
snap.Save(ctx, w, opts) error
mlx.LoadKVSnapshot(r, opts) (*KVSnapshot, error)
model.CaptureKVSnapshot(opts) (*KVSnapshot, error)
model.RestoreKVSnapshot(snap) error
```

The CaptureKVSnapshot / RestoreKVSnapshot methods are on `*metal.Model` — same model, different lifecycle phase.

## Memory cost

A 92k-token Gemma-4 KV cache is ~10GB in float32. In native bf16: ~5GB. In Q8: ~1.3GB. The encoding choice is per-snapshot; block-cache encoding can differ from snapshot encoding.

## Why version 3

- v1 — initial format, no encoding flag (float32 only)
- v2 — added encoding flag, added per-layer header for variable layer counts
- v3 — added reserved bytes for forward-compat, removed implicit-float32 fallback

A v1/v2 snapshot encountered today produces a clear "format version too old" error rather than silent corruption.

## Related

- [kv_snapshot_blocks.md](kv_snapshot_blocks.md) — chunking strategy
- [kv_snapshot_index.md](kv_snapshot_index.md) — bundle index across multiple snapshots
- [kv_snapshot_memvid.md](kv_snapshot_memvid.md) — memvid bundle integration
- [agent_memory.md](agent_memory.md) — Wake/Sleep that uses this
- [state_bundle.md](state_bundle.md) — the Bundle envelope wrapping snapshots
- `../../../go-inference/docs/inference/capability.md` — `CapabilityKVSnapshot` advertises this

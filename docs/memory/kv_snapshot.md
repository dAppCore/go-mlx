<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# kv/snapshot.go — portable KV cache encode/decode

**Package**: `dappco.re/go/mlx/kv`
**Files**: `go/kv/snapshot.go`, `go/kv/snapshot_encode.go`, `go/kv/snapshot_decode.go`

## What this is

The on-disk binary format for one KV cache snapshot. Captures the K/V tensors from a live model into a portable byte stream that can be saved, transported, decoded later, and restored into a fresh model with the same architecture.

This package owns the **format spec** (magic, version, encoding enum, save/load/capture options) and the marshal/unmarshal. Block chunking lives alongside it in `blocks*.go`; State integration lives in `state_store.go`.

## Format

```
+-----------------------------------------------------+
| magic = "MLXKV001"            (8 bytes)             |
| version = 6                   (4 bytes uint32)      |
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

`kv.SnapshotVersion = 6`. Older snapshots are not auto-upgraded — `kv.Load` returns an error and the caller decides whether to re-capture.

## Encoding

```go
type Encoding string

KVSnapshotEncodingFloat32 Encoding = "float32"  // exact float32 K/V — largest on disk (note the legacy "KVSnapshot" prefix kept for compatibility)
EncodingQ8                Encoding = "q8"        // symmetric int8 + scale per tile — ~4x smaller, lossy
EncodingNative             Encoding = "native"   // preserve captured dtype when available (bf16/fp16), falling back to float32
```

## Options

```go
type SaveOptions struct {
    KVEncoding Encoding   // float32 | q8 | native
}

type LoadOptions struct {
    RawKVOnly bool         // skip float32 side decode — for raw-byte transport
}

type CaptureOptions struct {
    RawKVOnly       bool   // capture native bytes only — skip float32 mirror
    BlockStartToken int    // skip capture of blocks ending at or before this token
}
```

`RawKVOnly` is the "I'm forwarding this to a peer, don't decode" path used by the disaggregated inference layer.

## Public API

```go
snap.Save(path) error
snap.SaveWithOptions(path, kv.SaveOptions{...}) error
kv.Load(path) (*kv.Snapshot, error)
kv.LoadWithOptions(path, kv.LoadOptions{...}) (*kv.Snapshot, error)

model.CaptureKV(prompt) (*kv.Snapshot, error)              // *mlx.Model — runs one prefill pass
model.CaptureKVWithOptions(prompt, kv.CaptureOptions{...}) (*kv.Snapshot, error)
model.WarmPromptCacheFromKV(snap) error                     // installs a captured prefix as the live prompt cache
```

`CaptureKV`/`WarmPromptCacheFromKV` are root `*mlx.Model` methods that delegate onto the internal `pkg/metal.Model`'s `CaptureKV`/`RestorePromptCacheFromKV` (same model, different lifecycle phase) — see [`docs/examples/model-ops/kv-snapshot.md`](../examples/model-ops/kv-snapshot.md) for a full walk-through.

## Memory cost

A 92k-token Gemma-4 KV cache is ~10GB in float32. In native bf16: ~5GB. In Q8: ~1.3GB. The encoding choice is per-snapshot; block-cache encoding can differ from snapshot encoding.

## Version History

- v1 — initial format, no encoding flag (float32 only)
- v2 — added encoding flag, added per-layer header for variable layer counts
- v3 — added reserved bytes for forward-compat, removed implicit-float32 fallback
- v4-v5 — intermediate revisions, not detailed here
- v6 — added each layer's source-cache `MaxSize` (window/rotation clamp) so a wake restore carries the slept window/rotation geometry instead of trusting wake-era model templates

A snapshot older than the running binary's `kv.SnapshotVersion` produces a clear "unsupported KV snapshot version" error rather than silent corruption.

## Related

- [kv_snapshot_blocks.md](kv_snapshot_blocks.md) — chunking strategy
- [kv_snapshot_index.md](kv_snapshot_index.md) — bundle index across multiple snapshots
- [kv_snapshot_state.md](kv_snapshot_state.md) — State bundle integration
- [agent_memory.md](agent_memory.md) — Wake/Sleep that uses this
- [state_bundle.md](state_bundle.md) — the Bundle envelope wrapping snapshots
- `../../../go-inference/docs/inference/capability.md` — `CapabilityKVSnapshot` advertises this

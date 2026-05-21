<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# kv_snapshot_blocks.go — block chunking for snapshots

**Package**: `dappco.re/go/mlx`
**File**: `go/kv_snapshot_blocks.go`

## What this is

The strategy for **chunking a KV snapshot into fixed-size blocks** so:

- Storage can hot-cache recent blocks while archiving cold blocks.
- Sleep with `ReuseParentPrefix` can share blocks between a child and its parent (identical prefix tokens → identical K/V → identical block hash → no rewrite).
- Wake can stream blocks lazily, restoring head blocks first to start generation early.
- State video encoding can address each block by `(chunk_id, frame_offset)`.

## Block size

```go
DefaultBlockSize = 256 tokens
```

256 tokens is a tuning compromise:

- Smaller blocks (64-128) → more parent-prefix reuse, more index overhead, slower restore.
- Larger blocks (512+) → fewer index entries, faster restore, less reuse for "branch from middle" cases.
- 256 hits the sweet spot for typical chat-style workloads.

Callable as a `SleepOptions.BlockSize` override per-sleep — long-form book bundles benefit from 512+, short-chat bundles from 128.

## Block layout

Each block is a contiguous KV span over `[token_start, token_start + BlockSize)`. Layout per block:

```
+-----------------+
| BlockHeader     |  layer count, token range, encoding, hash
+-----------------+
| per-layer K     |  flattened token-major
| per-layer V     |
+-----------------+
| block trailer   |  byte count, hash repeat for verification
+-----------------+
```

Hash is `blake3` of (BlockHeader + K + V) — used as the block identity for parent-reuse + cache lookup.

## Encoding per block

Block-level encoding is independent from snapshot-level encoding. A bundle can mix Q8 cold blocks (cheap storage) with native hot blocks (fast restore). The `block_cache.go` (in inference/) is the hot-tier; blocks not in cache fall through to bundle decode.

## Capture path

```go
blocks, err := captureBlocksFromSnapshot(snap, BlockSize)
```

Walks the snapshot's layers, partitions by token range, computes each block's hash, returns a `[]Block` ready to write.

## Restore path

```go
err := restoreBlocksIntoModel(model, blocks)
```

Per-block:

1. Verify hash against bundle index claim (skippable in trusted-bundle mode)
2. Decode K/V from block encoding
3. Inject into model's KV cache at the block's token range

## Block hash → identity

The hash IS the identity. Two parent/child bundles share a prefix → same blocks → same hashes → block deduplication at the storage layer.

This is what makes "1 base context + 100 divergent continuations" cheap: 100 bundles store only the divergent tails, not 100 copies of the base.

## Related

- [kv_snapshot.md](kv_snapshot.md) — snapshot format
- [kv_snapshot_index.md](kv_snapshot_index.md) — bundle index referencing blocks
- [kv_snapshot_state.md](kv_snapshot_state.md) — State chunks one block per frame range
- [block_cache.md](../inference/block_cache.md) — hot block cache
- [agent_memory.md](agent_memory.md) — Wake/Sleep that consumes blocks

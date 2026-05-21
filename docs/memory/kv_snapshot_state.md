<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# kv_snapshot_state.go — State QR-video bundle integration

**Package**: `dappco.re/go/mlx`
**File**: `go/kv_snapshot_state.go`

## What this is

The glue between `kv_snapshot_*` (the KV format) and State video store (the QR-video codec). When the bundle store is State video, KV blocks are packed into MP4 frames as QR codes; this file owns the framing strategy.

The result: an AI's runtime state shipped as a portable `.mp4` that can be scanned in by camera, dropped into a USB stick, streamed over HTTP, indexed by YouTube — see `design_coursera_for_ai_packs.md`.

## State bundle index

The State-flavoured bundle index. Adds:

- `FramesPerBlock` — how many video frames one block occupies (function of block size + QR density + error correction)
- `VideoMetadata` — frame rate, resolution, codec hint
- `IndexFrames` — if the index is embedded, which frames hold it

## Framing strategy

A block becomes N frames:

1. Block bytes are split into payloads sized for one QR code.
2. Each QR carries `(block_id, frame_offset, total_frames, payload, error_correction)`.
3. Frames are written sequentially in a single MP4 file at 24fps (default).

A 256-token Q8 block is ~256KB. At a typical QR density of ~2KB/frame, that's ~130 frames per block. A 92k-token bundle at BlockSize 256 = ~360 blocks × 130 frames = ~46k frames = ~32min of video at 24fps.

The block-cache layer ensures we don't actually decode 32 minutes of video on every wake — first wake decodes, subsequent wakes hit the cache.

## Read path

```go
idx, err := LoadStateIndex(ctx, store, indexURI)
entry, ok := idx.LookupURI(entryURI)
blocks, err := readBlocksFromState(ctx, store, entry.BlockRefs)
```

`readBlocksFromState` resolves each BlockRef → frame range → bytes via `state.RefBinaryResolver`. The State video `URIResolver` knows how to seek to a `frame_offset` and return the QR-decoded payload.

## Write path

```go
frames := encodeBlocksToStateFrames(blocks)
writer.PutBytesStream(ctx, totalSize, opts, func(w io.Writer) error {
    return encodeFramesToMP4(w, frames, framerate)
})
```

Streaming write — never materialises the whole bundle in memory. The encoder writes frames as it produces them.

## Error correction

QR codes carry their own ECC (L/M/Q/H levels). Production uses **M** (15% recovery) for portable bundles and **Q** (25%) for "scan by phone camera in poor lighting" intended bundles.

If a frame is unrecoverable (smudge on print, screen glitch during scan), the block-level hash catches it — the bundle reports "block X corrupt, skipping" and the wake fails for that block. Recovery: re-acquire the missing frames or fall back to the parent bundle.

## What this doesn't own

- The QR codec itself (State video store does).
- Video container choices (always MP4 today; future Theora/AV1 study tracked).
- YouTube-survival encoding (frame redundancy + error-correction tuning) — `design_coursera_for_ai_packs.md` future research.

## Related

- [kv_snapshot.md](kv_snapshot.md) — snapshot format
- [kv_snapshot_blocks.md](kv_snapshot_blocks.md) — blocks the frames carry
- [kv_snapshot_index.md](kv_snapshot_index.md) — base bundle index
- `pkg/memvid/` (deprecated compatibility path) — the codec
- `cmd/violet/` — sidecar that serves State wakes over Unix socket

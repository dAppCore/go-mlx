<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# kv_snapshot_index.go — bundle index

**Package**: `dappco.re/go/mlx`
**File**: `go/kv_snapshot_index.go`

## What this is

The **index** that lives alongside a bundle. Tells the wake side which blocks make up which entry, in what order, with what hashes. Without the index, a memvid bundle would be opaque — you couldn't enumerate entries or look up "the bundle for prompt X".

## Conceptual shape

```
Bundle Index
├── version
├── created_at
├── entries[]
│   ├── EntryURI ("memvid://aurelius/meditations/chapter-3")
│   ├── Title
│   ├── ParentEntryURI (optional)
│   ├── ModelIdentity + TokenizerIdentity
│   ├── PromptHash
│   ├── TokenStart, TokenCount
│   ├── BlockRefs[] (each = chunk_id + frame_offset + hash)
│   ├── Labels
│   └── Metadata
├── all_blocks[] (deduplicated — child entries reference parents)
└── trailer (signed hash of index for integrity)
```

## Why the index is separate from the bundle

Two reasons:

1. **Read-without-decode.** Walking a bundle's contents shouldn't require streaming the whole `.mp4`. The index is small (KBs); the bundle is GBs. A model picker reads the index to populate its UI.
2. **Cross-bundle linking.** Child bundles can reference parent blocks. The index records the reference; the parent bundle holds the actual bytes. No bundle is forced to be self-contained.

## Index storage

Two shapes ship:

- **Sidecar JSON** — `bundle.idx.json` next to `bundle.mp4`. Easy to read, easy to debug.
- **Embedded in QR frames** — first N frames of the memvid bundle are the index. Self-contained.

Production prefers sidecar for fast read, embedded for portable transfer.

## Operations

```go
idx, err := mlx.LoadBundleIndex(ctx, store, indexURI)
entry, ok := idx.LookupURI("memvid://aurelius/meditations/chapter-3")
idx.AddEntry(entry)
err := idx.Save(ctx, store, indexURI)
```

LookupURI is the wake-side hot path. AddEntry + Save run at sleep time.

## Deduplication

When `AddEntry` sees an entry whose parent already lives in `all_blocks`, it adds only the new (child-only) blocks. The wake side traverses the parent chain to assemble the full block list — same shape as git's commit-graph traversal.

## Compatibility check

The index records `ModelIdentity.Hash` + `TokenizerIdentity.Hash` per entry. A wake compares against the live model's identity and rejects mismatches (unless `SkipCompatibilityCheck`).

## Related

- [kv_snapshot.md](kv_snapshot.md) — snapshot format
- [kv_snapshot_blocks.md](kv_snapshot_blocks.md) — what BlockRefs point at
- [kv_snapshot_memvid.md](kv_snapshot_memvid.md) — memvid-specific framing of the index
- [agent_memory.md](agent_memory.md) — Wake/Sleep that uses LoadBundleIndex / AddEntry

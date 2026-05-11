<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# block_cache.go — KV block prefix cache

**Package**: `dappco.re/go/mlx`
**File**: `go/block_cache.go`
**Implements**: `inference.CacheService`

## What this is

The **block-prefix cache** that shares KV blocks across requests with identical prefixes. When two requests prefix-match (same system prompt, same first turn, same chat template), the second request reuses the first's prefill — instant time-to-first-token.

This is what `cache.warm` in the wider HTTP API actually warms.

## DefaultCacheBlockSize

```go
const DefaultCacheBlockSize = 128
```

128 tokens per block. Smaller than the snapshot-block size (256) because cache-share-hit-rate is sensitive to block size — smaller blocks → more chances to share a prefix mid-conversation.

## BlockCacheService

```go
type BlockCacheService struct {
    blocks    map[blockHash]cacheEntry
    diskPath  string
    mu        sync.Mutex
    // …
}
```

In-memory hot-set with optional disk-backed metadata at `BlockCacheDiskPathEnv` (env var override for the path).

## Operations

```go
svc.CacheStats(ctx)                            // current state
svc.WarmCache(ctx, CacheWarmRequest)            // prefetch a prompt's KV
svc.ClearCache(ctx, labels)                     // evict matching blocks
```

Implements `inference.CacheService` so it plugs into the OpenAI `/v1/cache/*` handlers via `register_metal_cache.go`.

## CacheStats

```go
type CacheStats struct {
    Blocks         int
    MemoryBytes    uint64
    DiskBytes      uint64
    Hits, Misses   uint64
    Evictions      uint64
    HitRate        float64
    RestoreMillis  float64
    CacheMode      string
}
```

Surfaced over `/v1/cache/stats` so monitoring can track cache health without scraping logs.

## How prefix matching works

1. Prompt is tokenised
2. Tokens are chunked into 128-token blocks
3. Each block's content hash is computed
4. For each block, the cache is queried:
   - Hit → KV bytes copied into the active model's cache at that prefix position
   - Miss → block runs prefill normally and the result is cached for future requests
5. Once first miss occurs, no further hits possible (prefix has diverged)

A common pattern hits the first N blocks (shared system prompt + few-shot examples), misses block N+1 (user-specific question), and gets ~80% of the prefill time saved.

## Cache modes

| Mode | Behaviour |
|------|-----------|
| `off` | no caching |
| `memory` | in-RAM only |
| `memory+disk` | RAM hot-set + disk cold-set (LRU between tiers) |

`MemoryPlan.PromptCache` decides default; user override via `WithCacheMode(...)` option.

## What's not cached

- Anything past block N+1 once any block has missed
- Adapter-specific blocks (different adapter → different KV → no cross-adapter share)
- Blocks where the tokenizer-template hash differs (chat-template upgrade invalidates blocks)

## Status

Production for memory-mode. Disk-mode in flight (Phase 1 parity item).

## Related

- [../memory/kv_snapshot_blocks.md](../memory/kv_snapshot_blocks.md) — same block concept, different lifetime (cache = ephemeral, snapshot = durable)
- [scheduler.md](scheduler.md) — scheduler drives cache lookups per request
- `../../../go-inference/docs/inference/contracts.md` — `CacheService` interface
- `../../../go-inference/docs/openai/services.md` — `/v1/cache/*` handlers using this
- `../../../go-inference/docs/inference/capability.md` — `CapabilityCacheBlocks` + `CapabilityCacheDisk` + `CapabilityCacheWarm` flags

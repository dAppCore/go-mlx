<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# agent_memory.go — Wake / Sleep on top of KV snapshots + memvid

**Package**: `dappco.re/go/mlx`
**File**: `go/agent_memory.go`
**Implements**: `inference/state.Session` (Wake/Sleep) — the reference implementation

## What this is

The **production Wake/Sleep/Fork** for the Metal backend. Translates the portable `state.WakeRequest` / `state.SleepRequest` contract into:

- KV-block read / write via the `kv_snapshot_*.go` family
- Memvid `.mp4` bundle encode/decode via `pkg/memvid`
- Filestore append-only logs via `state/filestore`
- Compatibility checking against `ModelIdentity` / `TokenizerIdentity`

This is the file that delivers the measured **55.2s cold-load of a 92k-token book** and **998ms warm-restore of a chapter**.

## DTOs (backend-specific extensions on top of state.*)

```go
AgentMemoryWakeOptions      // Index, IndexURI, EntryURI, Tokenizer, LoadOptions, SkipCompatibilityCheck
AgentMemoryWakeReport       // restored prefix counts + hashes for audit
AgentMemorySleepOptions     // EntryURI, BundleURI, IndexURI, parent URIs, Title, Model+ModelInfo, etc.
AgentMemorySleepReport      // written prefix counts + parent reuse stats
```

These are richer than the portable `state.WakeRequest/Result` because the Metal backend has more knobs (KV encoding, tokenizer handoff, native-vs-float32). The portable shape comes back at the call boundary — `Session.WakeState` / `Session.SleepState` take/return the portable types and adapt internally.

## Wake path

```
state.WakeRequest
   ↓
AgentMemoryWakeOptions    (translate)
   ↓
Resolve EntryURI in KVSnapshotMemvidBundleIndex
   ↓
Read bundle from Store     (memvid, filestore, or in-memory)
   ↓
Decode KV blocks            (kv_snapshot_blocks.go)
   ↓
Compatibility check vs current model + tokenizer  (skippable)
   ↓
Restore into live metal.Model KV cache
   ↓
AgentMemoryWakeReport       (counters + hashes)
   ↓
state.WakeResult            (project)
```

## Sleep path

```
state.SleepRequest
   ↓
AgentMemorySleepOptions     (translate)
   ↓
Capture KV from live model  (kv_snapshot.go — Q8 or native or float32)
   ↓
Chunk to blocks             (BlockSize, ReuseParentPrefix logic)
   ↓
Write bundle to Store        (memvid: encode QR frames; filestore: append records)
   ↓
Update bundle index          (kv_snapshot_index.go)
   ↓
AgentMemorySleepReport      (written + reused counters)
   ↓
state.SleepResult           (project)
```

## ReuseParentPrefix

The optimisation that makes append-mode bundles cheap. When a session sleeps with `ParentEntryURI` set + `ReuseParentPrefix: true`:

1. The bundle index records the parent.
2. KV blocks identical to the parent's blocks (by hash) are **not re-written** — the new bundle's KV refs point at the parent's blocks.
3. Only the delta — new tokens generated since wake — is written.

This is what makes "long-running session with periodic sleep" tractable. A 92k-token book bundle is ~10GB raw, but the next sleep after generating 200 tokens only writes those 200 tokens' KV.

## Compatibility check

Defaults on. Compares `WakeRequest.Model.Hash` / `Tokenizer.Hash` against bundle's stored identity:

- Match → restore proceeds
- Mismatch → return error with diff fields
- `SkipCompatibilityCheck: true` → bypass (used for explicit cross-version forensics)

Tokenizer mismatch is the more common failure — same model arch, different chat template hash. Bundles built before a chat-template upgrade can't be restored into the new tokenizer without warping the prompt boundary.

## Forker

The same file implements `state.Forker.ForkState` — spawns a **new** metal.Model from a bundle, leaving the calling session untouched. Used by speculative-rollout scenarios (Vi training, agent branching, "what if I had asked X instead") where you want two divergent continuations from the same prefix.

## Encoded probe events

Wake and Sleep emit probe events at every stage — bundle decode start/end, block read with hash, KV restore with prefix tokens, sleep block write with parent-reused count. Consumers (core/ide memory panel) render real-time progress without scraping internal logs.

## Used by

- `cmd/violet/` — sidecar exposes Wake/Sleep/Fork over Unix socket
- `core/ide` (planned) — agent inspector panel calls Wake when user selects a bundle
- `go-ai/ai/book_state_demo.go` — BookState wake before teacher call
- Vi training scripts — sleep training checkpoints + wake-and-continue

## Measured

| Operation | Bundle size | Latency |
|-----------|-------------|---------|
| Wake — chapter (warm cache) | ~500MB | 998ms |
| Wake — full book (warm cache) | ~10.5GB | 2.15s |
| Wake — full book (cold runner) | ~10.5GB | 55.2s |
| Sleep — incremental (ReuseParent on) | 200-token delta | <1s |

Cold load = process startup + memvid decoder warm + first-time block decode. Warm load = re-restore from already-decoded blocks (block cache hit). The "from cold runner, ever, in 55s" measurement is the AI-cognition-as-filesystem-object thesis made real — see `memory_plan_for_lethean.md` in core/plans.

## Related

- [kv_snapshot.md](kv_snapshot.md) — capture / restore the raw KV bytes
- [kv_snapshot_blocks.md](kv_snapshot_blocks.md) — chunk strategy
- [kv_snapshot_index.md](kv_snapshot_index.md) — bundle index
- [kv_snapshot_memvid.md](kv_snapshot_memvid.md) — memvid integration
- [medium.md](medium.md) — runtime Store abstraction
- [state_bundle.md](state_bundle.md) — Bundle encode/decode
- `../../../go-inference/docs/state/agent_memory.md` — the portable contract this implements

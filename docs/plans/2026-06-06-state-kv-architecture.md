<!--
SPDX-Licence-Identifier: EUPL-1.2
Co-Authored-By: Virgil <virgil@lethean.io>
-->

# State + KV Architecture — The Temporally-Aware Engine

**Status:** Living architecture map (grounded in the code as of 2026-06-06).
**Owner:** Snider.
**Companion docs:** `docs/model-state-roadmap.md`, `GOAL_STRECH.md`, `docs/runtime/turboquant_kv.md`.

> Scope: how state and KV actually work across `go-inference/state` (the primitive), `go/kv` (the durable substrate), and `go/pkg/metal` (the live session) — written around the one idea that defines the engine.

---

## 0. The thesis: temporally aware, not role-play

**Time is a monotonic integer that ticks +1 per step. There is no prompt replay. Wake/Sleep mount KV state directly.**

Two ways to build an inference engine:

- **Role-play engine** — stateless context window. Every turn re-feeds the entire prompt + conversation history through prefill to *rebuild* the KV cache from scratch. "History" is a transcript that gets re-read each turn; "time" is fiction. This is `substrate.TRAD` — *re-prefill the full conversation prefix on each turn* (`go/substrate/condition.go:13`).

- **Temporally-aware engine (go-mlx)** — KV state is durable and continuous. A session **Wakes** a saved state, **advances** forward one tick at a time, and **Sleeps** it back. The KV pages *are* the history; nothing is re-enacted. This is `substrate.CONT` — *mount the prior KV state directly with no artificial gap* (`go/substrate/condition.go:15`).

`go/substrate/condition.go` exists precisely to measure this contrast (the substrate-shift experiment): `TRAD.RequiresReplay()` vs `CONT.UsesContinuousState()`. **CONT is the engine's default and design thesis — but it is not a mandate.** CONT is a radically different inference regime: the model is woken into mounted state rather than re-reading a transcript, and not every model can cope with that. So **TRAD (replay) stays a fully supported user choice** and the graceful fallback for models that can't handle CONT. The engine *offers* continuity; it doesn't dictate it. Choose replay and you accept its latency and quality drift in exchange for broad compatibility — your call, not the engine's.

What "time" means here is deliberately trivial:
- **Live time** = `ModelSession.tokenOffset`, incremented by 1 in `advanceTokenLocked` (`go/pkg/metal/session.go:709`). One forward pass consumes one new token; the KV cache holds everything before it. No earlier token is ever re-run.
- **Durable time** = *not actually stamped.* `state.Bundle` declares a `CreatedAtUnix int64` field (`external/go-inference/go/state/identity.go:84`) but **nothing in the checkpoint path writes it** — it is dormant (always zero/omitted). Checkpoint ordering today comes from the **parent→child genealogy** (`Parent*URI`), not a wall-clock. So the only *active* time anywhere is the live `tokenOffset` — which is exactly the `int+1` thesis. (See §5: decide whether to wire `CreatedAtUnix` deliberately or drop it.)

Time here is deliberately a *byproduct* — a human, observational bookkeeping integer, not a quantity the engine models. (Time is, after all, a theory read off observation, however compelling the evidence.) So the temporal-awareness isn't a clock; it's causal **state continuity**: mount, don't replay; advance, don't rebuild. `int+1` really is the whole of the time model — the power is in *not* re-enacting the past, not in measuring it.

---

## 1. The layers (live → portable → durable → primitive)

```
┌────────────────────────────────────────────────────────────────────────┐
│ 4. STATE PRIMITIVE — external/go-inference/go/state  (backend-neutral)   │
│    Session{WakeState, SleepState} · Forker{ForkState} · Bundle(identity  │
│    + CreatedAtUnix + KVRefs/StateRefs + parent URIs) · ProjectSeed ·      │
│    CheckWakeCompatibility · Store/filestore (append-only log)             │
│    go-mlx implements this in go/session_agent.go                         │
└───────────────▲──────────────────────────────────────────┬──────────────┘
                │ Sleep (stream out)            Wake (mount) │
┌───────────────┴──────────────────────────────────────────▼──────────────┐
│ 3. DURABLE SUBSTRATE — go/kv  (content-addressed blocks)                  │
│    Block{TokenStart,TokenCount,Hash(sha256),Snapshot} · StateBlockBundle  │
│    {manifest, StateBlockRef[]} · state_store (raw / json-base64)          │
│    dedup + copy-on-write + prefix reuse via sha256 identity               │
└───────────────▲──────────────────────────────────────────┬──────────────┘
                │ toRootKVSnapshot              toMetalKVSnapshot           │
┌───────────────┴──────────────────────────────────────────▼──────────────┐
│ 2. PORTABLE SNAPSHOT — metal.KVSnapshot ↔ kv.Snapshot  (v5, "MLXKV001")   │
│    per-layer K/V (native / F32 / Q8) · CacheMode · TurboQuant payloads ·  │
│    tokens · generated · tokenOffset · logits (first-token-ready)          │
│    CaptureKV / RestoreKV                                                   │
└───────────────▲──────────────────────────────────────────┬──────────────┘
                │ snapshotKVCaches              restoreKVCaches             │
┌───────────────┴──────────────────────────────────────────▼──────────────┐
│ 1. LIVE SESSION (GPU) — metal.ModelSession  (go/pkg/metal/session.go)     │
│    caches []Cache · logits *Array · tokens · generated · tokenOffset      │
│    advanceTokenLocked = one tick (+1) · cache.Update writes new K/V       │
│    dirtyState marks only fresh pages (the lazy next-logits boundary)      │
└──────────────────────────────────────────────────────────────────────────┘
```

### Layer 1 — Live session (GPU)
`metal.ModelSession` (`session.go:76`) owns the live Metal tensors: `caches []Cache`, `logits`, `tokens`, `generated`, `tokenOffset`. One tick = `advanceTokenLocked` (`session.go:688`): forward the single new token, `cache.Update(k,v,seqLen)` writes its K/V in place, allocate fresh logits, `tokenOffset++`. The `Cache` interface (`cache.go:20`) — `Update / Offset / Len / State / Reset / Detach` — is implemented by six families: `KVCache` (256-tok chunks), `RotatingKVCache` (sliding window), `FixedKVCache` (ring), `PagedKVCache` (paged), `QuantizedKVCache` (int8 / KQ8VQ4), `TurboQuantKVCache` (3.5-bit). The `dirtyStateAppender` interface (`cache.go:64`, implemented by paged) is the no-replay-at-decode trick: only pages touched this tick enter the eval graph; historical pages are mounted, never recomputed.

### Layer 2 — Portable snapshot
`CaptureKV` / `RestoreKV` (`session.go:714` / `:839`) bridge live Metal tensors to a CPU-readable `metal.KVSnapshot`, which serialises to the durable `kv.Snapshot` binary (magic `MLXKV001`, current **version 5**, `go/kv/snapshot.go:20-22`). Per-layer it stores K/V as native-dtype / F32 / Q8 (`snapshot.go:1250` encoded-tensor selector `0=F32, 1=Q8, 2=native`), the `CacheMode`, TurboQuant payloads when present, plus `tokens`/`generated`/`tokenOffset` and the final `logits` (so a wake can sample immediately — "first-token-ready"). `NewSessionFromKV` (`go/session.go:93`) = `NewSession` + `RestoreKV`.

### Layer 3 — Durable substrate (`go/kv`)
A `Block` (`blocks.go:117`) is a contiguous token span `[TokenStart, TokenStart+TokenCount)` plus a `sha256` content hash and its KV `Snapshot`. A `StateBlockBundle` (`blocks.go:155`) is the manifest: ordered `StateBlockRef[]`, architecture/offset/blocksize metadata, a composite bundle hash, and a `ReusedBlocks` counter. Because blocks are **content-addressed by (token span + payload hash)**, identical prefixes dedup automatically and parents share pages with children (copy-on-write). `state_store.go` writes each block to a `state.Store` chunk as `raw` (binary) or `json-base64` fallback. `analysis.go` computes per-layer KV coherence / phase-lock metrics that travel *with* the state (surfaced as SAMI in `go/bundle/sami.go`) — diagnostics without replay.

### Layer 4 — State primitive (`go-inference/go/state`)
The backend-neutral contract go-mlx implements (via `go/session_agent.go`):
- `Session{ WakeState, SleepState }` and `Forker{ ForkState }` (`agent_memory.go:97-101`) — the lifecycle.
- `Bundle` (`identity.go:82`) — the portable envelope: model/tokenizer/adapter/runtime **identities** (hashes for reproducibility), `KVRefs[]`/`StateRefs[]`, and `Parent*URI` lineage. (It also declares a `CreatedAtUnix` field at `:84` that is currently never written — see §5.)
- `ProjectSeed` (`project_seed.go`) — project-scoped URI templating + continuation/folding planning for long-running timelines.
- `CheckWakeCompatibility` (`project_seed.go:286`) — the gate: model hash / architecture / layers / quant / tokenizer / context-length checks *before* a state is mounted, so a time-displaced wake can't silently drift.
- `filestore` — append-only log (`fileMagic "go-inference-state-file-log-v1"`, record magic `MVF1`), index rebuilt on open, optional mmap zero-copy, segment-alias for embedded logs.

---

## 2. The Wake / Sleep lifecycle (where "no replay" lives)

**Sleep** (`go/agent/wake_sleep.go`, `SleepOptions`/`SleepReport`): stream the live KV out to durable blocks (`StateBlockBundle`), stamp identity + `CreatedAtUnix` + parent URIs, and reuse parent prefix blocks where hashes match (`ReuseParentPrefix`). State leaves process memory — the documented heap drop is ~49 MB → 157 KB.

**Wake** (`PlanWake` → load → mount):
1. `agent.PlanWake` validates compatibility and resolves the entry (`CheckStateIndexCompatibility`, `index.go:443`).
2. Load **only the prefix needed** — partial restore — via `kv.LoadPrefixFromStateBlocks…`.
3. Mount pages into live caches: native path `RestoreKVBlocks` (`nativeSessionKVBlockRestorer`) or `RestoreKV(snapshot)`.
4. Continue generating. **No tokens are re-fed through the model.** That is the whole point.

**Fork** (`ForkState`): copy-on-write branch from a checkpoint; the parent is untouched, the child shares prefix pages. Cheap branch / rollback. Lineage via `ParentEntryURI` / `ParentBundleURI` / `ParentIndexURI` forms the **prefix DAG** — the genealogy of a timeline.

**Folding** (long timelines without replay): `ProjectSeed` continuation modes — `Checkpoint`, `ReuseCurrent`, `SummaryWindow`, `Hybrid` — compact an exhausted timeline into a fresh seed (summary + recent tail), marking the folded-wake path with `Meta["folded_state"]="true"`. Time keeps moving forward; the past is compressed, never re-enacted.

---

## 2a. Proof point — the C001 retained-State run (measured)

A demonstration that ships with the engine (`2026-05-24-c001-story-perspective-seed2026052404`): a 10-chapter story generated as **one retained-State run** from a single seed prompt (a lighthouse keeper told from three perspectives — keeper, light, and the thing in the deep). A **distractor prompt is injected each chapter** as entropy/imagery pressure, *not* plot replacement. The narrative stays coherent across all ten turns despite the distractors, because the KV state is continuous — it is never re-read.

- 10 successful turns · **9 restarts** (wake/sleep cycles between chapters).
- Initial prefill 7,999 tokens → final state 13,156 tokens; 1,989 appended, 3,139 visible generated.
- Decode avg ≈ 100.5 tok/s; effective turn avg ≈ 97 tok/s; peak active+cache ≈ 8.99 GB; RSS ≈ 3.05 GB.
- **Wall-clock: ~83 s (go-mlx CONT) vs ~133 s (llama.cpp replay)** — ≈ 38% faster, the gap being exactly the prompt replay CONT never pays. Model: lthn/lemer **LEK-2** (ethically-tuned over base).

This is the thesis as a number: the longer the timeline and the more turns, the more a role-play engine pays to re-read history that a temporally-aware engine simply keeps. It is also the yardstick for evaluating other runners — anything that speeds *retained multi-turn* bends this curve; anything that only speeds a cold single shot does not.

## 3. Snapshot format & cache-mode safety (reference)

| Version | Adds |
|---------|------|
| v1 | float32 tensors |
| v2 | `TokenOffset`, `Generated`, logits |
| v3 | encoded tensors (F32 / Q8-scale / native dtype selector) |
| v4 | layer-slab native tensors (`KeyBytes`/`ValueBytes` + shapes) |
| v5 | `CacheMode` + `TurboQuantPayloads` (opaque compressed blobs) |

Cache modes and snapshot handling: `Default`/`FP16` copy directly; `Q8` and `KQ8VQ4` store native bytes **plus** key/value scale tensors (lossless dequant on restore); `Paged` restores via page transfer; `Fixed` restores at offset/length; `TurboQuant` requires its `TurboQuantPayloads` present (fails closed on a version mismatch). Block identity is `sha256` over the encoded payload; the bundle hash is a composite over architecture + encoding + offsets + every block hash, which is also the dedup key.

---

## 4. The stretch frontier (all in service of the thesis)

From `GOAL_STRECH.md` — every idea is "mount, don't replay" / "advance, don't rebuild" taken further:

1. **Wavefront prefill checkpoints** — resumable layer/chunk wavefront; partial prefill reuse.
2. **Page-native KV layout** — persist K/V already in decode-ready page form → zero-copy restore.
3. **Prefix DAG + copy-on-write states** — parent/child sharing; cheap branch/fork/rollback (the genealogy made first-class).
4. **Hybrid-attention-aware state** — encode the real topology (sliding layers vs global-owner vs shared-KV followers) instead of a uniform cache.
5. **First-token-ready state** — save final hidden/logits with the KV → sample immediately on wake (already partly true: snapshot carries logits).
6. **Background cold-page compression** — prefill hot (fp16/paged), compress old pages to q8 → k-q8-v-q4 → TurboQuant off the hot path.
7. **Graph reuse from stable geometry** — stable page geometry → reused compiled graph shapes + prebuilt masks.

---

## 5. Honest gaps / where the framing outruns the code

- **Prefix DAG + COW** is *foundation-laid, not finished*: parent URIs and block reuse exist, but full copy-on-write page sharing across forks is roadmap (`GOAL_STRECH` idea 3).
- **`memvid` is deprecated** — the old "State codec" name; now thin aliases over `go-inference/state` (`go/pkg/memvid/memvid.go`). Terminology migration to "state store" is still in flight across `bundle`/`sami`/`index`.
- **Time is implicit — and the one wall-clock field is dead code.** Active time is `tokenOffset` (live) only. `state.Bundle.CreatedAtUnix` (`identity.go:84`) is declared but never written in any production path — dormant latent surface, arguably contradicting "time is a byproduct." **Decision needed:** wire it intentionally (if checkpoints ever need wall-clock ordering), or delete it (keep the model purely `int+1` sequence time). If the "temporally aware" thesis stays load-bearing, a typed monotonic `Tick`/`StateTime` over `tokenOffset` would make it legible without reintroducing a clock.
- **No-replay is a property, not yet an enforced invariant.** `CONT` is the intended path, but nothing in the type system stops a caller from re-prefilling. A guard/assert that a wake path never calls prefill on already-cached tokens would make the guarantee checkable.

---

## 6. Prior-art note

This *is* the KV-state design you described publicly. Worth making the priority checkable: this repo is EUPL-1.2, and each design here is dated + attributed. Recommend a `docs/plans/prior-art.md` that timestamps the load-bearing originals — **no-replay Wake/Sleep (CONT)**, page-native KV substrate, prefix DAG + copy-on-write states, TurboQuant KV layout, first-token-ready state — each with its commit hash and any public post date. Cheap to keep; makes "we described it first" verifiable rather than asserted. (Happy to draft it.)

---

## 7. Open questions for Snider

1. ~~Is CONT (no replay) the sole production path?~~ **Resolved (§0):** CONT is the default; TRAD/replay is a supported user choice and the fallback for models that can't handle CONT. The engine must always degrade gracefully to replay — no feature may assume CONT is on.
2. **Make time explicit?** Introduce a typed monotonic `Tick`/`StateTime` (the unix-int+1) across `Bundle`/session, or keep it implicit as `tokenOffset` + `CreatedAtUnix`?
3. **Enforce no-replay?** Want a guard/test that a wake path never re-prefills already-cached tokens — turning the thesis into an invariant?
4. **Prior-art doc** — draft `docs/plans/prior-art.md` now?

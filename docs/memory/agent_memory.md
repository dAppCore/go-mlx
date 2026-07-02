<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# session_agent.go — Wake / Sleep / Fold on top of KV snapshots + State

**Package**: `dappco.re/go/mlx`
**File**: `go/session_agent.go`
**Implements**: `inference/state.Session` (Wake/Sleep) — the reference implementation

## What this is

The **production Wake/Sleep/Fork/Fold** path for the Metal backend. Translates the portable `state.WakeRequest` / `state.SleepRequest` contract into:

- KV-block read / write via the `kv_snapshot_*.go` family
- State video `.mp4` bundle encode/decode via State video store
- Filestore append-only logs via `state/filestore`
- Compatibility checking against `ModelIdentity` / `TokenizerIdentity`

This is the file that delivers the measured **55.2s cold-load of a 92k-token book** and **998ms warm-restore of a chapter**.

## DTOs (backend-specific extensions on top of state.*)

```go
AgentMemoryWakeOptions      // Index, IndexURI, EntryURI, Tokenizer, LoadOptions, SkipCompatibilityCheck
AgentMemoryWakeReport       // restored prefix counts + hashes for audit
AgentMemorySleepOptions     // EntryURI, BundleURI, IndexURI, parent URIs, Title, Model+ModelInfo, etc.
AgentMemorySleepReport      // written prefix counts + parent reuse stats
AgentMemoryFoldOptions      // exhausted checkpoint options plus summary/tail folded-state prompt
AgentMemoryFoldReport       // checkpoint and folded-state reports plus byte accounting
```

These are richer than the portable `state.WakeRequest/Result` because the Metal backend has more knobs (KV encoding, tokenizer handoff, native-vs-float32). The portable shape comes back at the call boundary — `Session.WakeState` / `Session.SleepState` take/return the portable types and adapt internally.

## Wake path

```
state.WakeRequest
   ↓
AgentMemoryWakeOptions    (translate)
   ↓
Resolve EntryURI in State bundle index
   ↓
Read bundle from Store     (State video, filestore, or in-memory)
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
Write bundle to Store        (State video: encode QR frames; filestore: append records)
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

## Fold path

When a retained session reaches its live context budget, `Model.FoldAgentMemory`
creates the summary-plus-tail transition:

```
exhausted ModelSession
   ↓
SleepAgentMemory(checkpoint)       // exact exhausted KV state for audit/replay
   ↓
Model.NewSession()
   ↓
PrefillChunks(summary + recent tail)
   ↓
SleepAgentMemory(folded)           // fresh compacted state with parent lineage
   ↓
AgentMemoryFoldReport              // checkpoint + folded refs and byte counts
```

The folded index entry is labelled `folded-state` and records
`folded_state=true`, `folded_from_entry_uri`, `summary_bytes`,
`recent_tail_bytes`, and `folded_prompt_bytes` in metadata. The exhausted
checkpoint remains available for exact continuation or forensics, while future
turns wake the smaller folded state.

Folded entries are intentionally treated as compact semantic state, not as a
large raw K/V restore. When a wake target is labelled `folded-state` and its
prefix is within the compact-state budget, the Metal backend reads the folded
token prefix from the state file and prefills that small state into a fresh
session. The wake report records `restore_strategy=folded-prefill`. Larger
non-folded entries continue to use the K/V block restore path.

The `state-ramp-profile` benchmark can exercise this lifecycle directly with
`-fold-store <path>`. When the live state reaches its configured compaction
threshold, the report includes the checkpoint and folded
`SleepReport`, folded wake latency, and an optional folded wake/continue turn.
Pass `-fold-summary-file` and `-fold-tail-file` for semantic compaction; without
them the harness uses a metric-only lifecycle summary so the state transition is
measurable but not a useful agent memory.

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

## CLI: `generate -state`

`lthn-mlx generate -state <name>` (`cmd/mlx/generate.go`) is the simplest consumer of this Wake/Sleep path — a no-prompt-replay conversation turn from the command line, backed by the same `WakeAgentMemory`/`SleepAgentMemory` session methods documented above (via `agent.WakeOptions`/`agent.SleepOptions`):

```
lthn-mlx generate -state chat1 -prompt "Hello, who are you?" <model>
lthn-mlx generate -state chat1 -prompt "And what did I just ask you?" <model>
```

Each invocation is one turn: if `chat1` already exists in the store it is woken (KV restored from `.kv` blocks, no re-prefill of prior turns) and only the new turn is appended; otherwise the prompt opens a fresh session. After generation the session sleeps back to the store, so the next invocation picks up where this one left off. The `-native` flag routes the same `-state` turn loop through the no-cgo native engine (`pkg/model` + `pkg/native`) instead of the cgo Metal engine.

**Chat-framed by default.** Each turn is rendered through the model's chat template rather than treated as raw completion: a fresh session gets the full template from empty history (`FormatChatPrompt` — BOS, optional system/thinking preamble, the user turn, the assistant opener); a woken session gets only the new turn (`FormatChatContinuation` — closes the previously open assistant turn, renders the new user turn, reopens the assistant header) with no replay of the retained prefix. This mirrors `serve`'s stateless-request conversation continuity (`conversation_continuity.go`).

**`-raw` opts out of chat-framing.** With `-raw`, the prompt prefills or appends byte-for-byte with no template and no assistant opener — the original low-level completion-loop turn, useful as a token-loop instrument when you want to control the exact bytes going in. `-raw` is ignored when `-state` is not set.

**Named slots.** `-state <name>` resolves to entry URI `mlx://agent/<name>` (index URI `mlx://agent/<name>/index`) in the state store — one name is one durable conversation slot, reusable across process runs. `-state-store <path>` overrides the store file; the default is `~/Lethean/data/state/agent.kv`.

## Used by

- `lthn-mlx generate -state` — CLI conversation-turn loop, chat-framed by default, named store slots (see above)
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

Cold load = process startup + State decoder warm + first-time block decode. Warm load = re-restore from already-decoded blocks (block cache hit). The "from cold runner, ever, in 55s" measurement is the AI-cognition-as-filesystem-object thesis made real — see `memory_plan_for_lethean.md` in core/plans.

## Related

- [kv_snapshot.md](kv_snapshot.md) — capture / restore the raw KV bytes
- [kv_snapshot_blocks.md](kv_snapshot_blocks.md) — chunk strategy
- [kv_snapshot_index.md](kv_snapshot_index.md) — bundle index
- [kv_snapshot_state.md](kv_snapshot_state.md) — State integration
- [medium.md](medium.md) — runtime Store abstraction
- [state_bundle.md](state_bundle.md) — Bundle encode/decode
- `../../../go-inference/docs/state/agent_memory.md` — the portable contract this implements

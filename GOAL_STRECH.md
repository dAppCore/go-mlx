<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# go-mlx State-Store Stretch Goal

> **For agentic workers:** this is a stretch/R&D brief, not the active
> production gate. Keep `GOAL.md` as the source of truth for accepted work.
> Use this file when investigating state-store-driven performance ideas that
> may help go-mlx close the gap with faster backends such as go-rocm.

## Goal

Use the state store as a low-level, page-addressed, layer-aware KV substrate
rather than only as a saved prompt-cache artifact. The intent is not to bypass
causal dependencies. The intent is to expose stable cache pages, partial
prefill progress, shared prefixes, and reusable Metal/MLX graph shapes so the
runtime can avoid repeat work and schedule the unavoidable work better.

The first success criterion is evidence, not optimism: each idea below needs a
small focused prototype, a same-prompt control row, memory numbers, and a clear
answer about whether the state-store abstraction enables something the normal
temporary-array path cannot.

## Ground Rules

- Do not split a fresh prompt into independent parallel chunks and concatenate
  K/V as if causal attention did not exist. A later chunk still depends on
  earlier same-layer K/V and prior-layer hidden states.
- Treat prefill as a wavefront. Parallelise or pipeline only where layer/chunk
  dependencies are satisfied.
- Keep state files portable and versioned. A restored state must fail clearly
  if cache layout, dtype, quantisation, layer ownership, model hash, or prompt
  hash is incompatible.
- Do not benchmark this lane with broad paged-cache sweeps. Use focused
  one-shape commands and watch MLX active/cache memory.
- Use workspace-aware verification commands. Do not set `GOWORK=off` for this
  lane unless a separate release gate explicitly asks for standalone module
  resolution.

## Idea 1: Wavefront Prefill Checkpoints

**Hypothesis:** prefill can be represented as a resumable layer/chunk wavefront,
where each completed dependency-valid tile is written to the state store as soon
as its K/V and hidden outputs are valid.

Useful if it enables:

- Resuming an interrupted 30k-100k prefill without starting over.
- Sharing partial prefill progress between agents or branches.
- Scheduling Metal command buffers around completed state pages.
- Measuring exactly where time is spent by layer, chunk, and cache owner.

Initial implementation shape:

- [ ] Define a `PrefillTile` metadata shape: model hash, prompt hash, layer,
  cache owner, chunk token range, dtype, cache mode, hidden-state availability,
  and dependency parent tile IDs.
- [ ] Add a dry-run planner that emits the legal wavefront order for Gemma 4
  without writing state.
- [ ] Prototype writing completed K/V tiles for one native Gemma 4 E2B prompt
  shape, then resume from the last complete tile after an intentional stop.
- [ ] Benchmark against ordinary chunked prefill on the same 30k prompt.

Acceptance evidence:

- Same generated greedy output as ordinary prefill.
- Restore/resume avoids replaying already completed tiles.
- State metadata makes the dependency graph auditable.

## Idea 2: Page-Native KV Layout

**Hypothesis:** restore gets cheaper if the state store persists K/V in the same
page layout the decode kernels want, instead of saving generic arrays that must
be reshaped, copied, coalesced, or retyped after load.

Useful if it enables:

- Zero-copy or low-copy restore for paged K/V.
- Direct hydration of layer/cache-owner pages.
- Stable page sizes for native Metal kernels.
- Cleaner interop with future TurboQuant pages.

Initial implementation shape:

- [ ] Document the exact current Gemma 4 K/V physical layouts for `paged`,
  `fp16`, `q8`, `k-q8-v-q4`, and planned `turboquant`.
- [ ] Define a page-native state manifest: layer, cache owner, page index,
  token span, dtype, quantisation mode, RoPE-applied K flag, normalised K/V
  flag, and shared-KV reference count.
- [ ] Prototype state restore that returns page handles in decode-ready order.
- [ ] Compare restore time, active memory, and first-token latency against the
  current prompt-cache restore.

Acceptance evidence:

- Restore keeps the same model output.
- Restore time or memory pressure improves on 30k-40k retained workflows.
- Page metadata survives compact/sleep/wake cycles.

## Idea 3: Prefix DAG And Copy-On-Write States

**Hypothesis:** project memory, system prompt, repo map, and conversation
history should be content-addressed parent states. New turns and agent branches
should append child deltas without cloning base K/V pages.

Useful if it enables:

- Multiple agents sharing the same expensive prefix.
- Cheap branch/fork/rollback operations.
- State compaction that preserves exact continuation when wanted.
- Clear separation between durable memory and transient turn context.

Initial implementation shape:

- [ ] Define parent/child state manifest links by model hash, prompt hash,
  tokenizer hash, cache mode, and final token offset.
- [ ] Add copy-on-write page ownership for appended child turns.
- [ ] Add a state auditor that reports shared pages, private pages, and total
  physical bytes.
- [ ] Run a three-branch agent prompt where all branches share one 30k parent.

Acceptance evidence:

- Branches produce the same output as independently restored full states.
- Physical state bytes scale with deltas, not with full prompt length times
  branch count.
- Parent state remains immutable after child generation.

## Idea 4: Hybrid Attention State Exploitation

**Hypothesis:** Gemma 4 local/sliding layers and global/shared-KV layers should
not be represented as one uniform cache family. The state store can encode the
real attention topology and let decode restore only what each layer needs.

Useful if it enables:

- Sliding layers storing bounded recent windows.
- Global owner layers storing long pages.
- Shared-KV layers referencing owner pages instead of duplicating state.
- Cleaner memory planning for long contexts.

Initial implementation shape:

- [ ] Extend state metadata with attention family: sliding, global owner,
  shared global follower, or ordinary full cache.
- [ ] Record per-layer window bounds and shared-KV owner IDs.
- [ ] Restore a mixed topology state and prove follower layers read owner
  pages instead of cloned K/V.
- [ ] Compare memory and decode against uniform full-cache restore.

Acceptance evidence:

- Long-context state size reflects real Gemma 4 topology.
- No output drift from topology-aware restore.
- Memory planner can explain why each layer is retained, bounded, or shared.

## Idea 5: First-Token-Ready State

**Hypothesis:** a useful state file should optionally save more than K/V. It
can save final hidden/logits or enough suffix state to sample the next token or
start MTP without replaying the retained prefix.

Useful if it enables:

- Wake and immediately sample the next token.
- Attached Gemma 4 assistant MTP without replaying a suffix just to recover
  target hidden state.
- Better first-token latency reporting.
- Cleaner handoff between prompt-cache restore and generation.

Initial implementation shape:

- [ ] Define optional `FinalHidden` and `FinalLogits` state sections with model
  hash, token offset, dtype, and cache compatibility metadata.
- [ ] Add fail-closed validation when sampling settings, model revision, or
  cache layout make saved logits unsafe.
- [ ] Store final hidden for a retained E2B prompt and use it to start
  `gemma4_assistant` drafting.
- [ ] Compare first-token latency against KV-only restore plus suffix replay.

Acceptance evidence:

- Same greedy next token as normal restore.
- First-token latency improves or the added state size is rejected with data.
- MTP attachment can consume restored hidden without full-prefix replay.

## Idea 6: Background Compression

**Hypothesis:** the runtime can prefill into a high-quality hot format, then
compress cold state pages in the background. Recent pages stay fp16/paged while
old long-prefix pages move to q8, k-q8-v-q4, or TurboQuant.

Useful if it enables:

- Lower long-context memory after wake.
- Quality-preserving compression of cold prefix pages.
- Per-page downgrade/upgrade policy based on recency and attention family.
- TurboQuant experiments without forcing all pages into the same format.

Initial implementation shape:

- [ ] Add page versioning so a state can mix fp16, q8, k-q8-v-q4, and
  TurboQuant pages.
- [ ] Define a background compression queue that operates only after pages are
  immutable and dependency-complete.
- [ ] Start with q8/k-q8-v-q4 cold-page conversion before TurboQuant.
- [ ] Add a TurboQuant 3.5-bit cold-page experiment after the implementation
  note from `GOAL.md` exists.

Acceptance evidence:

- No output drift on greedy smoke prompts after cold-page conversion.
- Memory decreases after background compression completes.
- Decode does not regress enough to erase the memory win.

## Idea 7: Kernel And Graph Reuse From Stable State Geometry

**Hypothesis:** stable state page geometry can make Metal/MLX graph and kernel
reuse more predictable. The runtime can present repeated decode with the same
page shapes, masks, owner maps, and dtype layouts instead of arbitrary temporary
arrays each turn.

Useful if it enables:

- Reused compiled graph shapes for common retained workflows.
- Prebuilt masks and cache-owner maps.
- Fewer host-side shape decisions in the token loop.
- Better command-buffer scheduling around known state geometry.

Initial implementation shape:

- [ ] Record state geometry fingerprints: page size, token span, layer count,
  cache owner map, dtype map, mask family, and attention topology.
- [ ] Add a geometry cache that stores reusable mask/state descriptors for one
  E2B retained workflow.
- [ ] Benchmark decode with and without geometry reuse on the same restored
  state.
- [ ] Trace Go-side graph construction and MLX eval buckets before and after.

Acceptance evidence:

- Graph construction or first-token setup time decreases measurably.
- No output drift.
- Geometry cache invalidation is explicit when state shape or model changes.

## Measurement Plan

Use one narrow prompt shape at a time:

```bash
cd /Users/snider/Code/core/go-mlx
env GOCACHE=/private/tmp/codex-go-mlx-cache MLX_METALLIB_PATH=/Users/snider/Code/core/go-mlx/dist/lib/mlx.metallib go test ./go/internal/metal -run 'TestPromptCache|TestModelSession|TestState' -count=1
```

For performance claims, record JSON under `docs/runtime/` with:

- model path and exact revision/hash
- prompt token count and prompt hash
- context length and output budget
- cache mode and state-store layout version
- prefill time, restore time, first-token time, raw decode, wall time
- peak MLX active/cache memory and process RSS
- generated token counts and quality flags
- same-shape baseline without the stretch feature

## Non-Goals

- This file does not claim fresh 30k prompts can be split into independent
  chunks and recombined without respecting causal dependencies.
- This file does not replace `GOAL.md`.
- This file does not promote speculative/MTP or TurboQuant defaults.
- This file does not require broad benchmark sweeps. Keep experiments narrow
  until memory behaviour is understood.

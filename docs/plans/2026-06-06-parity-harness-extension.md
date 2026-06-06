<!--
SPDX-Licence-Identifier: EUPL-1.2
Co-Authored-By: Virgil <virgil@lethean.io>
-->

# Parity-Harness Extension — Safety Net for Gated Decode-Tail & Spec-Decode Work

**Status:** Draft spec for review.
**Last updated:** 2026-06-06.
**Owner:** Snider.
**Companion:** `docs/plans/2026-06-06-competitive-runner-research.md` (Tier C items C1–C5).

> Purpose: define the parity guard that must exist **before** any change touches the sampler / eval boundary / speculative-decode path — the area where probes have repeatedly regressed. This is the "extend the retained-session state-advance parity guard first" rule from `TODO.md`, written out as an actionable spec. The guard itself ships **no production change** — it only strengthens what we can prove, so the risky work has a net.

---

## 0. Why this exists first

`TODO.md` records a graveyard of rejected sampler/prefetch probes (prepared-sampler prefetch → 81.3 tok/s; C++ sampler wrapper 91.6→86.3; sampled-token lookahead → empty output; scalar sampled-token sync 91.0→89.2; zero-key random handle → 90.1; yield-before-prefetch → 88.0). The standing rule: **no sampler/lookahead change without first extending the retained-session state-advance parity guard.**

The Tier C work (prompt-lookup C1, fused on-device sampler C2, single-eval/async pipelining C3, `position_ids` C4, sample-aware verification C5) all land in exactly this area. So the guard goes in first.

---

## 1. What the guard covers today

| Test | Location | Pins |
|------|----------|------|
| `TestSample_PrefetchTokenEvalParity_Good` | `go/pkg/metal/sample_test.go:351` | First-token RNG + suppression parity: production `SampleTokenIDWithSuppressionGuard` (direct) vs `sampler.Sample` + `EvalAsync` (prefetched) over a single logits vector → identical token ID. |
| `TestModelSession_PrefetchTokenStateAdvanceParity_Good` | `go/pkg/metal/session_test.go:588` | 2-token retained-session advance over `NewPagedKVCache(0, 2)`: direct vs prefetched (`advanceTokenLocked` + `detachEvalState` + `appendCacheDirtyState` dirty-KV) → identical ID sequence. |

**Reference contract (do not change):** production stays on the explicit sampled-token eval path (`SampleTokenIDWithSuppressionGuard`, `sample.go`). Any candidate path must match it *exactly* under a fixed seed.

### What today's guard does NOT cover (the gaps the gated work needs)

1. **Horizon** — only 2 tokens. The probes that produced `empty_visible_output` / drift only showed up over longer traces.
2. **Cache families** — only `PagedKVCache(0, 2)`. A boundary change must not diverge on `KVCache`, `RotatingKVCache`, `FixedKVCache`, `QuantizedKVCache`, or `TurboQuantKVCache`.
3. **KV state equality** — current tests compare *token IDs only*, never the resulting cache contents. A change can emit the same first tokens yet corrupt later state.
4. **Sampler config matrix** — only `temp=1, topP=0.95, topK=4`. No greedy / minP / RepeatPenalty / large-vocab coverage.
5. **Multi-token (speculative) verification** — no test that accepting/rejecting a block of draft tokens yields the same output + state as the non-speculative baseline.
6. **`position_ids`** — no proof that adding explicit positions is a no-op for the contiguous (non-tree) case.

---

## 2. Design principles

- **One reference, many candidates.** The reference is today's production explicit-sampled-token eval. Each new technique is a "candidate runner." Parity = candidate produces an **identical token-ID sequence AND identical resulting KV-state hash** to the reference, under a fixed RNG seed.
- **Deterministic + CI-cheap by default.** Extend the existing synthetic `stateAdvanceParityModel` stub (`session_test.go:725`) for the matrix — no GPU model files needed. Add an *optional* real-model (Gemma-4) end-to-end parity behind the `/Volumes/Data/lem/safetensors` skip.
- **Bit-exact where the maths allows, statistical where it doesn't.** Greedy and shared-RNG temperature → sequence-exact. Independent-RNG sampling → distribution-equivalence (seeded chi-square, tolerance defined per layer).
- **House style.** `_Good`/`_Bad`/`_Ugly`; `requireMetalRuntime(t)`; UK English; one model per benchmark.

---

## 3. The layered guard

**Layer 0 — keep the two existing tests** as regression anchors (no change).

**Layer 1 — N-token prefetch-vs-direct parity across the cache matrix.** *(biggest immediate uplift; pure guard, no feature code)*
- Horizon `N` tokens (open decision §8).
- Cache families: `KVCache`, `RotatingKVCache`, `FixedKVCache`, `PagedKVCache`, `QuantizedKVCache`, `TurboQuantKVCache`.
- Sampler matrix: greedy(`temp=0`), `temp=1`+topP, topK-only, minP, suppression on/off, RepeatPenalty on/off.
- Assert per case: (a) identical token-ID sequence; (b) identical resulting **KV-state hash** — `CaptureKVWithOptions` → canonical bytes → sha256 (new helper `sessionKVStateHash`, mirroring the sha256 canonicalisation already in `kv/snapshot.go`).

**Layer 2 — `position_ids` parity (enabler for C4).**
- When the optional explicit-`position_ids` model-call path exists, assert that for **contiguous** positions it equals the integer-`offset` path (token IDs + KV hash). Guarantees `position_ids` is a no-op for the non-tree case *before* any tree-attention work builds on it.

**Layer 3 — fused-sampler-vs-reference-chain parity (guards C2).**
- The fused on-device argmax/sample kernel must produce identical token IDs to the reference `newSampler` chain (`sample.go`) across the sampler/seed/vocab matrix, including a **large (≈256k) vocab** and suppression. Bit-exact for greedy; shared-RNG-exact for sampled.

**Layer 4 — speculative-vs-baseline equivalence (guards C1, C5).**
- **Greedy (lossless contract):** the accepted token sequence **and** resulting KV-state hash from the speculative path must equal the non-speculative baseline, for *any* accept/reject pattern. This is the core correctness contract for prompt-lookup.
- **Sampling (`temp>0`):** with modified rejection sampling + a shared RNG stream → sequence-exact; otherwise distribution-equivalence via seeded chi-square (tolerance §8).
- **Adversarial cases (the ones that broke before):** full-reject block (every draft wrong → must equal baseline), partial-accept-then-correct, accept-all, and long-horizon drift (reuse `N` from Layer 1).

---

## 4. Reusable rig (so each new technique plugs in)

New helper file `go/pkg/metal/parity_test.go`:

```go
type parityCase struct {
    name      string
    newCache  func() Cache       // one per cache family
    sampler   samplerConfig      // temp, topP, topK, minP, suppress, repeatPenalty
    seed      uint64
    horizon   int
    candidate candidateRunner    // prefetchAsync | fusedSampler | positionIDs | speculative
}

// captureCanonicalIDs runs reference + candidate through one path and returns IDs.
// sessionKVStateHash canonicalises CaptureKVWithOptions output → sha256.
// assertParity(t, ref, cand) compares ID sequence AND KV-state hash.
```

Candidate runners (each a thin adapter onto an existing or new path):
- `prefetchAsync` — today's `sampler.Sample` + `EvalAsync` + dirty-KV (already exercised by Layer 0).
- `fusedSampler` — C2 kernel.
- `positionIDs` — C4 explicit-position call.
- `speculative` — C1 prompt-lookup drafter + C5 verifier.

Adding a technique = adding one runner + one table row, not a new bespoke test.

---

## 5. Benchmark gate (perf safety, not just correctness)

Correctness parity is necessary but not sufficient — the rejected probes were *correct* and still regressed throughput. Add `BenchmarkModelSession_RetainedDecodeTrace` emitting the `TokenPhaseTrace` split (notably `PrefetchLogitsDuration` — your headline cost — plus decode tok/s). Policy: a candidate that passes parity but regresses the retained trace is rejected, exactly per the existing probe log. Bench one model at a time.

---

## 6. CI / merge policy

- **Gate:** no sampler / lookahead / eval-boundary / spec-decode change merges unless Layers 0–N pass. Add the line to `TODO.md` and `CONTRIBUTING.md`.
- Synthetic-stub layers run in normal CI (no model files). The real-model layer runs where `/Volumes/Data/lem/safetensors` exists; `t.Skip` otherwise.

---

## 7. Sequencing

1. **Layer 1** — N-token + cache matrix + KV-state hash. Biggest coverage uplift, zero feature code, lands independently of any Tier C work. **Do first.**
2. **Layer 2** `position_ids` parity — ships alongside C4.
3. **Layer 3** fused-sampler parity — ships alongside C2.
4. **Layer 4** speculative equivalence — greedy-lossless test ships with C1 (prompt-lookup); the distribution test ships with C5 (sample-aware verify).

---

## 8. Open decisions for you (the forks)

1. **Horizon `N` for Layer 1** — 32 / 64 / 256? (longer catches more drift, costs more CI time). *Rec: 64.*
2. **KV-state assertion strength** — full KV-state hash equality (strong; the whole point is state integrity) vs token-IDs only (cheaper, weaker). *Rec: hash equality.*
3. **Sampling-speculative target (Layer 4)** — shared-RNG sequence-exact (strict, simplest to assert) vs distribution-equivalence chi-square (more faithful to independent sampling). *Rec: start sequence-exact, add chi-square later.*
4. **Stub-only or also a gated real Gemma-4 parity now?** *Rec: both — real one behind the model-path skip.*

---

## 9. File touch-points

| File | Change |
|------|--------|
| `go/pkg/metal/parity_test.go` *(new)* | Table-driven rig, `sessionKVStateHash`, `captureCanonicalIDs`, `assertParity`. |
| `go/pkg/metal/session_test.go` | Layer 1/2/4 tests reusing the rig; keep Layer 0 anchors. |
| `go/pkg/metal/sample_test.go` | Layer 3 fused-sampler parity; keep Layer 0 anchor. |
| `go/pkg/metal/session.go`, `generate.go` | **No change for the guard itself.** Production paths change only when C2/C4/C1/C5 land. |
| `TODO.md`, `CONTRIBUTING.md` | Merge-gate policy line. |

---

## 10. Acceptance criteria

- Layer 1 passes for all six cache families at horizon `N` with both ID-sequence and KV-state-hash equality, across the sampler matrix.
- The rig accepts a new candidate runner with one struct + one table row.
- `BenchmarkModelSession_RetainedDecodeTrace` reports the phase split and is wired into the perf-gate discipline.
- The merge-gate line is documented.
- No production decode path changed by this work.

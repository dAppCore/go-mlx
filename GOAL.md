# GOAL — implement the native API, do not reinvent the program

> **The one rule.** There is a working system (`pkg/metal` + `pkg/scheme` + `pkg/model`).
> The job is to implement the **no-cgo native compute backend** *behind* that system —
> consuming its registries, its declarative arch, and its quant-agnostic loading. It is
> **not** to build a second program inside `pkg/native`. A model arriving in bf16 / 4 / 5 /
> 6 / 8-bit must "just work" because the quant never reaches the model code — that is the
> whole point of the registries, and it already holds for `pkg/metal`.

---

## 0. Why this document exists

`pkg/native` was built as a parallel program: its own assembler, its own quant-weight
type, its own `fetchQuant`/`fetchNorm` per-weight branches, its own per-layer-FFN /
KV-share / MoE / PLE / embed handling, its own arch derivation. The consequences,
observed repeatedly:

- **Solved bugs get re-discovered** — per-layer FFN width, KV-share skipping, the
  global proportional+partial RoPE convention. `pkg/metal` already solves all of these
  once; the native rewrite re-hit each of them as a "new" bug.
- **A different quant breaks loading** — e4b (qat-4bit) quantises
  `per_layer_model_projection`; e2b (4bit) keeps it bf16. The native loader hard-codes
  that weight as bf16 (`fetchNorm`), so e4b fails where the working system would not
  even notice the difference.
- **Two implementations to maintain**, diverging, neither the source of truth.

This is the architecture being wrong, not a fix being missing. The remedy is to plug
native into the seam the system already defines, and **delete** the duplicated program.

---

## 1. The working system (the source of truth — consume, do not duplicate)

| Package | Role | Native must… |
|---|---|---|
| **`pkg/scheme`** | Pure-Go registries: `QuantScheme` (affine, q4_0, mxfp4, nvfp4…), `CacheScheme` (q8, turboquant, compaction, paged…), `Mixer` (softmax-hybrid, gla, mamba2…). | **Resolve** quant / cache / mixer from here. Register native compute as a scheme; never branch the engine on family. |
| **`pkg/model` — `backend.go`** | `Backend.DecodeForward(inputs [][]byte) ([][]byte, error)` — the seam. Activations cross as bf16 `[]byte`. | **Implement this and only this** as its model-facing surface. |
| **`pkg/model` — `quant.go`** | `QuantMatVec` + `RegisterBackendQuant(backend, q)` keyed `(backend, kind)`; `BackendQuant(backend, kind)`. | **Register** native quant compute; **resolve** every quant matvec through `BackendQuant("native", kind)`. |
| **`pkg/model/gemma4` — `arch.go` / `config.go`** | The **declarative arch**: `Arch` (all dims + gemma4 specifics), `LayerSpec` (per-layer attention type, KV-share, head-dim, MoE), `DeriveLayers`, `MaxHeadDim`, `MaxKVHeads`. | **Consume `Arch` as the sole source** of "what gemma4 is". Never re-derive layer types, KV-share maps, per-layer head-dim, MoE/PLE presence. |
| **`pkg/model/mistral`** | Shared Mistral `config.go` + `yarn.go`. | Consume; do not re-parse. |
| **`pkg/model` — `token.go` / `sample.go`** | Shared tokenisation + sampling above the seam. | Consume; the LM head + sampler sit on top of `DecodeForward`, shared. |
| **`pkg/metal`** + **`pkg/metal/model`** (`gemma4/`, `mistral/`) | The **proven** cgo engine + its model logic: the `Linear` / `QuantLoader` pattern ("model architecture is independent of quant format"), the loading, the arch execution, KV-share / MoE / PLE routing, the cache + mixer registries' use. | **READ-ONLY REFERENCE + parity oracle — NEVER edited.** It does not use `pkg/model` and won't. Read it to get native's behaviour right; reproduce the *pattern* in `pkg/native`. |

**The principle, in one line:** *arch* (what), *quant/cache/mixer* (how each piece is
provided), and *compute* (the kernels) are three separate concerns. The first two are
shared/registry-driven. Only the third is native's to author.

---

## 2. The seam — where native attaches

```
        ┌───────────────────────── shared, backend-agnostic ─────────────────────────┐
        │  pkg/model: Backend.DecodeForward  ·  QuantMatVec registry (backend,kind)   │
        │  pkg/model/gemma4: Arch + LayerSpec + DeriveLayers   pkg/scheme: 3 registries│
        └───────────────▲───────────────────────────────────────────▲────────────────┘
                         │ implements DecodeForward                   │ registers compute
        ┌────────────────┴───────────┐                ┌───────────────┴────────────────┐
        │ pkg/metal (cgo / mlx-c)     │                │ pkg/native (no-cgo / metallib)  │
        │ — the working reference     │                │ — COMPUTE ONLY (this work)      │
        └─────────────────────────────┘                └─────────────────────────────────┘
```

Both backends implement `DecodeForward` and register a `QuantMatVec`. Today native does
implement `DecodeForward` (`pkg/native/backend.go`) and register `native/affine`
(`pkg/native/model_quant.go`) — but the **loading + arch + per-weight quant decisions
around it are native-authored duplicates** instead of coming from the shared layer.
That gap is the work.

**Metal is never touched.** `pkg/metal` (and `pkg/metal/model`) is the working reference
and the parity oracle — it does not use `pkg/model` and won't. **All work is in
`pkg/native`** (plus `pkg/model` where a piece is genuinely shared, e.g. the arch already
there). `pkg/model` has the declarative arch (`gemma4/arch.go`, `config.go`) + the
contracts (`backend.go`, `quant.go`); the loading + orchestration logic native needs is
**read from** `pkg/metal/model` as the reference and implemented the right way in native —
quant-agnostic, arch-driven, registry-routed — rather than reinvented and re-bugged. No
metal edit, no "migration": native is brought up to the working pattern.

---

## 3. Requirements

Each is testable. Numbered for review/correction.

### R1 — Consume the shared declarative arch
Native's decode reads `pkg/model/gemma4.Arch` (dims, per-layer `LayerSpec`,
`DeriveLayers`, `MaxHeadDim`, `MaxKVHeads`, rope / MoE / PLE / `AttentionKEqV` /
`ValueNorm` fields) as the **single** source of truth. Native must not re-derive layer
attention types, KV-share maps, per-layer head-dim, or MoE/PLE presence. Where it
currently does (`decode_forward_arch*.go`), that derivation is deleted and replaced by
reading `Arch`.

### R2 — Weight loading is quant-agnostic *per weight* (the `Linear` pattern)
Every weight loads through **one** path that decides per weight:
`.scales` present ⇒ quantised (decoded via the registered `QuantMatVec`), else bf16.
No weight may hard-code its format. The `fetchNorm`-vs-`fetchQuant` split that assumes a
weight is "always bf16" or "always 4-bit" is removed. **Consequence (the acceptance
test):** a model in bf16/4/5/6/8, or one that quantises a weight another leaves bf16
(e4b's `per_layer_model_projection`), loads with **zero** native edits.

### R3 — All quant compute resolves through the `(backend,kind)` registry
Native registers its quant compute once per kind (`RegisterBackendQuant("native", …)` —
affine exists; further kinds register as schemes). Every quantised matvec in the decode
resolves via `model.BackendQuant("native", kind)`. Adding a format (q4_0, mxfp4, nvfp4,
5/6/8-bit) is one scheme registration + one backend impl — **no** decode/model edit.

### R4 — KV cache via the shared cache layout + `CacheScheme` registry
Cache topology (owner vs shared, sliding vs global, `CacheIndex` / `KVShareFrom`) comes
from `Arch.DeriveLayers`. Cache storage/mode resolves through `pkg/scheme.CacheFor(mode)`.
Native consumes both; it does not hand-roll a cache layout. *(This is the "where is the
KV reg" point.)*

### R5 — Sequence mixer via the shared `Mixer` registry
Attention vs hybrid/SSM is a `Mixer` (`pkg/scheme.MixerFor(kind)`); the mixer owns its
state kind; `scheme.Compatible(mixer, cache)` gates the pairing at load. Native consumes
the resolved mixer; it does not branch the decode on model family.

### R6 — Native's model-facing surface is `DecodeForward` and nothing else
Below the seam: native's kernels. Above it: the shared LM head + sampler. The arch,
the loading, and the quant decisions are shared/registry-driven, never native-authored.

### R7 — What native legitimately owns (and nothing more)
The no-cgo Metal **driver + compute**, which has no equivalent in the shared layer:

- purego / tmc-apple binding; PSO + command-encoder management.
- the metallib kernel dispatches: matmul, `affine_qmv`, RMSNorm, RoPE (incl.
  proportional / partial), SDPA, gelu-gate, MoE gather.
- the on-device decode loop; the ICB encode-bypass; the zero-copy mmap weight loader.
- the autorelease-pool / `LockOSThread` discipline.

These stay. They are the actual "native API" to implement.

### R8 — Delete the parallel program
Native files that duplicate shared concerns are removed or shrunk to compute-only: the
per-weight quant assembler logic, the re-derived arch, the duplicated FFN / KV-share /
MoE / PLE / embed handling now provided by `Arch` + the quant-agnostic loader. Net:
native shrinks; the shared layer is the single source.

### R9 — Correctness gate: prove it on the small models, mind the cache
Quant-agnosticism (R2/R3) is proved on the **small** models first, keeping the HF cache
from ballooning:

- **Proof set (required):** **e2b + e4b**, each across **{qat, non-qat} × {4-bit, 8-bit,
  bf16}**. These exercise every quant decision the load path makes — small models only, so
  the HF cache stays modest. This set passing **is** the proof of R2/R3.
- **Larger models (12b, 26b, 31b):** **deferred** until the proof set is green, then
  validated **opportunistically** — only quants already cached, never download-required.
  They load through the *same* path by construction, so the big models inherit the win;
  pulling every large-model quant just to re-prove the same code is wasted disk.

`native-smoke.sh` is parity-checked against the trusted metal engine on the shared arch.

### R10 — No new abstractions ("do NOT invent new")
Reuse `pkg/scheme` + `pkg/model` + the metal `Linear`/registry pattern. Native adds only
the no-cgo compute. If something seems to need a new abstraction, it almost certainly
exists already in the working system — find it and use it.

---

## 4. Plan of work (all in `pkg/native`; metal untouched)

1. **Audit `pkg/native`.** Tag every file: *compute (keep)* / *reinvented-duplicate
   (replace)* / *glue (rewire)*. Output: the delete/keep map.
2. **Quant-agnostic loader (R2).** One native load path keyed on `.scales` — present ⇒
   quant via the registered `QuantMatVec`, else bf16 — reproducing metal's `Linear`
   behaviour (read `pkg/metal/model` for the exact rules). Remove every hard-coded
   `fetchNorm`/`fetchQuant` per-weight assumption. First proof: e2b + e4b (qat + non-qat)
   through the one path.
3. **Arch from the contract (R1).** Native decode consumes `pkg/model/gemma4.Arch`;
   delete the native re-derivation.
4. **Quant compute through the registry (R3).** Native matvecs via
   `BackendQuant("native", kind)`; no hard-coded `QMVBF16` call sites.
5. **Cache + mixer from the registries (R4/R5).**
6. **Delete the reinvented duplicates (R8).** `pkg/native` is compute + glue only.
7. **Gate (R9).** native-smoke green across the proof set; parity to the (untouched) metal.

The order is logical, not staged: `pkg/native` is **not shipped yet** — there's no
keep-green-each-commit ceremony and nothing downstream depends on intermediate states.
Restructure freely toward the working end-state. **Metal is never edited** — only read
(reference) and run (parity oracle).

---

## 5. Definition of done

- Native is a registered backend whose only model-facing surface is `DecodeForward`.
- `pkg/model/gemma4.Arch` drives native; native re-derives nothing.
- Weight loading is quant-agnostic per weight; a new quant works with no native edit.
- Quant / cache / mixer all resolve through the registries.
- The duplicated native model code is gone.
- `native-smoke.sh` green for the proof set — e2b + e4b × {qat, non-qat} × {4-bit, 8-bit,
  bf16} — parity to metal. Larger models inherit the same path (validated opportunistically
  from what's already cached, never download-required).

---

## 6. Non-goals (explicitly out of scope here)

- **Performance tuning.** Speed (tok/s, ICB amortisation, fusion) is the AX-11 phase —
  *after* correctness-by-architecture. Do not optimise during this work.
- **New model families / quant formats** beyond what the cached variants need — though
  the design must make adding them a registration, not a rewrite.
- **go-rocm.** It inherits the shared layer for free once this is right; no work here.

---

## 7. Decisions (resolved)

1. **Scope.** No metal edit, no migration. `pkg/metal` + `pkg/metal/model` are read-only
   reference + parity oracle. All work is in `pkg/native` + `pkg/model`.
2. **Shared logic lands in `pkg/model`.** The backend-neutral loading + orchestration that
   `pkg/model` is missing is filled **into `pkg/model`** (reference: `pkg/metal/model`,
   read not edited) — config parsing, weight assembly with the per-weight quant decision,
   arch execution, KV-share / MoE / PLE routing — at the byte level. `pkg/native` consumes
   it and does the GPU upload + compute. Metal keeps its own, untouched.
3. **Keep the genuinely-native compute.** Kernels, ICB encode-bypass, zero-copy mmap loader
   stay in `pkg/native`, rewired behind the shared arch + quant-agnostic loader.
4. **No rollout ceremony.** `pkg/native` isn't used yet — restructure freely toward the
   working end-state; no keep-green-each-commit. Just make it work.

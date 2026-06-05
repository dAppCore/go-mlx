# Model ↔ Runtime SDK — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `pkg/metal/model/gemma4` a pure-Go `package gemma4` (the model *architecture*) that depends only on `metal`'s public SDK, while `metal` keeps the gemma4-specific *runtime* (speculative-decode assistant + fused cgo kernels) — driven through interfaces + request structs, never concrete `Gemma4*` types. Then merge to `dev` green.

## STATUS — extraction complete + verified (2026-06-03), pre-merge

**The gemma4 architecture is extracted into pure-Go `package gemma4`; all 4 builds green (metal/gemma4/cmd-mlx/mlx-root); no import cycle; behaviour-verified.** Done on branch `model-sdk` (not yet merged to dev):
- `eafbada` Cat 2 cache accessors · `74b193f` Cat 3 fused kernels→metal (reviewed faithful) · `0f74221` architecture compiles on SDK · `3522771`+`1cb85b7` assistant re-homed (reviewed behaviour-faithful) · `30a499d` gemma4 test pkg green.

**Task 3 is REVERSED** (Snider's call, mid-execution): the speculative-decode assistant spans the runtime↔architecture boundary, and severing it to metal would leak model cache-topology. So **the assistant stays IN `package gemma4`** and calls metal's exported runtime-author API (`metal/runtime_author.go`) — the accepted runtime-mgmt "leak", not a topology leak. The Task 3 body below (sever-into-metal) is superseded; keep for history.

**Remaining before merge to dev:**
1. **Test-straddle** — `metal/cache_profile_test.go`+`decode_test.go` reference gemma4 types from package metal (→ external `metal_test` pkg, or move to gemma4, or rework); go-root `backend/fast_eval/speculative_test.go` need `metal.Gemma4Assistant*`→`gemma4.*` + a `fakeNativeModel` test-seam rework (dispatch is now on concrete `*metal.Model`). The old Go-ignored `_parked_assistant_tests/` scratch copies were removed; restore coverage in real package tests only.
2. **Task 5** register/blank-import — likely effectively done (cmd/mlx builds); confirm registry + optional `GO_MLX_RUN_METAL_TESTS=1` smoke against a real target+drafter (closes the runtime-coverage loop the skipped tests leave).
3. **Task 6** squash + merge to dev (gated on `go test ./go/...` green).

---

**Architecture:** Three public API categories in `metal` — primitive surface (Cat 1) · cache accessors (Cat 2) · native-kernel request structs (Cat 3) — on top of the existing `metal.InternalModel` entry + `RegisterModelLoader` registry (both shipped). Design is `docs/RFC.model-sdk.md`.

**Boundary decision (the load-bearing call, made with Snider 2026-06-03 — "sever with interfaces"):**
The `gemma4/` folder the spike produced mixes two kinds of code, and they share *concrete* types (`Gemma4TextConfig`, `Gemma4DecoderLayer`, `Gemma4Attention`, `sharedKV`), so they cannot sit in separate packages without an import cycle unless the runtime reaches the model through *interfaces only*:

| Stays in `package metal` (runtime, cgo) | Moves to `package gemma4` (pure architecture) |
|---|---|
| speculative-decode assistant (`assistant_generate/pair/decode.go`) — written as `func (m *metal.Model)…`, reaches ~25 `metal.Model` internals (prompt-cache, device, slots, metrics, `lastErr`) | model + forward + attention + decoder_layer + experts + router + config + weights + load + masks + perlayer + methods + vision |
| fused cgo kernels (`nativeGemma4*` in `decode.go`, `import "C"`, `Array.ctx`) | calls metal via Cat 1 ops + Cat 2 accessors + Cat 3 request structs; **no cgo, no `Gemma4*` type named in metal** |
| `sharedKV`, `fixedGemma4AttentionMaskSet`, `gemma4RuntimeMaskCache` (runtime helpers) | |

**Corrected land order** (the spike's "rewire gemma4 in place" is impossible — illegal `func (m *metal.Model)` in package gemma4): Cat 2 (done) → Cat 3 kernels to metal (removes cgo from gemma4) → sever assistant to metal via interfaces (clears the cycle + ~140 errors) → wire architecture to the SDK + move its orphaned tests → register + green → land.

**Tech Stack:** Go 1.26 (workspace `go.work`); cgo + Apple MLX-C + Metal compute shaders (darwin/arm64 only). Build env for every command:
```
export GOWORK=/Users/snider/Code/core/go-mlx/go.work
export GOCACHE=/private/tmp/go-mlx-self/gocache
```
Green oracle: `go build ./go/pkg/metal/` is clean *now* (non-test build); `package metal`'s **test** build is pre-broken because the spike left three architecture tests behind (`cache_profile_test.go`, `decode_test.go`, `attention_bench_test.go` reference `Gemma4Model`/`Gemma4TextConfig`/`Gemma4DecoderLayer`/`buildGemma4SlidingMask`/mask-cache) — those move to gemma4 in Task 4. Full `go test ./go/...` green is the end-state (Task 5). Binary link check: `go build -ldflags "-extldflags=-mmacosx-version-min=26.0" -o /private/tmp/go-mlx-self/bin/lthn-mlx ./go/cmd/mlx`.

**Critical lessons from the spike — re-read before starting, do NOT repeat:**
- NEVER `git reset --hard`, `git checkout -- `, or `git stash` to "clean up" — uncommitted work is NOT "in git". Commit or branch first. If something looks wrong, STOP and report; do not recover by discarding.
- Verify every `cd` target with an absolute path. A `cd`-typo silently ran a sweep in the wrong directory and corrupted metal's own files.
- **Qualifying** a ref (`X` → `metal.X`): `gofmt -r 'X -> metal.X' -w *.go` — AST-safe, leaves selectors/method-defs/composite-literal keys alone. **Exporting** a symbol (rename def + all calls): `gofmt -r` does NOT rename func/method *definitions*, and blanket `perl s/\bfoo\b/Foo/g` BREAKS method-name collisions. Use careful per-symbol edits; build after every batch.
- cgo C types are package-private: a model package cannot use `metal.C.mlx_array`. Fused kernels stay in `metal`; the model passes data via request structs.

---

### Task 0: Resume on the work branch and snapshot the work-list  — ✅ DONE

Branched `model-sdk` off `wip/gemma4-split` (spike kept as fallback). Work-list captured: 198 errors, all in `model/gemma4/` (assistant_decode 74, assistant_generate 66, decode 39, rest in architecture files). Bridge accessors (`metal.ArrayHandle`/`ArrayFromHandle`/`DefaultStreamHandle`) confirmed present in `array.go` (kept for Cat 3 if a kernel needs the handle path; in-package cgo can use `Array.ctx`/`cArray` directly so they may end up unused — fine).

---

### Task 1: Cat 2 — cache accessors  — ✅ DONE (`eafbada`)

Added the RFC Cat 2 read-surface to the five cache types in `cache.go` + `cache_quantized.go`: `Keys()`/`Values()`/`Step()`/`MaxSize()`/`PageSize()`/`Bits()` as appropriate per type (reusing existing `Offset()`/`Len()`). No constructors (construction is runtime/metal-side). `go build ./go/pkg/metal/` clean. Trivial documented pass-throughs.

---

### Task 2: Cat 3 — move the fused cgo kernels into `metal` as request structs  [first sever bite]

**Files:**
- Create: `go/pkg/metal/gemma4_native.go` (package metal — the cgo kernels move here, taking request structs).
- Modify: `go/pkg/metal/model/gemma4/decode.go` (kernels leave) and the architecture call sites in `forward.go` / `attention.go` / `decoder_layer.go` / `router.go` (switch to `metal.Native…(req)`).

The kernels in `decode.go` are cgo (`import "C"`, `C.go_mlx_gemma4_*`) and take concrete `*Gemma4Attention`/`*Gemma4DecoderLayer`/`*Gemma4TextConfig`/`*Gemma4Model`. They must live in `metal` beside the C types. Expose each through a request struct of `*metal.Array` + scalars; the architecture fills it.

The kernels + their architecture call sites:
- `nativeGemma4FixedOwnerAttentionBlock` / `…ResidualBlock` (+ `…Available` predicates, `…Args` builder) ← `attention.go:41`, `decoder_layer.go:47`
- `nativeGemma4DecodeLayer` (+ `…Available`) ← `decoder_layer.go:28`
- `nativeGemma4FixedGreedyTokenWithArray` (+ `…/Available`/`…Reason`) ← `forward.go:165`
- `nativeGemma4LayerArgs`, and the leaf predicates `nativeGemma4NormsAvailable` / `…LayerAttentionAvailable` / `…AttentionAvailable` / `…SharedKVAvailable` / `…LayerSkipTraceName`
- the `metal.NativeGemma4*Enabled()` runtime gates already live in metal (`decode.go:147`+, `runtime_gate.go`) — leave them.

- [ ] **Step 1 (pattern kernel first):** in `gemma4_native.go` define `type Gemma4FixedAttentionRequest struct { X, Residual, KeyCache, ValueCache, Offset, Scale, Mask, QWeight, QScales, …, RopeFreqs *Array; NumAttentionHeads, NumKeyValueHeads, HeadDim, RopeDims int32; RopeBase float32 }` + `func NativeGemma4FixedOwnerAttention(req Gemma4FixedAttentionRequest) (out *Array, kv …, ok bool, err error)`. Move the cgo body across — in-package `Array.ctx`/`cArray` access is legal here. Build `./go/pkg/metal/`.
- [ ] **Step 2:** switch `attention.go`'s call site to fill the request from `a *Gemma4Attention` + `cfg`. The predicate (`…Block` returns `ok=false` when unavailable) folds the `…Available` check into the kernel's `ok` return where possible.
- [ ] **Step 3:** repeat for the decode-layer kernel, the greedy-token kernel, the args builders, and the predicates (one request struct each; predicates either take a request or collapse into `ok`). Build after each.
- [ ] **Step 4 (verify):** `grep -rl 'import "C"' go/pkg/metal/model/gemma4/` → EMPTY. `go build ./go/pkg/metal/ 2>&1 | grep -vE 'mmacosx|ld: warning'` clean. The 13 `Array.ctx` reaches gone from gemma4. (The architecture still won't fully build — assistant + Cat 1/qualify pending — but cgo + native-kernel errors are gone.)
- [ ] **Step 5:** `git add go/pkg/metal/gemma4_native.go go/pkg/metal/model/gemma4/{decode,attention,decoder_layer,forward,router}.go` + commit `feat(metal): gemma4 fused kernels as request structs; no cgo in model pkg (RFC.model-sdk Cat 3)`.

---

### Task 3: Sever the assistant speculative-decode subsystem back into `metal`  [cycle resolution]

**Files:**
- Move → metal: `assistant_generate.go`, `assistant_pair.go`, `assistant_decode.go` (+ `assistant_generate_test.go`) become `go/pkg/metal/gemma4_assistant_*.go`, `package metal`.
- Define (in metal): the model-facing interface(s) the assistant uses to read the architecture, so no `Gemma4*` architecture type is named in metal.

The assistant loop is `func (m *metal.Model) GenerateGemma4Assistant…` — illegal in package gemma4, and it reaches ~25 `metal.Model` internals (`lastMetrics`, `tokenizer`, `promptCache*`, `acquireSlot`, `withDevice`, `requireTextRuntime`, `newCachesWithRequestFixedSize`, `prefillChunkSize`, `lastErr`, …). It is runtime, RFC-owned by metal.

- [ ] **Step 1:** move the files to package metal; receivers `*metal.Model` → `*Model`. The ~25 internals + the assistant's own types (`Gemma4AssistantPair/Model/Layer/Attention`, `Gemma4Assistant*Result`) compile again in-package. `sharedKV` + `fixedGemma4AttentionMaskSet` + `gemma4RuntimeMaskCache` stay/return to metal (runtime helpers the assistant + kernels share).
- [ ] **Step 2 (the interface, the actual "sever"):** the assistant still reads architecture hyperparameters + layers (`*Gemma4TextConfig`, `*Gemma4DecoderLayer`, `*Gemma4Attention`). Replace those concrete reads with a model-facing **capability interface** the gemma4 architecture implements (e.g. extend `InternalModel`, or a `Gemma4RuntimeView` returning the scalar config + per-layer handles), OR a plain-data config the architecture hands metal at load via `RegisterModelLoader`. RULE: grep `go/pkg/metal/gemma4_assistant_*.go` + `gemma4_native.go` for `Gemma4` — every hit must be a metal-local type (`Gemma4AssistantPair`, the request structs) or an interface; NO `Gemma4Model`/`Gemma4TextConfig`/`Gemma4DecoderLayer`/`Gemma4Attention`.
- [ ] **Step 3 (verify):** `go build ./go/pkg/metal/` clean; the ~140 assistant errors gone from the gemma4 build. `go list -deps ./go/pkg/metal/ | grep model/gemma4` → EMPTY (metal must NOT import gemma4 — proves no cycle).
- [ ] **Step 4:** commit `refactor(metal): sever gemma4 assistant runtime into metal via interfaces (RFC.model-sdk)`.

---

### Task 4: Wire the gemma4 architecture to the SDK + relocate its tests

**Files:**
- Modify: architecture files (`config`/`weights`/`load`/`forward`/`attention`/`decoder_layer`/`masks`/`perlayer`/`router`/`methods`/`model`/`experts`/`vision`.go).
- Cat 1: export the metal helpers the architecture still calls (build-list-driven), keep plumbing internal.
- Move: `cache_profile_test.go`, `decode_test.go`, `attention_bench_test.go` from `go/pkg/metal/` → `go/pkg/metal/model/gemma4/` (`package gemma4`).

- [ ] **Step 1:** `go build ./go/pkg/metal/model/gemma4/ 2>&1 | grep '\.go:'` — the residual list. For each `cannot refer to unexported field` cache reach → Task 1 accessor (`c.keys`→`c.Keys()`, `c.maxSize`→`c.MaxSize()`, …).
- [ ] **Step 2 (Cat 1):** for each `undefined: <helper>` that is a genuine model-author primitive → export it (capitalise def + metal callers; leave method-name collisions; do NOT export plumbing — if a plumbing symbol is still needed it's a sign the code belongs in metal). Batch 5–10, build after each.
- [ ] **Step 3 (qualify):** verify `cd` to the gemma4 dir (absolute path), then `gofmt -r 'X -> metal.X' -w *.go` per exported-metal symbol the architecture references bare; `goimports -w *.go` to add the import. (Build the qualify list as in the spike: metal-exported ∩ gemma4-refs − gemma4-own − field-collisions.)
- [ ] **Step 4:** move the 3 orphaned tests into the gemma4 folder, change `package metal` → `package gemma4`, qualify their metal refs, fix to use the new accessors/exports. ALSO: `model_test.go` (gemma4) has ~29 stale lowercase `kv.clone()/free()/hasState()/hasPages()` calls broken by the Task 2 `sharedKV`→`metal.SharedKV` rename — update them to the exported forms (`Clone`/`Free`/`HasState`/`HasPages`); currently masked behind the assistant breakage.
- [ ] **Step 5 (verify):** `go build ./go/pkg/metal/model/gemma4/` clean; `go vet ./go/pkg/metal/model/gemma4/` clean; `grep -rl 'import "C"' …/gemma4/` EMPTY; `go test ./go/pkg/metal/model/gemma4/ 2>&1 | tail -3` green.
- [ ] **Step 6:** commit `refactor(gemma4): pure-Go architecture on the metal SDK; tests relocated (RFC.model-sdk Cat 1+2)`.

---

### Task 5: Register, blank-import, and full green

- [ ] **Step 1:** gemma4 self-registers its loader from `init()` via `metal.RegisterModelLoader("gemma4"/"gemma4_text", …)`; confirm `model_registry.go` in metal no longer names a concrete gemma4 type.
- [ ] **Step 2:** blank-import `_ "dappco.re/go/mlx/pkg/metal/model/gemma4"` from `go/cmd/mlx/main.go` (and any other binary that loads models).
- [ ] **Step 3:** `go build -ldflags "-extldflags=-mmacosx-version-min=26.0" -o /private/tmp/go-mlx-self/bin/lthn-mlx ./go/cmd/mlx && echo BINARY-OK`; then `~/.claude/skills/lethean-lem/scripts/lem.sh smoke` (or the gemma4 load test) — gemma4 loads + generates via the registry.
- [ ] **Step 4:** `go test ./go/... 2>&1 | grep -E '^(FAIL|ok)' | grep FAIL || echo ALL-GREEN`; `go vet ./go/pkg/metal/...` clean.
- [ ] **Step 5:** commit `feat(cmd): blank-import gemma4 package for self-registration (RFC.model-sdk)`.

---

### Task 6: Land on dev

- [ ] **Step 1:** squash `model-sdk` into the conceptual commits (Cat2 / Cat3 / sever / wire / register), dropping spike wip churn. (Interactive rebase is unsupported in the harness — do it via `git reset --soft a0357a9` + re-commit the final tree in staged conceptual commits; the tree is what matters.)
- [ ] **Step 2:** `git checkout dev && git merge --ff-only model-sdk` (or cherry-pick the conceptual commits); `go test ./go/...` green; push `for r in github homelab origin; do git push "$r" HEAD:dev; done`.
- [ ] **Step 3:** update go-mlx #45 (gemma4 architecture extracted; the SDK pattern — Cat 1/2/3 + the capability-interface sever — is ready for qwen3/llama). Delete the `wip/gemma4-split` fallback once dev is confirmed green.

---

## Self-review notes

- **Spec coverage:** Cat 1 → Task 4 Step 2; Cat 2 → Task 1 (done); Cat 3 → Task 2; the "sever with interfaces" boundary → Task 3 (the capability interface) + Task 2 (request structs); InternalModel/registry entry → Task 5; "shape for all" → the request-struct + capability-interface *patterns* reusable by qwen3/llama. All covered.
- **Why the order changed from the original plan:** the compiler proved the spike's split is blocked by illegal `func (m *metal.Model)` methods in package gemma4 and a real architecture↔runtime import cycle. "Rewire in place" can't work; the runtime (assistant + kernels) must return to metal behind interfaces/request-structs. Cat 3 (Task 2) goes first because it removes the cgo coupling cheaply; the assistant sever (Task 3) clears the cycle and 70% of the errors; only then is the architecture residual small enough to wire (Task 4).
- **Build-loop-driven:** the exact Cat 1 export list + the residual cache reaches are derived from `go build ./go/pkg/metal/model/gemma4/` at Task 4 time, not frozen here (they shrink as Tasks 2–3 land). Patterns are shown in full; application is mechanical + build-verified.
- **Harness caveat:** Task 6 squash via `reset --soft` + re-commit, not interactive rebase.

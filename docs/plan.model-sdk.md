# Model ↔ Runtime SDK — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the model↔runtime SDK in package `metal` so `pkg/metal/model/gemma4` compiles and its tests pass as a pure-Go `package gemma4`, with no model→metal-internal reaches — then merge it to `dev` green.

**Architecture:** Add three public API categories to `metal` (primitive surface · cache accessors · native-kernel request structs) on top of the existing `metal.InternalModel` entry point and `RegisterModelLoader` registry (both already shipped). Refactor gemma4 — already file-split into 12 feature files on branch `wip/gemma4-split` — to depend only on that public surface. Design is `docs/RFC.model-sdk.md`. Land order: Cat 1+2 (baseline → gemma4's generic `Forward` path compiles) → Cat 3 (native kernels → fused fast-path) → green.

**Tech Stack:** Go 1.26 (workspace `go.work`); cgo + Apple MLX-C + Metal compute shaders (darwin/arm64 only); go-mlx. Build env for every command:
```
export GOWORK=/Users/snider/Code/core/go-mlx/go.work
export GOCACHE=/private/tmp/go-mlx-self/gocache
```
Binary link check: `go build -ldflags "-extldflags=-mmacosx-version-min=26.0" -o /private/tmp/go-mlx-self/bin/lthn-mlx ./go/cmd/mlx`

**Critical lessons from the spike — re-read before starting, do NOT repeat:**
- NEVER `git reset --hard` on uncommitted work ("it's all in git" is FALSE for uncommitted). Commit or branch first.
- Verify every `cd` target with an absolute path. A `cd`-typo silently ran a sweep in the wrong directory and corrupted metal's own files.
- **Qualifying** a ref (`X` → `metal.X`): use `gofmt -r 'X -> metal.X' -w *.go` — AST-safe, leaves selectors (`x.X`), method defs, and composite-literal keys (`T{X:}`) alone. **Exporting** a symbol (rename def + all calls `foo`→`Foo`): `gofmt -r` does NOT rename func/method *definitions*, and a blanket `perl s/\bfoo\b/Foo/g` BREAKS method-name collisions (e.g. a `Gemma4DecoderLayer.foo` method). Use `gopls rename`/`gorename` (needs compiling code) or careful per-symbol edits. Build after every batch.
- cgo C types are package-private: a model package cannot use `metal.C.mlx_array`. Fused kernels stay in `metal`; the model passes data via request structs.

---

### Task 0: Resume on the work branch and snapshot the work-list

**Files:**
- Work branch: `wip/gemma4-split` (12-file gemma4 split + the `metal.ArrayHandle`/`ArrayFromHandle`/`DefaultStreamHandle` bridge accessors). Push it to a fresh branch first so the spike branch stays as a fallback.

- [ ] **Step 1: Branch off the spike so it stays a fallback**

```bash
cd /Users/snider/Code/core/go-mlx
git checkout wip/gemma4-split
git checkout -b model-sdk
```

- [ ] **Step 2: Capture the complete work-list from the compiler**

```bash
go build -gcflags=-e ./go/pkg/metal/... 2>/tmp/sdk.err; echo "exit $?"
grep -c 'cannot refer to unexported field\|has no field or method' /tmp/sdk.err   # Cat 2 reaches
grep -oE 'undefined: [A-Za-z0-9_]+' /tmp/sdk.err | sort -u                          # Cat 1 helpers + Cat 3 kernels
```
Expected: a list of unexported metal cache-field reaches (Cat 2) and undefined symbols split into generic helpers (Cat 1) and `*Gemma4*`/native-kernel names (Cat 3). This list is the live source of truth — re-run it after every task to see remaining work.

- [ ] **Step 3: Confirm the bridge accessors are present** (added during the spike, kept for Cat 3)

```bash
grep -n 'func ArrayHandle\|func ArrayFromHandle\|func DefaultStreamHandle' go/pkg/metal/array.go
```
Expected: all three present. (No commit — this is orientation.)

---

### Task 1: Cat 2 — cache accessors

**Files:**
- Modify: `go/pkg/metal/cache.go`, `cache_fixed_metal.go`, `cache_quantized.go` (wherever `KVCache`/`RotatingKVCache`/`FixedKVCache`/`PagedKVCache`/`QuantizedKVCache` are defined — `grep -l 'type .*KVCache struct' go/pkg/metal/*.go`)
- Test: `go/pkg/metal/cache_accessor_test.go` (new)

The model reaches into ~183 private cache fields. Replace each with a getter. These are trivial pass-throughs, but pin two with tests so the accessor↔field mapping can't silently drift.

- [ ] **Step 1: Write failing tests for the two highest-traffic accessors**

```go
// go/pkg/metal/cache_accessor_test.go
//go:build darwin && arm64

package metal

import "testing"

func TestKVCache_Accessors_Good(t *testing.T) {
	c := &KVCache{offset: 7, step: 256}
	if got := c.Offset(); got != 7 {
		t.Fatalf("Offset() = %d, want 7", got)
	}
	if got := c.Step(); got != 256 {
		t.Fatalf("Step() = %d, want 256", got)
	}
}

func TestRotatingKVCache_Accessors_Good(t *testing.T) {
	c := &RotatingKVCache{maxSize: 1024}
	if got := c.MaxSize(); got != 1024 {
		t.Fatalf("MaxSize() = %d, want 1024", got)
	}
}
```
(Adjust field names/types to the real struct defs — read them first. `KVCache` already exposes `Offset()` as a method per the spike; if so, drop that line and keep `Step()`.)

- [ ] **Step 2: Run, verify FAIL**

```bash
go test -run 'TestKVCache_Accessors|TestRotatingKVCache_Accessors' ./go/pkg/metal/
```
Expected: FAIL — `Step`/`MaxSize` undefined.

- [ ] **Step 3: Add the accessor methods, driven by the build-list**

For every field the model reaches (from `/tmp/sdk.err`'s "cannot refer to unexported field" lines), add a read accessor next to the cache type. Pattern:

```go
func (c *KVCache) Keys() *Array   { return c.keys }
func (c *KVCache) Values() *Array { return c.values }
func (c *KVCache) Step() int      { return c.step }
// RotatingKVCache / FixedKVCache / PagedKVCache / QuantizedKVCache:
func (c *RotatingKVCache) MaxSize() int { return c.maxSize }
func (c *FixedKVCache) MaxSize() int    { return c.maxSize }
func (c *PagedKVCache) PageSize() int   { return c.pageSize }
func (c *QuantizedKVCache) Bits() (key, value int) { return c.keyBits, c.valueBits }
```
Where the model *constructs* a cache by struct literal (e.g. `KVCache{keys, values, offset, step}` — appears in assistant_decode.go), add an exported constructor instead and use it from gemma4 in Task 3:

```go
func NewKVCacheFrom(keys, values *Array, offset, step int) *KVCache {
	return &KVCache{keys: keys, values: values, offset: offset, step: step}
}
```
Do NOT add a getter whose name collides with an existing method (`Offset()` already exists — reuse it).

- [ ] **Step 4: Run accessor tests + the full metal suite, verify PASS**

```bash
go test -run 'TestKVCache_Accessors|TestRotatingKVCache_Accessors' ./go/pkg/metal/
go test ./go/pkg/metal/   # behaviour preserved — must stay green (~1914 ok)
```
Expected: PASS; metal suite unchanged.

- [ ] **Step 5: Commit**

```bash
git add go/pkg/metal/cache*.go go/pkg/metal/cache_accessor_test.go
git commit -m "feat(metal): cache accessors for the model SDK (RFC.model-sdk Cat 2)"
```

---

### Task 2: Cat 1 — export the curated primitive surface

**Files:**
- Modify: the metal files defining the helpers gemma4 calls (find each: `grep -rln 'func <name>' go/pkg/metal/*.go`)
- No new test file — verification is "metal builds + suite green" (pure rename refactor).

Export only the genuine model-author primitives the build-list shows as `undefined:` and which are *not* `*Gemma4*`/native-kernel names (those are Task 3). Keep runtime plumbing internal.

- [ ] **Step 1: Classify the undefined-helper list**

```bash
grep -oE 'undefined: [A-Za-z0-9_]+' /tmp/sdk.err | sed 's/undefined: //' | sort -u > /tmp/undef.txt
# EXPORT these (model primitives): cacheLen cacheCapacity resolveModelRoot loadModelWeights
#   resolveWeight freeLinear freeEmbedding freeRMSNorm freeSwitchLinear sample* quantized*
#   gelu* normalizeQuantizationMode isAffineQuantizationMode ...  (read each; export if it's an
#   operation/value a model legitimately needs)
# DO NOT export plumbing: cArray lastError suppressIDsScratch appendNativePhaseTraceEvent
#   trace* — these are resolved by Cat 3 (kernels stay in metal) or by moving the using code.
```
For each plumbing symbol, note it: it must NOT cross the boundary — its using-code is a Cat 3 kernel that stays in metal, so it disappears from the model side once Task 4 lands. If a plumbing symbol is still needed by the model after Task 4, that is a design smell — surface it, don't export it.

- [ ] **Step 2: Export one primitive end-to-end as the pattern**

Take `cacheLen`. Rename its definition and all *metal* call sites to `CacheLen` (use `gopls rename` if available, else edit the def + grep the call sites):

```bash
# def + metal callers:
grep -rn '\bcacheLen\b' go/pkg/metal/*.go | grep -v model/
# rename def: func cacheLen(  ->  func CacheLen(   (edit the file)
# rename metal callers: gofmt -r 'cacheLen(a) -> CacheLen(a)' is NOT reliable for variadic;
#   prefer: for each metal caller line, edit cacheLen( -> CacheLen(  (these are package-func calls)
```
Then qualify the gemma4 side (Task 3 does the bulk; this proves the round-trip): in `model/gemma4`, `gofmt -r 'cacheLen -> metal.CacheLen' -w *.go` is wrong (it's now `CacheLen`); instead gemma4 will call `metal.CacheLen` — handled in Task 3's sweep.

- [ ] **Step 3: Export the rest of the primitive list (mechanical, batched)**

For each remaining EXPORT symbol: capitalise its definition + metal call sites. Work in small batches (5-10), `go build ./go/pkg/metal/` after each batch, fix any method-name-collision breakage immediately (a symbol that is also a method — leave the method, only the package-func def+calls get capitalised).

- [ ] **Step 4: Verify metal builds + suite green after every batch; final check**

```bash
go build ./go/pkg/metal/ 2>&1 | grep -v 'mmacosx\|ld: warning'
go test ./go/pkg/metal/
```
Expected: clean build; suite green (~1914 ok).

- [ ] **Step 5: Commit**

```bash
git add go/pkg/metal/*.go
git commit -m "feat(metal): export curated model-author primitives (RFC.model-sdk Cat 1)"
```

---

### Task 3: Wire gemma4's generic path to Cat 1+2 (baseline compiles)

**Files:**
- Modify: `go/pkg/metal/model/gemma4/*.go` (the 12 split files)

Qualify all metal references and replace the private-field reaches with the new accessors, so gemma4's *generic* (non-native-kernel) path compiles.

- [ ] **Step 1: Qualify metal symbol references (AST-safe sweep)**

Build the qualify list (metal exported decls ∩ gemma4 refs − gemma4's own − field-collisions) and sweep with `gofmt -r` — verify the `cd` target first:

```bash
G4=/Users/snider/Code/core/go-mlx/go/pkg/metal/model/gemma4
test -f "$G4/decode.go" && cd /Users/snider/Code/core/go-mlx/go/pkg/metal || { echo ABORT; exit 1; }
grep -hoE '^(type|func|var|const) [A-Z][A-Za-z0-9_]*' *.go | sed -E 's/^(type|func|var|const) //' | sort -u > /tmp/mexp.txt
grep -hoE '^(type|func|var|const) (\([^)]*\) )?[A-Z][A-Za-z0-9_]*' "$G4"/*.go | sed -E 's/^(type|func|var|const) (\([^)]*\) )?//' | sort -u > /tmp/g4own.txt
grep -hoE '\b[A-Z][A-Za-z0-9]+\b' "$G4"/*.go | sort -u > /tmp/g4refs.txt
grep -hoE '^	[A-Z][A-Za-z0-9]+ +[A-Za-z*[]' "$G4"/*.go | awk '{print $1}' | sort -u > /tmp/g4fld.txt   # field-name collisions
comm -12 /tmp/g4refs.txt /tmp/mexp.txt | comm -23 - /tmp/g4own.txt | comm -23 - /tmp/g4fld.txt > /tmp/qs.txt
cd "$G4" && pwd   # MUST print the gemma4 dir
while read -r s; do [ -n "$s" ] && gofmt -r "$s -> metal.$s" -w *.go; done < /tmp/qs.txt
"$(go env GOPATH)/bin/goimports" -w *.go   # adds the metal import; install once: GOWORK=off go install golang.org/x/tools/cmd/goimports@latest
```

- [ ] **Step 2: Qualify the field-collision metal *types* by hand (type positions only)**

For each symbol in `comm -12 /tmp/qs-pre.txt /tmp/g4fld.txt` (e.g. `MLP`, `Token`): `*MLP` → `*metal.MLP`, `[]Token` → `[]metal.Token`, `Token{` → `metal.Token{` — never the field name or `{MLP:` key. `perl -i -pe 's/\*MLP\b/*metal.MLP/g; s/\[\]Token\b/[]metal.Token/g; s/\bToken\{/metal.Token{/g' *.go` then gofmt.

- [ ] **Step 3: Replace private-field reaches with accessors + constructors**

Build → for each `cannot refer to unexported field` / `c.field undefined`, replace `c.keys`→`c.Keys()`, `c.maxSize`→`c.MaxSize()`, struct literals `KVCache{...}`→`metal.NewKVCacheFrom(...)`, etc. (from Task 1). Batch + build.

- [ ] **Step 4: Verify gemma4's generic path compiles** (Cat 3 kernels still error — expected)

```bash
cd /Users/snider/Code/core/go-mlx
go build -gcflags=-e ./go/pkg/metal/model/gemma4/ 2>&1 | grep -vE 'mmacosx|ld: warning' | grep '\.go:' | grep -v 'native\|Native\|C\.' | head
```
Expected: the ONLY remaining errors mention native kernels / `C.` (Task 4). All generic-path errors gone.

- [ ] **Step 5: Commit**

```bash
git add go/pkg/metal/model/gemma4/*.go
git commit -m "refactor(gemma4): generic path uses metal SDK primitives+accessors (RFC.model-sdk Cat 1+2)"
```

---

### Task 4: Cat 3 — native-kernel request structs (fused fast-path)

**Files:**
- Create: `go/pkg/metal/gemma4_native.go` (package metal — the 6 cgo kernel funcs move here from `model/gemma4/decode.go`, taking request structs)
- Modify: `go/pkg/metal/model/gemma4/decode.go` (the gemma4 side fills requests + calls metal)

The fused kernels (`nativeGemma4FixedOwnerAttention[Residual]`, `nativeGemma4DecodeLayer`, `nativeGemma4FixedGreedyTokenWithArray`, `nativeGemma4LayerArgs`, args builders) are cgo + model-shaped → they live in metal. Define a request struct per kernel; the model fills and calls.

- [ ] **Step 1: Define the request struct + kernel signature in metal (one kernel first)**

```go
// go/pkg/metal/gemma4_native.go  (package metal; has the cgo preamble via metal.go + decode_bridge.h)
type Gemma4FixedAttentionRequest struct {
	X, Residual, KeyCache, ValueCache, Offset, Scale, Mask *Array
	QWeight, QScales, QBiases, KWeight, KScales, KBiases   *Array
	VWeight, VScales, VBiases, OWeight, OScales, OBiases   *Array
	QNorm, KNorm, PostAttnNorm, RopeFreqs                  *Array
	NumAttentionHeads, NumKeyValueHeads, HeadDim, RopeDims int32
	RopeBase                                               float32
}

func NativeGemma4FixedOwnerAttention(req Gemma4FixedAttentionRequest) (out, newKeys, newValues *Array, ok bool, err error) {
	// body = the moved cgo func: build C.go_mlx_gemma4_fixed_attention_args from req fields
	// (req.X etc. are metal *Array — same package, use the existing C-handle path directly),
	// call C.go_mlx_gemma4_fixed_owner_attention, wrap results.
}
```
Move the matching `nativeGemma4FixedOwnerAttention*` funcs + their args-builder out of `model/gemma4/decode.go` into this file; swap the `*Gemma4Attention`/`*Gemma4TextConfig` params for the request struct.

- [ ] **Step 2: Call it from the gemma4 side**

```go
// model/gemma4/decode.go (package gemma4)
out, nk, nv, ok, err := metal.NativeGemma4FixedOwnerAttention(metal.Gemma4FixedAttentionRequest{
	X: x, Residual: residual, KeyCache: keyCache, /* ... */
	QWeight: attn.QProj.Weight, QScales: attn.QProj.Scales, /* ... */
	NumAttentionHeads: cfg.NumAttentionHeads, HeadDim: attn.HeadDim, /* ... */
})
```
Delete the gemma4-side cgo preamble bits + `cArray`/`gemma4DefaultStream` bridge helpers — no cgo remains in the model package.

- [ ] **Step 3: Repeat for the other 5 kernels** (same pattern, one request struct each). Build after each.

- [ ] **Step 4: Verify the whole package set builds + tests green**

```bash
cd /Users/snider/Code/core/go-mlx
go build ./go/pkg/metal/... 2>&1 | grep -vE 'mmacosx|ld: warning'
go test ./go/pkg/metal/... 2>&1 | grep -E '^(FAIL|ok)' | tail
go build -ldflags "-extldflags=-mmacosx-version-min=26.0" -o /private/tmp/go-mlx-self/bin/lthn-mlx ./go/cmd/mlx && echo BINARY-OK
```
Expected: clean build; gemma4 package + metal suite green; binary links. NO cgo in `model/gemma4` (`grep -rl 'import "C"' go/pkg/metal/model/gemma4/` → empty).

- [ ] **Step 5: Commit**

```bash
git add go/pkg/metal/gemma4_native.go go/pkg/metal/model/gemma4/*.go
git commit -m "feat(metal): gemma4 native-kernel request structs; gemma4 package is pure-Go (RFC.model-sdk Cat 3)"
```

---

### Task 5: Register, blank-import, and full green

**Files:**
- Modify: `go/cmd/mlx/main.go` (or wherever model loaders are wired) — add the blank import.
- Verify: model_registry.go init no longer registers gemma4 (gemma4 self-registers).

- [ ] **Step 1: Blank-import the gemma4 package so its init() registers**

```go
// go/cmd/mlx/main.go (and any other binary that loads models)
import _ "dappco.re/go/mlx/pkg/metal/model/gemma4"
```

- [ ] **Step 2: Smoke-test that gemma4 still loads + serves** (the registry round-trip)

```bash
go build -ldflags "-extldflags=-mmacosx-version-min=26.0" -o /private/tmp/go-mlx-self/bin/lthn-mlx ./go/cmd/mlx
# load a gemma4 checkpoint via the existing lethean-lem smoke harness:
~/.claude/skills/lethean-lem/scripts/lem.sh smoke   # or the project's gemma4 load test
```
Expected: gemma4 loads + generates (the registry resolves "gemma4"/"gemma4_text" to the package's loader).

- [ ] **Step 3: Full suite + vet**

```bash
go test ./go/... 2>&1 | grep -E '^(FAIL|ok)' | grep FAIL || echo ALL-GREEN
go vet ./go/pkg/metal/... 2>&1 | grep -vE 'mmacosx|ld: warning'
```
Expected: ALL-GREEN; vet clean.

- [ ] **Step 4: Commit**

```bash
git add go/cmd/mlx/*.go
git commit -m "feat(cmd): blank-import gemma4 package for self-registration (RFC.model-sdk)"
```

---

### Task 6: Land on dev

- [ ] **Step 1: Squash the model-sdk branch into a clean set** (Cat1 / Cat2 / Cat3 / wire — keep the 4 conceptual commits, drop the spike's wip churn)

```bash
git log --oneline a0357a9..HEAD   # review
git rebase -i a0357a9             # squash wip into the 4-5 conceptual commits  (NOTE: interactive rebase not supported in the harness — do this manually or via reset-to-a0357a9 + re-commit the final tree in 4 staged commits; the tree is what matters)
```

- [ ] **Step 2: Merge to dev + push**

```bash
git checkout dev && git merge --ff-only model-sdk   # or cherry-pick the conceptual commits
go test ./go/... 2>&1 | grep FAIL || echo GREEN
for r in github homelab origin; do git push "$r" HEAD:dev; done
```

- [ ] **Step 3: Close the loop**

Update go-mlx #45 (the package extraction is done for gemma4; the SDK pattern is ready for qwen3/llama). Delete the `wip/gemma4-split` fallback branch once dev is confirmed green.

---

## Self-review notes

- **Spec coverage:** Cat 1 → Task 2; Cat 2 → Task 1; Cat 3 → Task 4; InternalModel/registry entry → Tasks 0+5; layering (baseline→fused) → Task 3 then 4; "shape for all" → the request-struct *pattern* in Task 4 + curated exports reusable by qwen3/llama. All covered.
- **Build-loop-driven sets:** the exact 94-helper / 183-field lists are intentionally derived from `/tmp/sdk.err` at execution time (Task 0 Step 2), not statically frozen here — the branch tree is the source of truth, and a frozen list would drift. Patterns are shown in full; application is mechanical + build-verified.
- **Type consistency:** accessor names (`Keys`/`Values`/`Offset`/`Step`/`MaxSize`/`PageSize`/`Bits`) and `NewKVCacheFrom` are used consistently in Tasks 1 and 3; request-struct type names (`Gemma4FixedAttentionRequest`) are defined in Task 4 Step 1 and used in Step 2.
- **Harness caveat:** Task 6 Step 1 notes interactive rebase is unsupported in the harness — squash via staged re-commit of the final tree instead.

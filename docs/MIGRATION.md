# go-mlx → go-inference migration map

Produced by the 2026-07-03 test-audit campaign (final lane, imports-verified,
cross-checked against `external/go-inference`'s actual tree). The endgame this
maps onto:

1. Finish the pkg/native feature port (pkg/metal is the parity oracle until
   then, deleted after).
2. Reshape go-mlx to engine-only using this map.
3. Move `pkg/native` → `go-inference/engine/metal`; `pkg/model` follows into
   go-inference. The `lem` binary then compiles from go-inference alone.
4. go-mlx becomes the quarantine sandbox where the go-rocm hip++ port lands as
   `go/pkg/hip`; proven work migrates to `engine/hip` via audit-then-land.
   Unsupervised agents never edit go-inference.

Verdicts: **ENGINE-KEEP** rides with pkg/native into `engine/metal` ·
**MIGRATE-UP** belongs in go-inference (blockers named) ·
**DIES-WITH-METAL** deleted with the cgo engine.

| Package | Verdict | Blockers / notes |
|---|---|---|
| `go/` (root mlx) | MIXED | 18 files import `pkg/metal` directly (composition core included: `backend.go`, `mlx.go`, `session.go`, `eval.go`, `speculative.go`, `tokenizer.go`, `primitives.go`, `model_lora.go` — via concrete aliases like `type DeviceInfo = metal.DeviceInfo`); only 3 files import `pkg/native` — the native port is incomplete at this layer too. `register_metal*.go` + `metal_capabilities.go` → DIES-WITH-METAL. `register_native*.go`, `native_model.go`, `native_speculative_textmodel.go` → ENGINE-KEEP. The composition core is the MIGRATE-UP candidate, **but not a mechanical copy**: `go-inference/go/serving` is an independently designed backend-adapter layer (no go-mlx imports, talks through `inference.LoadModel`). **The single biggest open call of the merge: reconcile go-mlx composition into serving's shape, or the reverse.** `split_cpu_ffn*.go`, `split_executor.go`, `split_remote_ffn.go` are engine-import-free and portable; `split_native_runtime.go` imports pkg/metal. |
| `go/agent` | MIGRATE-UP | Clean (no engine imports). **Naming collision**: `go-inference/go/agent` exists and is semantically unrelated (SSH/worker/workflow) — rename on landing. |
| `go/artifact` | MIGRATE-UP | Clean; net-new in go-inference. |
| `go/bundle` | MIGRATE-UP | Clean; net-new in go-inference. |
| `go/pkg/daemon` | MIXED | Registry/Dispatch/wire types are backend-agnostic → MIGRATE-UP (go-inference/serving has no UDS/JSON-line daemon today — genuine gap). `native.go`'s concrete go-mlx runner glue → DIES-WITH-METAL once callers route through serving. |
| `go/kvconv` | DIES-WITH-METAL | Imports `mlx/kv` + `pkg/metal` for `metal.KVSnapshot`. pkg/native has **no** KVSnapshot equivalent yet — decide: native grows its own wire-shape converter, or consumes `kv.Snapshot` directly (cleaner). |
| `go/specprofile` | MIGRATE-UP | Own logic backend-agnostic (A/B profiling harness); blocked on the root composition-layer resolution (imports root `mlx` + `mlx/chat` types). |
| `go/model` | MIGRATE-UP | No engine imports, **not a straight copy**: still imports old go-mlx-local siblings (`mlx/gguf`, `mlx/pack`, `mlx/safetensors`, `mlx/profile`, `mlx/quant/autoround`) whose migrated equivalents already exist in go-inference (`inference/gguf`, `inference/safetensors`, `model/pack`, `modelmgmt`). Landing = re-pointing imports, not porting the siblings again. |
| `go/model/minimax/m2` | MIGRATE-UP | Clean. `m2_metal.go` is a CPU reference despite the name. Its `residency.go` (per-expert MoE weight residency) does not collide with `go-inference/go/residency` (per-model device residency). |
| `go/internal/sessionfake` | DIES-WITH-METAL (current form) | Sole import is `pkg/metal` (`metal.KVSnapshot` field). The concept must survive — re-point to `kv.Snapshot` so it outlives the metal deletion. |
| `go/internal/metaltest` | MIGRATE-UP | Trivial: build-tag gates + HF-cache walker, zero metal API coupling. Rename away from "metal" on landing; go-inference needs the identical gate pattern. |
| `go/cmd/mlx` | MIXED (45 files) | `admin_reload.go`/`admin_sft.go`/`admin_auth.go` skew ENGINE-KEEP / DIES-WITH-METAL. `admin_download.go`/`admin_hf.go` → MIGRATE-UP (backend-agnostic HF fetch+cache). CLI orchestration (`generate/audio/vision/diffuse/ebook/fuse/ssd*/pack/main`) migrates once the engine binding beneath it resolves. **Reconcile with `go-inference/go/cmd/lthn-model-pack`** (partial overlap with the pack subcommand) — no duplicate port. |
| `go/cmd/violet` | MIXED (thin) | Only imports core + `mlx/pkg/daemon`; fate follows pkg/daemon. |

Packages audited in earlier lanes and already dispositioned by prior work:
`gguf`, `safetensors`, `pkg/safetensors`, `pkg/tokenizer`, `pkg/scheme`,
`kv`, `blockcache`, `memory`, `substrate`, `probe`, `profile`, `adapter`,
`benchsummary`, `chat`, `chaptersmoke`, `dataset`, `ebook`, `hf`, `merge`,
`openai`, `pack`, `pkg/score`, `quant/autoround`, `internal/loraadapter`,
`train`, `lora`, `memorypretrain`, `distill`, `grpo`, `session`, `spine`,
`compute` — engine-agnostic members of this set follow the same MIGRATE-UP
path (several already have go-inference twins from the earlier push-up
lanes; reconcile, don't duplicate). `pkg/metal` and its model families are
DIES-WITH-METAL by definition; `pkg/native` + `pkg/model` are the engine
payload that moves to `engine/metal`.

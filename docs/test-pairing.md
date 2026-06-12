# Test ↔ source pairing map (go/)

The CoreGo convention pairs every test file with the source file it covers
(`<source>_test.go`, `<source>_bench_test.go`, `<source>_example_test.go`
beside `<source>.go`). This page is the one-place list of every test file
under `go/` that does NOT pair with a source file, after the 2026-06-12
orphan sweep relocated the genuinely lost ones
(`git log --grep="orphan sweep"`).

Regenerate the list (from `go/`):

```sh
python3 - <<'PY'
import os
SUFFIXES = ['_bench_test.go','_example_test.go','_internal_test.go','_live_test.go','_smoke_test.go','_golden_test.go','_test.go']
EXCLUDE = {'external','lib','.git','build','dist','testdata','.tmp'}
def base_of(n):
    for s in SUFFIXES:
        if n.endswith(s): return n[:-len(s)]
for root, dirs, files in os.walk('.'):
    dirs[:] = [d for d in dirs if d not in EXCLUDE]
    gofiles = set(f for f in files if f.endswith('.go'))
    sources = set(f[:-3] for f in gofiles if not f.endswith('_test.go'))
    for f in sorted(gofiles):
        if f.endswith('_test.go') and base_of(f) and base_of(f) not in sources:
            print(os.path.join(root, f))
PY
```

The audit's source→test direction (`core/go/tests/cli/v090-upgrade/audit.sh`)
currently reports **90 source files with no `<file>_test.go`** and **175 with
no `<file>_example_test.go`** — that is the AX-7 coverage lane, tracked
separately; this page tracks the test→source direction only.

## Deliberately unpaired — live / diagnostic instruments

Cross-file integration tests gated on a real model load
(`metaltest.RunMetalTests` / `_LiveModel` / metal-availability skips). They
exercise paths spanning many source files by design; pinning them to one
source file would be dishonest.

| File | What it exercises |
|------|-------------------|
| `compiled_layer_live_test.go` | compiled decode-layer vs eager parity (live model) |
| `compiled_layer_hits_live_test.go` | compiled-layer hit counters (live model) |
| `compiled_mlp_live_test.go` | compiled MLP parity (live model) |
| `det_probe_test.go` | decode-determinism instrument suite (all `_LiveModel`) |
| `mtp_live_test.go` | MTP assistant-pair speculative decode (live pair) |
| `serve_turn_phase_split_live_test.go` | serve turn phase split timing (live) |
| `substrate_parity_test.go` | substrate vs metal prompt-cache replay parity (live-gated) |
| `tests/smoke/small_model_smoke_test.go` | the supervised small-model smoke lane |

## Deliberately unpaired — shared fixtures and package-level examples

`testhelpers_test.go` / `*_test_helpers_test.go` / `*_testhelper_test.go` hold
shared fakes and skip-guards (the helper-file convention). `example_test.go`
files hold package-level `Example()` functions per Go's documented convention.

## Concern-named bench/feature files (subpackages)

The optimised packages group benches and regression tests by CONCERN rather
than by source file (e.g. `kv/dtype_bench_test.go`,
`pkg/metal/rope_bench_test.go`,
`pkg/metal/model/gemma4/decode_kernels_test.go`). These are findable by name
and deliberate; re-pairing them is churn without value. They are listed by
the regeneration snippet above — anything NEW should pair with its source
file instead of adding to this set.

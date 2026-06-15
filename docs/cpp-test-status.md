# Native MLX C++ test + benchmark status

The patched MLX (`lib/mlx`) carries `lthn patch` commits to the Metal backend,
scheduler, gguf I/O and compile cache. Those patches were never exercised by a
C++ test run, because **`lib/mlx-c/CMakeLists.txt` force-disables the suite**
(`set(MLX_BUILD_TESTS OFF)` / `EXAMPLES OFF` / `BENCHMARKS OFF`) before pulling
MLX in — so the main `cmake --build` can't reach a single test or benchmark, and
the `cpp/` CLion sandbox's cache is pinned to the pre-move repo path
(`~/Code/go-mlx`) and no longer configures.

## How to run

```bash
task test:cpp     # build patched lib/mlx STANDALONE + ctest
task bench:cpp    # build + run the 4 C++ benchmarks
```

Both build `lib/mlx` as its own top project (mlx-c out of the loop, so
`MLX_BUILD_TESTS`/`BENCHMARKS` enable normally) into `build/cpptest`. First run
cold-builds MLX (~15 min); subsequent runs are incremental.

## Test status — 239/246 pass, 7 red (as of dev `588f627d`)

| # | test | category | verdict |
|---|------|----------|---------|
| #221 | default stream in threads | patch `d02cc10b` "unbound threads adopt the process-canonical default stream" — the test still asserts the old per-thread-stream model (`thread_streams.size() == num_threads`) | **stale test, intentional divergence** — update/skip upstream test |
| #224 | thread local stream (subprocess aborted) | same stream-semantics patch family | **stale test** — same root |
| #245 | `tests` (subprocess aborted) | the aggregate doctest target — aborts after a sub-case crashes the process | **fallout**, not distinct |
| #12 | array shared buffer (**SEGFAULT**) | go-mlx does heavy custom array/buffer pooling; a segfault here is not explained by a known patch | **needs investigation — the real red flag** |
| #123 | gguf | `load_tests` expects empty metadata; patched `gguf.cpp` now populates it (go-mlx reads gguf metadata for model loading) | likely intentional — confirm + update test |
| #83 | export function with no inputs | function-graph serialise/import | unattributed — possibly upstream/env |
| #87 | export function with variable inputs | same | unattributed |

Net: only **#12 (segfault)** clearly warrants a root-cause pass. #221/#224 are
the cost of an intentional patch (the upstream tests encode the pre-patch
model); #123 likely the same for gguf metadata; #83/#87 unattributed.

## Benchmark baseline (M3 Ultra, gpu, Release, Metal, dev `588f627d`)

All 4 binaries run clean. Representative single-op timings:

- `single_ops`: elementwise/astype/reduce ops ~0.19–0.29 ms each on GPU.
- `irregular_strides`: strided/broadcast `add` ~0.23–0.32 ms.
- `autograd`: `value_and_grad` ~0.64–0.73 ms.
- `compare_devices`: small-size `add` — cpu ~0.015 ms vs gpu ~0.22 ms (dispatch
  floor dominates at tiny sizes; gpu wins as size grows).

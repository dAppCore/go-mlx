---
title: Metallib & build variants
description: What mlx.metallib is, why it must travel with the binary, the variant matrix, the bundling strategy, and the active CWD-resolution panic to work around.
---

# Metallib & build variants

`mlx.metallib` is a precompiled Metal GPU kernel archive (107 MB) that the MLX runtime loads at first GPU use. Without it, `lthn-mlx` panics inside `mlx_metal_load_library` the moment the model touches the GPU. Operators MUST know where it lives, which one to ship, and how the binary finds it at runtime — otherwise no model loads.

This doc covers four things:

1. **What it is** and the boundary it crosses.
2. **The variant matrix** — what actually differs between builds (chip family? macOS version? toolchain?).
3. **Bundling strategy** — three paths, the recommended one, and why.
4. **The CWD-resolution panic** that affects every build before the bundling work lands, and the env-var workaround.

---

## What it is

The metallib is the compiled output of `lib/mlx/mlx/backend/metal/kernels/` — every `.metal` source compiled to `.air`, then linked into one archive by `xcrun metallib`. MLX's C++ runtime calls `[MTLDevice newLibraryWithURL:]` against the path set in the `MLX_METALLIB_PATH` env var (or the binary-relative search path resolved by Go — see "Resolution" below) to load the archive, then dispatches named kernels by string lookup.

The committed metallib in `dist/lib/mlx.metallib` (107510692 bytes, MetalLib v1.2.9) was built from upstream MLX `v0.31.1` (the pinned submodule at `lib/mlx/`) on a baseline Apple toolchain. The duplicate at `build/_deps/mlx-build/mlx/backend/metal/kernels/mlx.metallib` (123677723 bytes) is a build-tree artefact from the local CMake run on this host — slightly larger because of unstripped debug paths.

**Why two on disk:** the `dist/lib/` copy is the install-tree artefact (the one consumers should use); the `build/_deps/` copy is the CMake build-tree artefact. They are semantically the same content, different containers. The Go runtime currently finds either via the CWD walk; the install-tree copy is canonical.

---

## The variant matrix

Snider asked: "if the lib is different for different apple versions, we need to know the variants that need building." Answer: **the chip family axis doesn't matter — Apple's Metal driver forward-compatibility handles M1→M5 from a single archive. The axis that matters is the build-host toolchain.** Specifically:

| Axis | Where decided | What changes in the metallib |
|------|---------------|------------------------------|
| **Metal language version** (≥320 unlocks `fence`; ≥400 + macOS SDK ≥26.2 unlocks the `nax` kernel family) | Detected at CMake configure from `xcrun -sdk macosx metal -E`. Effectively driven by installed Xcode / CommandLineTools version. | Which kernels exist in the archive. NAX kernels are the tensor-coprocessor fast paths (GEMM, attention, quantised matmul) — present on M4 onward, baseline for M5. |
| **macOS deployment target** | `CMAKE_OSX_DEPLOYMENT_TARGET` at CMake configure → `-mmacosx-version-min=…` per `.metal` compile | The earliest macOS runtime that will load this archive. Going lower is a downgrade; going higher is an upgrade-lock. |
| **MLX_METAL_JIT** | CMake option, default OFF | When ON, MLX compiles many kernels in-process at runtime instead of baking them into the metallib. The metallib still exists for the non-JIT'd subset, but is smaller. We do **not** use JIT mode — it pushes per-process startup cost into every consumer. |

The `26.0` deployment floor is intentional rather than a convenience default:
the native go-mlx path is aligned to Apple's Metal 4 API generation, which is
documented for macOS Tahoe 26 and includes the command API, explicit compiler
control, tensor resources, and machine-learning passes this lane is preparing
to use.

Reference links:

- [macOS Tahoe 26 release notes](https://developer.apple.com/documentation/macos-release-notes/macos-26-release-notes)
- [What's new in macOS 26](https://developer.apple.com/macos/whats-new/)
- [What's new in Metal](https://developer.apple.com/metal/whats-new/)
- [Understanding the Metal 4 core API](https://developer.apple.com/documentation/metal/understanding-the-metal-4-core-api)
- [Using the Metal 4 compilation API](https://developer.apple.com/documentation/metal/using-the-metal-4-compilation-api)
- [Metal machine learning passes](https://developer.apple.com/documentation/metal/machine-learning-passes)
- [Metal feature set tables](https://developer.apple.com/metal/capabilities/)

Evidence for the kernel-conditional behaviour (`lib/mlx/mlx/backend/metal/kernels/CMakeLists.txt:57,157`):

```cmake
if(MLX_METAL_VERSION GREATER_EQUAL 320)
  build_kernel(fence)
endif()

if((MLX_METAL_VERSION GREATER_EQUAL 400) AND (MACOS_SDK_VERSION GREATER_EQUAL 26.2))
  build_kernel(steel/gemm/kernels/steel_gemm_fused_nax  …)
  build_kernel(steel/gemm/kernels/steel_gemm_gather_nax …)
  build_kernel(steel/gemm/kernels/steel_gemm_splitk_nax …)
  build_kernel(quantized_nax         …)
  build_kernel(fp_quantized_nax      …)
  build_kernel(steel/attn/kernels/steel_attention_nax …)
else()
  target_compile_definitions(mlx PRIVATE MLX_METAL_NO_NAX)
endif()
```

### The practical ship matrix

Two variants cover everything we currently care about:

| Variant | Build conditions | Runs on | Use case |
|---------|------------------|---------|----------|
| **`mlx-nax.metallib`** | Metal ≥4.0 + SDK ≥26.2 (Xcode 26+), macOS deployment-min 26 | M1/M2/M3/M4/M5 on macOS 26+ ; NAX kernels dispatch on M4 + M5 | **Default ship.** M4 and M5 must dispatch tensor-coprocessor kernels — that's the entire perf advantage of the current two generations. Without NAX present, M4/M5 run M1-class kernels and the customer paid for hardware they don't get to use. |
| **`mlx-legacy.metallib`** | Metal ≥3.2 toolchain, macOS deployment-min 13 | M1/M2/M3/M4/M5 on macOS 13-25 | Legacy fallback for operators on macOS 13-25. Ship alongside NAX only when those operators exist. |

**Chip-family note:** there is no per-chip variant within a metallib. The Metal driver picks the right kernel encoding for the chip the program is running on; one archive serves M1 through M5. The NAX kernels in the default variant only *dispatch* on M4 + M5, but their presence/absence is a build-toolchain decision, not a runtime-target decision.

### Confidence + open questions

This matrix is **~85% confidence**. Three unknowns remain:

1. **Does the Metal driver refuse to load an entire metallib whose `-mmacosx-version-min` is higher than the runtime OS, or does it just refuse the affected kernels?** Likely whole-library reject; resolves if/when we encounter it.
2. **NAX kernel dispatch on M1-M3 hardware running the NAX metallib** — MLX must gate at dispatch time so M1-M3 chips fall back to the standard kernel path. Read of `lib/mlx/mlx/backend/metal/` dispatch code resolves it in ~20 min.
3. **M5 tensor-kernel API delta vs M4 NAX** — Apple shipped M5 with refined Neural Accelerators. The Metal-4 NAX symbol set is forward-compatible (M5 runs M4-generated NAX kernels), but if SDK 27+ exposes M5-specific kernels with measurable wins, a third variant could be warranted. Open until perf data justifies the split.

### How to identify what you have

```bash
file dist/lib/mlx.metallib
# MetalLib executable (MacOS), version 1.2.9
```

`version 1.2.9` is the MetalLib *container format* version (set by Apple's `metallib` tool), not the Metal language version. To inspect kernel contents:

```bash
xcrun metal-objdump --section-headers dist/lib/mlx.metallib | head -40
xcrun metal-objdump --symbols dist/lib/mlx.metallib | grep -i nax
# empty output = baseline metallib (no NAX kernels)
```

If `grep -i nax` returns symbols, you have the NAX-enabled variant.

---

## Bundling strategy

The metallib has to travel with the `lthn-mlx` binary. Three paths exist; the brief sketched all three. Recommendation + rationale below.

### Path A — embed → extract to `$TMPDIR/mlx-XXXX/` at startup

```go
//go:embed mlx.metallib
var metallibBytes []byte

func init() {
    dir, _ := os.MkdirTemp("", "mlx-")
    path := filepath.Join(dir, "mlx.metallib")
    os.WriteFile(path, metallibBytes, 0o644)
    os.Setenv("MLX_METALLIB_PATH", path)
}
```

- **Pros:** zero C++ change. Ships in one to two hours of work. Pure Go side.
- **Cons:** 107 MB extract on every process start. `$TMPDIR` is RAM-backed on some macOS configs (`/private/var/folders/…`), so the extract pressures the unified memory pool. Cleanup is best-effort — a crashed binary leaves the temp file behind until the OS sweeps. There's a brief filesystem race window where two binaries starting simultaneously could collide on the same temp dir (mitigated by `MkdirTemp` randomness).

### Path B — embed → bytes through CGO → `MTLDevice newLibraryWithData:`

```go
//go:embed mlx.metallib
var metallibBytes []byte

func init() {
    metal.SetMetallibBytes(metallibBytes) // new symbol — bridges into C++
}
```

C++ side gets a new helper `mlx_metal_load_library_from_data(const void *bytes, size_t len)` that wraps:

```objc
dispatch_data_t data = dispatch_data_create(bytes, len,
    dispatch_get_global_queue(QOS_CLASS_DEFAULT, 0), DISPATCH_DATA_DESTRUCTOR_DEFAULT);
id<MTLLibrary> lib = [device newLibraryWithData:data error:&err];
```

- **Pros:** one binary, one file. No temp artefact. No filesystem race. No `$TMPDIR` pressure. The Metal API is purpose-built for this — `newLibraryWithData:` is not a workaround. Matches Snider's "the actual model is the binary" boundary rule (the explicit 2026-05-25 framing in the brief).
- **Cons:** requires a `internal/metal/` C++ change. Adds one symbol to the cgo boundary. `dispatch_data_create` needs the destructor signal-flagged carefully so the Go GC doesn't reclaim `metallibBytes` while MLX is still reading it — straightforward with `runtime.KeepAlive` on the Go side and `DISPATCH_DATA_DESTRUCTOR_DEFAULT` (which makes a copy) on the C side.

### Path C — sidecar file next to binary

```
/usr/local/bin/lthn-mlx
/usr/local/bin/mlx.metallib
```

- **Pros:** simplest possible. Predictable.
- **Cons:** two artefacts to ship and not lose track of. Breaks Snider's one-binary boundary rule. Creates a new operator-error class — "deploy the binary, forget the metallib, runtime panic at first GPU dispatch." Not viable for App Store distribution where the bundle has to be self-contained.

### Recommendation

**Pick B as the canonical path, ship A first as the unblock, keep `MLX_METALLIB_PATH` as the dev override.**

Sequencing:

1. **Today / next session:** ship Path A. Unblocks the running-from-anywhere problem (see "CWD-resolution panic" below) in one to two hours. Functions as the immediate fix.
2. **Following session:** land Path B as the canonical replacement. A stops being used in production builds; the env var override survives for development workflows where you want to swap in a freshly-built metallib without rebuilding the Go binary.
3. **NAX as default ship:** done. NAX-class is the current baseline (M4 + M5 hardware, macOS 26+). The legacy variant exists for operators on macOS 13-25; ship it only when you have telemetry showing those operators exist.

Reasoning for B-over-A long-term: every process restart paying 107 MB of file IO + memory pressure is a real cost when this becomes a daemon. `newLibraryWithData:` skips it entirely — MLX maps directly off the embedded bytes via the Go-side `[]byte` pinned through one `runtime.KeepAlive`.

---

## The CWD-resolution panic (active blocker)

Until Path A or B lands, `lthn-mlx` only runs cleanly when invoked from inside the `core/go-mlx/` source checkout. From any other CWD it panics on first GPU dispatch.

### What's happening

`go/internal/metal/metal.go:204-224` (`defaultMetallibPath`) walks up to five levels above the process CWD looking for `dist/lib/mlx.metallib`:

```go
func defaultMetallibPath() string {
    const metallib = "mlx.metallib"
    var candidates []string
    if wd := core.Getwd(); wd.OK {
        root := wd.Value.(string)
        candidates = append(candidates,
            core.PathJoin(root, "dist", "lib", metallib),
            core.PathJoin(root, "..", "dist", "lib", metallib),
            // ... up to ../../../../../dist/lib/mlx.metallib
        )
    }
    for _, candidate := range candidates {
        if core.Stat(candidate).OK {
            return candidate
        }
    }
    return metallib // fallback — relative path, will not resolve
}
```

When `lthn-mlx` lives at `/usr/local/bin/lthn-mlx` and CWD is `~/projects/myapp/`, every candidate is `~/projects/myapp/[..]/dist/lib/mlx.metallib` and every one misses. The fallback returns `"mlx.metallib"` — a relative path that the Metal runtime then tries to resolve against the process CWD, fails, and panics inside `mlx_metal_load_library`.

This bug only didn't surface during dev because everyone's been invoking the binary from inside the repo, where the walk hits.

### Workaround until bundling lands

Set `MLX_METALLIB_PATH` to an absolute path before invoking:

```bash
export MLX_METALLIB_PATH=/Users/snider/Code/core/go-mlx/dist/lib/mlx.metallib
lthn-mlx serve --model /Volumes/Data/models/lemer-lite --addr :11434
```

Or inline for a single invocation:

```bash
MLX_METALLIB_PATH=/abs/path/mlx.metallib lthn-mlx serve --model … --addr :11434
```

The env var is checked at `metal.go:287` before the CWD walk fires, so a set path bypasses the buggy resolution entirely.

### Deployment guidance for systemd / launchd / Docker

Until bundling lands, **deployment scripts must set `MLX_METALLIB_PATH` explicitly**. Don't rely on the binary finding its own metallib. Pattern for a launchd plist:

```xml
<key>EnvironmentVariables</key>
<dict>
    <key>MLX_METALLIB_PATH</key>
    <string>/opt/lthn-mlx/lib/mlx.metallib</string>
</dict>
```

And ship the file there as part of the install package.

---

## Sources

- `go/internal/metal/metal.go:204-300` — CWD walk + env var precedence
- `lib/mlx/mlx/backend/metal/kernels/CMakeLists.txt:24,57,157` — kernel-set conditionals
- `lib/mlx/CMakeLists.txt:202` — Metal version detection via `xcrun metal -E`
- `dist/lib/mlx.metallib` + `build/_deps/mlx-build/mlx/backend/metal/kernels/mlx.metallib` — the two on-disk artefacts

## Cross-references

- [Deployment](deployment.md) — where to put the metallib in a real install
- [Troubleshooting](troubleshooting.md) — the panic signatures + what they mean

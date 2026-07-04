<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# compute.go — frame-compute API (non-LLM Metal)

**Package**: `dappco.re/go/mlx`
**File**: `go/compute.go` (plus `compute_darwin.go` / `compute_stub.go`)

## What this is

The **non-LLM Metal compute** surface — pixel buffers, kernels, frame pipelines. Lets callers use Apple GPU acceleration for **image / emulator / signal-processing workloads** without going through the LLM inference stack.

Origin: CoreAgent wants to ship retro-emulator UIs in its sub-apps (Nintendo, Mega Drive, etc.); those need fast image filters (CRT, scanline, nearest scale, soften, sharpen). Reusing the LLM Metal context for these saves the cost of a separate compute framework + duplicate device init.

## Public surface

```go
session, err := mlx.NewSession(mlx.WithSessionLabel("frame-pipeline"))
defer session.Close()

src, err := session.NewPixelBuffer(mlx.PixelBufferDesc{
    Width: 320, Height: 224, Stride: 640,
    Format: mlx.PixelRGB565,
})

dst, err := session.NewPixelBuffer(...)

err = session.BeginFrame()
err = session.RunKernel(mlx.KernelRGB565ToRGBA8, src, dst)
err = session.RunKernel(mlx.KernelCRTFilter, dst, dst)
err = session.FinishFrame()
```

## Pixel formats

| Format | Bits | Use |
|--------|------|-----|
| `PixelRGB565` | 16 | classic console framebuffer |
| `PixelRGBA8` | 32 | macOS native |
| `PixelBGRA8` | 32 | alternative byte order |
| `PixelGray8` | 8 | luminance-only |

## Kernels shipped

| Kernel | Effect |
|--------|--------|
| `KernelRGB565ToRGBA8` | colourspace convert |
| `KernelNearestScale` | upscale without smoothing |
| `KernelScanlineFilter` | CRT-style scanlines |
| `KernelCRTFilter` | full CRT emulation (mask + glow) |
| `KernelSoftenFilter` | gaussian blur |
| `KernelSharpenFilter` | sharpen mask |

Custom kernels can be registered at session init via `WithKernel(...)`.

## Session / Frame lifecycle

```go
session.BeginFrame()       // open the Metal command buffer
session.RunKernel(...)     // queue dispatches
session.RunKernel(...)
session.FinishFrame()      // commit + wait
```

Frame-coalesced — multiple kernel dispatches share one Metal command buffer, one commit, one wait. The win: a six-stage filter pipeline costs one frame round-trip, not six.

## Error model

Compute errors are typed (`ComputeErrorKind` enum + `*ComputeError` instances). Callers can check `errors.Is(err, mlx.ErrComputeClosed)` etc. without parsing strings.

The error kinds cover the failure shapes:

- `unavailable` — no Metal device
- `closed` — session already closed
- `invalid_state` — operation called out of order (kernel before BeginFrame)
- `invalid_descriptor` — buffer/kernel descriptor doesn't validate
- `unsupported_pixel_format` — kernel can't handle this format
- `buffer_size_mismatch` — kernel inputs don't agree on size
- `unknown_kernel` — kernel name not registered
- `internal` — Metal returned an error from the C side

## Why share with the LLM stack

Three reasons:

1. **One Metal device init.** Both LLM and frame-compute share `metal.GetDeviceInfo()` + the allocator.
2. **Shared memory budget.** When the LLM is hot, frame compute throttles; when frame is hot, LLM scheduler backs off.
3. **One package import.** Sub-apps that mix LLM ops (text-to-image prompt) and frame ops (filter the image) don't dual-bind.

## Status

Production for the six shipped kernels. Custom-kernel registration: planned. Image-generation kernels (diffusion-style): out of scope for the core runner.

## Related

- `../runtime/register_metal.md` — shared Metal device init
- `internal/metal/` — actual Metal kernel implementations
- CoreAgent retro-emulator sub-apps (not in this repo) — primary consumer

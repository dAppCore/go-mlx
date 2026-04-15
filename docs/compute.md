---
title: Compute
description: Frame-oriented Metal compute sessions, pixel buffers, kernels, and metrics.
---

# Compute

`dappco.re/go/mlx` exposes a small non-LLM compute surface for frame and image workloads on Apple Silicon. It is intended for pipelines such as emulator framebuffer conversion, low-copy pixel processing, and post-processing around an existing renderer or presenter.

This surface deliberately stays out of:

- window creation
- presentation and swapchains
- input and audio
- process management

Those concerns belong in downstream packages such as `core/gui` or `core/play`.

## API Shape

```go
type Compute interface {
    Available() bool
    DeviceInfo() DeviceInfo
    NewSession(opts ...SessionOption) (Session, error)
}

type Session interface {
    Close() error
    NewPixelBuffer(desc PixelBufferDesc) (PixelBuffer, error)
    NewByteBuffer(size int) (ByteBuffer, error)
    Run(kernel string, args KernelArgs) error
    Sync() error
    Metrics() SessionMetrics
}
```

Use `mlx.DefaultCompute()` when you want an explicit backend handle, or `mlx.NewSession()` when the package default is sufficient.

## Pixel Buffers

Pixel buffers are packed byte buffers with explicit width, height, stride, and format metadata:

```go
desc := mlx.PixelBufferDesc{
    Width:  320,
    Height: 224,
    Stride: 640,
    Format: mlx.PixelRGB565,
}
buf, err := session.NewPixelBuffer(desc)
```

Supported formats:

- `PixelRGBA8`
- `PixelBGRA8`
- `PixelRGB565`
- `PixelXRGB8888`
- `PixelIndexed8`

`Upload` copies Go-managed bytes into device-backed storage. `Read` synchronises the session and copies the current device contents back into Go memory.

## Kernels

The built-in kernels are string constants in the root package:

| Constant | Purpose |
|----------|---------|
| `KernelNearestScale` | Nearest-neighbour scaling for packed pixel buffers |
| `KernelIntegerScale` | Integer-factor nearest-neighbour scaling |
| `KernelBilinearScale` | Bilinear scaling for `rgba8` and `bgra8` |
| `KernelRGB565ToRGBA8` | RGB565 to RGBA8 conversion |
| `KernelRGBA8ToBGRA8` | RGBA/BGRA channel swap |
| `KernelBGRA8ToRGBA8` | BGRA/RGBA channel swap |
| `KernelXRGB8888ToRGBA8` | XRGB8888 to RGBA8 conversion |
| `KernelPaletteExpandRGBA` | Indexed 8-bit source plus RGBA palette to RGBA8 |

## Example Pipeline

This is the intended shape for emulator-style frame processing:

```go
session, err := mlx.NewSession(mlx.WithSessionLabel("retro-frame"))
if err != nil {
    panic(err)
}
defer session.Close()

src, _ := session.NewPixelBuffer(mlx.PixelBufferDesc{
    Width:  320,
    Height: 224,
    Stride: 640,
    Format: mlx.PixelRGB565,
})
rgba, _ := session.NewPixelBuffer(mlx.PixelBufferDesc{
    Width:  320,
    Height: 224,
    Stride: 1280,
    Format: mlx.PixelRGBA8,
})
scaled, _ := session.NewPixelBuffer(mlx.PixelBufferDesc{
    Width:  960,
    Height: 672,
    Stride: 3840,
    Format: mlx.PixelRGBA8,
})

frameBytes := make([]byte, src.Descriptor().SizeBytes())
if err := src.Upload(frameBytes); err != nil {
    panic(err)
}
if err := session.Run(mlx.KernelRGB565ToRGBA8, mlx.KernelArgs{
    Inputs:  map[string]mlx.Buffer{"src": src},
    Outputs: map[string]mlx.Buffer{"dst": rgba},
}); err != nil {
    panic(err)
}
if err := session.Run(mlx.KernelIntegerScale, mlx.KernelArgs{
    Inputs:  map[string]mlx.Buffer{"src": rgba},
    Outputs: map[string]mlx.Buffer{"dst": scaled},
}); err != nil {
    panic(err)
}
if err := session.Sync(); err != nil {
    panic(err)
}

presentable, err := scaled.Read()
if err != nil {
    panic(err)
}
_ = presentable
```

## Metrics

Each session accumulates coarse timing and memory figures:

```go
metrics := session.Metrics()
fmt.Println(metrics.Passes, metrics.LastKernel)
fmt.Println(metrics.LastDispatchDuration, metrics.LastSyncDuration)
fmt.Println(metrics.ActiveMemoryBytes, metrics.PeakMemoryBytes)
```

These metrics are designed for runtime policy decisions such as:

- enable GPU filters only on capable devices
- disable heavier passes when memory is tight
- fall back to CPU processing when a frame budget is exceeded

## Availability and Fallback

On unsupported builds, `mlx.DefaultCompute().Available()` returns `false` and `mlx.NewSession()` returns an availability error. Consumers should treat CPU fallback as an ordinary path rather than an exceptional one.

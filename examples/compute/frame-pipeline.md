# Frame Compute Pipeline

go-mlx exposes a non-LLM Metal compute API for image and emulator workloads. A `Session` owns its own command queue; `PixelBuffer` instances live on the Metal heap; `Run` dispatches a kernel against named input/output buffers and optional scalars.

## End-To-End: Emulator Frame Upscale + Scanline Filter

```go
package main

import (
    "fmt"
    "log"

    mlx "dappco.re/go/mlx"
)

func main() {
    session, err := mlx.NewSession(mlx.WithSessionLabel("emu-frame"))
    if err != nil { log.Fatal(err) }
    defer session.Close()

    // Source: 320x224 RGB565 (NES/SNES-style native resolution).
    src, err := session.NewPixelBuffer(mlx.PixelBufferDesc{
        Width:  320,
        Height: 224,
        Stride: 640,
        Format: mlx.PixelRGB565,
    })
    if err != nil { log.Fatal(err) }

    // Intermediate: 320x224 RGBA8 after format swizzle.
    rgba, err := session.NewPixelBuffer(mlx.PixelBufferDesc{
        Width:  320,
        Height: 224,
        Stride: 1280,
        Format: mlx.PixelRGBA8,
    })
    if err != nil { log.Fatal(err) }

    // Output: 960x672 RGBA8 at 3x scale.
    scaled, err := session.NewPixelBuffer(mlx.PixelBufferDesc{
        Width:  960,
        Height: 672,
        Stride: 3840,
        Format: mlx.PixelRGBA8,
    })
    if err != nil { log.Fatal(err) }

    // Upload the raw frame from the emulator.
    frame := getFrameFromEmulator() // []byte, length matches src.Descriptor().SizeBytes()
    if err := src.Upload(frame); err != nil { log.Fatal(err) }

    // Build one frame.
    if err := session.BeginFrame(); err != nil { log.Fatal(err) }

    // 1. RGB565 → RGBA8.
    if err := session.Run(mlx.KernelRGB565ToRGBA8, mlx.KernelArgs{
        Inputs:  map[string]mlx.Buffer{"src": src},
        Outputs: map[string]mlx.Buffer{"dst": rgba},
    }); err != nil { log.Fatal(err) }

    // 2. Nearest-neighbour 3x scale.
    if err := session.Run(mlx.KernelNearestScale, mlx.KernelArgs{
        Inputs:  map[string]mlx.Buffer{"src": rgba},
        Outputs: map[string]mlx.Buffer{"dst": scaled},
    }); err != nil { log.Fatal(err) }

    // 3. Apply scanline filter in-place on the scaled buffer.
    if err := session.Run(mlx.KernelScanlineFilter, mlx.KernelArgs{
        Inputs:  map[string]mlx.Buffer{"src": scaled},
        Outputs: map[string]mlx.Buffer{"dst": scaled},
        Scalars: map[string]float64{"strength": 0.3},
    }); err != nil { log.Fatal(err) }

    metrics, err := session.FinishFrame()
    if err != nil { log.Fatal(err) }

    out, err := scaled.Read()
    if err != nil { log.Fatal(err) }
    fmt.Printf("frame: %d bytes, gpu_time=%v\n", len(out), metrics.GPUDuration)
    presentToScreen(out)
}
```

## Available Kernels

| Kernel | Purpose |
|--------|---------|
| `KernelRGB565ToRGBA8` | Format swizzle (16-bit packed → 32-bit RGBA) |
| `KernelNearestScale`  | Integer nearest-neighbour upscale |
| `KernelScanlineFilter` | Darkens alternate rows (`strength` 0.0–1.0) |
| `KernelCRTFilter`     | CRT-style barrel + scanline + chromatic aberration |
| `KernelSoftenFilter`  | Slight gaussian blur |
| `KernelSharpenFilter` | Unsharp-mask sharpening |

## Per-Frame Lifecycle

```
NewSession → NewPixelBuffer ×N → loop {
    Upload(input)
    BeginFrame
    Run(...) ×K
    metrics = FinishFrame
    output = buffer.Read
}
→ Close
```

`BeginFrame` / `FinishFrame` is a single command-buffer commit + completion barrier. All kernel dispatches between them are batched into one Metal submission, which is the GPU-efficient way to drive 60 FPS pipelines.

## Reusing Buffers Between Frames

`PixelBuffer` instances are reusable. Allocate once at session setup, then reuse on every frame — no per-frame allocation, no GC pressure on the Metal heap.

## See Also

- [Compute Guide](../../docs/compute.md) — full reference
- [README Frame Compute section](../../README.md#frame-compute) — minimal example with code

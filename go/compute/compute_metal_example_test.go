// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package compute

import core "dappco.re/go"

// exampleSession opens a compute session for the runnable examples and arranges
// for it to close when the example returns. Examples cannot skip, so this
// assumes a usable Metal device — the same assumption the Apple-only compute
// package makes everywhere else.
func exampleSession() (Session, func()) {
	session, err := NewSession(WithSessionLabel("example"))
	if err != nil {
		core.Println("session error:", err)
		return nil, func() {}
	}
	return session, func() { _ = session.Close() }
}

// ExampleDefaultCompute reports whether the Metal compute backend is available
// on this device. On Apple silicon with the Metal runtime linked it is always
// available.
func ExampleDefaultCompute() {
	core.Println(DefaultCompute().Available())
	// Output: true
}

// ExampleNewSession opens a compute session backed by the default Metal
// backend and closes it. A fresh session reports no recorded passes yet.
func ExampleNewSession() {
	session, err := NewSession()
	if err != nil {
		core.Println("error:", err)
		return
	}
	core.Println("passes:", session.Metrics().Passes)
	core.Println("close:", session.Close())
	// Output:
	// passes: 0
	// close: <nil>
}

// ExampleSession_NewByteBuffer allocates a device byte buffer, uploads a small
// payload, and reads it straight back — the generic round-trip the palette and
// scalar inputs ride on. Read forces a device sync so the bytes are settled.
func ExampleSession_NewByteBuffer() {
	session, done := exampleSession()
	defer done()

	buffer, err := session.NewByteBuffer(4)
	if err != nil {
		core.Println("alloc error:", err)
		return
	}
	if err := buffer.Upload([]byte{1, 2, 3, 4}); err != nil {
		core.Println("upload error:", err)
		return
	}
	got, err := buffer.Read()
	if err != nil {
		core.Println("read error:", err)
		return
	}
	core.Println("size:", buffer.Size())
	core.Println("bytes:", got)
	// Output:
	// size: 4
	// bytes: [1 2 3 4]
}

// ExampleSession_NewPixelBuffer allocates a packed pixel buffer and reports the
// descriptor it was created from.
func ExampleSession_NewPixelBuffer() {
	session, done := exampleSession()
	defer done()

	buffer, err := session.NewPixelBuffer(PixelBufferDesc{Width: 2, Height: 2, Stride: 8, Format: PixelRGBA8})
	if err != nil {
		core.Println("alloc error:", err)
		return
	}
	desc := buffer.Descriptor()
	core.Println(desc.Width, desc.Height, desc.Stride, desc.Format, buffer.Size())
	// Output: 2 2 8 rgba8 16
}

// ExampleSession_Run dispatches the rgb565_to_rgba8 kernel over a synthetic
// two-pixel source (packed red, then green) and reads the unpacked RGBA8
// result. Run is the single entry point for every named compute kernel.
func ExampleSession_Run() {
	session, done := exampleSession()
	defer done()

	src, err := session.NewPixelBuffer(PixelBufferDesc{Width: 2, Height: 1, Stride: 4, Format: PixelRGB565})
	if err != nil {
		core.Println("src error:", err)
		return
	}
	dst, err := session.NewPixelBuffer(PixelBufferDesc{Width: 2, Height: 1, Stride: 8, Format: PixelRGBA8})
	if err != nil {
		core.Println("dst error:", err)
		return
	}
	if err := src.Upload([]byte{0x00, 0xF8, 0xE0, 0x07}); err != nil {
		core.Println("upload error:", err)
		return
	}
	if err := session.Run(KernelRGB565ToRGBA8, KernelArgs{
		Inputs:  map[string]Buffer{"src": src},
		Outputs: map[string]Buffer{"dst": dst},
	}); err != nil {
		core.Println("run error:", err)
		return
	}
	got, err := dst.Read()
	if err != nil {
		core.Println("read error:", err)
		return
	}
	core.Println(got)
	// Output: [255 0 0 255 0 255 0 255]
}

// ExampleSession_Run_filter applies the scanline filter at half strength to a
// 1x2 grey column. The kernel darkens odd rows, so the second pixel halves
// while the first is left untouched. Filters take their amount through the
// Scalars map.
func ExampleSession_Run_filter() {
	session, done := exampleSession()
	defer done()

	src, err := session.NewPixelBuffer(PixelBufferDesc{Width: 1, Height: 2, Stride: 4, Format: PixelRGBA8})
	if err != nil {
		core.Println("src error:", err)
		return
	}
	dst, err := session.NewPixelBuffer(PixelBufferDesc{Width: 1, Height: 2, Stride: 4, Format: PixelRGBA8})
	if err != nil {
		core.Println("dst error:", err)
		return
	}
	if err := src.Upload([]byte{200, 200, 200, 255, 200, 200, 200, 255}); err != nil {
		core.Println("upload error:", err)
		return
	}
	if err := session.Run(KernelScanlineFilter, KernelArgs{
		Inputs:  map[string]Buffer{"src": src},
		Outputs: map[string]Buffer{"dst": dst},
		Scalars: map[string]float64{"strength": 0.5},
	}); err != nil {
		core.Println("run error:", err)
		return
	}
	got, err := dst.Read()
	if err != nil {
		core.Println("read error:", err)
		return
	}
	core.Println(got)
	// Output: [200 200 200 255 100 100 100 255]
}

// ExampleSession_Sync flushes the pending dispatch queue without reading any
// buffer back, then reports the recorded sync via session metrics. Run records
// a pass; Sync settles it on the device.
func ExampleSession_Sync() {
	session, done := exampleSession()
	defer done()

	src, err := session.NewPixelBuffer(PixelBufferDesc{Width: 1, Height: 1, Stride: 2, Format: PixelRGB565})
	if err != nil {
		core.Println("src error:", err)
		return
	}
	dst, err := session.NewPixelBuffer(PixelBufferDesc{Width: 1, Height: 1, Stride: 4, Format: PixelRGBA8})
	if err != nil {
		core.Println("dst error:", err)
		return
	}
	if err := src.Upload([]byte{0x00, 0xF8}); err != nil {
		core.Println("upload error:", err)
		return
	}
	if err := session.Run(KernelRGB565ToRGBA8, KernelArgs{
		Inputs:  map[string]Buffer{"src": src},
		Outputs: map[string]Buffer{"dst": dst},
	}); err != nil {
		core.Println("run error:", err)
		return
	}
	core.Println("sync:", session.Sync())
	core.Println("last kernel:", session.Metrics().LastKernel)
	// Output:
	// sync: <nil>
	// last kernel: rgb565_to_rgba8
}

// ExampleSession_Metrics reports the cumulative pass count and last kernel name
// after a single dispatch-and-sync cycle.
func ExampleSession_Metrics() {
	session, done := exampleSession()
	defer done()

	src, err := session.NewPixelBuffer(PixelBufferDesc{Width: 1, Height: 1, Stride: 4, Format: PixelRGBA8})
	if err != nil {
		core.Println("src error:", err)
		return
	}
	dst, err := session.NewPixelBuffer(PixelBufferDesc{Width: 1, Height: 1, Stride: 4, Format: PixelBGRA8})
	if err != nil {
		core.Println("dst error:", err)
		return
	}
	if err := src.Upload([]byte{1, 2, 3, 4}); err != nil {
		core.Println("upload error:", err)
		return
	}
	if err := session.Run(KernelRGBA8ToBGRA8, KernelArgs{
		Inputs:  map[string]Buffer{"src": src},
		Outputs: map[string]Buffer{"dst": dst},
	}); err != nil {
		core.Println("run error:", err)
		return
	}
	if err := session.Sync(); err != nil {
		core.Println("sync error:", err)
		return
	}
	metrics := session.Metrics()
	core.Println("passes:", metrics.Passes)
	core.Println("last kernel:", metrics.LastKernel)
	// Output:
	// passes: 1
	// last kernel: rgba8_to_bgra8
}

// ExampleSession_BeginFrame opens an explicit frame, dispatches one kernel into
// it, and lets FinishFrame report the per-frame metrics. Within a frame the
// dispatches batch together and surface as a single frame index.
func ExampleSession_BeginFrame() {
	session, done := exampleSession()
	defer done()

	src, err := session.NewPixelBuffer(PixelBufferDesc{Width: 1, Height: 1, Stride: 4, Format: PixelRGBA8})
	if err != nil {
		core.Println("src error:", err)
		return
	}
	dst, err := session.NewPixelBuffer(PixelBufferDesc{Width: 1, Height: 1, Stride: 4, Format: PixelBGRA8})
	if err != nil {
		core.Println("dst error:", err)
		return
	}
	if err := src.Upload([]byte{10, 20, 30, 40}); err != nil {
		core.Println("upload error:", err)
		return
	}
	if err := session.BeginFrame(); err != nil {
		core.Println("begin error:", err)
		return
	}
	if err := session.Run(KernelRGBA8ToBGRA8, KernelArgs{
		Inputs:  map[string]Buffer{"src": src},
		Outputs: map[string]Buffer{"dst": dst},
	}); err != nil {
		core.Println("run error:", err)
		return
	}
	frame, err := session.FinishFrame()
	if err != nil {
		core.Println("finish error:", err)
		return
	}
	core.Println("frame:", frame.Frame)
	core.Println("passes:", frame.Passes)
	core.Println("last kernel:", frame.LastKernel)
	// Output:
	// frame: 1
	// passes: 1
	// last kernel: rgba8_to_bgra8
}

// ExampleSession_FinishFrame closes the frame opened implicitly by the first
// Run outside an explicit BeginFrame, returning the metrics for that frame.
func ExampleSession_FinishFrame() {
	session, done := exampleSession()
	defer done()

	src, err := session.NewPixelBuffer(PixelBufferDesc{Width: 1, Height: 1, Stride: 2, Format: PixelRGB565})
	if err != nil {
		core.Println("src error:", err)
		return
	}
	dst, err := session.NewPixelBuffer(PixelBufferDesc{Width: 1, Height: 1, Stride: 4, Format: PixelRGBA8})
	if err != nil {
		core.Println("dst error:", err)
		return
	}
	if err := src.Upload([]byte{0x00, 0xF8}); err != nil {
		core.Println("upload error:", err)
		return
	}
	if err := session.Run(KernelRGB565ToRGBA8, KernelArgs{
		Inputs:  map[string]Buffer{"src": src},
		Outputs: map[string]Buffer{"dst": dst},
	}); err != nil {
		core.Println("run error:", err)
		return
	}
	frame, err := session.FinishFrame()
	if err != nil {
		core.Println("finish error:", err)
		return
	}
	core.Println("frame:", frame.Frame, "passes:", frame.Passes)
	// Output: frame: 1 passes: 1
}

// ExampleSession_FrameMetrics reports the metrics of the most recently finished
// frame; FrameMetrics returns the same snapshot FinishFrame already returned.
func ExampleSession_FrameMetrics() {
	session, done := exampleSession()
	defer done()

	src, err := session.NewPixelBuffer(PixelBufferDesc{Width: 1, Height: 1, Stride: 2, Format: PixelRGB565})
	if err != nil {
		core.Println("src error:", err)
		return
	}
	dst, err := session.NewPixelBuffer(PixelBufferDesc{Width: 1, Height: 1, Stride: 4, Format: PixelRGBA8})
	if err != nil {
		core.Println("dst error:", err)
		return
	}
	if err := src.Upload([]byte{0x00, 0xF8}); err != nil {
		core.Println("upload error:", err)
		return
	}
	if err := session.Run(KernelRGB565ToRGBA8, KernelArgs{
		Inputs:  map[string]Buffer{"src": src},
		Outputs: map[string]Buffer{"dst": dst},
	}); err != nil {
		core.Println("run error:", err)
		return
	}
	finished, err := session.FinishFrame()
	if err != nil {
		core.Println("finish error:", err)
		return
	}
	core.Println("same:", session.FrameMetrics() == finished)
	// Output: same: true
}

// ExampleSession_Close releases a session's device resources. Closing an
// already-closed session is a no-op that still reports success.
func ExampleSession_Close() {
	session, err := NewSession()
	if err != nil {
		core.Println("error:", err)
		return
	}
	core.Println("first close:", session.Close())
	core.Println("second close:", session.Close())
	// Output:
	// first close: <nil>
	// second close: <nil>
}

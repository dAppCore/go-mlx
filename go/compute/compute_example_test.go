// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package compute

import core "dappco.re/go"

// ExamplePixelFormat_BytesPerPixel reports the packed byte width of each
// supported pixel format. Unknown formats report 0, which is how the validator
// detects an unsupported layout.
func ExamplePixelFormat_BytesPerPixel() {
	core.Println(PixelRGBA8.BytesPerPixel(), PixelRGB565.BytesPerPixel(), PixelIndexed8.BytesPerPixel(), PixelFormat("rgba16").BytesPerPixel())
	// Output: 4 2 1 0
}

// ExamplePixelBufferDesc_Validate accepts a well-formed descriptor and rejects
// one whose stride is narrower than width * bytes-per-pixel, naming the
// offending field in the structured error.
func ExamplePixelBufferDesc_Validate() {
	good := PixelBufferDesc{Width: 2, Height: 2, Stride: 8, Format: PixelRGBA8}
	core.Println("good:", good.Validate())

	narrow := PixelBufferDesc{Width: 2, Height: 2, Stride: 4, Format: PixelRGBA8}
	var bad *ComputeError
	core.As(narrow.Validate(), &bad)
	core.Println("bad:", bad.Kind, bad.Resource)
	// Output:
	// good: <nil>
	// bad: invalid_descriptor stride
}

// ExamplePixelBufferDesc_SizeBytes returns the packed byte length
// (Height * Stride) for a valid descriptor, and 0 for one that fails
// validation.
func ExamplePixelBufferDesc_SizeBytes() {
	valid := PixelBufferDesc{Width: 160, Height: 144, Stride: 640, Format: PixelRGBA8}
	invalid := PixelBufferDesc{Width: 1, Height: 1, Stride: 0, Format: PixelRGBA8}
	core.Println(valid.SizeBytes(), invalid.SizeBytes())
	// Output: 92160 0
}

// ExampleComputeError_Error renders the default message for a kind and folds a
// wrapped cause into the rendered text.
func ExampleComputeError_Error() {
	core.Println(ErrComputeUnknownKernel.Error())
	wrapped := computeWrap(ComputeErrorInternal, "dispatch_kernel", KernelNearestScale, "dst", "dispatch failed", core.NewError("metal blew up"))
	core.Println(wrapped.Error())
	// Output:
	// mlx: unknown compute kernel
	// mlx: dispatch failed: metal blew up
}

// ExampleComputeError_Is matches a wrapped error against a sentinel by kind,
// and against a more specific template by kind plus kernel. Only the fields the
// template sets are compared, so a template naming a different kernel does not
// match.
func ExampleComputeError_Is() {
	err := &ComputeError{Kind: ComputeErrorInvalidScalar, Kernel: KernelScanlineFilter, Resource: "strength"}
	core.Println(core.Is(err, ErrComputeInvalidScalar))
	core.Println(core.Is(err, &ComputeError{Kind: ComputeErrorInvalidScalar, Kernel: KernelScanlineFilter}))
	core.Println(core.Is(err, &ComputeError{Kind: ComputeErrorInvalidScalar, Kernel: KernelCRTFilter}))
	// Output:
	// true
	// true
	// false
}

// ExampleComputeError_Unwrap exposes the wrapped cause so core.Is can walk the
// chain to the original error.
func ExampleComputeError_Unwrap() {
	cause := core.NewError("backing store gone")
	err := computeWrap(ComputeErrorInternal, "read_buffer", "", "", "readback failed", cause)
	core.Println(core.Is(err, cause))
	// Output: true
}

// ExampleWithSessionLabel folds a human-readable label into the session
// configuration; the label is later sanitised into compiled kernel names so
// verbose logs can be traced back to a frame pipeline.
func ExampleWithSessionLabel() {
	cfg := newSessionConfig([]SessionOption{WithSessionLabel("Retro Frame / P1")})
	core.Println(cfg.label)
	core.Println(computeKernelRuntimeName(cfg.label, "frame_copy_scale"))
	// Output:
	// Retro Frame / P1
	// compute_retro_frame_p1__frame_copy_scale
}

// ExampleWithVerboseKernels enables verbose kernel-compilation logging on a
// session configuration.
func ExampleWithVerboseKernels() {
	cfg := newSessionConfig([]SessionOption{WithVerboseKernels(true)})
	core.Println(cfg.verboseKernels)
	// Output: true
}

// ExampleWithResetPeakMemory opts a session out of resetting the global MLX
// peak-memory counter at creation; the default (no options) leaves the reset
// enabled.
func ExampleWithResetPeakMemory() {
	opted := newSessionConfig([]SessionOption{WithResetPeakMemory(false)})
	defaults := newSessionConfig(nil)
	core.Println(opted.resetPeakMemory, defaults.resetPeakMemory)
	// Output: false true
}

// ExampleDefaultCompute reports whether the Metal compute backend is available
// on this device. On Apple silicon with the Metal runtime linked it is always
// available.
func ExampleDefaultCompute() {
	core.Println(DefaultCompute().Available())
	// Output: true
}

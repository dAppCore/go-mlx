// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for the non-LLM compute primitives that DON'T need a live
// Metal session. Per AX-11 — PixelBufferDesc.Validate fires per buffer
// per frame (validation gate before every kernel dispatch), unitScalar
// + quantizeUnitScalar fire per scalar arg per dispatch, sameDimensions
// + validateFilterBuffers fire per pixel-pair kernel, sanitizeComputeLabel
// fires once per kernel-name resolution which goes through a per-frame
// per-kernel cache lookup. Error format / Is dispatch is hot when frame
// pipelines surface compute errors back to the orchestrator.
// Anything that actually allocates a Metal Array / runs a kernel lives
// in compute_metal_*.go — those needs a GPU and are skipped here.
//
// Run:    go test -bench='BenchmarkCompute|BenchmarkPixelBufferDesc|BenchmarkSanitizeComputeLabel|BenchmarkUnitScalar|BenchmarkQuantizeUnitScalar|BenchmarkThreadGroup|BenchmarkSameDimensions|BenchmarkRequireBuffer|BenchmarkValidateFilterBuffers|BenchmarkComputeError|BenchmarkNewSessionConfig' -benchmem -run='^$' ./go/compute

package compute

import (
	"errors"
	"testing"
)

// Sinks defeat compiler DCE.
var (
	benchComputeInt        int
	benchComputeIntPair    [2]int
	benchComputeBool       bool
	benchComputeStr        string
	benchComputeErr        error
	benchComputeBytes      int
	benchComputeBuf        Buffer
	benchComputeSessionCfg sessionConfig
)

// --- PixelBufferDesc.Validate — gate before every Metal frame ---

func BenchmarkPixelBufferDesc_Validate_Valid(b *testing.B) {
	desc := PixelBufferDesc{Width: 320, Height: 224, Stride: 320 * 4, Format: PixelRGBA8}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchComputeErr = desc.Validate()
	}
}

// Typical 2048-wide framebuffer descriptor.
func BenchmarkPixelBufferDesc_Validate_LargeRGBA8(b *testing.B) {
	desc := PixelBufferDesc{Width: 2048, Height: 2048, Stride: 2048 * 4, Format: PixelRGBA8}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchComputeErr = desc.Validate()
	}
}

// Invalid descriptor — exercises the worst-case branch where the error
// path runs.
func BenchmarkPixelBufferDesc_Validate_InvalidStride(b *testing.B) {
	desc := PixelBufferDesc{Width: 320, Height: 224, Stride: 639, Format: PixelRGB565}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchComputeErr = desc.Validate()
	}
}

func BenchmarkPixelBufferDesc_SizeBytes_Valid(b *testing.B) {
	desc := PixelBufferDesc{Width: 1024, Height: 1024, Stride: 1024 * 4, Format: PixelRGBA8}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchComputeBytes = desc.SizeBytes()
	}
}

// --- PixelFormat.BytesPerPixel — fires per stride check ---

func BenchmarkPixelFormat_BytesPerPixel_RGBA8(b *testing.B) {
	format := PixelRGBA8
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchComputeInt = format.BytesPerPixel()
	}
}

func BenchmarkPixelFormat_BytesPerPixel_RGB565(b *testing.B) {
	format := PixelRGB565
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchComputeInt = format.BytesPerPixel()
	}
}

// --- sanitizeComputeLabel — fires per kernel runtime-name resolution ---

func BenchmarkSanitizeComputeLabel_Clean(b *testing.B) {
	label := "frame_pipeline_main"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchComputeStr = sanitizeComputeLabel(label)
	}
}

// Mixed-case + separators — every char goes through the unicode path.
func BenchmarkSanitizeComputeLabel_MixedCase(b *testing.B) {
	label := "Frame-Pipeline.Main Buffer-1"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchComputeStr = sanitizeComputeLabel(label)
	}
}

func BenchmarkSanitizeComputeLabel_LongUnicode(b *testing.B) {
	label := "  Café_Frame__Pipe-Stage  "
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchComputeStr = sanitizeComputeLabel(label)
	}
}

func BenchmarkComputeKernelRuntimeName_WithLabel(b *testing.B) {
	label := "frame_pipeline_main"
	kernel := KernelBilinearScale
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchComputeStr = computeKernelRuntimeName(label, kernel)
	}
}

func BenchmarkComputeKernelRuntimeName_EmptyLabel(b *testing.B) {
	kernel := KernelBilinearScale
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchComputeStr = computeKernelRuntimeName("", kernel)
	}
}

// --- unitScalar / quantizeUnitScalar — per-scalar per-dispatch ---

func BenchmarkUnitScalar_Default(b *testing.B) {
	args := KernelArgs{}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchComputeInt, benchComputeErr = unitScalar(args, KernelScanlineFilter, "strength", 0.25)
	}
}

func BenchmarkUnitScalar_Explicit(b *testing.B) {
	args := KernelArgs{Scalars: map[string]float64{"strength": 0.75}}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchComputeInt, benchComputeErr = unitScalar(args, KernelScanlineFilter, "strength", 0.25)
	}
}

func BenchmarkQuantizeUnitScalar_Mid(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchComputeInt = quantizeUnitScalar(0.5)
	}
}

func BenchmarkQuantizeUnitScalar_Clamped(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchComputeInt = quantizeUnitScalar(2.0)
	}
}

// --- threadGroup / minInt / maxInt — scalar inline math ---

func BenchmarkThreadGroup_Typical(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		x, y := threadGroup(2048, 2048)
		benchComputeIntPair = [2]int{x, y}
	}
}

func BenchmarkThreadGroup_Small(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		x, y := threadGroup(8, 3)
		benchComputeIntPair = [2]int{x, y}
	}
}

// --- sameDimensions — per pixel-pair validation ---

func BenchmarkSameDimensions_Match(b *testing.B) {
	a := PixelBufferDesc{Width: 1024, Height: 1024, Stride: 4096, Format: PixelRGBA8}
	c := PixelBufferDesc{Width: 1024, Height: 1024, Stride: 4096, Format: PixelRGBA8}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchComputeBool = sameDimensions(a, c)
	}
}

func BenchmarkSameDimensions_Mismatch(b *testing.B) {
	a := PixelBufferDesc{Width: 1024, Height: 1024, Stride: 4096, Format: PixelRGBA8}
	c := PixelBufferDesc{Width: 1024, Height: 512, Stride: 4096, Format: PixelRGBA8}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchComputeBool = sameDimensions(a, c)
	}
}

// --- requireBuffer — fires per kernel arg lookup ---

func BenchmarkRequireBuffer_Hit(b *testing.B) {
	src := &bufferbase{size: 4096}
	buffers := map[string]Buffer{"src": src, "dst": &bufferbase{size: 4096}}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchComputeBuf, benchComputeErr = requireBuffer(buffers, KernelNearestScale, "src")
	}
}

func BenchmarkRequireBuffer_Miss(b *testing.B) {
	buffers := map[string]Buffer{}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchComputeBuf, benchComputeErr = requireBuffer(buffers, KernelNearestScale, "src")
	}
}

// --- validateFilterBuffers — gate before every filter kernel ---

func BenchmarkValidateFilterBuffers_Valid(b *testing.B) {
	desc := PixelBufferDesc{Width: 320, Height: 224, Stride: 320 * 4, Format: PixelRGBA8}
	src := &pixelbuffer{desc: desc}
	dst := &pixelbuffer{desc: desc}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchComputeErr = validateFilterBuffers(src, dst, KernelScanlineFilter)
	}
}

// --- newSessionConfig — fires per NewSession; small options slice ---

func BenchmarkNewSessionConfig_NoOpts(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchComputeSessionCfg = newSessionConfig(nil)
	}
}

func BenchmarkNewSessionConfig_ThreeOpts(b *testing.B) {
	opts := []SessionOption{
		WithSessionLabel("frame-pipe"),
		WithVerboseKernels(true),
		WithResetPeakMemory(false),
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchComputeSessionCfg = newSessionConfig(opts)
	}
}

// --- ComputeError.Error / Is / Unwrap — fires on every compute-error
// surface back to the orchestrator. Each pipeline error walks Is() to
// match against the sentinel kinds. ---

func BenchmarkComputeError_Error_Default(b *testing.B) {
	err := &ComputeError{Kind: ComputeErrorInvalidDescriptor, Op: "validate_pixel_buffer", Resource: "stride"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchComputeStr = err.Error()
	}
}

func BenchmarkComputeError_Error_Wrapped(b *testing.B) {
	wrapped := errors.New("metal: bad command buffer")
	err := &ComputeError{Kind: ComputeErrorInternal, Op: "dispatch", Kernel: KernelBilinearScale, Err: wrapped}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchComputeStr = err.Error()
	}
}

func BenchmarkComputeError_Is_KindMatch(b *testing.B) {
	err := &ComputeError{Kind: ComputeErrorInvalidDescriptor, Op: "validate", Resource: "stride"}
	target := ErrComputeInvalidDescriptor
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchComputeBool = err.Is(target)
	}
}

func BenchmarkComputeError_Is_FullMatch(b *testing.B) {
	err := &ComputeError{Kind: ComputeErrorInvalidKernelArgs, Op: "dispatch", Kernel: KernelBilinearScale, Resource: "dst"}
	target := &ComputeError{Kind: ComputeErrorInvalidKernelArgs, Op: "dispatch", Kernel: KernelBilinearScale, Resource: "dst"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchComputeBool = err.Is(target)
	}
}

func BenchmarkComputeError_Unwrap_Wrapped(b *testing.B) {
	wrapped := errors.New("metal: bad command buffer")
	err := &ComputeError{Kind: ComputeErrorInternal, Err: wrapped}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchComputeErr = err.Unwrap()
	}
}

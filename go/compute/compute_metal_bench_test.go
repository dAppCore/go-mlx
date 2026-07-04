// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for the buffer readback path — bytebuffer.Read and
// pixelbuffer.Read — which the bench-coverage audit flagged 0%-covered.
// Both methods take the session lock and delegate to readLocked, which
// is the real GPU→host round-trip on the frame pipeline: syncLocked
// (stream sync + retire drain) → metal.Eval(array) → array.Bytes()
// (the inherent output buffer the caller owns). Read fires per frame
// per buffer that the orchestrator pulls back from the GPU, so its
// per-readback heap traffic is on the steady-state decode/frame path.
//
// Unlike the scalar/validation benches in compute_bench_test.go, these
// allocate a live Metal Array and evaluate it, so they need the GPU.
// Gated at runtime via metal.MetalAvailable() (the package is already
// darwin && arm64-only); skipped cleanly when Metal is unavailable.
//
// Run:    go test -bench='BenchmarkComputeRead' -benchmem -run='^$' ./go/compute

package compute

import (
	"testing"

	"dappco.re/go/mlx/pkg/metal"
)

// benchReadBytes sinks the returned slice into a package var so the
// caller-owned output buffer genuinely escapes — keeps the inherent-
// floor allocation honest (a no-escape sink would let the compiler
// stack-allocate it and hide the floor).
var (
	benchReadBytes []byte
	benchReadErr   error
)

func benchComputeSession(b *testing.B) Session {
	b.Helper()
	if !metal.MetalAvailable() {
		b.Skip("Metal runtime unavailable")
	}
	session, err := NewSession()
	if err != nil {
		b.Fatalf("NewSession: %v", err)
	}
	b.Cleanup(func() {
		if err := session.Close(); err != nil {
			b.Fatalf("Close: %v", err)
		}
	})
	return session
}

// makeReadByteFixture builds a byte buffer of size bytes, uploaded with a
// deterministic ramp so the readback returns real (non-zero) data.
func makeReadByteFixture(b *testing.B, size int) ByteBuffer {
	b.Helper()
	session := benchComputeSession(b)
	buffer, err := session.NewByteBuffer(size)
	if err != nil {
		b.Fatalf("NewByteBuffer(%d): %v", size, err)
	}
	data := make([]byte, size)
	for i := range data {
		data[i] = byte(i)
	}
	if err := buffer.Upload(data); err != nil {
		b.Fatalf("Upload: %v", err)
	}
	return buffer
}

// --- bytebuffer.Read — the byte-buffer GPU→host readback ---

// Tiny 4-byte buffer — readback dominated by fixed per-call overhead,
// not the output-buffer copy. Any B/op above ~the size class here is a
// fixed-overhead alloc hiding behind the inherent floor.
func BenchmarkComputeRead_ByteBuffer_Small(b *testing.B) {
	buffer := makeReadByteFixture(b, 4)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchReadBytes, benchReadErr = buffer.Read()
	}
}

// 1 MiB buffer — output-buffer copy dominates; B/op should track the
// payload size (the inherent make([]byte,n) floor scales with it, fixed
// overhead stays constant), proving floor vs overhead by size.
func BenchmarkComputeRead_ByteBuffer_1MiB(b *testing.B) {
	buffer := makeReadByteFixture(b, 1<<20)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchReadBytes, benchReadErr = buffer.Read()
	}
}

// --- pixelbuffer.Read — shares readLocked; covers the second Read type ---

// 320x224 RGBA8 framebuffer (286720 bytes) — a typical frame readback.
func BenchmarkComputeRead_PixelBuffer_FrameRGBA8(b *testing.B) {
	session := benchComputeSession(b)
	desc := PixelBufferDesc{Width: 320, Height: 224, Stride: 320 * 4, Format: PixelRGBA8}
	buffer, err := session.NewPixelBuffer(desc)
	if err != nil {
		b.Fatalf("NewPixelBuffer: %v", err)
	}
	data := make([]byte, desc.SizeBytes())
	for i := range data {
		data[i] = byte(i)
	}
	if err := buffer.Upload(data); err != nil {
		b.Fatalf("Upload: %v", err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchReadBytes, benchReadErr = buffer.Read()
	}
}

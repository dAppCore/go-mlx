// SPDX-Licence-Identifier: EUPL-1.2

package compute

import (
	"testing"

	core "dappco.re/go"

	"dappco.re/go/mlx/pkg/metal"
)

// requireCoverageSession mirrors requireComputeSession: a live Metal-backed
// session that is closed on cleanup. Kept local so this coverage file does not
// depend on the ordering of the other _test.go files.
func requireCoverageSession(t *testing.T) *computesession {
	t.Helper()
	if !metal.MetalAvailable() {
		t.Skip("Metal runtime unavailable")
	}
	session, err := NewSession()
	if err != nil {
		t.Fatalf("NewSession: %v", err)
	}
	t.Cleanup(func() {
		if err := session.Close(); err != nil {
			t.Fatalf("Close: %v", err)
		}
	})
	concrete, ok := session.(*computesession)
	if !ok {
		t.Fatalf("NewSession returned %T, want *computesession", session)
	}
	return concrete
}

func newRGBA8PixelBuffer(t *testing.T, session Session, w, h int) PixelBuffer {
	t.Helper()
	buffer, err := session.NewPixelBuffer(PixelBufferDesc{
		Width:  w,
		Height: h,
		Stride: w * 4,
		Format: PixelRGBA8,
	})
	if err != nil {
		t.Fatalf("NewPixelBuffer(rgba8 %dx%d): %v", w, h, err)
	}
	return buffer
}

// --- bufferbase.requireOpenLocked: nil-buffer and nil-array arms (L89-97) ---

func TestComputeCoverage_requireOpenLocked_NilArms_Bad(t *testing.T) {
	// nil receiver / nil session.
	var nilBase *bufferbase
	if err := nilBase.requireOpenLocked(); !core.Is(err, ErrComputeInvalidBuffer) {
		t.Fatalf("requireOpenLocked(nil) = %v, want ErrComputeInvalidBuffer", err)
	}

	if err := (&bufferbase{}).requireOpenLocked(); !core.Is(err, ErrComputeInvalidBuffer) {
		t.Fatalf("requireOpenLocked(nil session) = %v, want ErrComputeInvalidBuffer", err)
	}

	// Open session, but no backing array.
	session := requireCoverageSession(t)
	base := &bufferbase{session: session}
	if err := base.requireOpenLocked(); !core.Is(err, ErrComputeInvalidBuffer) {
		t.Fatalf("requireOpenLocked(nil array) = %v, want ErrComputeInvalidBuffer", err)
	}

	// Closed session takes priority over the missing array.
	closedSession := requireCoverageSession(t)
	if err := closedSession.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	closedBase := &bufferbase{session: closedSession}
	if err := closedBase.requireOpenLocked(); !core.Is(err, ErrComputeClosed) {
		t.Fatalf("requireOpenLocked(closed) = %v, want ErrComputeClosed", err)
	}
}

// --- session.retireArrayLocked(nil) early return (L383-385) ---

func TestComputeCoverage_retireArrayLocked_Nil_Good(t *testing.T) {
	session := requireCoverageSession(t)
	before := len(session.retired)
	session.retireArrayLocked(nil)
	if len(session.retired) != before {
		t.Fatalf("retireArrayLocked(nil) changed retired from %d to %d", before, len(session.retired))
	}
}

// --- session.updateFrameMetricsLocked early return when no frame (L414-416) ---

func TestComputeCoverage_updateFrameMetricsLocked_NoActiveFrame_Good(t *testing.T) {
	session := requireCoverageSession(t)
	if session.frame.active {
		t.Fatal("fresh session should not have an active frame")
	}
	// Must not panic or mutate frame metrics when no frame is active.
	session.updateFrameMetricsLocked()
	if session.frame.metrics != (FrameMetrics{}) {
		t.Fatalf("updateFrameMetricsLocked mutated metrics: %+v", session.frame.metrics)
	}
}

// --- session.kernelLocked missing-spec arm (L466-468) ---

func TestComputeCoverage_kernelLocked_MissingSpec_Bad(t *testing.T) {
	session := requireCoverageSession(t)
	_, err := session.kernelLocked("frame_does_not_exist")
	if !core.Is(err, ErrComputeInternal) {
		t.Fatalf("kernelLocked(missing) = %v, want ErrComputeInternal", err)
	}
	var ce *ComputeError
	if !core.As(err, &ce) {
		t.Fatalf("kernelLocked(missing) error = %T, want *ComputeError", err)
	}
	if ce.Kernel != "frame_does_not_exist" {
		t.Fatalf("Kernel = %q, want frame_does_not_exist", ce.Kernel)
	}
}

// --- pixelbufferLocked / bytebufferLocked wrong-type + requireOpen arms
// (L501-503, L509-517) ---

func TestComputeCoverage_pixelbufferLocked_WrongType_Bad(t *testing.T) {
	session := requireCoverageSession(t)

	// A byte buffer where a pixel buffer is required.
	bb, err := session.NewByteBuffer(16)
	if err != nil {
		t.Fatalf("NewByteBuffer: %v", err)
	}
	if _, err := session.pixelbufferLocked(bb, KernelNearestScale, "src"); !core.Is(err, ErrComputeInvalidBuffer) {
		t.Fatalf("pixelbufferLocked(byte buffer) = %v, want ErrComputeInvalidBuffer", err)
	}

	// A pixel buffer whose backing array has been cleared -> requireOpenLocked fails.
	pb := newRGBA8PixelBuffer(t, session, 1, 1).(*pixelbuffer)
	pb.array = nil
	if _, err := session.pixelbufferLocked(pb, KernelNearestScale, "src"); !core.Is(err, ErrComputeInvalidBuffer) {
		t.Fatalf("pixelbufferLocked(no storage) = %v, want ErrComputeInvalidBuffer", err)
	}
}

func TestComputeCoverage_bytebufferLocked_WrongTypeAndForeignAndOpen_Bad(t *testing.T) {
	session := requireCoverageSession(t)

	// A pixel buffer where a byte buffer is required.
	pb := newRGBA8PixelBuffer(t, session, 1, 1)
	if _, err := session.bytebufferLocked(pb, KernelPaletteExpandRGBA, "palette"); !core.Is(err, ErrComputeInvalidBuffer) {
		t.Fatalf("bytebufferLocked(pixel buffer) = %v, want ErrComputeInvalidBuffer", err)
	}

	// A byte buffer that belongs to a different session.
	other := requireCoverageSession(t)
	foreign, err := other.NewByteBuffer(16)
	if err != nil {
		t.Fatalf("NewByteBuffer(other): %v", err)
	}
	if _, err := session.bytebufferLocked(foreign, KernelPaletteExpandRGBA, "palette"); !core.Is(err, ErrComputeInvalidBuffer) {
		t.Fatalf("bytebufferLocked(foreign) = %v, want ErrComputeInvalidBuffer", err)
	}

	// A byte buffer of this session whose storage has been cleared.
	bb, err := session.NewByteBuffer(16)
	if err != nil {
		t.Fatalf("NewByteBuffer: %v", err)
	}
	concrete := bb.(*bytebuffer)
	concrete.array = nil
	if _, err := session.bytebufferLocked(bb, KernelPaletteExpandRGBA, "palette"); !core.Is(err, ErrComputeInvalidBuffer) {
		t.Fatalf("bytebufferLocked(no storage) = %v, want ErrComputeInvalidBuffer", err)
	}
}

// --- readLocked: requireOpenLocked failure arm (L109-111 already covered via
// Read on open buffers; here we hit the closed-buffer error return). ---

func TestComputeCoverage_readLocked_RequireOpenFails_Bad(t *testing.T) {
	session := requireCoverageSession(t)
	pb := newRGBA8PixelBuffer(t, session, 1, 1).(*pixelbuffer)
	pb.array = nil
	if _, err := pb.readLocked(); !core.Is(err, ErrComputeInvalidBuffer) {
		t.Fatalf("readLocked(no storage) = %v, want ErrComputeInvalidBuffer", err)
	}
}

// --- run*Locked: the "src ok, dst missing" and wrong-type / format-mismatch
// branches reached through the public Run API. Each sub-test pins one kernel's
// second-buffer and format-validation arms. ---

func runMissingDst(t *testing.T, session Session, kernel string, src Buffer) {
	t.Helper()
	err := session.Run(kernel, KernelArgs{
		Inputs: map[string]Buffer{"src": src},
	})
	if !core.Is(err, ErrComputeMissingKernelBuffer) {
		t.Fatalf("Run(%s, missing dst) = %v, want ErrComputeMissingKernelBuffer", kernel, err)
	}
	var ce *ComputeError
	if !core.As(err, &ce) {
		t.Fatalf("Run(%s, missing dst) error = %T, want *ComputeError", kernel, err)
	}
	if ce.Resource != "dst" {
		t.Fatalf("Run(%s, missing dst) resource = %q, want dst", kernel, ce.Resource)
	}
}

func runWrongDstType(t *testing.T, session Session, kernel string, src Buffer, dst Buffer) {
	t.Helper()
	err := session.Run(kernel, KernelArgs{
		Inputs:  map[string]Buffer{"src": src},
		Outputs: map[string]Buffer{"dst": dst},
	})
	if !core.Is(err, ErrComputeInvalidBuffer) {
		t.Fatalf("Run(%s, wrong dst type) = %v, want ErrComputeInvalidBuffer", kernel, err)
	}
}

func TestComputeCoverage_NearestScale_SecondBufferArms_Bad(t *testing.T) {
	session := requireCoverageSession(t)
	src := newRGBA8PixelBuffer(t, session, 2, 2)
	runMissingDst(t, session, KernelNearestScale, src)

	bb, err := session.NewByteBuffer(16)
	if err != nil {
		t.Fatalf("NewByteBuffer: %v", err)
	}
	runWrongDstType(t, session, KernelNearestScale, src, bb)
}

func TestComputeCoverage_BilinearScale_AllArms_Bad(t *testing.T) {
	session := requireCoverageSession(t)
	src := newRGBA8PixelBuffer(t, session, 2, 2)
	runMissingDst(t, session, KernelBilinearScale, src)

	bb, err := session.NewByteBuffer(16)
	if err != nil {
		t.Fatalf("NewByteBuffer: %v", err)
	}
	runWrongDstType(t, session, KernelBilinearScale, src, bb)

	// Wrong source type: byte buffer as src.
	dst := newRGBA8PixelBuffer(t, session, 2, 2)
	err = session.Run(KernelBilinearScale, KernelArgs{
		Inputs:  map[string]Buffer{"src": bb},
		Outputs: map[string]Buffer{"dst": dst},
	})
	if !core.Is(err, ErrComputeInvalidBuffer) {
		t.Fatalf("Run(bilinear, byte src) = %v, want ErrComputeInvalidBuffer", err)
	}

	// Format mismatch: src/dst differ in format.
	bgra, err := session.NewPixelBuffer(PixelBufferDesc{Width: 2, Height: 2, Stride: 8, Format: PixelBGRA8})
	if err != nil {
		t.Fatalf("NewPixelBuffer(bgra): %v", err)
	}
	err = session.Run(KernelBilinearScale, KernelArgs{
		Inputs:  map[string]Buffer{"src": src},
		Outputs: map[string]Buffer{"dst": bgra},
	})
	if !core.Is(err, ErrComputeInvalidKernelArgs) {
		t.Fatalf("Run(bilinear, format mismatch) = %v, want ErrComputeInvalidKernelArgs", err)
	}

	// Unsupported format: rgb565 src+dst (matching formats, but not rgba/bgra).
	s565, err := session.NewPixelBuffer(PixelBufferDesc{Width: 2, Height: 2, Stride: 4, Format: PixelRGB565})
	if err != nil {
		t.Fatalf("NewPixelBuffer(565 src): %v", err)
	}
	d565, err := session.NewPixelBuffer(PixelBufferDesc{Width: 2, Height: 2, Stride: 4, Format: PixelRGB565})
	if err != nil {
		t.Fatalf("NewPixelBuffer(565 dst): %v", err)
	}
	err = session.Run(KernelBilinearScale, KernelArgs{
		Inputs:  map[string]Buffer{"src": s565},
		Outputs: map[string]Buffer{"dst": d565},
	})
	if !core.Is(err, ErrComputeInvalidKernelArgs) {
		t.Fatalf("Run(bilinear, unsupported format) = %v, want ErrComputeInvalidKernelArgs", err)
	}
}

func TestComputeCoverage_RGB565ToRGBA8_AllArms_Bad(t *testing.T) {
	session := requireCoverageSession(t)
	src565, err := session.NewPixelBuffer(PixelBufferDesc{Width: 2, Height: 2, Stride: 4, Format: PixelRGB565})
	if err != nil {
		t.Fatalf("NewPixelBuffer(565): %v", err)
	}
	runMissingDst(t, session, KernelRGB565ToRGBA8, src565)

	bb, err := session.NewByteBuffer(16)
	if err != nil {
		t.Fatalf("NewByteBuffer: %v", err)
	}
	runWrongDstType(t, session, KernelRGB565ToRGBA8, src565, bb)

	// Wrong source format: rgba8 src.
	rgbaSrc := newRGBA8PixelBuffer(t, session, 2, 2)
	rgbaDst := newRGBA8PixelBuffer(t, session, 2, 2)
	err = session.Run(KernelRGB565ToRGBA8, KernelArgs{
		Inputs:  map[string]Buffer{"src": rgbaSrc},
		Outputs: map[string]Buffer{"dst": rgbaDst},
	})
	if !core.Is(err, ErrComputeInvalidKernelArgs) {
		t.Fatalf("Run(rgb565, wrong src format) = %v, want ErrComputeInvalidKernelArgs", err)
	}

	// Wrong destination format: rgb565 src but bgra8 dst.
	bgraDst, err := session.NewPixelBuffer(PixelBufferDesc{Width: 2, Height: 2, Stride: 8, Format: PixelBGRA8})
	if err != nil {
		t.Fatalf("NewPixelBuffer(bgra dst): %v", err)
	}
	err = session.Run(KernelRGB565ToRGBA8, KernelArgs{
		Inputs:  map[string]Buffer{"src": src565},
		Outputs: map[string]Buffer{"dst": bgraDst},
	})
	if !core.Is(err, ErrComputeInvalidKernelArgs) {
		t.Fatalf("Run(rgb565, wrong dst format) = %v, want ErrComputeInvalidKernelArgs", err)
	}

	// Dimension mismatch: matching formats, different sizes.
	bigDst, err := session.NewPixelBuffer(PixelBufferDesc{Width: 4, Height: 2, Stride: 16, Format: PixelRGBA8})
	if err != nil {
		t.Fatalf("NewPixelBuffer(big dst): %v", err)
	}
	err = session.Run(KernelRGB565ToRGBA8, KernelArgs{
		Inputs:  map[string]Buffer{"src": src565},
		Outputs: map[string]Buffer{"dst": bigDst},
	})
	if !core.Is(err, ErrComputeInvalidKernelArgs) {
		t.Fatalf("Run(rgb565, dimension mismatch) = %v, want ErrComputeInvalidKernelArgs", err)
	}
}

func TestComputeCoverage_ChannelSwizzle_AllArms_Bad(t *testing.T) {
	session := requireCoverageSession(t)
	rgbaSrc := newRGBA8PixelBuffer(t, session, 2, 2)
	runMissingDst(t, session, KernelRGBA8ToBGRA8, rgbaSrc)

	bb, err := session.NewByteBuffer(16)
	if err != nil {
		t.Fatalf("NewByteBuffer: %v", err)
	}
	runWrongDstType(t, session, KernelRGBA8ToBGRA8, rgbaSrc, bb)

	// Dimension mismatch.
	bigDst, err := session.NewPixelBuffer(PixelBufferDesc{Width: 4, Height: 2, Stride: 16, Format: PixelBGRA8})
	if err != nil {
		t.Fatalf("NewPixelBuffer(big bgra): %v", err)
	}
	err = session.Run(KernelRGBA8ToBGRA8, KernelArgs{
		Inputs:  map[string]Buffer{"src": rgbaSrc},
		Outputs: map[string]Buffer{"dst": bigDst},
	})
	if !core.Is(err, ErrComputeInvalidKernelArgs) {
		t.Fatalf("Run(rgba->bgra, dimension mismatch) = %v, want ErrComputeInvalidKernelArgs", err)
	}

	// rgba8_to_bgra8 with a wrong-format destination (rgba8 dst, want bgra8).
	rgbaDst := newRGBA8PixelBuffer(t, session, 2, 2)
	err = session.Run(KernelRGBA8ToBGRA8, KernelArgs{
		Inputs:  map[string]Buffer{"src": rgbaSrc},
		Outputs: map[string]Buffer{"dst": rgbaDst},
	})
	if !core.Is(err, ErrComputeInvalidKernelArgs) {
		t.Fatalf("Run(rgba->bgra, wrong dst format) = %v, want ErrComputeInvalidKernelArgs", err)
	}

	// bgra8_to_rgba8 dst-format arm: bgra source, bgra dst (want rgba dst).
	bgraSrc, err := session.NewPixelBuffer(PixelBufferDesc{Width: 2, Height: 2, Stride: 8, Format: PixelBGRA8})
	if err != nil {
		t.Fatalf("NewPixelBuffer(bgra src): %v", err)
	}
	bgraDst2, err := session.NewPixelBuffer(PixelBufferDesc{Width: 2, Height: 2, Stride: 8, Format: PixelBGRA8})
	if err != nil {
		t.Fatalf("NewPixelBuffer(bgra dst): %v", err)
	}
	err = session.Run(KernelBGRA8ToRGBA8, KernelArgs{
		Inputs:  map[string]Buffer{"src": bgraSrc},
		Outputs: map[string]Buffer{"dst": bgraDst2},
	})
	if !core.Is(err, ErrComputeInvalidKernelArgs) {
		t.Fatalf("Run(bgra->rgba, wrong dst format) = %v, want ErrComputeInvalidKernelArgs", err)
	}
}

func TestComputeCoverage_XRGB8888ToRGBA8_AllArms_Bad(t *testing.T) {
	session := requireCoverageSession(t)
	xrgbSrc, err := session.NewPixelBuffer(PixelBufferDesc{Width: 2, Height: 2, Stride: 8, Format: PixelXRGB8888})
	if err != nil {
		t.Fatalf("NewPixelBuffer(xrgb): %v", err)
	}
	runMissingDst(t, session, KernelXRGB8888ToRGBA8, xrgbSrc)

	bb, err := session.NewByteBuffer(16)
	if err != nil {
		t.Fatalf("NewByteBuffer: %v", err)
	}
	runWrongDstType(t, session, KernelXRGB8888ToRGBA8, xrgbSrc, bb)

	// Wrong source format: rgba8 src.
	rgbaSrc := newRGBA8PixelBuffer(t, session, 2, 2)
	rgbaDst := newRGBA8PixelBuffer(t, session, 2, 2)
	err = session.Run(KernelXRGB8888ToRGBA8, KernelArgs{
		Inputs:  map[string]Buffer{"src": rgbaSrc},
		Outputs: map[string]Buffer{"dst": rgbaDst},
	})
	if !core.Is(err, ErrComputeInvalidKernelArgs) {
		t.Fatalf("Run(xrgb, wrong src format) = %v, want ErrComputeInvalidKernelArgs", err)
	}

	// Wrong destination format: xrgb src, bgra dst.
	bgraDst, err := session.NewPixelBuffer(PixelBufferDesc{Width: 2, Height: 2, Stride: 8, Format: PixelBGRA8})
	if err != nil {
		t.Fatalf("NewPixelBuffer(bgra dst): %v", err)
	}
	err = session.Run(KernelXRGB8888ToRGBA8, KernelArgs{
		Inputs:  map[string]Buffer{"src": xrgbSrc},
		Outputs: map[string]Buffer{"dst": bgraDst},
	})
	if !core.Is(err, ErrComputeInvalidKernelArgs) {
		t.Fatalf("Run(xrgb, wrong dst format) = %v, want ErrComputeInvalidKernelArgs", err)
	}

	// Dimension mismatch.
	bigDst, err := session.NewPixelBuffer(PixelBufferDesc{Width: 4, Height: 2, Stride: 16, Format: PixelRGBA8})
	if err != nil {
		t.Fatalf("NewPixelBuffer(big dst): %v", err)
	}
	err = session.Run(KernelXRGB8888ToRGBA8, KernelArgs{
		Inputs:  map[string]Buffer{"src": xrgbSrc},
		Outputs: map[string]Buffer{"dst": bigDst},
	})
	if !core.Is(err, ErrComputeInvalidKernelArgs) {
		t.Fatalf("Run(xrgb, dimension mismatch) = %v, want ErrComputeInvalidKernelArgs", err)
	}
}

func TestComputeCoverage_PaletteExpand_AllArms_Bad(t *testing.T) {
	session := requireCoverageSession(t)
	idxSrc, err := session.NewPixelBuffer(PixelBufferDesc{Width: 2, Height: 2, Stride: 2, Format: PixelIndexed8})
	if err != nil {
		t.Fatalf("NewPixelBuffer(indexed): %v", err)
	}
	palette, err := session.NewByteBuffer(256 * 4)
	if err != nil {
		t.Fatalf("NewByteBuffer(palette): %v", err)
	}
	rgbaDst := newRGBA8PixelBuffer(t, session, 2, 2)

	// Missing palette (second input).
	err = session.Run(KernelPaletteExpandRGBA, KernelArgs{
		Inputs:  map[string]Buffer{"src": idxSrc},
		Outputs: map[string]Buffer{"dst": rgbaDst},
	})
	if !core.Is(err, ErrComputeMissingKernelBuffer) {
		t.Fatalf("Run(palette, missing palette) = %v, want ErrComputeMissingKernelBuffer", err)
	}

	// Missing dst.
	err = session.Run(KernelPaletteExpandRGBA, KernelArgs{
		Inputs: map[string]Buffer{"src": idxSrc, "palette": palette},
	})
	if !core.Is(err, ErrComputeMissingKernelBuffer) {
		t.Fatalf("Run(palette, missing dst) = %v, want ErrComputeMissingKernelBuffer", err)
	}

	// Wrong src type (byte buffer as src).
	bb, err := session.NewByteBuffer(16)
	if err != nil {
		t.Fatalf("NewByteBuffer: %v", err)
	}
	err = session.Run(KernelPaletteExpandRGBA, KernelArgs{
		Inputs:  map[string]Buffer{"src": bb, "palette": palette},
		Outputs: map[string]Buffer{"dst": rgbaDst},
	})
	if !core.Is(err, ErrComputeInvalidBuffer) {
		t.Fatalf("Run(palette, wrong src type) = %v, want ErrComputeInvalidBuffer", err)
	}

	// Wrong dst type (byte buffer as dst).
	err = session.Run(KernelPaletteExpandRGBA, KernelArgs{
		Inputs:  map[string]Buffer{"src": idxSrc, "palette": palette},
		Outputs: map[string]Buffer{"dst": bb},
	})
	if !core.Is(err, ErrComputeInvalidBuffer) {
		t.Fatalf("Run(palette, wrong dst type) = %v, want ErrComputeInvalidBuffer", err)
	}

	// Wrong source format: rgba8 src instead of indexed8.
	rgbaSrc := newRGBA8PixelBuffer(t, session, 2, 2)
	err = session.Run(KernelPaletteExpandRGBA, KernelArgs{
		Inputs:  map[string]Buffer{"src": rgbaSrc, "palette": palette},
		Outputs: map[string]Buffer{"dst": rgbaDst},
	})
	if !core.Is(err, ErrComputeInvalidKernelArgs) {
		t.Fatalf("Run(palette, wrong src format) = %v, want ErrComputeInvalidKernelArgs", err)
	}

	// Wrong destination format: indexed src, bgra dst.
	bgraDst, err := session.NewPixelBuffer(PixelBufferDesc{Width: 2, Height: 2, Stride: 8, Format: PixelBGRA8})
	if err != nil {
		t.Fatalf("NewPixelBuffer(bgra dst): %v", err)
	}
	err = session.Run(KernelPaletteExpandRGBA, KernelArgs{
		Inputs:  map[string]Buffer{"src": idxSrc, "palette": palette},
		Outputs: map[string]Buffer{"dst": bgraDst},
	})
	if !core.Is(err, ErrComputeInvalidKernelArgs) {
		t.Fatalf("Run(palette, wrong dst format) = %v, want ErrComputeInvalidKernelArgs", err)
	}

	// Dimension mismatch: indexed src vs larger rgba dst.
	bigDst, err := session.NewPixelBuffer(PixelBufferDesc{Width: 4, Height: 2, Stride: 16, Format: PixelRGBA8})
	if err != nil {
		t.Fatalf("NewPixelBuffer(big dst): %v", err)
	}
	err = session.Run(KernelPaletteExpandRGBA, KernelArgs{
		Inputs:  map[string]Buffer{"src": idxSrc, "palette": palette},
		Outputs: map[string]Buffer{"dst": bigDst},
	})
	if !core.Is(err, ErrComputeInvalidKernelArgs) {
		t.Fatalf("Run(palette, dimension mismatch) = %v, want ErrComputeInvalidKernelArgs", err)
	}

	// Palette too small.
	smallPalette, err := session.NewByteBuffer(16)
	if err != nil {
		t.Fatalf("NewByteBuffer(small palette): %v", err)
	}
	err = session.Run(KernelPaletteExpandRGBA, KernelArgs{
		Inputs:  map[string]Buffer{"src": idxSrc, "palette": smallPalette},
		Outputs: map[string]Buffer{"dst": rgbaDst},
	})
	if !core.Is(err, ErrComputeInvalidKernelArgs) {
		t.Fatalf("Run(palette, too small) = %v, want ErrComputeInvalidKernelArgs", err)
	}
}

// filterMissingAndWrongArms exercises the src/dst-fetch and filter-validation
// error branches shared by the scanline/crt/soften/sharpen kernels.
func filterMissingAndWrongArms(t *testing.T, kernel string) {
	t.Helper()
	session := requireCoverageSession(t)
	src := newRGBA8PixelBuffer(t, session, 2, 2)

	// Missing dst.
	runMissingDst(t, session, kernel, src)

	// Wrong dst type.
	bb, err := session.NewByteBuffer(16)
	if err != nil {
		t.Fatalf("NewByteBuffer: %v", err)
	}
	runWrongDstType(t, session, kernel, src, bb)

	// Wrong src type.
	dst := newRGBA8PixelBuffer(t, session, 2, 2)
	err = session.Run(kernel, KernelArgs{
		Inputs:  map[string]Buffer{"src": bb},
		Outputs: map[string]Buffer{"dst": dst},
	})
	if !core.Is(err, ErrComputeInvalidBuffer) {
		t.Fatalf("Run(%s, wrong src type) = %v, want ErrComputeInvalidBuffer", kernel, err)
	}

	// validateFilterBuffers failure: mismatched dimensions.
	bigDst, err := session.NewPixelBuffer(PixelBufferDesc{Width: 4, Height: 2, Stride: 16, Format: PixelRGBA8})
	if err != nil {
		t.Fatalf("NewPixelBuffer(big dst): %v", err)
	}
	err = session.Run(kernel, KernelArgs{
		Inputs:  map[string]Buffer{"src": src},
		Outputs: map[string]Buffer{"dst": bigDst},
	})
	if !core.Is(err, ErrComputeInvalidKernelArgs) {
		t.Fatalf("Run(%s, filter validation) = %v, want ErrComputeInvalidKernelArgs", kernel, err)
	}
}

func TestComputeCoverage_ScanlineFilter_SecondBufferArms_Bad(t *testing.T) {
	filterMissingAndWrongArms(t, KernelScanlineFilter)
}

func TestComputeCoverage_CRTFilter_SecondBufferArms_Bad(t *testing.T) {
	filterMissingAndWrongArms(t, KernelCRTFilter)

	// CRT-specific: both unitScalar arms (scanline_strength then mask_strength).
	session := requireCoverageSession(t)
	src := newRGBA8PixelBuffer(t, session, 2, 2)
	dst := newRGBA8PixelBuffer(t, session, 2, 2)
	err := session.Run(KernelCRTFilter, KernelArgs{
		Inputs:  map[string]Buffer{"src": src},
		Outputs: map[string]Buffer{"dst": dst},
		Scalars: map[string]float64{"scanline_strength": 2.0},
	})
	if !core.Is(err, ErrComputeInvalidScalar) {
		t.Fatalf("Run(crt, bad scanline_strength) = %v, want ErrComputeInvalidScalar", err)
	}
	err = session.Run(KernelCRTFilter, KernelArgs{
		Inputs:  map[string]Buffer{"src": src},
		Outputs: map[string]Buffer{"dst": dst},
		Scalars: map[string]float64{"mask_strength": 2.0},
	})
	if !core.Is(err, ErrComputeInvalidScalar) {
		t.Fatalf("Run(crt, bad mask_strength) = %v, want ErrComputeInvalidScalar", err)
	}
}

func TestComputeCoverage_SoftenFilter_SecondBufferArms_Bad(t *testing.T) {
	filterMissingAndWrongArms(t, KernelSoftenFilter)
}

func TestComputeCoverage_SharpenFilter_SecondBufferArms_Bad(t *testing.T) {
	filterMissingAndWrongArms(t, KernelSharpenFilter)
}

// bufferHandle is the unexported marker method on the Buffer interface; calling
// it keeps the type-assertion seam honest and covers the no-op body.
func TestComputeCoverage_bufferHandle_Good(t *testing.T) {
	var base bufferbase
	base.bufferHandle()
	if base.Size() != 0 {
		t.Fatalf("zero bufferbase Size() = %d, want 0", base.Size())
	}
}

// runMissingSrc exercises the FIRST requireBuffer (the "src" input) error arm of
// each kernel by dispatching with no inputs at all.
func runMissingSrc(t *testing.T, session Session, kernel string) {
	t.Helper()
	err := session.Run(kernel, KernelArgs{})
	if !core.Is(err, ErrComputeMissingKernelBuffer) {
		t.Fatalf("Run(%s, no inputs) = %v, want ErrComputeMissingKernelBuffer", kernel, err)
	}
	var ce *ComputeError
	if !core.As(err, &ce) {
		t.Fatalf("Run(%s, no inputs) error = %T, want *ComputeError", kernel, err)
	}
	if ce.Resource != "src" {
		t.Fatalf("Run(%s, no inputs) resource = %q, want src", kernel, ce.Resource)
	}
}

// runWrongSrcType exercises the src-side pixelbufferLocked error arm by passing
// a byte buffer (with a valid dst present) where a pixel buffer is required.
func runWrongSrcType(t *testing.T, session Session, kernel string, byteSrc Buffer, dst Buffer) {
	t.Helper()
	err := session.Run(kernel, KernelArgs{
		Inputs:  map[string]Buffer{"src": byteSrc},
		Outputs: map[string]Buffer{"dst": dst},
	})
	if !core.Is(err, ErrComputeInvalidBuffer) {
		t.Fatalf("Run(%s, byte src) = %v, want ErrComputeInvalidBuffer", kernel, err)
	}
}

func TestComputeCoverage_MissingSrc_AllKernels_Bad(t *testing.T) {
	session := requireCoverageSession(t)
	for _, kernel := range []string{
		KernelNearestScale,
		KernelIntegerScale,
		KernelBilinearScale,
		KernelRGB565ToRGBA8,
		KernelRGBA8ToBGRA8,
		KernelBGRA8ToRGBA8,
		KernelXRGB8888ToRGBA8,
		KernelPaletteExpandRGBA,
		KernelScanlineFilter,
		KernelCRTFilter,
		KernelSoftenFilter,
		KernelSharpenFilter,
	} {
		t.Run(kernel, func(t *testing.T) {
			runMissingSrc(t, session, kernel)
		})
	}
}

func TestComputeCoverage_WrongSrcType_PixelKernels_Bad(t *testing.T) {
	session := requireCoverageSession(t)
	bb, err := session.NewByteBuffer(16)
	if err != nil {
		t.Fatalf("NewByteBuffer: %v", err)
	}
	dst := newRGBA8PixelBuffer(t, session, 2, 2)

	for _, kernel := range []string{
		KernelNearestScale,
		KernelRGB565ToRGBA8,
		KernelRGBA8ToBGRA8,
		KernelBGRA8ToRGBA8,
		KernelXRGB8888ToRGBA8,
	} {
		t.Run(kernel, func(t *testing.T) {
			runWrongSrcType(t, session, kernel, bb, dst)
		})
	}
}

// TestComputeCoverage_PaletteExpand_SrcAndPaletteTypeArms_Bad pins the
// palette-expand src pixelbufferLocked arm (byte buffer as src) and the palette
// bytebufferLocked arm (pixel buffer as palette).
func TestComputeCoverage_PaletteExpand_SrcAndPaletteTypeArms_Bad(t *testing.T) {
	session := requireCoverageSession(t)
	idxSrc, err := session.NewPixelBuffer(PixelBufferDesc{Width: 2, Height: 2, Stride: 2, Format: PixelIndexed8})
	if err != nil {
		t.Fatalf("NewPixelBuffer(indexed): %v", err)
	}
	rgbaDst := newRGBA8PixelBuffer(t, session, 2, 2)
	bb, err := session.NewByteBuffer(256 * 4)
	if err != nil {
		t.Fatalf("NewByteBuffer: %v", err)
	}

	// Byte buffer as src -> src pixelbufferLocked fails.
	err = session.Run(KernelPaletteExpandRGBA, KernelArgs{
		Inputs:  map[string]Buffer{"src": bb, "palette": bb},
		Outputs: map[string]Buffer{"dst": rgbaDst},
	})
	if !core.Is(err, ErrComputeInvalidBuffer) {
		t.Fatalf("Run(palette, byte src) = %v, want ErrComputeInvalidBuffer", err)
	}

	// Pixel buffer as palette -> palette bytebufferLocked fails.
	err = session.Run(KernelPaletteExpandRGBA, KernelArgs{
		Inputs:  map[string]Buffer{"src": idxSrc, "palette": rgbaDst},
		Outputs: map[string]Buffer{"dst": rgbaDst},
	})
	if !core.Is(err, ErrComputeInvalidBuffer) {
		t.Fatalf("Run(palette, pixel palette) = %v, want ErrComputeInvalidBuffer", err)
	}
}

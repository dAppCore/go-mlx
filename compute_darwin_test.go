//go:build darwin && arm64 && !nomlx

package mlx

import (
	"errors"
	"testing"
)

func requireComputeSession(t *testing.T) Session {
	t.Helper()
	if !MetalAvailable() {
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
	return session
}

func TestComputeSession_ByteBufferRoundTrip_Good(t *testing.T) {
	session := requireComputeSession(t)

	buffer, err := session.NewByteBuffer(4)
	if err != nil {
		t.Fatalf("NewByteBuffer: %v", err)
	}
	if err := buffer.Upload([]byte{1, 2, 3, 4}); err != nil {
		t.Fatalf("Upload: %v", err)
	}
	got, err := buffer.Read()
	if err != nil {
		t.Fatalf("Read: %v", err)
	}
	want := []byte{1, 2, 3, 4}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("byte[%d] = %d, want %d", i, got[i], want[i])
		}
	}
}

func TestComputeSession_RGB565ToRGBA8_Good(t *testing.T) {
	session := requireComputeSession(t)

	src, err := session.NewPixelBuffer(PixelBufferDesc{
		Width:  2,
		Height: 1,
		Stride: 4,
		Format: PixelRGB565,
	})
	if err != nil {
		t.Fatalf("NewPixelBuffer(src): %v", err)
	}
	dst, err := session.NewPixelBuffer(PixelBufferDesc{
		Width:  2,
		Height: 1,
		Stride: 8,
		Format: PixelRGBA8,
	})
	if err != nil {
		t.Fatalf("NewPixelBuffer(dst): %v", err)
	}

	if err := src.Upload([]byte{
		0x00, 0xF8, // red
		0xE0, 0x07, // green
	}); err != nil {
		t.Fatalf("Upload(src): %v", err)
	}

	if err := session.Run(KernelRGB565ToRGBA8, KernelArgs{
		Inputs:  map[string]Buffer{"src": src},
		Outputs: map[string]Buffer{"dst": dst},
	}); err != nil {
		t.Fatalf("Run(rgb565_to_rgba8): %v", err)
	}
	if err := session.Sync(); err != nil {
		t.Fatalf("Sync: %v", err)
	}

	got, err := dst.Read()
	if err != nil {
		t.Fatalf("Read(dst): %v", err)
	}
	want := []byte{
		255, 0, 0, 255,
		0, 255, 0, 255,
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("rgba[%d] = %d, want %d", i, got[i], want[i])
		}
	}
}

func TestComputeSession_NearestScale_Good(t *testing.T) {
	session := requireComputeSession(t)

	src, err := session.NewPixelBuffer(PixelBufferDesc{
		Width:  2,
		Height: 2,
		Stride: 8,
		Format: PixelRGBA8,
	})
	if err != nil {
		t.Fatalf("NewPixelBuffer(src): %v", err)
	}
	dst, err := session.NewPixelBuffer(PixelBufferDesc{
		Width:  4,
		Height: 4,
		Stride: 16,
		Format: PixelRGBA8,
	})
	if err != nil {
		t.Fatalf("NewPixelBuffer(dst): %v", err)
	}

	if err := src.Upload([]byte{
		255, 0, 0, 255, 0, 255, 0, 255,
		0, 0, 255, 255, 255, 255, 255, 255,
	}); err != nil {
		t.Fatalf("Upload(src): %v", err)
	}

	if err := session.Run(KernelNearestScale, KernelArgs{
		Inputs:  map[string]Buffer{"src": src},
		Outputs: map[string]Buffer{"dst": dst},
	}); err != nil {
		t.Fatalf("Run(nearest_scale): %v", err)
	}
	if err := session.Sync(); err != nil {
		t.Fatalf("Sync: %v", err)
	}

	got, err := dst.Read()
	if err != nil {
		t.Fatalf("Read(dst): %v", err)
	}

	checkPixel := func(pixelX, pixelY int, want [4]byte) {
		base := pixelY*16 + pixelX*4
		for channel := 0; channel < 4; channel++ {
			if got[base+channel] != want[channel] {
				t.Fatalf("pixel (%d,%d) channel %d = %d, want %d", pixelX, pixelY, channel, got[base+channel], want[channel])
			}
		}
	}

	checkPixel(0, 0, [4]byte{255, 0, 0, 255})
	checkPixel(3, 0, [4]byte{0, 255, 0, 255})
	checkPixel(0, 3, [4]byte{0, 0, 255, 255})
	checkPixel(3, 3, [4]byte{255, 255, 255, 255})
}

func TestComputeSession_PaletteExpandRGBA_Good(t *testing.T) {
	session := requireComputeSession(t)

	src, err := session.NewPixelBuffer(PixelBufferDesc{
		Width:  2,
		Height: 1,
		Stride: 2,
		Format: PixelIndexed8,
	})
	if err != nil {
		t.Fatalf("NewPixelBuffer(src): %v", err)
	}
	dst, err := session.NewPixelBuffer(PixelBufferDesc{
		Width:  2,
		Height: 1,
		Stride: 8,
		Format: PixelRGBA8,
	})
	if err != nil {
		t.Fatalf("NewPixelBuffer(dst): %v", err)
	}
	palette, err := session.NewByteBuffer(256 * 4)
	if err != nil {
		t.Fatalf("NewByteBuffer(palette): %v", err)
	}

	paletteBytes := make([]byte, 256*4)
	copy(paletteBytes[0:4], []byte{255, 0, 0, 255})
	copy(paletteBytes[4:8], []byte{0, 0, 255, 255})
	if err := palette.Upload(paletteBytes); err != nil {
		t.Fatalf("Upload(palette): %v", err)
	}
	if err := src.Upload([]byte{0, 1}); err != nil {
		t.Fatalf("Upload(src): %v", err)
	}

	if err := session.Run(KernelPaletteExpandRGBA, KernelArgs{
		Inputs: map[string]Buffer{
			"src":     src,
			"palette": palette,
		},
		Outputs: map[string]Buffer{"dst": dst},
	}); err != nil {
		t.Fatalf("Run(palette_expand_rgba8): %v", err)
	}

	got, err := dst.Read()
	if err != nil {
		t.Fatalf("Read(dst): %v", err)
	}
	want := []byte{
		255, 0, 0, 255,
		0, 0, 255, 255,
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("palette rgba[%d] = %d, want %d", i, got[i], want[i])
		}
	}

	metrics := session.Metrics()
	if metrics.Passes == 0 {
		t.Fatal("expected session metrics to record at least one pass")
	}
	if metrics.LastKernel != KernelPaletteExpandRGBA {
		t.Fatalf("LastKernel = %q, want %q", metrics.LastKernel, KernelPaletteExpandRGBA)
	}
}

func TestComputeSession_IntegerScale_Good(t *testing.T) {
	session := requireComputeSession(t)

	src, err := session.NewPixelBuffer(PixelBufferDesc{
		Width:  2,
		Height: 2,
		Stride: 8,
		Format: PixelRGBA8,
	})
	if err != nil {
		t.Fatalf("NewPixelBuffer(src): %v", err)
	}
	dst, err := session.NewPixelBuffer(PixelBufferDesc{
		Width:  4,
		Height: 4,
		Stride: 16,
		Format: PixelRGBA8,
	})
	if err != nil {
		t.Fatalf("NewPixelBuffer(dst): %v", err)
	}

	if err := src.Upload([]byte{
		255, 0, 0, 255, 0, 255, 0, 255,
		0, 0, 255, 255, 255, 255, 255, 255,
	}); err != nil {
		t.Fatalf("Upload(src): %v", err)
	}

	if err := session.Run(KernelIntegerScale, KernelArgs{
		Inputs:  map[string]Buffer{"src": src},
		Outputs: map[string]Buffer{"dst": dst},
	}); err != nil {
		t.Fatalf("Run(integer_scale): %v", err)
	}

	got, err := dst.Read()
	if err != nil {
		t.Fatalf("Read(dst): %v", err)
	}

	checkPixel := func(pixelX, pixelY int, want [4]byte) {
		base := pixelY*16 + pixelX*4
		for channel := 0; channel < 4; channel++ {
			if got[base+channel] != want[channel] {
				t.Fatalf("pixel (%d,%d) channel %d = %d, want %d", pixelX, pixelY, channel, got[base+channel], want[channel])
			}
		}
	}

	checkPixel(0, 0, [4]byte{255, 0, 0, 255})
	checkPixel(3, 0, [4]byte{0, 255, 0, 255})
	checkPixel(0, 3, [4]byte{0, 0, 255, 255})
	checkPixel(3, 3, [4]byte{255, 255, 255, 255})
}

func TestComputeSession_IntegerScaleRejectsNonIntegerFactor_Bad(t *testing.T) {
	session := requireComputeSession(t)

	src, err := session.NewPixelBuffer(PixelBufferDesc{
		Width:  2,
		Height: 2,
		Stride: 8,
		Format: PixelRGBA8,
	})
	if err != nil {
		t.Fatalf("NewPixelBuffer(src): %v", err)
	}
	dst, err := session.NewPixelBuffer(PixelBufferDesc{
		Width:  3,
		Height: 4,
		Stride: 12,
		Format: PixelRGBA8,
	})
	if err != nil {
		t.Fatalf("NewPixelBuffer(dst): %v", err)
	}

	if err := session.Run(KernelIntegerScale, KernelArgs{
		Inputs:  map[string]Buffer{"src": src},
		Outputs: map[string]Buffer{"dst": dst},
	}); err == nil {
		t.Fatal("expected integer_scale to reject non-integer output dimensions")
	}
}

func TestComputeSession_BilinearScale_Good(t *testing.T) {
	session := requireComputeSession(t)

	src, err := session.NewPixelBuffer(PixelBufferDesc{
		Width:  2,
		Height: 1,
		Stride: 8,
		Format: PixelRGBA8,
	})
	if err != nil {
		t.Fatalf("NewPixelBuffer(src): %v", err)
	}
	dst, err := session.NewPixelBuffer(PixelBufferDesc{
		Width:  3,
		Height: 1,
		Stride: 12,
		Format: PixelRGBA8,
	})
	if err != nil {
		t.Fatalf("NewPixelBuffer(dst): %v", err)
	}

	if err := src.Upload([]byte{
		255, 0, 0, 255,
		0, 0, 255, 255,
	}); err != nil {
		t.Fatalf("Upload(src): %v", err)
	}

	if err := session.Run(KernelBilinearScale, KernelArgs{
		Inputs:  map[string]Buffer{"src": src},
		Outputs: map[string]Buffer{"dst": dst},
	}); err != nil {
		t.Fatalf("Run(bilinear_scale): %v", err)
	}

	got, err := dst.Read()
	if err != nil {
		t.Fatalf("Read(dst): %v", err)
	}

	wantMiddle := [4]byte{128, 0, 128, 255}
	for channel := 0; channel < 4; channel++ {
		if got[4+channel] != wantMiddle[channel] {
			t.Fatalf("middle pixel channel %d = %d, want %d", channel, got[4+channel], wantMiddle[channel])
		}
	}
}

func TestComputeSession_ChannelSwizzleRoundTrip_Good(t *testing.T) {
	session := requireComputeSession(t)

	rgba, err := session.NewPixelBuffer(PixelBufferDesc{
		Width:  1,
		Height: 1,
		Stride: 4,
		Format: PixelRGBA8,
	})
	if err != nil {
		t.Fatalf("NewPixelBuffer(rgba): %v", err)
	}
	bgra, err := session.NewPixelBuffer(PixelBufferDesc{
		Width:  1,
		Height: 1,
		Stride: 4,
		Format: PixelBGRA8,
	})
	if err != nil {
		t.Fatalf("NewPixelBuffer(bgra): %v", err)
	}
	roundTrip, err := session.NewPixelBuffer(PixelBufferDesc{
		Width:  1,
		Height: 1,
		Stride: 4,
		Format: PixelRGBA8,
	})
	if err != nil {
		t.Fatalf("NewPixelBuffer(roundTrip): %v", err)
	}

	original := []byte{1, 2, 3, 4}
	if err := rgba.Upload(original); err != nil {
		t.Fatalf("Upload(rgba): %v", err)
	}

	if err := session.Run(KernelRGBA8ToBGRA8, KernelArgs{
		Inputs:  map[string]Buffer{"src": rgba},
		Outputs: map[string]Buffer{"dst": bgra},
	}); err != nil {
		t.Fatalf("Run(rgba8_to_bgra8): %v", err)
	}

	swizzled, err := bgra.Read()
	if err != nil {
		t.Fatalf("Read(bgra): %v", err)
	}
	wantSwizzled := []byte{3, 2, 1, 4}
	for i := range wantSwizzled {
		if swizzled[i] != wantSwizzled[i] {
			t.Fatalf("swizzled[%d] = %d, want %d", i, swizzled[i], wantSwizzled[i])
		}
	}

	if err := session.Run(KernelBGRA8ToRGBA8, KernelArgs{
		Inputs:  map[string]Buffer{"src": bgra},
		Outputs: map[string]Buffer{"dst": roundTrip},
	}); err != nil {
		t.Fatalf("Run(bgra8_to_rgba8): %v", err)
	}

	got, err := roundTrip.Read()
	if err != nil {
		t.Fatalf("Read(roundTrip): %v", err)
	}
	for i := range original {
		if got[i] != original[i] {
			t.Fatalf("roundTrip[%d] = %d, want %d", i, got[i], original[i])
		}
	}
}

func TestComputeSession_XRGB8888ToRGBA8_Good(t *testing.T) {
	session := requireComputeSession(t)

	src, err := session.NewPixelBuffer(PixelBufferDesc{
		Width:  1,
		Height: 1,
		Stride: 4,
		Format: PixelXRGB8888,
	})
	if err != nil {
		t.Fatalf("NewPixelBuffer(src): %v", err)
	}
	dst, err := session.NewPixelBuffer(PixelBufferDesc{
		Width:  1,
		Height: 1,
		Stride: 4,
		Format: PixelRGBA8,
	})
	if err != nil {
		t.Fatalf("NewPixelBuffer(dst): %v", err)
	}

	if err := src.Upload([]byte{0x11, 0x22, 0x33, 0x00}); err != nil {
		t.Fatalf("Upload(src): %v", err)
	}

	if err := session.Run(KernelXRGB8888ToRGBA8, KernelArgs{
		Inputs:  map[string]Buffer{"src": src},
		Outputs: map[string]Buffer{"dst": dst},
	}); err != nil {
		t.Fatalf("Run(xrgb8888_to_rgba8): %v", err)
	}

	got, err := dst.Read()
	if err != nil {
		t.Fatalf("Read(dst): %v", err)
	}
	want := []byte{0x33, 0x22, 0x11, 0xFF}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("rgba[%d] = %d, want %d", i, got[i], want[i])
		}
	}
}

func TestComputeSession_ScanlineFilter_Good(t *testing.T) {
	session := requireComputeSession(t)

	src, err := session.NewPixelBuffer(PixelBufferDesc{
		Width:  1,
		Height: 2,
		Stride: 4,
		Format: PixelRGBA8,
	})
	if err != nil {
		t.Fatalf("NewPixelBuffer(src): %v", err)
	}
	dst, err := session.NewPixelBuffer(PixelBufferDesc{
		Width:  1,
		Height: 2,
		Stride: 4,
		Format: PixelRGBA8,
	})
	if err != nil {
		t.Fatalf("NewPixelBuffer(dst): %v", err)
	}

	if err := src.Upload([]byte{
		200, 200, 200, 255,
		200, 200, 200, 255,
	}); err != nil {
		t.Fatalf("Upload(src): %v", err)
	}

	if err := session.Run(KernelScanlineFilter, KernelArgs{
		Inputs:  map[string]Buffer{"src": src},
		Outputs: map[string]Buffer{"dst": dst},
		Scalars: map[string]float64{"strength": 0.5},
	}); err != nil {
		t.Fatalf("Run(scanline_filter): %v", err)
	}

	got, err := dst.Read()
	if err != nil {
		t.Fatalf("Read(dst): %v", err)
	}
	want := []byte{
		200, 200, 200, 255,
		100, 100, 100, 255,
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("scanline[%d] = %d, want %d", i, got[i], want[i])
		}
	}
}

func TestComputeSession_CRTFilter_Good(t *testing.T) {
	session := requireComputeSession(t)

	src, err := session.NewPixelBuffer(PixelBufferDesc{
		Width:  3,
		Height: 1,
		Stride: 12,
		Format: PixelRGBA8,
	})
	if err != nil {
		t.Fatalf("NewPixelBuffer(src): %v", err)
	}
	dst, err := session.NewPixelBuffer(PixelBufferDesc{
		Width:  3,
		Height: 1,
		Stride: 12,
		Format: PixelRGBA8,
	})
	if err != nil {
		t.Fatalf("NewPixelBuffer(dst): %v", err)
	}

	if err := src.Upload([]byte{
		240, 240, 240, 255,
		240, 240, 240, 255,
		240, 240, 240, 255,
	}); err != nil {
		t.Fatalf("Upload(src): %v", err)
	}

	if err := session.Run(KernelCRTFilter, KernelArgs{
		Inputs:  map[string]Buffer{"src": src},
		Outputs: map[string]Buffer{"dst": dst},
		Scalars: map[string]float64{"scanline_strength": 0, "mask_strength": 0.5},
	}); err != nil {
		t.Fatalf("Run(crt_filter): %v", err)
	}

	got, err := dst.Read()
	if err != nil {
		t.Fatalf("Read(dst): %v", err)
	}
	want := []byte{
		240, 120, 120, 255,
		120, 240, 120, 255,
		120, 120, 240, 255,
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("crt[%d] = %d, want %d", i, got[i], want[i])
		}
	}
}

func TestComputeSession_SoftenFilter_Good(t *testing.T) {
	session := requireComputeSession(t)

	src, err := session.NewPixelBuffer(PixelBufferDesc{
		Width:  3,
		Height: 1,
		Stride: 12,
		Format: PixelRGBA8,
	})
	if err != nil {
		t.Fatalf("NewPixelBuffer(src): %v", err)
	}
	dst, err := session.NewPixelBuffer(PixelBufferDesc{
		Width:  3,
		Height: 1,
		Stride: 12,
		Format: PixelRGBA8,
	})
	if err != nil {
		t.Fatalf("NewPixelBuffer(dst): %v", err)
	}

	if err := src.Upload([]byte{
		0, 0, 0, 255,
		255, 255, 255, 255,
		0, 0, 0, 255,
	}); err != nil {
		t.Fatalf("Upload(src): %v", err)
	}

	if err := session.Run(KernelSoftenFilter, KernelArgs{
		Inputs:  map[string]Buffer{"src": src},
		Outputs: map[string]Buffer{"dst": dst},
		Scalars: map[string]float64{"strength": 1.0},
	}); err != nil {
		t.Fatalf("Run(soften_filter): %v", err)
	}

	got, err := dst.Read()
	if err != nil {
		t.Fatalf("Read(dst): %v", err)
	}
	want := []byte{
		85, 85, 85, 255,
		85, 85, 85, 255,
		85, 85, 85, 255,
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("soften[%d] = %d, want %d", i, got[i], want[i])
		}
	}
}

func TestComputeSession_SharpenFilter_Good(t *testing.T) {
	session := requireComputeSession(t)

	src, err := session.NewPixelBuffer(PixelBufferDesc{
		Width:  3,
		Height: 1,
		Stride: 12,
		Format: PixelRGBA8,
	})
	if err != nil {
		t.Fatalf("NewPixelBuffer(src): %v", err)
	}
	dst, err := session.NewPixelBuffer(PixelBufferDesc{
		Width:  3,
		Height: 1,
		Stride: 12,
		Format: PixelRGBA8,
	})
	if err != nil {
		t.Fatalf("NewPixelBuffer(dst): %v", err)
	}

	if err := src.Upload([]byte{
		64, 64, 64, 255,
		128, 128, 128, 255,
		64, 64, 64, 255,
	}); err != nil {
		t.Fatalf("Upload(src): %v", err)
	}

	if err := session.Run(KernelSharpenFilter, KernelArgs{
		Inputs:  map[string]Buffer{"src": src},
		Outputs: map[string]Buffer{"dst": dst},
		Scalars: map[string]float64{"strength": 1.0},
	}); err != nil {
		t.Fatalf("Run(sharpen_filter): %v", err)
	}

	got, err := dst.Read()
	if err != nil {
		t.Fatalf("Read(dst): %v", err)
	}
	want := []byte{
		43, 43, 43, 255,
		171, 171, 171, 255,
		43, 43, 43, 255,
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("sharpen[%d] = %d, want %d", i, got[i], want[i])
		}
	}
}

func TestComputeSession_ScanlineFilterRejectsInvalidStrength_Bad(t *testing.T) {
	session := requireComputeSession(t)

	src, err := session.NewPixelBuffer(PixelBufferDesc{
		Width:  1,
		Height: 1,
		Stride: 4,
		Format: PixelRGBA8,
	})
	if err != nil {
		t.Fatalf("NewPixelBuffer(src): %v", err)
	}
	dst, err := session.NewPixelBuffer(PixelBufferDesc{
		Width:  1,
		Height: 1,
		Stride: 4,
		Format: PixelRGBA8,
	})
	if err != nil {
		t.Fatalf("NewPixelBuffer(dst): %v", err)
	}

	err = session.Run(KernelScanlineFilter, KernelArgs{
		Inputs:  map[string]Buffer{"src": src},
		Outputs: map[string]Buffer{"dst": dst},
		Scalars: map[string]float64{"strength": 1.5},
	})
	if err == nil {
		t.Fatal("expected scanline_filter to reject strength outside [0,1]")
	}
	if !errors.Is(err, ErrComputeInvalidScalar) {
		t.Fatalf("Run(scanline_filter) error = %v, want ErrComputeInvalidScalar", err)
	}
	var computeErr *ComputeError
	if !errors.As(err, &computeErr) {
		t.Fatalf("Run(scanline_filter) error = %T, want *ComputeError", err)
	}
	if computeErr.Kernel != KernelScanlineFilter || computeErr.Resource != "strength" {
		t.Fatalf("ComputeError = %+v, want kernel=%q resource=%q", computeErr, KernelScanlineFilter, "strength")
	}
}

func TestComputeSession_RunUnknownKernel_ReturnsStructuredError_Bad(t *testing.T) {
	session := requireComputeSession(t)

	err := session.Run("not_a_kernel", KernelArgs{})
	if err == nil {
		t.Fatal("expected unknown kernel error")
	}
	if !errors.Is(err, ErrComputeUnknownKernel) {
		t.Fatalf("Run(not_a_kernel) error = %v, want ErrComputeUnknownKernel", err)
	}
	var computeErr *ComputeError
	if !errors.As(err, &computeErr) {
		t.Fatalf("Run(not_a_kernel) error = %T, want *ComputeError", err)
	}
	if computeErr.Kernel != "not_a_kernel" {
		t.Fatalf("Kernel = %q, want %q", computeErr.Kernel, "not_a_kernel")
	}
}

func TestComputeSession_RunMissingBuffer_ReturnsStructuredError_Bad(t *testing.T) {
	session := requireComputeSession(t)

	err := session.Run(KernelRGB565ToRGBA8, KernelArgs{})
	if err == nil {
		t.Fatal("expected missing kernel buffer error")
	}
	if !errors.Is(err, ErrComputeMissingKernelBuffer) {
		t.Fatalf("Run(rgb565_to_rgba8) error = %v, want ErrComputeMissingKernelBuffer", err)
	}
	var computeErr *ComputeError
	if !errors.As(err, &computeErr) {
		t.Fatalf("Run(rgb565_to_rgba8) error = %T, want *ComputeError", err)
	}
	if computeErr.Kernel != KernelRGB565ToRGBA8 || computeErr.Resource != "src" {
		t.Fatalf("ComputeError = %+v, want kernel=%q resource=%q", computeErr, KernelRGB565ToRGBA8, "src")
	}
}

func TestComputeSession_IntegerScaleFormatErrorUsesPublicKernel_Bad(t *testing.T) {
	session := requireComputeSession(t)

	src, err := session.NewPixelBuffer(PixelBufferDesc{
		Width:  1,
		Height: 1,
		Stride: 4,
		Format: PixelRGBA8,
	})
	if err != nil {
		t.Fatalf("NewPixelBuffer(src): %v", err)
	}
	dst, err := session.NewPixelBuffer(PixelBufferDesc{
		Width:  2,
		Height: 2,
		Stride: 8,
		Format: PixelBGRA8,
	})
	if err != nil {
		t.Fatalf("NewPixelBuffer(dst): %v", err)
	}

	err = session.Run(KernelIntegerScale, KernelArgs{
		Inputs:  map[string]Buffer{"src": src},
		Outputs: map[string]Buffer{"dst": dst},
	})
	if err == nil {
		t.Fatal("expected integer_scale to reject mixed pixel formats")
	}
	if !errors.Is(err, ErrComputeInvalidKernelArgs) {
		t.Fatalf("Run(integer_scale) error = %v, want ErrComputeInvalidKernelArgs", err)
	}
	var computeErr *ComputeError
	if !errors.As(err, &computeErr) {
		t.Fatalf("Run(integer_scale) error = %T, want *ComputeError", err)
	}
	if computeErr.Kernel != KernelIntegerScale || computeErr.Resource != "format" {
		t.Fatalf("ComputeError = %+v, want kernel=%q resource=%q", computeErr, KernelIntegerScale, "format")
	}
}

func TestComputeSession_ChannelSwizzleErrorUsesRequestedKernel_Bad(t *testing.T) {
	session := requireComputeSession(t)

	src, err := session.NewPixelBuffer(PixelBufferDesc{
		Width:  1,
		Height: 1,
		Stride: 4,
		Format: PixelRGBA8,
	})
	if err != nil {
		t.Fatalf("NewPixelBuffer(src): %v", err)
	}
	dst, err := session.NewPixelBuffer(PixelBufferDesc{
		Width:  1,
		Height: 1,
		Stride: 4,
		Format: PixelRGBA8,
	})
	if err != nil {
		t.Fatalf("NewPixelBuffer(dst): %v", err)
	}

	err = session.Run(KernelBGRA8ToRGBA8, KernelArgs{
		Inputs:  map[string]Buffer{"src": src},
		Outputs: map[string]Buffer{"dst": dst},
	})
	if err == nil {
		t.Fatal("expected bgra8_to_rgba8 to reject an rgba8 source")
	}
	if !errors.Is(err, ErrComputeInvalidKernelArgs) {
		t.Fatalf("Run(bgra8_to_rgba8) error = %v, want ErrComputeInvalidKernelArgs", err)
	}
	var computeErr *ComputeError
	if !errors.As(err, &computeErr) {
		t.Fatalf("Run(bgra8_to_rgba8) error = %T, want *ComputeError", err)
	}
	if computeErr.Kernel != KernelBGRA8ToRGBA8 || computeErr.Resource != "src" {
		t.Fatalf("ComputeError = %+v, want kernel=%q resource=%q", computeErr, KernelBGRA8ToRGBA8, "src")
	}
}

func TestComputeSession_ClosedSessionReturnsStructuredError_Bad(t *testing.T) {
	session := requireComputeSession(t)
	if err := session.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	_, err := session.NewByteBuffer(8)
	if err == nil {
		t.Fatal("expected NewByteBuffer on a closed session to fail")
	}
	if !errors.Is(err, ErrComputeClosed) {
		t.Fatalf("NewByteBuffer() error = %v, want ErrComputeClosed", err)
	}
}

func TestComputeSession_MetricsTrackDispatchAndSync_Good(t *testing.T) {
	session := requireComputeSession(t)

	src, err := session.NewPixelBuffer(PixelBufferDesc{
		Width:  1,
		Height: 1,
		Stride: 2,
		Format: PixelRGB565,
	})
	if err != nil {
		t.Fatalf("NewPixelBuffer(src): %v", err)
	}
	dst, err := session.NewPixelBuffer(PixelBufferDesc{
		Width:  1,
		Height: 1,
		Stride: 4,
		Format: PixelRGBA8,
	})
	if err != nil {
		t.Fatalf("NewPixelBuffer(dst): %v", err)
	}

	if err := src.Upload([]byte{0x00, 0xF8}); err != nil {
		t.Fatalf("Upload(src): %v", err)
	}
	if err := session.Run(KernelRGB565ToRGBA8, KernelArgs{
		Inputs:  map[string]Buffer{"src": src},
		Outputs: map[string]Buffer{"dst": dst},
	}); err != nil {
		t.Fatalf("Run(rgb565_to_rgba8): %v", err)
	}
	if err := session.Sync(); err != nil {
		t.Fatalf("Sync: %v", err)
	}

	metrics := session.Metrics()
	if metrics.Passes != 1 {
		t.Fatalf("Passes = %d, want 1", metrics.Passes)
	}
	if metrics.LastKernel != KernelRGB565ToRGBA8 {
		t.Fatalf("LastKernel = %q, want %q", metrics.LastKernel, KernelRGB565ToRGBA8)
	}
	if metrics.LastDispatchDuration <= 0 {
		t.Fatalf("LastDispatchDuration = %v, want > 0", metrics.LastDispatchDuration)
	}
	if metrics.LastSyncDuration <= 0 {
		t.Fatalf("LastSyncDuration = %v, want > 0", metrics.LastSyncDuration)
	}
	if metrics.TotalDispatchDuration < metrics.LastDispatchDuration {
		t.Fatalf("TotalDispatchDuration = %v, want >= %v", metrics.TotalDispatchDuration, metrics.LastDispatchDuration)
	}
	if metrics.TotalSyncDuration < metrics.LastSyncDuration {
		t.Fatalf("TotalSyncDuration = %v, want >= %v", metrics.TotalSyncDuration, metrics.LastSyncDuration)
	}
	if metrics.PeakMemoryBytes < metrics.ActiveMemoryBytes {
		t.Fatalf("PeakMemoryBytes = %d, want >= ActiveMemoryBytes %d", metrics.PeakMemoryBytes, metrics.ActiveMemoryBytes)
	}
	if metrics.ActiveMemoryBytes == 0 {
		t.Fatal("ActiveMemoryBytes should report live session allocations")
	}
}

func TestComputeSession_MetricsClampToZeroWhenBelowBase_Good(t *testing.T) {
	session := &computeSession{
		metrics: SessionMetrics{
			ActiveMemoryBytes: 123,
			PeakMemoryBytes:   456,
		},
		frame: frameState{
			active: true,
			metrics: FrameMetrics{
				ActiveMemoryBytes: 789,
				PeakMemoryBytes:   321,
			},
			baseActiveMemory: ^uint64(0),
			basePeakMemory:   ^uint64(0),
		},
		baseActiveMemory: ^uint64(0),
		basePeakMemory:   ^uint64(0),
	}

	session.updateMemoryMetricsLocked()
	session.updateFrameMetricsLocked()

	if session.metrics.ActiveMemoryBytes != 0 || session.metrics.PeakMemoryBytes != 0 {
		t.Fatalf("SessionMetrics = %+v, want zeroed active/peak memory", session.metrics)
	}
	if session.frame.metrics.ActiveMemoryBytes != 0 || session.frame.metrics.PeakMemoryBytes != 0 {
		t.Fatalf("FrameMetrics = %+v, want zeroed active/peak memory", session.frame.metrics)
	}
}

func TestComputeSession_FrameLifecycle_Good(t *testing.T) {
	session := requireComputeSession(t)

	src, err := session.NewPixelBuffer(PixelBufferDesc{
		Width:  1,
		Height: 1,
		Stride: 2,
		Format: PixelRGB565,
	})
	if err != nil {
		t.Fatalf("NewPixelBuffer(src): %v", err)
	}
	dst, err := session.NewPixelBuffer(PixelBufferDesc{
		Width:  1,
		Height: 1,
		Stride: 4,
		Format: PixelRGBA8,
	})
	if err != nil {
		t.Fatalf("NewPixelBuffer(dst): %v", err)
	}

	if err := session.BeginFrame(); err != nil {
		t.Fatalf("BeginFrame: %v", err)
	}
	if err := src.Upload([]byte{0x00, 0xF8}); err != nil {
		t.Fatalf("Upload(src): %v", err)
	}
	if err := session.Run(KernelRGB565ToRGBA8, KernelArgs{
		Inputs:  map[string]Buffer{"src": src},
		Outputs: map[string]Buffer{"dst": dst},
	}); err != nil {
		t.Fatalf("Run(rgb565_to_rgba8): %v", err)
	}

	frameMetrics, err := session.FinishFrame()
	if err != nil {
		t.Fatalf("FinishFrame: %v", err)
	}
	if frameMetrics.Frame != 1 {
		t.Fatalf("Frame = %d, want 1", frameMetrics.Frame)
	}
	if frameMetrics.Passes != 1 {
		t.Fatalf("Passes = %d, want 1", frameMetrics.Passes)
	}
	if frameMetrics.LastKernel != KernelRGB565ToRGBA8 {
		t.Fatalf("LastKernel = %q, want %q", frameMetrics.LastKernel, KernelRGB565ToRGBA8)
	}
	if frameMetrics.DispatchDuration <= 0 {
		t.Fatalf("DispatchDuration = %v, want > 0", frameMetrics.DispatchDuration)
	}
	if frameMetrics.SyncDuration <= 0 {
		t.Fatalf("SyncDuration = %v, want > 0", frameMetrics.SyncDuration)
	}
	if frameMetrics.TotalDuration < frameMetrics.DispatchDuration {
		t.Fatalf("TotalDuration = %v, want >= %v", frameMetrics.TotalDuration, frameMetrics.DispatchDuration)
	}
	if got := session.FrameMetrics(); got != frameMetrics {
		t.Fatalf("FrameMetrics() = %+v, want %+v", got, frameMetrics)
	}
}

func TestComputeSession_RunImplicitFrameAndFinish_Good(t *testing.T) {
	session := requireComputeSession(t)

	src, err := session.NewPixelBuffer(PixelBufferDesc{
		Width:  1,
		Height: 1,
		Stride: 2,
		Format: PixelRGB565,
	})
	if err != nil {
		t.Fatalf("NewPixelBuffer(src): %v", err)
	}
	dst, err := session.NewPixelBuffer(PixelBufferDesc{
		Width:  1,
		Height: 1,
		Stride: 4,
		Format: PixelRGBA8,
	})
	if err != nil {
		t.Fatalf("NewPixelBuffer(dst): %v", err)
	}

	if err := src.Upload([]byte{0x00, 0xF8}); err != nil {
		t.Fatalf("Upload(src): %v", err)
	}
	if err := session.Run(KernelRGB565ToRGBA8, KernelArgs{
		Inputs:  map[string]Buffer{"src": src},
		Outputs: map[string]Buffer{"dst": dst},
	}); err != nil {
		t.Fatalf("Run(rgb565_to_rgba8): %v", err)
	}

	frameMetrics, err := session.FinishFrame()
	if err != nil {
		t.Fatalf("FinishFrame: %v", err)
	}
	if frameMetrics.Frame != 1 || frameMetrics.Passes != 1 {
		t.Fatalf("FinishFrame() = %+v, want frame=1 passes=1", frameMetrics)
	}
}

func TestComputeSession_BeginFrameWhileActive_ReturnsStructuredError_Bad(t *testing.T) {
	session := requireComputeSession(t)

	if err := session.BeginFrame(); err != nil {
		t.Fatalf("BeginFrame: %v", err)
	}
	err := session.BeginFrame()
	if err == nil {
		t.Fatal("expected BeginFrame to reject an already-active frame")
	}
	if !errors.Is(err, ErrComputeInvalidState) {
		t.Fatalf("BeginFrame() error = %v, want ErrComputeInvalidState", err)
	}
}

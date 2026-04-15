//go:build darwin && arm64 && !nomlx

package mlx

import "testing"

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

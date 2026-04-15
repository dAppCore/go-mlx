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

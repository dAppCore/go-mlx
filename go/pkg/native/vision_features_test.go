// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"bytes"
	"image"
	"image/color"
	"image/png"
	"testing"
)

func TestVisionImagePatchesNoResize_Good(t *testing.T) {
	img := image.NewNRGBA(image.Rect(0, 0, 16, 16))
	img.SetNRGBA(0, 0, color.NRGBA{R: 255, A: 255})
	img.SetNRGBA(15, 15, color.NRGBA{R: 128, G: 64, B: 32, A: 255})
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		t.Fatalf("encode png: %v", err)
	}

	patches, softTokens, err := VisionImagePatches(buf.Bytes(), &VisionImageFeatureConfig{
		PatchSize: 16, MaxSoftTokens: 1, PoolingKernelSize: 1, RescaleFactor: 1.0 / 255.0,
	})
	if err != nil {
		t.Fatalf("VisionImagePatches: %v", err)
	}
	if softTokens != 1 {
		t.Fatalf("soft tokens = %d, want 1", softTokens)
	}
	if len(patches) != 16*16*3*2 {
		t.Fatalf("patch bytes = %d, want %d", len(patches), 16*16*3*2)
	}
	got := bf16Floats(patches)
	if got[0] != 1 || got[1] != 0 || got[2] != 0 {
		t.Fatalf("first pixel = %.4f/%.4f/%.4f, want 1/0/0", got[0], got[1], got[2])
	}
	last := (16*16 - 1) * 3
	if got[last] == 0 || got[last+1] == 0 || got[last+2] == 0 {
		t.Fatalf("last pixel = %.4f/%.4f/%.4f, want rescaled RGB", got[last], got[last+1], got[last+2])
	}
}

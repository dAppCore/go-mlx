// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"bytes"
	"image"
	"image/color"
	"image/jpeg"
	"image/png"
	"math"
	"os"
	"testing"
)

type nativeVisionResizeGolden struct {
	name                string
	maxSoftTokens       int32
	targetH, targetW    int32
	softTokens          int
	meanR, meanG, meanB float64
	sampleCoords        [][2]int32
	samplePixels        []float32
}

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

func TestVisionImagePixelsPatchParityNoResize_Good(t *testing.T) {
	img := image.NewNRGBA(image.Rect(0, 0, 16, 16))
	img.SetNRGBA(0, 0, color.NRGBA{R: 255, A: 255})
	img.SetNRGBA(15, 15, color.NRGBA{G: 255, B: 128, A: 255})
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		t.Fatalf("encode png: %v", err)
	}
	cfg := &VisionImageFeatureConfig{
		PatchSize: 16, MaxSoftTokens: 1, PoolingKernelSize: 1, RescaleFactor: 1.0 / 255.0,
	}
	pixels, h, w, softTokens, err := VisionImagePixels(buf.Bytes(), cfg)
	if err != nil {
		t.Fatalf("VisionImagePixels: %v", err)
	}
	if h != 16 || w != 16 || softTokens != 1 {
		t.Fatalf("VisionImagePixels geometry = %dx%d softTokens=%d, want 16x16/1", h, w, softTokens)
	}
	patches, patchSoftTokens, err := VisionImagePatches(buf.Bytes(), cfg)
	if err != nil {
		t.Fatalf("VisionImagePatches: %v", err)
	}
	if patchSoftTokens != softTokens {
		t.Fatalf("soft tokens = pixels %d patches %d", softTokens, patchSoftTokens)
	}
	want := patchifyVisionPixelsBF16(pixels, h, w, cfg.PatchSize)
	if !bytes.Equal(patches, want) {
		t.Fatal("VisionImagePixels patchified output does not match VisionImagePatches")
	}
}

func TestVisionImagePatchesResizeGoldenParity_Good(t *testing.T) {
	const patch = int32(16)
	const pixelTolerance = 5.0 / 255.0
	golden := nativeVisionResizeGolden{
		name: "video_frame", maxSoftTokens: 70,
		targetH: 336, targetW: 432, softTokens: 63,
		meanR: 0.49806723, meanG: 0.49802658, meanB: 0.49824649,
		sampleCoords: [][2]int32{
			{19, 25},
			{19, 76},
			{19, 127},
			{19, 177},
		},
		samplePixels: []float32{
			0.0549019612, 0.0549019612, 0.905882359,
			0.176470593, 0.0549019612, 0,
			0.294117659, 0.0549019612, 0.898039222,
			0.407843143, 0.0549019612, 0.321568638,
		},
	}

	data, err := os.ReadFile("../metal/model/gemma4/testdata/vision_" + golden.name + ".png")
	if err != nil {
		t.Fatalf("%s: read golden png: %v", golden.name, err)
	}
	patches, softTokens, err := VisionImagePatches(data, &VisionImageFeatureConfig{
		PatchSize: patch, MaxSoftTokens: golden.maxSoftTokens, PoolingKernelSize: 3, RescaleFactor: 1.0 / 255.0, DoResize: true,
	})
	if err != nil {
		t.Fatalf("%s: VisionImagePatches: %v", golden.name, err)
	}
	if softTokens != golden.softTokens {
		t.Fatalf("%s: soft tokens = %d, want %d", golden.name, softTokens, golden.softTokens)
	}
	if got, want := len(patches), int((golden.targetH/patch)*(golden.targetW/patch))*int(patch*patch*3)*bf16Size; got != want {
		t.Fatalf("%s: patch bytes = %d, want %d", golden.name, got, want)
	}

	values := bf16Floats(patches)
	var sumR, sumG, sumB float64
	for y := int32(0); y < golden.targetH; y++ {
		for x := int32(0); x < golden.targetW; x++ {
			r, g, b := patchifiedVisionPixel(values, golden.targetW, patch, y, x)
			sumR += float64(r)
			sumG += float64(g)
			sumB += float64(b)
		}
	}
	n := float64(golden.targetH * golden.targetW)
	for c, pair := range [][2]float64{{sumR / n, golden.meanR}, {sumG / n, golden.meanG}, {sumB / n, golden.meanB}} {
		if diff := math.Abs(pair[0] - pair[1]); diff > 2e-3 {
			t.Fatalf("%s: channel %d mean = %.8f, want %.8f (diff %.8f)", golden.name, c, pair[0], pair[1], diff)
		}
	}

	maxDiff := 0.0
	for s, coord := range golden.sampleCoords {
		r, g, b := patchifiedVisionPixel(values, golden.targetW, patch, coord[0], coord[1])
		for c, got := range []float32{r, g, b} {
			want := golden.samplePixels[s*3+c]
			if diff := math.Abs(float64(got - want)); diff > maxDiff {
				maxDiff = diff
			}
			if diff := math.Abs(float64(got - want)); diff > pixelTolerance {
				t.Fatalf("%s: sample %d coord (%d,%d) channel %d = %.6f, want %.6f (diff %.6f)",
					golden.name, s, coord[0], coord[1], c, got, want, diff)
			}
		}
	}
	if maxDiff > pixelTolerance {
		t.Fatalf("%s: sampled pixel max diff = %.6f, want <= %.6f", golden.name, maxDiff, pixelTolerance)
	}
}

func patchifiedVisionPixel(values []float32, targetW, patch, y, x int32) (float32, float32, float32) {
	gridW := targetW / patch
	gy, gx := y/patch, x/patch
	py, px := y%patch, x%patch
	patchDim := int(patch * patch * 3)
	row := int(gy*gridW + gx)
	col := int((py*patch + px) * 3)
	base := row*patchDim + col
	return values[base], values[base+1], values[base+2]
}

func BenchmarkVisionImagePixelsResizeGolden(b *testing.B) {
	data, err := os.ReadFile("../metal/model/gemma4/testdata/vision_video_frame.png")
	if err != nil {
		b.Fatalf("read golden png: %v", err)
	}
	cfg := &VisionImageFeatureConfig{
		PatchSize: 16, MaxSoftTokens: 70, PoolingKernelSize: 3, RescaleFactor: 1.0 / 255.0, DoResize: true,
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		pixels, h, w, softTokens, err := VisionImagePixels(data, cfg)
		if err != nil {
			b.Fatalf("VisionImagePixels: %v", err)
		}
		if softTokens != 63 || len(pixels) != int(h*w*3) {
			b.Fatalf("VisionImagePixels softTokens=%d pixels=%d shape=%dx%d, want resized frame pixels", softTokens, len(pixels), h, w)
		}
	}
}

func BenchmarkVisionImagePatchesResizeGolden(b *testing.B) {
	data, err := os.ReadFile("../metal/model/gemma4/testdata/vision_video_frame.png")
	if err != nil {
		b.Fatalf("read golden png: %v", err)
	}
	cfg := &VisionImageFeatureConfig{
		PatchSize: 16, MaxSoftTokens: 70, PoolingKernelSize: 3, RescaleFactor: 1.0 / 255.0, DoResize: true,
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		patches, softTokens, err := VisionImagePatches(data, cfg)
		if err != nil {
			b.Fatalf("VisionImagePatches: %v", err)
		}
		if softTokens != 63 || len(patches) == 0 {
			b.Fatalf("VisionImagePatches softTokens=%d patchBytes=%d, want resized frame patches", softTokens, len(patches))
		}
	}
}

func BenchmarkVisionImagePatchesJPEGResizeGolden(b *testing.B) {
	data := jpegVisionGoldenData(b)
	cfg := &VisionImageFeatureConfig{
		PatchSize: 16, MaxSoftTokens: 70, PoolingKernelSize: 3, RescaleFactor: 1.0 / 255.0, DoResize: true,
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		patches, softTokens, err := VisionImagePatches(data, cfg)
		if err != nil {
			b.Fatalf("VisionImagePatches: %v", err)
		}
		if softTokens != 63 || len(patches) == 0 {
			b.Fatalf("VisionImagePatches softTokens=%d patchBytes=%d, want resized JPEG frame patches", softTokens, len(patches))
		}
	}
}

func TestVisionImagePatchesResizeAllocationBudget(t *testing.T) {
	data, err := os.ReadFile("../metal/model/gemma4/testdata/vision_video_frame.png")
	if err != nil {
		t.Fatalf("read golden png: %v", err)
	}
	cfg := &VisionImageFeatureConfig{
		PatchSize: 16, MaxSoftTokens: 70, PoolingKernelSize: 3, RescaleFactor: 1.0 / 255.0, DoResize: true,
	}
	if _, _, err := VisionImagePatches(data, cfg); err != nil {
		t.Fatalf("VisionImagePatches warmup: %v", err)
	}
	var patchErr error
	allocs := testing.AllocsPerRun(1, func() {
		var patches []byte
		var softTokens int
		patches, softTokens, patchErr = VisionImagePatches(data, cfg)
		if patchErr == nil && (softTokens != 63 || len(patches) == 0) {
			patchErr = os.ErrInvalid
		}
	})
	if patchErr != nil {
		t.Fatalf("VisionImagePatches: %v", patchErr)
	}
	if allocs > 100000 {
		t.Fatalf("VisionImagePatches resize allocations = %.0f, want <= 100000", allocs)
	}
}

func TestVisionImagePatchesJPEGResizeAllocationBudget(t *testing.T) {
	data := jpegVisionGoldenData(t)
	cfg := &VisionImageFeatureConfig{
		PatchSize: 16, MaxSoftTokens: 70, PoolingKernelSize: 3, RescaleFactor: 1.0 / 255.0, DoResize: true,
	}
	if _, _, err := VisionImagePatches(data, cfg); err != nil {
		t.Fatalf("VisionImagePatches warmup: %v", err)
	}
	var patchErr error
	allocs := testing.AllocsPerRun(1, func() {
		var patches []byte
		var softTokens int
		patches, softTokens, patchErr = VisionImagePatches(data, cfg)
		if patchErr == nil && (softTokens != 63 || len(patches) == 0) {
			patchErr = os.ErrInvalid
		}
	})
	if patchErr != nil {
		t.Fatalf("VisionImagePatches: %v", patchErr)
	}
	if allocs > 100000 {
		t.Fatalf("VisionImagePatches JPEG resize allocations = %.0f, want <= 100000", allocs)
	}
}

func jpegVisionGoldenData(tb testing.TB) []byte {
	tb.Helper()
	data, err := os.ReadFile("../metal/model/gemma4/testdata/vision_video_frame.png")
	if err != nil {
		tb.Fatalf("read golden png: %v", err)
	}
	img, _, err := image.Decode(bytes.NewReader(data))
	if err != nil {
		tb.Fatalf("decode golden png: %v", err)
	}
	var buf bytes.Buffer
	if err := jpeg.Encode(&buf, img, &jpeg.Options{Quality: 92}); err != nil {
		tb.Fatalf("encode jpeg fixture: %v", err)
	}
	return buf.Bytes()
}

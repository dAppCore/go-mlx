// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma4

import (
	"math"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/mlx/pkg/metal"
)

// The goldens in vision_features_golden_test.go are the actual outputs of
// the HF Gemma4ImageProcessorPil on the PNGs under testdata/ — reference
// parity for the image front-end. Geometry (target size, soft tokens) must
// match exactly; pixels within a small interpolation tolerance (PIL's
// integer-coefficient resampling vs our float64 path).
func TestGemma4_ImageFeatures_GoldenParity_Good(t *testing.T) {
	requireMetalRuntime(t)
	m := &Gemma4Model{MultiModalProjector: &Gemma4MultiModalProjector{}}
	const pixelTolerance = 4.0 / 255.0

	for _, golden := range imageFeatureGoldens {
		read := core.ReadFile("testdata/vision_" + golden.name + ".png")
		if !read.OK {
			t.Fatalf("%s: read testdata png failed", golden.name)
		}
		data, _ := read.Value.([]byte)
		cfg := &Gemma4ImageFeatureConfig{MaxSoftTokens: golden.maxSoftTokens, DoResize: true}
		pixels, softTokens, err := m.Gemma4ImagePixels(data, cfg)
		if err != nil {
			t.Fatalf("%s: Gemma4ImagePixels: %v", golden.name, err)
		}
		shape := pixels.Shape()
		if len(shape) != 3 || shape[0] != golden.targetH || shape[1] != golden.targetW || shape[2] != 3 {
			metal.Free(pixels)
			t.Fatalf("%s: target = %v, want [%d %d 3]", golden.name, shape, golden.targetH, golden.targetW)
		}
		if softTokens != golden.softTokens {
			metal.Free(pixels)
			t.Fatalf("%s: soft tokens = %d, want %d", golden.name, softTokens, golden.softTokens)
		}

		values := pixels.Floats()
		metal.Free(pixels)

		var sumR, sumG, sumB float64
		for i := 0; i < len(values); i += 3 {
			sumR += float64(values[i])
			sumG += float64(values[i+1])
			sumB += float64(values[i+2])
		}
		n := float64(len(values) / 3)
		for c, pair := range [][2]float64{{sumR / n, golden.meanR}, {sumG / n, golden.meanG}, {sumB / n, golden.meanB}} {
			if diff := math.Abs(pair[0] - pair[1]); diff > 1e-3 {
				t.Fatalf("%s: channel %d mean = %v, want %v (Δ %v)", golden.name, c, pair[0], pair[1], diff)
			}
		}

		maxDiff := 0.0
		for s, coord := range golden.sampleCoords {
			base := (int(coord[0])*int(golden.targetW) + int(coord[1])) * 3
			for c := 0; c < 3; c++ {
				diff := math.Abs(float64(values[base+c]) - float64(golden.samplePixels[s*3+c]))
				if diff > maxDiff {
					maxDiff = diff
				}
			}
		}
		t.Logf("%s: %dx%d, %d soft tokens, max sampled |Δ| vs HF = %.5f", golden.name, golden.targetH, golden.targetW, golden.softTokens, maxDiff)
		if maxDiff > pixelTolerance {
			t.Fatalf("%s: sampled pixel max |Δ| = %v exceeds %v", golden.name, maxDiff, pixelTolerance)
		}
	}
}

func TestGemma4_ImageFeatures_AspectMath_Good(t *testing.T) {
	// Pure geometry — mirrors get_aspect_ratio_preserving_size cases.
	cases := []struct{ h, w, max, th, tw int32 }{
		{480, 640, 2520, 672, 912},
		{64, 64, 2520, 768, 768},
		{100, 1200, 2520, 192, 2736},
		{480, 640, 630, 336, 432},
	}
	for _, c := range cases {
		th, tw, err := gemma4AspectPreservingSize(c.h, c.w, 16, c.max, 3)
		if err != nil || th != c.th || tw != c.tw {
			t.Fatalf("size(%dx%d, max %d) = %dx%d err=%v, want %dx%d", c.h, c.w, c.max, th, tw, err, c.th, c.tw)
		}
	}
	if _, _, err := gemma4AspectPreservingSize(0, 100, 16, 2520, 3); err == nil {
		t.Fatal("zero height accepted")
	}
}

func TestGemma4_ImageFeatures_Bad(t *testing.T) {
	requireMetalRuntime(t)
	m := &Gemma4Model{MultiModalProjector: &Gemma4MultiModalProjector{}}
	if _, _, err := m.Gemma4ImagePixels([]byte("not an image"), nil); err == nil {
		t.Fatal("garbage bytes decoded")
	}
	bare := &Gemma4Model{}
	if _, _, err := bare.Gemma4ImagePixels(nil, nil); err == nil {
		t.Fatal("vision-free model accepted an image")
	}
}

func TestGemma4_ImageFeatures_LoadConfigs_Good(t *testing.T) {
	dir := t.TempDir()
	payload := []byte(`{
		"image_processor": {"patch_size": 16, "max_soft_tokens": 280, "pooling_kernel_size": 3, "rescale_factor": 0.00392156862745098, "do_resize": true},
		"video_processor": {"patch_size": 16, "max_soft_tokens": 70, "pooling_kernel_size": 3, "do_resize": true, "num_frames": 32}
	}`)
	if r := core.WriteFile(core.PathJoin(dir, "processor_config.json"), payload, 0o600); !r.OK {
		t.Fatal("write processor_config.json failed")
	}
	imageCfg, videoCfg, err := LoadGemma4ImageFeatureConfigs(dir)
	if err != nil || imageCfg == nil || videoCfg == nil {
		t.Fatalf("load = (%v, %v, %v), want both sections", imageCfg, videoCfg, err)
	}
	if imageCfg.MaxSoftTokens != 280 || videoCfg.MaxSoftTokens != 70 || videoCfg.NumFrames != 32 {
		t.Fatalf("configs = %+v / %+v, want declared budgets", imageCfg, videoCfg)
	}
	none, _, err := LoadGemma4ImageFeatureConfigs(t.TempDir())
	if err != nil || none != nil {
		t.Fatalf("absent processor config gave (%+v, %v), want (nil, nil)", none, err)
	}
}

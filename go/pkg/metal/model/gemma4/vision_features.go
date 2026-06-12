// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma4

import (
	"bytes"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"math"

	core "dappco.re/go"
	"dappco.re/go/mlx/pkg/metal"
)

// The Gemma 4 image/video front-end (the vision twin of the audio mel
// extractor): encoded image bytes → aspect-ratio-preserving resize onto the
// patch budget → rescale to [0,1] → raw [H,W,3] pixels the vision tower's
// prepare() consumes (it normalises to [-1,1] and patchifies itself).
// Ported from the HF Gemma4ImageProcessor: targets are multiples of
// patch×pool (48 px) under max_soft_tokens×pool² patches, so the soft-token
// count is exactly (H/48)·(W/48)·... = patches/pool². The resize is
// PIL-style antialiased bicubic (a = -0.5, support widened by the scale
// factor when downscaling) — what torchvision's antialias path implements.

// Gemma 4 vision prompt tokens (tokenizer_config.json truth). Each image
// expands to BOI + ImageToken×softTokens + EOI; video frames expand to
// "mm:ss " + BOI + VideoToken×softTokens + EOI per frame.
const (
	Gemma4BOIToken   = "<|image>"
	Gemma4ImageToken = "<|image|>"
	Gemma4EOIToken   = "<image|>"
	Gemma4VideoToken = "<|video|>"
)

// Gemma4ImageFeatureConfig mirrors the image_processor / video_processor
// sections of processor_config.json (per-modality soft-token budgets).
type Gemma4ImageFeatureConfig struct {
	PatchSize         int32   `json:"patch_size"`
	MaxSoftTokens     int32   `json:"max_soft_tokens"`
	PoolingKernelSize int32   `json:"pooling_kernel_size"`
	RescaleFactor     float64 `json:"rescale_factor"`
	DoResize          bool    `json:"do_resize"`
	DoConvertRGB      bool    `json:"do_convert_rgb"`
	NumFrames         int32   `json:"num_frames"`
}

type gemma4VisionProcessorConfig struct {
	ImageProcessor *Gemma4ImageFeatureConfig `json:"image_processor"`
	VideoProcessor *Gemma4ImageFeatureConfig `json:"video_processor"`
}

// LoadGemma4ImageFeatureConfigs reads the image and video processor sections
// from the model directory's processor_config.json. Either may be nil when
// the model ships no section. (nil, nil, nil) = no processor config at all.
func LoadGemma4ImageFeatureConfigs(modelPath string) (imageCfg, videoCfg *Gemma4ImageFeatureConfig, err error) {
	read := core.ReadFile(core.PathJoin(modelPath, "processor_config.json"))
	if !read.OK {
		return nil, nil, nil
	}
	data, ok := read.Value.([]byte)
	if !ok {
		return nil, nil, core.E("gemma4.vision", "processor_config.json read returned non-byte data", nil)
	}
	var processor gemma4VisionProcessorConfig
	if r := core.JSONUnmarshal(data, &processor); !r.OK {
		return nil, nil, core.E("gemma4.vision", "parse processor_config.json", nil)
	}
	return normalizeGemma4ImageFeatureConfig(processor.ImageProcessor),
		normalizeGemma4ImageFeatureConfig(processor.VideoProcessor), nil
}

// normalizeGemma4ImageFeatureConfig fills absent fields with the HF
// Gemma4ImageProcessor defaults (published spec, mirroring the audio
// front-end's resolution policy).
func normalizeGemma4ImageFeatureConfig(cfg *Gemma4ImageFeatureConfig) *Gemma4ImageFeatureConfig {
	if cfg == nil {
		return nil
	}
	if cfg.PatchSize <= 0 {
		cfg.PatchSize = 16
	}
	if cfg.MaxSoftTokens <= 0 {
		cfg.MaxSoftTokens = 280
	}
	if cfg.PoolingKernelSize <= 0 {
		cfg.PoolingKernelSize = 3
	}
	if cfg.RescaleFactor <= 0 {
		cfg.RescaleFactor = 1.0 / 255.0
	}
	return cfg
}

// gemma4AspectPreservingSize ports get_aspect_ratio_preserving_size: the
// largest target producing at most maxPatches patches with both sides
// divisible by patch×pool. Mirrors the reference's zero-side edge cases.
func gemma4AspectPreservingSize(height, width, patchSize, maxPatches, pool int32) (int32, int32, error) {
	if height <= 0 || width <= 0 {
		return 0, 0, core.E("gemma4.vision", core.Sprintf("invalid image size %dx%d", height, width), nil)
	}
	targetPx := float64(maxPatches) * float64(patchSize) * float64(patchSize)
	factor := math.Sqrt(targetPx / (float64(height) * float64(width)))
	sideMult := pool * patchSize

	th := int32(math.Floor(factor*float64(height)/float64(sideMult))) * sideMult
	tw := int32(math.Floor(factor*float64(width)/float64(sideMult))) * sideMult

	if th == 0 && tw == 0 {
		return 0, 0, core.E("gemma4.vision", "image degenerates to 0x0 under the patch budget", nil)
	}
	maxSide := (maxPatches / (pool * pool)) * sideMult
	if th == 0 {
		th = sideMult
		tw = min(int32(math.Floor(float64(width)/float64(height)))*sideMult, maxSide)
	} else if tw == 0 {
		tw = sideMult
		th = min(int32(math.Floor(float64(height)/float64(width)))*sideMult, maxSide)
	}
	if int64(th)*int64(tw) > int64(targetPx) {
		return 0, 0, core.E("gemma4.vision", core.Sprintf("target %dx%d exceeds the %d-patch budget", th, tw, maxPatches), nil)
	}
	return th, tw, nil
}

// Gemma4ImagePixels decodes PNG/JPEG bytes and prepares them for the vision
// tower: aspect-preserving resize onto the patch budget, rescale to [0,1],
// returned as a [H, W, 3] float32 array plus the soft-token count the image
// occupies (the caller places that many placeholder tokens).
func (m *Gemma4Model) Gemma4ImagePixels(data []byte, cfg *Gemma4ImageFeatureConfig) (*metal.Array, int, error) {
	if m == nil || (m.VisionTower == nil && m.MultiModalProjector == nil) {
		return nil, 0, core.NewError("gemma4: model has no vision tower")
	}
	cfg = normalizeGemma4ImageFeatureConfig(cfg)
	if cfg == nil {
		return nil, 0, core.NewError("gemma4: image feature config is nil")
	}
	img, _, err := image.Decode(bytes.NewReader(data))
	if err != nil {
		return nil, 0, core.E("gemma4.vision", "decode image", err)
	}
	bounds := img.Bounds()
	h, w := int32(bounds.Dy()), int32(bounds.Dx())

	// uint8 RGB plane in HWC.
	src := make([]float64, int(h)*int(w)*3)
	idx := 0
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			r, g, b, _ := img.At(x, y).RGBA() // 16-bit premultiplied
			src[idx] = float64(r >> 8)
			src[idx+1] = float64(g >> 8)
			src[idx+2] = float64(b >> 8)
			idx += 3
		}
	}

	maxPatches := cfg.MaxSoftTokens * cfg.PoolingKernelSize * cfg.PoolingKernelSize
	th, tw := h, w
	if cfg.DoResize || th%(cfg.PatchSize*cfg.PoolingKernelSize) != 0 || tw%(cfg.PatchSize*cfg.PoolingKernelSize) != 0 {
		th, tw, err = gemma4AspectPreservingSize(h, w, cfg.PatchSize, maxPatches, cfg.PoolingKernelSize)
		if err != nil {
			return nil, 0, err
		}
	}
	resized := src
	if th != h || tw != w {
		resized = gemma4ResizeBicubicAA(src, h, w, th, tw)
	}

	// Round to uint8 like the reference (torchvision resizes uint8 tensors,
	// rounding back before the rescale), then rescale to [0,1].
	pixels := make([]float32, len(resized))
	for i, v := range resized {
		u := math.RoundToEven(v)
		if u < 0 {
			u = 0
		} else if u > 255 {
			u = 255
		}
		pixels[i] = float32(u * cfg.RescaleFactor)
	}
	grid := (th / cfg.PatchSize) * (tw / cfg.PatchSize)
	softTokens := int(grid / (cfg.PoolingKernelSize * cfg.PoolingKernelSize))
	return metal.FromValues(pixels, int(th), int(tw), 3), softTokens, nil
}

// gemma4ResizeBicubicAA is a separable antialiased bicubic resize
// (PIL-style: cubic a = -0.5, filter support widened by the scale factor
// when downscaling — the algorithm behind torchvision's antialias=True).
// src is [h, w, 3] float64 HWC; the result is [th, tw, 3].
func gemma4ResizeBicubicAA(src []float64, h, w, th, tw int32) []float64 {
	// Horizontal pass (w → tw), then vertical (h → th).
	horiz := make([]float64, int(h)*int(tw)*3)
	gemma4ResamplePass(src, horiz, int(w), int(tw), int(h), 3, true)
	out := make([]float64, int(th)*int(tw)*3)
	gemma4ResamplePass(horiz, out, int(h), int(th), int(tw), 3, false)
	return out
}

func gemma4CubicFilter(x float64) float64 {
	// PIL's bicubic kernel, a = -0.5.
	const a = -0.5
	if x < 0 {
		x = -x
	}
	switch {
	case x < 1:
		return ((a+2)*x-(a+3))*x*x + 1
	case x < 2:
		return (((x-5)*x+8)*x - 4) * a
	default:
		return 0
	}
}

// gemma4ResamplePass resamples one axis. horizontal=true treats rows of
// length inLen across `lines` rows; horizontal=false resamples columns
// (lines = row width). channels interleave fastest.
func gemma4ResamplePass(src, dst []float64, inLen, outLen, lines, channels int, horizontal bool) {
	scale := float64(inLen) / float64(outLen)
	filterScale := scale
	if filterScale < 1 {
		filterScale = 1
	}
	support := 2.0 * filterScale // bicubic base support 2

	weights := make([]float64, 0, int(support)*2+3)
	for out := 0; out < outLen; out++ {
		center := (float64(out) + 0.5) * scale
		xmin := int(center - support + 0.5)
		if xmin < 0 {
			xmin = 0
		}
		xmax := int(center + support + 0.5)
		if xmax > inLen {
			xmax = inLen
		}
		weights = weights[:0]
		sum := 0.0
		for x := xmin; x < xmax; x++ {
			wgt := gemma4CubicFilter((float64(x) - center + 0.5) / filterScale)
			weights = append(weights, wgt)
			sum += wgt
		}
		if sum != 0 {
			for i := range weights {
				weights[i] /= sum
			}
		}
		for line := 0; line < lines; line++ {
			for c := 0; c < channels; c++ {
				acc := 0.0
				for k, wgt := range weights {
					var at int
					if horizontal {
						at = (line*inLen + xmin + k) * channels
					} else {
						at = ((xmin+k)*lines + line) * channels
					}
					acc += src[at+c] * wgt
				}
				var to int
				if horizontal {
					to = (line*outLen + out) * channels
				} else {
					to = (out*lines + line) * channels
				}
				dst[to+c] = acc
			}
		}
	}
}

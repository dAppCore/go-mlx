// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"context"
	"flag"
	"io"

	core "dappco.re/go"
	"dappco.re/go/mlx/chat"
	"dappco.re/go/mlx/pkg/metal"
	"dappco.re/go/mlx/pkg/metal/model/gemma4"
	gemma4chat "dappco.re/go/mlx/pkg/metal/model/gemma4/chat"
)

// runVisionCommand answers a prompt about images and/or video frames through
// the Gemma 4 vision lane: PNG/JPEG → aspect-preserving resize onto the
// patch budget → vision tower soft tokens spliced over the prompt's
// placeholders. Video = frames through the same path under the video
// soft-token budget, each prefixed with its mm:ss timestamp (the HF
// processor convention).
func runVisionCommand(ctx context.Context, args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("vision", flag.ContinueOnError)
	fs.SetOutput(stderr)
	imagesFlag := fs.String("images", "", "comma-separated PNG/JPEG image paths")
	framesFlag := fs.String("video-frames", "", "comma-separated PNG/JPEG frame paths (one video)")
	fps := fs.Float64("fps", 1, "frame rate the video frames were sampled at (timestamps)")
	prompt := fs.String("prompt", "Describe what you see.", "question about the images/video")
	maxTokens := fs.Int("max-tokens", 256, "response length bound")
	chatFlag := fs.Bool("chat", true, "format with the model chat template")
	fs.Usage = func() {
		core.WriteString(stderr, "Usage: lthn-mlx vision -images a.png[,b.jpg] [flags] <model-path>\n\n")
		core.WriteString(stderr, "Answer a prompt about images and/or video frames (Gemma 4 vision tower).\n\n")
		core.WriteString(stderr, "Flags:\n")
		fs.PrintDefaults()
		core.WriteString(stderr, "\nExamples:\n")
		core.WriteString(stderr, "    lthn-mlx vision -images photo.png -prompt 'What is this?' <model>\n")
		core.WriteString(stderr, "    lthn-mlx vision -video-frames f1.png,f2.png,f3.png -fps 1 <model>\n")
	}
	if err := fs.Parse(args); err != nil {
		return 2
	}
	imagePaths := splitPathList(*imagesFlag)
	framePaths := splitPathList(*framesFlag)
	if fs.NArg() != 1 || (len(imagePaths) == 0 && len(framePaths) == 0) {
		fs.Usage()
		return 2
	}

	m, err := gemma4.LoadGemma4(fs.Arg(0))
	if err != nil {
		core.Print(stderr, "%s vision: load: %v", cliName(), err)
		return 1
	}
	defer m.CloseModel()
	if m.VisionTower == nil && m.MultiModalProjector == nil {
		core.Print(stderr, "%s vision: this checkpoint has no vision tower", cliName())
		return 1
	}
	if m.Cfg == nil || m.Cfg.ImageTokenID == 0 {
		core.Print(stderr, "%s vision: model config declares no image_token_id", cliName())
		return 1
	}
	imageCfg, videoCfg, err := gemma4.LoadGemma4ImageFeatureConfigs(metal.ResolveModelRoot(fs.Arg(0)))
	if err != nil {
		core.Print(stderr, "%s vision: %v", cliName(), err)
		return 1
	}

	loadPixels := func(path string, cfg *gemma4.Gemma4ImageFeatureConfig) (*metal.Array, int, error) {
		read := core.ReadFile(path)
		if !read.OK {
			return nil, 0, core.E("mlx.vision", core.Sprintf("read %s", path), nil)
		}
		data, ok := read.Value.([]byte)
		if !ok {
			return nil, 0, core.E("mlx.vision", core.Sprintf("read %s returned non-byte data", path), nil)
		}
		return m.Gemma4ImagePixels(data, cfg)
	}

	content := ""
	var imagePixels, videoFrames []*metal.Array
	defer func() {
		metal.Free(imagePixels...)
		metal.Free(videoFrames...)
	}()
	wantImageTokens := 0
	for _, path := range imagePaths {
		pixels, softTokens, loadErr := loadPixels(path, imageCfg)
		if loadErr != nil {
			core.Print(stderr, "%s vision: %s: %v", cliName(), path, loadErr)
			return 1
		}
		imagePixels = append(imagePixels, pixels)
		wantImageTokens += softTokens
		content += gemma4.Gemma4BOIToken
		for range softTokens {
			content += gemma4.Gemma4ImageToken
		}
		content += gemma4.Gemma4EOIToken + "\n"
	}
	wantVideoTokens := 0
	for i, path := range framePaths {
		pixels, softTokens, loadErr := loadPixels(path, videoCfg)
		if loadErr != nil {
			core.Print(stderr, "%s vision: %s: %v", cliName(), path, loadErr)
			return 1
		}
		videoFrames = append(videoFrames, pixels)
		wantVideoTokens += softTokens
		seconds := 0
		if *fps > 0 {
			seconds = int(float64(i) / *fps)
		}
		content += core.Sprintf("%02d:%02d ", seconds/60, seconds%60)
		content += gemma4.Gemma4BOIToken
		for range softTokens {
			content += gemma4.Gemma4VideoToken
		}
		content += gemma4.Gemma4EOIToken + " "
	}
	if len(framePaths) > 0 {
		content += "\n"
	}
	content += *prompt

	formatted := content
	if *chatFlag {
		formatted = gemma4chat.Format([]chat.Message{{Role: "user", Content: content}}, chat.Config{})
	}
	ids := m.Tok.Encode(formatted)
	if got := countTokenID(ids, m.Cfg.ImageTokenID); got != wantImageTokens {
		core.Print(stderr, "%s vision: tokenizer produced %d image placeholders, want %d", cliName(), got, wantImageTokens)
		return 1
	}
	if m.Cfg.VideoTokenID != 0 {
		if got := countTokenID(ids, m.Cfg.VideoTokenID); got != wantVideoTokens {
			core.Print(stderr, "%s vision: tokenizer produced %d video placeholders, want %d", cliName(), got, wantVideoTokens)
			return 1
		}
	} else if wantVideoTokens > 0 {
		core.Print(stderr, "%s vision: model config declares no video_token_id", cliName())
		return 1
	}

	res, err := multimodalGreedyDecode(ctx, m, ids, imagePixels, nil, videoFrames, *maxTokens)
	if err != nil {
		core.Print(stderr, "%s vision: %v", cliName(), err)
		return 1
	}

	core.WriteString(stdout, m.Tok.Decode(res.Generated))
	core.WriteString(stdout, "\n\n")
	rate := 0.0
	if res.DecodeDur > 0 {
		rate = float64(len(res.Generated)) / res.DecodeDur.Seconds()
	}
	core.WriteString(stdout, core.Sprintf(
		"vision %d image(s) %d frame(s) · %d soft tokens · prefill %dms · %d generated · %.1f tok/s\n",
		len(imagePixels), len(videoFrames), wantImageTokens+wantVideoTokens,
		res.PrefillDur.Milliseconds(), len(res.Generated), rate))
	return 0
}

func splitPathList(list string) []string {
	if core.Trim(list) == "" {
		return nil
	}
	parts := core.Split(list, ",")
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		if trimmed := core.Trim(p); trimmed != "" {
			out = append(out, trimmed)
		}
	}
	return out
}

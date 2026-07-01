// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"context"
	"flag"
	"io"
	"time"

	core "dappco.re/go"
	"dappco.re/go/mlx/chat"
	"dappco.re/go/mlx/pkg/model"
	"dappco.re/go/mlx/pkg/model/gemma4"
	gemma4chat "dappco.re/go/mlx/pkg/model/gemma4/chat"
	"dappco.re/go/mlx/pkg/native"
	"dappco.re/go/mlx/pkg/tokenizer"
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

	return runNativeVisionCommand(ctx, fs.Arg(0), imagePaths, framePaths, *fps, *prompt, *maxTokens, *chatFlag, stdout, stderr)
}

type nativeVisionCommandModel interface {
	model.SessionModel
	AcceptsImageInput() bool
	ImagePlaceholderTokenID() int32
	ImagePlaceholderBlock(int) string
	VideoPlaceholderTokenID() int32
	VideoPlaceholderBlock(int) string
	ProjectImageFeatures([]byte) ([]byte, error)
}

type nativeVisionCommandSession interface {
	PrefillTokenEmbeddings([]int32, [][]byte) error
	GenerateFromCacheEach(int, int, func(int32) bool) ([]int32, error)
}

func runNativeVisionCommand(ctx context.Context, modelPath string, imagePaths, framePaths []string, fps float64, prompt string, maxTokens int, chatFlag bool, stdout, stderr io.Writer) int {
	if maxTokens <= 0 {
		core.Print(stderr, "%s vision: max-tokens must be > 0", cliName())
		return 1
	}

	tm, err := native.LoadTokenModelDir(modelPath, maxTokens+4096)
	if err != nil {
		core.Print(stderr, "%s vision: load: %v", cliName(), err)
		return 1
	}
	if c, ok := tm.(interface{ Close() error }); ok {
		defer func() { _ = c.Close() }()
	}
	m, ok := tm.(nativeVisionCommandModel)
	if !ok || !m.AcceptsImageInput() {
		core.Print(stderr, "%s vision: this checkpoint has no vision tower", cliName())
		return 1
	}
	if m.ImagePlaceholderTokenID() == 0 {
		core.Print(stderr, "%s vision: model config declares no image_token_id", cliName())
		return 1
	}

	imageCfg, videoCfg, err := gemma4.LoadGemma4ImageFeatureConfigs(modelPath)
	if err != nil {
		core.Print(stderr, "%s vision: %v", cliName(), err)
		return 1
	}

	tok, err := tokenizer.LoadTokenizer(core.PathJoin(modelPath, "tokenizer.json"))
	if err != nil {
		core.Print(stderr, "%s vision: tokenizer: %v", cliName(), err)
		return 1
	}

	content := ""
	var imageFeatures, videoFeatures []byte
	wantImageTokens := 0
	for _, path := range imagePaths {
		projected, softTokens, loadErr := nativeVisionProjectedFeatures(m, path, imageCfg)
		if loadErr != nil {
			core.Print(stderr, "%s vision: %s: %v", cliName(), path, loadErr)
			return 1
		}
		imageFeatures = append(imageFeatures, projected...)
		wantImageTokens += softTokens
		block := m.ImagePlaceholderBlock(softTokens)
		if block == "" {
			core.Print(stderr, "%s vision: model config declares no image placeholder tokens", cliName())
			return 1
		}
		content += block + "\n"
	}
	wantVideoTokens := 0
	for i, path := range framePaths {
		if m.VideoPlaceholderTokenID() == 0 {
			core.Print(stderr, "%s vision: model config declares no video_token_id", cliName())
			return 1
		}
		projected, softTokens, loadErr := nativeVisionProjectedFeatures(m, path, videoCfg)
		if loadErr != nil {
			core.Print(stderr, "%s vision: %s: %v", cliName(), path, loadErr)
			return 1
		}
		videoFeatures = append(videoFeatures, projected...)
		wantVideoTokens += softTokens
		seconds := 0
		if fps > 0 {
			seconds = int(float64(i) / fps)
		}
		content += core.Sprintf("%02d:%02d ", seconds/60, seconds%60)
		block := m.VideoPlaceholderBlock(softTokens)
		if block == "" {
			core.Print(stderr, "%s vision: model config declares no video placeholder tokens", cliName())
			return 1
		}
		content += block + " "
	}
	if len(framePaths) > 0 {
		content += "\n"
	}
	content += prompt

	formatted := content
	if chatFlag {
		formatted = gemma4chat.Format([]chat.Message{{Role: "user", Content: content}}, chat.Config{})
	}
	ids := tok.Encode(formatted)
	if got := countTokenID(ids, m.ImagePlaceholderTokenID()); got != wantImageTokens {
		core.Print(stderr, "%s vision: tokenizer produced %d image placeholders, want %d", cliName(), got, wantImageTokens)
		return 1
	}
	if wantVideoTokens > 0 {
		if got := countTokenID(ids, m.VideoPlaceholderTokenID()); got != wantVideoTokens {
			core.Print(stderr, "%s vision: tokenizer produced %d video placeholders, want %d", cliName(), got, wantVideoTokens)
			return 1
		}
	}

	res, err := nativeVisionGreedyDecode(ctx, m, tok, ids, imageFeatures, videoFeatures, maxTokens)
	if err != nil {
		core.Print(stderr, "%s vision: %v", cliName(), err)
		return 1
	}

	core.WriteString(stdout, tok.Decode(res.Generated))
	core.WriteString(stdout, "\n\n")
	rate := 0.0
	if res.DecodeDur > 0 {
		rate = float64(len(res.Generated)) / res.DecodeDur.Seconds()
	}
	core.WriteString(stdout, core.Sprintf(
		"vision %d image(s) %d frame(s) · %d soft tokens · prefill %dms · %d generated · %.1f tok/s\n",
		len(imagePaths), len(framePaths), wantImageTokens+wantVideoTokens,
		res.PrefillDur.Milliseconds(), len(res.Generated), rate))
	return 0
}

func nativeVisionProjectedFeatures(m nativeVisionCommandModel, path string, cfg *gemma4.Gemma4ImageFeatureConfig) ([]byte, int, error) {
	read := core.ReadFile(path)
	if !read.OK {
		return nil, 0, core.E("mlx.vision", core.Sprintf("read %s", path), nil)
	}
	data, ok := read.Value.([]byte)
	if !ok {
		return nil, 0, core.E("mlx.vision", core.Sprintf("read %s returned non-byte data", path), nil)
	}
	patches, softTokens, err := native.VisionImagePatches(data, nativeVisionFeatureConfig(cfg))
	if err != nil {
		return nil, 0, err
	}
	if softTokens <= 0 {
		return nil, 0, core.NewError("native vision features produced no soft tokens")
	}
	projected, err := m.ProjectImageFeatures(patches)
	if err != nil {
		return nil, 0, core.E("mlx.vision", "project", err)
	}
	return projected, softTokens, nil
}

func nativeVisionFeatureConfig(cfg *gemma4.Gemma4ImageFeatureConfig) *native.VisionImageFeatureConfig {
	if cfg == nil {
		return nil
	}
	return &native.VisionImageFeatureConfig{
		PatchSize:         cfg.PatchSize,
		MaxSoftTokens:     cfg.MaxSoftTokens,
		PoolingKernelSize: cfg.PoolingKernelSize,
		RescaleFactor:     cfg.RescaleFactor,
		DoResize:          cfg.DoResize,
		DoConvertRGB:      cfg.DoConvertRGB,
	}
}

func nativeVisionGreedyDecode(ctx context.Context, m nativeVisionCommandModel, tok *tokenizer.Tokenizer, ids []int32, imageFeatures, videoFeatures []byte, maxTokens int) (multimodalDecodeResult, error) {
	var res multimodalDecodeResult

	start := time.Now()
	embeddings, err := nativeVisionPromptEmbeddings(m, ids, imageFeatures, videoFeatures)
	if err != nil {
		return res, err
	}
	stepper, err := m.OpenSession()
	if err != nil {
		return res, err
	}
	if c, ok := stepper.(interface{ Close() error }); ok {
		defer func() { _ = c.Close() }()
	}
	sess, ok := stepper.(nativeVisionCommandSession)
	if !ok {
		return res, core.NewError("native vision session does not support multimodal prefill")
	}
	if err := sess.PrefillTokenEmbeddings(ids, embeddings); err != nil {
		return res, err
	}
	res.PrefillDur = time.Since(start)

	stopIDs := nativeCommandStopIDs(tok)
	eos := -1
	if tok.HasEOSToken() {
		eos = int(tok.EOSToken())
	}
	emit := func(id int32) bool {
		if ctx != nil {
			if err := ctx.Err(); err != nil {
				return false
			}
		}
		_, stop := stopIDs[id]
		return !stop
	}

	decodeStart := time.Now()
	res.Generated, err = sess.GenerateFromCacheEach(maxTokens, eos, emit)
	res.DecodeDur = time.Since(decodeStart)
	if err != nil {
		return res, err
	}
	if ctx != nil {
		if err := ctx.Err(); err != nil {
			return res, err
		}
	}
	if len(res.Generated) > 0 {
		if _, stop := stopIDs[res.Generated[len(res.Generated)-1]]; stop {
			res.Generated = res.Generated[:len(res.Generated)-1]
		}
	}
	return res, nil
}

func nativeVisionPromptEmbeddings(m nativeVisionCommandModel, ids []int32, imageFeatures, videoFeatures []byte) ([][]byte, error) {
	imageID := m.ImagePlaceholderTokenID()
	videoID := m.VideoPlaceholderTokenID()
	embeddings := make([][]byte, len(ids))
	imageOff, videoOff := 0, 0
	imageUsed, videoUsed := 0, 0
	for i, id := range ids {
		emb, err := m.Embed(id)
		if err != nil {
			return nil, err
		}
		if len(emb) == 0 {
			return nil, core.NewError("native vision prompt embedding is empty")
		}
		switch {
		case id == imageID:
			if imageOff+len(emb) > len(imageFeatures) {
				return nil, core.NewError("native image feature rows do not match placeholder embeddings")
			}
			embeddings[i] = append([]byte(nil), imageFeatures[imageOff:imageOff+len(emb)]...)
			imageOff += len(emb)
			imageUsed++
		case videoID != 0 && id == videoID:
			if videoOff+len(emb) > len(videoFeatures) {
				return nil, core.NewError("native video feature rows do not match placeholder embeddings")
			}
			embeddings[i] = append([]byte(nil), videoFeatures[videoOff:videoOff+len(emb)]...)
			videoOff += len(emb)
			videoUsed++
		default:
			embeddings[i] = append([]byte(nil), emb...)
		}
	}
	if len(imageFeatures) > 0 && imageUsed == 0 {
		return nil, core.NewError("native vision prompt has no image placeholders")
	}
	if len(videoFeatures) > 0 && videoUsed == 0 {
		return nil, core.NewError("native vision prompt has no video placeholders")
	}
	if imageOff != len(imageFeatures) {
		return nil, core.NewError("native vision prompt has unused image feature rows")
	}
	if videoOff != len(videoFeatures) {
		return nil, core.NewError("native vision prompt has unused video feature rows")
	}
	return embeddings, nil
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

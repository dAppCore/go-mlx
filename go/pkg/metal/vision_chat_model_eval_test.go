// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"bytes"
	"context"
	"image"
	"image/color"
	"image/png"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/mlx/internal/metaltest"
)

// TestVisionChat_ImageInput_Eval drives the image-input lane on a real
// multimodal model (gemma4 -it takes images): the SigLIP tower, image-token
// splicing, and multimodal prefill in vision_chat.go are entirely dark in the
// text-only suite. A synthetic PNG goes through ChatMessage.Images. Run:
//
//	GO_MLX_BENCH_MODEL=google/gemma-4-e2b-it go test \
//	  -tags 'metal_runtime model_eval' -run TestVisionChat_ImageInput_Eval ./pkg/metal/
func TestVisionChat_ImageInput_Eval(t *testing.T) {
	if !metaltest.RunModelEvalTests {
		t.Skip("model-eval test; build with -tags model_eval")
	}
	isolateRuntimeGates(t)
	restore := DefaultEngineFeatures().Apply()
	defer restore()

	repo := core.Getenv("GO_MLX_BENCH_MODEL")
	if repo == "" {
		repo = "google/gemma-4-e2b-it"
	}
	dir := metaltest.HFModelPath(t, repo)
	model, err := LoadAndInit(dir, LoadConfig{ContextLen: 4096})
	if err != nil {
		t.Fatalf("LoadAndInit: %v", err)
	}
	defer model.Close()
	ctx := context.Background()

	// A small synthetic gradient image — the processor resizes it to the vision
	// tower's expected input, so exact dimensions don't matter.
	img := image.NewRGBA(image.Rect(0, 0, 64, 64))
	for y := 0; y < 64; y++ {
		for x := 0; x < 64; x++ {
			img.Set(x, y, color.RGBA{R: uint8(x * 4), G: uint8(y * 4), B: 128, A: 255})
		}
	}
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		t.Fatalf("encode png: %v", err)
	}

	msgs := []ChatMessage{{
		Role:    "user",
		Content: "Describe this image briefly.",
		Images:  [][]byte{buf.Bytes()},
	}}
	n := 0
	for range model.Chat(ctx, msgs, GenerateConfig{MaxTokens: 16}) {
		n++
	}
	if err := model.Err(); err != nil {
		t.Fatalf("vision Chat: %v", err)
	}
	if n == 0 {
		t.Fatal("vision chat produced no tokens")
	}
}

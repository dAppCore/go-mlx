// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && metal_runtime

// Runnable usage-in-situ examples for the Gemma 4 vision lane, built on the
// synthetic patch-conv fixture (gemma4VisionConfigJSON / gemma4VisionTinyWeights,
// shared with vision_forward_test.go's loadGemma4VisionTestModel) — no real
// multi-GB checkpoint, no live model load from disk beyond the tiny synthetic
// safetensors this file writes to a scratch temp dir. They are tagged
// metal_runtime for the same reason audio_example_test.go's are: an Example
// cannot skip (it has no *testing.T), so it must compile out of an untagged
// `go test` rather than run Metal ops where the runtime may be unavailable;
// under `-tags metal_runtime` they run and document the contract.
//
// Each Example prints structural facts (shapes / grid) rather than floats —
// the maths is deterministic, but exact float formatting is not a stable
// golden (same rule audio_example_test.go documents).
//
// Previously every function here called LoadGemma4("/models/gemma4"), a path
// guaranteed absent in any test environment, and none carried an "// Output:"
// comment — per Go's testing semantics that means they were compiled but NEVER
// EXECUTED by `go test`, silently proving nothing. gemma4VisionExampleModel
// below builds the same synthetic checkpoint vision_forward_test.go's
// non-Example tests already exercise, so these now really run.
package gemma4

import (
	"os"

	core "dappco.re/go"
	coreio "dappco.re/go/io"
	"dappco.re/go/mlx/pkg/metal"
)

// gemma4VisionExampleModel builds and loads the same tiny synthetic multimodal
// checkpoint vision_forward_test.go's loadGemma4VisionTestModel uses (patch-conv
// vision tower + dense 2-layer text trunk, vocab 10, image token 5), without a
// *testing.T — Example functions carry none. The caller must invoke the
// returned cleanup func (always non-nil) exactly once.
func gemma4VisionExampleModel() (model *Gemma4Model, cleanup func(), err error) {
	dir, mkErr := os.MkdirTemp("", "gemma4-vision-example")
	if mkErr != nil {
		return nil, func() {}, mkErr
	}
	cleanup = func() { os.RemoveAll(dir) }

	if err := coreio.Local.Write(core.JoinPath(dir, "config.json"), gemma4VisionConfigJSON); err != nil {
		cleanup()
		return nil, func() {}, err
	}
	tokenizer := `{
		"model": {
			"type": "BPE",
			"vocab": {"<pad>": 0, "<eos>": 1, "<bos>": 2, "hello": 3, "world": 4},
			"merges": []
		},
		"added_tokens": [
			{"id": 0, "content": "<pad>", "special": true},
			{"id": 1, "content": "<eos>", "special": true},
			{"id": 2, "content": "<bos>", "special": true}
		]
	}`
	if err := coreio.Local.Write(core.JoinPath(dir, "tokenizer.json"), tokenizer); err != nil {
		cleanup()
		return nil, func() {}, err
	}
	if err := metal.SaveSafetensors(core.JoinPath(dir, "model.safetensors"), gemma4VisionTinyWeights()); err != nil {
		cleanup()
		return nil, func() {}, err
	}

	model, err = LoadGemma4(dir)
	if err != nil {
		cleanup()
		return nil, func() {}, err
	}
	return model, cleanup, nil
}

// ExampleGemma4Model_ForwardMultiModal shows the multimodal-prefill contract: a
// token sequence carrying 4 image-placeholder slots (matching the synthetic
// tower's 4 pooled soft tokens) plus the raw pixels comes back as logits shaped
// [batch, seqLen, vocab].
func ExampleGemma4Model_ForwardMultiModal() {
	model, cleanup, err := gemma4VisionExampleModel()
	if err != nil {
		core.Println(core.Sprintf("model unavailable: %v", err))
		return
	}
	defer cleanup()
	defer closeGemma4(model)

	tokenIDs := []int32{2, gemma4VisionImageTok, gemma4VisionImageTok, gemma4VisionImageTok, gemma4VisionImageTok, 4}
	tokens := metal.FromValues(tokenIDs, 1, len(tokenIDs))
	image := visionTestPixels()
	caches := model.NewCache()
	logits := model.ForwardMultiModal(tokens, []*metal.Array{image}, caches)
	defer func() {
		metal.Free(tokens, image, logits)
		metal.FreeCaches(caches)
	}()
	if err := metal.Eval(logits); err != nil {
		core.Println(core.Sprintf("eval error: %v", err))
		return
	}
	shape := logits.Shape()
	core.Println(core.Sprintf("logits %dx%dx%d", shape[0], shape[1], shape[2]))
	// Output: logits 1x6x10
}

// ExampleGemma4VisionModel_Forward shows the whole tower contract: patch embed
// -> encoder stack -> post-layernorm -> pooler, emerging as [nSoftTokens,
// hidden]. Pooling kernel 2 over the 4x4 patch grid halves both axes -> 4
// pooled rows.
func ExampleGemma4VisionModel_Forward() {
	model, cleanup, err := gemma4VisionExampleModel()
	if err != nil || model.VisionTower == nil {
		core.Println(core.Sprintf("vision tower unavailable: %v", err))
		return
	}
	defer cleanup()
	defer closeGemma4(model)

	pixels := visionTestPixels()
	features := model.VisionTower.Forward(pixels)
	defer metal.Free(pixels, features)
	if err := metal.Eval(features); err != nil {
		core.Println(core.Sprintf("eval error: %v", err))
		return
	}
	shape := features.Shape()
	core.Println(core.Sprintf("vision features %dx%d", shape[0], shape[1]))
	// Output: vision features 4x16
}

// ExampleGemma4VisionPatchEmbedder_Forward shows the patch-conv contract: a
// [1, 32, 32, 3] NHWC image with patch size 8 makes a 4x4 patch grid, emerging
// as [1, 16, hidden] patch rows.
func ExampleGemma4VisionPatchEmbedder_Forward() {
	model, cleanup, err := gemma4VisionExampleModel()
	if err != nil || model.VisionTower == nil || model.VisionTower.PatchEmbedder == nil {
		core.Println(core.Sprintf("patch embedder unavailable: %v", err))
		return
	}
	defer cleanup()
	defer closeGemma4(model)

	pixels := visionTestPixels()
	patches, gridH, gridW := model.VisionTower.PatchEmbedder.Forward(pixels)
	defer metal.Free(pixels, patches)
	if err := metal.Eval(patches); err != nil {
		core.Println(core.Sprintf("eval error: %v", err))
		return
	}
	shape := patches.Shape()
	core.Println(core.Sprintf("patches %dx%dx%d grid %dx%d", shape[0], shape[1], shape[2], gridH, gridW))
	// Output: patches 1x16x16 grid 4x4
}

// ExampleGemma4VisionEncoder_Forward shows the encoder-stack contract: the
// patch rows flow through every encoder layer's residual attention+MLP blocks,
// which preserve the [batch, patches, hidden] shape throughout.
func ExampleGemma4VisionEncoder_Forward() {
	model, cleanup, err := gemma4VisionExampleModel()
	if err != nil || model.VisionTower == nil || model.VisionTower.PatchEmbedder == nil || model.VisionTower.Encoder == nil {
		core.Println(core.Sprintf("encoder unavailable: %v", err))
		return
	}
	defer cleanup()
	defer closeGemma4(model)

	pixels := visionTestPixels()
	patches, gridH, gridW := model.VisionTower.PatchEmbedder.Forward(pixels)
	encoded := model.VisionTower.Encoder.Forward(patches, gridH, gridW)
	defer metal.Free(pixels, patches, encoded)
	if err := metal.Eval(encoded); err != nil {
		core.Println(core.Sprintf("eval error: %v", err))
		return
	}
	shape := encoded.Shape()
	core.Println(core.Sprintf("encoded %dx%dx%d", shape[0], shape[1], shape[2]))
	// Output: encoded 1x16x16
}

// ExampleGemma4VisionEncoderLayer_Forward shows a single encoder layer's
// contract in isolation: input-norm -> attention -> residual -> pre-FF-norm ->
// MLP -> residual, shape-preserving end to end.
func ExampleGemma4VisionEncoderLayer_Forward() {
	model, cleanup, err := gemma4VisionExampleModel()
	if err != nil || model.VisionTower == nil || model.VisionTower.PatchEmbedder == nil || model.VisionTower.Encoder == nil || len(model.VisionTower.Encoder.Layers) == 0 {
		core.Println(core.Sprintf("encoder layer unavailable: %v", err))
		return
	}
	defer cleanup()
	defer closeGemma4(model)

	pixels := visionTestPixels()
	patches, gridH, gridW := model.VisionTower.PatchEmbedder.Forward(pixels)
	next := model.VisionTower.Encoder.Layers[0].Forward(patches, gridH, gridW, model.VisionTower.Cfg)
	defer metal.Free(pixels, patches, next)
	if err := metal.Eval(next); err != nil {
		core.Println(core.Sprintf("eval error: %v", err))
		return
	}
	shape := next.Shape()
	core.Println(core.Sprintf("layer output %dx%dx%d", shape[0], shape[1], shape[2]))
	// Output: layer output 1x16x16
}

// ExampleGemma4VisionAttention_Forward shows one encoder layer's attention
// block directly: Q/K/V projections + 2-D RoPE + SDPA + output projection,
// preserving the [batch, patches, hidden] shape.
func ExampleGemma4VisionAttention_Forward() {
	model, cleanup, err := gemma4VisionExampleModel()
	if err != nil || model.VisionTower == nil || model.VisionTower.PatchEmbedder == nil || model.VisionTower.Encoder == nil || len(model.VisionTower.Encoder.Layers) == 0 {
		core.Println(core.Sprintf("attention unavailable: %v", err))
		return
	}
	defer cleanup()
	defer closeGemma4(model)

	layer := model.VisionTower.Encoder.Layers[0]
	if layer == nil || layer.Attention == nil {
		core.Println("attention unavailable: nil layer")
		return
	}
	pixels := visionTestPixels()
	patches, gridH, gridW := model.VisionTower.PatchEmbedder.Forward(pixels)
	out := layer.Attention.Forward(patches, gridH, gridW, model.VisionTower.Cfg)
	defer metal.Free(pixels, patches, out)
	if err := metal.Eval(out); err != nil {
		core.Println(core.Sprintf("eval error: %v", err))
		return
	}
	shape := out.Shape()
	core.Println(core.Sprintf("attention output %dx%dx%d", shape[0], shape[1], shape[2]))
	// Output: attention output 1x16x16
}

// ExampleGemma4VisionMLP_Forward shows one encoder layer's gated MLP directly:
// gate/up projections to the intermediate width, GELU-gated multiply, then the
// down projection back to hidden width.
func ExampleGemma4VisionMLP_Forward() {
	model, cleanup, err := gemma4VisionExampleModel()
	if err != nil || model.VisionTower == nil || model.VisionTower.PatchEmbedder == nil || model.VisionTower.Encoder == nil || len(model.VisionTower.Encoder.Layers) == 0 {
		core.Println(core.Sprintf("mlp unavailable: %v", err))
		return
	}
	defer cleanup()
	defer closeGemma4(model)

	layer := model.VisionTower.Encoder.Layers[0]
	if layer == nil || layer.MLP == nil {
		core.Println("mlp unavailable: nil layer")
		return
	}
	pixels := visionTestPixels()
	patches, _, _ := model.VisionTower.PatchEmbedder.Forward(pixels)
	out := layer.MLP.Forward(patches)
	defer metal.Free(pixels, patches, out)
	if err := metal.Eval(out); err != nil {
		core.Println(core.Sprintf("eval error: %v", err))
		return
	}
	shape := out.Shape()
	core.Println(core.Sprintf("mlp output %dx%dx%d", shape[0], shape[1], shape[2]))
	// Output: mlp output 1x16x16
}

// ExampleGemma4VisionPooler_Forward shows the spatial-pooling contract
// directly: a [batch, patches, hidden] tensor over a known grid pools
// kernel x kernel patch blocks into one row each, flattening batch and grid
// into the row axis.
func ExampleGemma4VisionPooler_Forward() {
	model, cleanup, err := gemma4VisionExampleModel()
	if err != nil || model.VisionTower == nil || model.VisionTower.PatchEmbedder == nil || model.VisionTower.Pooler == nil {
		core.Println(core.Sprintf("pooler unavailable: %v", err))
		return
	}
	defer cleanup()
	defer closeGemma4(model)

	pixels := visionTestPixels()
	patches, gridH, gridW := model.VisionTower.PatchEmbedder.Forward(pixels)
	pooled := model.VisionTower.Pooler.Forward(patches, gridH, gridW)
	defer metal.Free(pixels, patches, pooled)
	if err := metal.Eval(pooled); err != nil {
		core.Println(core.Sprintf("eval error: %v", err))
		return
	}
	shape := pooled.Shape()
	core.Println(core.Sprintf("pooled %dx%d", shape[0], shape[1]))
	// Output: pooled 4x16
}

// ExampleGemma4MultiModalProjector_Forward shows the engine-facing projection
// seam: tower forward + multimodal projector, mapping the pooled vision hidden
// width down to the text decoder's hidden width.
func ExampleGemma4MultiModalProjector_Forward() {
	model, cleanup, err := gemma4VisionExampleModel()
	if err != nil || model.VisionTower == nil || model.MultiModalProjector == nil {
		core.Println(core.Sprintf("projector unavailable: %v", err))
		return
	}
	defer cleanup()
	defer closeGemma4(model)

	pixels := visionTestPixels()
	features := model.VisionTower.Forward(pixels)
	projected := model.MultiModalProjector.Forward(features)
	defer metal.Free(pixels, features, projected)
	if err := metal.Eval(projected); err != nil {
		core.Println(core.Sprintf("eval error: %v", err))
		return
	}
	shape := projected.Shape()
	core.Println(core.Sprintf("projected %dx%d", shape[0], shape[1]))
	// Output: projected 4x8
}

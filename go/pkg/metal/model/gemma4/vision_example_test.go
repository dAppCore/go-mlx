// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma4

import "dappco.re/go/mlx/pkg/metal"

func ExampleGemma4Model_ForwardMultiModal() {
	model, err := LoadGemma4("/models/gemma4")
	if err != nil {
		return
	}
	defer closeGemma4(model)

	tokens := metal.FromValues([]int32{model.Cfg.ImageTokenID}, 1, 1)
	image := metal.Zeros([]int32{1, 896, 896, 3}, metal.DTypeFloat32)
	caches := model.NewCache()
	logits := model.ForwardMultiModal(tokens, []*metal.Array{image}, caches)
	metal.Free(tokens, image, logits)
	metal.FreeCaches(caches)
}

func ExampleGemma4VisionModel_Forward() {
	model, err := LoadGemma4("/models/gemma4")
	if err != nil || model.VisionTower == nil {
		return
	}
	defer closeGemma4(model)

	pixels := metal.Zeros([]int32{1, 896, 896, 3}, metal.DTypeFloat32)
	features := model.VisionTower.Forward(pixels)
	metal.Free(pixels, features)
}

func ExampleGemma4VisionPatchEmbedder_Forward() {
	model, err := LoadGemma4("/models/gemma4")
	if err != nil || model.VisionTower == nil || model.VisionTower.PatchEmbedder == nil {
		return
	}
	defer closeGemma4(model)

	pixels := metal.Zeros([]int32{1, 896, 896, 3}, metal.DTypeFloat32)
	patches, gridH, gridW := model.VisionTower.PatchEmbedder.Forward(pixels)
	_, _ = gridH, gridW
	metal.Free(pixels, patches)
}

func ExampleGemma4VisionEncoder_Forward() {
	model, err := LoadGemma4("/models/gemma4")
	if err != nil || model.VisionTower == nil || model.VisionTower.PatchEmbedder == nil || model.VisionTower.Encoder == nil {
		return
	}
	defer closeGemma4(model)

	pixels := metal.Zeros([]int32{1, 896, 896, 3}, metal.DTypeFloat32)
	patches, gridH, gridW := model.VisionTower.PatchEmbedder.Forward(pixels)
	encoded := model.VisionTower.Encoder.Forward(patches, gridH, gridW)
	metal.Free(pixels, patches, encoded)
}

func ExampleGemma4VisionEncoderLayer_Forward() {
	model, err := LoadGemma4("/models/gemma4")
	if err != nil || model.VisionTower == nil || model.VisionTower.PatchEmbedder == nil || model.VisionTower.Encoder == nil || len(model.VisionTower.Encoder.Layers) == 0 {
		return
	}
	defer closeGemma4(model)

	pixels := metal.Zeros([]int32{1, 896, 896, 3}, metal.DTypeFloat32)
	patches, gridH, gridW := model.VisionTower.PatchEmbedder.Forward(pixels)
	next := model.VisionTower.Encoder.Layers[0].Forward(patches, gridH, gridW, model.VisionTower.Cfg)
	metal.Free(pixels, patches, next)
}

func ExampleGemma4VisionAttention_Forward() {
	model, err := LoadGemma4("/models/gemma4")
	if err != nil || model.VisionTower == nil || model.VisionTower.PatchEmbedder == nil || model.VisionTower.Encoder == nil || len(model.VisionTower.Encoder.Layers) == 0 {
		return
	}
	defer closeGemma4(model)

	layer := model.VisionTower.Encoder.Layers[0]
	if layer == nil || layer.Attention == nil {
		return
	}
	pixels := metal.Zeros([]int32{1, 896, 896, 3}, metal.DTypeFloat32)
	patches, gridH, gridW := model.VisionTower.PatchEmbedder.Forward(pixels)
	out := layer.Attention.Forward(patches, gridH, gridW, model.VisionTower.Cfg)
	metal.Free(pixels, patches, out)
}

func ExampleGemma4VisionMLP_Forward() {
	model, err := LoadGemma4("/models/gemma4")
	if err != nil || model.VisionTower == nil || model.VisionTower.PatchEmbedder == nil || model.VisionTower.Encoder == nil || len(model.VisionTower.Encoder.Layers) == 0 {
		return
	}
	defer closeGemma4(model)

	layer := model.VisionTower.Encoder.Layers[0]
	if layer == nil || layer.MLP == nil {
		return
	}
	pixels := metal.Zeros([]int32{1, 896, 896, 3}, metal.DTypeFloat32)
	patches, _, _ := model.VisionTower.PatchEmbedder.Forward(pixels)
	out := layer.MLP.Forward(patches)
	metal.Free(pixels, patches, out)
}

func ExampleGemma4VisionPooler_Forward() {
	model, err := LoadGemma4("/models/gemma4")
	if err != nil || model.VisionTower == nil || model.VisionTower.PatchEmbedder == nil || model.VisionTower.Pooler == nil {
		return
	}
	defer closeGemma4(model)

	pixels := metal.Zeros([]int32{1, 896, 896, 3}, metal.DTypeFloat32)
	patches, gridH, gridW := model.VisionTower.PatchEmbedder.Forward(pixels)
	pooled := model.VisionTower.Pooler.Forward(patches, gridH, gridW)
	metal.Free(pixels, patches, pooled)
}

func ExampleGemma4MultiModalProjector_Forward() {
	model, err := LoadGemma4("/models/gemma4")
	if err != nil || model.VisionTower == nil || model.MultiModalProjector == nil {
		return
	}
	defer closeGemma4(model)

	pixels := metal.Zeros([]int32{1, 896, 896, 3}, metal.DTypeFloat32)
	features := model.VisionTower.Forward(pixels)
	projected := model.MultiModalProjector.Forward(features)
	metal.Free(pixels, features, projected)
}

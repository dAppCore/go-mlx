// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma4

import (
	core "dappco.re/go"
	"dappco.re/go/mlx/pkg/metal"
)

func ExampleLoadGemma4() {
	model, err := LoadGemma4("/models/gemma4")
	if err != nil {
		return
	}
	defer closeGemma4(model)
	core.Println(model.ModelType())
}

func ExampleGemma4Model_Forward() {
	model, err := LoadGemma4("/models/gemma4")
	if err != nil {
		return
	}
	defer closeGemma4(model)

	tokens := metal.FromValues([]int32{2}, 1, 1)
	caches := model.NewCache()
	logits := model.Forward(tokens, caches)
	metal.Free(tokens, logits)
	metal.FreeCaches(caches)
}

func ExampleGemma4Model_ForwardMasked() {
	model, err := LoadGemma4("/models/gemma4")
	if err != nil {
		return
	}
	defer closeGemma4(model)

	tokens := metal.FromValues([]int32{2}, 1, 1)
	mask := metal.Zeros([]int32{1, 1, 1, 1}, metal.DTypeFloat32)
	caches := model.NewCache()
	logits := model.ForwardMasked(tokens, mask, caches)
	metal.Free(tokens, mask, logits)
	metal.FreeCaches(caches)
}

func ExampleGemma4Model_NewCache() {
	model, err := LoadGemma4("/models/gemma4")
	if err != nil {
		return
	}
	defer closeGemma4(model)

	caches := model.NewCache()
	defer metal.FreeCaches(caches)
	core.Println(len(caches) > 0)
}

func ExampleGemma4Model_NumLayers() {
	model := &Gemma4Model{Layers: make([]*Gemma4DecoderLayer, 2)}
	core.Println(model.NumLayers())
	// Output: 2
}

func ExampleGemma4Model_Tokenizer() {
	model, err := LoadGemma4("/models/gemma4")
	if err != nil {
		return
	}
	defer closeGemma4(model)

	tokenizer := model.Tokenizer()
	_ = tokenizer
}

func ExampleGemma4Model_ModelType() {
	model := &Gemma4Model{modelType: "gemma4_text"}
	core.Println(model.ModelType())
	// Output: gemma4_text
}

func ExampleGemma4Model_ApplyLoRA() {
	model := &Gemma4Model{}
	adapter := model.ApplyLoRA(metal.LoRAConfig{})
	core.Println(adapter.Config.TargetKeys, len(adapter.Layers))
	// Output: [q_proj v_proj o_proj] 0
}

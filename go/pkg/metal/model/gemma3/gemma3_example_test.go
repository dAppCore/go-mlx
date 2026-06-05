// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma3

import (
	core "dappco.re/go"
	"dappco.re/go/mlx/pkg/metal"
)

func ExampleLoadGemma3() {
	model, err := LoadGemma3("/path/to/gemma3")
	_, _ = model, err
}

func ExampleGemmaModel_Forward() {
	var (
		model  *GemmaModel
		tokens *metal.Array
		caches []metal.Cache
	)
	if model == nil {
		return
	}
	logits := model.Forward(tokens, caches)
	_ = logits
}

func ExampleGemmaModel_ForwardMasked() {
	var (
		model  *GemmaModel
		tokens *metal.Array
		mask   *metal.Array
		caches []metal.Cache
	)
	if model == nil {
		return
	}
	logits := model.ForwardMasked(tokens, mask, caches)
	_ = logits
}

func ExampleGemmaModel_NewCache() {
	model := &GemmaModel{
		Layers: []*DecoderLayer{
			{IsSliding: true},
			{},
		},
		Cfg: &TextConfig{SlidingWindow: 64},
	}

	caches := model.NewCache()

	core.Println(len(caches), core.Sprintf("%T", caches[0]), core.Sprintf("%T", caches[1]))
	// Output: 2 *metal.RotatingKVCache *metal.KVCache
}

func ExampleGemmaModel_NumLayers() {
	model := &GemmaModel{
		Layers: []*DecoderLayer{
			{},
			{},
			{},
		},
	}

	core.Println(model.NumLayers())
	// Output: 3
}

func ExampleGemmaModel_Tokenizer() {
	var model *GemmaModel
	if model == nil {
		return
	}
	tok := model.Tokenizer()
	_ = tok
}

func ExampleGemmaModel_ModelType() {
	model := &GemmaModel{modelType: "gemma3_text"}

	core.Println(model.ModelType(), (&GemmaModel{}).ModelType())
	// Output: gemma3_text gemma3
}

func ExampleGemmaModel_ApplyLoRA() {
	model := &GemmaModel{}
	adapter := model.ApplyLoRA(metal.LoRAConfig{
		Rank:         2,
		Scale:        4,
		TargetLayers: []string{"gate_proj"},
	})

	core.Println(adapter.Config.TargetKeys, adapter.Config.Rank, adapter.Config.Alpha, adapter.Config.Scale, len(adapter.Layers))
	// Output: [gate_proj] 2 8 4 0
}

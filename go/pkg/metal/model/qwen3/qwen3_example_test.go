// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package qwen3

import (
	core "dappco.re/go"
	"dappco.re/go/mlx/pkg/metal"
)

func ExampleLoadQwen3() {
	model, err := LoadQwen3("/path/to/qwen3")
	_, _ = model, err
}

func ExampleQwen3Model_Forward() {
	var (
		model  *Qwen3Model
		tokens *metal.Array
		caches []metal.Cache
	)
	if model == nil {
		return
	}
	logits := model.Forward(tokens, caches)
	_ = logits
}

func ExampleQwen3Model_ForwardMasked() {
	var (
		model  *Qwen3Model
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

func ExampleQwen3Model_NewCache() {
	model := &Qwen3Model{
		Layers: []*metal.DenseDecoderLayer{
			nil,
			nil,
		},
	}

	caches := model.NewCache()

	core.Println(len(caches), core.Sprintf("%T", caches[0]), core.Sprintf("%T", caches[1]))
	// Output: 2 *metal.KVCache *metal.KVCache
}

func ExampleQwen3Model_NumLayers() {
	model := &Qwen3Model{
		Layers: []*metal.DenseDecoderLayer{
			nil,
			nil,
			nil,
		},
	}

	core.Println(model.NumLayers())
	// Output: 3
}

func ExampleQwen3Model_Tokenizer() {
	var model *Qwen3Model
	if model == nil {
		return
	}
	tok := model.Tokenizer()
	_ = tok
}

func ExampleQwen3Model_ModelType() {
	model := &Qwen3Model{modelType: "qwen3"}

	core.Println(model.ModelType())
	// Output: qwen3
}

func ExampleQwen3Model_ApplyLoRA() {
	model := &Qwen3Model{}
	adapter := model.ApplyLoRA(metal.LoRAConfig{
		Rank:         2,
		Scale:        4,
		TargetLayers: []string{"gate_proj"},
	})

	core.Println(adapter.Config.TargetKeys, adapter.Config.Rank, adapter.Config.Alpha, adapter.Config.Scale, len(adapter.Layers))
	// Output: [gate_proj] 2 8 4 0
}

// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gptoss

import (
	core "dappco.re/go"
	"dappco.re/go/mlx/pkg/metal"
)

// ExampleLoadGptOss loads a GPT-OSS sparse-MoE checkpoint from a model
// directory (config.json + tokenizer.json + *.safetensors). The returned model
// satisfies metal.InternalModel and is also registered under the "gpt_oss"
// model type, so metal.LoadAndInit dispatches here automatically.
func ExampleLoadGptOss() {
	model, err := LoadGptOss("/path/to/gpt-oss")
	_, _ = model, err
}

// ExampleGptOssModel_Forward runs a token batch [B, L] through the decoder
// stack and returns logits [B, L, vocab]. caches must hold one cache per layer
// (see NewCache); attention appends to them across decode steps.
func ExampleGptOssModel_Forward() {
	var (
		model  *GptOssModel
		tokens *metal.Array
		caches []metal.Cache
	)
	if model == nil {
		return
	}
	logits := model.Forward(tokens, caches)
	_ = logits
}

// ExampleGptOssModel_ForwardMasked is the explicit-mask variant of Forward; a
// nil mask is the single-token decode path. Forward delegates to it with a nil
// mask.
func ExampleGptOssModel_ForwardMasked() {
	var (
		model  *GptOssModel
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

// ExampleGptOssModel_NewCache allocates one KV cache per decoder layer — the
// caches slice Forward/ForwardMasked expect.
func ExampleGptOssModel_NewCache() {
	model := &GptOssModel{
		Layers: []*GptOssDecoderLayer{
			nil,
			nil,
		},
	}

	caches := model.NewCache()

	core.Println(len(caches), core.Sprintf("%T", caches[0]), core.Sprintf("%T", caches[1]))
	// Output: 2 *metal.KVCache *metal.KVCache
}

// ExampleGptOssModel_NumLayers reports the decoder-layer count.
func ExampleGptOssModel_NumLayers() {
	model := &GptOssModel{
		Layers: []*GptOssDecoderLayer{
			nil,
			nil,
			nil,
		},
	}

	core.Println(model.NumLayers())
	// Output: 3
}

// ExampleGptOssModel_ModelType returns the canonical architecture token used by
// the loader registry and diagnostics.
func ExampleGptOssModel_ModelType() {
	model := &GptOssModel{modelType: "gpt_oss"}

	core.Println(model.ModelType())
	// Output: gpt_oss
}

// ExampleGptOssModel_MoETextDecodeFamily returns the family token reported in
// "native MoE decode unavailable" diagnostics.
func ExampleGptOssModel_MoETextDecodeFamily() {
	model := &GptOssModel{}

	core.Println(model.MoETextDecodeFamily())
	// Output: gpt_oss
}

// ExampleGptOssModel_ApplyLoRA attaches LoRA adapters to the named projections.
// On an empty model no projections exist, so the adapter is created but holds
// no layers — the config is still normalised (Alpha defaults from Rank).
func ExampleGptOssModel_ApplyLoRA() {
	model := &GptOssModel{}
	adapter := model.ApplyLoRA(metal.LoRAConfig{
		Rank:         2,
		Alpha:        4,
		TargetLayers: []string{"q_proj"},
	})

	core.Println(adapter.Config.Rank, adapter.Config.Alpha, len(adapter.Layers))
	// Output: 2 4 0
}

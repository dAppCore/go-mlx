// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package kimi

import (
	core "dappco.re/go"
	"dappco.re/go/mlx/pkg/metal"
)

// ExampleLoadKimi loads a Kimi sparse-MoE checkpoint from a model directory
// (config.json + tokenizer.json + *.safetensors). The returned model satisfies
// metal.InternalModel and is also registered under the "kimi" model type, so
// metal.LoadAndInit dispatches here automatically.
func ExampleLoadKimi() {
	model, err := LoadKimi("/path/to/kimi")
	_, _ = model, err
}

// ExampleKimiModel_Forward runs a token batch [B, L] through the decoder stack
// and returns logits [B, L, vocab]. caches must hold one cache per layer (see
// NewCache); attention appends to them across decode steps.
func ExampleKimiModel_Forward() {
	var (
		model  *KimiModel
		tokens *metal.Array
		caches []metal.Cache
	)
	if model == nil {
		return
	}
	logits := model.Forward(tokens, caches)
	_ = logits
}

// ExampleKimiModel_ForwardMasked is the explicit-mask variant of Forward; a nil
// mask is the single-token decode path. Forward delegates to it with a nil mask.
func ExampleKimiModel_ForwardMasked() {
	var (
		model  *KimiModel
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

// ExampleKimiModel_NewCache allocates one KV cache per decoder layer — the
// caches slice Forward/ForwardMasked expect.
func ExampleKimiModel_NewCache() {
	model := &KimiModel{
		Layers: []*KimiDecoderLayer{
			nil,
			nil,
		},
	}

	caches := model.NewCache()

	core.Println(len(caches), core.Sprintf("%T", caches[0]), core.Sprintf("%T", caches[1]))
	// Output: 2 *metal.KVCache *metal.KVCache
}

// ExampleKimiModel_NumLayers reports the decoder-layer count.
func ExampleKimiModel_NumLayers() {
	model := &KimiModel{
		Layers: []*KimiDecoderLayer{
			nil,
			nil,
			nil,
		},
	}

	core.Println(model.NumLayers())
	// Output: 3
}

// ExampleKimiModel_ModelType returns the canonical architecture token used by
// the loader registry and diagnostics.
func ExampleKimiModel_ModelType() {
	model := &KimiModel{modelType: "kimi"}

	core.Println(model.ModelType())
	// Output: kimi
}

// ExampleKimiModel_MoETextDecodeFamily returns the family token reported in
// "native MoE decode unavailable" diagnostics.
func ExampleKimiModel_MoETextDecodeFamily() {
	model := &KimiModel{}

	core.Println(model.MoETextDecodeFamily())
	// Output: kimi
}

// ExampleKimiModel_ApplyLoRA attaches LoRA adapters to the named projections. On
// an empty model no projections exist, so the adapter is created but holds no
// layers — the config is still normalised (Alpha defaults from Rank).
func ExampleKimiModel_ApplyLoRA() {
	model := &KimiModel{}
	adapter := model.ApplyLoRA(metal.LoRAConfig{
		Rank:         2,
		Alpha:        4,
		TargetLayers: []string{"q_proj"},
	})

	core.Println(adapter.Config.Rank, adapter.Config.Alpha, len(adapter.Layers))
	// Output: 2 4 0
}

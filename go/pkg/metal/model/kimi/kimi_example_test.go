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
// metal.LoadAndInit dispatches here automatically. This illustrates the load →
// decode wiring; it is not run (a real checkpoint + Metal runtime are required —
// see TestModel_LoadKimi_Good / TestModel_Forward_Good for the executed paths).
func ExampleLoadKimi() {
	model, err := LoadKimi("/path/to/kimi")
	if err != nil {
		return
	}
	defer model.CloseModel()

	tokens := metal.FromValues([]int32{2, 9, 14}, 1, 3) // [B=1, L=3]
	caches := model.NewCache()                          // one KV cache per layer
	logits := model.Forward(tokens, caches)             // [B, L, vocab]
	_ = logits
}

// ExampleKimiModel_Forward runs a token batch [B, L] through the decoder stack
// and returns logits [B, L, vocab]. caches must hold one cache per layer (see
// NewCache); attention appends to them across decode steps. Illustrative only —
// TestModel_Forward_Good runs this over a synthetic fixture.
func ExampleKimiModel_Forward() {
	var model *KimiModel // a loaded *KimiModel (see ExampleLoadKimi)
	if model == nil {
		return
	}
	tokens := metal.FromValues([]int32{2, 9, 14}, 1, 3)
	caches := model.NewCache()
	logits := model.Forward(tokens, caches)
	_ = logits
}

// ExampleKimiModel_ForwardMasked is the explicit-mask variant of Forward; a nil
// mask is the single-token decode path. Forward delegates to it with a nil mask.
// Illustrative only — TestModel_ForwardMasked_Good runs the nil-mask path.
func ExampleKimiModel_ForwardMasked() {
	var model *KimiModel // a loaded *KimiModel (see ExampleLoadKimi)
	if model == nil {
		return
	}
	tokens := metal.FromValues([]int32{2}, 1, 1)
	caches := model.NewCache()
	logits := model.ForwardMasked(tokens, nil, caches) // nil mask = decode step
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

// ExampleKimiModel_FillModelInfo copies vocab/hidden/context sizing and (when
// present) quantization from the config into a metal.ModelInfo. Hand-built
// config, no load.
func ExampleKimiModel_FillModelInfo() {
	model := &KimiModel{Cfg: &KimiConfig{
		VocabSize:             163840,
		HiddenSize:            2048,
		MaxPositionEmbeddings: 131072,
		Quantization:          &metal.QuantizationConfig{Bits: 4, GroupSize: 64},
	}}

	info := &metal.ModelInfo{}
	model.FillModelInfo(info)

	core.Println(info.VocabSize, info.HiddenSize, info.ContextLength, info.QuantBits, info.QuantGroup)
	// Output: 163840 2048 131072 4 64
}

// ExampleKimiModel_MoETextRuntimeAvailable reports whether the native
// selected-expert decode kernels are linked for every layer. An empty model has
// no layers wired, so the native fast path is unavailable.
func ExampleKimiModel_MoETextRuntimeAvailable() {
	model := &KimiModel{}

	core.Println(model.MoETextRuntimeAvailable())
	// Output: false
}

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

// ExampleGemmaModel_ResolveLoRALinear_outOfRange shows the bounds-safe contract:
// a layer index past the end of the model returns nil rather than panicking, so
// a LoRA resolver can probe optimistically.
func ExampleGemmaModel_ResolveLoRALinear_outOfRange() {
	model := &GemmaModel{Layers: []*DecoderLayer{{Attention: &Attention{}}}}

	// layerIdx 5 is past the single layer — nil, no panic.
	proj := model.ResolveLoRALinear(5, "self_attn.q_proj")
	core.Println(proj == nil)
	// Output: true
}

// Example_parseConfig shows how a sparse Gemma 3 config.json is normalised:
// the RoPE thetas, RMSNorm epsilon, and sliding-window pattern fall back to
// their Gemma 3 defaults, while vocab_size is deliberately left at zero —
// it is a dimension derived from the embedding tensor at load, never fabricated
// from the config.
func Example_parseConfig() {
	cfg, err := parseConfig([]byte(`{
		"hidden_size": 1152,
		"num_hidden_layers": 26,
		"num_attention_heads": 4,
		"head_dim": 256
	}`))
	if err != nil {
		core.Println("error:", err.Error())
		return
	}

	core.Println(cfg.RopeTheta, cfg.RopeLocalBaseFreq, cfg.RMSNormEps, cfg.SlidingWindowPattern, cfg.VocabSize, cfg.ModelType)
	// Output: 1e+06 10000 1e-06 6 0 gemma3
}

// Example_isLayerSliding shows the Gemma 3 attention pattern: with a
// sliding-window pattern of 6, every layer is local (sliding) except every 6th,
// which is global. Layer indices are zero-based, so layer index 5 (the 6th) is
// the first global layer.
func Example_isLayerSliding() {
	const pattern = 6
	for idx := int32(0); idx < 7; idx++ {
		core.Println(idx, isLayerSliding(idx, pattern))
	}
	// Output:
	// 0 true
	// 1 true
	// 2 true
	// 3 true
	// 4 true
	// 5 false
	// 6 true
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

func ExampleGemmaModel_NumQueryHeads() {
	model := &GemmaModel{Cfg: &TextConfig{NumAttentionHeads: 8}}

	// Zero when the config is unavailable (load failed before Cfg attached).
	core.Println(model.NumQueryHeads(), (&GemmaModel{}).NumQueryHeads())
	// Output: 8 0
}

func ExampleGemmaModel_ResolveLoRALinear() {
	model := &GemmaModel{
		Layers: []*DecoderLayer{
			{Attention: &Attention{QProj: &metal.Linear{}}},
		},
	}

	known := model.ResolveLoRALinear(0, "self_attn.q_proj")
	unknown := model.ResolveLoRALinear(0, "mlp.gate_proj")
	outOfRange := model.ResolveLoRALinear(9, "self_attn.q_proj")

	core.Println(known != nil, unknown == nil, outOfRange == nil)
	// Output: true true true
}

func ExampleGemmaModel_FillModelInfo() {
	model := &GemmaModel{Cfg: &TextConfig{
		VocabSize:             262144,
		HiddenSize:            1152,
		MaxPositionEmbeddings: 32768,
		Quantization:          &metal.QuantizationConfig{Bits: 4, GroupSize: 64},
	}}

	var info metal.ModelInfo
	model.FillModelInfo(&info)

	core.Println(info.VocabSize, info.HiddenSize, info.ContextLength, info.QuantBits, info.QuantGroup)
	// Output: 262144 1152 32768 4 64
}

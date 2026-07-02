// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	core "dappco.re/go"
	coreio "dappco.re/go/io"
	"dappco.re/go/mlx/gguf"
	g4 "dappco.re/go/mlx/pkg/model/gemma4"
	"dappco.re/go/mlx/pkg/safetensors"
	"dappco.re/go/mlx/pkg/tokenizer"
)

const nativeGemma4AssistantGGUFArch = "gemma4-assistant"

// ResolveGemma4AssistantGGUFDrafterFile reports whether path is a GGUF
// assistant drafter source: either a .gguf file directly or a directory with
// exactly one .gguf file. Ambiguous directories stand down.
func ResolveGemma4AssistantGGUFDrafterFile(path string) (string, bool) {
	if path == "" {
		return "", false
	}
	if nativeHasGGUFSuffix(path) {
		if _, err := coreio.Local.Stat(path); err != nil {
			return "", false
		}
		return path, true
	}
	entries, err := coreio.Local.List(path)
	if err != nil {
		return "", false
	}
	var matches []string
	for _, entry := range entries {
		if nativeHasGGUFSuffix(entry.Name()) {
			matches = append(matches, core.PathJoin(path, entry.Name()))
		}
	}
	if len(matches) != 1 {
		return "", false
	}
	return matches[0], true
}

func nativeHasGGUFSuffix(path string) bool {
	return core.HasSuffix(core.Lower(path), ".gguf")
}

func nativeGemma4AssistantGGUFWeightName(name string) string {
	switch name {
	case "token_embd.weight":
		return "model.embed_tokens.weight"
	case "output_norm.weight":
		return "model.norm.weight"
	case "nextn.pre_projection.weight":
		return "pre_projection.weight"
	case "nextn.post_projection.weight":
		return "post_projection.weight"
	}
	if !core.HasPrefix(name, "blk.") {
		return ""
	}
	rest := core.TrimPrefix(name, "blk.")
	dot := -1
	for i := 0; i < len(rest); i++ {
		if rest[i] == '.' {
			dot = i
			break
		}
	}
	if dot <= 0 {
		return ""
	}
	layer, leaf := rest[:dot], rest[dot+1:]
	prefix := "model.layers." + layer
	switch leaf {
	case "attn_norm.weight":
		return prefix + ".input_layernorm.weight"
	case "post_attention_norm.weight":
		return prefix + ".post_attention_layernorm.weight"
	case "ffn_norm.weight":
		return prefix + ".pre_feedforward_layernorm.weight"
	case "post_ffw_norm.weight":
		return prefix + ".post_feedforward_layernorm.weight"
	case "attn_q.weight":
		return prefix + ".self_attn.q_proj.weight"
	case "attn_q_norm.weight":
		return prefix + ".self_attn.q_norm.weight"
	case "attn_output.weight":
		return prefix + ".self_attn.o_proj.weight"
	case "ffn_gate.weight":
		return prefix + ".mlp.gate_proj.weight"
	case "ffn_up.weight":
		return prefix + ".mlp.up_proj.weight"
	case "ffn_down.weight":
		return prefix + ".mlp.down_proj.weight"
	case "layer_output_scale.weight":
		return prefix + ".layer_scalar.weight"
	}
	return ""
}

func nativeGGUFMetaInt(meta map[string]any, key string) int {
	switch v := meta[key].(type) {
	case uint32:
		return int(v)
	case int32:
		return int(v)
	case uint64:
		return int(v)
	case int64:
		return int(v)
	case int:
		return v
	case float64:
		return int(v)
	}
	return 0
}

func nativeGGUFMetaFloat(meta map[string]any, key string) float32 {
	switch v := meta[key].(type) {
	case float32:
		return v
	case float64:
		return float32(v)
	case uint32:
		return float32(v)
	case int32:
		return float32(v)
	}
	return 0
}

func nativeGemma4AssistantConfigFromGGUF(meta map[string]any) (Gemma4AssistantConfig, error) {
	if arch, _ := meta["general.architecture"].(string); arch != nativeGemma4AssistantGGUFArch {
		return Gemma4AssistantConfig{}, core.E("native.gemma4.assistant.gguf", "general.architecture is not gemma4-assistant", nil)
	}
	const p = nativeGemma4AssistantGGUFArch + "."
	layers := nativeGGUFMetaInt(meta, p+"block_count")
	hidden := nativeGGUFMetaInt(meta, p+"embedding_length")
	heads := nativeGGUFMetaInt(meta, p+"attention.head_count")
	headDim := nativeGGUFMetaInt(meta, p+"attention.key_length")
	if layers <= 0 || hidden <= 0 || heads <= 0 || headDim <= 0 {
		return Gemma4AssistantConfig{}, core.E("native.gemma4.assistant.gguf",
			"drafter gguf is missing block_count / embedding_length / head_count / key_length metadata", nil)
	}
	backbone := nativeGGUFMetaInt(meta, p+"embedding_length_out")
	if backbone <= 0 {
		backbone = hidden
	}
	pattern := nativeGGUFMetaInt(meta, p+"attention.sliding_window_pattern")
	if pattern <= 0 {
		pattern = 1
	}
	layerTypes := make([]string, layers)
	for i := range layerTypes {
		if (i+1)%pattern == 0 {
			layerTypes[i] = "full_attention"
		} else {
			layerTypes[i] = "sliding_attention"
		}
	}
	eps := nativeGGUFMetaFloat(meta, p+"attention.layer_norm_rms_epsilon")
	if eps == 0 {
		eps = 1e-6
	}
	freqBase := nativeGGUFMetaFloat(meta, p+"rope.freq_base")
	if freqBase == 0 {
		freqBase = 1000000
	}
	freqBaseSWA := nativeGGUFMetaFloat(meta, p+"rope.freq_base_swa")
	if freqBaseSWA == 0 {
		freqBaseSWA = 10000
	}
	rotaryFactor := func(dimKey string) float32 {
		if dims := nativeGGUFMetaInt(meta, dimKey); dims > 0 && headDim > 0 {
			return float32(dims) / float32(headDim)
		}
		return 1
	}
	text := g4.Config{
		HiddenSize:              hidden,
		NumHiddenLayers:         layers,
		IntermediateSize:        nativeGGUFMetaInt(meta, p+"feed_forward_length"),
		NumAttentionHeads:       heads,
		NumKeyValueHeads:        nativeGGUFMetaInt(meta, p+"attention.head_count_kv"),
		HeadDim:                 headDim,
		VocabSize:               nativeGGUFMetaInt(meta, p+"vocab_size"),
		RMSNormEps:              eps,
		SlidingWindow:           nativeGGUFMetaInt(meta, p+"attention.sliding_window"),
		MaxPositionEmbeddings:   nativeGGUFMetaInt(meta, p+"context_length"),
		NumKVSharedLayers:       nativeGGUFMetaInt(meta, p+"attention.shared_kv_layers"),
		HiddenSizePerLayerInput: nativeGGUFMetaInt(meta, p+"embedding_length_per_layer_input"),
		LayerTypes:              layerTypes,
		RopeParameters: map[string]g4.RopeParam{
			"full_attention": {
				RopeTheta:           freqBase,
				RopeType:            "default",
				Factor:              1,
				PartialRotaryFactor: rotaryFactor(p + "rope.dimension_count"),
			},
			"sliding_attention": {
				RopeTheta:           freqBaseSWA,
				RopeType:            "default",
				Factor:              1,
				PartialRotaryFactor: rotaryFactor(p + "rope.dimension_count_swa"),
			},
		},
	}
	if text.NumKeyValueHeads <= 0 {
		text.NumKeyValueHeads = heads
	}
	return Gemma4AssistantConfig{
		ModelType:          "gemma4_assistant",
		BackboneHiddenSize: backbone,
		TextConfig:         text,
	}, nil
}

func loadNativeGemma4AssistantFromGGUF(file string, tok *tokenizer.Tokenizer) (*Gemma4AssistantModel, error) {
	if tok == nil {
		return nil, core.E("native.gemma4.assistant.gguf", "target tokenizer required", nil)
	}
	meta, err := gguf.Metadata(file)
	if err != nil {
		return nil, core.E("native.gemma4.assistant.gguf", "read gguf metadata", err)
	}
	cfg, err := nativeGemma4AssistantConfigFromGGUF(meta)
	if err != nil {
		return nil, err
	}
	raw, err := gguf.LoadTensors(file)
	if err != nil {
		return nil, core.E("native.gemma4.assistant.gguf", "load gguf tensors", err)
	}
	m, err := buildNativeGemma4AssistantFromGGUFTensors(cfg, raw, tok)
	if err != nil {
		_ = raw.Close()
		return nil, err
	}
	return m, nil
}

func buildNativeGemma4AssistantFromGGUFTensors(cfg Gemma4AssistantConfig, raw *gguf.TensorMapping, tok *tokenizer.Tokenizer) (*Gemma4AssistantModel, error) {
	if raw == nil {
		return nil, core.NewError("native.gemma4.assistant.gguf tensor map is nil")
	}
	weights := make(map[string]safetensors.Tensor, len(raw.Tensors))
	for name, tensor := range raw.Tensors {
		mapped := nativeGemma4AssistantGGUFWeightName(name)
		if mapped == "" {
			continue
		}
		weights[mapped] = tensor
	}
	if cfg.TextConfig.VocabSize == 0 {
		if embed, ok := weights["model.embed_tokens.weight"]; ok && len(embed.Shape) > 0 {
			cfg.TextConfig.VocabSize = embed.Shape[0]
		}
	}
	if err := validateNativeGemma4AssistantConfig(cfg); err != nil {
		return nil, err
	}
	arch, err := cfg.TextConfig.Arch()
	if err != nil {
		return nil, core.E("native.gemma4.assistant.gguf", "derive arch", err)
	}
	m := &Gemma4AssistantModel{
		Config:                   cfg,
		Arch:                     arch,
		Tensors:                  weights,
		BackboneHiddenSize:       cfg.BackboneHiddenSize,
		NumCentroids:             cfg.NumCentroids,
		CentroidIntermediateTopK: cfg.CentroidIntermediateTopK,
		UseOrderedEmbeddings:     cfg.UseOrderedEmbeddings,
		Tok:                      tok,
		gguf:                     raw,
	}
	if err := validateNativeGemma4AssistantModel(m); err != nil {
		_ = m.Close()
		return nil, core.E("native.gemma4.assistant.gguf", "validate tensors", err)
	}
	return m, nil
}

// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma4

import (
	core "dappco.re/go"
	coreio "dappco.re/go/io"
	"dappco.re/go/mlx/gguf"
	"dappco.re/go/mlx/pkg/metal"
)

// GGUF drafter lane (#92): unsloth ships gemma-4 MTP drafters as single
// .gguf files (repo-root mtp-*.gguf + MTP/<precision>-MTP.gguf, arch tag
// "gemma4-assistant", llama.cpp PR ggml-org/llama.cpp#23398). The file
// carries everything but the tokenizer config we already hold: drafter
// dims in gemma4-assistant.* metadata keys, tensors under llama.cpp
// names. This lane maps both onto the exact shapes
// buildGemma4AssistantFromWeights expects and borrows the TARGET's
// tokenizer (identical vocab by construction — the drafter proposes
// tokens the target verifies).

// ggufAssistantArch is the general.architecture tag the drafter declares.
const ggufAssistantArch = "gemma4-assistant"

// resolveGGUFDrafterFile reports whether path is a GGUF drafter source:
// either a .gguf file directly, or a directory holding exactly one .gguf
// (the unsloth MTP/ directory shape). Ambiguous directories stand down —
// detection stays reactive, never guessing between siblings.
func resolveGGUFDrafterFile(path string) (string, bool) {
	if path == "" {
		return "", false
	}
	if hasGGUFSuffix(path) {
		if _, err := coreio.Local.Stat(path); err != nil {
			return "", false
		}
		return path, true
	}
	entries, err := coreio.Local.List(path)
	if err != nil {
		return "", false
	}
	var ggufs []string
	for _, entry := range entries {
		if hasGGUFSuffix(entry.Name()) {
			ggufs = append(ggufs, core.JoinPath(path, entry.Name()))
		}
	}
	if len(ggufs) != 1 {
		return "", false
	}
	return ggufs[0], true
}

func hasGGUFSuffix(path string) bool {
	return core.HasSuffix(core.Lower(path), ".gguf")
}

// ggufAssistantWeightName maps one llama.cpp drafter tensor name onto the
// name buildGemma4AssistantFromWeights expects. Unknown names map to ""
// and are dropped (rope frequency tables etc. are rebuilt, not loaded).
//
// The full map, verified against unsloth/gemma-4-31B-it-GGUF's MTP header:
//
//	token_embd.weight             -> model.embed_tokens.weight
//	output_norm.weight            -> model.norm.weight
//	nextn.pre_projection.weight   -> pre_projection.weight
//	nextn.post_projection.weight  -> post_projection.weight
//	blk.N.attn_norm               -> model.layers.N.input_layernorm
//	blk.N.post_attention_norm     -> model.layers.N.post_attention_layernorm
//	blk.N.ffn_norm                -> model.layers.N.pre_feedforward_layernorm
//	blk.N.post_ffw_norm           -> model.layers.N.post_feedforward_layernorm
//	blk.N.attn_q                  -> model.layers.N.self_attn.q_proj
//	blk.N.attn_q_norm             -> model.layers.N.self_attn.q_norm
//	blk.N.attn_output             -> model.layers.N.self_attn.o_proj
//	blk.N.ffn_{gate,up,down}      -> model.layers.N.mlp.{gate,up,down}_proj
//	blk.N.layer_output_scale      -> model.layers.N.layer_scalar
//
// No attn_k / attn_v exist — the drafter rides the target's shared KV.
func ggufAssistantWeightName(name string) string {
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

// ggufMetaInt32 coerces a gguf metadata value (the parser yields assorted
// integer widths) to int32; absent or non-numeric yields 0.
func ggufMetaInt32(meta map[string]any, key string) int32 {
	switch v := meta[key].(type) {
	case uint32:
		return int32(v)
	case int32:
		return v
	case uint64:
		return int32(v)
	case int64:
		return int32(v)
	case int:
		return int32(v)
	case float64:
		return int32(v)
	}
	return 0
}

func ggufMetaFloat32(meta map[string]any, key string) float32 {
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

// gemma4AssistantConfigFromGGUF builds the drafter config from the
// gemma4-assistant.* metadata keys. The layer schedule derives from the
// DECLARED sliding_window_pattern — in gguf-land the pattern key IS the
// model's schedule declaration (llama.cpp derives identically); this is
// reading a declaration, not synthesising from a guessed period.
func gemma4AssistantConfigFromGGUF(meta map[string]any) (*Gemma4AssistantConfig, error) {
	if arch, _ := meta["general.architecture"].(string); arch != ggufAssistantArch {
		return nil, core.E("gemma4.assistant.gguf", "general.architecture is not gemma4-assistant", nil)
	}
	const p = ggufAssistantArch + "."
	layers := ggufMetaInt32(meta, p+"block_count")
	hidden := ggufMetaInt32(meta, p+"embedding_length")
	heads := ggufMetaInt32(meta, p+"attention.head_count")
	headDim := ggufMetaInt32(meta, p+"attention.key_length")
	if layers <= 0 || hidden <= 0 || heads <= 0 || headDim <= 0 {
		return nil, core.E("gemma4.assistant.gguf",
			"drafter gguf is missing block_count / embedding_length / head_count / key_length metadata", nil)
	}
	backbone := ggufMetaInt32(meta, p+"embedding_length_out")
	if backbone <= 0 {
		// Older exports may omit the out-projection width; the pair
		// validator cross-checks against the target either way.
		backbone = hidden
	}

	pattern := ggufMetaInt32(meta, p+"attention.sliding_window_pattern")
	if pattern <= 0 {
		pattern = 1
	}
	layerTypes := make([]string, layers)
	for i := range layerTypes {
		if (int32(i)+1)%pattern == 0 {
			layerTypes[i] = "full_attention"
		} else {
			layerTypes[i] = "sliding_attention"
		}
	}

	eps := ggufMetaFloat32(meta, p+"attention.layer_norm_rms_epsilon")
	if eps == 0 {
		eps = 1e-6
	}
	freqBase := ggufMetaFloat32(meta, p+"rope.freq_base")
	if freqBase == 0 {
		freqBase = 1000000.0
	}
	freqBaseSWA := ggufMetaFloat32(meta, p+"rope.freq_base_swa")
	if freqBaseSWA == 0 {
		freqBaseSWA = 10000.0
	}
	rotaryFactor := func(dimKey string) float32 {
		if dims := ggufMetaInt32(meta, dimKey); dims > 0 && headDim > 0 {
			return float32(dims) / float32(headDim)
		}
		return 1.0
	}

	text := &Gemma4TextConfig{}
	text.ModelType = "gemma4_text"
	text.HiddenSize = hidden
	text.NumHiddenLayers = layers
	text.IntermediateSize = ggufMetaInt32(meta, p+"feed_forward_length")
	text.NumAttentionHeads = heads
	text.NumKeyValueHeads = ggufMetaInt32(meta, p+"attention.head_count_kv")
	if text.NumKeyValueHeads <= 0 {
		text.NumKeyValueHeads = heads
	}
	text.HeadDim = headDim
	text.RMSNormEps = eps
	text.SlidingWindow = ggufMetaInt32(meta, p+"attention.sliding_window")
	text.SlidingWindowPattern = pattern
	text.NumKVSharedLayers = ggufMetaInt32(meta, p+"attention.shared_kv_layers")
	text.HiddenSizePerLayerInput = ggufMetaInt32(meta, p+"embedding_length_per_layer_input")
	text.MaxPositionEmbeddings = ggufMetaInt32(meta, p+"context_length")
	text.LayerTypesInput = layerTypes
	text.LayerTypes = append([]string(nil), layerTypes...)
	text.RopeParameters = map[string]RopeParams{
		"full_attention": {
			RopeTheta:           float64(freqBase),
			RopeType:            "default",
			Factor:              1.0,
			PartialRotaryFactor: rotaryFactor(p + "rope.dimension_count"),
		},
		"sliding_attention": {
			RopeTheta:           float64(freqBaseSWA),
			RopeType:            "default",
			Factor:              1.0,
			PartialRotaryFactor: rotaryFactor(p + "rope.dimension_count_swa"),
		},
	}
	gemma4FinaliseEmbeddingScales(text)

	return &Gemma4AssistantConfig{
		ModelType:          "gemma4_assistant",
		BackboneHiddenSize: backbone,
		TextConfig:         text,
	}, nil
}

// loadGemma4AssistantFromGGUF loads an unsloth-style single-file drafter:
// metadata -> config, tensors -> renamed weight map, tokenizer borrowed
// from the target. mlx's gguf reader dequantises Q8_0 to fp16 on load, so
// the dense (nil-quantization) builder path applies for the published
// BF16 / F16 / Q8_0 drafters alike.
func loadGemma4AssistantFromGGUF(file string, tok *metal.Tokenizer) (*Gemma4AssistantModel, error) {
	if tok == nil {
		return nil, core.E("gemma4.assistant.gguf", "target tokenizer required (drafter shares the target vocab)", nil)
	}
	meta, err := gguf.Metadata(file)
	if err != nil {
		return nil, core.E("gemma4.assistant.gguf", "read gguf metadata", err)
	}
	cfg, err := gemma4AssistantConfigFromGGUF(meta)
	if err != nil {
		return nil, err
	}
	raw, err := metal.LoadAllGGUF(file)
	if err != nil {
		return nil, core.E("gemma4.assistant.gguf", "load gguf tensors", err)
	}
	return buildGemma4AssistantFromGGUFTensors(cfg, raw, tok)
}

// buildGemma4AssistantFromGGUFTensors renames llama.cpp tensor names onto
// the builder's expectations and assembles + validates the drafter. Split
// from the file loader so tests can drive it with SaveGGUF round-trip
// tensors and a hand-built config (mlx's gguf writer emits no metadata).
func buildGemma4AssistantFromGGUFTensors(cfg *Gemma4AssistantConfig, raw map[string]*metal.Array, tok *metal.Tokenizer) (*Gemma4AssistantModel, error) {
	weights := make(map[string]*metal.Array, len(raw))
	for name, arr := range raw {
		mapped := ggufAssistantWeightName(name)
		if mapped == "" {
			metal.Free(arr)
			continue
		}
		weights[mapped] = arr
	}
	if cfg.TextConfig.VocabSize == 0 {
		if embed, ok := weights["model.embed_tokens.weight"]; ok && embed != nil {
			cfg.TextConfig.VocabSize = int32(embed.Dim(0))
		}
	}
	m := buildGemma4AssistantFromWeights(cfg, weights, tok)

	loadSucceeded := false
	defer func() {
		if loadSucceeded {
			return
		}
		retained := gemma4AssistantRetainedWeights(m)
		gemma4FreeUnusedWeights(weights, retained)
		closeGemma4Assistant(m)
		metal.ClearCache()
	}()

	if err := validateGemma4AssistantModel(m); err != nil {
		return nil, core.E("gemma4.assistant.gguf", "validate tensors", err)
	}
	retained := gemma4AssistantRetainedWeights(m)
	gemma4FreeUnusedWeights(weights, retained)
	gemma4MaterializeRetainedWeights(retained, nil)
	loadSucceeded = true
	return m, nil
}

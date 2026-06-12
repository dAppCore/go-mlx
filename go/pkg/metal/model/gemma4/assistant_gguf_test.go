// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma4

import (
	"testing"

	core "dappco.re/go"
	coreio "dappco.re/go/io"
	"dappco.re/go/mlx/pkg/metal"
)

// ggufNameForTest inverts ggufAssistantWeightName for the fixture's names —
// the round-trip writes a gguf exactly as unsloth would.
func ggufNameForTest(t *testing.T, hf string) string {
	t.Helper()
	switch hf {
	case "model.embed_tokens.weight":
		return "token_embd.weight"
	case "model.norm.weight":
		return "output_norm.weight"
	case "pre_projection.weight":
		return "nextn.pre_projection.weight"
	case "post_projection.weight":
		return "nextn.post_projection.weight"
	}
	if !core.HasPrefix(hf, "model.layers.") {
		t.Fatalf("no gguf name for fixture weight %q", hf)
	}
	rest := core.TrimPrefix(hf, "model.layers.")
	dot := -1
	for i := 0; i < len(rest); i++ {
		if rest[i] == '.' {
			dot = i
			break
		}
	}
	layer, leaf := rest[:dot], rest[dot+1:]
	blk := "blk." + layer + "."
	switch leaf {
	case "input_layernorm.weight":
		return blk + "attn_norm.weight"
	case "post_attention_layernorm.weight":
		return blk + "post_attention_norm.weight"
	case "pre_feedforward_layernorm.weight":
		return blk + "ffn_norm.weight"
	case "post_feedforward_layernorm.weight":
		return blk + "post_ffw_norm.weight"
	case "layer_scalar":
		return blk + "layer_output_scale.weight"
	case "self_attn.q_proj.weight":
		return blk + "attn_q.weight"
	case "self_attn.o_proj.weight":
		return blk + "attn_output.weight"
	case "self_attn.q_norm.weight":
		return blk + "attn_q_norm.weight"
	case "mlp.gate_proj.weight":
		return blk + "ffn_gate.weight"
	case "mlp.up_proj.weight":
		return blk + "ffn_up.weight"
	case "mlp.down_proj.weight":
		return blk + "ffn_down.weight"
	}
	t.Fatalf("no gguf name for fixture layer weight %q", hf)
	return ""
}

func TestAssistantGGUF_WeightNameMap_Good(t *testing.T) {
	cases := map[string]string{
		"token_embd.weight":                "model.embed_tokens.weight",
		"output_norm.weight":               "model.norm.weight",
		"nextn.pre_projection.weight":      "pre_projection.weight",
		"nextn.post_projection.weight":     "post_projection.weight",
		"blk.0.attn_norm.weight":           "model.layers.0.input_layernorm.weight",
		"blk.3.post_attention_norm.weight": "model.layers.3.post_attention_layernorm.weight",
		"blk.1.ffn_norm.weight":            "model.layers.1.pre_feedforward_layernorm.weight",
		"blk.2.post_ffw_norm.weight":       "model.layers.2.post_feedforward_layernorm.weight",
		"blk.0.attn_q.weight":              "model.layers.0.self_attn.q_proj.weight",
		"blk.0.attn_q_norm.weight":         "model.layers.0.self_attn.q_norm.weight",
		"blk.0.attn_output.weight":         "model.layers.0.self_attn.o_proj.weight",
		"blk.1.ffn_gate.weight":            "model.layers.1.mlp.gate_proj.weight",
		"blk.1.ffn_up.weight":              "model.layers.1.mlp.up_proj.weight",
		"blk.1.ffn_down.weight":            "model.layers.1.mlp.down_proj.weight",
		"blk.2.layer_output_scale.weight":  "model.layers.2.layer_scalar.weight",
	}
	for gguf, want := range cases {
		if got := ggufAssistantWeightName(gguf); got != want {
			t.Fatalf("ggufAssistantWeightName(%q) = %q, want %q", gguf, got, want)
		}
	}
	for _, unknown := range []string{"rope_freqs.weight", "blk.0.attn_k.weight", "blk.notanumber", "output.weight"} {
		if got := ggufAssistantWeightName(unknown); got != "" {
			t.Fatalf("ggufAssistantWeightName(%q) = %q, want dropped", unknown, got)
		}
	}
}

func TestAssistantGGUF_ConfigFromMetadata_Good(t *testing.T) {
	meta := map[string]any{
		"general.architecture":                              "gemma4-assistant",
		"gemma4-assistant.block_count":                      uint32(4),
		"gemma4-assistant.embedding_length":                 uint32(16),
		"gemma4-assistant.feed_forward_length":              uint32(32),
		"gemma4-assistant.attention.head_count":             uint32(2),
		"gemma4-assistant.attention.head_count_kv":          uint32(1),
		"gemma4-assistant.attention.key_length":             uint32(8),
		"gemma4-assistant.attention.layer_norm_rms_epsilon": float32(1e-5),
		"gemma4-assistant.attention.sliding_window":         uint32(512),
		"gemma4-assistant.attention.sliding_window_pattern": uint32(2),
		"gemma4-assistant.context_length":                   uint32(131072),
		"gemma4-assistant.embedding_length_out":             uint32(64),
		"gemma4-assistant.rope.freq_base":                   float32(1000000),
		"gemma4-assistant.rope.freq_base_swa":               float32(10000),
		"gemma4-assistant.rope.dimension_count":             uint32(4),
		"gemma4-assistant.rope.dimension_count_swa":         uint32(8),
	}
	cfg, err := gemma4AssistantConfigFromGGUF(meta)
	if err != nil {
		t.Fatalf("gemma4AssistantConfigFromGGUF: %v", err)
	}
	text := cfg.TextConfig
	if cfg.BackboneHiddenSize != 64 || text.HiddenSize != 16 || text.NumHiddenLayers != 4 {
		t.Fatalf("dims = backbone %d hidden %d layers %d, want 64/16/4",
			cfg.BackboneHiddenSize, text.HiddenSize, text.NumHiddenLayers)
	}
	if text.HeadDim != 8 || text.NumAttentionHeads != 2 || text.NumKeyValueHeads != 1 {
		t.Fatalf("heads = dim %d q %d kv %d, want 8/2/1", text.HeadDim, text.NumAttentionHeads, text.NumKeyValueHeads)
	}
	wantTypes := []string{"sliding_attention", "full_attention", "sliding_attention", "full_attention"}
	for i, want := range wantTypes {
		if text.LayerTypes[i] != want {
			t.Fatalf("layer %d type = %q, want %q (pattern-declared schedule)", i, text.LayerTypes[i], want)
		}
	}
	full := text.RopeParameters["full_attention"]
	sliding := text.RopeParameters["sliding_attention"]
	if full.RopeTheta != 1000000 || sliding.RopeTheta != 10000 {
		t.Fatalf("rope thetas = %v/%v, want 1e6/1e4", full.RopeTheta, sliding.RopeTheta)
	}
	if full.PartialRotaryFactor != 0.5 || sliding.PartialRotaryFactor != 1.0 {
		t.Fatalf("partial rotary = %v/%v, want 0.5/1.0 (dimension_count/key_length)",
			full.PartialRotaryFactor, sliding.PartialRotaryFactor)
	}
}

func TestAssistantGGUF_ConfigFromMetadata_Bad(t *testing.T) {
	if _, err := gemma4AssistantConfigFromGGUF(map[string]any{
		"general.architecture": "gemma4",
	}); err == nil {
		t.Fatal("wrong architecture must be refused")
	}
	if _, err := gemma4AssistantConfigFromGGUF(map[string]any{
		"general.architecture":         "gemma4-assistant",
		"gemma4-assistant.block_count": uint32(4),
	}); err == nil {
		t.Fatal("missing dims must be refused")
	}
}

func TestAssistantGGUF_ResolveDrafterFile_Good(t *testing.T) {
	dir := t.TempDir()
	file := core.JoinPath(dir, "mtp-tiny.gguf")
	if err := coreio.Local.Write(file, "stub"); err != nil {
		t.Fatalf("write stub: %v", err)
	}
	if got, ok := resolveGGUFDrafterFile(file); !ok || got != file {
		t.Fatalf("file path = %q/%v, want direct hit", got, ok)
	}
	if got, ok := resolveGGUFDrafterFile(dir); !ok || got != file {
		t.Fatalf("single-gguf dir = %q/%v, want the one file", got, ok)
	}
	if err := coreio.Local.Write(core.JoinPath(dir, "second.gguf"), "stub"); err != nil {
		t.Fatalf("write second: %v", err)
	}
	if _, ok := resolveGGUFDrafterFile(dir); ok {
		t.Fatal("ambiguous two-gguf dir must stand down")
	}
	if _, ok := resolveGGUFDrafterFile(core.JoinPath(dir, "missing.gguf")); ok {
		t.Fatal("missing file must stand down")
	}
}

// The round-trip: the tiny fixture saved under unsloth names loads through
// the gguf tensor lane bit-identical to the safetensors reference.
func TestAssistantGGUF_RoundTripMatchesReference_Good(t *testing.T) {
	requireMetalRuntime(t)

	refDir := t.TempDir()
	writeGemma4AssistantConfig(t, refDir, false)
	writeMinimalTokenizer(t, refDir)
	if err := metal.SaveSafetensors(core.JoinPath(refDir, "model.safetensors"), gemma4AssistantTinyWeights(false)); err != nil {
		t.Fatalf("SaveSafetensors: %v", err)
	}
	reference, err := LoadGemma4Assistant(refDir)
	if err != nil {
		t.Fatalf("LoadGemma4Assistant(reference): %v", err)
	}
	defer func() { _ = reference.Close() }()

	ggufWeights := map[string]*metal.Array{}
	for hf, arr := range gemma4AssistantTinyWeights(false) {
		ggufWeights[ggufNameForTest(t, hf)] = arr
	}
	ggufPath := core.JoinPath(t.TempDir(), "mtp-tiny.gguf")
	if err := metal.SaveGGUF(ggufPath, ggufWeights); err != nil {
		t.Fatalf("SaveGGUF: %v", err)
	}
	metal.Free(mapValues(ggufWeights)...)

	str, err := coreio.Local.Read(core.JoinPath(refDir, "config.json"))
	if err != nil {
		t.Fatalf("read fixture config: %v", err)
	}
	cfg, err := parseGemma4AssistantConfig([]byte(str))
	if err != nil {
		t.Fatalf("parse fixture config: %v", err)
	}
	tok, err := metal.LoadTokenizer(core.JoinPath(refDir, "tokenizer.json"))
	if err != nil {
		t.Fatalf("load tokenizer: %v", err)
	}
	raw, err := metal.LoadAllGGUF(ggufPath)
	if err != nil {
		t.Fatalf("LoadAllGGUF: %v", err)
	}
	loaded, err := buildGemma4AssistantFromGGUFTensors(cfg, raw, tok)
	if err != nil {
		t.Fatalf("buildGemma4AssistantFromGGUFTensors: %v", err)
	}
	defer func() { _ = loaded.Close() }()

	assertSameTensor := func(label string, a, b *metal.Array) {
		t.Helper()
		if a == nil || b == nil {
			t.Fatalf("%s: nil tensor (ref %v, gguf %v)", label, a != nil, b != nil)
		}
		av, bv := a.Floats(), b.Floats()
		if len(av) != len(bv) {
			t.Fatalf("%s: %d vs %d elements — shape drift through the gguf lane", label, len(av), len(bv))
		}
		for i := range av {
			if av[i] != bv[i] {
				t.Fatalf("%s: element %d = %g vs %g — value drift through the gguf lane", label, i, av[i], bv[i])
			}
		}
	}
	assertSameTensor("embed_tokens", reference.EmbedTokens.Weight, loaded.EmbedTokens.Weight)
	assertSameTensor("pre_projection", reference.PreProjection.Weight, loaded.PreProjection.Weight)
	assertSameTensor("layer0.q_proj", reference.Layers[0].Attention.QProj.Weight, loaded.Layers[0].Attention.QProj.Weight)
	assertSameTensor("layer1.down_proj", reference.Layers[1].MLP.DownProj.Weight, loaded.Layers[1].MLP.DownProj.Weight)
	assertSameTensor("layer0.layer_scalar", reference.Layers[0].LayerScalar, loaded.Layers[0].LayerScalar)
	assertSameTensor("norm", reference.Norm.Weight, loaded.Norm.Weight)
}

func mapValues(m map[string]*metal.Array) []*metal.Array {
	out := make([]*metal.Array, 0, len(m))
	for _, v := range m {
		out = append(out, v)
	}
	return out
}

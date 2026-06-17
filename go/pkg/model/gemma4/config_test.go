// SPDX-Licence-Identifier: EUPL-1.2

package gemma4

import (
	"reflect"
	"testing"

	core "dappco.re/go"
)

// TestConfigArchDense fills an Arch from a dense (non-MoE) config and checks every
// neutral dim, the gemma4-specifics, and that the per-layer specs equal DeriveLayers
// with no MoE flag set.
func TestConfigArchDense(t *testing.T) {
	c := Config{
		HiddenSize: 256, NumHiddenLayers: 4, IntermediateSize: 512,
		NumAttentionHeads: 8, NumKeyValueHeads: 2, HeadDim: 64,
		VocabSize: 1000, RMSNormEps: 1e-5, RopeTheta: 10000,
		FinalLogitSoftcapping: 30, SlidingWindow: 128, NumKVSharedLayers: 1,
		LayerTypes:             []string{"full_attention", "sliding_attention", "full_attention", "sliding_attention"},
		VocabSizePerLayerInput: 500, HiddenSizePerLayerInput: 64, AttentionKEqV: true,
	}
	a, err := c.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	want := Arch{
		Hidden: 256, Heads: 8, KVHeads: 2, HeadDim: 64, FF: 512, Vocab: 1000,
		Experts: 0, TopK: 0, ExpertFF: 0,
		Eps: 1e-5, RopeBase: 10000, RopeLocalBase: defaultRopeLocalTheta, RopeScale: 1, SoftCap: 30, SlidingWindow: 128,
		PerLayerInputVocab: 500, PerLayerInputHidden: 64, AttentionKEqV: true,
		Layer: DeriveLayers(c.LayerTypes, 1),
	}
	if !reflect.DeepEqual(a, want) {
		t.Fatalf("dense Arch mismatch:\n got %+v\nwant %+v", a, want)
	}
	for i, l := range a.Layer {
		if l.MoE {
			t.Fatalf("layer %d marked MoE in a dense config", i)
		}
	}
	t.Logf("dense Arch: all dims filled, %d layer specs ≡ DeriveLayers, no MoE", len(a.Layer))
}

// TestConfigArchMoE fills an Arch from a MoE config and checks the MoE dims plus that
// EVERY layer is marked MoE (gemma4 applies MoE uniformly, not interleaved).
func TestConfigArchMoE(t *testing.T) {
	c := Config{
		HiddenSize: 512, NumHiddenLayers: 3, IntermediateSize: 1024,
		NumAttentionHeads: 8, NumKeyValueHeads: 4, HeadDim: 64, VocabSize: 2000,
		LayerTypes:     []string{"full_attention", "full_attention", "sliding_attention"},
		EnableMoEBlock: true, NumExperts: 16, TopKExperts: 4, MoEIntermediateSize: 384,
	}
	a, err := c.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	if a.Experts != 16 || a.TopK != 4 || a.ExpertFF != 384 {
		t.Fatalf("MoE dims: got Experts=%d TopK=%d ExpertFF=%d, want 16/4/384", a.Experts, a.TopK, a.ExpertFF)
	}
	wantLayers := DeriveLayers(c.LayerTypes, 0)
	for i := range wantLayers {
		wantLayers[i].MoE = true
	}
	if !reflect.DeepEqual(a.Layer, wantLayers) {
		t.Fatalf("MoE layer specs mismatch:\n got %+v\nwant %+v", a.Layer, wantLayers)
	}
	t.Logf("MoE Arch: Experts=%d TopK=%d ExpertFF=%d, all %d layers MoE", a.Experts, a.TopK, a.ExpertFF, len(a.Layer))
}

// TestConfigArchDefaults checks the omitted-field defaults: head_dim ← hidden/heads,
// num_key_value_heads ← num_attention_heads, eps/rope ← gemma4 defaults, and absent
// layer_types ← all full_attention.
func TestConfigArchDefaults(t *testing.T) {
	c := Config{HiddenSize: 512, NumHiddenLayers: 2, IntermediateSize: 1024, NumAttentionHeads: 8, VocabSize: 100}
	a, err := c.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	if a.HeadDim != 64 {
		t.Fatalf("HeadDim default: got %d, want 64 (512/8)", a.HeadDim)
	}
	if a.KVHeads != 8 {
		t.Fatalf("KVHeads default: got %d, want 8 (= heads)", a.KVHeads)
	}
	if a.Eps != defaultRMSNormEps || a.RopeBase != defaultRopeTheta || a.RopeScale != 1 {
		t.Fatalf("defaults: eps=%v rope=%v scale=%v", a.Eps, a.RopeBase, a.RopeScale)
	}
	if len(a.Layer) != 2 || a.Layer[0].Attention != GlobalAttention || a.Layer[1].Attention != GlobalAttention {
		t.Fatalf("absent layer_types should default to 2 global layers, got %+v", a.Layer)
	}
	t.Logf("defaults: HeadDim %d, KVHeads %d, eps %v, rope %v, %d global layers", a.HeadDim, a.KVHeads, a.Eps, a.RopeBase, len(a.Layer))
}

// TestConfigUnmarshal proves the json tags: a config.json-shaped document unmarshals
// (via core.JSONUnmarshal, the loader's path) into Config and fills the Arch.
func TestConfigUnmarshal(t *testing.T) {
	js := `{
		"hidden_size": 640, "num_hidden_layers": 2, "intermediate_size": 2048,
		"num_attention_heads": 4, "num_key_value_heads": 1, "head_dim": 256,
		"vocab_size": 262144, "rms_norm_eps": 1e-6, "rope_theta": 1000000,
		"sliding_window": 512, "num_kv_shared_layers": 1,
		"layer_types": ["sliding_attention", "full_attention"],
		"hidden_size_per_layer_input": 256, "vocab_size_per_layer_input": 262144,
		"enable_moe_block": true, "num_experts": 8, "top_k_experts": 2, "moe_intermediate_size": 1024
	}`
	var c Config
	if r := core.JSONUnmarshal([]byte(js), &c); !r.OK {
		t.Fatalf("JSONUnmarshal failed")
	}
	a, err := c.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	if a.Hidden != 640 || a.Heads != 4 || a.KVHeads != 1 || a.HeadDim != 256 || a.FF != 2048 || a.Vocab != 262144 {
		t.Fatalf("unmarshalled dims wrong: %+v", a)
	}
	if a.Experts != 8 || a.TopK != 2 || a.ExpertFF != 1024 || !a.Layer[0].MoE {
		t.Fatalf("unmarshalled MoE wrong: Experts=%d TopK=%d ExpertFF=%d MoE0=%v", a.Experts, a.TopK, a.ExpertFF, a.Layer[0].MoE)
	}
	if a.SlidingWindow != 512 || a.PerLayerInputHidden != 256 || a.Layer[0].Attention != SlidingAttention {
		t.Fatalf("unmarshalled gemma4-specifics wrong: %+v", a)
	}
	t.Logf("json → Config → Arch: hidden %d, %d layers, MoE %dx top-%d, sliding %d", a.Hidden, len(a.Layer), a.Experts, a.TopK, a.SlidingWindow)
}

// TestConfigArchErrors checks the load-bearing validations reject malformed configs.
func TestConfigArchErrors(t *testing.T) {
	cases := []struct {
		name string
		c    Config
	}{
		{"zero hidden", Config{HiddenSize: 0, NumHiddenLayers: 2, NumAttentionHeads: 8}},
		{"heads not multiple of kv", Config{HiddenSize: 256, NumHiddenLayers: 2, NumAttentionHeads: 8, NumKeyValueHeads: 3, HeadDim: 32}},
		{"layer_types length mismatch", Config{HiddenSize: 256, NumHiddenLayers: 4, NumAttentionHeads: 8, HeadDim: 32, LayerTypes: []string{"full_attention", "full_attention", "full_attention"}}},
		{"moe without experts", Config{HiddenSize: 256, NumHiddenLayers: 2, NumAttentionHeads: 8, HeadDim: 32, EnableMoEBlock: true}},
		{"topK exceeds experts", Config{HiddenSize: 256, NumHiddenLayers: 2, NumAttentionHeads: 8, HeadDim: 32, EnableMoEBlock: true, NumExperts: 4, TopKExperts: 8}},
		{"head_dim absent, indivisible", Config{HiddenSize: 100, NumHiddenLayers: 2, NumAttentionHeads: 8}},
	}
	for _, tc := range cases {
		if _, err := tc.c.Arch(); err == nil {
			t.Fatalf("%s: expected an error, got nil", tc.name)
		}
	}
	t.Logf("validation: all %d malformed configs rejected", len(cases))
}

// TestConfigRope checks per-attention-type RoPE: defaults (global 1e6 / sliding 1e4),
// top-level rope_theta sets the global, and rope_parameters overrides both.
func TestConfigRope(t *testing.T) {
	base := Config{HiddenSize: 128, NumHiddenLayers: 1, NumAttentionHeads: 2, HeadDim: 64, VocabSize: 10}
	a, err := base.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	if a.RopeBase != defaultRopeTheta || a.RopeLocalBase != defaultRopeLocalTheta {
		t.Fatalf("defaults: RopeBase %v (want %v), RopeLocalBase %v (want %v)", a.RopeBase, defaultRopeTheta, a.RopeLocalBase, defaultRopeLocalTheta)
	}

	c := base
	c.RopeTheta = 500000
	a, _ = c.Arch()
	if a.RopeBase != 500000 || a.RopeLocalBase != defaultRopeLocalTheta {
		t.Fatalf("rope_theta: RopeBase %v (want 5e5), RopeLocalBase %v (want %v)", a.RopeBase, a.RopeLocalBase, defaultRopeLocalTheta)
	}

	c = base
	c.RopeParameters = map[string]RopeParam{
		"full_attention":    {RopeTheta: 2000000},
		"sliding_attention": {RopeTheta: 5000},
	}
	a, _ = c.Arch()
	if a.RopeBase != 2000000 || a.RopeLocalBase != 5000 {
		t.Fatalf("rope_parameters: RopeBase %v (want 2e6), RopeLocalBase %v (want 5e3)", a.RopeBase, a.RopeLocalBase)
	}
	t.Logf("rope: defaults 1e6/1e4, rope_theta sets global, rope_parameters overrides both (global %v, local %v)", a.RopeBase, a.RopeLocalBase)
}

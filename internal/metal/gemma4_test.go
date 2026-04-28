// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"os"
	"strings"
	"testing"

	"dappco.re/go"

	coreio "dappco.re/go/io"
)

func requireMetalRuntime(t *testing.T) {
	t.Helper()
	if os.Getenv("GO_MLX_RUN_METAL_TESTS") != "1" {
		t.Skip("set GO_MLX_RUN_METAL_TESTS=1 to enable Metal runtime tests")
	}
	if !MetalAvailable() {
		t.Skip("Metal runtime unavailable")
	}
}

func freeWeightMap(weights map[string]*Array) {
	for _, arr := range weights {
		Free(arr)
	}
}

func TestGemma4_ParseConfig_Defaults_Good(t *testing.T) {
	cfg, err := parseGemma4Config([]byte(`{
		"model_type": "gemma4_text",
		"hidden_size": 1024,
		"num_hidden_layers": 6,
		"intermediate_size": 2048,
		"num_attention_heads": 4,
		"num_key_value_heads": 1,
		"head_dim": 256
	}`))
	if err != nil {
		t.Fatalf("parseGemma4Config: %v", err)
	}
	if cfg.GlobalHeadDim != 512 {
		t.Errorf("GlobalHeadDim = %d, want 512", cfg.GlobalHeadDim)
	}
	if cfg.HiddenSizePerLayerInput != 256 {
		t.Errorf("HiddenSizePerLayerInput = %d, want 256", cfg.HiddenSizePerLayerInput)
	}
	if !cfg.UseDoubleWideMLP {
		t.Error("UseDoubleWideMLP = false, want true")
	}
	if !cfg.TieWordEmbeddings {
		t.Error("TieWordEmbeddings = false, want true")
	}
	if cfg.SlidingWindow != 512 {
		t.Errorf("SlidingWindow = %d, want 512", cfg.SlidingWindow)
	}
	if cfg.NumKVSharedLayers != 20 {
		t.Errorf("NumKVSharedLayers = %d, want 20", cfg.NumKVSharedLayers)
	}
	if cfg.FinalLogitSoftcapping != 30 {
		t.Errorf("FinalLogitSoftcapping = %f, want 30", cfg.FinalLogitSoftcapping)
	}
	if len(cfg.LayerTypes) != 6 {
		t.Fatalf("LayerTypes len = %d, want 6", len(cfg.LayerTypes))
	}
	want := []string{
		"sliding_attention",
		"sliding_attention",
		"sliding_attention",
		"sliding_attention",
		"full_attention",
		"sliding_attention",
	}
	for i, got := range cfg.LayerTypes {
		if got != want[i] {
			t.Fatalf("LayerTypes[%d] = %q, want %q", i, got, want[i])
		}
	}
	if cfg.RopeParameters["full_attention"].RopeType != "proportional" {
		t.Errorf("full attention rope type = %q, want proportional", cfg.RopeParameters["full_attention"].RopeType)
	}
	if cfg.RopeParameters["sliding_attention"].RopeTheta != 10000 {
		t.Errorf("sliding attention rope theta = %f, want 10000", cfg.RopeParameters["sliding_attention"].RopeTheta)
	}
}

func TestGemma4_ParseConfig_ExplicitZeroSharedKV_Good(t *testing.T) {
	cfg, err := parseGemma4Config([]byte(`{
		"model_type": "gemma4_text",
		"hidden_size": 1024,
		"num_hidden_layers": 6,
		"intermediate_size": 2048,
		"num_attention_heads": 4,
		"num_key_value_heads": 1,
		"head_dim": 256,
		"num_kv_shared_layers": 0
	}`))
	if err != nil {
		t.Fatalf("parseGemma4Config: %v", err)
	}
	if cfg.NumKVSharedLayers != 0 {
		t.Fatalf("NumKVSharedLayers = %d, want 0", cfg.NumKVSharedLayers)
	}
}

func TestGemma4_ParseConfig_NegativeDimensions_Bad(t *testing.T) {
	_, err := parseGemma4Config([]byte(`{
		"model_type": "gemma4_text",
		"hidden_size": 1024,
		"num_hidden_layers": -1,
		"intermediate_size": 2048,
		"num_attention_heads": 4,
		"num_key_value_heads": 1,
		"head_dim": 256
	}`))
	if err == nil {
		t.Fatal("parseGemma4Config succeeded, want error")
	}
	if !strings.Contains(err.Error(), "negative num_hidden_layers") {
		t.Fatalf("parseGemma4Config error = %v, want negative num_hidden_layers", err)
	}
}

func TestGemma4_ParseConfig_VisionConfig_Good(t *testing.T) {
	cfg, err := parseGemma4Config([]byte(`{
		"model_type": "gemma4",
		"image_token_id": 258880,
		"text_config": {
			"model_type": "gemma4_text",
			"pad_token_id": 0,
			"hidden_size": 1024,
			"num_hidden_layers": 2,
			"intermediate_size": 2048,
			"num_attention_heads": 4,
			"num_key_value_heads": 1,
			"head_dim": 256
		},
		"vision_config": {
			"model_type": "gemma4_vision",
			"hidden_size": 48,
			"intermediate_size": 96,
			"num_hidden_layers": 3,
			"num_attention_heads": 4,
			"num_key_value_heads": 4,
			"patch_size": 8,
			"pooling_kernel_size": 2,
			"position_embedding_size": 32,
			"rope_parameters": {
				"rope_type": "default",
				"rope_theta": 100
			}
		}
	}`))
	if err != nil {
		t.Fatalf("parseGemma4Config: %v", err)
	}
	if cfg.ImageTokenID != 258880 {
		t.Fatalf("ImageTokenID = %d, want 258880", cfg.ImageTokenID)
	}
	if cfg.VisionConfig == nil {
		t.Fatal("VisionConfig = nil, want parsed vision config")
	}
	if cfg.VisionConfig.HiddenSize != 48 {
		t.Fatalf("VisionConfig.HiddenSize = %d, want 48", cfg.VisionConfig.HiddenSize)
	}
	if cfg.VisionConfig.HeadDim != 12 {
		t.Fatalf("VisionConfig.HeadDim = %d, want inferred 12", cfg.VisionConfig.HeadDim)
	}
	if cfg.VisionConfig.RMSNormEps == 0 {
		t.Fatal("VisionConfig.RMSNormEps = 0, want default")
	}
}

func TestGemma4_ParseConfig_PartialRopeParameters_Good(t *testing.T) {
	cfg, err := parseGemma4Config([]byte(`{
		"model_type": "gemma4_text",
		"hidden_size": 1024,
		"num_hidden_layers": 6,
		"intermediate_size": 2048,
		"num_attention_heads": 4,
		"num_key_value_heads": 1,
		"head_dim": 256,
		"rope_parameters": {
			"full_attention": {
				"rope_theta": 123456
			}
		}
	}`))
	if err != nil {
		t.Fatalf("parseGemma4Config: %v", err)
	}
	full := cfg.RopeParameters["full_attention"]
	if full.RopeTheta != 123456 {
		t.Fatalf("full rope theta = %f, want 123456", full.RopeTheta)
	}
	if full.PartialRotaryFactor != 0.25 {
		t.Fatalf("full partial rotary factor = %f, want 0.25", full.PartialRotaryFactor)
	}
	if full.RopeType != "proportional" {
		t.Fatalf("full rope type = %q, want proportional", full.RopeType)
	}
	if full.Factor != 1.0 {
		t.Fatalf("full factor = %f, want 1.0", full.Factor)
	}

	sliding := cfg.RopeParameters["sliding_attention"]
	if sliding.RopeTheta != 10000 {
		t.Fatalf("sliding rope theta = %f, want 10000", sliding.RopeTheta)
	}
	if sliding.PartialRotaryFactor != 1.0 {
		t.Fatalf("sliding partial rotary factor = %f, want 1.0", sliding.PartialRotaryFactor)
	}
	if sliding.RopeType != "default" {
		t.Fatalf("sliding rope type = %q, want default", sliding.RopeType)
	}
}

func TestGemma4_ParseConfig_MoEDefaults_Good(t *testing.T) {
	cfg, err := parseGemma4Config([]byte(`{
		"model_type": "gemma4_text",
		"hidden_size": 1024,
		"num_hidden_layers": 2,
		"intermediate_size": 2048,
		"num_attention_heads": 4,
		"num_key_value_heads": 1,
		"head_dim": 256,
		"enable_moe_block": true
	}`))
	if err != nil {
		t.Fatalf("parseGemma4Config: %v", err)
	}
	if cfg.NumExperts == nil || *cfg.NumExperts != 128 {
		t.Fatalf("NumExperts = %v, want 128", cfg.NumExperts)
	}
	if cfg.TopKExperts == nil || *cfg.TopKExperts != 8 {
		t.Fatalf("TopKExperts = %v, want 8", cfg.TopKExperts)
	}
}

func TestGemma4_ParseConfig_NestedQuantization_Good(t *testing.T) {
	cfg, err := parseGemma4Config([]byte(`{
		"model_type": "gemma4",
		"text_config": {
			"hidden_size": 1024,
			"num_hidden_layers": 2,
			"intermediate_size": 2048,
			"num_attention_heads": 4,
			"num_key_value_heads": 1,
			"head_dim": 256,
			"layer_types": ["sliding_attention", "full_attention"],
			"quantization": {"group_size": 64, "bits": 4}
		}
	}`))
	if err != nil {
		t.Fatalf("parseGemma4Config: %v", err)
	}
	if cfg.ModelType != "gemma4" {
		t.Fatalf("ModelType = %q, want gemma4", cfg.ModelType)
	}
	if cfg.Quantization == nil || cfg.Quantization.GroupSize != 64 || cfg.Quantization.Bits != 4 {
		t.Fatalf("Quantization = %+v, want group_size=64 bits=4", cfg.Quantization)
	}
	if got := cfg.LayerTypes; len(got) != 2 || got[0] != "sliding_attention" || got[1] != "full_attention" {
		t.Fatalf("LayerTypes = %v, want explicit nested layer types", got)
	}
}

func TestGemma4_ParseConfig_NestedTopLevelOverrides_Good(t *testing.T) {
	cfg, err := parseGemma4Config([]byte(`{
		"model_type": "gemma4_text",
		"num_kv_shared_layers": 7,
		"global_head_dim": 384,
		"hidden_size_per_layer_input": 128,
		"use_double_wide_mlp": true,
		"tie_word_embeddings": true,
		"text_config": {
			"hidden_size": 1024,
			"num_hidden_layers": 6,
			"intermediate_size": 2048,
			"num_attention_heads": 4,
			"num_key_value_heads": 1,
			"head_dim": 256,
			"layer_types": [
				"sliding_attention",
				"sliding_attention",
				"sliding_attention",
				"sliding_attention",
				"full_attention",
				"sliding_attention"
			]
		}
	}`))
	if err != nil {
		t.Fatalf("parseGemma4Config: %v", err)
	}
	if cfg.NumKVSharedLayers != 7 {
		t.Fatalf("NumKVSharedLayers = %d, want 7", cfg.NumKVSharedLayers)
	}
	if cfg.GlobalHeadDim != 384 {
		t.Fatalf("GlobalHeadDim = %d, want 384", cfg.GlobalHeadDim)
	}
	if cfg.HiddenSizePerLayerInput != 128 {
		t.Fatalf("HiddenSizePerLayerInput = %d, want 128", cfg.HiddenSizePerLayerInput)
	}
	if !cfg.UseDoubleWideMLP {
		t.Fatal("UseDoubleWideMLP = false, want true")
	}
	if !cfg.TieWordEmbeddings {
		t.Fatal("TieWordEmbeddings = false, want true")
	}
}

func TestGemma4_ParseConfig_NestedTopLevelGemma4Fields_Good(t *testing.T) {
	cfg, err := parseGemma4Config([]byte(`{
		"model_type": "gemma4",
		"attention_k_eq_v": true,
		"num_global_key_value_heads": 2,
		"enable_moe_block": true,
		"num_experts": 64,
		"top_k_experts": 4,
		"moe_intermediate_size": 4096,
		"sliding_window": 256,
		"final_logit_softcapping": 12.5,
		"rope_parameters": {
			"full_attention": {
				"partial_rotary_factor": 0.125,
				"rope_theta": 424242,
				"rope_type": "proportional"
			}
		},
		"text_config": {
			"hidden_size": 1024,
			"num_hidden_layers": 2,
			"intermediate_size": 2048,
			"num_attention_heads": 4,
			"num_key_value_heads": 1,
			"head_dim": 256,
			"layer_types": ["sliding_attention", "full_attention"]
		}
	}`))
	if err != nil {
		t.Fatalf("parseGemma4Config: %v", err)
	}
	if cfg.ModelType != "gemma4" {
		t.Fatalf("ModelType = %q, want gemma4", cfg.ModelType)
	}
	if !cfg.AttentionKEqV {
		t.Fatal("AttentionKEqV = false, want true")
	}
	if cfg.NumGlobalKeyValueHeads == nil || *cfg.NumGlobalKeyValueHeads != 2 {
		t.Fatalf("NumGlobalKeyValueHeads = %v, want 2", cfg.NumGlobalKeyValueHeads)
	}
	if !cfg.EnableMoEBlock {
		t.Fatal("EnableMoEBlock = false, want true")
	}
	if cfg.NumExperts == nil || *cfg.NumExperts != 64 {
		t.Fatalf("NumExperts = %v, want 64", cfg.NumExperts)
	}
	if cfg.TopKExperts == nil || *cfg.TopKExperts != 4 {
		t.Fatalf("TopKExperts = %v, want 4", cfg.TopKExperts)
	}
	if cfg.MoEIntermediateSize == nil || *cfg.MoEIntermediateSize != 4096 {
		t.Fatalf("MoEIntermediateSize = %v, want 4096", cfg.MoEIntermediateSize)
	}
	if cfg.SlidingWindow != 256 {
		t.Fatalf("SlidingWindow = %d, want 256", cfg.SlidingWindow)
	}
	if cfg.FinalLogitSoftcapping != 12.5 {
		t.Fatalf("FinalLogitSoftcapping = %f, want 12.5", cfg.FinalLogitSoftcapping)
	}
	full := cfg.RopeParameters["full_attention"]
	if full.RopeTheta != 424242 {
		t.Fatalf("full rope theta = %f, want 424242", full.RopeTheta)
	}
	if full.PartialRotaryFactor != 0.125 {
		t.Fatalf("full partial rotary factor = %f, want 0.125", full.PartialRotaryFactor)
	}
	if full.RopeType != "proportional" {
		t.Fatalf("full rope type = %q, want proportional", full.RopeType)
	}
}

func TestGemma4_ParseConfig_NestedTopLevelFalseOverrides_Good(t *testing.T) {
	cfg, err := parseGemma4Config([]byte(`{
		"model_type": "gemma4",
		"attention_k_eq_v": false,
		"enable_moe_block": false,
		"use_double_wide_mlp": false,
		"tie_word_embeddings": false,
		"text_config": {
			"model_type": "gemma4_text",
			"hidden_size": 1024,
			"num_hidden_layers": 2,
			"intermediate_size": 2048,
			"num_attention_heads": 4,
			"num_key_value_heads": 1,
			"head_dim": 256,
			"attention_k_eq_v": true,
			"enable_moe_block": true,
			"use_double_wide_mlp": true,
			"tie_word_embeddings": true
		}
	}`))
	if err != nil {
		t.Fatalf("parseGemma4Config: %v", err)
	}
	if cfg.AttentionKEqV {
		t.Fatal("AttentionKEqV = true, want false")
	}
	if cfg.EnableMoEBlock {
		t.Fatal("EnableMoEBlock = true, want false")
	}
	if cfg.UseDoubleWideMLP {
		t.Fatal("UseDoubleWideMLP = true, want false")
	}
	if cfg.TieWordEmbeddings {
		t.Fatal("TieWordEmbeddings = true, want false")
	}
}

func TestGemma4_ParseConfig_NestedTopLevelNumericOverrides_Good(t *testing.T) {
	cfg, err := parseGemma4Config([]byte(`{
		"model_type": "gemma4",
		"num_global_key_value_heads": 2,
		"global_head_dim": 384,
		"global_partial_rotary_factor": 0.125,
		"sliding_window": 256,
		"final_logit_softcapping": 12.5,
		"rope_parameters": {
			"full_attention": {
				"rope_theta": 424242
			}
		},
		"text_config": {
			"model_type": "gemma4_text",
			"hidden_size": 1024,
			"num_hidden_layers": 2,
			"intermediate_size": 2048,
			"num_attention_heads": 4,
			"num_key_value_heads": 1,
			"num_global_key_value_heads": 4,
			"head_dim": 256,
			"global_head_dim": 768,
			"global_partial_rotary_factor": 0.5,
			"sliding_window": 128,
			"final_logit_softcapping": 30,
			"rope_parameters": {
				"full_attention": {
					"rope_theta": 111111,
					"rope_type": "proportional"
				}
			}
		}
	}`))
	if err != nil {
		t.Fatalf("parseGemma4Config: %v", err)
	}
	if cfg.NumGlobalKeyValueHeads == nil || *cfg.NumGlobalKeyValueHeads != 2 {
		t.Fatalf("NumGlobalKeyValueHeads = %v, want 2", cfg.NumGlobalKeyValueHeads)
	}
	if cfg.GlobalHeadDim != 384 {
		t.Fatalf("GlobalHeadDim = %d, want 384", cfg.GlobalHeadDim)
	}
	if cfg.GlobalPartialRotaryFactor != 0.125 {
		t.Fatalf("GlobalPartialRotaryFactor = %f, want 0.125", cfg.GlobalPartialRotaryFactor)
	}
	if cfg.SlidingWindow != 256 {
		t.Fatalf("SlidingWindow = %d, want 256", cfg.SlidingWindow)
	}
	if cfg.FinalLogitSoftcapping != 12.5 {
		t.Fatalf("FinalLogitSoftcapping = %f, want 12.5", cfg.FinalLogitSoftcapping)
	}
	full := cfg.RopeParameters["full_attention"]
	if full.RopeTheta != 424242 {
		t.Fatalf("full rope theta = %f, want 424242", full.RopeTheta)
	}
	if full.RopeType != "proportional" {
		t.Fatalf("full rope type = %q, want proportional", full.RopeType)
	}
}

func TestGemma4_InferPerLayerInputSize_StructuredEmbedding_Good(t *testing.T) {
	requireMetalRuntime(t)

	embed := seqArray(0.10, 10, 3, 4)
	defer Free(embed)

	got := inferGemma4PerLayerInputSize(map[string]*Array{
		"model.embed_tokens_per_layer.weight": embed,
	}, 3)
	if got != 4 {
		t.Fatalf("inferGemma4PerLayerInputSize() = %d, want 4", got)
	}
}

func TestGemma4_InferPerLayerInputSize_GatingFallback_Good(t *testing.T) {
	requireMetalRuntime(t)

	gate := seqArray(0.20, 6, 8)
	proj := seqArray(0.30, 8, 6)
	defer Free(gate, proj)

	got := inferGemma4PerLayerInputSize(map[string]*Array{
		"model.layers.0.per_layer_input_gate.weight": gate,
		"model.layers.0.per_layer_projection.weight": proj,
	}, 2)
	if got != 6 {
		t.Fatalf("inferGemma4PerLayerInputSize() = %d, want 6", got)
	}
}

func TestGemma4_NormalizePerLayerTensor_TransposedEmbedding_Good(t *testing.T) {
	requireMetalRuntime(t)

	input := FromValues([]float32{1, 2, 3, 4, 5, 6}, 1, 1, 2, 3)
	output := gemma4NormalizePerLayerTensor(input, 1, 1, 3, 2)
	if err := Eval(output); err != nil {
		t.Fatalf("Eval: %v", err)
	}
	defer Free(input, output)

	if got := output.Shape(); len(got) != 4 || got[0] != 1 || got[1] != 1 || got[2] != 3 || got[3] != 2 {
		t.Fatalf("normalized shape = %v, want [1 1 3 2]", got)
	}

	floatSliceApprox(t, output.Floats(), []float32{1, 4, 2, 5, 3, 6})
}

func TestGemma4_OutputLinear_TiedFallback_Good(t *testing.T) {
	embed := &Embedding{}
	output, err := gemma4OutputLinear(map[string]*Array{}, &Gemma4TextConfig{
		TieWordEmbeddings: true,
	}, embed)
	if err != nil {
		t.Fatalf("gemma4OutputLinear: %v", err)
	}
	if output == nil {
		t.Fatal("expected tied output linear")
	}
	if output.Weight != embed.Weight || output.Scales != embed.Scales || output.Biases != embed.Biases {
		t.Fatal("tied output should reuse embedding weights")
	}
}

func TestGemma4_OutputLinear_UntiedMissingLMHead_Bad(t *testing.T) {
	_, err := gemma4OutputLinear(map[string]*Array{}, &Gemma4TextConfig{}, &Embedding{})
	if err == nil {
		t.Fatal("expected error when untied Gemma4 model lacks lm_head.weight")
	}
	if !core.Contains(err.Error(), "lm_head.weight") {
		t.Fatalf("expected lm_head.weight error, got: %v", err)
	}
}

func TestGemma4_AttentionScale_Good(t *testing.T) {
	got := gemma4AttentionScale(512)
	if got != 1.0 {
		t.Fatalf("gemma4AttentionScale(512) = %f, want 1.0", got)
	}
}

func TestGemma4_SwitchLinear_PrefixFallback_Good(t *testing.T) {
	requireMetalRuntime(t)

	switchWeight := func(scale float32) *Array {
		return FromValues([]float32{
			scale, 0,
			0, scale,
		}, 1, 2, 2)
	}

	cases := []struct {
		name    string
		weights map[string]*Array
	}{
		{
			name: "rfc_switch_glu",
			weights: map[string]*Array{
				"model.layers.0.experts.switch_glu.gate_proj.weight": switchWeight(1.0),
			},
		},
		{
			name: "legacy_direct",
			weights: map[string]*Array{
				"model.layers.0.experts.gate_proj.weight": switchWeight(1.0),
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			layer := gemma4SwitchLinear(tc.weights, nil,
				"model.layers.0.experts.switch_glu.gate_proj",
				"model.layers.0.experts.gate_proj",
			)
			if layer == nil {
				t.Fatal("expected gemma4SwitchLinear to resolve the expert weight")
			}
			freeSwitchLinear(layer)
		})
	}
}

func TestGemma4_Linear_QuantizedWithoutConfig_Good(t *testing.T) {
	requireMetalRuntime(t)

	weight := seqArray(0.10, 2, 8)
	scales := seqArray(0.20, 2, 1)
	biases := seqArray(0.30, 2, 1)
	defer Free(weight, scales, biases)

	layer := gemma4Linear(map[string]*Array{
		"model.layers.0.self_attn.q_proj.weight": weight,
		"model.layers.0.self_attn.q_proj.scales": scales,
		"model.layers.0.self_attn.q_proj.biases": biases,
	}, "model.layers.0.self_attn.q_proj", nil)
	if layer == nil {
		t.Fatal("expected quantized layer")
	}
	defer freeLinear(layer)

	if layer.Scales != scales || layer.Biases != biases {
		t.Fatal("quantized Gemma4 layer should preserve scales/biases when config is absent")
	}
	if layer.GroupSize != 0 || layer.Bits != 0 {
		t.Fatalf("quantized Gemma4 layer should defer to MLX affine defaults, got group_size=%d bits=%d", layer.GroupSize, layer.Bits)
	}
}

func TestGemma4_SwitchLinear_QuantizedWithoutConfig_Good(t *testing.T) {
	requireMetalRuntime(t)

	weight := seqArray(0.10, 1, 2, 8)
	scales := seqArray(0.20, 1, 2, 1)
	biases := seqArray(0.30, 1, 2, 1)
	defer Free(weight, scales, biases)

	layer := gemma4SwitchLinear(map[string]*Array{
		"model.layers.0.experts.switch_glu.gate_proj.weight": weight,
		"model.layers.0.experts.switch_glu.gate_proj.scales": scales,
		"model.layers.0.experts.switch_glu.gate_proj.biases": biases,
	}, nil, "model.layers.0.experts.switch_glu.gate_proj")
	if layer == nil {
		t.Fatal("expected quantized switch layer")
	}
	defer freeSwitchLinear(layer)

	if layer.Scales != scales || layer.Biases != biases {
		t.Fatal("quantized Gemma4 switch layer should preserve scales/biases when config is absent")
	}
	if layer.GroupSize != 0 || layer.Bits != 0 {
		t.Fatalf("quantized Gemma4 switch layer should defer to MLX affine defaults, got group_size=%d bits=%d", layer.GroupSize, layer.Bits)
	}
}

func TestGemma4_QuantPredicate_RouterForces8Bit_Good(t *testing.T) {
	defaultQ := &QuantizationConfig{GroupSize: 128, Bits: 4}

	routerQ := gemma4QuantPredicate("model.layers.0.router.proj", defaultQ)
	if routerQ == nil {
		t.Fatal("router quantization predicate returned nil")
	}
	if routerQ.GroupSize != 64 || routerQ.Bits != 8 {
		t.Fatalf("router quantization = %+v, want group_size=64 bits=8", routerQ)
	}

	mlpQ := gemma4QuantPredicate("model.layers.0.mlp.gate_proj", defaultQ)
	if mlpQ != defaultQ {
		t.Fatalf("non-router quantization should preserve default config pointer, got %+v want %+v", mlpQ, defaultQ)
	}
}

func TestGemma4_SanitizeWeights_GateUpProj_Good(t *testing.T) {
	requireMetalRuntime(t)

	gateUp := FromValues([]float32{
		1, 2,
		3, 4,
		5, 6,
		7, 8,
	}, 1, 4, 2)
	Materialize(gateUp)
	vision := FromValues([]float32{1}, 1)
	rotary := FromValues([]float32{1}, 1)

	sanitized := sanitizeGemma4Weights(map[string]*Array{
		"model.layers.0.experts.gate_up_proj.weight": gateUp,
		"model.vision_tower.block.weight":            vision,
		"model.layers.0.self_attn.rotary_emb.inv":    rotary,
	})

	gate := sanitized["model.layers.0.experts.switch_glu.gate_proj.weight"]
	up := sanitized["model.layers.0.experts.switch_glu.up_proj.weight"]
	if gate == nil || up == nil {
		t.Fatal("expected split switch_glu gate_proj and up_proj weights")
	}
	if _, ok := sanitized["model.layers.0.experts.gate_up_proj.weight"]; ok {
		t.Fatal("gate_up_proj should be replaced by split weights")
	}
	if _, ok := sanitized["model.layers.0.experts.gate_proj.weight"]; ok {
		t.Fatal("legacy direct gate_proj key should not be emitted during sanitization")
	}
	if _, ok := sanitized["model.layers.0.experts.up_proj.weight"]; ok {
		t.Fatal("legacy direct up_proj key should not be emitted during sanitization")
	}
	if _, ok := sanitized["model.vision_tower.block.weight"]; ok {
		t.Fatal("vision tower weights should be stripped")
	}
	if _, ok := sanitized["model.layers.0.self_attn.rotary_emb.inv"]; ok {
		t.Fatal("rotary embedding weights should be stripped")
	}
	if got := gate.Shape(); len(got) != 3 || got[1] != 2 {
		t.Fatalf("gate split shape = %v, want [1 2 2]", got)
	}
	if got := up.Shape(); len(got) != 3 || got[1] != 2 {
		t.Fatalf("up split shape = %v, want [1 2 2]", got)
	}
	if !gate.IsRowContiguous() {
		t.Fatal("gate split should be row-contiguous")
	}
	if !up.IsRowContiguous() {
		t.Fatal("up split should be row-contiguous")
	}
	if gateUp.Valid() {
		t.Fatal("gate_up source tensor should be freed after split sanitization")
	}
	if vision.Valid() {
		t.Fatal("vision tower tensor should be freed after sanitization")
	}
	if rotary.Valid() {
		t.Fatal("rotary embedding tensor should be freed after sanitization")
	}
}

func TestGemma4_SanitizeWeights_GateUpProjBias2D_Good(t *testing.T) {
	requireMetalRuntime(t)

	biases := FromValues([]float32{
		1, 2, 3, 4,
		5, 6, 7, 8,
	}, 2, 4)
	Materialize(biases)

	sanitized := sanitizeGemma4Weights(map[string]*Array{
		"model.layers.0.experts.gate_up_proj.biases": biases,
	})

	gate := sanitized["model.layers.0.experts.switch_glu.gate_proj.biases"]
	up := sanitized["model.layers.0.experts.switch_glu.up_proj.biases"]
	if gate == nil || up == nil {
		t.Fatal("expected split switch_glu gate_proj and up_proj biases")
	}
	if got := gate.Shape(); len(got) != 2 || got[0] != 2 || got[1] != 2 {
		t.Fatalf("gate bias split shape = %v, want [2 2]", got)
	}
	if got := up.Shape(); len(got) != 2 || got[0] != 2 || got[1] != 2 {
		t.Fatalf("up bias split shape = %v, want [2 2]", got)
	}
}

func TestGemma4_SanitizeWeights_DownProjRemap_Good(t *testing.T) {
	requireMetalRuntime(t)

	down := FromValues([]float32{
		1, 2,
		3, 4,
	}, 1, 2, 2)
	Materialize(down)

	sanitized := sanitizeGemma4Weights(map[string]*Array{
		"model.layers.0.experts.down_proj.weight": down,
	})

	remapped := sanitized["model.layers.0.experts.switch_glu.down_proj.weight"]
	if remapped == nil {
		t.Fatal("expected down_proj to be remapped to switch_glu.down_proj")
	}
	if remapped != down {
		t.Fatal("down_proj remap should retain the original tensor")
	}
	if _, ok := sanitized["model.layers.0.experts.down_proj.weight"]; ok {
		t.Fatal("legacy direct down_proj key should not be emitted during sanitization")
	}
	if !down.Valid() {
		t.Fatal("down_proj tensor should be retained after key remap")
	}
	Free(down)
}

func TestGemma4_SanitizeWeights_LanguageModelPrefix_Good(t *testing.T) {
	sanitized := sanitizeGemma4Weights(map[string]*Array{
		"language_model.model.embed_tokens.weight":       nil,
		"language_model.model.norm.weight":               nil,
		"language_model.model.vision_tower.block.weight": nil,
		"language_model.multi_modal_projector.weight":    nil,
	})

	if _, ok := sanitized["model.embed_tokens.weight"]; !ok {
		t.Fatal("expected embed_tokens weight to be normalised to model.*")
	}
	if _, ok := sanitized["model.norm.weight"]; !ok {
		t.Fatal("expected norm weight to be normalised to model.*")
	}
	if _, ok := sanitized["language_model.model.embed_tokens.weight"]; ok {
		t.Fatal("expected language_model.model prefix to be stripped")
	}
	if _, ok := sanitized["language_model.model.vision_tower.block.weight"]; ok {
		t.Fatal("vision tower weights should be stripped even under language_model.model")
	}
	if _, ok := sanitized["language_model.multi_modal_projector.weight"]; ok {
		t.Fatal("multimodal projector weights should be stripped even under language_model")
	}
}

func TestGemma4_SanitizeVisionWeights_Good(t *testing.T) {
	raw := map[string]*Array{
		"language_model.model.vision_tower.patch_embedder.input_proj.weight": nil,
		"language_model.embed_vision.embedding_projection.weight":            nil,
		"language_model.model.embed_tokens.weight":                           nil,
	}

	vision := sanitizeGemma4VisionWeights(raw)
	if _, ok := vision["patch_embedder.input_proj.weight"]; !ok {
		t.Fatal("expected vision tower prefix to be stripped")
	}
	if _, ok := vision["embed_vision.embedding_projection.weight"]; !ok {
		t.Fatal("expected embed_vision projector weight to be retained")
	}
	if _, ok := raw["language_model.model.vision_tower.patch_embedder.input_proj.weight"]; ok {
		t.Fatal("expected vision weight to be removed from raw map")
	}
	if _, ok := raw["language_model.embed_vision.embedding_projection.weight"]; ok {
		t.Fatal("expected projector weight to be removed from raw map")
	}
	if _, ok := raw["language_model.model.embed_tokens.weight"]; !ok {
		t.Fatal("expected text weight to remain in raw map")
	}
}

func TestGemma4_SanitizeWeights_RepeatedWrapperPrefixes_Good(t *testing.T) {
	sanitized := sanitizeGemma4Weights(map[string]*Array{
		"model.model.embed_tokens.weight":                        nil,
		"language_model.model.model.norm.weight":                 nil,
		"model.language_model.model.model.vision_tower.block.w":  nil,
		"language_model.model.model.audio_tower.encoder.weight":  nil,
		"model.model.layers.0.self_attn.rotary_emb.inv_freq":     nil,
		"model.language_model.model.model.layers.0.layer_scalar": nil,
	})

	if _, ok := sanitized["model.embed_tokens.weight"]; !ok {
		t.Fatal("expected nested model.model prefix to collapse to model.*")
	}
	if _, ok := sanitized["model.norm.weight"]; !ok {
		t.Fatal("expected repeated language_model.model prefixes to collapse to model.*")
	}
	if _, ok := sanitized["model.layers.0.layer_scalar"]; !ok {
		t.Fatal("expected repeated wrapper prefixes on layer weights to collapse to model.*")
	}
	if _, ok := sanitized["model.model.embed_tokens.weight"]; ok {
		t.Fatal("expected model.model prefix to be stripped")
	}
	if _, ok := sanitized["language_model.model.model.norm.weight"]; ok {
		t.Fatal("expected repeated language_model.model prefixes to be stripped")
	}
	if _, ok := sanitized["model.language_model.model.model.vision_tower.block.w"]; ok {
		t.Fatal("vision tower weights should be stripped even under repeated wrapper prefixes")
	}
	if _, ok := sanitized["language_model.model.model.audio_tower.encoder.weight"]; ok {
		t.Fatal("audio tower weights should be stripped even under repeated wrapper prefixes")
	}
	if _, ok := sanitized["model.model.layers.0.self_attn.rotary_emb.inv_freq"]; ok {
		t.Fatal("rotary embedding weights should be stripped even under repeated wrapper prefixes")
	}
}

func TestGemma4_BuildPreviousKVs_Good(t *testing.T) {
	layers := []*Gemma4DecoderLayer{
		{LayerType: "sliding_attention"},
		{LayerType: "full_attention"},
		{LayerType: "sliding_attention"},
		{LayerType: "full_attention"},
	}
	got := buildGemma4PreviousKVs(layers, 2)
	want := []int32{0, 1, 0, 1}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("PreviousKVs[%d] = %d, want %d", i, got[i], want[i])
		}
	}
}

func TestGemma4_BuildCacheLayout_PromotesMissingOwner_Good(t *testing.T) {
	layers := []*Gemma4DecoderLayer{
		{LayerType: "sliding_attention"},
		{LayerType: "sliding_attention"},
		{LayerType: "sliding_attention"},
		{LayerType: "sliding_attention"},
		{LayerType: "full_attention"},
		{LayerType: "sliding_attention"},
	}

	previous, cacheIndexByLayer := buildGemma4CacheLayout(layers, 2)

	wantPrevious := []int32{0, 1, 2, 3, 4, 3}
	for i, want := range wantPrevious {
		if previous[i] != want {
			t.Fatalf("PreviousKVs[%d] = %d, want %d", i, previous[i], want)
		}
	}

	wantCacheIndex := []int32{0, 1, 2, 3, 4, -1}
	for i, want := range wantCacheIndex {
		if cacheIndexByLayer[i] != want {
			t.Fatalf("CacheIndexByLayer[%d] = %d, want %d", i, cacheIndexByLayer[i], want)
		}
	}
}

func TestGemma4_NewCache_SharedLayers_Good(t *testing.T) {
	model := &Gemma4Model{
		Cfg: &Gemma4TextConfig{
			NumHiddenLayers:   4,
			NumKVSharedLayers: 2,
			SlidingWindow:     32,
		},
		Layers: []*Gemma4DecoderLayer{
			{LayerType: "sliding_attention"},
			{LayerType: "full_attention"},
			{LayerType: "sliding_attention"},
			{LayerType: "full_attention"},
		},
	}
	caches := model.NewCache()
	if len(caches) != 2 {
		t.Fatalf("len(caches) = %d, want 2", len(caches))
	}
	if _, ok := caches[0].(*RotatingKVCache); !ok {
		t.Fatalf("cache[0] = %T, want *RotatingKVCache", caches[0])
	}
	if _, ok := caches[1].(*KVCache); !ok {
		t.Fatalf("cache[1] = %T, want *KVCache", caches[1])
	}
}

func TestGemma4_NewCache_PromotedOwner_Good(t *testing.T) {
	model := &Gemma4Model{
		Cfg: &Gemma4TextConfig{
			NumHiddenLayers:   6,
			NumKVSharedLayers: 2,
			SlidingWindow:     32,
		},
		Layers: []*Gemma4DecoderLayer{
			{LayerType: "sliding_attention"},
			{LayerType: "sliding_attention"},
			{LayerType: "sliding_attention"},
			{LayerType: "sliding_attention"},
			{LayerType: "full_attention"},
			{LayerType: "sliding_attention"},
		},
	}

	caches := model.NewCache()
	if len(caches) != 5 {
		t.Fatalf("len(caches) = %d, want 5", len(caches))
	}
	if _, ok := caches[4].(*KVCache); !ok {
		t.Fatalf("cache[4] = %T, want *KVCache for promoted full-attention owner", caches[4])
	}
	if got := model.PreviousKVs[4]; got != 4 {
		t.Fatalf("PreviousKVs[4] = %d, want 4", got)
	}
	if got := model.CacheIndexByLayer[4]; got != 4 {
		t.Fatalf("CacheIndexByLayer[4] = %d, want 4", got)
	}
}

func TestGemma4_LoadModel_Dispatch_Good(t *testing.T) {
	dir := t.TempDir()
	_ = coreio.Local.Write(core.JoinPath(dir, "config.json"), `{
		"model_type": "gemma4_text",
		"hidden_size": 8,
		"num_hidden_layers": 1,
		"intermediate_size": 16,
		"num_attention_heads": 1,
		"num_key_value_heads": 1,
		"head_dim": 4,
		"hidden_size_per_layer_input": 0
	}`)

	_, err := loadModel(dir)
	if err == nil {
		t.Fatal("expected tokenizer error, proving dispatch reached Gemma4 loader")
	}
	if !core.Contains(err.Error(), "tokenizer") && !core.Contains(err.Error(), "gemma4") {
		t.Fatalf("expected gemma4 loader error, got: %v", err)
	}
}

func TestGemma4_LoadAndForwardDenseModel_Good(t *testing.T) {
	requireMetalRuntime(t)

	dir := t.TempDir()
	config := `{
		"model_type": "gemma4_text",
		"hidden_size": 8,
		"num_hidden_layers": 2,
		"intermediate_size": 16,
		"num_attention_heads": 1,
		"num_key_value_heads": 1,
		"head_dim": 4,
		"global_head_dim": 8,
		"vocab_size": 10,
		"rms_norm_eps": 1e-6,
		"sliding_window": 4,
		"sliding_window_pattern": 2,
		"num_kv_shared_layers": 0,
		"hidden_size_per_layer_input": 0,
		"layer_types": ["sliding_attention", "full_attention"]
	}`
	if err := coreio.Local.Write(core.JoinPath(dir, "config.json"), config); err != nil {
		t.Fatalf("write config.json: %v", err)
	}
	writeMinimalTokenizer(t, dir)
	if err := SaveSafetensors(core.JoinPath(dir, "model.safetensors"), gemma4TinyWeights()); err != nil {
		t.Fatalf("SaveSafetensors: %v", err)
	}

	model, err := LoadGemma4(dir)
	if err != nil {
		t.Fatalf("LoadGemma4: %v", err)
	}
	defer closeGemma4(model)

	tokens := FromValues([]int32{2, 3, 4}, 1, 3)
	caches := model.NewCache()
	logits := model.Forward(tokens, caches)
	if err := Eval(logits); err != nil {
		t.Fatalf("Eval logits: %v", err)
	}
	defer func() {
		Free(tokens, logits)
		freeCaches(caches)
	}()

	shape := logits.Shape()
	if len(shape) != 3 {
		t.Fatalf("logits dims = %v, want rank 3", shape)
	}
	if shape[0] != 1 || shape[1] != 3 || shape[2] != 10 {
		t.Fatalf("logits shape = %v, want [1 3 10]", shape)
	}
}

func TestGemma4_LoadAndForwardDenseModel_LongSlidingPrompt_Good(t *testing.T) {
	requireMetalRuntime(t)

	dir := t.TempDir()
	config := `{
		"model_type": "gemma4_text",
		"hidden_size": 8,
		"num_hidden_layers": 2,
		"intermediate_size": 16,
		"num_attention_heads": 1,
		"num_key_value_heads": 1,
		"head_dim": 4,
		"global_head_dim": 8,
		"vocab_size": 10,
		"rms_norm_eps": 1e-6,
		"sliding_window": 2,
		"sliding_window_pattern": 2,
		"num_kv_shared_layers": 0,
		"hidden_size_per_layer_input": 0,
		"layer_types": ["sliding_attention", "full_attention"]
	}`
	if err := coreio.Local.Write(core.JoinPath(dir, "config.json"), config); err != nil {
		t.Fatalf("write config.json: %v", err)
	}
	writeMinimalTokenizer(t, dir)
	if err := SaveSafetensors(core.JoinPath(dir, "model.safetensors"), gemma4TinyWeights()); err != nil {
		t.Fatalf("SaveSafetensors: %v", err)
	}

	model, err := LoadGemma4(dir)
	if err != nil {
		t.Fatalf("LoadGemma4: %v", err)
	}
	defer closeGemma4(model)

	tokens := FromValues([]int32{2, 3, 4, 5}, 1, 4)
	caches := model.NewCache()
	logits := model.Forward(tokens, caches)
	if err := Eval(logits); err != nil {
		t.Fatalf("Eval logits: %v", err)
	}
	defer func() {
		Free(tokens, logits)
		freeCaches(caches)
	}()

	shape := logits.Shape()
	if len(shape) != 3 {
		t.Fatalf("logits dims = %v, want rank 3", shape)
	}
	if shape[0] != 1 || shape[1] != 4 || shape[2] != 10 {
		t.Fatalf("logits shape = %v, want [1 4 10]", shape)
	}
}

func TestGemma4_LoadAndForwardDenseModelFromGGUF_Good(t *testing.T) {
	requireMetalRuntime(t)

	dir := t.TempDir()
	config := `{
		"model_type": "gemma4_text",
		"hidden_size": 8,
		"num_hidden_layers": 2,
		"intermediate_size": 16,
		"num_attention_heads": 1,
		"num_key_value_heads": 1,
		"head_dim": 4,
		"global_head_dim": 8,
		"vocab_size": 10,
		"rms_norm_eps": 1e-6,
		"sliding_window": 4,
		"sliding_window_pattern": 2,
		"num_kv_shared_layers": 0,
		"hidden_size_per_layer_input": 0,
		"layer_types": ["sliding_attention", "full_attention"]
	}`
	if err := coreio.Local.Write(core.JoinPath(dir, "config.json"), config); err != nil {
		t.Fatalf("write config.json: %v", err)
	}
	writeMinimalTokenizer(t, dir)
	if err := SaveGGUF(core.JoinPath(dir, "model.gguf"), gemma4TinyWeights()); err != nil {
		t.Fatalf("SaveGGUF: %v", err)
	}

	model, err := LoadGemma4(core.JoinPath(dir, "model.gguf"))
	if err != nil {
		t.Fatalf("LoadGemma4: %v", err)
	}
	defer closeGemma4(model)

	tokens := FromValues([]int32{2, 3, 4}, 1, 3)
	caches := model.NewCache()
	logits := model.Forward(tokens, caches)
	if err := Eval(logits); err != nil {
		t.Fatalf("Eval logits: %v", err)
	}
	defer func() {
		Free(tokens, logits)
		freeCaches(caches)
	}()

	shape := logits.Shape()
	if len(shape) != 3 {
		t.Fatalf("logits dims = %v, want rank 3", shape)
	}
	if shape[0] != 1 || shape[1] != 3 || shape[2] != 10 {
		t.Fatalf("logits shape = %v, want [1 3 10]", shape)
	}
}

func TestGemma4_LoadAndForwardWrapperModel_Good(t *testing.T) {
	requireMetalRuntime(t)

	dir := t.TempDir()
	config := `{
		"model_type": "gemma4",
		"text_config": {
			"hidden_size": 8,
			"num_hidden_layers": 2,
			"intermediate_size": 16,
			"num_attention_heads": 1,
			"num_key_value_heads": 1,
			"head_dim": 4,
			"global_head_dim": 8,
			"vocab_size": 10,
			"rms_norm_eps": 1e-6,
			"sliding_window": 4,
			"sliding_window_pattern": 2,
			"num_kv_shared_layers": 0,
			"hidden_size_per_layer_input": 0,
			"layer_types": ["sliding_attention", "full_attention"]
		}
	}`
	if err := coreio.Local.Write(core.JoinPath(dir, "config.json"), config); err != nil {
		t.Fatalf("write config.json: %v", err)
	}
	writeMinimalTokenizer(t, dir)

	weights := gemma4TinyWeights()
	weights["vision_tower.encoder.weight"] = FromValues([]float32{1, 2, 3, 4}, 2, 2)
	weights["language_model.model.layers.0.self_attn.rotary_emb.inv_freq"] = FromValues([]float32{1, 2}, 2)
	defer Free(weights["vision_tower.encoder.weight"], weights["language_model.model.layers.0.self_attn.rotary_emb.inv_freq"])
	if err := SaveSafetensors(core.JoinPath(dir, "model.safetensors"), weights); err != nil {
		t.Fatalf("SaveSafetensors: %v", err)
	}

	model, err := LoadGemma4(dir)
	if err != nil {
		t.Fatalf("LoadGemma4: %v", err)
	}
	defer closeGemma4(model)

	if got := model.ModelType(); got != "gemma4" {
		t.Fatalf("ModelType() = %q, want gemma4", got)
	}

	tokens := FromValues([]int32{2, 3, 4}, 1, 3)
	caches := model.NewCache()
	logits := model.Forward(tokens, caches)
	if err := Eval(logits); err != nil {
		t.Fatalf("Eval logits: %v", err)
	}
	defer func() {
		Free(tokens, logits)
		freeCaches(caches)
	}()

	shape := logits.Shape()
	if len(shape) != 3 {
		t.Fatalf("logits dims = %v, want rank 3", shape)
	}
	if shape[0] != 1 || shape[1] != 3 || shape[2] != 10 {
		t.Fatalf("logits shape = %v, want [1 3 10]", shape)
	}
}

func TestGemma4_LoadModel_UntiedOutputFailureReleasesAllocatedWeights_Good(t *testing.T) {
	requireMetalRuntime(t)

	dir := t.TempDir()
	config := `{
		"model_type": "gemma4_text",
		"hidden_size": 8,
		"num_hidden_layers": 2,
		"intermediate_size": 16,
		"num_attention_heads": 1,
		"num_key_value_heads": 1,
		"head_dim": 4,
		"global_head_dim": 8,
		"vocab_size": 10,
		"rms_norm_eps": 1e-6,
		"sliding_window": 4,
		"sliding_window_pattern": 2,
		"num_kv_shared_layers": 0,
		"tie_word_embeddings": false,
		"layer_types": ["sliding_attention", "full_attention"]
	}`
	if err := coreio.Local.Write(core.JoinPath(dir, "config.json"), config); err != nil {
		t.Fatalf("write config.json: %v", err)
	}
	writeMinimalTokenizer(t, dir)

	weights := gemma4TinyWeights()
	if err := SaveSafetensors(core.JoinPath(dir, "model.safetensors"), weights); err != nil {
		t.Fatalf("SaveSafetensors: %v", err)
	}
	freeWeightMap(weights)
	ClearCache()

	baseline := GetActiveMemory()
	_, err := LoadGemma4(dir)
	if err == nil {
		t.Fatal("expected untied Gemma4 load to fail without lm_head.weight")
	}
	if !core.Contains(err.Error(), "lm_head.weight") {
		t.Fatalf("expected lm_head.weight error, got: %v", err)
	}

	activeAfterFailure := GetActiveMemory()
	if activeAfterFailure > baseline {
		t.Fatalf("active memory after failed load = %d, want <= %d", activeAfterFailure, baseline)
	}
}

func TestGemma4_DecoderLayer_MoEAppliesFinalPostFFNorm_Good(t *testing.T) {
	requireMetalRuntime(t)

	zeros2x2 := func() *Array {
		return FromValues([]float32{
			0, 0,
			0, 0,
		}, 2, 2)
	}
	ones2 := func() *Array {
		return FromValues([]float32{1, 1}, 2)
	}
	switchWeight := func(scale float32) *Array {
		return FromValues([]float32{
			scale, 0,
			0, scale,
		}, 1, 2, 2)
	}

	layer := &Gemma4DecoderLayer{
		Attention: &Gemma4Attention{
			QProj:          NewLinear(zeros2x2(), nil),
			KProj:          NewLinear(zeros2x2(), nil),
			VProj:          NewLinear(zeros2x2(), nil),
			OProj:          NewLinear(zeros2x2(), nil),
			QNormScaled:    ones2(),
			KNormScaled:    ones2(),
			HeadDim:        2,
			NKVHeads:       1,
			Scale:          1.0,
			RopeBase:       10000,
			RopeRotatedDim: 2,
		},
		MLP: &MLP{
			GateProj: NewLinear(FromValues([]float32{
				0.8, 0.1,
				0.2, 0.7,
			}, 2, 2), nil),
			UpProj: NewLinear(FromValues([]float32{
				0.5, -0.1,
				0.3, 0.6,
			}, 2, 2), nil),
			DownProj: NewLinear(FromValues([]float32{
				0.4, 0.2,
				-0.3, 0.9,
			}, 2, 2), nil),
		},
		EnableMoE:          true,
		InputNormScaled:    ones2(),
		PostAttnNormScaled: ones2(),
		PreFFNormScaled:    ones2(),
		PostFFNormScaled:   FromValues([]float32{2.0, 0.5}, 2),
		PreFFNorm2Scaled:   ones2(),
		PostFFNorm1Scaled:  ones2(),
		PostFFNorm2Scaled:  ones2(),
		Router: &Gemma4Router{
			Proj:           NewLinear(FromValues([]float32{1.0, -0.25}, 1, 2), nil),
			Scale:          ones2(),
			PerExpertScale: FromValues([]float32{1}, 1),
			ScaleScaled:    ones2(),
			TopK:           1,
			Eps:            1e-6,
		},
		Experts: &Gemma4Experts{
			GateProj: NewSwitchLinear(switchWeight(0.9), nil),
			UpProj:   NewSwitchLinear(switchWeight(0.6), nil),
			DownProj: NewSwitchLinear(switchWeight(0.7), nil),
		},
	}
	defer closeGemma4(&Gemma4Model{Layers: []*Gemma4DecoderLayer{layer}})

	cfg := &Gemma4TextConfig{
		HiddenSize:        2,
		NumAttentionHeads: 1,
		NumKeyValueHeads:  1,
		RMSNormEps:        1e-6,
	}
	x := FromValues([]float32{0.3, -0.2}, 1, 1, 2)

	got, kv := layer.forward(x, nil, 1, 1, nil, nil, sharedKV{}, cfg)
	defer Free(kv.Keys, kv.Values)

	h1In := RMSNorm(x, layer.PreFFNormScaled, cfg.RMSNormEps)
	h1 := layer.MLP.forward(h1In)
	Free(h1In)
	h1Normed := RMSNorm(h1, layer.PostFFNorm1Scaled, cfg.RMSNormEps)
	Free(h1)

	h2In := RMSNorm(x, layer.PreFFNorm2Scaled, cfg.RMSNormEps)
	topKIndices, topKWeights := layer.Router.forward(h2In)
	h2 := layer.Experts.forward(h2In, topKIndices, topKWeights)
	Free(h2In, topKIndices, topKWeights)
	h2Normed := RMSNorm(h2, layer.PostFFNorm2Scaled, cfg.RMSNormEps)
	Free(h2)

	combined := Add(h1Normed, h2Normed)
	Free(h1Normed, h2Normed)
	combinedNormed := RMSNorm(combined, layer.PostFFNormScaled, cfg.RMSNormEps)
	Free(combined)
	want := Add(x, combinedNormed)
	Free(combinedNormed)

	if err := Eval(got, want); err != nil {
		t.Fatalf("Eval: %v", err)
	}
	defer Free(x, got, want)

	floatSliceApprox(t, got.Floats(), want.Floats())
}

func TestGemma4_DecoderLayer_MoERouterUsesPreFFNorm2Input_Good(t *testing.T) {
	requireMetalRuntime(t)

	zeros2x2 := func() *Array {
		return FromValues([]float32{
			0, 0,
			0, 0,
		}, 2, 2)
	}
	ones2 := func() *Array {
		return FromValues([]float32{1, 1}, 2)
	}
	expertWeight := func(e0, e1 []float32) *Array {
		data := append(append([]float32{}, e0...), e1...)
		return FromValues(data, 2, 2, 2)
	}

	layer := &Gemma4DecoderLayer{
		Attention: &Gemma4Attention{
			QProj:          NewLinear(zeros2x2(), nil),
			KProj:          NewLinear(zeros2x2(), nil),
			VProj:          NewLinear(zeros2x2(), nil),
			OProj:          NewLinear(zeros2x2(), nil),
			QNormScaled:    ones2(),
			KNormScaled:    ones2(),
			HeadDim:        2,
			NKVHeads:       1,
			Scale:          1.0,
			RopeBase:       10000,
			RopeRotatedDim: 2,
		},
		MLP: &MLP{
			GateProj: NewLinear(zeros2x2(), nil),
			UpProj:   NewLinear(zeros2x2(), nil),
			DownProj: NewLinear(zeros2x2(), nil),
		},
		EnableMoE:          true,
		InputNormScaled:    ones2(),
		PostAttnNormScaled: ones2(),
		PreFFNormScaled:    ones2(),
		PostFFNormScaled:   ones2(),
		PreFFNorm2Scaled:   FromValues([]float32{0.1, 2.0}, 2),
		PostFFNorm1Scaled:  ones2(),
		PostFFNorm2Scaled:  ones2(),
		Router: &Gemma4Router{
			Proj: NewLinear(FromValues([]float32{
				1, -1,
				-1, 1,
			}, 2, 2), nil),
			Scale:          ones2(),
			PerExpertScale: FromValues([]float32{1, 1}, 2),
			ScaleScaled:    ones2(),
			TopK:           1,
			Eps:            1e-6,
		},
		Experts: &Gemma4Experts{
			GateProj: NewSwitchLinear(expertWeight(
				[]float32{1, 0, 0, 1},
				[]float32{1, 0, 0, 1},
			), nil),
			UpProj: NewSwitchLinear(expertWeight(
				[]float32{1, 0, 0, 1},
				[]float32{1, 0, 0, 1},
			), nil),
			DownProj: NewSwitchLinear(expertWeight(
				[]float32{1, 0, 0, 1},
				[]float32{-1, 0, 0, -1},
			), nil),
		},
	}
	defer closeGemma4(&Gemma4Model{Layers: []*Gemma4DecoderLayer{layer}})

	cfg := &Gemma4TextConfig{
		HiddenSize:        2,
		NumAttentionHeads: 1,
		NumKeyValueHeads:  1,
		RMSNormEps:        1e-6,
	}
	x := FromValues([]float32{2, 1}, 1, 1, 2)

	got, kv := layer.forward(x, nil, 1, 1, nil, nil, sharedKV{}, cfg)
	defer Free(kv.Keys, kv.Values)

	h2InForCheck := RMSNorm(x, layer.PreFFNorm2Scaled, cfg.RMSNormEps)
	residualIndices, residualWeights := layer.Router.forward(x)
	normedIndices, normedWeights := layer.Router.forward(h2InForCheck)
	if err := Eval(residualIndices, normedIndices); err != nil {
		t.Fatalf("Eval indices: %v", err)
	}
	if residualIndices.DataInt32()[0] == normedIndices.DataInt32()[0] {
		t.Fatal("expected residual-stream and pre-normalized router inputs to pick different experts")
	}
	Free(residualIndices, residualWeights)

	h1In := RMSNorm(x, layer.PreFFNormScaled, cfg.RMSNormEps)
	h1 := layer.MLP.forward(h1In)
	Free(h1In)
	h1Normed := RMSNorm(h1, layer.PostFFNorm1Scaled, cfg.RMSNormEps)
	Free(h1)

	h2 := layer.Experts.forward(h2InForCheck, normedIndices, normedWeights)
	Free(h2InForCheck, normedIndices, normedWeights)
	h2Normed := RMSNorm(h2, layer.PostFFNorm2Scaled, cfg.RMSNormEps)
	Free(h2)

	combined := Add(h1Normed, h2Normed)
	Free(h1Normed, h2Normed)
	combinedNormed := RMSNorm(combined, layer.PostFFNormScaled, cfg.RMSNormEps)
	Free(combined)
	want := Add(x, combinedNormed)
	Free(combinedNormed)

	if err := Eval(got, want); err != nil {
		t.Fatalf("Eval: %v", err)
	}
	defer Free(x, got, want)

	floatSliceApprox(t, got.Floats(), want.Floats())
}

func TestGemma4_LoadAndForwardPerLayerInputModel_Good(t *testing.T) {
	requireMetalRuntime(t)

	dir := t.TempDir()
	config := `{
		"model_type": "gemma4_text",
		"hidden_size": 8,
		"num_hidden_layers": 2,
		"intermediate_size": 16,
		"num_attention_heads": 1,
		"num_key_value_heads": 1,
		"head_dim": 4,
		"global_head_dim": 8,
		"vocab_size": 10,
		"vocab_size_per_layer_input": 10,
		"rms_norm_eps": 1e-6,
		"sliding_window": 4,
		"sliding_window_pattern": 2,
		"num_kv_shared_layers": 0,
		"layer_types": ["sliding_attention", "full_attention"]
	}`
	if err := coreio.Local.Write(core.JoinPath(dir, "config.json"), config); err != nil {
		t.Fatalf("write config.json: %v", err)
	}
	writeMinimalTokenizer(t, dir)
	if err := SaveSafetensors(core.JoinPath(dir, "model.safetensors"), gemma4TinyWeightsWithPerLayerInputs()); err != nil {
		t.Fatalf("SaveSafetensors: %v", err)
	}

	model, err := LoadGemma4(dir)
	if err != nil {
		t.Fatalf("LoadGemma4: %v", err)
	}
	defer closeGemma4(model)

	if model.EmbedTokensPerLayer == nil {
		t.Fatal("expected per-layer embedding table to load")
	}
	if model.PerLayerModelProj == nil {
		t.Fatal("expected per-layer model projection to load")
	}
	if model.PerLayerProjNorm == nil || model.PerLayerProjNorm.Weight == nil {
		t.Fatal("expected per-layer projection norm to load")
	}
	for i, layer := range model.Layers {
		if layer.PerLayerInputGate == nil {
			t.Fatalf("layer %d missing per_layer_input_gate", i)
		}
		if layer.PerLayerProjection == nil {
			t.Fatalf("layer %d missing per_layer_projection", i)
		}
		if layer.PostPerLayerInputNorm == nil || layer.PostPerLayerInputNorm.Weight == nil {
			t.Fatalf("layer %d missing post_per_layer_input_norm", i)
		}
	}

	tokens := FromValues([]int32{2, 3, 4}, 1, 3)
	caches := model.NewCache()
	logits := model.Forward(tokens, caches)
	if err := Eval(logits); err != nil {
		t.Fatalf("Eval logits: %v", err)
	}
	defer func() {
		Free(tokens, logits)
		freeCaches(caches)
	}()

	shape := logits.Shape()
	if len(shape) != 3 {
		t.Fatalf("logits dims = %v, want rank 3", shape)
	}
	if shape[0] != 1 || shape[1] != 3 || shape[2] != 10 {
		t.Fatalf("logits shape = %v, want [1 3 10]", shape)
	}
}

func TestGemma4_LoadDisablesPerLayerInputsWithoutProjectionNorm_Good(t *testing.T) {
	requireMetalRuntime(t)

	dir := t.TempDir()
	config := `{
		"model_type": "gemma4_text",
		"hidden_size": 8,
		"num_hidden_layers": 2,
		"intermediate_size": 16,
		"num_attention_heads": 1,
		"num_key_value_heads": 1,
		"head_dim": 4,
		"global_head_dim": 8,
		"vocab_size": 10,
		"vocab_size_per_layer_input": 10,
		"rms_norm_eps": 1e-6,
		"sliding_window": 4,
		"sliding_window_pattern": 2,
		"num_kv_shared_layers": 0,
		"layer_types": ["sliding_attention", "full_attention"]
	}`
	if err := coreio.Local.Write(core.JoinPath(dir, "config.json"), config); err != nil {
		t.Fatalf("write config.json: %v", err)
	}
	writeMinimalTokenizer(t, dir)

	weights := gemma4TinyWeightsWithPerLayerInputs()
	delete(weights, "model.per_layer_projection_norm.weight")
	if err := SaveSafetensors(core.JoinPath(dir, "model.safetensors"), weights); err != nil {
		t.Fatalf("SaveSafetensors: %v", err)
	}

	model, err := LoadGemma4(dir)
	if err != nil {
		t.Fatalf("LoadGemma4: %v", err)
	}
	defer closeGemma4(model)

	if model.EmbedTokensPerLayer != nil {
		t.Fatal("per-layer embedding table should be disabled without projection norm")
	}
	if model.PerLayerModelProj != nil {
		t.Fatal("per-layer model projection should be disabled without projection norm")
	}
	if model.PerLayerProjNorm != nil {
		t.Fatal("per-layer projection norm should be nil when per-layer inputs are disabled")
	}
	for i, layer := range model.Layers {
		if layer.PerLayerInputGate != nil {
			t.Fatalf("layer %d per_layer_input_gate should be disabled", i)
		}
		if layer.PerLayerProjection != nil {
			t.Fatalf("layer %d per_layer_projection should be disabled", i)
		}
		if layer.PostPerLayerInputNorm != nil {
			t.Fatalf("layer %d post_per_layer_input_norm should be disabled", i)
		}
	}
}

func TestGemma4_LoadDisablesPerLayerInputsWithoutProjectionNorm_ReleasesUnusedWeights_Good(t *testing.T) {
	requireMetalRuntime(t)

	dir := t.TempDir()
	config := `{
		"model_type": "gemma4_text",
		"hidden_size": 8,
		"num_hidden_layers": 2,
		"intermediate_size": 16,
		"num_attention_heads": 1,
		"num_key_value_heads": 1,
		"head_dim": 4,
		"global_head_dim": 8,
		"vocab_size": 10,
		"vocab_size_per_layer_input": 10,
		"rms_norm_eps": 1e-6,
		"sliding_window": 4,
		"sliding_window_pattern": 2,
		"num_kv_shared_layers": 0,
		"layer_types": ["sliding_attention", "full_attention"]
	}`
	if err := coreio.Local.Write(core.JoinPath(dir, "config.json"), config); err != nil {
		t.Fatalf("write config.json: %v", err)
	}
	writeMinimalTokenizer(t, dir)

	weights := gemma4TinyWeightsWithPerLayerInputs()
	delete(weights, "model.per_layer_projection_norm.weight")
	if err := SaveSafetensors(core.JoinPath(dir, "model.safetensors"), weights); err != nil {
		t.Fatalf("SaveSafetensors: %v", err)
	}
	freeWeightMap(weights)

	ClearCache()
	baseline := GetActiveMemory()

	model, err := LoadGemma4(dir)
	if err != nil {
		t.Fatalf("LoadGemma4: %v", err)
	}

	closeGemma4(model)
	ClearCache()

	if active := GetActiveMemory(); active > baseline {
		t.Fatalf("active memory after close = %d, want <= %d", active, baseline)
	}
}

func TestGemma4_LoadKEqVModel_ReleasesUnusedVProjWeights_Good(t *testing.T) {
	requireMetalRuntime(t)

	dir := t.TempDir()
	config := `{
		"model_type": "gemma4_text",
		"hidden_size": 8,
		"num_hidden_layers": 1,
		"intermediate_size": 16,
		"num_attention_heads": 1,
		"num_key_value_heads": 1,
		"num_global_key_value_heads": 1,
		"head_dim": 4,
		"global_head_dim": 8,
		"attention_k_eq_v": true,
		"vocab_size": 10,
		"rms_norm_eps": 1e-6,
		"sliding_window": 4,
		"sliding_window_pattern": 1,
		"num_kv_shared_layers": 0,
		"hidden_size_per_layer_input": 0,
		"layer_types": ["full_attention"]
	}`
	if err := coreio.Local.Write(core.JoinPath(dir, "config.json"), config); err != nil {
		t.Fatalf("write config.json: %v", err)
	}
	writeMinimalTokenizer(t, dir)

	weights := map[string]*Array{
		"model.embed_tokens.weight":                        seqArray(0.01, 10, 8),
		"model.norm.weight":                                seqArray(0.02, 8),
		"model.layers.0.input_layernorm.weight":            seqArray(0.03, 8),
		"model.layers.0.post_attention_layernorm.weight":   seqArray(0.04, 8),
		"model.layers.0.pre_feedforward_layernorm.weight":  seqArray(0.05, 8),
		"model.layers.0.post_feedforward_layernorm.weight": seqArray(0.06, 8),
		"model.layers.0.layer_scalar":                      FromValues([]float32{1}, 1),
		"model.layers.0.self_attn.q_proj.weight":           seqArray(0.10, 8, 8),
		"model.layers.0.self_attn.k_proj.weight":           seqArray(0.20, 8, 8),
		"model.layers.0.self_attn.v_proj.weight":           seqArray(0.30, 8, 8),
		"model.layers.0.self_attn.o_proj.weight":           seqArray(0.40, 8, 8),
		"model.layers.0.self_attn.q_norm.weight":           seqArray(0.50, 8),
		"model.layers.0.self_attn.k_norm.weight":           seqArray(0.60, 8),
		"model.layers.0.mlp.gate_proj.weight":              seqArray(0.70, 16, 8),
		"model.layers.0.mlp.up_proj.weight":                seqArray(0.80, 16, 8),
		"model.layers.0.mlp.down_proj.weight":              seqArray(0.90, 8, 16),
	}
	if err := SaveSafetensors(core.JoinPath(dir, "model.safetensors"), weights); err != nil {
		t.Fatalf("SaveSafetensors: %v", err)
	}
	freeWeightMap(weights)

	ClearCache()
	baseline := GetActiveMemory()

	model, err := LoadGemma4(dir)
	if err != nil {
		t.Fatalf("LoadGemma4: %v", err)
	}

	if got := model.Layers[0].Attention.VProj; got != nil {
		t.Fatal("expected K-equals-V full-attention layer to drop v_proj")
	}

	closeGemma4(model)
	ClearCache()

	if active := GetActiveMemory(); active > baseline {
		t.Fatalf("active memory after close = %d, want <= %d", active, baseline)
	}
}

func gemma4TinyWeights() map[string]*Array {
	weights := map[string]*Array{
		"model.embed_tokens.weight": seqArray(0.01, 10, 8),
		"model.norm.weight":         seqArray(0.02, 8),
	}

	addLayer := func(idx int, sliding bool) {
		prefix := core.Sprintf("model.layers.%d", idx)
		headDim := 4
		oIn := 4
		if !sliding {
			headDim = 8
			oIn = 8
		}
		weights[prefix+".input_layernorm.weight"] = seqArray(0.03+float32(idx), 8)
		weights[prefix+".post_attention_layernorm.weight"] = seqArray(0.04+float32(idx), 8)
		weights[prefix+".pre_feedforward_layernorm.weight"] = seqArray(0.05+float32(idx), 8)
		weights[prefix+".post_feedforward_layernorm.weight"] = seqArray(0.06+float32(idx), 8)
		weights[prefix+".layer_scalar"] = FromValues([]float32{1}, 1)

		weights[prefix+".self_attn.q_proj.weight"] = seqArray(0.10+float32(idx), headDim, 8)
		weights[prefix+".self_attn.k_proj.weight"] = seqArray(0.20+float32(idx), headDim, 8)
		weights[prefix+".self_attn.v_proj.weight"] = seqArray(0.30+float32(idx), headDim, 8)
		weights[prefix+".self_attn.o_proj.weight"] = seqArray(0.40+float32(idx), 8, oIn)
		weights[prefix+".self_attn.q_norm.weight"] = seqArray(0.50+float32(idx), headDim)
		weights[prefix+".self_attn.k_norm.weight"] = seqArray(0.60+float32(idx), headDim)

		weights[prefix+".mlp.gate_proj.weight"] = seqArray(0.70+float32(idx), 16, 8)
		weights[prefix+".mlp.up_proj.weight"] = seqArray(0.80+float32(idx), 16, 8)
		weights[prefix+".mlp.down_proj.weight"] = seqArray(0.90+float32(idx), 8, 16)
	}

	addLayer(0, true)
	addLayer(1, false)
	return weights
}

func gemma4TinyWeightsWithPerLayerInputs() map[string]*Array {
	weights := gemma4TinyWeights()
	weights["model.embed_tokens_per_layer.weight"] = seqArray(1.10, 10, 4)
	weights["model.per_layer_model_projection.weight"] = seqArray(1.20, 4, 8)
	weights["model.per_layer_projection_norm.weight"] = seqArray(1.30, 2)

	for idx := 0; idx < 2; idx++ {
		prefix := core.Sprintf("model.layers.%d", idx)
		weights[prefix+".per_layer_input_gate.weight"] = seqArray(1.40+float32(idx), 2, 8)
		weights[prefix+".per_layer_projection.weight"] = seqArray(1.50+float32(idx), 8, 2)
		weights[prefix+".post_per_layer_input_norm.weight"] = seqArray(1.60+float32(idx), 8)
	}

	return weights
}

func seqArray(start float32, shape ...int) *Array {
	size := 1
	for _, dim := range shape {
		size *= dim
	}
	data := make([]float32, size)
	for i := range size {
		data[i] = start + 0.01*float32(i)
	}
	return FromValues(data, shape...)
}

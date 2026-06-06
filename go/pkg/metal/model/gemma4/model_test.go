// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma4

import (
	"math"
	"reflect"
	"slices"
	"strings"
	"testing"

	core "dappco.re/go"

	coreio "dappco.re/go/io"
	"dappco.re/go/mlx/pkg/metal"
)

func requireMetalRuntime(t testing.TB) {
	t.Helper()
	if core.Getenv("GO_MLX_RUN_METAL_TESTS") != "1" {
		t.Skip("set GO_MLX_RUN_METAL_TESTS=1 to enable Metal runtime tests")
	}
	if !metal.MetalAvailable() {
		t.Skip("Metal runtime unavailable")
	}
}

func freeWeightMap(weights map[string]*metal.Array) {
	for _, arr := range weights {
		metal.Free(arr)
	}
}

func arraySetContains(set map[*metal.Array]struct{}, arr *metal.Array) bool {
	_, ok := set[arr]
	return ok
}

func arraySliceContains(arrays []*metal.Array, needle *metal.Array) bool {
	return slices.Contains(arrays, needle)
}

const floatApproxTol = 1e-5

func floatSliceApprox(t *testing.T, got []float32, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("length mismatch: got %d, want %d", len(got), len(want))
	}
	for i := range got {
		if math.Abs(float64(got[i])-float64(want[i])) >= floatApproxTol {
			t.Errorf("[%d] = %f, want %f", i, got[i], want[i])
		}
	}
}

// TestGemma4_ParseConfig_InvariantDefaultsAndDeclaredFields_Good locks the two
// honest halves of the parser: the genuine architectural invariants
// (head_dim 256, global_head_dim 512, vocab_size 262144, rms_norm_eps 1e-6) are
// defaulted when the pack omits them, while everything that varies per pack
// (sliding_window, max_position_embeddings, use_double_wide_mlp, the core dims,
// layer_types) is read straight from the declared config and never guessed. The
// config below omits the invariant-defaulted fields but declares every varying
// one, so each defaulted value proves a real invariant and each declared value
// proves a real read.
func TestGemma4_ParseConfig_InvariantDefaultsAndDeclaredFields_Good(t *testing.T) {
	cfg, err := parseGemma4Config([]byte(`{
		"model_type": "gemma4_text",
		"hidden_size": 1024,
		"num_hidden_layers": 6,
		"intermediate_size": 2048,
		"num_attention_heads": 4,
		"num_key_value_heads": 1,
		"sliding_window": 512,
		"max_position_embeddings": 131072,
		"use_double_wide_mlp": true,
		"layer_types": [
			"sliding_attention",
			"sliding_attention",
			"sliding_attention",
			"sliding_attention",
			"sliding_attention",
			"full_attention"
		]
	}`))
	if err != nil {
		t.Fatalf("parseGemma4Config: %v", err)
	}
	// Invariant defaults applied (the pack omitted these).
	if cfg.HeadDim != 256 {
		t.Errorf("HeadDim = %d, want 256 (invariant default)", cfg.HeadDim)
	}
	if cfg.GlobalHeadDim != 512 {
		t.Errorf("GlobalHeadDim = %d, want 512 (invariant default)", cfg.GlobalHeadDim)
	}
	if cfg.VocabSize != 262144 {
		t.Errorf("VocabSize = %d, want 262144 (invariant default)", cfg.VocabSize)
	}
	if cfg.RMSNormEps != 1e-6 {
		t.Errorf("RMSNormEps = %g, want 1e-6 (invariant default)", cfg.RMSNormEps)
	}
	// Declared varying fields read verbatim.
	if cfg.SlidingWindow != 512 {
		t.Errorf("SlidingWindow = %d, want declared 512", cfg.SlidingWindow)
	}
	if cfg.MaxPositionEmbeddings != 131072 {
		t.Errorf("MaxPositionEmbeddings = %d, want declared 131072", cfg.MaxPositionEmbeddings)
	}
	if !cfg.UseDoubleWideMLP {
		t.Error("UseDoubleWideMLP = false, want declared true")
	}
	// tie_word_embeddings still follows the transformers convention when omitted.
	if !cfg.TieWordEmbeddings {
		t.Error("TieWordEmbeddings = false, want true (convention default)")
	}
	// final_logit_softcapping omitted → stays 0, never fabricated.
	if cfg.FinalLogitSoftcapping != 0 {
		t.Errorf("FinalLogitSoftcapping = %f, want 0 (config omits it — no fabricated softcap)", cfg.FinalLogitSoftcapping)
	}
	if cfg.NumKVSharedLayers != 0 {
		t.Errorf("NumKVSharedLayers = %d, want 0", cfg.NumKVSharedLayers)
	}
	// layer_types is read verbatim from the declared schedule — index 5 is
	// full_attention because the pack DECLARED it, not because the parser
	// force-set the final layer global.
	want := []string{
		"sliding_attention",
		"sliding_attention",
		"sliding_attention",
		"sliding_attention",
		"sliding_attention",
		"full_attention",
	}
	if len(cfg.LayerTypes) != len(want) {
		t.Fatalf("LayerTypes len = %d, want %d", len(cfg.LayerTypes), len(want))
	}
	for i, got := range cfg.LayerTypes {
		if got != want[i] {
			t.Fatalf("LayerTypes[%d] = %q, want %q", i, got, want[i])
		}
	}
	if cfg.LayerTypes[5] != "full_attention" {
		t.Errorf("LayerTypes[5] = %q, want full_attention (declared, not force-set)", cfg.LayerTypes[5])
	}
	if cfg.RopeParameters["full_attention"].RopeType != "proportional" {
		t.Errorf("full attention rope type = %q, want proportional", cfg.RopeParameters["full_attention"].RopeType)
	}
	if cfg.RopeParameters["sliding_attention"].RopeTheta != 10000 {
		t.Errorf("sliding attention rope theta = %f, want 10000", cfg.RopeParameters["sliding_attention"].RopeTheta)
	}
}

// TestGemma4_ParseConfig_RequiresDeclaredLayerTypes_Bad locks the deletion of
// the old layer_types synthesis. Every gemma-4 pack declares its per-layer
// sliding/full schedule, so an omitted or wrong-length layer_types is a
// malformed pack — the parser must fail loud rather than fabricate a schedule
// from a guessed period (the old "every 6th" rule was even wrong for E2B, which
// is every 5th).
func TestGemma4_ParseConfig_RequiresDeclaredLayerTypes_Bad(t *testing.T) {
	// (a) layer_types omitted entirely.
	_, err := parseGemma4Config([]byte(`{
		"model_type": "gemma4_text",
		"hidden_size": 1024,
		"num_hidden_layers": 7,
		"intermediate_size": 2048,
		"num_attention_heads": 4,
		"num_key_value_heads": 1,
		"head_dim": 256,
		"use_double_wide_mlp": true,
		"sliding_window": 512,
		"max_position_embeddings": 131072
	}`))
	if err == nil {
		t.Fatal("parseGemma4Config succeeded with omitted layer_types, want error")
	}
	if !core.Contains(err.Error(), "layer_types must be declared") {
		t.Fatalf("parseGemma4Config error = %v, want layer_types must be declared", err)
	}

	// (b) layer_types declared but length != num_hidden_layers.
	_, err = parseGemma4Config([]byte(`{
		"model_type": "gemma4_text",
		"hidden_size": 1024,
		"num_hidden_layers": 7,
		"intermediate_size": 2048,
		"num_attention_heads": 4,
		"num_key_value_heads": 1,
		"head_dim": 256,
		"use_double_wide_mlp": true,
		"sliding_window": 512,
		"max_position_embeddings": 131072,
		"layer_types": ["sliding_attention", "full_attention"]
	}`))
	if err == nil {
		t.Fatal("parseGemma4Config succeeded with short layer_types, want error")
	}
	if !core.Contains(err.Error(), "layer_types") {
		t.Fatalf("parseGemma4Config error = %v, want layer_types length error", err)
	}
}

func TestGemma4_ParseConfig_PreservesE2BLayerMetadata_Good(t *testing.T) {
	cfg, err := parseGemma4Config([]byte(`{
		"model_type": "gemma4",
		"text_config": {
			"model_type": "gemma4_text",
			"hidden_size": 1536,
			"num_hidden_layers": 35,
			"intermediate_size": 6144,
			"num_attention_heads": 8,
			"num_key_value_heads": 1,
			"head_dim": 256,
			"global_head_dim": 512,
			"hidden_size_per_layer_input": 256,
			"num_kv_shared_layers": 20,
			"sliding_window": 512,
			"max_position_embeddings": 131072,
			"use_double_wide_mlp": true,
			"layer_types": [
				"sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
				"sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
				"sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
				"sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
				"sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
				"sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
				"sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention"
			],
			"rope_parameters": {
				"full_attention": {
					"partial_rotary_factor": 0.25,
					"rope_theta": 1000000.0,
					"rope_type": "proportional"
				},
				"sliding_attention": {
					"rope_theta": 10000.0,
					"rope_type": "default"
				}
			}
		}
	}`))
	if err != nil {
		t.Fatalf("parseGemma4Config: %v", err)
	}
	if cfg.SlidingWindow != 512 {
		t.Fatalf("SlidingWindow = %d, want 512", cfg.SlidingWindow)
	}
	if cfg.NumKVSharedLayers != 20 {
		t.Fatalf("NumKVSharedLayers = %d, want 20", cfg.NumKVSharedLayers)
	}
	if len(cfg.LayerTypes) != 35 {
		t.Fatalf("LayerTypes len = %d, want 35", len(cfg.LayerTypes))
	}
	fullLayers := map[int]bool{4: true, 9: true, 14: true, 19: true, 24: true, 29: true, 34: true}
	for i, got := range cfg.LayerTypes {
		want := "sliding_attention"
		if fullLayers[i] {
			want = "full_attention"
		}
		if got != want {
			t.Fatalf("LayerTypes[%d] = %q, want %q", i, got, want)
		}
	}
	full := cfg.RopeParameters["full_attention"]
	if full.RopeType != "proportional" || full.PartialRotaryFactor != 0.25 || full.RopeTheta != 1000000 {
		t.Fatalf("full rope params = %+v, want proportional p-RoPE", full)
	}

	layers := make([]*Gemma4DecoderLayer, len(cfg.LayerTypes))
	for i, layerType := range cfg.LayerTypes {
		layers[i] = &Gemma4DecoderLayer{LayerType: layerType}
	}
	previous, cacheIndexByLayer := buildGemma4CacheLayout(layers, cfg.NumKVSharedLayers)
	ownerCount := 0
	for _, cacheIdx := range cacheIndexByLayer {
		if cacheIdx >= 0 {
			ownerCount++
		}
	}
	if ownerCount != 15 {
		t.Fatalf("owner cache count = %d, want 15 pre-sharing owners", ownerCount)
	}
	if previous[15] != 13 {
		t.Fatalf("PreviousKVs[15] = %d, want sliding owner 13", previous[15])
	}
	if previous[19] != 14 {
		t.Fatalf("PreviousKVs[19] = %d, want full owner 14", previous[19])
	}
	if previous[34] != 14 {
		t.Fatalf("PreviousKVs[34] = %d, want full owner 14", previous[34])
	}
	if cacheIndexByLayer[15] != -1 || cacheIndexByLayer[19] != -1 || cacheIndexByLayer[34] != -1 {
		t.Fatalf("shared layers allocated caches: layer15=%d layer19=%d layer34=%d", cacheIndexByLayer[15], cacheIndexByLayer[19], cacheIndexByLayer[34])
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
		"num_kv_shared_layers": 0,
		"use_double_wide_mlp": true,
		"sliding_window": 512,
		"max_position_embeddings": 131072,
		"layer_types": [
			"sliding_attention",
			"sliding_attention",
			"sliding_attention",
			"sliding_attention",
			"sliding_attention",
			"full_attention"
		]
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
		"head_dim": 256,
		"use_double_wide_mlp": true,
		"sliding_window": 512,
		"max_position_embeddings": 131072
	}`))
	if err == nil {
		t.Fatal("parseGemma4Config succeeded, want error")
	}
	if !core.Contains(err.Error(), "negative num_hidden_layers") {
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
			"head_dim": 256,
			"use_double_wide_mlp": true,
			"sliding_window": 512,
			"max_position_embeddings": 131072,
			"layer_types": ["sliding_attention", "full_attention"]
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

func TestGemma4_ParseConfig_Official12BUnified_Good(t *testing.T) {
	layerTypes := make([]string, 0, 48)
	for i := 0; i < 48; i++ {
		if (i+1)%6 == 0 {
			layerTypes = append(layerTypes, `"full_attention"`)
		} else {
			layerTypes = append(layerTypes, `"sliding_attention"`)
		}
	}
	cfgJSON := core.Sprintf(`{
		"architectures": ["Gemma4UnifiedForConditionalGeneration"],
		"audio_config": {
			"audio_embed_dim": 640,
			"audio_samples_per_token": 640,
			"hidden_size": 640,
			"model_type": "gemma4_unified_audio",
			"output_proj_dims": 640,
			"rms_norm_eps": 1e-06
		},
		"audio_token_id": 258881,
		"boa_token_id": 256000,
		"boi_token_id": 255999,
		"eoa_token_index": 258883,
		"eoi_token_id": 258882,
		"model_type": "gemma4_unified",
		"text_config": {
			"attention_k_eq_v": true,
			"final_logit_softcapping": 30.0,
			"global_head_dim": 512,
			"head_dim": 256,
			"hidden_size": 3840,
			"hidden_size_per_layer_input": 0,
			"intermediate_size": 15360,
			"layer_types": [%s],
			"max_position_embeddings": 262144,
			"model_type": "gemma4_unified_text",
			"num_attention_heads": 16,
			"num_global_key_value_heads": 1,
			"num_hidden_layers": 48,
			"num_key_value_heads": 8,
			"num_kv_shared_layers": 0,
			"rms_norm_eps": 1e-06,
			"rope_parameters": {
				"full_attention": {
					"partial_rotary_factor": 0.25,
					"rope_theta": 1000000.0,
					"rope_type": "proportional"
				},
				"sliding_attention": {
					"rope_theta": 10000.0,
					"rope_type": "default"
				}
			},
			"sliding_window": 1024,
			"tie_word_embeddings": true,
			"use_double_wide_mlp": false,
			"vocab_size": 262144,
			"vocab_size_per_layer_input": 262144
		},
		"video_token_id": 258884,
		"vision_config": {
			"mm_embed_dim": 3840,
			"mm_posemb_size": 1120,
			"model_patch_size": 48,
			"model_type": "gemma4_unified_vision",
			"num_soft_tokens": 280,
			"output_proj_dims": 3840,
			"patch_size": 16,
			"pooling_kernel_size": 3,
			"rms_norm_eps": 1e-06
		}
	}`, strings.Join(layerTypes, ","))
	cfg, err := parseGemma4Config([]byte(cfgJSON))
	if err != nil {
		t.Fatalf("parseGemma4Config: %v", err)
	}
	if cfg.ModelType != "gemma4_unified" {
		t.Fatalf("ModelType = %q, want gemma4_unified", cfg.ModelType)
	}
	if cfg.HiddenSize != 3840 || cfg.NumHiddenLayers != 48 || cfg.IntermediateSize != 15360 {
		t.Fatalf("text shape hidden=%d layers=%d intermediate=%d, want 3840/48/15360", cfg.HiddenSize, cfg.NumHiddenLayers, cfg.IntermediateSize)
	}
	if cfg.SlidingWindow != 1024 || cfg.MaxPositionEmbeddings != 262144 || cfg.VocabSize != 262144 {
		t.Fatalf("window/context/vocab = %d/%d/%d, want 1024/262144/262144", cfg.SlidingWindow, cfg.MaxPositionEmbeddings, cfg.VocabSize)
	}
	if cfg.HiddenSizePerLayerInput != 0 {
		t.Fatalf("HiddenSizePerLayerInput = %d, want 0 for encoder-free 12B Unified", cfg.HiddenSizePerLayerInput)
	}
	if !cfg.AttentionKEqV || cfg.NumGlobalKeyValueHeads == nil || *cfg.NumGlobalKeyValueHeads != 1 {
		t.Fatalf("attention K=V/global kv heads = %v/%v, want true/1", cfg.AttentionKEqV, cfg.NumGlobalKeyValueHeads)
	}
	if cfg.UseDoubleWideMLP {
		t.Fatal("UseDoubleWideMLP = true, want false for 12B Unified config")
	}
	if len(cfg.LayerTypes) != 48 || cfg.LayerTypes[5] != "full_attention" || cfg.LayerTypes[47] != "full_attention" {
		t.Fatalf("LayerTypes summary len=%d layer5=%q last=%q, want 48/full/full", len(cfg.LayerTypes), cfg.LayerTypes[5], cfg.LayerTypes[47])
	}
	if cfg.RopeParameters["full_attention"].RopeType != "proportional" || cfg.RopeParameters["full_attention"].RopeTheta != 1000000 {
		t.Fatalf("full attention rope = %+v, want p-RoPE theta 1e6", cfg.RopeParameters["full_attention"])
	}
	if cfg.VisionConfig == nil || cfg.VisionConfig.ModelType != "gemma4_unified_vision" {
		t.Fatalf("VisionConfig = %+v, want unified vision config", cfg.VisionConfig)
	}
	if cfg.VisionConfig.MMEmbedDim != 3840 || cfg.VisionConfig.MMPosembSize != 1120 || cfg.VisionConfig.ModelPatchSize != 48 {
		t.Fatalf("vision projector dims = %d/%d/%d, want 3840/1120/48", cfg.VisionConfig.MMEmbedDim, cfg.VisionConfig.MMPosembSize, cfg.VisionConfig.ModelPatchSize)
	}
	if cfg.VisionConfig.NumSoftTokens != 280 || cfg.VisionConfig.OutputProjDims != 3840 {
		t.Fatalf("vision soft/output dims = %d/%d, want 280/3840", cfg.VisionConfig.NumSoftTokens, cfg.VisionConfig.OutputProjDims)
	}
	if cfg.AudioConfig == nil || cfg.AudioConfig.ModelType != "gemma4_unified_audio" {
		t.Fatalf("AudioConfig = %+v, want unified audio config", cfg.AudioConfig)
	}
	if cfg.AudioConfig.AudioEmbedDim != 640 || cfg.AudioConfig.AudioSamplesPerToken != 640 || cfg.AudioConfig.OutputProjDims != 640 {
		t.Fatalf("audio dims = %d/%d/%d, want 640/640/640", cfg.AudioConfig.AudioEmbedDim, cfg.AudioConfig.AudioSamplesPerToken, cfg.AudioConfig.OutputProjDims)
	}
	if cfg.AudioTokenID != 258881 || cfg.VideoTokenID != 258884 || cfg.BOITokenID != 255999 || cfg.BOATokenID != 256000 || cfg.EOITokenID != 258882 || cfg.EOATokenIndex != 258883 {
		t.Fatalf("unified token ids audio=%d video=%d boi=%d boa=%d eoi=%d eoa=%d, want official 12B ids", cfg.AudioTokenID, cfg.VideoTokenID, cfg.BOITokenID, cfg.BOATokenID, cfg.EOITokenID, cfg.EOATokenIndex)
	}
}

// TestGemma4_ParseConfig_UnifiedHasNoSpecialDefaults_Good locks the deletion of
// the unified-branch special-casing. A gemma4_unified config is now parsed like
// any other pack: the values it declares are read verbatim and nothing is
// fabricated for it. The old branch guessed sliding_window 1024 /
// max_position_embeddings 262144 / unified token-id defaults for an omitting
// config — all gone. Below the unified pack DECLARES its sizes and the test
// proves they are read, then a sub-case proves unified gets no exemption from
// the use_double_wide_mlp requirement.
func TestGemma4_ParseConfig_UnifiedHasNoSpecialDefaults_Good(t *testing.T) {
	layerTypes := make([]string, 0, 48)
	for i := 0; i < 48; i++ {
		if (i+1)%6 == 0 {
			layerTypes = append(layerTypes, `"full_attention"`)
		} else {
			layerTypes = append(layerTypes, `"sliding_attention"`)
		}
	}
	cfgJSON := core.Sprintf(`{
		"architectures": ["Gemma4UnifiedForConditionalGeneration"],
		"audio_config": {"model_type": "gemma4_unified_audio"},
		"model_type": "gemma4_unified",
		"text_config": {
			"attention_k_eq_v": true,
			"global_head_dim": 512,
			"head_dim": 256,
			"hidden_size": 3840,
			"hidden_size_per_layer_input": 0,
			"intermediate_size": 15360,
			"layer_types": [%s],
			"max_position_embeddings": 262144,
			"model_type": "gemma4_unified_text",
			"num_attention_heads": 16,
			"num_global_key_value_heads": 1,
			"num_hidden_layers": 48,
			"num_key_value_heads": 8,
			"num_kv_shared_layers": 0,
			"sliding_window": 1024,
			"use_double_wide_mlp": false,
			"vocab_size": 262144
		},
		"vision_config": {"model_type": "gemma4_unified_vision"}
	}`, strings.Join(layerTypes, ","))
	cfg, err := parseGemma4Config([]byte(cfgJSON))
	if err != nil {
		t.Fatalf("parseGemma4Config(unified): %v", err)
	}
	// Declared values read verbatim — no unified special-casing.
	if cfg.SlidingWindow != 1024 {
		t.Fatalf("SlidingWindow = %d, want declared 1024", cfg.SlidingWindow)
	}
	if cfg.MaxPositionEmbeddings != 262144 {
		t.Fatalf("MaxPositionEmbeddings = %d, want declared 262144", cfg.MaxPositionEmbeddings)
	}
	if cfg.HiddenSizePerLayerInput != 0 {
		t.Fatalf("HiddenSizePerLayerInput = %d, want declared 0 for encoder-free 12B Unified", cfg.HiddenSizePerLayerInput)
	}
	if cfg.UseDoubleWideMLP {
		t.Fatal("UseDoubleWideMLP = true, want declared false for dense 12B Unified")
	}
	if !cfg.TieWordEmbeddings {
		t.Fatal("TieWordEmbeddings = false, want tied embeddings by convention default")
	}

	// Unified gets no exemption: drop use_double_wide_mlp and the parser must
	// fail loud exactly as it would for any other pack.
	noMLPJSON := core.Sprintf(`{
		"architectures": ["Gemma4UnifiedForConditionalGeneration"],
		"audio_config": {"model_type": "gemma4_unified_audio"},
		"model_type": "gemma4_unified",
		"text_config": {
			"attention_k_eq_v": true,
			"global_head_dim": 512,
			"head_dim": 256,
			"hidden_size": 3840,
			"hidden_size_per_layer_input": 0,
			"intermediate_size": 15360,
			"layer_types": [%s],
			"max_position_embeddings": 262144,
			"model_type": "gemma4_unified_text",
			"num_attention_heads": 16,
			"num_global_key_value_heads": 1,
			"num_hidden_layers": 48,
			"num_key_value_heads": 8,
			"num_kv_shared_layers": 0,
			"sliding_window": 1024,
			"vocab_size": 262144
		},
		"vision_config": {"model_type": "gemma4_unified_vision"}
	}`, strings.Join(layerTypes, ","))
	if _, err := parseGemma4Config([]byte(noMLPJSON)); err == nil {
		t.Fatal("parseGemma4Config succeeded for unified config without use_double_wide_mlp, want error")
	} else if !core.Contains(err.Error(), "use_double_wide_mlp is required") {
		t.Fatalf("parseGemma4Config error = %v, want use_double_wide_mlp is required", err)
	}
}

// TestGemma4_ParseConfig_TopLevelMaxPosWinsOverTextBackbone_Good locks the
// 31B / 26B-MoE fix: those models carry their real 256K context (262144) at
// the top level and the text backbone's 128K (131072) inside text_config. The
// merge must take the LARGER — reading text_config first cramped the two
// biggest models to 128K. Both values are declared here, so no guess fires.
func TestGemma4_ParseConfig_TopLevelMaxPosWinsOverTextBackbone_Good(t *testing.T) {
	cfg, err := parseGemma4Config([]byte(`{
		"model_type": "gemma4",
		"max_position_embeddings": 262144,
		"text_config": {
			"model_type": "gemma4_text",
			"hidden_size": 1024,
			"intermediate_size": 2048,
			"num_attention_heads": 4,
			"num_key_value_heads": 1,
			"head_dim": 256,
			"global_head_dim": 512,
			"num_hidden_layers": 6,
			"sliding_window": 1024,
			"max_position_embeddings": 131072,
			"use_double_wide_mlp": true,
			"layer_types": [
				"sliding_attention",
				"sliding_attention",
				"sliding_attention",
				"sliding_attention",
				"sliding_attention",
				"full_attention"
			]
		}
	}`))
	if err != nil {
		t.Fatalf("parseGemma4Config: %v", err)
	}
	if cfg.MaxPositionEmbeddings != 262144 {
		t.Fatalf("MaxPositionEmbeddings = %d, want 262144 (top-level 256K must win over the text backbone's 131072)", cfg.MaxPositionEmbeddings)
	}
	if cfg.SlidingWindow != 1024 {
		t.Fatalf("SlidingWindow = %d, want 1024 (declared in text_config)", cfg.SlidingWindow)
	}
}

func TestGemma4_ParseConfig_FamilyVariants_Good(t *testing.T) {
	cases := []struct {
		name                    string
		modelType               string
		textModelType           string
		hiddenSize              int32
		numLayers               int32
		intermediateSize        int32
		numAttentionHeads       int32
		numKeyValueHeads        int32
		hiddenSizePerLayerInput int32
		numKVSharedLayers       int32
		slidingWindow           int32
		maxPositionEmbeddings   int32
		layerPattern            int
		fullLayers              int
		slidingLayers           int
		attentionKEqV           bool
		useDoubleWideMLP        bool
		moe                     bool
		numExperts              int32
		topKExperts             int32
		moeIntermediateSize     int32
	}{
		{
			name:                    "E2B",
			modelType:               "gemma4",
			textModelType:           "gemma4_text",
			hiddenSize:              1536,
			numLayers:               35,
			intermediateSize:        6144,
			numAttentionHeads:       8,
			numKeyValueHeads:        1,
			hiddenSizePerLayerInput: 256,
			numKVSharedLayers:       20,
			slidingWindow:           512,
			maxPositionEmbeddings:   131072,
			layerPattern:            5,
			fullLayers:              7,
			slidingLayers:           28,
			useDoubleWideMLP:        true,
		},
		{
			name:                    "E4B",
			modelType:               "gemma4",
			textModelType:           "gemma4_text",
			hiddenSize:              2560,
			numLayers:               42,
			intermediateSize:        10240,
			numAttentionHeads:       8,
			numKeyValueHeads:        2,
			hiddenSizePerLayerInput: 256,
			numKVSharedLayers:       18,
			slidingWindow:           512,
			maxPositionEmbeddings:   131072,
			layerPattern:            6,
			fullLayers:              7,
			slidingLayers:           35,
			useDoubleWideMLP:        true,
		},
		{
			name:                  "12B Unified",
			modelType:             "gemma4_unified",
			textModelType:         "gemma4_unified_text",
			hiddenSize:            3840,
			numLayers:             48,
			intermediateSize:      15360,
			numAttentionHeads:     16,
			numKeyValueHeads:      8,
			slidingWindow:         1024,
			maxPositionEmbeddings: 262144,
			layerPattern:          6,
			fullLayers:            8,
			slidingLayers:         40,
			attentionKEqV:         true,
		},
		{
			name:                  "31B",
			modelType:             "gemma4",
			textModelType:         "gemma4_text",
			hiddenSize:            5376,
			numLayers:             60,
			intermediateSize:      21504,
			numAttentionHeads:     32,
			numKeyValueHeads:      16,
			slidingWindow:         1024,
			maxPositionEmbeddings: 262144,
			layerPattern:          6,
			fullLayers:            10,
			slidingLayers:         50,
			attentionKEqV:         true,
		},
		{
			name:                    "26B A4B MoE",
			modelType:               "gemma4",
			textModelType:           "gemma4_text",
			hiddenSize:              2816,
			numLayers:               30,
			intermediateSize:        2112,
			numAttentionHeads:       16,
			numKeyValueHeads:        8,
			slidingWindow:           1024,
			maxPositionEmbeddings:   262144,
			layerPattern:            6,
			fullLayers:              5,
			slidingLayers:           25,
			attentionKEqV:           true,
			moe:                     true,
			numExperts:              128,
			topKExperts:             8,
			moeIntermediateSize:     704,
			hiddenSizePerLayerInput: 0,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cfg, err := parseGemma4Config([]byte(gemma4FamilyConfigJSON(tc.modelType, tc.textModelType, tc.numLayers, tc.layerPattern, tc.hiddenSize, tc.intermediateSize, tc.numAttentionHeads, tc.numKeyValueHeads, tc.hiddenSizePerLayerInput, tc.numKVSharedLayers, tc.slidingWindow, tc.maxPositionEmbeddings, tc.attentionKEqV, tc.useDoubleWideMLP, tc.moe, tc.numExperts, tc.topKExperts, tc.moeIntermediateSize)))
			if err != nil {
				t.Fatalf("parseGemma4Config(%s): %v", tc.name, err)
			}
			if cfg.ModelType != tc.modelType {
				t.Fatalf("ModelType = %q, want %q", cfg.ModelType, tc.modelType)
			}
			if cfg.HiddenSize != tc.hiddenSize || cfg.NumHiddenLayers != tc.numLayers || cfg.IntermediateSize != tc.intermediateSize {
				t.Fatalf("shape hidden/layers/intermediate = %d/%d/%d, want %d/%d/%d", cfg.HiddenSize, cfg.NumHiddenLayers, cfg.IntermediateSize, tc.hiddenSize, tc.numLayers, tc.intermediateSize)
			}
			if cfg.NumAttentionHeads != tc.numAttentionHeads || cfg.NumKeyValueHeads != tc.numKeyValueHeads {
				t.Fatalf("attention heads = %d/%d, want %d/%d", cfg.NumAttentionHeads, cfg.NumKeyValueHeads, tc.numAttentionHeads, tc.numKeyValueHeads)
			}
			if cfg.HiddenSizePerLayerInput != tc.hiddenSizePerLayerInput || cfg.NumKVSharedLayers != tc.numKVSharedLayers {
				t.Fatalf("PLE/shared KV = %d/%d, want %d/%d", cfg.HiddenSizePerLayerInput, cfg.NumKVSharedLayers, tc.hiddenSizePerLayerInput, tc.numKVSharedLayers)
			}
			if cfg.SlidingWindow != tc.slidingWindow || cfg.MaxPositionEmbeddings != tc.maxPositionEmbeddings {
				t.Fatalf("window/context = %d/%d, want %d/%d", cfg.SlidingWindow, cfg.MaxPositionEmbeddings, tc.slidingWindow, tc.maxPositionEmbeddings)
			}
			if cfg.AttentionKEqV != tc.attentionKEqV {
				t.Fatalf("AttentionKEqV = %v, want %v", cfg.AttentionKEqV, tc.attentionKEqV)
			}
			if cfg.UseDoubleWideMLP != tc.useDoubleWideMLP {
				t.Fatalf("UseDoubleWideMLP = %v, want %v", cfg.UseDoubleWideMLP, tc.useDoubleWideMLP)
			}
			sliding, full := gemma4CountLayerTypes(cfg.LayerTypes)
			if sliding != tc.slidingLayers || full != tc.fullLayers {
				t.Fatalf("layer topology sliding/full = %d/%d, want %d/%d", sliding, full, tc.slidingLayers, tc.fullLayers)
			}
			if cfg.EnableMoEBlock != tc.moe {
				t.Fatalf("EnableMoEBlock = %v, want %v", cfg.EnableMoEBlock, tc.moe)
			}
			if tc.moe {
				if cfg.NumExperts == nil || *cfg.NumExperts != tc.numExperts || cfg.TopKExperts == nil || *cfg.TopKExperts != tc.topKExperts || cfg.MoEIntermediateSize == nil || *cfg.MoEIntermediateSize != tc.moeIntermediateSize {
					t.Fatalf("MoE config experts/topK/intermediate = %v/%v/%v, want %d/%d/%d", cfg.NumExperts, cfg.TopKExperts, cfg.MoEIntermediateSize, tc.numExperts, tc.topKExperts, tc.moeIntermediateSize)
				}
			}
		})
	}
}

func gemma4FamilyConfigJSON(modelType, textModelType string, numLayers int32, pattern int, hiddenSize, intermediateSize, numAttentionHeads, numKeyValueHeads, hiddenSizePerLayerInput, numKVSharedLayers, slidingWindow, maxPositionEmbeddings int32, attentionKEqV, useDoubleWideMLP, moe bool, numExperts, topKExperts, moeIntermediateSize int32) string {
	moeFields := `"enable_moe_block": false, "num_experts": null, "top_k_experts": null, "moe_intermediate_size": null`
	if moe {
		moeFields = core.Sprintf(`"enable_moe_block": true, "num_experts": %d, "top_k_experts": %d, "moe_intermediate_size": %d`, numExperts, topKExperts, moeIntermediateSize)
	}
	return core.Sprintf(`{
		"model_type": %q,
		"text_config": {
			"attention_k_eq_v": %t,
			"global_head_dim": 512,
			"head_dim": 256,
			"hidden_size": %d,
			"hidden_size_per_layer_input": %d,
			"intermediate_size": %d,
			"layer_types": [%s],
			"max_position_embeddings": %d,
			"model_type": %q,
			%s,
			"num_attention_heads": %d,
			"num_hidden_layers": %d,
			"num_key_value_heads": %d,
			"num_kv_shared_layers": %d,
			"sliding_window": %d,
			"use_double_wide_mlp": %t,
			"vocab_size": 262144
		}
	}`, modelType, attentionKEqV, hiddenSize, hiddenSizePerLayerInput, intermediateSize, gemma4FamilyLayerTypesJSON(int(numLayers), pattern), maxPositionEmbeddings, textModelType, moeFields, numAttentionHeads, numLayers, numKeyValueHeads, numKVSharedLayers, slidingWindow, useDoubleWideMLP)
}

func gemma4FamilyLayerTypesJSON(numLayers, pattern int) string {
	layerTypes := make([]string, numLayers)
	for i := range layerTypes {
		layerType := "full_attention"
		if pattern > 1 && (i+1)%pattern != 0 {
			layerType = "sliding_attention"
		}
		if i == len(layerTypes)-1 {
			layerType = "full_attention"
		}
		layerTypes[i] = core.Sprintf("%q", layerType)
	}
	return strings.Join(layerTypes, ",")
}

func gemma4CountLayerTypes(layerTypes []string) (sliding, full int) {
	for _, layerType := range layerTypes {
		switch layerType {
		case "sliding_attention":
			sliding++
		case "full_attention":
			full++
		}
	}
	return sliding, full
}

func TestGemma4_FinalizeMultiModalModelTypePreservesUnified_Good(t *testing.T) {
	model := &Gemma4Model{Cfg: &Gemma4TextConfig{
		TransformerConfig: metal.TransformerConfig{ModelType: "gemma4_unified"},
	}}

	finalizeGemma4MultiModalModelType(model)
	if got := model.ModelType(); got != "gemma4_unified" {
		t.Fatalf("ModelType() = %q, want gemma4_unified", got)
	}
	if model.Cfg.ModelType != "gemma4_unified" {
		t.Fatalf("Cfg.ModelType = %q, want gemma4_unified", model.Cfg.ModelType)
	}
}

func TestGemma4_UnifiedModalTokenCountsIncludesVideo_Good(t *testing.T) {
	cfg := &Gemma4TextConfig{
		ImageTokenID: 258880,
		AudioTokenID: 258881,
		VideoTokenID: 258884,
	}

	imageCount, audioCount, videoCount := gemma4UnifiedModalTokenCounts(cfg, []int32{1, 258884, 258880, 258881, 258884})
	if imageCount != 1 || audioCount != 1 || videoCount != 2 {
		t.Fatalf("modal counts = image:%d audio:%d video:%d, want 1/1/2", imageCount, audioCount, videoCount)
	}
}

func TestGemma4_UnifiedModalTokenCountsIgnoreUnsetIDs_Good(t *testing.T) {
	cfg := &Gemma4TextConfig{
		ImageTokenID: 258880,
	}

	imageCount, audioCount, videoCount := gemma4UnifiedModalTokenCounts(cfg, []int32{0, 258880, 0})
	if imageCount != 1 || audioCount != 0 || videoCount != 0 {
		t.Fatalf("modal counts = image:%d audio:%d video:%d, want 1/0/0", imageCount, audioCount, videoCount)
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
		"use_double_wide_mlp": true,
		"sliding_window": 512,
		"max_position_embeddings": 131072,
		"layer_types": [
			"sliding_attention",
			"sliding_attention",
			"sliding_attention",
			"sliding_attention",
			"sliding_attention",
			"full_attention"
		],
		"rope_parameters": {
			"full_attention": {
				"partial_rotary_factor": 0.25,
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
	// partial_rotary_factor is read straight from the declared override (the old
	// parser hard-defaulted GlobalPartialRotaryFactor to a fabricated 0.25; that
	// guess is gone, so the pack must declare it).
	if full.PartialRotaryFactor != 0.25 {
		t.Fatalf("full partial rotary factor = %f, want declared 0.25", full.PartialRotaryFactor)
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

// TestGemma4_ParseConfig_MoEDeclaredExperts_Good locks the MoE expert counts to
// the model-declared num_experts / top_k_experts. The old 128 / 8 default was a
// fabricated guess; an MoE pack that omits the counts is malformed and must fail
// loud rather than load a wrong router width.
func TestGemma4_ParseConfig_MoEDeclaredExperts_Good(t *testing.T) {
	cfg, err := parseGemma4Config([]byte(`{
		"model_type": "gemma4_text",
		"hidden_size": 1024,
		"num_hidden_layers": 2,
		"intermediate_size": 2048,
		"num_attention_heads": 4,
		"num_key_value_heads": 1,
		"head_dim": 256,
		"use_double_wide_mlp": true,
		"sliding_window": 512,
		"max_position_embeddings": 131072,
		"layer_types": ["sliding_attention", "full_attention"],
		"enable_moe_block": true,
		"num_experts": 32,
		"top_k_experts": 2
	}`))
	if err != nil {
		t.Fatalf("parseGemma4Config: %v", err)
	}
	if !cfg.EnableMoEBlock {
		t.Fatal("EnableMoEBlock = false, want true")
	}
	if cfg.NumExperts == nil || *cfg.NumExperts != 32 {
		t.Fatalf("NumExperts = %v, want declared 32", cfg.NumExperts)
	}
	if cfg.TopKExperts == nil || *cfg.TopKExperts != 2 {
		t.Fatalf("TopKExperts = %v, want declared 2", cfg.TopKExperts)
	}

	// An MoE pack that enables the block but omits the expert counts is malformed
	// — the parser must reject it, never substitute the deleted 128 / 8 guess.
	_, err = parseGemma4Config([]byte(`{
		"model_type": "gemma4_text",
		"hidden_size": 1024,
		"num_hidden_layers": 2,
		"intermediate_size": 2048,
		"num_attention_heads": 4,
		"num_key_value_heads": 1,
		"head_dim": 256,
		"use_double_wide_mlp": true,
		"sliding_window": 512,
		"max_position_embeddings": 131072,
		"layer_types": ["sliding_attention", "full_attention"],
		"enable_moe_block": true
	}`))
	if err == nil {
		t.Fatal("parseGemma4Config succeeded with MoE block but no expert counts, want error")
	}
	if !core.Contains(err.Error(), "num_experts") || !core.Contains(err.Error(), "top_k_experts") {
		t.Fatalf("parseGemma4Config error = %v, want num_experts / top_k_experts not declared", err)
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
			"use_double_wide_mlp": true,
			"sliding_window": 512,
			"max_position_embeddings": 131072,
			"layer_types": ["sliding_attention", "full_attention"],
			"quantization": {"group_size": 64, "bits": 4, "mode": "affine"}
		}
	}`))
	if err != nil {
		t.Fatalf("parseGemma4Config: %v", err)
	}
	if cfg.ModelType != "gemma4" {
		t.Fatalf("ModelType = %q, want gemma4", cfg.ModelType)
	}
	if cfg.Quantization == nil || cfg.Quantization.GroupSize != 64 || cfg.Quantization.Bits != 4 || cfg.Quantization.Mode != "affine" {
		t.Fatalf("Quantization = %+v, want group_size=64 bits=4 mode=affine", cfg.Quantization)
	}
	if got := cfg.LayerTypes; len(got) != 2 || got[0] != "sliding_attention" || got[1] != "full_attention" {
		t.Fatalf("LayerTypes = %v, want explicit nested layer types", got)
	}
}

func TestGemma4_ParseConfig_TopLevelMXFPQuantization_Good(t *testing.T) {
	cfg, err := parseGemma4Config([]byte(`{
		"model_type": "gemma4",
		"quantization": {"group_size": 32, "bits": 8, "mode": "mxfp8"},
		"text_config": {
			"hidden_size": 1024,
			"num_hidden_layers": 2,
			"intermediate_size": 2048,
			"num_attention_heads": 4,
			"num_key_value_heads": 1,
			"head_dim": 256,
			"use_double_wide_mlp": true,
			"sliding_window": 512,
			"max_position_embeddings": 131072,
			"layer_types": ["sliding_attention", "full_attention"]
		}
	}`))
	if err != nil {
		t.Fatalf("parseGemma4Config: %v", err)
	}
	if cfg.Quantization == nil || cfg.Quantization.GroupSize != 32 || cfg.Quantization.Bits != 8 || cfg.Quantization.Mode != "mxfp8" {
		t.Fatalf("Quantization = %+v, want group_size=32 bits=8 mode=mxfp8", cfg.Quantization)
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
			"sliding_window": 512,
			"max_position_embeddings": 131072,
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
			"use_double_wide_mlp": true,
			"max_position_embeddings": 131072,
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
			"sliding_window": 512,
			"max_position_embeddings": 131072,
			"layer_types": ["sliding_attention", "full_attention"],
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
			"max_position_embeddings": 131072,
			"use_double_wide_mlp": true,
			"layer_types": ["sliding_attention", "full_attention"],
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
	defer metal.Free(embed)

	got := inferGemma4PerLayerInputSize(map[string]*metal.Array{
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
	defer metal.Free(gate, proj)

	got := inferGemma4PerLayerInputSize(map[string]*metal.Array{
		"model.layers.0.per_layer_input_gate.weight": gate,
		"model.layers.0.per_layer_projection.weight": proj,
	}, 2)
	if got != 6 {
		t.Fatalf("inferGemma4PerLayerInputSize() = %d, want 6", got)
	}
}

func TestGemma4_InferPerLayerInputSize_PackedEmbeddingProjectionWins_Good(t *testing.T) {
	requireMetalRuntime(t)

	embeddingPacked := metal.FromValues(make([]uint32, 16*32), 16, 32)
	projection := seqArray(1.20, 256, 8)
	defer metal.Free(embeddingPacked, projection)

	got := inferGemma4PerLayerInputSize(map[string]*metal.Array{
		"model.embed_tokens_per_layer.weight":     embeddingPacked,
		"model.per_layer_model_projection.weight": projection,
	}, 4)
	if got != 64 {
		t.Fatalf("inferGemma4PerLayerInputSize() = %d, want 64", got)
	}
}

func TestGemma4_NormalizePerLayerTensor_TransposedEmbedding_Good(t *testing.T) {
	requireMetalRuntime(t)

	input := metal.FromValues([]float32{1, 2, 3, 4, 5, 6}, 1, 1, 2, 3)
	output := gemma4NormalizePerLayerTensor(input, 1, 1, 3, 2)
	if err := metal.Eval(output); err != nil {
		t.Fatalf("Eval: %v", err)
	}
	defer metal.Free(input, output)

	if got := output.Shape(); len(got) != 4 || got[0] != 1 || got[1] != 1 || got[2] != 3 || got[3] != 2 {
		t.Fatalf("normalized shape = %v, want [1 1 3 2]", got)
	}

	floatSliceApprox(t, output.Floats(), []float32{1, 4, 2, 5, 3, 6})
}

func TestGemma4_CompiledPerLayerInputsMatchesGoGraph_Good(t *testing.T) {
	requireMetalRuntime(t)

	m := &Gemma4Model{
		EmbedTokensPerLayer: &metal.Embedding{Weight: metal.FromValues([]float32{
			0.1, 0.2, 0.3, 0.4,
			0.5, 0.6, 0.7, 0.8,
			0.9, 1.0, 1.1, 1.2,
		}, 3, 4)},
		PerLayerModelProj: metal.NewLinear(metal.FromValues([]float32{0.2, 0.1, -0.3, 0.4, 0.5, -0.2, 0.7, 0.6}, 4, 2), nil),
		PerLayerProjNorm:  &metal.RMSNormModule{Weight: metal.FromValues([]float32{1, 1}, 2)},
		PerLayerProjNormScaled: metal.FromValues([]float32{
			1, 1,
		}, 2),
		Cfg: &Gemma4TextConfig{
			TransformerConfig: metal.TransformerConfig{
				HiddenSize:      2,
				NumHiddenLayers: 2,
				RMSNormEps:      1e-6,
			},
			HiddenSizePerLayerInput: 2,
		},
	}
	defer closeGemma4(m)

	tokens := metal.FromValues([]int32{1}, 1, 1)
	hidden := metal.FromValues([]float32{0.5, -0.25}, 1, 1, 2)
	defer metal.Free(tokens, hidden)

	old := enableCompiledGemma4PerLayerInputs
	enableCompiledGemma4PerLayerInputs = false
	base := m.computePerLayerInputs(tokens, hidden)
	if err := metal.Eval(base...); err != nil {
		t.Fatalf("base per-layer inputs eval: %v", err)
	}
	baseFloats := make([][]float32, len(base))
	for i := range base {
		baseFloats[i] = append([]float32(nil), base[i].Floats()...)
	}
	metal.Free(base...)

	enableCompiledGemma4PerLayerInputs = true
	t.Cleanup(func() { enableCompiledGemma4PerLayerInputs = old })
	compiled := m.computePerLayerInputs(tokens, hidden)
	defer metal.Free(compiled...)
	if err := metal.Eval(compiled...); err != nil {
		t.Fatalf("compiled per-layer inputs eval: %v", err)
	}
	if len(compiled) != len(baseFloats) {
		t.Fatalf("compiled per-layer count = %d, want %d", len(compiled), len(baseFloats))
	}
	for i := range compiled {
		floatSliceApprox(t, compiled[i].Floats(), baseFloats[i])
	}
}

func TestGemma4_PerLayerInputForLayerMatchesSplit_Good(t *testing.T) {
	requireMetalRuntime(t)

	m := &Gemma4Model{
		Cfg: &Gemma4TextConfig{
			TransformerConfig: metal.TransformerConfig{
				NumHiddenLayers: 3,
			},
			HiddenSizePerLayerInput: 2,
		},
	}
	combined := metal.FromValues([]float32{
		0.1, 0.2,
		0.3, 0.4,
		0.5, 0.6,
	}, 1, 1, 3, 2)
	defer metal.Free(combined)

	split := m.splitPerLayerInputTensor(combined.Clone())
	defer metal.Free(split...)
	if len(split) != int(m.Cfg.NumHiddenLayers) {
		t.Fatalf("split layer count = %d, want %d", len(split), m.Cfg.NumHiddenLayers)
	}
	for i := range split {
		streamed := m.perLayerInputForLayer(combined, 1, 1, int32(i))
		if streamed == nil || !streamed.Valid() {
			t.Fatalf("streamed layer %d is invalid", i)
		}
		floatSliceApprox(t, streamed.Floats(), split[i].Floats())
		metal.Free(streamed)
	}
}

func TestGemma4_PerLayerEmbeddingRetainedLazy_Good(t *testing.T) {
	requireMetalRuntime(t)

	model := &Gemma4Model{
		EmbedTokensPerLayer: &metal.Embedding{
			Weight: metal.FromValues([]float32{0.1, 0.2, 0.3, 0.4}, 2, 2),
			Scales: metal.FromValues([]float32{1.0, 1.0}, 2, 1),
			Biases: metal.FromValues([]float32{0.0, 0.0}, 2, 1),
		},
		PerLayerModelProj: metal.NewLinear(metal.FromValues([]float32{0.2, 0.1, -0.3, 0.4}, 2, 2), nil),
		Output:            metal.NewLinear(metal.FromValues([]float32{0.5, -0.2, 0.7, 0.6}, 2, 2), nil),
	}
	defer closeGemma4(model)

	retained := gemma4RetainedWeights(model)
	lazy := gemma4LazyRetainedWeights(model)
	materializable := gemma4MaterializableRetainedWeights(retained, lazy)

	for _, arr := range []*metal.Array{
		model.EmbedTokensPerLayer.Weight,
		model.EmbedTokensPerLayer.Scales,
		model.EmbedTokensPerLayer.Biases,
	} {
		if !arraySetContains(retained, arr) {
			t.Fatal("per-layer embedding arrays must stay retained for model lifetime")
		}
		if !arraySetContains(lazy, arr) {
			t.Fatal("per-layer embedding arrays should stay lazy at load time")
		}
		if arraySliceContains(materializable, arr) {
			t.Fatal("per-layer embedding arrays should not be eagerly materialized")
		}
	}

	if !arraySliceContains(materializable, model.PerLayerModelProj.Weight) {
		t.Fatal("per-layer projection should still be eagerly materialized")
	}
	if !arraySliceContains(materializable, model.Output.Weight) {
		t.Fatal("output projection should still be eagerly materialized")
	}
}

func TestGemma4_DisablePerLayerInputsDiagnostic_Bad(t *testing.T) {
	requireMetalRuntime(t)

	m := &Gemma4Model{
		EmbedTokensPerLayer:    &metal.Embedding{Weight: metal.FromValues([]float32{0.1, 0.2, 0.3, 0.4}, 2, 2)},
		PerLayerModelProj:      metal.NewLinear(metal.FromValues([]float32{0.2, 0.1, -0.3, 0.4}, 2, 2), nil),
		PerLayerProjNorm:       &metal.RMSNormModule{Weight: metal.FromValues([]float32{1, 1}, 2)},
		PerLayerProjNormScaled: metal.FromValues([]float32{1, 1}, 2),
		Cfg: &Gemma4TextConfig{
			TransformerConfig: metal.TransformerConfig{
				HiddenSize:      2,
				NumHiddenLayers: 1,
				RMSNormEps:      1e-6,
			},
			HiddenSizePerLayerInput: 2,
		},
	}
	defer closeGemma4(m)

	old := disableGemma4PerLayerInputs
	disableGemma4PerLayerInputs = true
	t.Cleanup(func() { disableGemma4PerLayerInputs = old })

	tokens := metal.FromValues([]int32{1}, 1, 1)
	hidden := metal.FromValues([]float32{0.5, -0.25}, 1, 1, 2)
	defer metal.Free(tokens, hidden)

	if got := m.computePerLayerInputs(tokens, hidden); got != nil {
		metal.Free(got...)
		t.Fatal("computePerLayerInputs() = non-nil with diagnostic disable gate")
	}
}

func TestGemma4_FixedAttentionMaskCapacityOffset_Good(t *testing.T) {
	capacity, offset, ok := fixedGemma4AttentionMaskCapacityOffset(metal.NewFixedKVCacheAtOffset(2336, 2204, 2204), sharedKV{}, 1)
	if !ok || capacity != 2336 || offset != 2204 {
		t.Fatalf("full fixed mask = capacity %d offset %d ok %v, want 2336/2204/true", capacity, offset, ok)
	}

	if _, _, ok := fixedGemma4AttentionMaskCapacityOffset(metal.NewFixedKVCacheAtOffset(1024, 2204, 1024), sharedKV{}, 1); ok {
		t.Fatal("overflowed sliding fixed cache should not build an absolute-position causal mask")
	}

	if _, _, ok := fixedGemma4AttentionMaskCapacityOffset(metal.NewFixedKVCacheAtOffset(2336, 2204, 2204), sharedKV{}, 2); ok {
		t.Fatal("multi-token decode should not use the single-token shared fixed mask")
	}
}

func TestGemma4_OutputLinear_TiedFallback_Good(t *testing.T) {
	embed := &metal.Embedding{}
	output, err := gemma4OutputLinear(map[string]*metal.Array{}, &Gemma4TextConfig{
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
	_, err := gemma4OutputLinear(map[string]*metal.Array{}, &Gemma4TextConfig{}, &metal.Embedding{})
	if err == nil {
		t.Fatal("expected error when untied Gemma4 model lacks lm_head.weight")
	}
	if !core.Contains(err.Error(), "lm_head.weight") {
		t.Fatalf("expected lm_head.weight error, got: %v", err)
	}
}

func TestGemma4_PreferNativeLastTokenOutputLogits_Good(t *testing.T) {
	if gemma4PreferNativeLastTokenOutputLogits(nil) {
		t.Fatal("nil output should not use native last-token logits")
	}
	if !gemma4PreferNativeLastTokenOutputLogits(metal.NewLinear(&metal.Array{}, nil)) {
		t.Fatal("dense output should use native last-token logits")
	}
	if gemma4PreferNativeLastTokenOutputLogits(metal.NewQuantizedLinear(&metal.Array{}, &metal.Array{}, &metal.Array{}, nil, 64, 4)) {
		t.Fatal("quantized output should stay on the graph path")
	}
}

func TestGemma4_AttentionScale_Good(t *testing.T) {
	got := gemma4AttentionScale(512)
	if got != 1.0 {
		t.Fatalf("gemma4AttentionScale(512) = %f, want 1.0", got)
	}
}

func TestGemma4_PrecomputeNormWeightsUsesDirectScale_Good(t *testing.T) {
	requireMetalRuntime(t)
	weight := metal.FromValues([]float32{0.125, 2.5}, 2)
	defer metal.Free(weight)
	model := &Gemma4Model{
		Norm:             &metal.RMSNormModule{Weight: weight},
		PerLayerProjNorm: &metal.RMSNormModule{Weight: weight},
		Layers: []*Gemma4DecoderLayer{{
			InputNorm:             &metal.RMSNormModule{Weight: weight},
			PostAttnNorm:          &metal.RMSNormModule{Weight: weight},
			PreFFNorm:             &metal.RMSNormModule{Weight: weight},
			PostFFNorm:            &metal.RMSNormModule{Weight: weight},
			PreFFNorm2:            &metal.RMSNormModule{Weight: weight},
			PostFFNorm1:           &metal.RMSNormModule{Weight: weight},
			PostFFNorm2:           &metal.RMSNormModule{Weight: weight},
			PostPerLayerInputNorm: &metal.RMSNormModule{Weight: weight},
			Attention: &Gemma4Attention{
				QNorm: &metal.RMSNormModule{Weight: weight},
				KNorm: &metal.RMSNormModule{Weight: weight},
			},
		}},
	}
	precomputeGemma4ScaledWeights(model)
	layer := model.Layers[0]
	defer metal.Free(
		model.NormScaled,
		model.PerLayerProjNormScaled,
		layer.InputNormScaled,
		layer.PostAttnNormScaled,
		layer.PreFFNormScaled,
		layer.PostFFNormScaled,
		layer.PreFFNorm2Scaled,
		layer.PostFFNorm1Scaled,
		layer.PostFFNorm2Scaled,
		layer.PostPerLayerInputNormScaled,
		layer.Attention.QNormScaled,
		layer.Attention.KNormScaled,
	)

	if err := metal.Eval(
		model.NormScaled,
		model.PerLayerProjNormScaled,
		layer.InputNormScaled,
		layer.PostAttnNormScaled,
		layer.PreFFNormScaled,
		layer.PostFFNormScaled,
		layer.PreFFNorm2Scaled,
		layer.PostFFNorm1Scaled,
		layer.PostFFNorm2Scaled,
		layer.PostPerLayerInputNormScaled,
		layer.Attention.QNormScaled,
		layer.Attention.KNormScaled,
	); err != nil {
		t.Fatalf("Eval scaled norm weights: %v", err)
	}
	floatSliceApprox(t, model.NormScaled.Floats(), []float32{0.125, 2.5})
	floatSliceApprox(t, model.PerLayerProjNormScaled.Floats(), []float32{0.125, 2.5})
	floatSliceApprox(t, layer.InputNormScaled.Floats(), []float32{0.125, 2.5})
	floatSliceApprox(t, layer.PostAttnNormScaled.Floats(), []float32{0.125, 2.5})
	floatSliceApprox(t, layer.PreFFNormScaled.Floats(), []float32{0.125, 2.5})
	floatSliceApprox(t, layer.PostFFNormScaled.Floats(), []float32{0.125, 2.5})
	floatSliceApprox(t, layer.PreFFNorm2Scaled.Floats(), []float32{0.125, 2.5})
	floatSliceApprox(t, layer.PostFFNorm1Scaled.Floats(), []float32{0.125, 2.5})
	floatSliceApprox(t, layer.PostFFNorm2Scaled.Floats(), []float32{0.125, 2.5})
	floatSliceApprox(t, layer.PostPerLayerInputNormScaled.Floats(), []float32{0.125, 2.5})
	floatSliceApprox(t, layer.Attention.QNormScaled.Floats(), []float32{0.125, 2.5})
	floatSliceApprox(t, layer.Attention.KNormScaled.Floats(), []float32{0.125, 2.5})
}

func TestGemma4_ProportionalRoPEFreqsMatchesHFDefinition_Good(t *testing.T) {
	requireMetalRuntime(t)

	freqs := gemma4ProportionalFreqs(512, 128, 1000000, 1)
	defer metal.Free(freqs)
	if got := freqs.Shape(); len(got) != 1 || got[0] != 256 {
		t.Fatalf("freq shape = %v, want [256]", got)
	}
	if err := metal.Eval(freqs); err != nil {
		t.Fatalf("Eval p-RoPE freqs: %v", err)
	}

	values := freqs.Floats()
	for _, idx := range []int{0, 1, 63} {
		want := math.Pow(1000000, float64(idx*2)/512.0)
		got := float64(values[idx])
		tolerance := math.Max(1e-5, math.Abs(want)*1e-5)
		if math.Abs(got-want) > tolerance {
			t.Fatalf("freq[%d] = %f, want %f", idx, got, want)
		}
	}
	for i := 64; i < len(values); i++ {
		if !math.IsInf(float64(values[i]), 1) {
			t.Fatalf("freq[%d] = %f, want +Inf unrotated p-RoPE tail", i, values[i])
		}
	}
}

func TestGemma4_SwitchLinear_PrefixFallback_Good(t *testing.T) {
	requireMetalRuntime(t)

	switchWeight := func(scale float32) *metal.Array {
		return metal.FromValues([]float32{
			scale, 0,
			0, scale,
		}, 1, 2, 2)
	}

	cases := []struct {
		name    string
		weights map[string]*metal.Array
	}{
		{
			name: "rfc_switch_glu",
			weights: map[string]*metal.Array{
				"model.layers.0.experts.switch_glu.gate_proj.weight": switchWeight(1.0),
			},
		},
		{
			name: "legacy_direct",
			weights: map[string]*metal.Array{
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
			metal.FreeSwitchLinear(layer)
		})
	}
}

func TestGemma4_Linear_QuantizedWithoutConfig_Good(t *testing.T) {
	requireMetalRuntime(t)

	weight := seqArray(0.10, 2, 8)
	scales := seqArray(0.20, 2, 1)
	biases := seqArray(0.30, 2, 1)
	defer metal.Free(weight, scales, biases)

	layer := gemma4Linear(map[string]*metal.Array{
		"model.layers.0.self_attn.q_proj.weight": weight,
		"model.layers.0.self_attn.q_proj.scales": scales,
		"model.layers.0.self_attn.q_proj.biases": biases,
	}, "model.layers.0.self_attn.q_proj", nil)
	if layer == nil {
		t.Fatal("expected quantized layer")
	}
	defer metal.FreeLinear(layer)

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
	defer metal.Free(weight, scales, biases)

	layer := gemma4SwitchLinear(map[string]*metal.Array{
		"model.layers.0.experts.switch_glu.gate_proj.weight": weight,
		"model.layers.0.experts.switch_glu.gate_proj.scales": scales,
		"model.layers.0.experts.switch_glu.gate_proj.biases": biases,
	}, nil, "model.layers.0.experts.switch_glu.gate_proj")
	if layer == nil {
		t.Fatal("expected quantized switch layer")
	}
	defer metal.FreeSwitchLinear(layer)

	if layer.Scales != scales || layer.Biases != biases {
		t.Fatal("quantized Gemma4 switch layer should preserve scales/biases when config is absent")
	}
	if layer.GroupSize != 0 || layer.Bits != 0 {
		t.Fatalf("quantized Gemma4 switch layer should defer to MLX affine defaults, got group_size=%d bits=%d", layer.GroupSize, layer.Bits)
	}
}

func TestGemma4_QuantPredicate_RouterForces8Bit_Good(t *testing.T) {
	defaultQ := &metal.QuantizationConfig{GroupSize: 128, Bits: 4}

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

func TestGemma4_QuantPredicate_RouterPreservesMXFPMode_Good(t *testing.T) {
	defaultQ := &metal.QuantizationConfig{GroupSize: 32, Bits: 8, Mode: "mxfp8"}

	routerQ := gemma4QuantPredicate("model.layers.0.router.proj", defaultQ)
	if routerQ == nil {
		t.Fatal("router quantization predicate returned nil")
	}
	if routerQ.GroupSize != 32 || routerQ.Bits != 8 || routerQ.Mode != "mxfp8" {
		t.Fatalf("router quantization = %+v, want mxfp8 group_size=32 bits=8", routerQ)
	}
}

func TestGemma4_QuantForWeight_AllowsMLXCommunityVariants_Good(t *testing.T) {
	cases := []struct {
		name string
		in   *metal.QuantizationConfig
		want *metal.QuantizationConfig
	}{
		{name: "mxfp4", in: &metal.QuantizationConfig{GroupSize: 32, Bits: 4, Mode: "mxfp4"}, want: &metal.QuantizationConfig{GroupSize: 32, Bits: 4, Mode: "mxfp4"}},
		{name: "mxfp8", in: &metal.QuantizationConfig{GroupSize: 32, Bits: 8, Mode: "mxfp8"}, want: &metal.QuantizationConfig{GroupSize: 32, Bits: 8, Mode: "mxfp8"}},
		{name: "affine5", in: &metal.QuantizationConfig{GroupSize: 64, Bits: 5, Mode: "affine"}, want: &metal.QuantizationConfig{GroupSize: 64, Bits: 5, Mode: "affine"}},
		{name: "affine6", in: &metal.QuantizationConfig{GroupSize: 64, Bits: 6, Mode: "affine"}, want: &metal.QuantizationConfig{GroupSize: 64, Bits: 6, Mode: "affine"}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := gemma4QuantForWeight("model.layers.0.mlp.gate_proj", tc.in, nil, nil)
			if got == nil {
				t.Fatal("gemma4QuantForWeight returned nil")
			}
			if got.GroupSize != tc.want.GroupSize || got.Bits != tc.want.Bits || got.Mode != tc.want.Mode {
				t.Fatalf("quantization = %+v, want %+v", got, tc.want)
			}
		})
	}
}

func TestGemma4_QuantForWeight_DetectsAffineOverrideInsideMXFP_Good(t *testing.T) {
	requireMetalRuntime(t)

	weight := metal.Zeros([]int32{2112, 704}, metal.DTypeUint32)
	scales := metal.Zeros([]int32{2112, 44}, metal.DTypeFloat32)
	defer metal.Free(weight, scales)

	got := gemma4QuantForWeight("model.layers.0.mlp.gate_proj", &metal.QuantizationConfig{
		GroupSize: 32,
		Bits:      4,
		Mode:      "mxfp4",
	}, weight, scales)
	if got == nil {
		t.Fatal("gemma4QuantForWeight returned nil")
	}
	if got.Mode != "affine" || got.GroupSize != 64 || got.Bits != 8 {
		t.Fatalf("quantization = %+v, want affine group_size=64 bits=8", got)
	}
}

func TestGemma4_QuantForWeight_InfersAffineDefaultsFromPackedWeights_Good(t *testing.T) {
	requireMetalRuntime(t)

	weight := metal.Zeros([]int32{256, 192}, metal.DTypeUint32)
	scales := metal.Zeros([]int32{256, 24}, metal.DTypeFloat32)
	defer metal.Free(weight, scales)

	got := gemma4QuantForWeight("model.layers.0.self_attn.k_proj", nil, weight, scales)
	if got == nil {
		t.Fatal("gemma4QuantForWeight returned nil")
	}
	if got.Mode != "affine" || got.GroupSize != 64 || got.Bits != 4 {
		t.Fatalf("quantization = %+v, want inferred affine group_size=64 bits=4", got)
	}
}

func TestGemma4_ValidateQuantizationConfig_Bad(t *testing.T) {
	err := validateGemma4QuantizationConfig(&metal.QuantizationConfig{GroupSize: 32, Bits: 7, Mode: "mxfp8"})
	if err == nil || !core.Contains(err.Error(), "mxfp8") {
		t.Fatalf("validateGemma4QuantizationConfig error = %v, want mxfp8 bits diagnostic", err)
	}
}

func TestGemma4_Linear_Infers8BitOverrideFromScales_Good(t *testing.T) {
	requireMetalRuntime(t)

	weight := metal.Zeros([]int32{2112, 704}, metal.DTypeUint32)
	scales := metal.Zeros([]int32{2112, 44}, metal.DTypeFloat32)
	biases := metal.Zeros([]int32{2112, 44}, metal.DTypeFloat32)
	defer metal.Free(weight, scales, biases)

	layer := gemma4Linear(map[string]*metal.Array{
		"model.layers.0.mlp.gate_proj.weight": weight,
		"model.layers.0.mlp.gate_proj.scales": scales,
		"model.layers.0.mlp.gate_proj.biases": biases,
	}, "model.layers.0.mlp.gate_proj", &metal.QuantizationConfig{GroupSize: 64, Bits: 4})
	if layer == nil {
		t.Fatal("expected quantized layer")
	}
	defer metal.FreeLinear(layer)

	if layer.GroupSize != 64 || layer.Bits != 8 {
		t.Fatalf("quantization = group_size=%d bits=%d, want group_size=64 bits=8", layer.GroupSize, layer.Bits)
	}
}

func TestGemma4_SwitchLinear_Preserves4BitWhenShapesMatchDefault_Good(t *testing.T) {
	requireMetalRuntime(t)

	weight := metal.Zeros([]int32{128, 2112, 352}, metal.DTypeUint32)
	scales := metal.Zeros([]int32{128, 2112, 44}, metal.DTypeFloat32)
	biases := metal.Zeros([]int32{128, 2112, 44}, metal.DTypeFloat32)
	defer metal.Free(weight, scales, biases)

	layer := gemma4SwitchLinear(map[string]*metal.Array{
		"model.layers.0.experts.switch_glu.gate_proj.weight": weight,
		"model.layers.0.experts.switch_glu.gate_proj.scales": scales,
		"model.layers.0.experts.switch_glu.gate_proj.biases": biases,
	}, &metal.QuantizationConfig{GroupSize: 64, Bits: 4}, "model.layers.0.experts.switch_glu.gate_proj")
	if layer == nil {
		t.Fatal("expected quantized switch layer")
	}
	defer metal.FreeSwitchLinear(layer)

	if layer.GroupSize != 64 || layer.Bits != 4 {
		t.Fatalf("quantization = group_size=%d bits=%d, want group_size=64 bits=4", layer.GroupSize, layer.Bits)
	}
}

func TestGemma4_SanitizeWeights_GateUpProj_Good(t *testing.T) {
	requireMetalRuntime(t)

	gateUp := metal.FromValues([]float32{
		1, 2,
		3, 4,
		5, 6,
		7, 8,
	}, 1, 4, 2)
	metal.Materialize(gateUp)
	vision := metal.FromValues([]float32{1}, 1)
	rotary := metal.FromValues([]float32{1}, 1)

	sanitized := sanitizeGemma4Weights(map[string]*metal.Array{
		"model.layers.0.experts.gate_up_proj.weight": gateUp,
		"model.vision_tower.block.weight":            vision,
		"model.layers.0.self_attn.rotary_emb.inv":    rotary,
	})

	gate := sanitized["model.layers.0.experts.switch_glu.gate_proj.weight"]
	up := sanitized["model.layers.0.experts.switch_glu.up_proj.weight"]
	fused := sanitized["model.layers.0.experts.switch_glu.gate_up_proj.weight"]
	if gate == nil || up == nil {
		t.Fatal("expected split switch_glu gate_proj and up_proj weights")
	}
	if fused != gateUp {
		t.Fatal("expected sanitization to retain fused switch_glu gate_up_proj weight")
	}
	if _, ok := sanitized["model.layers.0.experts.gate_up_proj.weight"]; ok {
		t.Fatal("legacy gate_up_proj key should be replaced by switch_glu keys")
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
	if !gateUp.Valid() {
		t.Fatal("gate_up source tensor should be retained for fused expert projection")
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

	biases := metal.FromValues([]float32{
		1, 2, 3, 4,
		5, 6, 7, 8,
	}, 2, 4)
	metal.Materialize(biases)

	sanitized := sanitizeGemma4Weights(map[string]*metal.Array{
		"model.layers.0.experts.gate_up_proj.biases": biases,
	})

	gate := sanitized["model.layers.0.experts.switch_glu.gate_proj.biases"]
	up := sanitized["model.layers.0.experts.switch_glu.up_proj.biases"]
	fused := sanitized["model.layers.0.experts.switch_glu.gate_up_proj.biases"]
	if gate == nil || up == nil {
		t.Fatal("expected split switch_glu gate_proj and up_proj biases")
	}
	if fused != biases {
		t.Fatal("expected fused switch_glu gate_up_proj biases to be retained")
	}
	if got := gate.Shape(); len(got) != 2 || got[0] != 2 || got[1] != 2 {
		t.Fatalf("gate bias split shape = %v, want [2 2]", got)
	}
	if got := up.Shape(); len(got) != 2 || got[0] != 2 || got[1] != 2 {
		t.Fatalf("up bias split shape = %v, want [2 2]", got)
	}
}

func TestGemma4_Experts_FusedGateUpMatchesSplit_Good(t *testing.T) {
	requireMetalRuntime(t)

	expertWeight := func(e0, e1 []float32) *metal.Array {
		data := append(append([]float32{}, e0...), e1...)
		return metal.FromValues(data, 2, 2, 2)
	}
	gateValues0 := []float32{1.0, 0.2, -0.1, 0.7}
	gateValues1 := []float32{0.3, -0.6, 0.9, 0.1}
	upValues0 := []float32{0.5, -0.4, 0.8, 0.2}
	upValues1 := []float32{-0.2, 0.4, 0.1, 0.6}
	downValues0 := []float32{0.6, -0.2, 0.4, 0.8}
	downValues1 := []float32{0.1, 0.5, -0.3, 0.7}

	splitGateWeight := expertWeight(gateValues0, gateValues1)
	splitUpWeight := expertWeight(upValues0, upValues1)
	splitDownWeight := expertWeight(downValues0, downValues1)
	fusedGateWeight := expertWeight(gateValues0, gateValues1)
	fusedUpWeight := expertWeight(upValues0, upValues1)
	fusedWeight := metal.Concatenate([]*metal.Array{fusedGateWeight, fusedUpWeight}, 1)
	metal.Materialize(fusedWeight)
	metal.Free(fusedGateWeight, fusedUpWeight)
	fusedDownWeight := expertWeight(downValues0, downValues1)

	splitExperts := &Gemma4Experts{
		GateProj: metal.NewSwitchLinear(splitGateWeight, nil),
		UpProj:   metal.NewSwitchLinear(splitUpWeight, nil),
		DownProj: metal.NewSwitchLinear(splitDownWeight, nil),
	}
	fusedExperts := &Gemma4Experts{
		GateUpProj: metal.NewSwitchLinear(fusedWeight, nil),
		GateProj:   metal.NewSwitchLinear(expertWeight(gateValues0, gateValues1), nil),
		UpProj:     metal.NewSwitchLinear(expertWeight(upValues0, upValues1), nil),
		DownProj:   metal.NewSwitchLinear(fusedDownWeight, nil),
	}
	defer func() {
		metal.FreeSwitchLinear(splitExperts.GateProj)
		metal.FreeSwitchLinear(splitExperts.UpProj)
		metal.FreeSwitchLinear(splitExperts.DownProj)
		metal.FreeSwitchLinear(fusedExperts.GateUpProj)
		metal.FreeSwitchLinear(fusedExperts.GateProj)
		metal.FreeSwitchLinear(fusedExperts.UpProj)
		metal.FreeSwitchLinear(fusedExperts.DownProj)
	}()

	x := metal.FromValues([]float32{0.25, -0.75}, 1, 1, 2)
	topKIndices := metal.FromValues([]int32{1}, 1, 1, 1)
	topKWeights := metal.FromValues([]float32{0.8}, 1, 1, 1)
	defer metal.Free(x, topKIndices, topKWeights)

	want := splitExperts.forward(x, topKIndices, topKWeights, "")
	got := fusedExperts.forward(x, topKIndices, topKWeights, "")
	defer metal.Free(want, got)

	if err := metal.Eval(want, got); err != nil {
		t.Fatalf("Eval: %v", err)
	}
	floatSliceApprox(t, got.Floats(), want.Floats())
}

func TestGemma4_Experts_FusedGateUpDecodeOnly_Bad(t *testing.T) {
	requireMetalRuntime(t)

	decode := metal.FromValues([]float32{0.25, -0.75}, 1, 1, 2)
	prefill := metal.FromValues([]float32{
		0.25, -0.75,
		0.5, 0.125,
	}, 1, 2, 2)
	defer metal.Free(decode, prefill)

	if !gemma4UseFusedExpertGateUp(decode) {
		t.Fatal("single-token decode should use fused gate_up projection")
	}
	if gemma4UseFusedExpertGateUp(prefill) {
		t.Fatal("multi-token prefill should keep split gate/up projections")
	}
}

func TestGemma4_SanitizeWeights_DownProjRemap_Good(t *testing.T) {
	requireMetalRuntime(t)

	down := metal.FromValues([]float32{
		1, 2,
		3, 4,
	}, 1, 2, 2)
	metal.Materialize(down)

	sanitized := sanitizeGemma4Weights(map[string]*metal.Array{
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
	metal.Free(down)
}

func TestGemma4_SanitizeWeights_LanguageModelPrefix_Good(t *testing.T) {
	sanitized := sanitizeGemma4Weights(map[string]*metal.Array{
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
	raw := map[string]*metal.Array{
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

func TestGemma4_SanitizeAudioWeights_Good(t *testing.T) {
	raw := map[string]*metal.Array{
		"language_model.embed_audio.embedding_projection.weight": nil,
		"language_model.embed_audio.embedding_projection.scales": nil,
		"language_model.embed_audio.embedding_projection.biases": nil,
		"language_model.model.embed_tokens.weight":               nil,
	}

	audio := sanitizeGemma4AudioWeights(raw)
	if _, ok := audio["embed_audio.embedding_projection.weight"]; !ok {
		t.Fatal("expected official embed_audio projection weight to be retained")
	}
	if _, ok := audio["embed_audio.embedding_projection.scales"]; !ok {
		t.Fatal("expected official embed_audio projection scales to be retained")
	}
	if _, ok := audio["embed_audio.embedding_projection.biases"]; !ok {
		t.Fatal("expected official embed_audio projection biases to be retained")
	}
	if _, ok := raw["language_model.embed_audio.embedding_projection.weight"]; ok {
		t.Fatal("expected audio weight to be removed from raw map before text sanitization")
	}
	if _, ok := raw["language_model.model.embed_tokens.weight"]; !ok {
		t.Fatal("expected text weight to remain in raw map")
	}
}

func TestGemma4_AudioProjectionRetainedWeights_Good(t *testing.T) {
	requireMetalRuntime(t)
	weight := seqArray(0.1, 4, 2)
	bias := seqArray(0.2, 4)
	weights := map[string]*metal.Array{
		"embed_audio.embedding_projection.weight": weight,
		"embed_audio.embedding_projection.bias":   bias,
	}

	projector := buildGemma4AudioProjector(&Gemma4TextConfig{
		TransformerConfig: metal.TransformerConfig{HiddenSize: 4},
		AudioConfig:       normalizeGemma4AudioConfig(&Gemma4AudioConfig{HiddenSize: 2, OutputProjDims: 4}),
	}, weights)
	if projector == nil || projector.Projection == nil {
		t.Fatal("audio projector = nil, want official embed_audio projection")
	}
	defer closeGemma4AudioProjector(projector)

	model := &Gemma4Model{AudioProjector: projector}
	retained := gemma4RetainedWeights(model)
	if !arraySetContains(retained, weight) || !arraySetContains(retained, bias) {
		t.Fatal("audio projector weights were not retained by Gemma4Model")
	}
}

func TestGemma4_VisionProjectionWithoutTowerRetainedWeights_Good(t *testing.T) {
	requireMetalRuntime(t)
	weight := seqArray(0.1, 4, 2)
	scales := seqArray(0.2, 4, 1)
	biases := seqArray(0.3, 4, 1)
	visionWeights := map[string]*metal.Array{
		"embed_vision.embedding_projection.weight": weight,
		"embed_vision.embedding_projection.scales": scales,
		"embed_vision.embedding_projection.biases": biases,
	}

	vision, projector, err := buildGemma4VisionComponents(&Gemma4TextConfig{
		TransformerConfig: metal.TransformerConfig{HiddenSize: 4},
		VisionConfig: normalizeGemma4VisionConfig(&Gemma4VisionConfig{
			TransformerConfig: metal.TransformerConfig{HiddenSize: 2},
			MMEmbedDim:        2,
			OutputProjDims:    4,
		}),
	}, visionWeights)
	if err != nil {
		t.Fatalf("buildGemma4VisionComponents: %v", err)
	}
	if vision != nil {
		t.Fatal("vision tower = non-nil, want encoder-free projection-only Unified path")
	}
	if projector == nil || projector.Projection == nil {
		t.Fatal("projector = nil, want official embed_vision projection retained without tower")
	}
	if projector.Projection.Scales != scales || projector.Projection.Biases != biases {
		t.Fatal("projection-only vision quant side tensors were not attached")
	}
	defer closeGemma4Vision(vision, projector)

	model := &Gemma4Model{MultiModalProjector: projector}
	retained := gemma4RetainedWeights(model)
	if !arraySetContains(retained, weight) || !arraySetContains(retained, scales) || !arraySetContains(retained, biases) {
		t.Fatal("projection-only vision weight was not retained by Gemma4Model")
	}
}

func TestGemma4_UnifiedVisionComponentPolicyEncoderFree_Good(t *testing.T) {
	cfg := &Gemma4TextConfig{
		TransformerConfig: metal.TransformerConfig{
			ModelType:  "gemma4_unified",
			HiddenSize: 4,
		},
		VisionConfig: normalizeGemma4VisionConfig(&Gemma4VisionConfig{
			TransformerConfig: metal.TransformerConfig{
				ModelType:  "gemma4_unified_vision",
				HiddenSize: 2,
			},
			MMEmbedDim:     2,
			OutputProjDims: 4,
		}),
	}

	if gemma4VisionShouldBuildEncoderTower(cfg) {
		t.Fatal("gemma4VisionShouldBuildEncoderTower(unified) = true, want encoder-free projection path")
	}
	cfg.ModelType = "gemma4"
	cfg.VisionConfig.ModelType = "gemma4_vision"
	if !gemma4VisionShouldBuildEncoderTower(cfg) {
		t.Fatal("gemma4VisionShouldBuildEncoderTower(encoder model) = false, want tower path preserved for non-unified Gemma4")
	}
}

func TestGemma4_EncodeImagesUsesProjectionWithoutTower_Good(t *testing.T) {
	requireMetalRuntime(t)
	projector := &Gemma4MultiModalProjector{Projection: metal.NewLinear(seqArray(0.1, 4, 2), nil), Eps: 1e-6}
	defer closeGemma4Vision(nil, projector)
	model := &Gemma4Model{MultiModalProjector: projector}
	imagePatches := seqArray(1, 1, 2)

	got := model.encodeGemma4Images([]*metal.Array{imagePatches})
	defer metal.Free(got, imagePatches)
	if got == nil || !got.Valid() {
		t.Fatal("encodeGemma4Images() = nil, want projection output without a vision tower")
	}
	if got.Dim(1) != 4 {
		t.Fatalf("projected image hidden dim = %d, want 4", got.Dim(1))
	}
}

func TestGemma4_InjectAudioFeatures_Good(t *testing.T) {
	requireMetalRuntime(t)
	model := &Gemma4Model{Cfg: &Gemma4TextConfig{AudioTokenID: 258881}}
	h := seqArray(0, 1, 3, 4)
	features := seqArray(10, 1, 4)

	got := model.injectGemma4TokenFeatures(h, []int32{7, 258881, 9}, []int32{1, 3}, features, 258881, "audio")
	defer metal.Free(got, features)
	if err := metal.Eval(got); err != nil {
		t.Fatalf("Eval injected audio features: %v", err)
	}
	values := got.Floats()
	floatSliceApprox(t, values[4:8], []float32{10, 10.01, 10.02, 10.03})
	floatSliceApprox(t, values[0:4], []float32{0, 0.01, 0.02, 0.03})
}

func TestGemma4_SanitizeWeights_RepeatedWrapperPrefixes_Good(t *testing.T) {
	sanitized := sanitizeGemma4Weights(map[string]*metal.Array{
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

func gemma4TestPatternLayers(numLayers int, pattern int32) []*Gemma4DecoderLayer {
	layers := make([]*Gemma4DecoderLayer, numLayers)
	for i := range layers {
		layerType := "full_attention"
		if pattern > 1 && (i+1)%int(pattern) != 0 {
			layerType = "sliding_attention"
		}
		if i == len(layers)-1 {
			layerType = "full_attention"
		}
		layers[i] = &Gemma4DecoderLayer{
			LayerType: layerType,
			IsSliding: layerType == "sliding_attention",
		}
	}
	return layers
}

func TestGemma4_E4BSharedCacheLayoutUsesLayerTypes_Good(t *testing.T) {
	layers := gemma4TestPatternLayers(42, 6)

	previous, cacheIndexByLayer := buildGemma4CacheLayout(layers, 18)

	ownerCount := 0
	for _, cacheIdx := range cacheIndexByLayer {
		if cacheIdx >= 0 {
			ownerCount++
		}
	}
	if ownerCount != 24 {
		t.Fatalf("owner cache count = %d, want 24 pre-sharing owners", ownerCount)
	}
	if previous[24] != 22 {
		t.Fatalf("PreviousKVs[24] = %d, want sliding owner 22", previous[24])
	}
	if previous[29] != 23 || previous[41] != 23 {
		t.Fatalf("full shared PreviousKVs = %d/%d, want owner 23", previous[29], previous[41])
	}
	if cacheIndexByLayer[24] != -1 || cacheIndexByLayer[29] != -1 || cacheIndexByLayer[41] != -1 {
		t.Fatalf("shared layers allocated caches: layer24=%d layer29=%d layer41=%d", cacheIndexByLayer[24], cacheIndexByLayer[29], cacheIndexByLayer[41])
	}

	model := &Gemma4Model{
		Cfg: &Gemma4TextConfig{
			TransformerConfig: metal.TransformerConfig{
				NumHiddenLayers: 42,
			},
			NumKVSharedLayers: 18,
			SlidingWindow:     512,
		},
		Layers: layers,
	}
	caches := model.NewCache()
	if len(caches) != 24 {
		t.Fatalf("len(caches) = %d, want 24", len(caches))
	}
	sliding, ok := caches[0].(*metal.RotatingKVCache)
	if !ok {
		t.Fatalf("cache[0] = %T, want *RotatingKVCache", caches[0])
	}
	if sliding.MaxSize() != 512 {
		t.Fatalf("sliding cache maxSize = %d, want 512", sliding.MaxSize())
	}
	if _, ok := caches[5].(*metal.KVCache); !ok {
		t.Fatalf("cache[5] = %T, want *KVCache for first full-attention owner", caches[5])
	}
}

func TestGemma4_SharedKVInvalidPages_Bad(t *testing.T) {
	kv := sharedKV{
		Pages: metal.PagedKVState{
			Keys:   []*metal.Array{nil},
			Values: []*metal.Array{nil},
		},
	}
	if kv.HasPages() {
		t.Fatal("nil page handles should not count as usable K/V state")
	}
	if kv.HasState() {
		t.Fatal("invalid pages should not count as usable K/V state")
	}
}

func TestGemma4_SharedKVBorrowedFreePreservesFixedState_Good(t *testing.T) {
	requireMetalRuntime(t)

	keys := metal.FromValues([]float32{1, 2}, 1, 1, 1, 2)
	values := metal.FromValues([]float32{3, 4}, 1, 1, 1, 2)
	defer metal.Free(keys, values)

	kv := sharedKV{Keys: keys, Values: values, Fixed: true, Borrowed: true}
	kv.Free()

	if !keys.Valid() || !values.Valid() {
		t.Fatal("borrowed sharedKV.free invalidated cache-owned fixed K/V handles")
	}
}

func TestGemma4_SharedKVCloneRetainsBorrowedFixedState_Good(t *testing.T) {
	requireMetalRuntime(t)

	keys := metal.FromValues([]float32{1, 2}, 1, 1, 1, 2)
	values := metal.FromValues([]float32{3, 4}, 1, 1, 1, 2)
	kv := sharedKV{Keys: keys, Values: values, Fixed: true, Borrowed: true}

	retained := kv.Clone()
	kv.Free()
	metal.Free(keys, values)
	defer retained.Free()

	if !retained.HasState() {
		t.Fatal("retained sharedKV clone lost fixed K/V handles after original cache wrappers were freed")
	}
	if retained.Borrowed {
		t.Fatal("retained sharedKV clone should own its ref-counted handles")
	}
}

func TestGemma4_SharedKVCloneRetainsBorrowedPagedState_Good(t *testing.T) {
	requireMetalRuntime(t)

	k, v := makeSingleTokenKVShape(1, 2, 4)
	defer metal.Free(k, v)

	cache := metal.NewPagedKVCache(0, 2)
	pages := cache.UpdateBorrowedPages(k, v, 1)
	kv := sharedKV{Pages: pages, Offset: cache.Offset()}
	retained := kv.Clone()
	kv.Free()
	cache.Reset()
	defer retained.Free()

	if !retained.HasPages() {
		t.Fatal("retained sharedKV clone lost paged K/V handles after source cache reset")
	}
	if len(retained.Pages.Owned) != 2 {
		t.Fatalf("retained owned page handles = %d, want 2", len(retained.Pages.Owned))
	}
}

func TestGemma4_SharedKVMoveTransfersOwnerWithoutClone_Good(t *testing.T) {
	requireMetalRuntime(t)

	k, v := makeSingleTokenKVShape(1, 2, 4)
	defer metal.Free(k, v)

	cache := metal.NewPagedKVCache(0, 2)
	pages := cache.UpdateBorrowedPages(k, v, 1)
	kv := sharedKV{Pages: pages, Offset: cache.Offset()}
	retained := moveSharedKV(&kv)
	defer cache.Reset()
	defer retained.Free()

	if kv.HasState() || kv.HasPages() {
		t.Fatal("moved sharedKV source still owns state")
	}
	if !retained.HasPages() {
		t.Fatal("moved sharedKV lost paged state")
	}
	if len(retained.Pages.Owned) != len(pages.Owned) {
		t.Fatalf("moved owned page handles = %d, want %d", len(retained.Pages.Owned), len(pages.Owned))
	}
	if len(retained.Pages.Keys) == 0 || retained.Pages.Keys[0] != pages.Keys[0] {
		t.Fatal("moved sharedKV cloned or replaced borrowed page handles")
	}
}

func TestGemma4_NewCache_SharedLayers_Good(t *testing.T) {
	model := &Gemma4Model{
		Cfg: &Gemma4TextConfig{
			TransformerConfig: metal.TransformerConfig{
				NumHiddenLayers: 4,
			},
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
	if _, ok := caches[0].(*metal.RotatingKVCache); !ok {
		t.Fatalf("cache[0] = %T, want *RotatingKVCache", caches[0])
	}
	if _, ok := caches[1].(*metal.KVCache); !ok {
		t.Fatalf("cache[1] = %T, want *KVCache", caches[1])
	}
}

func TestGemma4_NewCache_PromotedOwner_Good(t *testing.T) {
	model := &Gemma4Model{
		Cfg: &Gemma4TextConfig{
			TransformerConfig: metal.TransformerConfig{
				NumHiddenLayers: 6,
			},
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
	if _, ok := caches[4].(*metal.KVCache); !ok {
		t.Fatalf("cache[4] = %T, want *KVCache for promoted full-attention owner", caches[4])
	}
	if got := model.PreviousKVs[4]; got != 4 {
		t.Fatalf("PreviousKVs[4] = %d, want 4", got)
	}
	if got := model.CacheIndexByLayer[4]; got != 4 {
		t.Fatalf("CacheIndexByLayer[4] = %d, want 4", got)
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
		"max_position_embeddings": 16,
		"rms_norm_eps": 1e-6,
		"sliding_window": 4,
		"sliding_window_pattern": 2,
		"num_kv_shared_layers": 0,
		"hidden_size_per_layer_input": 0,
		"use_double_wide_mlp": false,
		"layer_types": ["sliding_attention", "full_attention"]
	}`
	if err := coreio.Local.Write(core.JoinPath(dir, "config.json"), config); err != nil {
		t.Fatalf("write config.json: %v", err)
	}
	writeMinimalTokenizer(t, dir)
	if err := metal.SaveSafetensors(core.JoinPath(dir, "model.safetensors"), gemma4TinyWeights()); err != nil {
		t.Fatalf("SaveSafetensors: %v", err)
	}

	model, err := LoadGemma4(dir)
	if err != nil {
		t.Fatalf("LoadGemma4: %v", err)
	}
	defer closeGemma4(model)

	tokens := metal.FromValues([]int32{2, 3, 4}, 1, 3)
	caches := model.NewCache()
	logits := model.Forward(tokens, caches)
	if err := metal.Eval(logits); err != nil {
		t.Fatalf("Eval logits: %v", err)
	}
	defer func() {
		metal.Free(tokens, logits)
		metal.FreeCaches(caches)
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
		"max_position_embeddings": 16,
		"rms_norm_eps": 1e-6,
		"sliding_window": 2,
		"sliding_window_pattern": 2,
		"num_kv_shared_layers": 0,
		"hidden_size_per_layer_input": 0,
		"use_double_wide_mlp": false,
		"layer_types": ["sliding_attention", "full_attention"]
	}`
	if err := coreio.Local.Write(core.JoinPath(dir, "config.json"), config); err != nil {
		t.Fatalf("write config.json: %v", err)
	}
	writeMinimalTokenizer(t, dir)
	if err := metal.SaveSafetensors(core.JoinPath(dir, "model.safetensors"), gemma4TinyWeights()); err != nil {
		t.Fatalf("SaveSafetensors: %v", err)
	}

	model, err := LoadGemma4(dir)
	if err != nil {
		t.Fatalf("LoadGemma4: %v", err)
	}
	defer closeGemma4(model)

	tokens := metal.FromValues([]int32{2, 3, 4, 5}, 1, 4)
	caches := model.NewCache()
	logits := model.Forward(tokens, caches)
	if err := metal.Eval(logits); err != nil {
		t.Fatalf("Eval logits: %v", err)
	}
	defer func() {
		metal.Free(tokens, logits)
		metal.FreeCaches(caches)
	}()

	shape := logits.Shape()
	if len(shape) != 3 {
		t.Fatalf("logits dims = %v, want rank 3", shape)
	}
	if shape[0] != 1 || shape[1] != 4 || shape[2] != 10 {
		t.Fatalf("logits shape = %v, want [1 4 10]", shape)
	}
}

func TestGemma4_LastSequenceHidden_Good_HandlesRankVariants(t *testing.T) {
	requireMetalRuntime(t)

	rank3 := metal.FromValues([]float32{
		1, 2,
		3, 4,
		5, 6,
	}, 1, 3, 2)
	last3 := gemma4LastSequenceHidden(rank3, 3)
	defer metal.Free(last3)
	if got := last3.Shape(); len(got) != 3 || got[0] != 1 || got[1] != 1 || got[2] != 2 {
		t.Fatalf("rank3 last shape = %v, want [1 1 2]", got)
	}

	rank2 := metal.FromValues([]float32{
		1, 2,
		3, 4,
		5, 6,
	}, 3, 2)
	last2 := gemma4LastSequenceHidden(rank2, 3)
	if got := last2.Shape(); len(got) != 2 || got[0] != 1 || got[1] != 2 {
		t.Fatalf("rank2 last shape = %v, want [1 2]", got)
	}
	proj2 := gemma4ProjectionHidden(last2)
	if got := proj2.Shape(); len(got) != 3 || got[0] != 1 || got[1] != 1 || got[2] != 2 {
		t.Fatalf("rank2 projection shape = %v, want [1 1 2]", got)
	}
	contig2 := gemma4ContiguousHidden(proj2)
	defer metal.Free(contig2)
	if err := metal.Eval(contig2); err != nil {
		t.Fatalf("Eval(contig2) error = %v", err)
	}
	if !contig2.IsRowContiguous() {
		t.Fatalf("rank2 projection is not contiguous")
	}

	rank1 := metal.FromValues([]float32{1, 2}, 2)
	last1 := gemma4LastSequenceHidden(rank1, 3)
	if got := last1.Shape(); len(got) != 1 || got[0] != 2 {
		t.Fatalf("rank1 last shape = %v, want [2]", got)
	}
	proj1 := gemma4ProjectionHidden(last1)
	defer metal.Free(proj1)
	if got := proj1.Shape(); len(got) != 3 || got[0] != 1 || got[1] != 1 || got[2] != 2 {
		t.Fatalf("rank1 projection shape = %v, want [1 1 2]", got)
	}
}

func TestGemma4_CachedAttentionMask_Good_OffsetsAndWindow(t *testing.T) {
	requireMetalRuntime(t)

	mask := buildGemma4CachedAttentionMask(1, 2, 5, 3, 0, 2)
	defer metal.Free(mask)
	values := mask.Floats()
	if len(values) != 10 {
		t.Fatalf("mask values = %d, want 10", len(values))
	}
	negInf := float32(math.Inf(-1))
	want := []float32{
		negInf, negInf, 0, 0, negInf,
		negInf, negInf, negInf, 0, 0,
	}
	for i := range want {
		if values[i] != want[i] {
			t.Fatalf("mask[%d] = %v, want %v (all=%v)", i, values[i], want[i], values)
		}
	}
}

func TestGemma4_CachedAttentionMask_Good_TrimmedKeyStart(t *testing.T) {
	requireMetalRuntime(t)

	mask := buildGemma4CachedAttentionMask(1, 2, 5, 8, 5, 4)
	defer metal.Free(mask)
	values := mask.Floats()
	if len(values) != 10 {
		t.Fatalf("mask values = %d, want 10", len(values))
	}
	negInf := float32(math.Inf(-1))
	want := []float32{
		0, 0, 0, 0, negInf,
		negInf, 0, 0, 0, 0,
	}
	for i := range want {
		if values[i] != want[i] {
			t.Fatalf("mask[%d] = %v, want %v (all=%v)", i, values[i], want[i], values)
		}
	}
}

func TestGemma4_RuntimeMaskCache_Good_ReusesChunkMasks(t *testing.T) {
	requireMetalRuntime(t)

	cache := newGemma4RuntimeMaskCache()
	defer cache.Free()

	first := cache.CachedAttentionMask(1, 2, 5, 8, 5, 4)
	second := cache.CachedAttentionMask(1, 2, 5, 8, 5, 4)
	if first == nil || !first.Valid() {
		t.Fatal("first cached attention mask is invalid")
	}
	if first != second {
		t.Fatal("cached attention mask was rebuilt for identical shape/window")
	}
	if len(cache.owned) != 1 {
		t.Fatalf("runtime mask cache owns %d masks, want 1", len(cache.owned))
	}

	otherWindow := cache.CachedAttentionMask(1, 2, 5, 8, 5, 2)
	if otherWindow == nil || !otherWindow.Valid() {
		t.Fatal("other-window cached attention mask is invalid")
	}
	if otherWindow == first {
		t.Fatal("runtime mask cache reused a mask with a different sliding window")
	}
	if len(cache.owned) != 2 {
		t.Fatalf("runtime mask cache owns %d masks after window split, want 2", len(cache.owned))
	}
}

func TestGemma4_SlidingCausalContextLen_Good(t *testing.T) {
	if got := gemma4SlidingCausalContextLen(512, 1024, 512); got != 1023 {
		t.Fatalf("context len = %d, want 1023 for previous window plus current chunk", got)
	}
	if got := gemma4SlidingCausalContextLen(128, 2048, 512); got != 639 {
		t.Fatalf("context len = %d, want 639 for 512-token window and 128-token chunk", got)
	}
	if got := gemma4SlidingCausalContextLen(513, 2048, 512); got != 2048 {
		t.Fatalf("context len = %d, want full key span when chunk exceeds window", got)
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
		"max_position_embeddings": 16,
		"rms_norm_eps": 1e-6,
		"sliding_window": 4,
		"sliding_window_pattern": 2,
		"num_kv_shared_layers": 0,
		"hidden_size_per_layer_input": 0,
		"use_double_wide_mlp": false,
		"layer_types": ["sliding_attention", "full_attention"]
	}`
	if err := coreio.Local.Write(core.JoinPath(dir, "config.json"), config); err != nil {
		t.Fatalf("write config.json: %v", err)
	}
	writeMinimalTokenizer(t, dir)
	if err := metal.SaveGGUF(core.JoinPath(dir, "model.gguf"), gemma4TinyWeights()); err != nil {
		t.Fatalf("SaveGGUF: %v", err)
	}

	model, err := LoadGemma4(core.JoinPath(dir, "model.gguf"))
	if err != nil {
		t.Fatalf("LoadGemma4: %v", err)
	}
	defer closeGemma4(model)

	tokens := metal.FromValues([]int32{2, 3, 4}, 1, 3)
	caches := model.NewCache()
	logits := model.Forward(tokens, caches)
	if err := metal.Eval(logits); err != nil {
		t.Fatalf("Eval logits: %v", err)
	}
	defer func() {
		metal.Free(tokens, logits)
		metal.FreeCaches(caches)
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
			"max_position_embeddings": 16,
			"rms_norm_eps": 1e-6,
			"sliding_window": 4,
			"sliding_window_pattern": 2,
			"num_kv_shared_layers": 0,
			"hidden_size_per_layer_input": 0,
			"use_double_wide_mlp": false,
			"layer_types": ["sliding_attention", "full_attention"]
		}
	}`
	if err := coreio.Local.Write(core.JoinPath(dir, "config.json"), config); err != nil {
		t.Fatalf("write config.json: %v", err)
	}
	writeMinimalTokenizer(t, dir)

	weights := gemma4TinyWeights()
	weights["vision_tower.encoder.weight"] = metal.FromValues([]float32{1, 2, 3, 4}, 2, 2)
	weights["language_model.model.layers.0.self_attn.rotary_emb.inv_freq"] = metal.FromValues([]float32{1, 2}, 2)
	defer metal.Free(weights["vision_tower.encoder.weight"], weights["language_model.model.layers.0.self_attn.rotary_emb.inv_freq"])
	if err := metal.SaveSafetensors(core.JoinPath(dir, "model.safetensors"), weights); err != nil {
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

	tokens := metal.FromValues([]int32{2, 3, 4}, 1, 3)
	caches := model.NewCache()
	logits := model.Forward(tokens, caches)
	if err := metal.Eval(logits); err != nil {
		t.Fatalf("Eval logits: %v", err)
	}
	defer func() {
		metal.Free(tokens, logits)
		metal.FreeCaches(caches)
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
		"max_position_embeddings": 16,
		"rms_norm_eps": 1e-6,
		"sliding_window": 4,
		"sliding_window_pattern": 2,
		"num_kv_shared_layers": 0,
		"tie_word_embeddings": false,
		"use_double_wide_mlp": false,
		"layer_types": ["sliding_attention", "full_attention"]
	}`
	if err := coreio.Local.Write(core.JoinPath(dir, "config.json"), config); err != nil {
		t.Fatalf("write config.json: %v", err)
	}
	writeMinimalTokenizer(t, dir)

	weights := gemma4TinyWeights()
	if err := metal.SaveSafetensors(core.JoinPath(dir, "model.safetensors"), weights); err != nil {
		t.Fatalf("SaveSafetensors: %v", err)
	}
	freeWeightMap(weights)
	metal.ClearCache()

	baseline := metal.GetActiveMemory()
	_, err := LoadGemma4(dir)
	if err == nil {
		t.Fatal("expected untied Gemma4 load to fail without lm_head.weight")
	}
	if !core.Contains(err.Error(), "lm_head.weight") {
		t.Fatalf("expected lm_head.weight error, got: %v", err)
	}

	activeAfterFailure := metal.GetActiveMemory()
	if activeAfterFailure > baseline {
		t.Fatalf("active memory after failed load = %d, want <= %d", activeAfterFailure, baseline)
	}
}

func TestGemma4_DecoderLayer_MoEAppliesFinalPostFFNorm_Good(t *testing.T) {
	requireMetalRuntime(t)

	zeros2x2 := func() *metal.Array {
		return metal.FromValues([]float32{
			0, 0,
			0, 0,
		}, 2, 2)
	}
	ones2 := func() *metal.Array {
		return metal.FromValues([]float32{1, 1}, 2)
	}
	switchWeight := func(scale float32) *metal.Array {
		return metal.FromValues([]float32{
			scale, 0,
			0, scale,
		}, 1, 2, 2)
	}

	layer := &Gemma4DecoderLayer{
		Attention: &Gemma4Attention{
			QProj:          metal.NewLinear(zeros2x2(), nil),
			KProj:          metal.NewLinear(zeros2x2(), nil),
			VProj:          metal.NewLinear(zeros2x2(), nil),
			OProj:          metal.NewLinear(zeros2x2(), nil),
			QNormScaled:    ones2(),
			KNormScaled:    ones2(),
			HeadDim:        2,
			NKVHeads:       1,
			Scale:          1.0,
			RopeBase:       10000,
			RopeRotatedDim: 2,
		},
		MLP: &metal.MLP{
			GateProj: metal.NewLinear(metal.FromValues([]float32{
				0.8, 0.1,
				0.2, 0.7,
			}, 2, 2), nil),
			UpProj: metal.NewLinear(metal.FromValues([]float32{
				0.5, -0.1,
				0.3, 0.6,
			}, 2, 2), nil),
			DownProj: metal.NewLinear(metal.FromValues([]float32{
				0.4, 0.2,
				-0.3, 0.9,
			}, 2, 2), nil),
		},
		EnableMoE:          true,
		InputNormScaled:    ones2(),
		PostAttnNormScaled: ones2(),
		PreFFNormScaled:    ones2(),
		PostFFNormScaled:   metal.FromValues([]float32{2.0, 0.5}, 2),
		PreFFNorm2Scaled:   ones2(),
		PostFFNorm1Scaled:  ones2(),
		PostFFNorm2Scaled:  ones2(),
		Router: &Gemma4Router{
			Proj:           metal.NewLinear(metal.FromValues([]float32{1.0, -0.25}, 1, 2), nil),
			Scale:          ones2(),
			PerExpertScale: metal.FromValues([]float32{1}, 1),
			ScaleScaled:    ones2(),
			TopK:           1,
			Eps:            1e-6,
		},
		Experts: &Gemma4Experts{
			GateProj: metal.NewSwitchLinear(switchWeight(0.9), nil),
			UpProj:   metal.NewSwitchLinear(switchWeight(0.6), nil),
			DownProj: metal.NewSwitchLinear(switchWeight(0.7), nil),
		},
	}
	defer closeGemma4(&Gemma4Model{Layers: []*Gemma4DecoderLayer{layer}})

	cfg := &Gemma4TextConfig{
		TransformerConfig: metal.TransformerConfig{
			HiddenSize:        2,
			NumAttentionHeads: 1,
			NumKeyValueHeads:  1,
			RMSNormEps:        1e-6,
		},
	}
	x := metal.FromValues([]float32{0.3, -0.2}, 1, 1, 2)

	got, kv := layer.forward(x, nil, 1, 1, nil, nil, sharedKV{}, cfg, nil, nil, false)
	defer metal.Free(kv.Keys, kv.Values)

	h1In := metal.RMSNorm(x, layer.PreFFNormScaled, cfg.RMSNormEps)
	h1 := layer.MLP.Forward(h1In)
	metal.Free(h1In)
	h1Normed := metal.RMSNorm(h1, layer.PostFFNorm1Scaled, cfg.RMSNormEps)
	metal.Free(h1)

	h2In := metal.RMSNorm(x, layer.PreFFNorm2Scaled, cfg.RMSNormEps)
	topKIndices, topKWeights := layer.Router.forward(x)
	h2 := layer.Experts.forward(h2In, topKIndices, topKWeights, "")
	metal.Free(h2In, topKIndices, topKWeights)
	h2Normed := metal.RMSNorm(h2, layer.PostFFNorm2Scaled, cfg.RMSNormEps)
	metal.Free(h2)

	combined := metal.Add(h1Normed, h2Normed)
	metal.Free(h1Normed, h2Normed)
	combinedNormed := metal.RMSNorm(combined, layer.PostFFNormScaled, cfg.RMSNormEps)
	metal.Free(combined)
	want := metal.Add(x, combinedNormed)
	metal.Free(combinedNormed)

	if err := metal.Eval(got, want); err != nil {
		t.Fatalf("Eval: %v", err)
	}
	defer metal.Free(x, got, want)

	floatSliceApprox(t, got.Floats(), want.Floats())
}

type gemma4TestFFNMemoryAugmenter struct {
	layerID int32
	scale   float32
	called  bool
}

func (a *gemma4TestFFNMemoryAugmenter) AugmentFFNMemory(layerID int32, ffnOutput, mlpInput *metal.Array) (*metal.Array, bool, error) {
	a.layerID = layerID
	a.called = true
	delta := metal.MulScalar(mlpInput, a.scale)
	out := metal.Add(ffnOutput, delta)
	metal.Free(delta)
	return out, true, nil
}

func TestGemma4_DecoderLayer_FFNMemoryAugmenterAddsBeforePostFFNorm_Good(t *testing.T) {
	requireMetalRuntime(t)

	zeros2x2 := func() *metal.Array {
		return metal.FromValues([]float32{
			0, 0,
			0, 0,
		}, 2, 2)
	}
	ones2 := func() *metal.Array {
		return metal.FromValues([]float32{1, 1}, 2)
	}
	augmenter := &gemma4TestFFNMemoryAugmenter{scale: 2}
	layer := &Gemma4DecoderLayer{
		Attention: &Gemma4Attention{
			QProj:          metal.NewLinear(zeros2x2(), nil),
			KProj:          metal.NewLinear(zeros2x2(), nil),
			VProj:          metal.NewLinear(zeros2x2(), nil),
			OProj:          metal.NewLinear(zeros2x2(), nil),
			QNormScaled:    ones2(),
			KNormScaled:    ones2(),
			HeadDim:        2,
			NKVHeads:       1,
			Scale:          1.0,
			RopeBase:       10000,
			RopeRotatedDim: 2,
		},
		MLP: &metal.MLP{
			GateProj: metal.NewLinear(zeros2x2(), nil),
			UpProj:   metal.NewLinear(zeros2x2(), nil),
			DownProj: metal.NewLinear(zeros2x2(), nil),
		},
		InputNormScaled:    ones2(),
		PostAttnNormScaled: ones2(),
		PreFFNormScaled:    ones2(),
		PostFFNormScaled:   ones2(),
		LayerScalar:        metal.FromValues([]float32{1}, 1),
		LayerIdx:           7,
		FFNMemory:          augmenter,
	}
	defer closeGemma4(&Gemma4Model{Layers: []*Gemma4DecoderLayer{layer}})

	cfg := &Gemma4TextConfig{
		TransformerConfig: metal.TransformerConfig{
			HiddenSize:        2,
			NumAttentionHeads: 1,
			NumKeyValueHeads:  1,
			RMSNormEps:        1e-6,
		},
	}
	x := metal.FromValues([]float32{0.3, -0.2}, 1, 1, 2)

	got, kv := layer.forward(x, nil, 1, 1, nil, nil, sharedKV{}, cfg, nil, nil, false)
	defer metal.Free(kv.Keys, kv.Values)

	if !augmenter.called || augmenter.layerID != 7 {
		t.Fatalf("augmenter called=%v layer=%d, want layer 7", augmenter.called, augmenter.layerID)
	}
	ffIn := metal.RMSNorm(x, layer.PreFFNormScaled, cfg.RMSNormEps)
	augmented := metal.MulScalar(ffIn, 2)
	metal.Free(ffIn)
	ffResidual := metal.RMSNorm(augmented, layer.PostFFNormScaled, cfg.RMSNormEps)
	metal.Free(augmented)
	want := metal.Add(x, ffResidual)
	metal.Free(ffResidual)

	if err := metal.Eval(got, want); err != nil {
		t.Fatalf("Eval: %v", err)
	}
	defer metal.Free(x, got, want)

	floatSliceApprox(t, got.Floats(), want.Floats())
}

func TestGemma4_DecoderLayer_MoERouterUsesAttentionResidualInput_Good(t *testing.T) {
	requireMetalRuntime(t)

	zeros2x2 := func() *metal.Array {
		return metal.FromValues([]float32{
			0, 0,
			0, 0,
		}, 2, 2)
	}
	ones2 := func() *metal.Array {
		return metal.FromValues([]float32{1, 1}, 2)
	}
	expertWeight := func(e0, e1 []float32) *metal.Array {
		data := append(append([]float32{}, e0...), e1...)
		return metal.FromValues(data, 2, 2, 2)
	}

	layer := &Gemma4DecoderLayer{
		Attention: &Gemma4Attention{
			QProj:          metal.NewLinear(zeros2x2(), nil),
			KProj:          metal.NewLinear(zeros2x2(), nil),
			VProj:          metal.NewLinear(zeros2x2(), nil),
			OProj:          metal.NewLinear(zeros2x2(), nil),
			QNormScaled:    ones2(),
			KNormScaled:    ones2(),
			HeadDim:        2,
			NKVHeads:       1,
			Scale:          1.0,
			RopeBase:       10000,
			RopeRotatedDim: 2,
		},
		MLP: &metal.MLP{
			GateProj: metal.NewLinear(zeros2x2(), nil),
			UpProj:   metal.NewLinear(zeros2x2(), nil),
			DownProj: metal.NewLinear(zeros2x2(), nil),
		},
		EnableMoE:          true,
		InputNormScaled:    ones2(),
		PostAttnNormScaled: ones2(),
		PreFFNormScaled:    ones2(),
		PostFFNormScaled:   ones2(),
		PreFFNorm2Scaled:   metal.FromValues([]float32{0.1, 2.0}, 2),
		PostFFNorm1Scaled:  ones2(),
		PostFFNorm2Scaled:  ones2(),
		Router: &Gemma4Router{
			Proj: metal.NewLinear(metal.FromValues([]float32{
				1, -1,
				-1, 1,
			}, 2, 2), nil),
			Scale:          ones2(),
			PerExpertScale: metal.FromValues([]float32{1, 1}, 2),
			ScaleScaled:    ones2(),
			TopK:           1,
			Eps:            1e-6,
		},
		Experts: &Gemma4Experts{
			GateProj: metal.NewSwitchLinear(expertWeight(
				[]float32{1, 0, 0, 1},
				[]float32{1, 0, 0, 1},
			), nil),
			UpProj: metal.NewSwitchLinear(expertWeight(
				[]float32{1, 0, 0, 1},
				[]float32{1, 0, 0, 1},
			), nil),
			DownProj: metal.NewSwitchLinear(expertWeight(
				[]float32{1, 0, 0, 1},
				[]float32{-1, 0, 0, -1},
			), nil),
		},
	}
	defer closeGemma4(&Gemma4Model{Layers: []*Gemma4DecoderLayer{layer}})

	cfg := &Gemma4TextConfig{
		TransformerConfig: metal.TransformerConfig{
			HiddenSize:        2,
			NumAttentionHeads: 1,
			NumKeyValueHeads:  1,
			RMSNormEps:        1e-6,
		},
	}
	x := metal.FromValues([]float32{2, 1}, 1, 1, 2)

	got, kv := layer.forward(x, nil, 1, 1, nil, nil, sharedKV{}, cfg, nil, nil, false)
	defer metal.Free(kv.Keys, kv.Values)

	h2InForCheck := metal.RMSNorm(x, layer.PreFFNorm2Scaled, cfg.RMSNormEps)
	residualIndices, residualWeights := layer.Router.forward(x)
	normedIndices, normedWeights := layer.Router.forward(h2InForCheck)
	if err := metal.Eval(residualIndices, normedIndices); err != nil {
		t.Fatalf("Eval indices: %v", err)
	}
	if residualIndices.DataInt32()[0] == normedIndices.DataInt32()[0] {
		t.Fatal("expected residual-stream and pre-normalized router inputs to pick different experts")
	}

	h1In := metal.RMSNorm(x, layer.PreFFNormScaled, cfg.RMSNormEps)
	h1 := layer.MLP.Forward(h1In)
	metal.Free(h1In)
	h1Normed := metal.RMSNorm(h1, layer.PostFFNorm1Scaled, cfg.RMSNormEps)
	metal.Free(h1)

	h2 := layer.Experts.forward(h2InForCheck, residualIndices, residualWeights, "")
	metal.Free(h2InForCheck, normedIndices, normedWeights, residualIndices, residualWeights)
	h2Normed := metal.RMSNorm(h2, layer.PostFFNorm2Scaled, cfg.RMSNormEps)
	metal.Free(h2)

	combined := metal.Add(h1Normed, h2Normed)
	metal.Free(h1Normed, h2Normed)
	combinedNormed := metal.RMSNorm(combined, layer.PostFFNormScaled, cfg.RMSNormEps)
	metal.Free(combined)
	want := metal.Add(x, combinedNormed)
	metal.Free(combinedNormed)

	if err := metal.Eval(got, want); err != nil {
		t.Fatalf("Eval: %v", err)
	}
	defer metal.Free(x, got, want)

	floatSliceApprox(t, got.Floats(), want.Floats())
}

func TestGemma4_AttentionPagedCacheReturnsSharedPages_Good(t *testing.T) {
	requireMetalRuntime(t)

	identity := func() *metal.Array {
		return metal.FromValues([]float32{
			1, 0,
			0, 1,
		}, 2, 2)
	}
	ones := func() *metal.Array { return metal.FromValues([]float32{1, 1}, 2) }
	attention := &Gemma4Attention{
		QProj:          metal.NewLinear(identity(), nil),
		KProj:          metal.NewLinear(identity(), nil),
		VProj:          metal.NewLinear(identity(), nil),
		OProj:          metal.NewLinear(identity(), nil),
		QNormScaled:    ones(),
		KNormScaled:    ones(),
		HeadDim:        2,
		NKVHeads:       1,
		Scale:          1,
		RopeBase:       10000,
		RopeRotatedDim: 2,
	}
	defer closeGemma4(&Gemma4Model{Layers: []*Gemma4DecoderLayer{{Attention: attention}}})

	cfg := &Gemma4TextConfig{
		TransformerConfig: metal.TransformerConfig{
			HiddenSize:        2,
			NumAttentionHeads: 1,
			NumKeyValueHeads:  1,
			RMSNormEps:        1e-6,
		},
	}
	cache := metal.NewPagedKVCache(8, 2)
	defer cache.Reset()
	x := metal.FromValues([]float32{0.25, -0.5}, 1, 1, 2)

	out, kv := attention.forward(x, cache, 1, 1, nil, sharedKV{}, cfg, 0, nil, nil, false)
	defer func() {
		metal.Free(x, out)
		kv.Free()
	}()
	if err := metal.Eval(out); err != nil {
		t.Fatalf("Eval(out): %v", err)
	}

	if kv.Keys != nil || kv.Values != nil {
		t.Fatalf("shared KV used concatenated arrays: %v/%v", kv.Keys != nil, kv.Values != nil)
	}
	if len(kv.Pages.Keys) != 1 || len(kv.Pages.Values) != 1 {
		t.Fatalf("shared pages = %d/%d, want one K/V page", len(kv.Pages.Keys), len(kv.Pages.Values))
	}
}

func TestGemma4_AttentionFixedCacheUsesNativeBridge_Good(t *testing.T) {
	requireMetalRuntime(t)

	identity := func() *metal.Array {
		return metal.FromValues([]float32{
			1, 0,
			0, 1,
		}, 2, 2)
	}
	ones := func() *metal.Array { return metal.FromValues([]float32{1, 1}, 2) }
	attention := &Gemma4Attention{
		QProj:          metal.NewLinear(identity(), nil),
		KProj:          metal.NewLinear(identity(), nil),
		VProj:          metal.NewLinear(identity(), nil),
		OProj:          metal.NewLinear(identity(), nil),
		QNormScaled:    ones(),
		KNormScaled:    ones(),
		HeadDim:        2,
		NKVHeads:       1,
		Scale:          1,
		RopeBase:       10000,
		RopeRotatedDim: 2,
	}
	defer closeGemma4(&Gemma4Model{Layers: []*Gemma4DecoderLayer{{Attention: attention}}})

	cfg := &Gemma4TextConfig{
		TransformerConfig: metal.TransformerConfig{
			HiddenSize:        2,
			NumAttentionHeads: 1,
			NumKeyValueHeads:  1,
			RMSNormEps:        1e-6,
		},
	}
	fixed := metal.NewFixedKVCache(4)
	paged := metal.NewPagedKVCache(4, 2)
	defer fixed.Reset()
	defer paged.Reset()

	fixedX := metal.FromValues([]float32{0.25, -0.5}, 1, 1, 2)
	pagedX := fixedX.Clone()
	defer metal.Free(fixedX, pagedX)

	fixedOut, fixedKV := attention.forward(fixedX, fixed, 1, 1, nil, sharedKV{}, cfg, 0, nil, nil, false)
	pagedOut, pagedKV := attention.forward(pagedX, paged, 1, 1, nil, sharedKV{}, cfg, 0, nil, nil, false)
	defer metal.Free(fixedOut, pagedOut)
	defer fixedKV.Free()
	defer pagedKV.Free()
	if !fixedKV.Fixed {
		t.Fatal("fixed-cache attention did not return fixed shared KV from native bridge")
	}
	if state := fixed.State(); len(state) != 2 || state[0].Dim(2) != 4 || state[1].Dim(2) != 4 {
		t.Fatalf("fixed cache state shape = %v, want full-capacity state", state)
	}
	if err := metal.Eval(fixedOut, pagedOut); err != nil {
		t.Fatalf("Eval(fixed/paged attention) error = %v", err)
	}
	floatSliceApprox(t, fixedOut.Floats(), pagedOut.Floats())
}

func TestGemma4_AttentionSharedPagedKVSkipsKVProjection_Good(t *testing.T) {
	requireMetalRuntime(t)

	identity := func() *metal.Array {
		return metal.FromValues([]float32{
			1, 0,
			0, 1,
		}, 2, 2)
	}
	attention := &Gemma4Attention{
		QProj:          metal.NewLinear(identity(), nil),
		OProj:          metal.NewLinear(identity(), nil),
		QNormScaled:    metal.FromValues([]float32{1, 1}, 2),
		HeadDim:        2,
		NKVHeads:       1,
		Scale:          1,
		RopeBase:       10000,
		RopeRotatedDim: 2,
	}
	defer closeGemma4(&Gemma4Model{Layers: []*Gemma4DecoderLayer{{Attention: attention}}})

	keyPage := metal.FromValues([]float32{
		1, 0,
		0, 1,
	}, 1, 1, 2, 2)
	valuePage := metal.FromValues([]float32{
		2, 0,
		0, 3,
	}, 1, 1, 2, 2)
	prev := sharedKV{
		Pages: metal.PagedKVState{
			Keys:   []*metal.Array{keyPage},
			Values: []*metal.Array{valuePage},
			Owned:  []*metal.Array{keyPage, valuePage},
			Length: 2,
		},
		Offset: 2,
	}
	cfg := &Gemma4TextConfig{
		TransformerConfig: metal.TransformerConfig{
			HiddenSize:        2,
			NumAttentionHeads: 1,
			NumKeyValueHeads:  1,
			RMSNormEps:        1e-6,
		},
	}
	x := metal.FromValues([]float32{0.5, 0.25}, 1, 1, 2)

	out, kv := attention.forward(x, nil, 1, 1, nil, prev, cfg, 0, nil, nil, false)
	defer func() {
		metal.Free(x, out)
		kv.Free()
	}()
	if err := metal.Eval(out); err != nil {
		t.Fatalf("Eval(out): %v", err)
	}
	if kv.Keys != nil || kv.Values != nil {
		t.Fatalf("shared KV materialized contiguous arrays: %v/%v", kv.Keys != nil, kv.Values != nil)
	}
}

func TestGemma4_AttentionPagedFastConcatCachesFullKVForSharedReuse_Good(t *testing.T) {
	requireMetalRuntime(t)
	t.Cleanup(metal.SetRuntimeGate("GO_MLX_ENABLE_PAGED_DECODE_FAST_CONCAT", "1"))
	t.Cleanup(metal.SetRuntimeGate("GO_MLX_ENABLE_NATIVE_PAGED_ATTENTION", "1"))

	identity := func() *metal.Array {
		return metal.FromValues([]float32{
			1, 0,
			0, 1,
		}, 2, 2)
	}
	ones := func() *metal.Array { return metal.FromValues([]float32{1, 1}, 2) }
	attention := &Gemma4Attention{
		QProj:          metal.NewLinear(identity(), nil),
		KProj:          metal.NewLinear(identity(), nil),
		VProj:          metal.NewLinear(identity(), nil),
		OProj:          metal.NewLinear(identity(), nil),
		QNormScaled:    ones(),
		KNormScaled:    ones(),
		HeadDim:        2,
		NKVHeads:       1,
		Scale:          1,
		RopeBase:       10000,
		RopeRotatedDim: 2,
	}
	defer closeGemma4(&Gemma4Model{Layers: []*Gemma4DecoderLayer{{Attention: attention}}})

	cfg := &Gemma4TextConfig{
		TransformerConfig: metal.TransformerConfig{
			HiddenSize:        2,
			NumAttentionHeads: 1,
			NumKeyValueHeads:  1,
			RMSNormEps:        1e-6,
		},
	}
	cache := metal.NewPagedKVCache(8, 1)
	defer cache.Reset()

	x1 := metal.FromValues([]float32{0.25, -0.5}, 1, 1, 2)
	out1, kv1 := attention.forward(x1, cache, 1, 1, nil, sharedKV{}, cfg, 0, nil, nil, false)
	if err := metal.Eval(out1); err != nil {
		t.Fatalf("Eval(out1): %v", err)
	}
	metal.Free(x1, out1)
	kv1.Free()

	x2 := metal.FromValues([]float32{0.5, 0.25}, 1, 1, 2)
	out2, kv2 := attention.forward(x2, cache, 1, 1, nil, sharedKV{}, cfg, 0, nil, nil, true)
	defer kv2.Free()
	if err := metal.Eval(out2); err != nil {
		t.Fatalf("Eval(out2): %v", err)
	}
	metal.Free(x2, out2)
	if !kv2.HasPages() {
		t.Fatal("owner paged attention did not keep page state")
	}
	if !gemma4ValidKV(kv2.Keys, kv2.Values) {
		t.Fatal("owner paged fast-concat did not retain contiguous K/V for shared reuse")
	}

	x3 := metal.FromValues([]float32{-0.25, 0.75}, 1, 1, 2)
	out3, kv3 := attention.forward(x3, nil, 1, 1, nil, kv2, cfg, 0, nil, nil, false)
	defer metal.Free(x3, out3)
	if err := metal.Eval(out3); err != nil {
		t.Fatalf("Eval(out3): %v", err)
	}
	if kv3.Keys != kv2.Keys || kv3.Values != kv2.Values {
		t.Fatal("shared paged attention should reuse owner contiguous K/V handles")
	}
}

func TestGemma4_AttentionPagedStorageDTypeKeepsAttentionEvaluable_Good(t *testing.T) {
	requireMetalRuntime(t)
	t.Cleanup(metal.SetRuntimeGate("GO_MLX_ENABLE_PAGED_DECODE_FAST_CONCAT", "1"))

	identity := func() *metal.Array {
		return metal.FromValues([]float32{
			1, 0,
			0, 1,
		}, 2, 2)
	}
	ones := func() *metal.Array { return metal.FromValues([]float32{1, 1}, 2) }
	attention := &Gemma4Attention{
		QProj:          metal.NewLinear(identity(), nil),
		KProj:          metal.NewLinear(identity(), nil),
		VProj:          metal.NewLinear(identity(), nil),
		OProj:          metal.NewLinear(identity(), nil),
		QNormScaled:    ones(),
		KNormScaled:    ones(),
		HeadDim:        2,
		NKVHeads:       1,
		Scale:          1,
		RopeBase:       10000,
		RopeRotatedDim: 2,
	}
	defer closeGemma4(&Gemma4Model{Layers: []*Gemma4DecoderLayer{{Attention: attention}}})

	cfg := &Gemma4TextConfig{
		TransformerConfig: metal.TransformerConfig{
			HiddenSize:        2,
			NumAttentionHeads: 1,
			NumKeyValueHeads:  1,
			RMSNormEps:        1e-6,
		},
	}
	cache := metal.NewPagedKVCacheWithDType(8, 1, metal.DTypeBFloat16)
	defer cache.Reset()

	x1 := metal.FromValues([]float32{0.25, -0.5}, 1, 1, 2)
	out1, kv1 := attention.forward(x1, cache, 1, 1, nil, sharedKV{}, cfg, 0, nil, nil, false)
	if err := metal.Eval(out1); err != nil {
		t.Fatalf("Eval(out1): %v", err)
	}
	metal.Free(x1, out1)
	kv1.Free()

	x2 := metal.FromValues([]float32{0.5, 0.25}, 1, 1, 2)
	out2, kv2 := attention.forward(x2, cache, 1, 1, nil, sharedKV{}, cfg, 0, nil, nil, false)
	defer kv2.Free()
	defer metal.Free(x2, out2)
	if err := metal.Eval(out2); err != nil {
		t.Fatalf("Eval(out2): %v", err)
	}
	if !kv2.HasPages() || !gemma4ValidKV(kv2.Keys, kv2.Values) {
		t.Fatal("typed owner paged attention did not return usable page and contiguous state")
	}
	if kv2.Pages.Keys[0].Dtype() != metal.DTypeBFloat16 || kv2.Keys.Dtype() != metal.DTypeBFloat16 {
		t.Fatalf("typed K/V dtypes = page %v contiguous %v, want bfloat16", kv2.Pages.Keys[0].Dtype(), kv2.Keys.Dtype())
	}
}

func TestGemma4_AttentionPagedDoesNotRetainFullMaterializedKV_Good(t *testing.T) {
	requireMetalRuntime(t)
	t.Cleanup(metal.SetRuntimeGate("GO_MLX_ENABLE_NATIVE_PAGED_ATTENTION", "1"))

	identity := func() *metal.Array {
		return metal.FromValues([]float32{
			1, 0,
			0, 1,
		}, 2, 2)
	}
	ones := func() *metal.Array { return metal.FromValues([]float32{1, 1}, 2) }
	attention := &Gemma4Attention{
		QProj:          metal.NewLinear(identity(), nil),
		KProj:          metal.NewLinear(identity(), nil),
		VProj:          metal.NewLinear(identity(), nil),
		OProj:          metal.NewLinear(identity(), nil),
		QNormScaled:    ones(),
		KNormScaled:    ones(),
		HeadDim:        2,
		NKVHeads:       1,
		Scale:          1,
		RopeBase:       10000,
		RopeRotatedDim: 2,
	}
	defer closeGemma4(&Gemma4Model{Layers: []*Gemma4DecoderLayer{{Attention: attention}}})

	cfg := &Gemma4TextConfig{
		TransformerConfig: metal.TransformerConfig{
			HiddenSize:        2,
			NumAttentionHeads: 1,
			NumKeyValueHeads:  1,
			RMSNormEps:        1e-6,
		},
	}
	cache := metal.NewPagedKVCache(8, 1)
	defer cache.Reset()

	x1 := metal.FromValues([]float32{0.25, -0.5}, 1, 1, 2)
	out1, kv1 := attention.forward(x1, cache, 1, 1, nil, sharedKV{}, cfg, 0, nil, nil, false)
	if err := metal.Eval(out1); err != nil {
		t.Fatalf("Eval(out1): %v", err)
	}
	metal.Free(x1, out1)
	kv1.Free()

	x2 := metal.FromValues([]float32{0.5, 0.25}, 1, 1, 2)
	out2, kv2 := attention.forward(x2, cache, 1, 1, nil, sharedKV{}, cfg, 0, nil, nil, false)
	defer kv2.Free()
	if err := metal.Eval(out2); err != nil {
		t.Fatalf("Eval(out2): %v", err)
	}
	metal.Free(x2, out2)
	if !kv2.HasPages() {
		t.Fatal("owner paged attention did not keep page state")
	}
	if gemma4ValidKV(kv2.Keys, kv2.Values) {
		t.Fatal("owner paged attention returned retained full-materialized K/V views")
	}
	state := cache.BorrowedPageState()
	defer state.Free()
	if state.Length != 2 || len(state.Keys) != 2 || len(state.Values) != 2 {
		t.Fatalf("paged state = len %d K pages %d V pages %d, want 2/2/2 without materialized backing", state.Length, len(state.Keys), len(state.Values))
	}
}

func TestGemma4_AttentionForward_FallsBackWhenCacheUpdateReturnsNil_Ugly(t *testing.T) {
	requireMetalRuntime(t)

	identity := func() *metal.Array {
		return metal.FromValues([]float32{
			1, 0,
			0, 1,
		}, 2, 2)
	}
	attention := &Gemma4Attention{
		QProj:          metal.NewLinear(identity(), nil),
		KProj:          metal.NewLinear(identity(), nil),
		OProj:          metal.NewLinear(identity(), nil),
		QNormScaled:    metal.FromValues([]float32{1, 1}, 2),
		KNormScaled:    metal.FromValues([]float32{1, 1}, 2),
		HeadDim:        2,
		NKVHeads:       1,
		UseKEqV:        true,
		Scale:          1,
		RopeBase:       10000,
		RopeRotatedDim: 2,
	}
	defer closeGemma4(&Gemma4Model{Layers: []*Gemma4DecoderLayer{{Attention: attention}}})

	cfg := &Gemma4TextConfig{
		TransformerConfig: metal.TransformerConfig{
			HiddenSize:        2,
			NumAttentionHeads: 1,
			NumKeyValueHeads:  1,
			RMSNormEps:        1e-6,
		},
	}
	x := metal.FromValues([]float32{0.5, 0.25}, 1, 1, 2)
	out, kv := attention.forward(x, &fakeDetachCache{}, 1, 1, nil, sharedKV{}, cfg, 0, nil, nil, false)
	defer func() {
		metal.Free(x, out)
		kv.Free()
	}()

	if !gemma4ValidKV(kv.Keys, kv.Values) {
		t.Fatal("local K/V fallback was not retained after cache update returned nil")
	}
	if err := metal.Eval(out); err != nil {
		t.Fatalf("Eval(out): %v", err)
	}
}

func TestGemma4_AttentionKEqVDoesNotAliasFinalCache_Good(t *testing.T) {
	requireMetalRuntime(t)

	identity := func() *metal.Array {
		return metal.FromValues([]float32{
			1, 0,
			0, 1,
		}, 2, 2)
	}
	attention := &Gemma4Attention{
		QProj:          metal.NewLinear(identity(), nil),
		KProj:          metal.NewLinear(identity(), nil),
		OProj:          metal.NewLinear(identity(), nil),
		QNormScaled:    metal.FromValues([]float32{1, 1}, 2),
		KNormScaled:    metal.FromValues([]float32{1, 1}, 2),
		HeadDim:        2,
		NKVHeads:       1,
		UseKEqV:        true,
		Scale:          1,
		RopeBase:       10000,
		RopeRotatedDim: 2,
	}
	defer closeGemma4(&Gemma4Model{Layers: []*Gemma4DecoderLayer{{Attention: attention}}})

	cfg := &Gemma4TextConfig{
		TransformerConfig: metal.TransformerConfig{
			HiddenSize:        2,
			NumAttentionHeads: 1,
			NumKeyValueHeads:  1,
			RMSNormEps:        1e-6,
		},
	}
	x := metal.FromValues([]float32{
		1, 0,
		0, 1,
	}, 1, 2, 2)
	out, kv := attention.forward(x, &fakeDetachCache{}, 1, 2, nil, sharedKV{}, cfg, 0, nil, nil, false)
	defer func() {
		metal.Free(x, out)
		kv.Free()
	}()

	if !gemma4ValidKV(kv.Keys, kv.Values) {
		t.Fatal("K=V path did not retain final K/V tensors")
	}
	if err := metal.Eval(kv.Keys, kv.Values); err != nil {
		t.Fatalf("Eval(K/V): %v", err)
	}
	keys := kv.Keys.Floats()
	values := kv.Values.Floats()
	if len(keys) != len(values) {
		t.Fatalf("K/V lengths = %d/%d, want same shape", len(keys), len(values))
	}
	if reflect.DeepEqual(keys, values) {
		t.Fatal("K=V final cache tensors unexpectedly alias; KNorm/RoPE and value RMSNorm should diverge")
	}
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
		"max_position_embeddings": 16,
		"rms_norm_eps": 1e-6,
		"sliding_window": 4,
		"sliding_window_pattern": 2,
		"num_kv_shared_layers": 0,
		"use_double_wide_mlp": false,
		"layer_types": ["sliding_attention", "full_attention"]
	}`
	if err := coreio.Local.Write(core.JoinPath(dir, "config.json"), config); err != nil {
		t.Fatalf("write config.json: %v", err)
	}
	writeMinimalTokenizer(t, dir)
	if err := metal.SaveSafetensors(core.JoinPath(dir, "model.safetensors"), gemma4TinyWeightsWithPerLayerInputs()); err != nil {
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

	tokens := metal.FromValues([]int32{2, 3, 4}, 1, 3)
	caches := model.NewCache()
	logits := model.Forward(tokens, caches)
	if err := metal.Eval(logits); err != nil {
		t.Fatalf("Eval logits: %v", err)
	}
	defer func() {
		metal.Free(tokens, logits)
		metal.FreeCaches(caches)
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
		"max_position_embeddings": 16,
		"rms_norm_eps": 1e-6,
		"sliding_window": 4,
		"sliding_window_pattern": 2,
		"num_kv_shared_layers": 0,
		"use_double_wide_mlp": false,
		"layer_types": ["sliding_attention", "full_attention"]
	}`
	if err := coreio.Local.Write(core.JoinPath(dir, "config.json"), config); err != nil {
		t.Fatalf("write config.json: %v", err)
	}
	writeMinimalTokenizer(t, dir)

	weights := gemma4TinyWeightsWithPerLayerInputs()
	delete(weights, "model.per_layer_projection_norm.weight")
	if err := metal.SaveSafetensors(core.JoinPath(dir, "model.safetensors"), weights); err != nil {
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
		"max_position_embeddings": 16,
		"rms_norm_eps": 1e-6,
		"sliding_window": 4,
		"sliding_window_pattern": 2,
		"num_kv_shared_layers": 0,
		"use_double_wide_mlp": false,
		"layer_types": ["sliding_attention", "full_attention"]
	}`
	if err := coreio.Local.Write(core.JoinPath(dir, "config.json"), config); err != nil {
		t.Fatalf("write config.json: %v", err)
	}
	writeMinimalTokenizer(t, dir)

	weights := gemma4TinyWeightsWithPerLayerInputs()
	delete(weights, "model.per_layer_projection_norm.weight")
	if err := metal.SaveSafetensors(core.JoinPath(dir, "model.safetensors"), weights); err != nil {
		t.Fatalf("SaveSafetensors: %v", err)
	}
	freeWeightMap(weights)

	metal.ClearCache()
	baseline := metal.GetActiveMemory()

	model, err := LoadGemma4(dir)
	if err != nil {
		t.Fatalf("LoadGemma4: %v", err)
	}

	closeGemma4(model)
	metal.ClearCache()

	if active := metal.GetActiveMemory(); active > baseline {
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
		"max_position_embeddings": 16,
		"rms_norm_eps": 1e-6,
		"sliding_window": 4,
		"sliding_window_pattern": 1,
		"num_kv_shared_layers": 0,
		"hidden_size_per_layer_input": 0,
		"use_double_wide_mlp": false,
		"layer_types": ["full_attention"]
	}`
	if err := coreio.Local.Write(core.JoinPath(dir, "config.json"), config); err != nil {
		t.Fatalf("write config.json: %v", err)
	}
	writeMinimalTokenizer(t, dir)

	weights := map[string]*metal.Array{
		"model.embed_tokens.weight":                        seqArray(0.01, 10, 8),
		"model.norm.weight":                                seqArray(0.02, 8),
		"model.layers.0.input_layernorm.weight":            seqArray(0.03, 8),
		"model.layers.0.post_attention_layernorm.weight":   seqArray(0.04, 8),
		"model.layers.0.pre_feedforward_layernorm.weight":  seqArray(0.05, 8),
		"model.layers.0.post_feedforward_layernorm.weight": seqArray(0.06, 8),
		"model.layers.0.layer_scalar":                      metal.FromValues([]float32{1}, 1),
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
	if err := metal.SaveSafetensors(core.JoinPath(dir, "model.safetensors"), weights); err != nil {
		t.Fatalf("SaveSafetensors: %v", err)
	}
	freeWeightMap(weights)

	metal.ClearCache()
	baseline := metal.GetActiveMemory()

	model, err := LoadGemma4(dir)
	if err != nil {
		t.Fatalf("LoadGemma4: %v", err)
	}

	if got := model.Layers[0].Attention.VProj; got != nil {
		t.Fatal("expected K-equals-V full-attention layer to drop v_proj")
	}

	closeGemma4(model)
	metal.ClearCache()

	if active := metal.GetActiveMemory(); active > baseline {
		t.Fatalf("active memory after close = %d, want <= %d", active, baseline)
	}
}

func gemma4TinyWeights() map[string]*metal.Array {
	weights := map[string]*metal.Array{
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
		weights[prefix+".layer_scalar"] = metal.FromValues([]float32{1}, 1)

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

func gemma4TinyWeightsWithPerLayerInputs() map[string]*metal.Array {
	weights := gemma4TinyWeights()
	weights["model.embed_tokens_per_layer.weight"] = seqArray(1.10, 10, 4)
	weights["model.per_layer_model_projection.weight"] = seqArray(1.20, 4, 8)
	weights["model.per_layer_projection_norm.weight"] = seqArray(1.30, 2)

	for idx := range 2 {
		prefix := core.Sprintf("model.layers.%d", idx)
		weights[prefix+".per_layer_input_gate.weight"] = seqArray(1.40+float32(idx), 2, 8)
		weights[prefix+".per_layer_projection.weight"] = seqArray(1.50+float32(idx), 8, 2)
		weights[prefix+".post_per_layer_input_norm.weight"] = seqArray(1.60+float32(idx), 8)
	}

	return weights
}

func seqArray(start float32, shape ...int) *metal.Array {
	size := 1
	for _, dim := range shape {
		size *= dim
	}
	data := make([]float32, size)
	for i := range size {
		data[i] = start + 0.01*float32(i)
	}
	return metal.FromValues(data, shape...)
}

func TestGemma4_parseConfig_EmbeddingScalesCached_Good(t *testing.T) {
	type pair struct{ hidden, perLayer int32 }
	cases := []pair{
		{hidden: 2, perLayer: 2},
		{hidden: 1024, perLayer: 256},
		{hidden: 2048, perLayer: 256},
		{hidden: 3072, perLayer: 384},
		{hidden: 4096, perLayer: 0}, // disabled per-layer path
	}
	for _, c := range cases {
		cfg := &Gemma4TextConfig{
			TransformerConfig: metal.TransformerConfig{
				HiddenSize: c.hidden,
			},
			HiddenSizePerLayerInput: c.perLayer,
		}
		gemma4FinaliseEmbeddingScales(cfg)

		wantH := float32(math.Sqrt(float64(c.hidden)))
		if cfg.EmbeddingScale != wantH {
			t.Fatalf("EmbeddingScale(hidden=%d): cached %v != per-call %v", c.hidden, cfg.EmbeddingScale, wantH)
		}
		var wantP float32
		if c.perLayer > 0 {
			wantP = float32(math.Sqrt(float64(c.perLayer)))
		}
		if cfg.PerLayerInputEmbeddingScale != wantP {
			t.Fatalf("PerLayerInputEmbeddingScale(perLayer=%d): cached %v != per-call %v", c.perLayer, cfg.PerLayerInputEmbeddingScale, wantP)
		}
		wantProj := float32(math.Pow(float64(c.hidden), -0.5))
		if cfg.PerLayerProjectionScale != wantProj {
			t.Fatalf("PerLayerProjectionScale(hidden=%d): cached %v != per-call %v", c.hidden, cfg.PerLayerProjectionScale, wantProj)
		}
	}
}

func TestGemma4_perLayerCombineScale_MatchesMathPow_Good(t *testing.T) {
	// gemma4PerLayerCombineScale folds the per-token math.Pow(2, -0.5)
	// inside perLayerInputTensor; the constant must remain bit-exact
	// against the float32 narrowing of the live computation so the
	// forward pass output is unchanged.
	want := float32(math.Pow(2, -0.5))
	if gemma4PerLayerCombineScale != want {
		t.Fatalf("gemma4PerLayerCombineScale = %v, want %v (1/sqrt(2))", gemma4PerLayerCombineScale, want)
	}
}

func TestGemma4_parseConfig_EmbeddingScalesCached_ResetsOnZero_Good(t *testing.T) {
	// LoadGemma4 may clear HiddenSizePerLayerInput when weights are missing;
	// the second invocation of gemma4FinaliseEmbeddingScales must zero the
	// cached scale rather than retain a stale value.
	cfg := &Gemma4TextConfig{
		TransformerConfig: metal.TransformerConfig{
			HiddenSize: 2048,
		},
		HiddenSizePerLayerInput: 256,
	}
	gemma4FinaliseEmbeddingScales(cfg)
	if cfg.PerLayerInputEmbeddingScale == 0 {
		t.Fatal("PerLayerInputEmbeddingScale = 0, want positive after first finalise")
	}
	cfg.HiddenSizePerLayerInput = 0
	gemma4FinaliseEmbeddingScales(cfg)
	if cfg.PerLayerInputEmbeddingScale != 0 {
		t.Fatalf("PerLayerInputEmbeddingScale = %v, want 0 after per-layer reset", cfg.PerLayerInputEmbeddingScale)
	}
	if cfg.EmbeddingScale == 0 {
		t.Fatal("EmbeddingScale = 0, want unchanged main embedding scale")
	}
	if cfg.PerLayerProjectionScale == 0 {
		t.Fatal("PerLayerProjectionScale = 0, want unchanged when only per-layer reset")
	}
	// A second zeroing of HiddenSize must also zero PerLayerProjectionScale
	// — the loader may clear HiddenSize in pathological configs and the
	// projection scale tracks HiddenSize.
	cfg.HiddenSize = 0
	gemma4FinaliseEmbeddingScales(cfg)
	if cfg.PerLayerProjectionScale != 0 {
		t.Fatalf("PerLayerProjectionScale = %v, want 0 after HiddenSize reset", cfg.PerLayerProjectionScale)
	}
}

// writeMinimalTokenizer drops a tiny BPE tokenizer.json into dir for the
// load/attach tests. Self-contained (no metal internals); previously lived in
// metal's model_test.go beside the gemma4 architecture tests, relocated here
// with those tests.
func writeMinimalTokenizer(t testing.TB, dir string) {
	t.Helper()
	tokenizer := `{
		"model": {
			"type": "BPE",
			"vocab": {"<pad>": 0, "<eos>": 1, "<bos>": 2, "hello": 3, "world": 4},
			"merges": []
		},
		"added_tokens": [
			{"id": 0, "content": "<pad>", "special": true},
			{"id": 1, "content": "<eos>", "special": true},
			{"id": 2, "content": "<bos>", "special": true}
		]
	}`
	if err := coreio.Local.Write(core.JoinPath(dir, "tokenizer.json"), tokenizer); err != nil {
		t.Fatalf("write tokenizer.json: %v", err)
	}
}

// makeSingleTokenKVShape returns [B, H, 1, D] random K/V tensors for the
// single-token attention tests. Self-contained over the public metal API;
// relocated here with the architecture tests.
func makeSingleTokenKVShape(B, H, D int32) (*metal.Array, *metal.Array) {
	k := metal.RandomUniform(0, 1, []int32{B, H, 1, D}, metal.DTypeFloat32)
	v := metal.RandomUniform(0, 1, []int32{B, H, 1, D}, metal.DTypeFloat32)
	metal.Materialize(k, v)
	return k, v
}

// fakeDetachCache is a no-op Cache that counts Detach calls, used by the
// architecture attention tests. It implements the public metal.Cache interface;
// relocated here with the architecture tests.
type fakeDetachCache struct {
	detachCalls int
}

func (f *fakeDetachCache) Update(_ *metal.Array, _ *metal.Array, _ int) (*metal.Array, *metal.Array) {
	return nil, nil
}
func (f *fakeDetachCache) Offset() int           { return 0 }
func (f *fakeDetachCache) Len() int              { return 0 }
func (f *fakeDetachCache) State() []*metal.Array { return nil }
func (f *fakeDetachCache) Reset()                {}
func (f *fakeDetachCache) Detach()               { f.detachCalls++ }

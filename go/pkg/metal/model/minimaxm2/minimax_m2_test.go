// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package minimaxm2

import (
	"encoding/binary"
	"math"
	"testing"

	"dappco.re/go"

	coreio "dappco.re/go/io"

	"dappco.re/go/mlx/pkg/metal"
)

func TestMiniMaxM2Native_ReadPayloadsAndForwardSelectedExpert_Good(t *testing.T) {
	requireMetalRuntime(t)

	dir := t.TempDir()
	config := `{
		"model_type": "minimax_m2",
		"hidden_size": 2,
		"intermediate_size": 2,
		"num_hidden_layers": 1,
		"num_attention_heads": 1,
		"num_key_value_heads": 1,
		"head_dim": 2,
		"vocab_size": 32,
		"num_local_experts": 1,
		"num_experts_per_tok": 1
	}`
	if err := coreio.Local.Write(core.JoinPath(dir, "config.json"), config); err != nil {
		t.Fatalf("write config.json: %v", err)
	}
	writeMiniMaxM2TinyJANGConfig(t, dir)
	writeMiniMaxM2TinyPayloadSafetensors(t, core.JoinPath(dir, "model.safetensors"))

	plan, err := prepareMiniMaxM2NativeLoad(dir, []byte(config))
	if err != nil {
		t.Fatalf("prepareMiniMaxM2NativeLoad() error = %v", err)
	}
	payloads, err := plan.ReadExpertPayloads(0, []int{0})
	if err != nil {
		t.Fatalf("ReadExpertPayloads() error = %v", err)
	}

	payload := payloads[0]
	if payload.PackedBytes != 3 || len(payload.GateProj.Packed) != 1 || len(payload.GateProj.Scales) != 1 {
		t.Fatalf("payload = %+v, want three one-byte projections with sidecars", payload)
	}
	got, err := forwardMiniMaxM2NativeExpertPayload([]float32{1, 2}, payload)
	if err != nil {
		t.Fatalf("forwardMiniMaxM2NativeExpertPayload() error = %v", err)
	}

	want := []float32{float32(silu64(1) * 1), float32(silu64(2) * 2)}
	floatSliceApprox(t, got, want)
}

func TestMiniMaxM2Native_ForwardSparseLayerRoutesLoadsSelectedExperts_Good(t *testing.T) {
	requireMetalRuntime(t)

	dir := t.TempDir()
	config := `{
		"model_type": "minimax_m2",
		"hidden_size": 2,
		"intermediate_size": 2,
		"num_hidden_layers": 1,
		"num_attention_heads": 1,
		"num_key_value_heads": 1,
		"head_dim": 2,
		"vocab_size": 32,
		"num_local_experts": 3,
		"num_experts_per_tok": 1
	}`
	if err := coreio.Local.Write(core.JoinPath(dir, "config.json"), config); err != nil {
		t.Fatalf("write config.json: %v", err)
	}
	writeMiniMaxM2TinyJANGConfig(t, dir)
	writeMiniMaxM2TinyRoutedPayloadSafetensors(t, core.JoinPath(dir, "model.safetensors"))

	plan, err := prepareMiniMaxM2NativeLoad(dir, []byte(config))
	if err != nil {
		t.Fatalf("prepareMiniMaxM2NativeLoad() error = %v", err)
	}
	got, err := plan.ForwardSparseLayer(0, [][]float32{{1, 0}})
	if err != nil {
		t.Fatalf("ForwardSparseLayer() error = %v", err)
	}

	if len(got.Decisions) != 1 || len(got.Decisions[0].ExpertIDs) != 1 || got.Decisions[0].ExpertIDs[0] != 2 {
		t.Fatalf("decision = %+v, want expert 2", got.Decisions)
	}
	if len(got.SelectedExpertIDs) != 1 || got.SelectedExpertIDs[0] != 2 {
		t.Fatalf("selected experts = %+v, want [2]", got.SelectedExpertIDs)
	}
	if got.LoadedPackedBytes != 3 {
		t.Fatalf("LoadedPackedBytes = %d, want one three-projection expert", got.LoadedPackedBytes)
	}
	if len(got.Output) != 1 {
		t.Fatalf("output tokens = %d, want 1", len(got.Output))
	}
	floatSliceApprox(t, got.Output[0], []float32{float32(silu64(1)), 0})
}

func TestMiniMaxM2_LoadMiniMaxM2StagedModel_Good(t *testing.T) {
	dir := t.TempDir()
	config := `{
		"model_type": "minimax_m2",
		"architectures": ["MiniMaxM2ForCausalLM"],
		"hidden_size": 3072,
		"intermediate_size": 1536,
		"num_hidden_layers": 62,
		"num_attention_heads": 48,
		"num_key_value_heads": 8,
		"head_dim": 128,
		"vocab_size": 200064,
		"max_position_embeddings": 1048576,
		"num_local_experts": 256,
		"num_experts_per_tok": 8,
		"use_routing_bias": true
	}`
	if err := coreio.Local.Write(core.JoinPath(dir, "config.json"), config); err != nil {
		t.Fatalf("write config.json: %v", err)
	}
	writeMinimalTokenizer(t, dir)
	writeMiniMaxM2JANGConfig(t, dir)
	writeMiniMaxM2SafetensorsHeader(t, core.JoinPath(dir, "model.safetensors"), miniMaxM2FirstLayerTensorNames(false))

	model, err := loadMiniMaxM2StagedModel(dir, []byte(config))
	if err != nil {
		t.Fatalf("loadMiniMaxM2StagedModel() error = %v", err)
	}
	if model.ModelType() != "minimax_m2" {
		t.Fatalf("ModelType() = %q, want minimax_m2", model.ModelType())
	}
	if model.NumLayers() != 62 {
		t.Fatalf("NumLayers() = %d, want 62", model.NumLayers())
	}
	if caches := model.NewCache(); caches != nil {
		t.Fatalf("NewCache() = %#v, want nil until MiniMax decode kernels are linked", caches)
	}
	if model.Tokenizer() == nil {
		t.Fatal("Tokenizer() = nil, want staged loader to expose tokenizer metadata")
	}
	info := &metal.ModelInfo{}
	model.FillModelInfo(info)
	if info.VocabSize != 200064 || info.HiddenSize != 3072 || info.ContextLength != 1048576 {
		t.Fatalf("Info() = %+v, want MiniMax config metadata", info)
	}
	if info.QuantBits != 2 || info.QuantGroup != 64 {
		t.Fatalf("Info() quant = %d/%d, want 2/64", info.QuantBits, info.QuantGroup)
	}
	if len(model.plan.LayerSkeleton.Attention) != 4 || model.plan.LayerSkeleton.RouterGate.Name == "" || model.plan.LayerSkeleton.RouterBias == nil {
		t.Fatalf("LayerSkeleton = %+v, want attention plus router metadata", model.plan.LayerSkeleton)
	}
	if model.plan.LayerSkeleton.Attention[0].PackedBytes == 0 {
		t.Fatalf("LayerSkeleton attention = %+v, want packed byte metadata", model.plan.LayerSkeleton.Attention)
	}
	payloadRefs, err := model.plan.ResolveExpertPayloadRefs(0, []int{0})
	if err != nil {
		t.Fatalf("ResolveExpertPayloadRefs() error = %v", err)
	}
	expert0 := payloadRefs[0]
	if expert0.PackedBytes == 0 || expert0.GateProj.Path == "" || expert0.GateProj.DataStart <= 0 {
		t.Fatalf("expert payload refs = %+v, want packed byte refs without payload loading", expert0)
	}
	if expert0.GateProj.ByteLen != 1179648 || expert0.UpProj.ByteLen != 1179648 || expert0.DownProj.ByteLen != 1179648 {
		t.Fatalf("expert payload byte lengths = gate:%d up:%d down:%d, want JANGTQ packed expert refs", expert0.GateProj.ByteLen, expert0.UpProj.ByteLen, expert0.DownProj.ByteLen)
	}
}

func TestMiniMaxM2_LoadMiniMaxM2MissingTokenizer_Bad(t *testing.T) {
	dir := t.TempDir()
	config := `{
		"model_type": "minimax_m2",
		"architectures": ["MiniMaxM2ForCausalLM"],
		"hidden_size": 3072,
		"intermediate_size": 1536,
		"num_hidden_layers": 62,
		"num_attention_heads": 48,
		"num_key_value_heads": 8,
		"head_dim": 128,
		"vocab_size": 200064,
		"num_local_experts": 256,
		"num_experts_per_tok": 8,
		"use_routing_bias": true
	}`
	if err := coreio.Local.Write(core.JoinPath(dir, "config.json"), config); err != nil {
		t.Fatalf("write config.json: %v", err)
	}
	writeMiniMaxM2JANGConfig(t, dir)
	writeMiniMaxM2SafetensorsHeader(t, core.JoinPath(dir, "model.safetensors"), miniMaxM2FirstLayerTensorNames(false))

	_, err := loadMiniMaxM2StagedModel(dir, []byte(config))
	if err == nil {
		t.Fatal("expected MiniMax staged loader tokenizer error")
	}
	if !core.Contains(err.Error(), "minimax_m2") || !core.Contains(err.Error(), "tokenizer") {
		t.Fatalf("error = %v, want minimax_m2 tokenizer diagnostic", err)
	}
}

func TestMiniMaxM2_LoadMiniMaxM2MissingTensor_Bad(t *testing.T) {
	dir := t.TempDir()
	config := `{
		"model_type": "minimax_m2",
		"architectures": ["MiniMaxM2ForCausalLM"],
		"hidden_size": 3072,
		"intermediate_size": 1536,
		"num_hidden_layers": 62,
		"num_attention_heads": 48,
		"num_key_value_heads": 8,
		"head_dim": 128,
		"vocab_size": 200064,
		"num_local_experts": 256,
		"num_experts_per_tok": 8,
		"use_routing_bias": true
	}`
	if err := coreio.Local.Write(core.JoinPath(dir, "config.json"), config); err != nil {
		t.Fatalf("write config.json: %v", err)
	}
	writeMiniMaxM2JANGConfig(t, dir)
	writeMiniMaxM2SafetensorsHeader(t, core.JoinPath(dir, "model.safetensors"), miniMaxM2FirstLayerTensorNames(true))

	_, err := loadMiniMaxM2StagedModel(dir, []byte(config))
	if err == nil {
		t.Fatal("expected MiniMax tensor validation error")
	}
	if !core.Contains(err.Error(), "minimax_m2") || !core.Contains(err.Error(), "up_proj") {
		t.Fatalf("error = %v, want missing expert up_proj diagnostic", err)
	}
}

// minimaxM2FullConfigJSON is the production-shape config used by the loader
// tests that drive the staged model end to end. The huge dims match the real
// MiniMax-M2 release so packed-byte arithmetic exercises the int64 paths.
const minimaxM2FullConfigJSON = `{
	"model_type": "minimax_m2",
	"architectures": ["MiniMaxM2ForCausalLM"],
	"hidden_size": 3072,
	"intermediate_size": 1536,
	"num_hidden_layers": 62,
	"num_attention_heads": 48,
	"num_key_value_heads": 8,
	"head_dim": 128,
	"vocab_size": 200064,
	"max_position_embeddings": 1048576,
	"num_local_experts": 256,
	"num_experts_per_tok": 8,
	"use_routing_bias": true
}`

// writeMiniMaxM2FullModelDir lays down a complete (header-only) MiniMax-M2
// checkpoint directory — config.json + tokenizer.json + jang_config.json +
// model.safetensors — that the staged loader accepts. omitTokenizer and
// omitExpertUp toggle the two negative paths.
func writeMiniMaxM2FullModelDir(t *testing.T, omitTokenizer, omitExpertUp bool) string {
	t.Helper()
	dir := t.TempDir()
	if err := coreio.Local.Write(core.JoinPath(dir, "config.json"), minimaxM2FullConfigJSON); err != nil {
		t.Fatalf("write config.json: %v", err)
	}
	if !omitTokenizer {
		writeMinimalTokenizer(t, dir)
	}
	writeMiniMaxM2JANGConfig(t, dir)
	writeMiniMaxM2SafetensorsHeader(t, core.JoinPath(dir, "model.safetensors"), miniMaxM2FirstLayerTensorNames(omitExpertUp))
	return dir
}

// TestMiniMaxM2_RegisteredLoader_Good drives the init()-registered loader
// closure through the public metal.LoadAndInit dispatch (model_type lookup →
// closure → loadMiniMaxM2StagedModel → success return), covering the success
// arm of the registered closure rather than only the import-time registration.
func TestMiniMaxM2_RegisteredLoader_Good(t *testing.T) {
	requireMetalRuntime(t)
	dir := writeMiniMaxM2FullModelDir(t, false, false)

	model, err := metal.LoadAndInit(dir)
	if err != nil {
		t.Fatalf("metal.LoadAndInit() error = %v", err)
	}
	defer model.Close()
	if model.ModelType() != "minimax_m2" {
		t.Fatalf("ModelType() = %q, want minimax_m2", model.ModelType())
	}
	if model.NumLayers() != 62 {
		t.Fatalf("NumLayers() = %d, want 62", model.NumLayers())
	}
}

// TestMiniMaxM2_RegisteredLoader_Bad drives the same registered closure with a
// checkpoint missing tokenizer.json so the closure's core.E("model.loadModel",
// ...) error-wrap arm runs and surfaces through LoadAndInit.
func TestMiniMaxM2_RegisteredLoader_Bad(t *testing.T) {
	requireMetalRuntime(t)
	dir := writeMiniMaxM2FullModelDir(t, true, false)

	_, err := metal.LoadAndInit(dir)
	if err == nil {
		t.Fatal("expected metal.LoadAndInit error for missing tokenizer")
	}
	if !core.Contains(err.Error(), "minimax_m2") {
		t.Fatalf("error = %v, want minimax_m2 diagnostic from registered loader", err)
	}
}

// TestMiniMaxM2_ValidateMiniMaxM2NativeLoad_Good checks the cheap header-only
// validation entry returns the JANGTQ plan summary for a sound checkpoint.
func TestMiniMaxM2_ValidateMiniMaxM2NativeLoad_Good(t *testing.T) {
	dir := writeMiniMaxM2FullModelDir(t, true, false) // tokenizer not needed for validate
	summary, err := validateMiniMaxM2NativeLoad(dir, []byte(minimaxM2FullConfigJSON))
	if err != nil {
		t.Fatalf("validateMiniMaxM2NativeLoad() error = %v", err)
	}
	if !core.Contains(summary, "minimax_m2") || !core.Contains(summary, "JANGTQ") {
		t.Fatalf("summary = %q, want minimax_m2 JANGTQ plan summary", summary)
	}
}

// TestMiniMaxM2_ValidateMiniMaxM2NativeLoad_Bad checks the validation entry
// propagates the config-parse/validate failure for a non-MiniMax config.
func TestMiniMaxM2_ValidateMiniMaxM2NativeLoad_Bad(t *testing.T) {
	dir := t.TempDir()
	_, err := validateMiniMaxM2NativeLoad(dir, []byte(`{"model_type":"llama","hidden_size":8}`))
	if err == nil {
		t.Fatal("expected validateMiniMaxM2NativeLoad error for non-MiniMax config")
	}
	if !core.Contains(err.Error(), "minimax_m2") {
		t.Fatalf("error = %v, want minimax_m2 validation diagnostic", err)
	}
}

// TestMiniMaxM2_DecodeUnavailableError_Good asserts the staged model's decode
// sentinel carries the operation label and the no-kernels reason.
func TestMiniMaxM2_DecodeUnavailableError_Good(t *testing.T) {
	model := &miniMaxM2StagedModel{}
	err := model.DecodeUnavailableError("generate")
	if err == nil {
		t.Fatal("DecodeUnavailableError() = nil, want staged decode error")
	}
	if !core.Contains(err.Error(), "generate") || !core.Contains(err.Error(), "no native decode kernels") {
		t.Fatalf("error = %v, want operation label plus no-kernels reason", err)
	}
}

// TestMiniMaxM2_FillModelInfo_SlidingAndBitsFallback_Good exercises the two
// fallback arms of FillModelInfo: ContextLength from sliding_window when
// max_position_embeddings is absent, and QuantBits from quantization.bits_default
// when mxtq_bits.routed_expert is absent.
func TestMiniMaxM2_FillModelInfo_SlidingAndBitsFallback_Good(t *testing.T) {
	model := &miniMaxM2StagedModel{
		plan: miniMaxM2NativeLoadPlan{
			Config: miniMaxM2LoadConfig{
				VocabSize:             100,
				HiddenSize:            16,
				MaxPositionEmbeddings: 0,
				SlidingWindow:         4096,
			},
			JANG: miniMaxM2JANGLoadConfig{},
		},
	}
	model.plan.JANG.Quantization.GroupSize = 32
	model.plan.JANG.Quantization.BitsDefault = 4

	info := &metal.ModelInfo{}
	model.FillModelInfo(info)
	if info.ContextLength != 4096 {
		t.Fatalf("ContextLength = %d, want 4096 (sliding_window fallback)", info.ContextLength)
	}
	if info.QuantBits != 4 {
		t.Fatalf("QuantBits = %d, want 4 (bits_default fallback)", info.QuantBits)
	}
	if info.QuantGroup != 32 {
		t.Fatalf("QuantGroup = %d, want 32", info.QuantGroup)
	}
}

// TestMiniMaxM2_Validate_Bad walks every rejection branch of the load-config
// validator with a table of one-field-broken configs.
func TestMiniMaxM2_Validate_Bad(t *testing.T) {
	base := miniMaxM2LoadConfig{
		ModelType:          "minimax_m2",
		HiddenSize:         8,
		IntermediateSize:   8,
		NumHiddenLayers:    1,
		NumAttentionHeads:  2,
		NumKeyValueHeads:   1,
		HeadDim:            4,
		NumLocalExperts:    4,
		NumExpertsPerToken: 2,
	}
	if err := base.validate(); err != nil {
		t.Fatalf("base config should validate, got %v", err)
	}
	cases := map[string]func(c *miniMaxM2LoadConfig){
		"wrong model type":  func(c *miniMaxM2LoadConfig) { c.ModelType = "llama" },
		"zero hidden":       func(c *miniMaxM2LoadConfig) { c.HiddenSize = 0 },
		"zero intermediate": func(c *miniMaxM2LoadConfig) { c.IntermediateSize = 0 },
		"zero layers":       func(c *miniMaxM2LoadConfig) { c.NumHiddenLayers = 0 },
		"zero heads":        func(c *miniMaxM2LoadConfig) { c.NumAttentionHeads = 0 },
		"zero kv heads":     func(c *miniMaxM2LoadConfig) { c.NumKeyValueHeads = 0 },
		"zero head dim":     func(c *miniMaxM2LoadConfig) { c.HeadDim = 0 },
		"zero experts":      func(c *miniMaxM2LoadConfig) { c.NumLocalExperts = 0 },
		"zero top-k":        func(c *miniMaxM2LoadConfig) { c.NumExpertsPerToken = 0 },
		"top-k over count":  func(c *miniMaxM2LoadConfig) { c.NumExpertsPerToken = c.NumLocalExperts + 1 },
	}
	for name, mutate := range cases {
		cfg := base
		mutate(&cfg)
		if err := cfg.validate(); err == nil {
			t.Fatalf("validate(%s) = nil, want rejection", name)
		}
	}
}

// TestMiniMaxM2_ParseLoadConfig_Bad checks the config parser rejects malformed
// JSON via the core.JSONUnmarshal failure arm.
func TestMiniMaxM2_ParseLoadConfig_Bad(t *testing.T) {
	_, err := parseMiniMaxM2LoadConfig([]byte(`{not valid json`))
	if err == nil {
		t.Fatal("parseMiniMaxM2LoadConfig() = nil, want JSON parse error")
	}
}

// TestMiniMaxM2_ParseLoadConfig_ArchitectureNormalises_Good checks model_type
// is derived from the architectures list when model_type is absent.
func TestMiniMaxM2_ParseLoadConfig_ArchitectureNormalises_Good(t *testing.T) {
	cfg, err := parseMiniMaxM2LoadConfig([]byte(`{"architectures":["MiniMaxM2ForCausalLM"],"hidden_size":8}`))
	if err != nil {
		t.Fatalf("parseMiniMaxM2LoadConfig() error = %v", err)
	}
	if cfg.ModelType != "minimax_m2" {
		t.Fatalf("ModelType = %q, want minimax_m2 derived from architectures", cfg.ModelType)
	}
}

// TestMiniMaxM2_LoadRouter_WithBias_Good loads a router that carries an
// e_score_correction_bias and verifies both weight and bias are read, covering
// the UseRoutingBias arm of LoadRouter.
func TestMiniMaxM2_LoadRouter_WithBias_Good(t *testing.T) {
	dir := t.TempDir()
	config := `{
		"model_type": "minimax_m2",
		"hidden_size": 2,
		"intermediate_size": 2,
		"num_hidden_layers": 1,
		"num_attention_heads": 1,
		"num_key_value_heads": 1,
		"head_dim": 2,
		"vocab_size": 32,
		"num_local_experts": 3,
		"num_experts_per_tok": 1,
		"use_routing_bias": true
	}`
	if err := coreio.Local.Write(core.JoinPath(dir, "config.json"), config); err != nil {
		t.Fatalf("write config.json: %v", err)
	}
	writeMiniMaxM2TinyJANGConfig(t, dir)
	identity := packMiniMaxM2TinyQ2(t, []uint8{1, 0, 0, 1})
	tensors := []miniMaxM2TinyTensor{
		miniMaxM2TinyU8Tensor("model.layers.0.self_attn.q_proj.weight", []byte{0, 0, 0, 0}, 4),
		miniMaxM2TinyU8Tensor("model.layers.0.self_attn.k_proj.weight", []byte{0, 0, 0, 0}, 4),
		miniMaxM2TinyU8Tensor("model.layers.0.self_attn.v_proj.weight", []byte{0, 0, 0, 0}, 4),
		miniMaxM2TinyU8Tensor("model.layers.0.self_attn.o_proj.weight", []byte{0, 0, 0, 0}, 4),
		miniMaxM2TinyF32Tensor("model.layers.0.block_sparse_moe.gate.weight", []float32{0, 0, -2, 0, 3, 0}, 3, 2),
		miniMaxM2TinyF32Tensor("model.layers.0.block_sparse_moe.e_score_correction_bias", []float32{0.5, -0.5, 1.0}, 3),
	}
	tensors = append(tensors, miniMaxM2TinyExpertPayloadTensors(t, 0, identity)...)
	tensors = append(tensors, miniMaxM2TinyExpertPayloadTensors(t, 2, identity)...)
	writeMiniMaxM2TinySafetensors(t, core.JoinPath(dir, "model.safetensors"), tensors)

	plan, err := prepareMiniMaxM2NativeLoad(dir, []byte(config))
	if err != nil {
		t.Fatalf("prepareMiniMaxM2NativeLoad() error = %v", err)
	}
	router, err := plan.LoadRouter(0)
	if err != nil {
		t.Fatalf("LoadRouter() error = %v", err)
	}
	if router.NumExperts != 3 || router.HiddenSize != 2 {
		t.Fatalf("router meta = %d experts / %d hidden, want 3 / 2", router.NumExperts, router.HiddenSize)
	}
	if len(router.Weight) != 6 {
		t.Fatalf("router weight len = %d, want 6", len(router.Weight))
	}
	if len(router.Bias) != 3 || router.Bias[0] != 0.5 || router.Bias[2] != 1.0 {
		t.Fatalf("router bias = %v, want [0.5 -0.5 1.0]", router.Bias)
	}
}

// TestMiniMaxM2_LoadRouter_Bad covers the out-of-range layer rejection.
func TestMiniMaxM2_LoadRouter_Bad(t *testing.T) {
	plan := miniMaxM2NativeLoadPlan{Config: miniMaxM2LoadConfig{NumHiddenLayers: 1}}
	if _, err := plan.LoadRouter(5); err == nil {
		t.Fatal("LoadRouter(out-of-range) = nil, want range error")
	}
	if _, err := plan.LoadRouter(-1); err == nil {
		t.Fatal("LoadRouter(negative) = nil, want range error")
	}
}

// TestMiniMaxM2_RouterProject_Bad walks the Project validation arms: bad
// metadata, mismatched weight length, mismatched bias length, and a token whose
// hidden width disagrees with the router.
func TestMiniMaxM2_RouterProject_Bad(t *testing.T) {
	cases := map[string]miniMaxM2NativeRouterWeights{
		"zero meta":      {NumExperts: 0, HiddenSize: 0},
		"bad weight len": {NumExperts: 2, HiddenSize: 2, Weight: []float32{1, 2, 3}},
		"bad bias len":   {NumExperts: 2, HiddenSize: 2, Weight: []float32{1, 2, 3, 4}, Bias: []float32{1}},
	}
	for name, router := range cases {
		if _, err := router.Project([][]float32{{1, 1}}); err == nil {
			t.Fatalf("Project(%s) = nil, want error", name)
		}
	}
	good := miniMaxM2NativeRouterWeights{NumExperts: 2, HiddenSize: 2, Weight: []float32{1, 0, 0, 1}}
	if _, err := good.Project([][]float32{{1, 2, 3}}); err == nil {
		t.Fatal("Project(wrong token width) = nil, want hidden-width error")
	}
}

// TestMiniMaxM2_RouterProject_Good checks the dense router projection plus the
// bias-add arm produce the expected per-expert scores.
func TestMiniMaxM2_RouterProject_Good(t *testing.T) {
	router := miniMaxM2NativeRouterWeights{
		NumExperts: 2,
		HiddenSize: 2,
		Weight:     []float32{1, 0, 0, 2}, // expert0 = x0, expert1 = 2*x1
		Bias:       []float32{0.5, -1},
	}
	scores, err := router.Project([][]float32{{3, 4}})
	if err != nil {
		t.Fatalf("Project() error = %v", err)
	}
	floatSliceApprox(t, scores[0], []float32{3*1 + 0.5, 4*2 - 1})
}

// TestMiniMaxM2_RouteTokens_Bad covers the top-k metadata and per-token score
// count rejection arms of routeMiniMaxM2NativeTokens.
func TestMiniMaxM2_RouteTokens_Bad(t *testing.T) {
	if _, _, err := routeMiniMaxM2NativeTokens(miniMaxM2LoadConfig{NumLocalExperts: 4, NumExpertsPerToken: 0}, [][]float32{{1, 2, 3, 4}}); err == nil {
		t.Fatal("routeMiniMaxM2NativeTokens(zero top-k) = nil, want metadata error")
	}
	if _, _, err := routeMiniMaxM2NativeTokens(miniMaxM2LoadConfig{NumLocalExperts: 4, NumExpertsPerToken: 5}, [][]float32{{1, 2, 3, 4}}); err == nil {
		t.Fatal("routeMiniMaxM2NativeTokens(top-k>experts) = nil, want metadata error")
	}
	if _, _, err := routeMiniMaxM2NativeTokens(miniMaxM2LoadConfig{NumLocalExperts: 4, NumExpertsPerToken: 2}, [][]float32{{1, 2}}); err == nil {
		t.Fatal("routeMiniMaxM2NativeTokens(short score row) = nil, want count error")
	}
}

// TestMiniMaxM2_DispatchExperts_Bad covers the dispatch guards: token/decision
// count mismatch, misindexed decision, and a missing expert payload.
func TestMiniMaxM2_DispatchExperts_Bad(t *testing.T) {
	hidden := [][]float32{{1, 1}}
	if _, err := dispatchMiniMaxM2NativeExperts(hidden, nil, nil); err == nil {
		t.Fatal("dispatch(token/decision mismatch) = nil, want error")
	}
	misindexed := []miniMaxM2NativeRouterDecision{{TokenIndex: 7, ExpertIDs: []int{0}}}
	if _, err := dispatchMiniMaxM2NativeExperts(hidden, misindexed, nil); err == nil {
		t.Fatal("dispatch(misindexed decision) = nil, want error")
	}
	missing := []miniMaxM2NativeRouterDecision{{TokenIndex: 0, ExpertIDs: []int{3}, Weights: []float32{1}}}
	if _, err := dispatchMiniMaxM2NativeExperts(hidden, missing, map[int]miniMaxM2NativeExpertPayload{}); err == nil {
		t.Fatal("dispatch(missing expert payload) = nil, want error")
	}
}

// TestMiniMaxM2_ResolveExpertPayloadRefs_Bad covers the no-metadata and
// out-of-range expert arms.
func TestMiniMaxM2_ResolveExpertPayloadRefs_Bad(t *testing.T) {
	empty := miniMaxM2NativeLoadPlan{}
	if _, err := empty.ResolveExpertPayloadRefs(0, []int{0}); err == nil {
		t.Fatal("ResolveExpertPayloadRefs(no metadata) = nil, want error")
	}
	plan := miniMaxM2NativeLoadPlan{
		Config:     miniMaxM2LoadConfig{NumLocalExperts: 2},
		TensorRefs: map[string]miniMaxM2SafetensorTensorRef{"x": {Name: "x"}},
	}
	if _, err := plan.ResolveExpertPayloadRefs(0, []int{5}); err == nil {
		t.Fatal("ResolveExpertPayloadRefs(out-of-range expert) = nil, want error")
	}
	if _, err := plan.ResolveExpertPayloadRefs(0, []int{-1}); err == nil {
		t.Fatal("ResolveExpertPayloadRefs(negative expert) = nil, want error")
	}
}

// TestMiniMaxM2_ReadSafetensorNamesWrappers_Good drives the two thin name-set
// wrappers (readMiniMaxM2SafetensorNames + readMiniMaxM2SafetensorHeaderNames)
// against a written checkpoint, asserting they surface the tensor names.
func TestMiniMaxM2_ReadSafetensorNamesWrappers_Good(t *testing.T) {
	dir := t.TempDir()
	path := core.JoinPath(dir, "model.safetensors")
	writeMiniMaxM2SafetensorsHeader(t, path, miniMaxM2FirstLayerTensorNames(false))

	names, shards, err := readMiniMaxM2SafetensorNames(dir, dir)
	if err != nil {
		t.Fatalf("readMiniMaxM2SafetensorNames() error = %v", err)
	}
	if shards != 1 {
		t.Fatalf("shards = %d, want 1", shards)
	}
	if !names["model.layers.0.self_attn.q_proj.weight"] {
		t.Fatalf("names missing q_proj; got %d entries", len(names))
	}
	headerNames, err := readMiniMaxM2SafetensorHeaderNames(path)
	if err != nil {
		t.Fatalf("readMiniMaxM2SafetensorHeaderNames() error = %v", err)
	}
	if !headerNames["model.layers.0.block_sparse_moe.gate.weight"] {
		t.Fatalf("header names missing gate weight; got %d entries", len(headerNames))
	}
}

// TestMiniMaxM2_ReadSafetensorRefs_Bad covers the no-shards arm of
// readMiniMaxM2SafetensorRefs.
func TestMiniMaxM2_ReadSafetensorRefs_Bad(t *testing.T) {
	empty := t.TempDir()
	if _, _, err := readMiniMaxM2SafetensorRefs(empty, empty); err == nil {
		t.Fatal("readMiniMaxM2SafetensorRefs(no shards) = nil, want error")
	}
}

// TestMiniMaxM2_ReadSafetensorRefs_SingleFile_Good exercises the explicit
// single-.safetensors modelPath branch (not the directory glob).
func TestMiniMaxM2_ReadSafetensorRefs_SingleFile_Good(t *testing.T) {
	dir := t.TempDir()
	path := core.JoinPath(dir, "model.safetensors")
	writeMiniMaxM2SafetensorsHeader(t, path, miniMaxM2FirstLayerTensorNames(false))
	tensors, shards, err := readMiniMaxM2SafetensorRefs(path, dir)
	if err != nil {
		t.Fatalf("readMiniMaxM2SafetensorRefs(single file) error = %v", err)
	}
	if shards != 1 || len(tensors) == 0 {
		t.Fatalf("single-file read = %d shards / %d tensors, want 1 / >0", shards, len(tensors))
	}
}

// TestMiniMaxM2_ReadSafetensorHeaderRefs_Bad covers the invalid header-length
// guard with a zero-length-header file and a file truncated before the 8-byte
// length prefix completes.
func TestMiniMaxM2_ReadSafetensorHeaderRefs_Bad(t *testing.T) {
	dir := t.TempDir()
	zero := core.JoinPath(dir, "zero.safetensors")
	buf := make([]byte, 8)
	binary.LittleEndian.PutUint64(buf, 0)
	if result := core.WriteFile(zero, buf, 0o644); !result.OK {
		t.Fatalf("write zero header: %v", result.Value)
	}
	if _, err := readMiniMaxM2SafetensorHeaderRefs(zero); err == nil {
		t.Fatal("readMiniMaxM2SafetensorHeaderRefs(zero header) = nil, want invalid-length error")
	}
	short := core.JoinPath(dir, "short.safetensors")
	if result := core.WriteFile(short, []byte{1, 2, 3}, 0o644); !result.OK {
		t.Fatalf("write short header: %v", result.Value)
	}
	if _, err := readMiniMaxM2SafetensorHeaderRefs(short); err == nil {
		t.Fatal("readMiniMaxM2SafetensorHeaderRefs(short header) = nil, want read error")
	}
}

// TestMiniMaxM2_ReadSafetensorFloat32_DTypes_Good reads F16, BF16, F32 and F64
// tensors through the float reader, asserting each dtype branch decodes to the
// same float32 values. The existing tests only cover F32, leaving the other
// three branches and the float16 converter dark.
func TestMiniMaxM2_ReadSafetensorFloat32_DTypes_Good(t *testing.T) {
	dir := t.TempDir()
	path := core.JoinPath(dir, "floats.safetensors")
	tensors := []miniMaxM2TinyTensor{
		{Name: "f16", DType: "F16", Shape: []int64{3}, Raw: miniMaxM2F16Bytes(1.0, -2.0, 0.5)},
		{Name: "bf16", DType: "BF16", Shape: []int64{3}, Raw: miniMaxM2BF16Bytes(1.0, -2.0, 0.5)},
		miniMaxM2TinyF32Tensor("f32", []float32{1.0, -2.0, 0.5}, 3),
		{Name: "f64", DType: "F64", Shape: []int64{3}, Raw: miniMaxM2F64Bytes(1.0, -2.0, 0.5)},
	}
	writeMiniMaxM2TinySafetensors(t, path, tensors)
	refs, err := readMiniMaxM2SafetensorHeaderRefs(path)
	if err != nil {
		t.Fatalf("readMiniMaxM2SafetensorHeaderRefs() error = %v", err)
	}
	want := []float32{1.0, -2.0, 0.5}
	for _, name := range []string{"f16", "bf16", "f32", "f64"} {
		got, err := readMiniMaxM2SafetensorFloat32(refs[name])
		if err != nil {
			t.Fatalf("readMiniMaxM2SafetensorFloat32(%s) error = %v", name, err)
		}
		floatSliceApprox(t, got, want)
	}
}

// TestMiniMaxM2_ReadSafetensorFloat32_Bad covers the non-float dtype rejection
// and a per-dtype byte-length mismatch arm.
func TestMiniMaxM2_ReadSafetensorFloat32_Bad(t *testing.T) {
	dir := t.TempDir()
	path := core.JoinPath(dir, "bad.safetensors")
	tensors := []miniMaxM2TinyTensor{
		miniMaxM2TinyU8Tensor("u8", []byte{1, 2}, 2),
		// F16 header claims 4 elements but only 2 elements of bytes → length mismatch.
		{Name: "f16short", DType: "F16", Shape: []int64{4}, Raw: miniMaxM2F16Bytes(1.0, 2.0)},
	}
	writeMiniMaxM2TinySafetensors(t, path, tensors)
	refs, err := readMiniMaxM2SafetensorHeaderRefs(path)
	if err != nil {
		t.Fatalf("readMiniMaxM2SafetensorHeaderRefs() error = %v", err)
	}
	if _, err := readMiniMaxM2SafetensorFloat32(refs["u8"]); err == nil {
		t.Fatal("readMiniMaxM2SafetensorFloat32(U8) = nil, want non-float error")
	}
	if _, err := readMiniMaxM2SafetensorFloat32(refs["f16short"]); err == nil {
		t.Fatal("readMiniMaxM2SafetensorFloat32(short F16) = nil, want byte-length error")
	}
}

// TestMiniMaxM2_Float16ToFloat32_GoodBadUgly checks the half-precision decoder
// across normal values, the zero/infinity boundary, and subnormal/NaN edges.
func TestMiniMaxM2_Float16ToFloat32_GoodBadUgly(t *testing.T) {
	// Good — ordinary normalised values.
	good := map[uint16]float32{
		0x3C00: 1.0,  // 1.0
		0xC000: -2.0, // -2.0
		0x3800: 0.5,  // 0.5
		0x0000: 0.0,  // +0
	}
	for bits, want := range good {
		if got := miniMaxM2NativeFloat16ToFloat32(bits); float32(math.Abs(float64(got-want))) > 1e-6 {
			t.Fatalf("float16(0x%04x) = %v, want %v", bits, got, want)
		}
	}
	// Boundary — signed zero and infinities round-trip exactly.
	if got := miniMaxM2NativeFloat16ToFloat32(0x8000); !math.Signbit(float64(got)) || got != 0 {
		t.Fatalf("float16(0x8000) = %v, want -0", got)
	}
	if got := miniMaxM2NativeFloat16ToFloat32(0x7C00); !math.IsInf(float64(got), 1) {
		t.Fatalf("float16(0x7C00) = %v, want +Inf", got)
	}
	if got := miniMaxM2NativeFloat16ToFloat32(0xFC00); !math.IsInf(float64(got), -1) {
		t.Fatalf("float16(0xFC00) = %v, want -Inf", got)
	}
	// Ugly — smallest subnormal (2^-24) and a NaN.
	if got := miniMaxM2NativeFloat16ToFloat32(0x0001); float32(math.Abs(float64(got)-math.Pow(2, -24))) > 1e-12 {
		t.Fatalf("float16(0x0001) = %v, want 2^-24", got)
	}
	if got := miniMaxM2NativeFloat16ToFloat32(0x7E00); !math.IsNaN(float64(got)) {
		t.Fatalf("float16(0x7E00) = %v, want NaN", got)
	}
}

// TestMiniMaxM2_SoftmaxWeights_GoodBadUgly checks the routed-expert softmax: a
// normal mix, the empty-id short circuit, and the all-equal / overflow uniform
// fallback.
func TestMiniMaxM2_SoftmaxWeights_GoodBadUgly(t *testing.T) {
	// Good — distinct scores produce a monotonic, normalised distribution.
	w := miniMaxM2NativeSoftmaxWeights([]float32{0, 1, 2}, []int{0, 1, 2})
	if len(w) != 3 {
		t.Fatalf("len = %d, want 3", len(w))
	}
	sum := float64(0)
	for _, v := range w {
		sum += float64(v)
	}
	if math.Abs(sum-1) > 1e-5 {
		t.Fatalf("softmax sum = %v, want 1", sum)
	}
	if !(w[2] > w[1] && w[1] > w[0]) {
		t.Fatalf("weights = %v, want strictly increasing", w)
	}
	// Bad — empty id set short-circuits to nil.
	if got := miniMaxM2NativeSoftmaxWeights([]float32{1, 2}, nil); got != nil {
		t.Fatalf("softmax(empty ids) = %v, want nil", got)
	}
	// Ugly — non-finite scores drive the sum to NaN/Inf → uniform fallback.
	inf := float32(math.Inf(1))
	uni := miniMaxM2NativeSoftmaxWeights([]float32{inf, inf}, []int{0, 1})
	floatSliceApprox(t, uni, []float32{0.5, 0.5})
}

// TestMiniMaxM2_DTypeAndBitsHelpers_Good is a table over the small dtype/bits
// predicates and accessors, covering the branches the loader tests leave at the
// default arm only.
func TestMiniMaxM2_DTypeAndBitsHelpers_Good(t *testing.T) {
	for _, dt := range []string{"U8", "UINT8", "u8"} {
		if !miniMaxM2NativePackedDType(dt) {
			t.Fatalf("miniMaxM2NativePackedDType(%q) = false, want true", dt)
		}
	}
	if miniMaxM2NativePackedDType("F16") {
		t.Fatal("miniMaxM2NativePackedDType(F16) = true, want false")
	}
	for _, dt := range []string{"F16", "BF16", "F32", "F64", "f32"} {
		if !miniMaxM2NativeFloatDType(dt) {
			t.Fatalf("miniMaxM2NativeFloatDType(%q) = false, want true", dt)
		}
	}
	if miniMaxM2NativeFloatDType("U8") {
		t.Fatal("miniMaxM2NativeFloatDType(U8) = true, want false")
	}
	bytesWant := map[string]int64{"F16": 2, "BF16": 2, "F32": 4, "F64": 8, "U8": 0}
	for dt, want := range bytesWant {
		if got := miniMaxM2NativeDTypeBytes(dt); got != want {
			t.Fatalf("miniMaxM2NativeDTypeBytes(%q) = %d, want %d", dt, got, want)
		}
	}
	// Attention bits: explicit value vs default 8.
	explicit := miniMaxM2JANGLoadConfig{}
	explicit.MXTQBits.Attention = 6
	if got := miniMaxM2NativeAttentionBits(explicit); got != 6 {
		t.Fatalf("attention bits explicit = %d, want 6", got)
	}
	if got := miniMaxM2NativeAttentionBits(miniMaxM2JANGLoadConfig{}); got != 8 {
		t.Fatalf("attention bits default = %d, want 8", got)
	}
	// Routed-expert bits: mxtq override → bits_default → 2.
	jang := miniMaxM2JANGLoadConfig{}
	if got := miniMaxM2NativeRoutedExpertBits(jang); got != 2 {
		t.Fatalf("routed bits default = %d, want 2", got)
	}
	jang.Quantization.BitsDefault = 4
	if got := miniMaxM2NativeRoutedExpertBits(jang); got != 4 {
		t.Fatalf("routed bits from bits_default = %d, want 4", got)
	}
	jang.MXTQBits.RoutedExpert = 3
	if got := miniMaxM2NativeRoutedExpertBits(jang); got != 3 {
		t.Fatalf("routed bits from mxtq = %d, want 3", got)
	}
}

// TestMiniMaxM2_SmallHelpers_Good covers the remaining pure helpers: slice
// equality, int32 shape conversion (good + invalid), packed-byte sizing, and
// the suffix trimmers.
func TestMiniMaxM2_SmallHelpers_Good(t *testing.T) {
	if !sameMiniMaxM2Uint64Slice([]uint64{1, 2}, []uint64{1, 2}) {
		t.Fatal("sameMiniMaxM2Uint64Slice(equal) = false")
	}
	if sameMiniMaxM2Uint64Slice([]uint64{1, 2}, []uint64{1, 3}) {
		t.Fatal("sameMiniMaxM2Uint64Slice(value mismatch) = true")
	}
	if sameMiniMaxM2Uint64Slice([]uint64{1}, []uint64{1, 2}) {
		t.Fatal("sameMiniMaxM2Uint64Slice(length mismatch) = true")
	}
	shape, err := miniMaxM2NativeInt32Shape([]uint64{2, 3})
	if err != nil || len(shape) != 2 || shape[0] != 2 || shape[1] != 3 {
		t.Fatalf("miniMaxM2NativeInt32Shape good = %v, %v", shape, err)
	}
	if _, err := miniMaxM2NativeInt32Shape(nil); err == nil {
		t.Fatal("miniMaxM2NativeInt32Shape(empty) = nil error, want error")
	}
	if _, err := miniMaxM2NativeInt32Shape([]uint64{0}); err == nil {
		t.Fatal("miniMaxM2NativeInt32Shape(zero dim) = nil error, want error")
	}
	// Packed bytes: 4 elements at 2 bits = 1 byte; a zero dim collapses to 0;
	// bits<=0 defaults to 8.
	if got := miniMaxM2NativePackedBytes([]uint64{2, 2}, 2); got != 1 {
		t.Fatalf("packed bytes 2x2 q2 = %d, want 1", got)
	}
	if got := miniMaxM2NativePackedBytes([]uint64{0, 4}, 2); got != 0 {
		t.Fatalf("packed bytes with zero dim = %d, want 0", got)
	}
	if got := miniMaxM2NativePackedBytes([]uint64{2}, 0); got != 2 {
		t.Fatalf("packed bytes bits<=0 default8 = %d, want 2", got)
	}
	if got := trimMiniMaxM2NativeWeightSuffix("a.b.weight"); got != "a.b" {
		t.Fatalf("trim weight suffix = %q, want a.b", got)
	}
	if got := trimMiniMaxM2NativeWeightSuffix("a.b"); got != "a.b" {
		t.Fatalf("trim weight suffix (none) = %q, want a.b", got)
	}
	if got := trimMiniMaxM2NativePackedSuffix("a.b.qweight"); got != "a.b" {
		t.Fatalf("trim packed suffix = %q, want a.b", got)
	}
	if got := trimMiniMaxM2NativePackedSuffix("a.b"); got != "a.b" {
		t.Fatalf("trim packed suffix (none) = %q, want a.b", got)
	}
}

// TestMiniMaxM2_FirstHelpers_Good covers the firstPositiveInt /
// firstNonEmptyString / firstNonEmptyUpper fallback walkers.
func TestMiniMaxM2_FirstHelpers_Good(t *testing.T) {
	if got := firstPositiveInt(0, -1, 7, 9); got != 7 {
		t.Fatalf("firstPositiveInt = %d, want 7", got)
	}
	if got := firstPositiveInt(0, -1); got != 0 {
		t.Fatalf("firstPositiveInt(none) = %d, want 0", got)
	}
	if got := firstNonEmptyString("", "", "x", "y"); got != "x" {
		t.Fatalf("firstNonEmptyString = %q, want x", got)
	}
	if got := firstNonEmptyString("", ""); got != "" {
		t.Fatalf("firstNonEmptyString(none) = %q, want empty", got)
	}
	if got := firstNonEmptyUpper("", "mxtq"); got != "MXTQ" {
		t.Fatalf("firstNonEmptyUpper = %q, want MXTQ", got)
	}
	if got := firstNonEmptyUpper("", ""); got != "" {
		t.Fatalf("firstNonEmptyUpper(none) = %q, want empty", got)
	}
}

// miniMaxM2F16Bytes encodes float32 values into little-endian IEEE-754 half
// precision for the dtype reader tests.
func miniMaxM2F16Bytes(values ...float32) []byte {
	out := make([]byte, len(values)*2)
	for i, v := range values {
		binary.LittleEndian.PutUint16(out[i*2:], miniMaxM2Float32ToFloat16(v))
	}
	return out
}

// miniMaxM2BF16Bytes encodes float32 values into little-endian bfloat16 (the
// high 16 bits of the float32 representation).
func miniMaxM2BF16Bytes(values ...float32) []byte {
	out := make([]byte, len(values)*2)
	for i, v := range values {
		binary.LittleEndian.PutUint16(out[i*2:], uint16(math.Float32bits(v)>>16))
	}
	return out
}

// miniMaxM2F64Bytes encodes float32 values widened to little-endian float64.
func miniMaxM2F64Bytes(values ...float32) []byte {
	out := make([]byte, len(values)*8)
	for i, v := range values {
		binary.LittleEndian.PutUint64(out[i*8:], math.Float64bits(float64(v)))
	}
	return out
}

// miniMaxM2Float32ToFloat16 is a round-to-nearest float32→half encoder used
// only to build test fixtures (the test values are exactly representable).
func miniMaxM2Float32ToFloat16(f float32) uint16 {
	bits := math.Float32bits(f)
	sign := uint16((bits >> 16) & 0x8000)
	exp := int((bits>>23)&0xff) - 127 + 15
	frac := bits & 0x7fffff
	if exp <= 0 {
		return sign
	}
	if exp >= 0x1f {
		return sign | 0x7c00
	}
	return sign | uint16(exp<<10) | uint16(frac>>13)
}

func writeMiniMaxM2TinyJANGConfig(t *testing.T, dir string) {
	t.Helper()
	if err := coreio.Local.Write(core.JoinPath(dir, "jang_config.json"), `{
		"weight_format": "mxtq",
		"profile": "JANGTQ",
		"mxtq_bits": {"attention": 8, "routed_expert": 2},
		"quantization": {"method": "affine+mxtq", "group_size": 4, "bits_default": 2}
	}`); err != nil {
		t.Fatalf("write jang_config.json: %v", err)
	}
}

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

func writeMiniMaxM2JANGConfig(t *testing.T, dir string) {
	t.Helper()
	if err := coreio.Local.Write(core.JoinPath(dir, "jang_config.json"), `{
		"version": 1,
		"weight_format": "mxtq",
		"profile": "JANGTQ_K",
		"mxtq_bits": {
			"attention": 8,
			"routed_expert": 2,
			"embed_tokens": 8,
			"lm_head": 8
		},
		"quantization": {
			"method": "affine+mxtq",
			"group_size": 64,
			"bits_default": 2
		}
	}`); err != nil {
		t.Fatalf("write jang_config.json: %v", err)
	}
}

func miniMaxM2FirstLayerTensorNames(omitExpertUp bool) []string {
	names := []string{
		"model.layers.0.self_attn.q_proj.weight",
		"model.layers.0.self_attn.k_proj.weight",
		"model.layers.0.self_attn.v_proj.weight",
		"model.layers.0.self_attn.o_proj.weight",
		"model.layers.0.block_sparse_moe.gate.weight",
		"model.layers.0.block_sparse_moe.e_score_correction_bias",
		"model.layers.0.block_sparse_moe.experts.0.gate_proj.weight",
		"model.layers.0.block_sparse_moe.experts.0.down_proj.weight",
	}
	if !omitExpertUp {
		names = append(names, "model.layers.0.block_sparse_moe.experts.0.up_proj.weight")
	}
	return names
}

func writeMiniMaxM2SafetensorsHeader(t *testing.T, path string, names []string) {
	t.Helper()
	type entry struct {
		DType       string `json:"dtype"`
		Shape       []int  `json:"shape"`
		DataOffsets [2]int `json:"data_offsets"`
	}
	header := map[string]entry{}
	cursor := 0
	for _, name := range names {
		dtype, shape, byteLen := miniMaxM2TestSafetensorsTensorLayout(name)
		header[name] = entry{DType: dtype, Shape: shape, DataOffsets: [2]int{cursor, cursor + byteLen}}
		cursor += byteLen
	}
	encoded := core.JSONMarshal(header)
	if !encoded.OK {
		t.Fatalf("marshal safetensors header: %v", encoded.Value)
	}
	headerBytes := encoded.Value.([]byte)
	out := make([]byte, 8+len(headerBytes))
	binary.LittleEndian.PutUint64(out[:8], uint64(len(headerBytes)))
	copy(out[8:], headerBytes)
	if result := core.WriteFile(path, out, 0o644); !result.OK {
		t.Fatalf("write safetensors header: %v", result.Value)
	}
}

func miniMaxM2TestSafetensorsTensorLayout(name string) (string, []int, int) {
	const (
		hidden       = 3072
		qSize        = 6144
		kvSize       = 1024
		intermediate = 1536
		experts      = 256
	)
	switch {
	case core.Contains(name, "self_attn.q_proj.weight"):
		bytes := qSize * hidden
		return "U8", []int{bytes}, bytes
	case core.Contains(name, "self_attn.k_proj.weight"), core.Contains(name, "self_attn.v_proj.weight"):
		bytes := kvSize * hidden
		return "U8", []int{bytes}, bytes
	case core.Contains(name, "self_attn.o_proj.weight"):
		bytes := hidden * qSize
		return "U8", []int{bytes}, bytes
	case core.Contains(name, "block_sparse_moe.gate.weight"):
		return "F32", []int{experts, hidden}, experts * hidden * 4
	case core.Contains(name, "e_score_correction_bias"):
		return "F32", []int{experts}, experts * 4
	case core.Contains(name, ".gate_proj.weight"), core.Contains(name, ".up_proj.weight"):
		bytes := (intermediate * hidden * 2) / 8
		return "U8", []int{bytes}, bytes
	case core.Contains(name, ".down_proj.weight"):
		bytes := (hidden * intermediate * 2) / 8
		return "U8", []int{bytes}, bytes
	default:
		return "F32", []int{1}, 4
	}
}

func writeMiniMaxM2TinyPayloadSafetensors(t *testing.T, path string) {
	t.Helper()
	identity := packMiniMaxM2TinyQ2(t, []uint8{1, 0, 0, 1})
	tensors := []miniMaxM2TinyTensor{
		miniMaxM2TinyU8Tensor("model.layers.0.self_attn.q_proj.weight", []byte{0, 0, 0, 0}, 4),
		miniMaxM2TinyU8Tensor("model.layers.0.self_attn.k_proj.weight", []byte{0, 0, 0, 0}, 4),
		miniMaxM2TinyU8Tensor("model.layers.0.self_attn.v_proj.weight", []byte{0, 0, 0, 0}, 4),
		miniMaxM2TinyU8Tensor("model.layers.0.self_attn.o_proj.weight", []byte{0, 0, 0, 0}, 4),
		miniMaxM2TinyF32Tensor("model.layers.0.block_sparse_moe.gate.weight", []float32{1, 0}, 1, 2),
		miniMaxM2TinyU8Tensor("model.layers.0.block_sparse_moe.experts.0.gate_proj.weight", identity, 1),
		miniMaxM2TinyF32Tensor("model.layers.0.block_sparse_moe.experts.0.gate_proj.weight.scales", []float32{1}, 1),
		miniMaxM2TinyF32Tensor("model.layers.0.block_sparse_moe.experts.0.gate_proj.weight.biases", []float32{0}, 1),
		miniMaxM2TinyU8Tensor("model.layers.0.block_sparse_moe.experts.0.up_proj.weight", identity, 1),
		miniMaxM2TinyF32Tensor("model.layers.0.block_sparse_moe.experts.0.up_proj.weight.scales", []float32{1}, 1),
		miniMaxM2TinyF32Tensor("model.layers.0.block_sparse_moe.experts.0.up_proj.weight.biases", []float32{0}, 1),
		miniMaxM2TinyU8Tensor("model.layers.0.block_sparse_moe.experts.0.down_proj.weight", identity, 1),
		miniMaxM2TinyF32Tensor("model.layers.0.block_sparse_moe.experts.0.down_proj.weight.scales", []float32{1}, 1),
		miniMaxM2TinyF32Tensor("model.layers.0.block_sparse_moe.experts.0.down_proj.weight.biases", []float32{0}, 1),
	}
	writeMiniMaxM2TinySafetensors(t, path, tensors)
}

func writeMiniMaxM2TinyRoutedPayloadSafetensors(t *testing.T, path string) {
	t.Helper()
	identity := packMiniMaxM2TinyQ2(t, []uint8{1, 0, 0, 1})
	tensors := []miniMaxM2TinyTensor{
		miniMaxM2TinyU8Tensor("model.layers.0.self_attn.q_proj.weight", []byte{0, 0, 0, 0}, 4),
		miniMaxM2TinyU8Tensor("model.layers.0.self_attn.k_proj.weight", []byte{0, 0, 0, 0}, 4),
		miniMaxM2TinyU8Tensor("model.layers.0.self_attn.v_proj.weight", []byte{0, 0, 0, 0}, 4),
		miniMaxM2TinyU8Tensor("model.layers.0.self_attn.o_proj.weight", []byte{0, 0, 0, 0}, 4),
		miniMaxM2TinyF32Tensor("model.layers.0.block_sparse_moe.gate.weight", []float32{
			0, 0,
			-2, 0,
			3, 0,
		}, 3, 2),
	}
	tensors = append(tensors, miniMaxM2TinyExpertPayloadTensors(t, 0, identity)...)
	tensors = append(tensors, miniMaxM2TinyExpertPayloadTensors(t, 2, identity)...)
	writeMiniMaxM2TinySafetensors(t, path, tensors)
}

func miniMaxM2TinyExpertPayloadTensors(t *testing.T, expertID int, packed []byte) []miniMaxM2TinyTensor {
	t.Helper()
	prefix := core.Sprintf("model.layers.0.block_sparse_moe.experts.%d.", expertID)
	return []miniMaxM2TinyTensor{
		miniMaxM2TinyU8Tensor(prefix+"gate_proj.weight", packed, 1),
		miniMaxM2TinyF32Tensor(prefix+"gate_proj.weight.scales", []float32{1}, 1),
		miniMaxM2TinyF32Tensor(prefix+"gate_proj.weight.biases", []float32{0}, 1),
		miniMaxM2TinyU8Tensor(prefix+"up_proj.weight", packed, 1),
		miniMaxM2TinyF32Tensor(prefix+"up_proj.weight.scales", []float32{1}, 1),
		miniMaxM2TinyF32Tensor(prefix+"up_proj.weight.biases", []float32{0}, 1),
		miniMaxM2TinyU8Tensor(prefix+"down_proj.weight", packed, 1),
		miniMaxM2TinyF32Tensor(prefix+"down_proj.weight.scales", []float32{1}, 1),
		miniMaxM2TinyF32Tensor(prefix+"down_proj.weight.biases", []float32{0}, 1),
	}
}

type miniMaxM2TinyTensor struct {
	Name  string
	DType string
	Shape []int64
	Raw   []byte
}

func miniMaxM2TinyU8Tensor(name string, raw []byte, shape ...int64) miniMaxM2TinyTensor {
	return miniMaxM2TinyTensor{Name: name, DType: "U8", Shape: shape, Raw: append([]byte(nil), raw...)}
}

func miniMaxM2TinyF32Tensor(name string, values []float32, shape ...int64) miniMaxM2TinyTensor {
	raw := make([]byte, len(values)*4)
	for i, value := range values {
		binary.LittleEndian.PutUint32(raw[i*4:], math.Float32bits(value))
	}
	return miniMaxM2TinyTensor{Name: name, DType: "F32", Shape: shape, Raw: raw}
}

func writeMiniMaxM2TinySafetensors(t *testing.T, path string, tensors []miniMaxM2TinyTensor) {
	t.Helper()
	type entry struct {
		DType       string  `json:"dtype"`
		Shape       []int64 `json:"shape"`
		DataOffsets []int64 `json:"data_offsets"`
	}
	header := map[string]entry{}
	var payload []byte
	for _, tensor := range tensors {
		start := int64(len(payload))
		payload = append(payload, tensor.Raw...)
		header[tensor.Name] = entry{DType: tensor.DType, Shape: tensor.Shape, DataOffsets: []int64{start, int64(len(payload))}}
	}
	encoded := core.JSONMarshal(header)
	if !encoded.OK {
		t.Fatalf("marshal safetensors header: %v", encoded.Value)
	}
	headerBytes := encoded.Value.([]byte)
	out := make([]byte, 8+len(headerBytes)+len(payload))
	binary.LittleEndian.PutUint64(out[:8], uint64(len(headerBytes)))
	copy(out[8:], headerBytes)
	copy(out[8+len(headerBytes):], payload)
	if result := core.WriteFile(path, out, 0o644); !result.OK {
		t.Fatalf("write safetensors: %v", result.Value)
	}
}

func packMiniMaxM2TinyQ2(t *testing.T, values []uint8) []byte {
	t.Helper()
	out := make([]byte, (len(values)*2+7)/8)
	for i, value := range values {
		if value > 3 {
			t.Fatalf("q2 value %d exceeds max 3", value)
		}
		out[i/4] |= byte(value << ((i % 4) * 2))
	}
	return out
}

func silu64(value float64) float64 {
	return value / (1 + math.Exp(-value))
}

func requireMetalRuntime(t testing.TB) {
	t.Helper()
	if !metal.MetalAvailable() {
		t.Skip("Metal runtime unavailable")
	}
}

func floatSliceApprox(t *testing.T, got []float32, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("len(got) = %d, want %d; got=%v want=%v", len(got), len(want), got, want)
	}
	const tolerance = 1e-3
	for i := range got {
		diff := math.Abs(float64(got[i] - want[i]))
		if diff > tolerance {
			t.Fatalf("got[%d] = %.6f, want %.6f (diff %.6f); got=%v want=%v", i, got[i], want[i], diff, got, want)
		}
	}
}

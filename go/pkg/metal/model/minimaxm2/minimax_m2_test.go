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

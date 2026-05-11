// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"encoding/binary"
	"math"
	"testing"

	"dappco.re/go"

	coreio "dappco.re/go/io"
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

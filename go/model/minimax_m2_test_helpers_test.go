// SPDX-Licence-Identifier: EUPL-1.2

package model

import (
	"encoding/binary"
	"math"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/quant/jang"
	"dappco.re/go/mlx/model/minimax/m2"
)

// MiniMax M2 fixture config + safetensors helpers shared between
// jang_darwin_test.go and model_pack_test.go. The canonical fixture
// data also lives at go-mlx/model/minimax/m2/m2_test.go; these
// duplicates exist because Go test packages cannot import each other's
// internal test helpers.

const miniMaxM2FixtureConfig = `{
	"architectures": ["MiniMaxM2ForCausalLM"],
	"model_type": "minimax_m2",
	"vocab_size": 200064,
	"hidden_size": 3072,
	"intermediate_size": 1536,
	"num_hidden_layers": 62,
	"num_attention_heads": 48,
	"num_key_value_heads": 8,
	"head_dim": 128,
	"max_position_embeddings": 196608,
	"num_local_experts": 256,
	"num_experts_per_tok": 8,
	"scoring_func": "sigmoid",
	"use_routing_bias": true,
	"use_mtp": true,
	"num_mtp_modules": 3,
	"mtp_transformer_layers": 1,
	"use_qk_norm": true,
	"rotary_dim": 64,
	"rope_theta": 5000000
}`

func findMiniMaxM2Spec(specs []m2.TensorSpec, role m2.TensorRole) m2.TensorSpec {
	for _, spec := range specs {
		if spec.Role == role {
			return spec
		}
	}
	return m2.TensorSpec{}
}

func miniMaxM2SkeletonRawTensors(t *testing.T, plan m2.TensorPlan, badAttentionShape bool) []miniMaxM2RawSafetensor {
	t.Helper()
	specs, err := plan.LayerTensorSpecs(0, 0)
	if err != nil {
		t.Fatalf("LayerTensorSpecs() error = %v", err)
	}
	var tensors []miniMaxM2RawSafetensor
	for _, role := range []m2.TensorRole{
		m2.TensorRoleAttentionQ,
		m2.TensorRoleAttentionK,
		m2.TensorRoleAttentionV,
		m2.TensorRoleAttentionO,
	} {
		spec := findMiniMaxM2Spec(specs, role)
		if spec.Packed == nil {
			t.Fatalf("attention spec %s has no packed descriptor", role)
		}
		packedBytes := spec.Packed.PackedBytes
		if badAttentionShape && role == m2.TensorRoleAttentionQ {
			packedBytes--
		}
		tensors = append(tensors, miniMaxM2RawSafetensor{
			Name:  spec.Name,
			DType: "U8",
			Shape: []int{packedBytes},
			Raw:   make([]byte, packedBytes),
		})
	}
	tensors = append(tensors,
		miniMaxM2F32RawTensor("model.layers.0.block_sparse_moe.gate.weight", []float32{
			1, 0, 0, 1,
			0, 1, 1, 0,
			1, 1, 0, 0,
		}, 3, 4),
	)
	if plan.Config.UseRoutingBias {
		tensors = append(tensors, miniMaxM2F32RawTensor("model.layers.0.block_sparse_moe.e_score_correction_bias", []float32{0, 0.25, -0.25}, 3))
	}
	return tensors
}

type miniMaxM2RawSafetensor struct {
	Name  string
	DType string
	Shape []int
	Raw   []byte
}

func miniMaxM2F32RawTensor(name string, values []float32, shape ...int) miniMaxM2RawSafetensor {
	raw := make([]byte, len(values)*4)
	for i, value := range values {
		binary.LittleEndian.PutUint32(raw[i*4:], math.Float32bits(value))
	}
	if len(shape) == 0 {
		shape = []int{len(values)}
	}
	return miniMaxM2RawSafetensor{Name: name, DType: "F32", Shape: append([]int(nil), shape...), Raw: raw}
}

func writeMiniMaxM2RawSafetensors(t *testing.T, path string, tensors []miniMaxM2RawSafetensor) {
	t.Helper()
	type entry struct {
		DType       string `json:"dtype"`
		Shape       []int  `json:"shape"`
		DataOffsets []int  `json:"data_offsets"`
	}
	header := map[string]entry{}
	var data []byte
	for _, tensor := range tensors {
		start := len(data)
		data = append(data, tensor.Raw...)
		header[tensor.Name] = entry{
			DType:       tensor.DType,
			Shape:       tensor.Shape,
			DataOffsets: []int{start, len(data)},
		}
	}
	encoded := core.JSONMarshal(header)
	if !encoded.OK {
		t.Fatalf("marshal safetensors header: %v", encoded.Value)
	}
	headerBytes := encoded.Value.([]byte)
	out := make([]byte, 8+len(headerBytes)+len(data))
	binary.LittleEndian.PutUint64(out[:8], uint64(len(headerBytes)))
	copy(out[8:], headerBytes)
	copy(out[8+len(headerBytes):], data)
	if result := core.WriteFile(path, out, 0o644); !result.OK {
		t.Fatalf("write safetensors: %v", result.Value)
	}
}

// silence unused-import in non-darwin builds
var _ = jang.Info{}

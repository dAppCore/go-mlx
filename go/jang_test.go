// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	core "dappco.re/go"
	"dappco.re/go/inference/quant/jang"
	"dappco.re/go/mlx/model/minimax/m2"
	mlxjang "dappco.re/go/mlx/quant/jang"
	"encoding/binary"
	"math"
	"testing"
)

func testJANGTQInfo() *jang.Info {
	info := &jang.Info{
		Version:          2,
		WeightFormat:     "mxtq",
		Profile:          "JANGTQ",
		Method:           "affine+mxtq",
		GroupSize:        4,
		BitsDefault:      2,
		AttentionBits:    8,
		SharedExpertBits: 8,
		RoutedExpertBits: 2,
		EmbedTokensBits:  8,
		LMHeadBits:       8,
	}
	info.Packed = jang.BuildPackedProfile(info)
	return info
}

func TestJANGNative_DequantizePackedTensorMetalMatchesReference_Good(t *testing.T) {
	skipIfNoUsableMetal(t)

	cfg, err := m2.ParseConfig([]byte(miniMaxM2FixtureConfig))
	if err != nil {
		t.Fatalf("ParseMiniMaxM2Config() error = %v", err)
	}
	plan, err := m2.BuildTensorPlan(cfg, testJANGTQInfo())
	if err != nil {
		t.Fatalf("BuildMiniMaxM2TensorPlan() error = %v", err)
	}
	specs, err := plan.LayerTensorSpecs(0, 0)
	if err != nil {
		t.Fatalf("LayerTensorSpecs() error = %v", err)
	}
	expert := findMiniMaxM2Spec(specs, m2.TensorRoleExpertGate)
	if expert.Packed == nil {
		t.Fatal("expert packed descriptor is nil")
	}
	desc := *expert.Packed
	desc.Shape = []uint64{2, 4}
	desc.Elements = 8
	desc.GroupSize = 4
	desc.Groups = 2
	desc.PackedBytes = 2
	desc.ScaleCount = 2
	desc.BiasCount = 2

	values := []uint8{0, 1, 2, 3, 3, 2, 1, 0}
	packed, err := jang.PackQuantizedValues(desc, values)
	if err != nil {
		t.Fatalf("jang.PackQuantizedValues() error = %v", err)
	}
	scales := []float32{0.5, 1.25}
	biases := []float32{-1, 2}
	want, err := jang.DequantizePackedTensor(desc, packed, scales, biases)
	if err != nil {
		t.Fatalf("jang.DequantizePackedTensor() error = %v", err)
	}

	got, err := mlxjang.DequantizePackedTensor(desc, packed, scales, biases)
	if err != nil {
		t.Fatalf("mlxjang.DequantizePackedTensor() error = %v", err)
	}
	if !float32SlicesRoughlyEqual(got, want, 1e-5) {
		t.Fatalf("got = %+v, want %+v", got, want)
	}
}

func TestJANGNative_ProjectPackedTensorMetalMatchesCPUProjection_Good(t *testing.T) {
	skipIfNoUsableMetal(t)

	desc := jang.PackedTensorDescriptor{
		Name:          "model.layers.0.block_sparse_moe.experts.0.gate_proj.weight",
		Type:          "jangtq",
		Format:        "mxtq",
		Role:          jang.TensorRoleRoutedExpert,
		Shape:         []uint64{3, 4},
		Elements:      12,
		Bits:          2,
		GroupSize:     4,
		Groups:        3,
		PackedBytes:   3,
		ValuesPerByte: 4,
		ScaleCount:    3,
		BiasCount:     3,
		BitOrder:      jang.BitOrderLSB0,
		Encoding:      jang.EncodingAffine,
	}
	values := []uint8{0, 1, 2, 3, 3, 2, 1, 0, 1, 1, 2, 2}
	packed, err := jang.PackQuantizedValues(desc, values)
	if err != nil {
		t.Fatalf("jang.PackQuantizedValues() error = %v", err)
	}
	scales := []float32{0.5, 1.25, -0.75}
	biases := []float32{-1, 2, 5}
	input := []float32{
		1, 2, 3, 4,
		-1, 0.5, 2, -0.5,
	}
	projBias := []float32{0.25, -1, 2}

	got, err := mlxjang.ProjectPackedTensor(desc, packed, scales, biases, input, []int32{2, 4}, projBias)
	if err != nil {
		t.Fatalf("mlxjang.ProjectPackedTensor() error = %v", err)
	}
	weight, err := jang.DequantizePackedTensor(desc, packed, scales, biases)
	if err != nil {
		t.Fatalf("jang.DequantizePackedTensor() error = %v", err)
	}
	want := denseProjectionReference(input, 2, weight, 3, 4, projBias)
	if !float32SlicesRoughlyEqual(got.Values, want, 1e-5) {
		t.Fatalf("got = %+v, want %+v", got.Values, want)
	}
	if len(got.Shape) != 2 || got.Shape[0] != 2 || got.Shape[1] != 3 {
		t.Fatalf("shape = %+v, want [2 3]", got.Shape)
	}
}

func TestJANGNative_ProjectPackedTensorMetalFusedMatchesComposedProjection_Good(t *testing.T) {
	skipIfNoUsableMetal(t)

	desc := jang.PackedTensorDescriptor{
		Name:          "model.layers.0.block_sparse_moe.experts.0.gate_proj.weight",
		Type:          "jangtq",
		Format:        "mxtq",
		Role:          jang.TensorRoleRoutedExpert,
		Shape:         []uint64{3, 4},
		Elements:      12,
		Bits:          2,
		GroupSize:     4,
		Groups:        3,
		PackedBytes:   3,
		ValuesPerByte: 4,
		ScaleCount:    3,
		BiasCount:     3,
		BitOrder:      jang.BitOrderLSB0,
		Encoding:      jang.EncodingAffine,
	}
	values := []uint8{0, 1, 2, 3, 3, 2, 1, 0, 1, 1, 2, 2}
	packed, err := jang.PackQuantizedValues(desc, values)
	if err != nil {
		t.Fatalf("jang.PackQuantizedValues() error = %v", err)
	}
	scales := []float32{0.5, 1.25, -0.75}
	biases := []float32{-1, 2, 5}
	input := []float32{
		1, 2, 3, 4,
		-1, 0.5, 2, -0.5,
	}
	projBias := []float32{0.25, -1, 2}

	got, err := mlxjang.ProjectPackedTensorFused(desc, packed, scales, biases, input, []int32{2, 4}, projBias)
	if err != nil {
		t.Fatalf("mlxjang.ProjectPackedTensorFused() error = %v", err)
	}
	want, err := mlxjang.ProjectPackedTensor(desc, packed, scales, biases, input, []int32{2, 4}, projBias)
	if err != nil {
		t.Fatalf("mlxjang.ProjectPackedTensor() error = %v", err)
	}
	if !float32SlicesRoughlyEqual(got.Values, want.Values, 1e-5) {
		t.Fatalf("got = %+v, want %+v", got.Values, want.Values)
	}
	if len(got.Shape) != 2 || got.Shape[0] != 2 || got.Shape[1] != 3 {
		t.Fatalf("shape = %+v, want [2 3]", got.Shape)
	}
}

func TestJANGNative_ProjectPackedTensorMetalRejectsInputMismatch_Bad(t *testing.T) {
	desc := jang.PackedTensorDescriptor{
		Name:        "bad",
		Shape:       []uint64{3, 4},
		Elements:    12,
		Bits:        2,
		GroupSize:   4,
		Groups:      3,
		PackedBytes: 3,
		ScaleCount:  3,
		BiasCount:   3,
	}
	_, err := mlxjang.ProjectPackedTensor(desc, []byte{0, 0, 0}, []float32{1, 1, 1}, []float32{0, 0, 0}, []float32{1, 2, 3}, []int32{1, 3}, nil)
	if err == nil {
		t.Fatal("expected input shape error")
	}
}

func TestJANGNative_ShapeValidationHelpers_Bad(t *testing.T) {
	if _, err := mlxjang.MetalShape(nil); err == nil {
		t.Fatal("expected empty JANG metal shape error")
	}
	if _, err := mlxjang.MetalShape([]uint64{0}); err == nil {
		t.Fatal("expected zero JANG metal shape error")
	}
	if _, err := mlxjang.MetalShape([]uint64{uint64(^uint32(0)>>1) + 1}); err == nil {
		t.Fatal("expected oversized JANG metal shape error")
	}
	shape, err := mlxjang.MetalShape([]uint64{2, 3})
	if err != nil {
		t.Fatalf("mlxjang.MetalShape(valid) error = %v", err)
	}
	if !equalInt32Slices(shape, []int32{2, 3}) {
		t.Fatalf("shape = %v, want [2 3]", shape)
	}
	if _, err := mlxjang.ShapeElements(nil); err == nil {
		t.Fatal("expected empty projection input shape error")
	}
	if _, err := mlxjang.ShapeElements([]int32{2, 0}); err == nil {
		t.Fatal("expected invalid projection input shape error")
	}
	if _, err := mlxjang.ShapeElements([]int32{1 << 30, 1 << 30, 8}); err == nil {
		t.Fatal("expected oversized projection input shape error")
	}
	if elements, err := mlxjang.ShapeElements([]int32{2, 3, 4}); err != nil || elements != 24 {
		t.Fatalf("mlxjang.ShapeElements(valid) = %d/%v, want 24/nil", elements, err)
	}
	if got := mlxjang.Int32SliceToInts([]int32{4, 5}); !equalIntSlices(got, []int{4, 5}) {
		t.Fatalf("mlxjang.Int32SliceToInts() = %v, want [4 5]", got)
	}
}

func float32SlicesRoughlyEqual(a, b []float32, epsilon float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		diff := a[i] - b[i]
		if diff < 0 {
			diff = -diff
		}
		if diff > epsilon {
			return false
		}
	}
	return true
}

func denseProjectionReference(input []float32, rows int, weight []float32, outDim, inDim int, bias []float32) []float32 {
	out := make([]float32, rows*outDim)
	for row := 0; row < rows; row++ {
		for outIndex := 0; outIndex < outDim; outIndex++ {
			sum := float32(0)
			for inIndex := 0; inIndex < inDim; inIndex++ {
				sum += input[row*inDim+inIndex] * weight[outIndex*inDim+inIndex]
			}
			if len(bias) > 0 {
				sum += bias[outIndex]
			}
			out[row*outDim+outIndex] = sum
		}
	}
	return out
}

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

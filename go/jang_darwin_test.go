// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && !nomlx

package mlx

import (
	"testing"

	"dappco.re/go/inference/quant/jang"
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

	cfg, err := ParseMiniMaxM2Config([]byte(miniMaxM2FixtureConfig))
	if err != nil {
		t.Fatalf("ParseMiniMaxM2Config() error = %v", err)
	}
	plan, err := BuildMiniMaxM2TensorPlan(cfg, testJANGTQInfo())
	if err != nil {
		t.Fatalf("BuildMiniMaxM2TensorPlan() error = %v", err)
	}
	specs, err := plan.LayerTensorSpecs(0, 0)
	if err != nil {
		t.Fatalf("LayerTensorSpecs() error = %v", err)
	}
	expert := findMiniMaxM2Spec(specs, MiniMaxM2TensorRoleExpertGate)
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

	got, err := DequantizeJANGPackedTensorMetal(desc, packed, scales, biases)
	if err != nil {
		t.Fatalf("DequantizeJANGPackedTensorMetal() error = %v", err)
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

	got, err := ProjectJANGPackedTensorMetal(desc, packed, scales, biases, input, []int32{2, 4}, projBias)
	if err != nil {
		t.Fatalf("ProjectJANGPackedTensorMetal() error = %v", err)
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

	got, err := ProjectJANGPackedTensorMetalFused(desc, packed, scales, biases, input, []int32{2, 4}, projBias)
	if err != nil {
		t.Fatalf("ProjectJANGPackedTensorMetalFused() error = %v", err)
	}
	want, err := ProjectJANGPackedTensorMetal(desc, packed, scales, biases, input, []int32{2, 4}, projBias)
	if err != nil {
		t.Fatalf("ProjectJANGPackedTensorMetal() error = %v", err)
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
	_, err := ProjectJANGPackedTensorMetal(desc, []byte{0, 0, 0}, []float32{1, 1, 1}, []float32{0, 0, 0}, []float32{1, 2, 3}, []int32{1, 3}, nil)
	if err == nil {
		t.Fatal("expected input shape error")
	}
}

func TestJANGNative_ShapeValidationHelpers_Bad(t *testing.T) {
	if _, err := jangMetalShape(nil); err == nil {
		t.Fatal("expected empty JANG metal shape error")
	}
	if _, err := jangMetalShape([]uint64{0}); err == nil {
		t.Fatal("expected zero JANG metal shape error")
	}
	if _, err := jangMetalShape([]uint64{uint64(^uint32(0)>>1) + 1}); err == nil {
		t.Fatal("expected oversized JANG metal shape error")
	}
	shape, err := jangMetalShape([]uint64{2, 3})
	if err != nil {
		t.Fatalf("jangMetalShape(valid) error = %v", err)
	}
	if !equalInt32Slices(shape, []int32{2, 3}) {
		t.Fatalf("shape = %v, want [2 3]", shape)
	}
	if _, err := jangMetalShapeElements(nil); err == nil {
		t.Fatal("expected empty projection input shape error")
	}
	if _, err := jangMetalShapeElements([]int32{2, 0}); err == nil {
		t.Fatal("expected invalid projection input shape error")
	}
	if _, err := jangMetalShapeElements([]int32{1 << 30, 1 << 30, 8}); err == nil {
		t.Fatal("expected oversized projection input shape error")
	}
	if elements, err := jangMetalShapeElements([]int32{2, 3, 4}); err != nil || elements != 24 {
		t.Fatalf("jangMetalShapeElements(valid) = %d/%v, want 24/nil", elements, err)
	}
	if got := int32SliceToInts([]int32{4, 5}); !equalIntSlices(got, []int{4, 5}) {
		t.Fatalf("int32SliceToInts() = %v, want [4 5]", got)
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

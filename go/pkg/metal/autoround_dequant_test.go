// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/quant/autoround"
)

func TestAutoRoundDequant_DequantizePackedW4MatchesCPUReference_Good(t *testing.T) {
	weights := make([]float32, 32)
	for i := range weights {
		weights[i] = float32(i-16) / 7
	}
	quantized, err := autoround.QuantizeWeights(weights, autoround.QuantizeConfig{Scheme: autoround.SchemeW4A16, GroupSize: 32, Iters: 0})
	if err != nil {
		t.Fatalf("QuantizeWeights() error = %v", err)
	}
	packed, err := autoround.PackQuantizedWeights(quantized, []int32{4, 8})
	if err != nil {
		t.Fatalf("PackQuantizedWeights() error = %v", err)
	}
	want, err := autoround.DequantizePackedWeights(packed)
	if err != nil {
		t.Fatalf("DequantizePackedWeights() error = %v", err)
	}

	gotArray, err := DequantizeAutoRoundPacked(
		FromValues(packed.Packed, len(packed.Packed)),
		FromValues(packed.Scales, len(packed.Scales)),
		FromValues(packed.ZeroPoints, len(packed.ZeroPoints)),
		packed.Shape,
		packed.GroupSize,
		packed.Bits,
		packed.QMin,
	)
	if err != nil {
		t.Fatalf("DequantizeAutoRoundPacked() error = %v", err)
	}
	Materialize(gotArray)

	assertFloat32SliceClose(t, gotArray.Floats(), want, 1e-5)
	if shape := gotArray.Shape(); len(shape) != 2 || shape[0] != 4 || shape[1] != 8 {
		t.Fatalf("shape = %+v, want [4 8]", shape)
	}
}

func TestAutoRoundDequant_FusedPackedLinearMatchesComposedProjection_Good(t *testing.T) {
	weights := []float32{
		-1.5, -0.75, 0, 0.5,
		1.25, -1, 0.25, 1.75,
		-0.5, 0.75, -1.25, 1,
	}
	quantized, err := autoround.QuantizeWeights(weights, autoround.QuantizeConfig{Scheme: autoround.SchemeW2A16, GroupSize: 32, Iters: 0})
	if err != nil {
		t.Fatalf("QuantizeWeights() error = %v", err)
	}
	packed, err := autoround.PackQuantizedWeights(quantized, []int32{3, 4})
	if err != nil {
		t.Fatalf("PackQuantizedWeights() error = %v", err)
	}
	input := FromValues([]float32{
		1, 2, 3, 4,
		-1, 0.5, 2, -0.5,
	}, 1, 2, 4)
	bias := FromValues([]float32{0.25, -1, 2}, 3)
	packedArray := FromValues(packed.Packed, len(packed.Packed))
	scaleArray := FromValues(packed.Scales, len(packed.Scales))
	zeroArray := FromValues(packed.ZeroPoints, len(packed.ZeroPoints))

	gotArray, err := AutoRoundPackedLinearFused(input, packedArray, scaleArray, zeroArray, bias, packed.Shape, packed.GroupSize, packed.Bits, packed.QMin)
	if err != nil {
		t.Fatalf("AutoRoundPackedLinearFused() error = %v", err)
	}
	wantArray, err := AutoRoundPackedLinear(input, packedArray, scaleArray, zeroArray, bias, packed.Shape, packed.GroupSize, packed.Bits, packed.QMin)
	if err != nil {
		t.Fatalf("AutoRoundPackedLinear() error = %v", err)
	}
	Materialize(gotArray, wantArray)

	assertFloat32SliceClose(t, gotArray.Floats(), wantArray.Floats(), 1e-5)
	if shape := gotArray.Shape(); len(shape) != 3 || shape[0] != 1 || shape[1] != 2 || shape[2] != 3 {
		t.Fatalf("shape = %+v, want [1 2 3]", shape)
	}
}

func TestAutoRoundDequant_FusedProjectionConsumesLoadedPayload_Good(t *testing.T) {
	projection := autoround.PackedProjection{
		Tensor: autoround.PackTensor{
			Name:       "model.layers.0.self_attn.q_proj.weight",
			Packed:     "model.layers.0.self_attn.q_proj.weight.packed",
			Scales:     "model.layers.0.self_attn.q_proj.weight.scales",
			ZeroPoints: "model.layers.0.self_attn.q_proj.weight.zeros",
			Shape:      []int32{3, 4},
			Bits:       2,
			GroupSize:  32,
			Symmetric:  true,
			QMin:       -2,
			QMax:       1,
		},
		Weights: autoround.PackedWeights{
			Bits:       2,
			GroupSize:  32,
			Symmetric:  true,
			Shape:      []int32{3, 4},
			Packed:     []byte{0b11100100, 0b01001110, 0b00111001},
			Scales:     []float32{0.5},
			ZeroPoints: []float32{0},
			QMin:       -2,
			QMax:       1,
		},
		Bias: []float32{0.25, -1, 2},
	}
	input := FromValues([]float32{
		1, 2, 3, 4,
		-1, 0.5, 2, -0.5,
	}, 2, 4)

	gotArray, err := AutoRoundPackedProjectionLinearFused(input, projection)
	if err != nil {
		t.Fatalf("AutoRoundPackedProjectionLinearFused() error = %v", err)
	}
	wantArray, err := AutoRoundPackedLinearFused(
		input,
		FromValues(projection.Weights.Packed, len(projection.Weights.Packed)),
		FromValues(projection.Weights.Scales, len(projection.Weights.Scales)),
		FromValues(projection.Weights.ZeroPoints, len(projection.Weights.ZeroPoints)),
		FromValues(projection.Bias, len(projection.Bias)),
		projection.Weights.Shape,
		projection.Weights.GroupSize,
		projection.Weights.Bits,
		projection.Weights.QMin,
	)
	if err != nil {
		t.Fatalf("AutoRoundPackedLinearFused() error = %v", err)
	}
	Materialize(gotArray, wantArray)

	assertFloat32SliceClose(t, gotArray.Floats(), wantArray.Floats(), 1e-5)
	if shape := gotArray.Shape(); len(shape) != 2 || shape[0] != 2 || shape[1] != 3 {
		t.Fatalf("shape = %+v, want [2 3]", shape)
	}
}

func TestAutoRoundDequant_DequantizePackedRejectsBadMetadata_Bad(t *testing.T) {
	_, err := DequantizeAutoRoundPacked(FromValues([]uint8{0}, 1), FromValues([]float32{1}, 1), FromValues([]float32{0}, 1), []int32{2}, 1, 5, -16)
	if err == nil || !core.Contains(err.Error(), "bits") {
		t.Fatalf("error = %v, want unsupported bits diagnostic", err)
	}

	_, err = DequantizeAutoRoundPacked(FromValues([]uint8{0}, 1), FromValues([]float32{1}, 1), FromValues([]float32{0}, 1), []int32{5}, 8, 2, -2)
	if err == nil || !core.Contains(err.Error(), "packed") {
		t.Fatalf("error = %v, want packed length diagnostic", err)
	}
}

func TestAutoRoundDequant_PackedLinearRejectsShapeMismatch_Bad(t *testing.T) {
	_, err := AutoRoundPackedLinear(FromValues([]float32{1, 2, 3}, 1, 3), FromValues([]uint8{0}, 1), FromValues([]float32{1}, 1), FromValues([]float32{0}, 1), nil, []int32{2, 2}, 32, 4, -8)
	if err == nil || !core.Contains(err.Error(), "input") {
		t.Fatalf("error = %v, want input shape diagnostic", err)
	}
}

// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"math"
	"testing"

	core "dappco.re/go"
)

func TestJANGDequant_DequantizePackedQ2MatchesCPUReference_Good(t *testing.T) {
	quantized := []uint8{0, 1, 2, 3, 3, 2, 1, 0, 2, 1}
	packed := packJANGTestValues(t, quantized, 2)
	scales := []float32{0.5, 1.25, -0.75}
	biases := []float32{-1, 2, 5}

	gotArray, err := DequantizeJANGPacked(FromValues(packed, len(packed)), FromValues(scales, len(scales)), FromValues(biases, len(biases)), []int32{2, 5}, 4, 2)
	if err != nil {
		t.Fatalf("DequantizeJANGPacked() error = %v", err)
	}
	Materialize(gotArray)

	got := gotArray.Floats()
	want := dequantizeJANGTestValues(quantized, scales, biases, 4)
	assertFloat32SliceClose(t, got, want, 1e-5)
	if shape := gotArray.Shape(); len(shape) != 2 || shape[0] != 2 || shape[1] != 5 {
		t.Fatalf("shape = %+v, want [2 5]", shape)
	}
}

func TestJANGDequant_DequantizePackedQ8MatchesCPUReference_Good(t *testing.T) {
	quantized := []uint8{0, 7, 128, 255, 64, 3}
	scales := []float32{0.25, -0.5}
	biases := []float32{1, 8}

	gotArray, err := DequantizeJANGPacked(FromValues(quantized, len(quantized)), FromValues(scales, len(scales)), FromValues(biases, len(biases)), []int32{2, 3}, 3, 8)
	if err != nil {
		t.Fatalf("DequantizeJANGPacked() error = %v", err)
	}
	Materialize(gotArray)

	got := gotArray.Floats()
	want := dequantizeJANGTestValues(quantized, scales, biases, 3)
	assertFloat32SliceClose(t, got, want, 1e-5)
}

func TestJANGDequant_DequantizePackedRejectsBadMetadata_Bad(t *testing.T) {
	_, err := DequantizeJANGPacked(FromValues([]uint8{0}, 1), FromValues([]float32{1}, 1), FromValues([]float32{0}, 1), []int32{2}, 1, 5)
	if err == nil || !core.Contains(err.Error(), "bits") {
		t.Fatalf("error = %v, want unsupported bits diagnostic", err)
	}

	_, err = DequantizeJANGPacked(FromValues([]uint8{0}, 1), FromValues([]float32{1}, 1), FromValues([]float32{0}, 1), []int32{5}, 8, 2)
	if err == nil || !core.Contains(err.Error(), "packed") {
		t.Fatalf("error = %v, want packed length diagnostic", err)
	}
}

func TestJANGDequant_PackedLinearMatchesDenseProjection_Good(t *testing.T) {
	quantizedWeight := []uint8{
		0, 1, 2, 3,
		3, 2, 1, 0,
		1, 1, 2, 2,
	}
	packed := packJANGTestValues(t, quantizedWeight, 2)
	scales := []float32{0.5, 1.25, -0.75}
	biases := []float32{-1, 2, 5}
	input := FromValues([]float32{
		1, 2, 3, 4,
		-1, 0.5, 2, -0.5,
	}, 2, 4)
	bias := FromValues([]float32{0.25, -1, 2}, 3)

	gotArray, err := JANGPackedLinear(input, FromValues(packed, len(packed)), FromValues(scales, len(scales)), FromValues(biases, len(biases)), bias, []int32{3, 4}, 4, 2)
	if err != nil {
		t.Fatalf("JANGPackedLinear() error = %v", err)
	}
	Materialize(gotArray)

	denseWeight := FromValues(dequantizeJANGTestValues(quantizedWeight, scales, biases, 4), 3, 4)
	denseWeightT := Transpose(denseWeight)
	wantArray := Add(Matmul(input, denseWeightT), bias)
	Materialize(wantArray)

	assertFloat32SliceClose(t, gotArray.Floats(), wantArray.Floats(), 1e-5)
	if shape := gotArray.Shape(); len(shape) != 2 || shape[0] != 2 || shape[1] != 3 {
		t.Fatalf("shape = %+v, want [2 3]", shape)
	}
}

func TestJANGDequant_FusedPackedLinearMatchesComposedProjection_Good(t *testing.T) {
	quantizedWeight := []uint8{
		0, 1, 2, 3,
		3, 2, 1, 0,
		1, 1, 2, 2,
	}
	packed := packJANGTestValues(t, quantizedWeight, 2)
	scales := []float32{0.5, 1.25, -0.75}
	biases := []float32{-1, 2, 5}
	input := FromValues([]float32{
		1, 2, 3, 4,
		-1, 0.5, 2, -0.5,
	}, 1, 2, 4)
	bias := FromValues([]float32{0.25, -1, 2}, 3)
	packedArray := FromValues(packed, len(packed))
	scaleArray := FromValues(scales, len(scales))
	biasArray := FromValues(biases, len(biases))

	gotArray, err := JANGPackedLinearFused(input, packedArray, scaleArray, biasArray, bias, []int32{3, 4}, 4, 2)
	if err != nil {
		t.Fatalf("JANGPackedLinearFused() error = %v", err)
	}
	wantArray, err := JANGPackedLinear(input, packedArray, scaleArray, biasArray, bias, []int32{3, 4}, 4, 2)
	if err != nil {
		t.Fatalf("JANGPackedLinear() error = %v", err)
	}
	Materialize(gotArray, wantArray)

	assertFloat32SliceClose(t, gotArray.Floats(), wantArray.Floats(), 1e-5)
	if shape := gotArray.Shape(); len(shape) != 3 || shape[0] != 1 || shape[1] != 2 || shape[2] != 3 {
		t.Fatalf("shape = %+v, want [1 2 3]", shape)
	}
}

func TestJANGDequant_FusedPackedLinearMatchesComposedProjectionNoBias_Good(t *testing.T) {
	quantizedWeight := []uint8{0, 1, 2, 3, 3, 2, 1, 0}
	packed := packJANGTestValues(t, quantizedWeight, 2)
	scales := []float32{0.5, 1.25}
	biases := []float32{-1, 2}
	input := FromValues([]float32{1, 2, 3, 4}, 1, 4)
	packedArray := FromValues(packed, len(packed))
	scaleArray := FromValues(scales, len(scales))
	biasArray := FromValues(biases, len(biases))

	gotArray, err := JANGPackedLinearFused(input, packedArray, scaleArray, biasArray, nil, []int32{2, 4}, 4, 2)
	if err != nil {
		t.Fatalf("JANGPackedLinearFused() error = %v", err)
	}
	wantArray, err := JANGPackedLinear(input, packedArray, scaleArray, biasArray, nil, []int32{2, 4}, 4, 2)
	if err != nil {
		t.Fatalf("JANGPackedLinear() error = %v", err)
	}
	Materialize(gotArray, wantArray)
	assertFloat32SliceClose(t, gotArray.Floats(), wantArray.Floats(), 1e-5)
}

func TestJANGDequant_PackedLinearRejectsShapeMismatch_Bad(t *testing.T) {
	_, err := JANGPackedLinear(FromValues([]float32{1, 2, 3}, 1, 3), FromValues([]uint8{0}, 1), FromValues([]float32{1}, 1), FromValues([]float32{0}, 1), nil, []int32{2, 2}, 4, 2)
	if err == nil || !core.Contains(err.Error(), "input") {
		t.Fatalf("error = %v, want input shape diagnostic", err)
	}
}

func TestJANGDequant_FusedPackedLinearRejectsShapeMismatch_Bad(t *testing.T) {
	_, err := JANGPackedLinearFused(FromValues([]float32{1, 2, 3}, 1, 3), FromValues([]uint8{0}, 1), FromValues([]float32{1}, 1), FromValues([]float32{0}, 1), nil, []int32{2, 2}, 4, 2)
	if err == nil || !core.Contains(err.Error(), "input") {
		t.Fatalf("error = %v, want input shape diagnostic", err)
	}
}

func packJANGTestValues(t *testing.T, values []uint8, bits int) []uint8 {
	t.Helper()
	packed := make([]uint8, (len(values)*bits+7)/8)
	maxValue := uint8((1 << bits) - 1)
	for i, value := range values {
		if value > maxValue {
			t.Fatalf("value %d exceeds %d-bit max", value, bits)
		}
		bitOffset := i * bits
		byteIndex := bitOffset / 8
		shift := bitOffset % 8
		packed[byteIndex] |= value << shift
		if shift+bits > 8 {
			packed[byteIndex+1] |= value >> (8 - shift)
		}
	}
	return packed
}

func dequantizeJANGTestValues(values []uint8, scales, biases []float32, groupSize int) []float32 {
	out := make([]float32, len(values))
	for i, value := range values {
		group := i / groupSize
		out[i] = float32(value)*scales[group] + biases[group]
	}
	return out
}

func assertFloat32SliceClose(t *testing.T, got, want []float32, epsilon float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("len(got) = %d, want %d", len(got), len(want))
	}
	for i := range got {
		if math.Abs(float64(got[i]-want[i])) > epsilon {
			t.Fatalf("value[%d] = %f, want %f", i, got[i], want[i])
		}
	}
}
